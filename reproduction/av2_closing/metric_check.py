"""Read-only AV2 metric reconstruction and focal scene-paired confidence intervals.

Uses saved predictions only. No model is loaded, trained, or selected here.
Python 3.8 / NumPy 1.19 / PyTorch 2.0 compatible.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch


METHODS = ('NDPNet', 'NDPNet_refined', 'OldRCIM_refined',
           'GeometryB_refined', 'GeometryB_two_refiners',
           'Full_without_RCIM', 'FullRCIM')
SEEDS = ('42', '43', '44')
METRICS = ('scored_plus_focal_ADE6s', 'scored_plus_focal_FDE6s',
           'focal_ADE6s', 'focal_FDE6s')
REFERENCES = ('NDPNet', 'NDPNet_refined', 'OldRCIM_refined',
              'GeometryB_two_refiners', 'Full_without_RCIM')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def close(actual, expected, description):
    actual, expected = float(actual), float(expected)
    assert math.isfinite(actual) and math.isfinite(expected), description
    tolerance = 1e-8 + 1e-10 * abs(expected)
    assert abs(actual - expected) <= tolerance, (
        description, actual, expected, tolerance)
    return abs(actual - expected)


def frozen_check(root, frozen):
    for relative, digest in frozen['sha256'].items():
        assert sha(root / relative) == digest, relative
    return len(frozen['sha256'])


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(root, output):
    begun = time.perf_counter()
    cpu_start = time.process_time()
    torch.set_num_threads(1)
    result_path = root / 'AV2_RESULTS.json'
    frozen_path = root / 'MODELS_FROZEN.json'
    manifest_path = root / 'manifests/val_files.txt'
    old_result_sha = sha(result_path)
    old_frozen_sha = sha(frozen_path)
    original = json.loads(result_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    assert original['passed'] and original['scenes'] == 24988
    frozen_files = frozen_check(root, frozen)
    manifest_lines = [line.strip() for line in manifest_path.read_text().splitlines() if line.strip()]
    manifest_ids = [Path(line).stem for line in manifest_lines]
    assert len(manifest_ids) == len(set(manifest_ids)) == 24988
    source_files = sorted((root / 'evaluation/shards').glob('*.pt'))
    assert len(source_files) == 391, len(source_files)

    # Each record needs only 3 seeds x 7 methods x 4 error sums, about 17 MB total.
    numerators = np.zeros((24988, 3, len(METHODS), 4), dtype=np.float64)
    denominators = np.zeros((24988, 4), dtype=np.int64)
    totals_pairs = np.zeros((3, len(METHODS)), dtype=np.int64)
    totals_conflicts = np.zeros_like(totals_pairs)
    totals_steps = np.zeros_like(totals_pairs)
    totals_predicted = np.zeros_like(totals_pairs)
    scenario_ids = []
    seen = set()
    metadata_sha = hashlib.sha256()
    evaluation_inventory = hashlib.sha256()
    cache_inventory = hashlib.sha256()
    max_stored_error = 0.
    geometry_denominators_checked = 0
    reconstructed_seed_method_records = 0
    at = 0
    focal_count_histogram = {}

    for file_index, prediction_path in enumerate(source_files, 1):
        cache_path = root / 'cache/val' / prediction_path.name
        assert cache_path.is_file(), str(cache_path)
        evaluation_inventory.update((prediction_path.name + ':' + sha(prediction_path) + '\n').encode())
        cache_inventory.update((cache_path.name + ':' + sha(cache_path) + '\n').encode())
        prediction_package = torch.load(prediction_path, map_location='cpu')
        assert prediction_package['frozen_manifest_sha256'] == old_frozen_sha
        saved = prediction_package['records']
        raw_package = torch.load(cache_path, map_location='cpu')
        raw = raw_package['rows']
        assert [r['scene'] for r in saved] == [r['scene'] for r in raw]
        for record, row in zip(saved, raw):
            scene = row['scene']
            assert scene not in seen and at < 24988, scene
            seen.add(scene)
            scenario_ids.append(scene)
            agent_ids = [str(v) for v in row['original_agent_id']]
            assert agent_ids == [str(v) for v in record['original_agent_id']]
            assert len(agent_ids) == len(set(agent_ids))
            metadata_sha.update(json.dumps([scene, agent_ids], separators=(',', ':')).encode())
            gt = array(row['gt']).astype(np.float64, copy=False)
            valid = array(row['valid']).astype(bool, copy=False)
            is_focal = array(row['is_focal']).astype(bool, copy=False)
            count = len(agent_ids)
            assert gt.shape == (count, 60, 2)
            assert valid.shape == (count, 60) and is_focal.shape == (count,)
            assert np.isfinite(gt).all()
            endpoint_valid = valid[:, 59]
            focal_valid = endpoint_valid & is_focal
            eligible = int(endpoint_valid.sum())
            focal = int(focal_valid.sum())
            observed_focal_count = int(is_focal.sum())
            focal_count_histogram[str(observed_focal_count)] = focal_count_histogram.get(str(observed_focal_count), 0) + 1
            denominators[at] = (eligible, eligible, focal, focal)
            expected_pairs = count * (count - 1) // 2
            valid_count = np.maximum(valid.sum(axis=-1), 1)
            for seed_index, seed in enumerate(SEEDS):
                for method_index, method in enumerate(METHODS):
                    path = array(record['predictions'][seed][method]).astype(np.float64, copy=False)
                    assert path.shape == gt.shape and np.isfinite(path).all(), (scene, method)
                    error = np.linalg.norm(path - gt, axis=-1)
                    ade = (error * valid).sum(axis=-1) / valid_count
                    values = np.array((ade[endpoint_valid].sum(), error[endpoint_valid, 59].sum(),
                                       ade[focal_valid].sum(), error[focal_valid, 59].sum()), dtype=np.float64)
                    assert np.isfinite(values).all(), (scene, seed, method)
                    numerators[at, seed_index, method_index] = values
                    saved_metric = record['methods'][seed][method]
                    for key, value in zip(('ade_sum', 'fde_sum', 'focal_ade_sum', 'focal_fde_sum'), values):
                        max_stored_error = max(max_stored_error, close(value, saved_metric[key], (scene, seed, method, key)))
                    assert saved_metric['targets'] == eligible
                    assert saved_metric['focal_targets'] == focal
                    assert saved_metric['all_predicted_targets'] == count
                    assert saved_metric['pairs'] == expected_pairs
                    totals_pairs[seed_index, method_index] += expected_pairs
                    totals_predicted[seed_index, method_index] += count
                    totals_conflicts[seed_index, method_index] += saved_metric['overlap']
                    totals_steps[seed_index, method_index] += saved_metric['overlap_steps']
                    geometry_denominators_checked += 1
                    reconstructed_seed_method_records += 1
            at += 1
        # Release all predictions immediately; no model or full prediction set is retained.
        del prediction_package, saved, raw_package, raw
        if file_index % 40 == 0 or file_index == len(source_files):
            print('AV2_METRIC_RECHECK %d/%d shards | %d/24988 scenes | elapsed_s=%.1f' %
                  (file_index, len(source_files), at, time.perf_counter() - begun), flush=True)
    assert at == 24988 and seen == set(manifest_ids)
    assert len(seen) == len(scenario_ids)
    assert int(denominators[:, 0].sum()) == 97424
    assert int(denominators[:, 2].sum()) == 24988
    assert np.all(totals_pairs == 246531)

    # Canonical scene order makes bootstrap results independent of shard traversal order.
    order = np.argsort(np.asarray(scenario_ids))
    numerators = numerators[order]
    denominators = denominators[order]
    scenario_ids = [scenario_ids[int(i)] for i in order]
    den_sum = denominators.sum(axis=0)
    aggregate = numerators.sum(axis=0) / np.maximum(den_sum, 1)[None, None, :]
    per_seed = {}
    mean_sd = {}
    root_metric_keys = ('ADE6', 'FDE6', 'focal_ADE6', 'focal_FDE6')
    for seed_index, seed in enumerate(SEEDS):
        per_seed[seed] = {}
        for method_index, method in enumerate(METHODS):
            values = aggregate[seed_index, method_index]
            prior = original['per_seed'][seed][method]
            for key, value in zip(root_metric_keys, values):
                close(value, prior[key], ('whole_set', seed, method, key))
            assert prior['valid6_targets'] == int(den_sum[0])
            assert prior['focal_targets'] == int(den_sum[2])
            assert prior['eligible_pairs'] == int(totals_pairs[seed_index, method_index])
            assert prior['center_conflict_pairs'] == int(totals_conflicts[seed_index, method_index])
            assert prior['center_conflict_steps'] == int(totals_steps[seed_index, method_index])
            assert prior['all_predicted_targets'] == int(totals_predicted[seed_index, method_index])
            per_seed[seed][method] = dict(zip(METRICS, (float(v) for v in values)))
            per_seed[seed][method].update(
                valid6_targets=int(den_sum[0]), focal_valid6_targets=int(den_sum[2]),
                predicted_targets=int(totals_predicted[seed_index, method_index]),
                eligible_geometry_pairs=int(totals_pairs[seed_index, method_index]),
                stored_center_conflict_pairs=int(totals_conflicts[seed_index, method_index]),
                stored_center_conflict_steps=int(totals_steps[seed_index, method_index]))
    for method_index, method in enumerate(METHODS):
        mean_sd[method] = {}
        for metric_index, metric in enumerate(METRICS):
            values = aggregate[:, method_index, metric_index]
            mean_sd[method][metric] = {'mean': float(values.mean()), 'sample_sd': float(values.std(ddof=1))}

    reconstruction_seconds = time.perf_counter() - begun
    mean_numerator = numerators.mean(axis=1)
    new_index = METHODS.index('FullRCIM')
    differences = np.stack([mean_numerator[:, new_index] - mean_numerator[:, METHODS.index(old)]
                            for old in REFERENCES], axis=1)
    rng = np.random.default_rng(9413)
    draws = np.empty((5000, len(REFERENCES), len(METRICS)), dtype=np.float64)
    for repetition in range(5000):
        selected = rng.integers(0, at, at)
        this_denominator = denominators[selected].sum(axis=0)
        assert (this_denominator > 0).all()
        draws[repetition] = differences[selected].sum(axis=0) / this_denominator[None, :]
        if (repetition + 1) % 1000 == 0:
            print('AV2_PAIRED_BOOTSTRAP %d/5000 scene resamples' % (repetition + 1), flush=True)
    point = differences.sum(axis=0) / den_sum[None, :]
    low, high = np.quantile(draws, [.025, .975], axis=0)
    intervals = {}
    for reference_index, reference in enumerate(REFERENCES):
        intervals['FullRCIM_minus_' + reference] = {
            metric: {'delta': float(point[reference_index, metric_index]),
                     'CI95': [float(low[reference_index, metric_index]), float(high[reference_index, metric_index])]}
            for metric_index, metric in enumerate(METRICS)}

    # The old metric file, model/code/config inventory and original weights stay unchanged.
    assert sha(result_path) == old_result_sha
    assert sha(frozen_path) == old_frozen_sha
    assert frozen_check(root, frozen) == frozen_files
    scope = {
        'prediction': 'Saved selected single trajectory per requested target; native 6 s, 60 future frames.',
        'ADE': 'Per-target mean Euclidean error over its valid future steps; aggregate only targets valid at frame59.',
        'FDE': 'Euclidean error at frame59; identical endpoint-valid target mask for every method.',
        'focal': 'Additional observed-category focal mask from the frozen feature cache, then endpoint validity.',
        'not_official_min_of_six': True,
        'conflict': 'Stored AV2 center-distance<1m proxy counts reused; geometry denominators checked, not a new geometry computation.',
        'bootstrap': '5000 paired scene resamples, same scene draw across references and metrics; three fixed refiner seeds averaged within each scene.',
        'training_randomness': 'RCIM seed42 fixed; refiner seeds42/43/44. CIs condition on these frozen models, not all possible training seeds.',
        'exposure': 'AV2 validation historically selected the frozen local backbone. This closing check neither tunes models nor makes a new unexposed-test claim.',
        'GT_use': 'GT and future validity used exclusively for offline metric reconstruction, not inference or selection.'}
    value = {
        'passed': True, 'scenes': at, 'valid6_scored_plus_focal_targets': int(den_sum[0]),
        'valid6_focal_targets': int(den_sum[2]), 'eligible_geometry_pairs': int(totals_pairs[0, 0]),
        'observed_focal_count_per_scene_histogram': focal_count_histogram,
        'reconstructed_seed_method_records': reconstructed_seed_method_records,
        'geometry_denominator_checks': geometry_denominators_checked,
        'maximum_absolute_stored_sum_difference': max_stored_error,
        'scene_ids_unique_and_match_frozen_manifest': True,
        'per_seed': per_seed, 'three_seed_mean_and_sample_sd': mean_sd,
        'paired_scene_bootstrap': intervals, 'bootstrap_seed': 9413, 'bootstrap_replicates': 5000,
        'scope': scope, 'frozen_files_checked_before_and_after': frozen_files,
        'original_assets_unchanged': True,
        'provenance': {'code_path': str(Path(__file__)), 'code_SHA256': sha(__file__),
                       'AV2_RESULTS_SHA256': old_result_sha, 'MODELS_FROZEN_SHA256': old_frozen_sha,
                       'val_scene_manifest': str(manifest_path), 'val_scene_manifest_SHA256': sha(manifest_path),
                       'ordered_prediction_shards_inventory_SHA256': evaluation_inventory.hexdigest(),
                       'ordered_cache_shards_inventory_SHA256': cache_inventory.hexdigest(),
                       'scene_and_original_agent_ID_mapping_SHA256': metadata_sha.hexdigest(),
                       'sorted_scene_ID_SHA256': hashlib.sha256(('\n'.join(scenario_ids) + '\n').encode()).hexdigest()},
        'resources': {'torch_threads': 1, 'GPU_used': False, 'reconstruction_seconds': reconstruction_seconds,
                      'total_wall_seconds': time.perf_counter() - begun,
                      'process_CPU_seconds': time.process_time() - cpu_start,
                      'peak_RSS_GiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.**2,
                      'torch_version': torch.__version__, 'numpy_version': np.__version__}}
    atomic_json(output / 'METRIC_RECHECK.json', value)
    print('AV2_METRIC_RECHECK_COMPLETE', json.dumps({'passed': True, 'scenes': at,
          'focal_targets': int(den_sum[2]), 'seconds': value['resources']['total_wall_seconds'],
          'output': str(output / 'METRIC_RECHECK.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/workspace/output'))
    parser.add_argument('--out', type=Path, default=Path('/workspace/checks'))
    options = parser.parse_args()
    main(options.root, options.out)
