# Optional native reproducers

These are preserved native experiment entrypoints, not replacement algorithms.
They are intentionally separated from the portable core CLI. This packaging task
did not rerun any of them. Do not treat their presence as a passed external-method
or raw-backbone reproduction.

## Table 10: isolated AV2 inference cost

`av2_closing/inference_cost.py` is the exact final timing script, not the earlier
all-variant diagnostic bundle. Its native first-party AV2 dependencies are included
in `native_av2/code`; exact ordered filename manifests are regenerated into the local `CCC_SPLIT_ROOT` directory (see docs/data.md).
`stage_native_av2.py` stages these assets and links user-supplied weights/caches.
It verifies the frozen checkpoint/source hashes and creates a new local integrity
manifest without copying dataset or checkpoint contents into the repository.

```bash
python scripts/stage_native_av2.py --weights local/weights_av2.json \
 --backbone checkpoints/av2_backbone.ckpt --cache-root work/av2/cache \
 --workdir work/native_av2
python scripts/native_av2_cost.py --workspace work/native_av2 \
 --processed-root data/av2 --output work/av2_cost --mode prepare
python scripts/native_av2_cost.py --workspace work/native_av2 \
 --processed-root data/av2 --output work/av2_cost --mode verify
python scripts/native_av2_cost.py --workspace work/native_av2 \
 --processed-root data/av2 --output work/av2_cost --mode benchmark --method FullRCIM
```

Use the frozen AV2 Torch2 runtime and an authorized idle GPU. Each method must run
in a separate process, with the same256FITscenes,20warmup,3passes. The wrapper only
routes I/O roots and the equivalent packaged feature-loader path adapter; it invokes
native benchmark functions unchanged. It does not run the benchmark automatically.
If using Docker, bind symlink target directories read-only at accessible matching
paths, as well as the writable result output. A staged local integrity-manifest SHA
is not impersonated as the original archive's byte-identical metadata hash.

`metric_check.py --help` exposes the original saved-output closing check. That
archived-output checker still requires its original frozen manifest/layout to
verify existing archived shards. For newly reproduced portable outputs, use
`scripts/evaluate.py`; do not bypass archived-output integrity assertions.
No latency experiment was rerun while assembling this release.

## Table 11: adapted external postprocessors

`external/code` preserves the NDPNet-specific SmartRefine/SRefiner adapters and
native training wrappers. The actual upstream neural source is **not bundled**;
obtain the original licensed source versions matching
`external/SOURCE_MANIFEST.json` before reconstructing `sources/SmartRefine` and
`sources/SRefiner`. SRefiner's CUDA-compatibility source lineage must also be
restored from the original approved adapter archive; no silent substitute is
provided. The archived MIT text has unfilled copyright placeholders; clarify attribution before vendoring that source.

Native mounts: `/workspace/external` contains the authorized source/adapters,
protocols, prepared rows and external model weights; `/workspace/output` contains
the frozen AV2 experiment assets; `/workspace/validation` is a new result output.
The original scripts verify code/config/checkpoint hashes and fail when assets
are missing. They do not download sources or bypass those checks.

Actual entrypoints (only after those assets/permissions are supplied):

```bash
python reproduction/external/code/baseline_train.py train --method SmartRefine --seed 42
python reproduction/external/learned/validation_worker.py evaluate --method SmartRefine --seed 42
python reproduction/external/classical/worker.py evaluate --method ORCA
```

Do not run `tune` on Validation or reselect parameters. Frozen classical
configurations are in `configs/external_classical_frozen.json`. The original
worker expects separate `FROZEN_METHOD.json` files and native workspace integrity
manifests; these are metadata/asset setup requirements, not new hyperparameters.

The ORCA C++ source is the original Apache-2.0 RVO2 dynamic-agent port, with its
copyright header retained, original LP core and disclosed forecasting adaptations.
Compile the supplied source in its native classical workspace using
`g++ -std=c++11 -O2 -shared -fPIC orca_port.cpp -o liborca_adapter.so`; no binary
is distributed. The archived library SHA is build-specific and is not promised
to match a different compiler. CHOMP/SCP are labelled forecasting adaptations.
Table11 is not claimed to reproduce the external authors' released backbones.

## Restrictions

These supplemental scripts retain native container-layout contracts. Their imports
and full asset reconstruction are **not certified by the core fit smoke**.
Full external reproducibility remains pending source/attribution and compatibility
archive restoration and an independent native-layout run. The official numbers
are archived in results/external_av2_validation.json, not fabricated by these
examples. None of these commands was launched during release preparation.
