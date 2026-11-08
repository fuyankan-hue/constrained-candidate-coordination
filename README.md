# Constrained Candidate Coordination and Staged Trajectory Refinement

Code and frozen inference models for **Constrained Candidate Coordination and
Staged Trajectory Refinement for Multi-Agent Interaction Consistency**, submitted
to Engineering Applications of Artificial Intelligence. Author details are
withheld for anonymous review.

## Method

The framework coordinates the six trajectories of a frozen NDPNet predictor,
then refines the selected paths:

**NDPNet → RCIM → constrained joint selection → B-spline R1 + P1 → R2 + P2.**

- RCIM learns candidate-pair compatibility from mode-specific Refine features and
  predicted relative motion. Near/Far hidden inputs are zero; those backbone
  stages remain intact.
- CJM20 admits candidates within 20 m of the independent top-1 path. Eligible
  candidates are projected into a 4 m whole-path neighborhood of that anchor.
  Unary and pair scores are extended linearly and bilinearly, respectively.
- Geometry-first joint selection applies baseline-relative feasibility rules.
  Two bounded B-spline refiners use scene-level protection after each stage.
- The final path stays within 5 m of the original top-1. This limits trajectory
  change, not absolute prediction error or real-world collision risk.

RCIM has **44,449** parameters; each refiner has **76,026**, for **196,501** added
parameters. RCIM seed 42 is fixed; refiner seeds are 42, 43 and 44.

The backbone is [NDPNet](https://github.com/HuaHu-yizhou/NDPNet), built on
[QCNet](https://github.com/ZikangZhou/QCNet). The used source snapshots and their
licenses are retained in `third_party/`; see [source provenance](docs/source_manifest.json).

## Environment

The experiments used Linux and separate environments:

| Stage | Python | PyTorch | NumPy |
|---|---|---|---|
| WOMD backbone/scoring; all AV2 model stages | 3.8.10 | 2.0.1+cu118 | 1.24.3 |
| WOMD continuous refinement | 3.8.10 | 1.13.1+cu117 | 1.19.5 |
| WOMD raw-data reader | 3.11.17 | 2.0.1+cpu | 1.24.3 |

Use `requirements-scoring.txt`, `requirements.txt` and `requirements-reader.txt`
for their respective stages. These record observed versions, not a single
resolver-tested lockfile. CUDA/PyG pairing and the reader SDK dependency conflict
are explained in [environment.md](docs/environment.md). Do not combine the three
requirements files. A clean installation remains unverified.

## Data and weights

Obtain [WOMD Scenario v1.3.0](https://waymo.com/open/download/) and
[Argoverse 2 Motion Forecasting](https://argoverse.github.io/user-guide/tasks/motion_forecasting.html)
under their provider terms. Raw data and feature caches are not included.
See [data.md](docs/data.md) for indexing, preprocessing, extraction and manifests.

Download the complete review repository archive, including `assets/review`.
Then run from the repository root:

```bash
python scripts/download_paper_artifacts.py --dataset all
```

Despite its filename, this command assembles the included assets **offline**.
It verifies chunk, archive and checkpoint SHA256 values without contacting a
personal hosting service. It provides 40 inference exports and generates:

| Dataset | Weights map | Backbone checkpoint |
|---|---|---|
| WOMD | `local/paper_artifacts/womd/weights.json` | `local/paper_artifacts/womd/checkpoints/backbone.ckpt` |
| AV2 | `local/paper_artifacts/av2/weights.json` | `local/paper_artifacts/av2/checkpoints/backbone.ckpt` |

The exports preserve all frozen model tensors; training-only metadata is omitted.
[checkpoints.md](docs/checkpoints.md) describes the model terms, export hashes and
recovery of the original seven cohort documents from user-owned dataset indexes.
After recovery, set `CCC_SPLIT_ROOT` to the generated membership directory:

```bash
export CCC_SPLIT_ROOT="$PWD/local/splits"
```

## Frozen evaluation

The example below uses the fixed WOMD audit manifest prepared in
[data.md](docs/data.md). Use a new output directory for each cohort/configuration.
Selection runs in the scoring environment; WOMD refinement runs in its separate
refiner environment.

```bash
# Scoring environment: generate joint choices once.
python scripts/infer.py --config configs/womd.json \
  --manifest local/audit.jsonl --split audit --phase selection \
  --weights local/paper_artifacts/womd/weights.json --output work/eval_womd

# WOMD refiner environment: evaluate all three fixed refiner seeds.
for seed in 42 43 44; do
  python scripts/infer.py --config configs/womd.json \
    --manifest local/audit.jsonl --split audit --phase refinement \
    --weights local/paper_artifacts/womd/weights.json --seed "$seed" --device cpu \
    --methods A A_I1 E_I1 B2 NoRCIM Full --output work/eval_womd
done

python scripts/evaluate.py --config configs/womd.json --input work/eval_womd \
  --output work/eval_womd/summary.json --main Full --references A A_I1 B2 NoRCIM
```

For AV2 use `configs/av2.json`, `local/paper_artifacts/av2/weights.json` and the AV2
manifest/cohort described in the data guide. Both phases use the AV2 environment.
For the fixed-refiner score ablation, run `--methods Full Full_fixed_OFF` and
summarize with `--main Full --references Full_fixed_OFF`. This shares Full's two
heads and differs from the separately matched `NoRCIM` second head.

## Training

The native recipes and full preparation sequence are in
[reproduction.md](docs/reproduction.md). The following WOMD commands assume the
corresponding RCIM rows and refiner banks are already prepared. Expose one
authorized GPU and use fresh training outputs.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_rcim.py \
  --config configs/womd.json --workdir work/womd \
  --output work/womd/rcim_models --seed 42

CUDA_VISIBLE_DEVICES=0 python scripts/train_refiner.py \
  --config configs/womd.json --workdir work/womd --stage i1 --arm E --seed 42

CUDA_VISIBLE_DEVICES=0 python scripts/train_refiner.py \
  --config configs/womd.json --workdir work/womd --stage i2 --arm RCIM --seed 42 \
  --weights local/paper_artifacts/womd/weights.json
```

RCIM uses Joint CE, AdamW (learning rate 2e-4, weight decay 1e-4), clipping at 5,
at most eight epochs and patience three. Each refiner uses 8,192 training and
1,024 tuning scenes, batch size 16, ten epochs and 5,120 updates. RCIM checkpoint
selection uses calibration Joint CE; refiners use tuning ADE + 0.25 FDE.
Repeat refiner training for seeds 42/43/44 and the required matched arms. See the
reproduction guide before chaining newly trained checkpoints.

## Results and checks

`results/paper_tables.json` contains the eleven manuscript tables. Other files in
`results/` contain the saved WOMD audit, confirmation, Validation, AV2 Validation,
fixed-refiner ablation, external comparisons and inference-cost summaries.
[reproduction.md](docs/reproduction.md) maps Tables 2–11 to entrypoints and metrics.

WOMD uses oriented-box overlap; AV2 uses a center-distance proxy below 1 m.
Neither is a real collision measurement. Intervals use whole-scene paired
bootstrap samples with predictions averaged across the three fixed models.
The 20,000-scene cohort is held out from added-module development, not certified
unseen to the backbone. WOMD and AV2 backbone Validation exposure is retained in
the experimental record.

```bash
python scripts/check_release.py
python scripts/check_release.py --imports
python -m unittest discover -s tests
```

These checks cover integrity, imports and local contracts. Prior small CPU
forward checks, current distribution status and remaining limitations are
separated in [validation.json](docs/validation.json) and
[release notes](docs/PUBLIC_RELEASE_NOTES.md). Full raw-data-to-table reproduction
on a clean machine and optional external-method dependencies remain unverified.

## License

Original code is Apache-2.0; author identification is withheld for review.
NDPNet, QCNet, RVO2 and other third-party notices retain their respective scope.
See `LICENSE`, `NOTICE` and [LICENSING.md](docs/LICENSING.md).

This software was made using the Waymo Open Dataset, provided by Waymo LLC under
the [Waymo Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms/),
and your access and use of such work are governed by the terms and conditions
therein. WOD model use remains subject to that Agreement, including downstream
non-commercial restrictions. AV2 data/documentation are CC BY-NC-SA 4.0 and its
API is MIT. Source-code Apache-2.0 does not replace these separate terms.

## Citation

```bibtex
@misc{constrained_candidate_coordination,
  author = {Anonymous authors},
  title = {Constrained Candidate Coordination and Staged Trajectory Refinement
           for Multi-Agent Interaction Consistency},
  year = {2026},
  note = {Manuscript submitted to Engineering Applications of Artificial Intelligence}
}
```

Please also cite NDPNet, QCNet, WOMD and Argoverse 2 using their original references.
