# Training recipes and paper results

Use the recorded environments in `docs/environment.md` and recover the fixed
cohorts using `docs/data.md`. Frozen inference needs no retraining; assemble the
weights with `scripts/download_paper_artifacts.py` first. Its default maps are:

- WOMD: `local/paper_artifacts/womd/weights.json`
- AV2: `local/paper_artifacts/av2/weights.json`

Use new training outputs. The examples below preserve the original loss, scene
ordering, seed and optimizer settings. They have not been executed as new training
runs during documentation maintenance.

## WOMD RCIM

Build the manifests once with `scripts/build_training_manifests.py`, as shown in
the data guide. Then prepare the native rows and run Joint-CE RCIM:

```bash
python scripts/prepare_training.py --config configs/womd.json \
  --manifest local/womd_rcim_training.jsonl --stage rcim --workdir work/womd
CUDA_VISIBLE_DEVICES=0 python scripts/train_rcim.py --config configs/womd.json \
  --workdir work/womd --output work/womd/rcim_models --seed 42
```

The original shard/record order and per-scene eligible-edge mean are retained.
The selected checkpoint is `work/womd/rcim_models/refine_only42_best.pt`.

## WOMD refiners

Generate train and tune choices separately in the Torch 2 scoring environment.
The inference contract records the cohort, so the two subsets need separate
output directories. Merge their disjoint choice files for bank preparation:

```bash
for split in fit tune; do
  python scripts/infer.py --config configs/womd.json \
    --manifest local/refiner_train_tune.jsonl --split "$split" --phase selection \
    --weights local/paper_artifacts/womd/weights.json \
    --output "work/train_choices/$split"
done
mkdir -p work/train_choices/selection
cp -n work/train_choices/fit/selection/*.pt work/train_choices/selection/
cp -n work/train_choices/tune/selection/*.pt work/train_choices/selection/
```

Switch to the WOMD refiner environment and prepare R1 banks:

```bash
python scripts/prepare_training.py --config configs/womd.json \
  --manifest local/refiner_train_tune.jsonl --stage i1 --workdir work/womd \
  --weights local/paper_artifacts/womd/weights.json \
  --selection-root work/train_choices/selection
CUDA_VISIBLE_DEVICES=0 python scripts/train_refiner.py --config configs/womd.json \
  --workdir work/womd --stage i1 --arm E --seed 42
```

R1 arms are A/E/B. WOMD output is `work/womd/models/LABEL/seedSEED/best.pt`, with
LABEL A/E42/B05 respectively. The A arm uses its own anchor bank and weights.

R2 preparation requires the matching, selected R1 heads for each seed:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_training.py --config configs/womd.json \
  --manifest local/refiner_train_tune.jsonl --stage i2 --seed 42 \
  --workdir work/womd --weights local/paper_artifacts/womd/weights.json \
  --selection-root work/train_choices/selection
CUDA_VISIBLE_DEVICES=0 python scripts/train_refiner.py --config configs/womd.json \
  --workdir work/womd --stage i2 --arm RCIM --seed 42 \
  --weights local/paper_artifacts/womd/weights.json
```

Repeat for seeds 42/43/44 and R2 arms RCIM/NoRCIM/B. WOMD R2 outputs are
`work/womd/seedSEED/models/BRANCH/best.pt`, with BRANCH RCIM/NoRCIM/B2.
Full and NoRCIM initialize from the same E-matched R1 with a zeroed output head;
the B branch uses B-matched R1 and only permits R2 updates where B selected A's
mode. Matched NoRCIM has its own R2; `Full_fixed_OFF` keeps Full's R1/R2 unchanged.

### Chaining newly trained modules

The supplied maps point to the frozen paper checkpoints. The commands above can
reproduce an individual refiner stage against those frozen parents. To chain a
newly trained RCIM/R1/R2, update the **local map** to the selected new checkpoint
paths before generating dependent choices/banks or evaluating. Keep the same
seed/arm mapping and regenerate dependent preparation into fresh directories.
For AV2, R2 also loads its parent from the native output layout below; that parent
must match the R1 used to prepare its rows. Do not mix frozen-parent preparation
with a different newly trained parent.

## AV2 recipes

AV2 uses `configs/av2.json` and `local/paper_artifacts/av2/weights.json` throughout.
After preprocessing and extraction in the data guide, prepare RCIM from a local
Training manifest and run its fixed native output convention:

```bash
python scripts/make_manifest.py --config configs/av2.json \
  --cache-root work/av2/cache/train --scene-ids local/splits/av2_refiner_train.txt \
  --split fit --output local/av2_refiner_fit.jsonl
python scripts/make_manifest.py --config configs/av2.json \
  --cache-root work/av2/cache/train --scene-ids local/splits/av2_refiner_tune.txt \
  --split tune --output local/av2_refiner_tune.jsonl
cat local/av2_refiner_fit.jsonl local/av2_refiner_tune.jsonl \
  > local/av2_refiner_train_tune.jsonl
```

These commands index the 8,192 train and 1,024 tune scenes for R1/R2 preparation.
Run them into new output paths. RCIM preparation uses the complete
native `work/av2/cache/train` shards, so its manifest may be this existing local
Training index; it validates its own full fit membership independently.

```bash
python scripts/prepare_training.py --config configs/av2.json \
  --manifest local/av2_refiner_fit.jsonl --stage rcim --workdir work/av2
CUDA_VISIBLE_DEVICES=0 python scripts/train_rcim.py --config configs/av2.json \
  --workdir work/av2 --output work/av2/models/rcim_seed42 --seed 42
```

For R1, prepare the combined manifest and train a matched arm:

```bash
python scripts/prepare_training.py --config configs/av2.json \
  --manifest local/av2_refiner_train_tune.jsonl --stage i1 --workdir work/av2 \
  --weights local/paper_artifacts/av2/weights.json
CUDA_VISIBLE_DEVICES=0 python scripts/train_refiner.py --config configs/av2.json \
  --workdir work/av2 --stage i1 --arm E --seed 42
```

To reproduce R2 against the frozen paper parents, use a separate R2 workdir and
place those parents at the native trainer's expected paths. On Linux:

```bash
python - <<'PY'
import json
from pathlib import Path
weights = json.loads(Path('local/paper_artifacts/av2/weights.json').read_text())
for seed in (42, 43, 44):
    for label, arm in (('E', 'Full'), ('B', 'B2')):
        parent = Path(weights['heads'][str(seed)][arm]['i1']).resolve()
        assert parent.is_file()
        target = Path(f'work/av2_r2/models/refiners/seed{seed}/i1_{label}/best.pt')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(parent)
PY
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_training.py --config configs/av2.json \
  --manifest local/av2_refiner_train_tune.jsonl --stage i2 --seed 42 \
  --workdir work/av2_r2 --weights local/paper_artifacts/av2/weights.json
CUDA_VISIBLE_DEVICES=0 python scripts/train_refiner.py --config configs/av2.json \
  --workdir work/av2_r2 --stage i2 --arm RCIM --seed 42
```

Use a fresh workdir; the example does not overwrite existing parent checkpoints.
Selection is computed during preparation in the AV2 runtime. Prepared rows are
under `prepared/i1/{train,tune}` and `prepared/i2/seedSEED/{train,tune}`.
Repeat the matching preparations for seeds 42/43/44. R1 arms are A/E/B; R2 arms
are RCIM/NoRCIM/B. Outputs are `WORKDIR/models/refiners/seedSEED/STAGE_ARM/best.pt`.

## Training protocol details

R1 loss is masked SmoothL1 position + 0.25 endpoint + 0.05 residual norm squared
+ 0.002 residual acceleration squared, with gradient clipping at 5. R2 uses
Euclidean position/endpoint errors, radius-scaled actual residual regularization
and clipping at 1. WOMD R2 steps its scheduler before tuning; R1/AV2 step it after
tuning. Refiner loops run ten epochs/5,120 updates; the inherited `patience` field
does not enable early stopping. RCIM retains its original Joint-CE objective.

## Tables 2–11

| Paper table | Cohort / entry | Notes |
|---|---|---|
|2|WOMD 44,097 Validation; `scripts/infer.py` + `scripts/evaluate.py`|A,A_I1,E_I1,B2,NoRCIM,Full; archived `results/womd_validation.json`|
|3|20,000 frozen Training confirmation; same entry|local historical fingerprint-verified IDs; public token recovery or original qualification/exclusion inputs; backbone membership unknown|
|4|paired summary from Tables 2/3|same scene draw across three fixed models; original archived CIs retained|
|5|5,000 exposed Training audit; `--methods Full_coord Full_I1 Full_noP2 Full`|shared Full heads; identifies P2 contribution, not retraining|
|6|fixed-head ON/OFF; `--methods Full Full_fixed_OFF`|Full I1/I2 unchanged; differs from matched NoRCIM; RNG 20261103|
|7|per-seed summaries from Tables 2/3|RCIM42 fixed; descriptive sample SD of three refiner seeds|
|8|AV2 24,988 Validation; AV2 `infer.py` + `evaluate.py`|focal and all endpoint-valid scored/focal metrics kept separately|
|9|same AV2 predictions|center-distance < 1 m proxy; all requested pair denominator 246,531|
|10|`reproduction/av2_closing/inference_cost.py`|native isolated GPU benchmark, 256 fit scenes, 20 warmup, 3 passes; bundled native interfaces; use stage_native_av2.py then native_av2_cost.py|
|11|`reproduction/external/` native learned and classical entries|adapted forecasting ports; external neural/compatibility assets required, see below|

Table 10 timings are AV2's actual isolated benchmark. The generic `infer.py` adapter
also computes diagnostic legacy branches and is **not** a replacement latency
measurement. No latency experiment was rerun in this packaging task.

`results/paper_tables.json` is the exact manuscript source snapshot. Official
metric snapshots and checkpoint hashes are references, use them alongside the included frozen inference weights and authorized datasets.
They are saved experiment results, not a new end-to-end reproduction.

## Evaluation definitions

ADE is a per-target mean over valid future positions, aggregated only over targets
valid at the endpoint. FDE uses the same endpoint mask:79WOMD,59AV2. Inference
targets and geometric pair denominators do not depend on future validity.
The additional >5m count compares selected final FDE to original A with
threshold 5.000001; it is not a missing-rate or collision statistic.

WOMD box overlap uses observed valid vehicle sizes and original oriented-box SAT
with 1e-6 penetration tolerance and heading carry-over for displacements below 0.05 m. AV2 uses
center distance < 1 m because dimensions are unavailable. Do not compare these as
the same geometric metric. P1/P2 reject new conflict identities or an increase
in total conflict steps relative to the incoming trajectory set; they do not
guarantee zero inherited conflicts.

Whole-scene paired bootstrap uses 5,000 resamples; model predictions are averaged
within a scene before resampling. Three seed outputs are never 15,000/3N independent
observations. WOMD's native final summary used RNG 20261030 and its saved shard
order. The portable summary fixes lexical scene order: point estimates agree;
finite-resample interval endpoints need not match the archived order-dependent
last decimals. AV2's closing accuracy reconstruction fixes lexical order and
RNG 9413; earlier all-target AV2 summary used 4201. The archived closing result
preserves the exact focal intervals. AV2 geometric-rate CIs in original results
may include zero; this is not rounded into a significance claim.

## External methods and native-layout scripts

SmartRefine/SRefiner neural implementations, weights and required compatibility
assets are external dependencies; sources and licenses are listed in `docs/LICENSING.md`.
`reproduction/external/SOURCE_MANIFEST.json` records the exact upstream source
hashes. Do not replace an adapted method with an upstream released-backbone number.
The external baseline scripts retain their explicit container mount layout, documented in
`reproduction/README.md`; they are supplemental entrypoints, not certified by
the earlier small CPU smoke. Their external dependencies must be prepared separately.

## Verification scope

See `docs/validation.json` and `docs/PUBLIC_RELEASE_NOTES.md`: syntax, import, config, argument, deterministic
decomposition/geometry tests and a few frozen fit forwards are reported separately
from full experiment reproduction. No optimizer step, full Validation pass or
new threshold selection is part of packaging.

Freshly trained checkpoint files contain packaged-code/protocol metadata, so their
file SHA need not equal the archived binary even with matching model tensors.
Use the published checkpoint hashes for frozen evaluation, and state-dict and
prediction checks for a fresh training reproduction. No fresh training is claimed here.
