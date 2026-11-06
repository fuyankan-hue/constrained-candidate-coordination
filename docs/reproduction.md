# Native recipes and paper table map

## Training preparation

The source losses, network definitions and update loops are retained. New wrappers
only assemble the original bank/row structures and configure local paths. Training
was **not** run while preparing this release. Use a fresh writable work directory;
never point it at formal experiment output.

Create WOMD training manifests without scanning trajectory tensors:

```bash
python scripts/build_training_manifests.py --raw-index local/womd_training_raw.jsonl \
 --cache-root work/womd/cache/training --output-dir local
```

Released checkpoints and exact cohort-token recovery are documented in `docs/checkpoints.md`;
use its downloader for inference without retraining. Then prepare RCIM native rows from the explicit local manifest (first generate the local split metadata per docs/data.md and set CCC_SPLIT_ROOT):

```bash
python scripts/prepare_training.py --config configs/womd.json \
 --manifest local/womd_rcim_training.jsonl --stage rcim --workdir work/womd
```

This retains the original per-scene eligible-edge loss mean and shard/record order.
For AV2, extraction already writes native numeric `cache/train/*.pt` shards;
use the same preparation command with AV2 config/workdir. It performs only a
forward finite-loss preflight and records that backward was not executed.

Generate selection files in the WOMD Torch2 scoring environment for all fixed
train/tune scenes, then prepare I1 banks in the refiner environment:

```bash
python scripts/prepare_training.py --config configs/womd.json \
 --manifest local/refiner_train_tune.jsonl --stage i1 --workdir work/womd \
 --weights local/weights_womd.json --selection-root work/train_choices/selection
```

The native banks are E42/B05. A-matched uses the exact Stage-W `bankA` anchor0
duplication, not E-matched weights. `train_refiner.py --stage i1 --arm A|E|B`
selects the corresponding native recipe; seeds42/43/44 must all be run.
After a same-seed I1 checkpoint is selected, prepare I2 rows:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_training.py --config configs/womd.json \
 --manifest local/refiner_train_tune.jsonl --stage i2 --seed 42 \
 --workdir work/womd --weights local/weights_womd.json \
 --selection-root work/train_choices/selection
```

Repeat for each seed. The Full/NoRCIM I2 initializer is the same-seed E-matched
I1 with output head zeroed; B I2 uses B-matched I1. The latter only permits a
second residual on targets whose B mode equals A; it is not the same feasible
space as Full. Each matched NoRCIM I2 is separately trained under identical
budget/order. In contrast, `Full_fixed_OFF` must load **Full's unchanged I2**.

For AV2 first use `scripts/preprocess_av2.py` and the local split generation steps in `docs/data.md`. AV2 then uses the same `scripts/prepare_training.py` stages with its config and workdir;
native I1 arms A/E/B, native I2 arms RCIM/NoRCIM/B. Paths are
`prepared/i1/{train,tune}` and `prepared/i2/seedSEED/{train,tune}`. Native trainer
assertions verify exact scene membership and counts.

I1 loss is masked SmoothL1 positions+.25SmoothL1 endpoint+.05residual norm²+
.002residual acceleration², clip5. I2 uses Euclidean position/endpoint errors,
radius-scaled actual residual regularization and clip1. Neither is Conditional CE.
The native WOMD I2 steps the scheduler before tune evaluation; native I1/AV2 step
after evaluation. Those original loops are intentionally preserved. The config
field `patience` inherited from an early I1 protocol does not enable early stop:
the formal matched-refiner loops execute all10epochs/5120updates.

## Table2–11 reproduction entries

| Paper table | Cohort / entry | Notes |
|---|---|---|
|2|WOMD44,097 Validation; `scripts/infer.py` + `scripts/evaluate.py`|A,A_I1,E_I1,B2,NoRCIM,Full; archived `results/womd_validation.json`|
|3|20,000 frozen Training confirmation; same entry|local historical fingerprint-verified IDs; public token recovery or original qualification/exclusion inputs; backbone membership unknown|
|4|paired summary from Tables2/3|same scene draw across three fixed models; original archived CIs retained|
|5|5,000 exposed Training audit; `--methods Full_coord Full_I1 Full_noP2 Full`|shared Full heads; identifies P2 contribution, not retraining|
|6|fixed-head ON/OFF; `--methods Full Full_fixed_OFF`|Full I1/I2 unchanged; differs from matched NoRCIM; RNG20261103|
|7|per-seed summaries from Tables2/3|RCIM42 fixed; descriptive sample SD of three refiner seeds|
|8|AV224,988 Validation; AV2 `infer.py` + `evaluate.py`|focal and all endpoint-valid scored/focal metrics kept separately|
|9|same AV2 predictions|center-distance<1m proxy; all requested pair denominator246,531|
|10|`reproduction/av2_closing/inference_cost.py`|native isolated GPU benchmark,256fit scenes,20warmup,3passes; bundled native interfaces; use stage_native_av2.py then native_av2_cost.py|
|11|`reproduction/external/` native learned and classical entries|adapted forecasting ports; permissions/extra assets pending, see below|

Table10 timings are AV2's actual isolated benchmark. The generic `infer.py` adapter
also computes diagnostic legacy branches and is **not** a replacement latency
measurement. No latency experiment was rerun in this packaging task.

`results/paper_tables.json` is the exact manuscript source snapshot. Official
metric snapshots and checkpoint hashes are references, not a claim that a fresh
clone can already reproduce all tables without authorized weights and datasets.

## Evaluation definitions

ADE is a per-target mean over valid future positions, aggregated only over targets
valid at the endpoint. FDE uses the same endpoint mask:79WOMD,59AV2. Inference
targets and geometric pair denominators do not depend on future validity.
The additional >5m count compares selected final FDE to original A with
threshold5.000001; it is not a missing-rate or collision statistic.

WOMD box overlap uses observed valid vehicle sizes and original oriented-box SAT
with1e-6 penetration tolerance and heading carry-over for<.05m steps. AV2 uses
center distance<1m because dimensions are unavailable. Do not compare these as
the same geometric metric. P1/P2 reject new conflict identities or an increase
in total conflict steps relative to the incoming trajectory set; they do not
guarantee zero inherited conflicts.

Whole-scene paired Bootstrap has5000resamples; model predictions are averaged
within a scene before resampling. Three seed outputs are never15000/3N independent
observations. WOMD's native final summary used RNG20261030 and its saved shard
order. The portable summary fixes lexical scene order: point estimates agree;
finite-resample interval endpoints need not match the archived order-dependent
last decimals. AV2's closing accuracy reconstruction fixes lexical order and
RNG9413; earlier all-target AV2 summary used4201. The archived closing result
preserves the exact focal intervals. AV2 geometric-rate CIs in original results
may include zero; this is not rounded into a significance claim.

## External methods and native-layout scripts

The native postprocessor scripts are provided without copied SmartRefine/SRefiner
neural implementations or their weights, outside the upload set; archived licenses and remaining attribution issues are documented in docs/LICENSING.md.
`reproduction/external/SOURCE_MANIFEST.json` records the exact upstream source
hashes. Do not replace an adapted method with an upstream released-backbone number.
The external baseline scripts retain their explicit container mount layout, documented in
`reproduction/README.md`; they are supplemental entrypoints, not certified by
the core two-fit-scene smoke. Their current asset/dependency blockers are not
hidden behind a fabricated universal launch command.

## Validation performed in this release

See `docs/validation.json` and `docs/PUBLIC_RELEASE_NOTES.md`: syntax, import, config, argument, deterministic
decomposition/geometry tests and a few frozen fit forwards are reported separately
from full experiment reproduction. No optimizer step, full Validation pass or
new threshold selection is part of packaging.

Freshly trained checkpoint files contain packaged-code/protocol metadata, so their
file SHA need not equal the archived binary even with matching model tensors.
Use the published checkpoint hashes for frozen evaluation, and state-dict and
prediction checks for a fresh training reproduction. No fresh training is claimed here.
