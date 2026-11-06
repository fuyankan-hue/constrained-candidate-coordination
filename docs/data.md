# Data preparation and native cache schema

Acquire datasets from the official links in README, under their terms. WOMD data
has separate noncommercial terms; AV2 datasets are distributed under their stated
CC BY-NC-SA4.0 terms. No dataset license is replaced by the code license. Do not
commit raw files, processed graphs, feature caches, predictions or local manifests
containing absolute paths. Scene-ID membership lists are not redistributed; only deterministic selection code and fingerprints are provided.

## WOMD

Formal raw layout is `DATA/scenario/training/raw/training.tfrecord-00000-of-01000`
and `.../validation/raw/validation.tfrecord-00000-of-00150`, v1.3.0 Scenario format.
The exact native indexer is `ccc.data.womd_index`: it CRC-checks all records,
parses protobufs, records byte offsets and payload SHA256, and validates requested
track IDs and91-frame histories. Use `scripts/index_womd.py` to invoke it and
produce a **local** raw-record manifest. No indexing was rerun in this task.

```bash
python scripts/index_womd.py --raw-root data/womd/scenario \
  --split training --workdir work/womd_index --output local/womd_training_raw.jsonl
python scripts/preprocess_womd.py --raw-root data/womd/scenario \
  --manifest local/womd_training_raw.jsonl --split training --output work/womd_graphs
# Switch from the CPU reader to the Python3.8 backbone environment.
CUDA_VISIBLE_DEVICES=0 python scripts/extract_features.py \
  --preprocessed-root work/womd_graphs --config configs/womd.json --checkpoint checkpoints/womd_backbone.ckpt \
  --raw-root data/womd/scenario --manifest local/womd_training_raw.jsonl \
  --split training --workdir work/womd
```

The extraction uses the retained dataset parser, the exact two-point map-arc
interpolation safeguard, TargetBuilder(11,80), GT-free backbone forward and actual
Near/Far/Refine hooks. It predicts all official requested targets, including those
whose historical vector validity differs from current presence. It never uses
future validity to define the inference target set.

Each native file is `cache/training/SHARD/SCENE.pt`, with:

* `scene`, `checkpoint_sha256`, `track_ids`, `cache_contract`;
* `m1`: `world[N,6,80,2]`, `pi[N,6]`, `original_agent_id[N]`, native unary/pair
  features, `gt[N,80,2]`, `valid[N,80]`, `valid8=valid[:,79]`, `fde[N,6]`;
* `m2`: `near/far/refine[N,6,128]`, `world`, `pi`, `current[N,2]`, `target_ids`,
  `context_ids`, matching m1 original context-index IDs exactly;
* `history`: the11 observed frames only, including position,velocity,heading,
  validity,length,width,type. Actual WOMD track IDs are recorded separately from
  context tensor indices; do not interchange them.

Training fit is `int(sha256(scene)[:8],16)%5!=0`. Local generation below
reconstructs RCIM calibration/audit, matched-refiner train/tune, and optionally
final cohorts, using the original namespaces and ordering. Historical exclusion
inputs are required; no new random split is substituted.

For existing caches `scripts/make_manifest.py` indexes scene/path mappings using
an explicit scene list. For RCIM training, keep `shard` and `record_index` from
the raw manifest and use `fit`, `calibration`, `audit` split labels; the native
prepared order is shard then record order. Keep empty-supervision scenes in the
index but exclude them through the original eligibility function `item` retained in `src/ccc/core/womd_labels.py`.

## AV2

The AV2 graph parser comes from QCNet, not the NDPNet WOMD dataset class. The
now-bundled `third_party/ndpnet_av2/datasets/argoverse_v2_dataset.py` is unmodified
QCNet commit `55cacb418cbbce3753119c1f157360e66993d0d0`, SHA256
`851071ad9bd4fef01b2d7ba705b71ea2c1db6dc299a4ea2c1a18d9e756437399`.
It matches the parser file in the existing local QCNet preprocessing environment.
Official pinned source:
https://github.com/ZikangZhou/QCNet/blob/55cacb418cbbce3753119c1f157360e66993d0d0/datasets/argoverse_v2_dataset.py .
Its original Apache-2.0 copyright is retained; the corresponding license copy is
`third_party/licenses/QCNet_LICENSE`. Raw AV2 data and the av2-api dependency are
separate: use AV2 API0.2.1 in the documented Torch2 scoring/backbone environment.
The historical .pkl archive does not embed a preprocessing-parser SHA, so exact
raw-to-archived-graph reconstruction is not retrospectively certified here.

Download/unpack official AV2 scenarios yourself into
`data/av2/train/raw/SCENE/` and `data/av2/val/raw/SCENE/`, each containing its
scenario parquet and map JSON. The native parser uses50observed/60future steps,
dim3, predict_unseen_agents=False and vector_repr=True. The thin offline entry
loads the file directly, avoiding the NDPNet dataset package's WOMD-only export:

```bash
# Import/default check only; reads no dataset and runs no preprocessing.
python scripts/preprocess_av2.py --root data/av2 --split train --check-only
# Explicit preprocessing of user-owned raw data, not run in this release repair.
python scripts/preprocess_av2.py --root data/av2 --split train
python scripts/preprocess_av2.py --root data/av2 --split val
python scripts/generate_splits.py av2 --processed-root data/av2 --output local/splits
```

The offline entry requires a complete raw split and an empty output directory
(or skips a complete existing graph split). It disables upstream automatic
download/extraction and refuses to overwrite partial processed output. The parser
writes `data/av2/train/processed/SCENE.pkl` and the analogous val tree. No parser
algorithm has been replaced. Actual end-to-end raw preprocessing remains untested
in this documentation repair.

The exact native ordered filenames are regenerated from user-owned processed
files. The generator verifies the original ordered-list fingerprints before writing
local files. Preserve that order for the native RCIM shuffling protocol, then:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/extract_features.py \
  --config configs/av2.json --checkpoint checkpoints/av2_backbone.ckpt \
  --raw-root data/av2 --manifest local/splits/av2_train_files.txt \
  --split train --workdir work/av2
```

The actual frozen Extractor applies AV2QCNetAdapter(50,60), TargetBuilder(50,60),
zeros future backbone fields, and selects observed-current category>=2 targets.
It retains actual agent IDs, context indices and focal-category flags. It checks
the historical local backbone epoch4/global_step124945. Numeric64-scene shards
and their row order must be preserved for exact native RCIM shuffle order.

AV2 rows contain world[N,6,60,2],pi[N,6],near/far/refine[N,6,128],current,heading,
the last11 observed history frames, observed_type,is_focal,original_agent_id,
context_ids,gt,valid. No observed vehicle dimensions are invented. The native
geometry proxy is center distance<1m, not WOMD boxes.

AV2 native Training membership: SHA256(scene) first8 hex modulo10:0calibration,
1reserved development,2..9fit. Refiner train8192/tune1024 IDs are deterministically regenerated locally.
Native Validation has24,988 scenes and is not used to choose added-module epochs.
Backbone historical Validation exposure is explicitly retained in the metadata.

## Manifest contract and safety

The portable inference manifest is JSONL:
`{"scene":"SCENE","path":"relative/or/user-local/cache.pt","split":"fit"}`.
A relative path resolves against the manifest directory. Duplicate scene IDs,
missing scene-to-cache matches and target-ID mismatches fail rather than guess.
Only unpickle caches/checkpoints from sources you trust; torch.load/pickle are not
safe input formats for arbitrary untrusted files.

## Local deterministic membership reconstruction

No original scene-ID or filename list is distributed. The expected counts and
ordered canonical SHA256 fingerprints are in `configs/split_rules.json`.
This is not a replacement dataset split. The script writes nothing if fingerprints
do not match the paper's historical membership and ordering.

AV2 reads only native processed basenames (no pickle/trajectory labels):

```bash
python scripts/generate_splits.py av2 --processed-root data/av2 --output local/splits
```

It expects 199,908 train and 24,988 val files, sorted lexically. Hash(scene) modulo10
defines fit/calibration/development; the exact `AV2_refiner_fit` and
`AV2_refiner_tune` namespace rankings select8192/1024, stored lexically as in the
original deployment code.

For WOMD first create the raw Training index using index_womd.py. Supply your own
authorized historical metadata, kept outside the public file set:

```bash
python scripts/generate_splits.py womd --raw-index local/womd_training_raw.jsonl \
 --known-exposure local/known_previous_scenes.json \
 --old-probe local/old_probe_scene_ids.json --output local/splits
export CCC_SPLIT_ROOT="$PWD/local/splits"
```

`local/known_previous_scenes.json` is the exact historical exclusion union from the
original experiment. `local/old_probe_scene_ids.json` contains the64 pre-Stage-R probe
IDs from the original protocol. Both accept a JSON list; a `scene_ids` list or
`records` list with scene fields is also accepted. Neither list is bundled.
The raw index must preserve shard/record_index and observed requested-target counts.
WOMD RCIM calibration2000/audit5000 are namespace-ranked after exclusion. Refiner
train8192/tune1024 are the original StageR corrected-fit ranked slices, excluding
known history and those64 probe scenes. These are development/training cohorts,
not claims that the pretrained backbone never saw them.

To also regenerate Table3's final cohorts, use a new output directory and add
`--confirmation-qualified local/qualified_confirmation_records.json` and
`--validation-index local/womd_validation_raw.jsonl`. The first input must contain
the original84,828 structurally qualified, exposure-audited Training candidates
from the H protocol. It is not a list selected by prediction error. The generator
selects20,000 using the original confirmation namespace and tie-break, and orders
Validation by shard/record_index. Do not merely hash all non-fit scenes and call
that the original confirmation cohort. Without that provenance/qualification
input, exact Table3 reproduction is not available from a fresh raw download alone.
No future GT, valid8 or FDE is used by the metadata selection code.

Generated outputs are ignored under `local/`; `CCC_SPLIT_ROOT` routes preparation
and native-layout staging to those files. Supplying an existing authorized exact
local copy is also possible, but redistribution permission is not implied.
Never commit generated metadata, raw indices, cache-path manifests or data lists.
