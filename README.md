# Constrained Candidate Coordination and Staged Trajectory Refinement

Code-complete Public release candidate for the EAAI submission **Constrained Candidate
Coordination and Staged Trajectory Refinement for Multi-Agent Interaction
Consistency**. This independent candidate has not been pushed or published. Its Git history does not inherit the Private repository.

## Method and scope

The method coordinates a frozen predictor's six mode-specific trajectories and
then refines the selected continuous paths:

**Frozen NDPNet → RCIM → CJM20 + anchor-centered projection + geometry-first
joint selection → B-spline Refiner 1 + P1 → B-spline Refiner 2 + P2.**

RCIM (Refine-Conditioned Candidate Interaction Modeling) learns candidate-pair compatibility from Refine features and predicted
relative motion. Its 44,449-parameter architecture is unchanged: Near/Far hidden
inputs are zero, while the backbone's Near/Far stages and geometry branches are
retained. Original candidate eligibility is 20 m maximum whole-path deviation;
eligible candidates are radially projected into a 4 m whole-path neighborhood of
the independent unary top-1 anchor. Unary scores are linearly extended and pair
scores bilinearly extended. Geometry-first lexicographic selection preserves the
anchor-relative feasibility contract. Two 76,026-parameter residual refiners and
scene-level protection retain a 5 m whole-path displacement bound from the original
top-1. That is a displacement guarantee, not a bound on absolute FDE.

The added parameter count is **196,501**. Pair-overlap metrics are prediction-side
geometric proxies, not real collisions or closed-loop planning performance.

The core functions are copied from the code that produced the frozen experiments;
packaging changes configure paths, imports and command-line I/O. This is not a new
implementation of the algorithms. `docs/source_manifest.json` records source-file
and per-symbol hashes and declared portability edits. Historical exploration,
Conditional CE, AIR, CJR and FCR are not release defaults.

## Relationship to NDPNet and QCNet

The backbone is [NDPNet](https://github.com/HuaHu-yizhou/NDPNet), built upon
[QCNet](https://github.com/ZikangZhou/QCNet). Exact locally used source snapshots
are retained under `third_party/ndpnet_womd` and `third_party/ndpnet_av2`, including
their original copyright notices and Apache-2.0 LICENSE files. There is no upstream
Git commit in the experiment archives; file hashes identify the actual snapshots.
The AV2 backbone is a historical local adaptation, not claimed to be an official
author-released AV2 checkpoint. See `third_party/NOTICE.md`.

## Environment

Linux / Python **3.8.10**. The frozen WOMD experiment used **two environments**:

| Stage | PyTorch | NumPy | Additional observed dependencies |
|---|---|---|---|
| Backbone / WOMD candidate scoring | 2.0.1+cu118 | 1.24.3 | PyG 2.3.0, Lightning 2.0.4, AV2 API 0.2.1 |
| Continuous-refiner training/inference | 1.13.1+cu117 | 1.19.5 | SciPy 1.10.1 |

AV2 backbone, RCIM, refiner training/inference and the isolated benchmark used
**Torch2.0.1+cu118/NumPy1.24.3**, as confirmed by the native launchers and timing
records. WOMD raw parsing used a third CPU-reader environment: Python3.11.17,
TensorFlow-CPU2.13.0, Torch2.0.1+cpu and Waymo protobuf package1.6.7. Do not silently replace either runtime with a
new Torch version. `requirements.txt` and `requirements-scoring.txt` pin the direct
core dependencies; the retained backbone `environment.yml` and
`docs/environment.md` cover extraction dependencies and CUDA-specific PyG wheels.
No environment was installed or modified while assembling this candidate.

## Data and checkpoint preparation

Obtain WOMD and AV2 yourself under their official terms; no raw data, processed
cache or checkpoint is distributed here:

* [WOMD official download](https://waymo.com/open/download/) and
  [terms](https://waymo.com/open/terms/). Experiments used Scenario TFRecords v1.3.0,
  11 observed and 80 predicted frames at 10 Hz.
* [AV2 official task/download guide](https://argoverse.github.io/user-guide/tasks/motion_forecasting.html),
  50 observed and 60 predicted frames at 10 Hz.

`docs/data.md` specifies raw/processed layouts, native extraction commands and
cache contracts. Dataset-specific scene/file lists are omitted from this Public candidate. Generate the exact local metadata with `scripts/generate_splits.py`; see `docs/data.md`. Never combine a
WOMD oriented-box denominator with the AV2 center-distance denominator.

Checkpoint SHA256 values and selected epochs are in `configs/checkpoint_hashes.json`.
Checkpoint distribution permission and hosting are pending. There is no invented download
link. Copy `configs/weights.example.json` to an ignored local file and fill the
actual checkpoint paths. `Full_fixed_OFF` deliberately uses the same Full I1/I2
heads; matched `NoRCIM` uses its own I2. Loading is strict; the native WOMD I2
`network.` state-dict prefix is an I/O wrapper, not a changed model.

## Training commands (not executed during release preparation)

Run from this repository root. Choose one authorized idle GPU yourself after
checking `nvidia-smi`; a trainer expects exactly one visible GPU. There are no
automatic GPU-selection or other-user termination commands in this release.

First regenerate local metadata as described in `docs/data.md`, then set:

```bash
export CCC_SPLIT_ROOT="$PWD/local/splits"
```

Keep these generated lists out of Git. The CLI checks historical membership/order
fingerprints and refuses mismatches. WOMD historical exclusion/probe/qualification
metadata must be supplied locally; it cannot be reconstructed from a hash alone.
Then prepare native rows as described in `docs/reproduction.md` and run:

```bash
# Original Joint-CE RCIM; same command/config family for AV2.
CUDA_VISIBLE_DEVICES=0 python scripts/train_rcim.py \
  --config configs/womd.json --workdir work/womd \
  --output work/womd/rcim_models --seed 42

# A/E/B matched first heads; repeat each arm for seeds 42, 43 and 44.
CUDA_VISIBLE_DEVICES=0 python scripts/train_refiner.py \
  --config configs/womd.json --workdir work/womd --stage i1 --arm E --seed 42

# Separate RCIM / NoRCIM / B second heads; same-seed frozen first head required.
CUDA_VISIBLE_DEVICES=0 python scripts/train_refiner.py \
  --config configs/womd.json --workdir work/womd --stage i2 --arm RCIM \
  --seed 42 --weights local/weights_womd.json
```

For AV2 use `configs/av2.json`, `work/av2`, and RCIM output
`work/av2/models/rcim_seed42`. Native refiner recipes use the fixed manifests,
architecture and output convention documented in `docs/reproduction.md`.
No public checkpoint is currently available, so the commands cannot reproduce
the published numbers from a fresh clone without separately authorized weights.

## Frozen evaluation

First create a **local** JSONL manifest with `scene`, `path` and `split` from
authorized native caches. WOMD RCIM preparation also requires `shard` and
`record_index`, preserving native scene order. See the data guide; the example
below does not silently guess a cohort.

```bash
# Run WOMD selection in the scoring environment; only prediction-side fields enter it.
python scripts/infer.py --config configs/womd.json \
  --manifest local/audit.jsonl --weights local/weights_womd.json \
  --split audit --phase selection --output work/eval_womd

# Switch to the frozen refiner environment; repeat for seeds 42, 43, 44.
python scripts/infer.py --config configs/womd.json \
  --manifest local/audit.jsonl --weights local/weights_womd.json \
  --split audit --phase refinement --seed 42 --device cpu \
  --methods A A_I1 E_I1 B2 NoRCIM Full --output work/eval_womd

python scripts/evaluate.py --config configs/womd.json \
  --input work/eval_womd --output work/eval_womd/summary.json \
  --main Full --references A A_I1 B2 NoRCIM
```

Use `--device cuda` for refiner inference after checking GPU availability. Dataset
cohorts are explicit: `fit`, `calibration`, `tune`, `audit`, `confirmation`,
`validation`. For AV2 select its config and native cache manifest. These commands
do not authorize changing a method after observing held-out results.

For the fixed-head attribution (Table 6), use
`--methods Full Full_fixed_OFF` with the Full heads for each seed, and summarize
with `--main Full --references Full_fixed_OFF`. This is different from separately
matched `NoRCIM`. For Table 5 use `Full_coord Full_I1 Full_noP2 Full`.

## Verified experimental settings

| Item | Frozen value |
|---|---|
| RCIM | seed42, Joint CE over36 pairs, q=softmax(-r/3), r=relative ADE+.25 relative FDE |
| RCIM training | AdamW 2e-4, weight decay1e-4, clip5, at most8 epochs, patience3, min_delta1e-4 |
| RCIM selection | edge-weighted calibration Joint CE; no audit checkpoint selection |
| Supervised edges | both endpoints valid, observed distance<=75m, common valid future>=60 WOMD /45 AV2 |
| Prediction graph | observed current distance **<75m**; boundary intentionally differs from supervised eligibility |
| Refiner seeds | 42/43/44 with fixed RCIM42; not three independently trained RCIMs |
| Each refiner | 8192 train/1024 tune, batch16, exactly10 epochs/5120 updates |
| Optimizer | AdamW 2e-4, wd1e-4, cosine to2e-5; I1 clip5, I2 clip1 |
| Loss | position+.25endpoint+.05residual L2+.002residual acceleration L2; I1 SmoothL1, I2 Euclidean |
| Model choice | unprotected tune endpoint-valid ADE+.25FDE; epoch0 eligible; improvement>1e-4 |
| Protection | beta in[1,.5,.25,.125,0], common scene-level accepted update |
| Search | exact when product<=65536, otherwise at most8 ascending coordinate sweeps from A |

WOMD RCIM has327,578 usable supervised fit scenes and1,692 usable calibration
scenes from the fixed2,000 calibration IDs. Do not confuse selected scene counts
with scenes containing eligible supervision. AV2 native Training split:159,753 fit,
20,136 calibration,20,019 reserved development; refiner train/tune8192/1024.

## Results and table reproduction

`results/paper_tables.json` preserves the exact eleven manuscript table bodies and
their source SHA256. Numeric official summaries are kept separately for WOMD
audit,20k confirmation,44,097 Validation, AV2 24,988 Validation, closing focal
checks, isolated latency and external methods. These are archived results, not new
experiment claims from this packaging task. See `docs/reproduction.md` for the
Table2–11 entrypoint map, scientific definitions, fairness boundaries and remaining
release gates. Original WOMD Validation had historical exposure; the20k cohort is
not represented as guaranteed-unseen to the pretrained backbone. AV2's backbone
was historically selected using its Validation set. These boundaries are retained.

## Release checks and current limitations

```bash
python scripts/check_release.py
python scripts/check_release.py --imports
python -m unittest discover -s tests
```

The read-only CPU release smoke uses a few existing **fit** records and frozen
weights. It is neither a new full benchmark nor a training reproduction. Training
entrypoints are syntax/import/argument checked, not retrained here. Raw-data
extraction and optional external-source reconstruction require their separate
backbone dependencies and have not been rerun over the datasets during packaging.
`docs/validation.json` reports actual pass/fail coverage, rather than claiming all
experiments were freshly reproduced.

## License, attribution and citation

**Copyright 2026 Yankan Fu.** The copyright holder has expressly authorized release
of the original project contributions under **Apache License 2.0**. Upstream
NDPNet/QCNet/RVO2 copyrights and independent licenses remain. SmartRefine/SRefiner
neural code and weights are not bundled; use their official sources and the
retained archive fingerprints. Third-party ownership and dataset terms remain
separate.

**Dataset conditions remain separate.** This software was made using the Waymo
Open Dataset, provided by Waymo LLC under the Waymo Dataset License Agreement for
Non-Commercial Use, available at https://waymo.com/open/terms/, and your access and
use of such work are governed by the terms and conditions therein. WOMD-developed
trained model architectures and associated derivatives that constitute Distributed
WOD Models are subject to that Agreement and its downstream non-commercial,
production-system and vehicle-operation restrictions. Its full agreement text is
in `LICENSES/Waymo_Dataset_License_March_2025.html`. The Apache license does not
override those conditions; absence of weight files does not waive them.

AV2 data/documentation are CC BY-NC-SA 4.0 and its API code is MIT, per
https://www.argoverse.org/about.html#terms-of-use. No data, feature caches, model
weights, scene IDs or filename inventories are included. See `NOTICE` and
`docs/LICENSING.md`; do not describe this entire stack as unrestricted commercial
Apache software.

```bibtex
@misc{constrained_candidate_coordination,
  author={Fu, Yankan and Li, Luyang and Liang, Haijian and Han, Huiyan
          and Zhao, Rong and Xue, Hongxin},
  year={2026},
  title={Constrained Candidate Coordination and Staged Trajectory Refinement
         for Multi-Agent Interaction Consistency},
  note={Manuscript prepared for Engineering Applications of Artificial Intelligence}
}
```

Authors/year follow the manuscript; DOI/repository URL remain pending. No acceptance,
DOI or hosting URL is implied. Please also cite NDPNet, QCNet, WOMD and AV2 using
their upstream citation information. We acknowledge the original implementations
and dataset providers; retained code headers identify their copyrights.

## Public candidate checks and limits

`PUBLIC_RELEASE_MANIFEST.json` classifies the file set and `RELEASE_SHA256.json`
fingerprints it. `scripts/audit_git.py --require-staged --expect-no-remote` checks
working files, the staged index, every stored object (including unreachable blobs),
refs and reflogs. This directory was created without copying `.git`; no history
from the Private repository, data lists, checkpoint or secret is inherited.

Dataset-specific historical metadata is local-only. Generation is deterministic
and checked against published fingerprints, but an absent historical exclusion or
qualification input is a real reproduction blocker: the tool fails rather than
fabricates the original split or declares a new split equivalent. Aggregate paper
numbers are archived results, not new benchmark results from preparing this release.

The three requirements files describe separate observed runtimes, not one combined
installation or resolver-tested lock. All observed versions are recorded in
`configs/runtime_versions.json` and `docs/environment.md`. The WOMD reader SDK
metadata conflicts with its observed NumPy/package layout; existing protobuf/reader
imports work, but clean `pip install -r requirements-reader.txt` is **not validated**.
No environment installation, optimizer step, full raw extraction or Table2–11
end-to-end rerun was performed here. Optional external learned methods require
their upstream source and original approved compatibility assets. Checkpoints are
not published and no download URL is invented. See `docs/PUBLIC_RELEASE_NOTES.md`.
