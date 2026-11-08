# Frozen paper checkpoints and exact cohort recovery

The inference checkpoints for WOMD and AV2 are provided as checksum-pinned
encoded assets under assets/review in this anonymous repository. Download the
complete review archive and retain its directory layout; no personal hosting
service or original-repository visit is needed.
This contains two backbone exports, two RCIM42 exports, and 36 matched Refiner
exports (R1 A/B/E and R2 B/NoRCIM/RCIM, seeds 42/43/44): 40 unique checkpoints.
Full and matched NoRCIM share the same E-matched R1. Fixed-head OFF uses Full's
unchanged R1/R2. No training was performed for this release.

## Model terms and provenance

WOD models are released under the **Waymo Dataset License Agreement for
Non-Commercial Use (March 2025)**. All downstream use/modification is subject to
that Agreement, including Section 4. Each checkpoint archive contains a copy and
a prominent MODEL_NOTICE.txt. Source-code Apache-2.0 does not remove those
restrictions. Argoverse 2 attribution, CC BY-NC-SA 4.0 dataset terms, and the
independent NDPNet/QCNet copyrights remain applicable as described in the notice.
No commercial, production-system or vehicle-operation license is granted.
See https://waymo.com/open/terms/ and https://www.argoverse.org/about.html#terms-of-use.

The archives are **inference exports**, not exact copies of training checkpoint
containers. Every state_dict tensor, including normalization buffers, is bitwise
equal to its original frozen checkpoint. Optimizers, callbacks, private paths and
unused training metadata are removed. Architecture hyperparameters and required
epoch/global_step fields are retained. The original checkpoints are unchanged.
`configs/artifact_release.json` records export-file SHA256, original-file SHA256,
and a canonical tensor digest separately. Original hashes/epochs in
`configs/checkpoint_hashes.json` still identify the historical experiment files.
Export hashes legitimately differ; do not relabel them as original file hashes.
These exports support inference; exact optimizer-state resume is not provided.

## Assemble and verify review assets

From the repository root, using only Python's standard library:

```bash
python scripts/download_paper_artifacts.py --dataset all --output local/paper_artifacts
```

Use `--dataset womd` or `--dataset av2` for one dataset. The entrypoint verifies the encoded chunks, reconstructed archives and every
extracted checkpoint checksum, rejects unsafe archive members,
and refuses to overwrite differing local files. It produces these user-local maps:

* `local/paper_artifacts/womd/weights.json`
* `local/paper_artifacts/av2/weights.json`

Supply the corresponding file to the existing `--weights` CLI; the `backbone`
field points to the exported backbone for `extract_features.py --checkpoint`.
Paths in generated maps are local to your machine and must not be committed.
`stage_native_av2.py` accepts either the original file digest or the exact pinned
inference export mapped to that original digest. Core algorithms are unchanged.

Only load checkpoints obtained from a trusted release and checked against the
pinned digests. Asset checksums establish integrity, not unrestricted usage
rights or a fresh end-to-end reproduction claim.

## Recover the original paper's scene memberships

The encoded review assets also reconstruct `paper-split-hashes.json.gz`: ordered one-way SHA256
membership tokens for the original seven cohort documents. It contains no raw
scene IDs, dataset filenames, records, trajectories, GT, predictions, or private
paths. Tokens encode our research cohort choices, not replacement random splits.
They are not an access-control or anonymization mechanism: someone with the
original dataset can match them locally, which is the purpose of this interface.
Dataset acquisition/use remains governed by the respective provider agreements.

For WOMD, first index your authorized v1.3.0 Training and Validation TFRecords
using `index_womd.py`. This reads metadata; no evaluation or model selection is
needed. Then:

```bash
python scripts/restore_paper_splits.py womd \
  --training-index local/womd_training_raw.jsonl \
  --validation-index local/womd_validation_raw.jsonl \
  --output local/splits
export CCC_SPLIT_ROOT="$PWD/local/splits"
```

For AV2 after the documented preprocessing:

```bash
python scripts/restore_paper_splits.py av2 \
  --processed-root data/av2 --output local/splits
```

Restoration verifies the compressed asset digest, matches only local identifiers,
and validates every original canonical membership/order fingerprint before
writing. WOMD restores RCIM calibration/audit, Refiner train/tune and final
confirmation/Validation cohorts. AV2 restores ordered train/val filenames and
Refiner train/tune. **The historical exclusion union, 64 probe IDs and 84,828
qualification records are no longer external prerequisites for reconstructing
the final cohorts.** They are not redistributed; their effect is already fixed
in the published final-membership tokens. This recovers cohort membership/order,
not the original private exposure-audit records or a new proof of data novelty.

The older `generate_splits.py` remains available for independently re-deriving
cohorts from the original historical inputs; it does not silently replace those
inputs. Both routes must produce the same historical canonical fingerprints.
Never commit the restored local identifiers or raw indexes.

## Verification limits

All 40 exported tensor dictionaries were compared bit-for-bit with their original
files. All seven restored cohorts are checked against the existing paper
fingerprints. These checks do not rerun benchmarks, retrain models, certify a clean
environment install, or resolve external SmartRefine/SRefiner dependencies.
The full raw-data-to-table pipeline on a new machine remains unverified.
