# Checkpoints and cohort recovery

The review archive includes 40 frozen inference exports: two backbones, two
RCIM42 models, and 36 refiner models covering R1 A/B/E and R2 B/NoRCIM/RCIM for
seeds 42/43/44. Encoded files are stored under `assets/review`.

## Assemble the assets

From the complete repository archive, run:

```bash
python scripts/download_paper_artifacts.py --dataset all
```

This is an offline assembler; the historical script name is retained. Use
`--dataset womd` or `--dataset av2` to assemble one model family. It checks every
chunk, reconstructed archive and extracted checkpoint against
`configs/artifact_release.json`, and refuses to overwrite mismatching files.

| Dataset | Generated map | Backbone |
|---|---|---|
| WOMD | `local/paper_artifacts/womd/weights.json` | `local/paper_artifacts/womd/checkpoints/backbone.ckpt` |
| AV2 | `local/paper_artifacts/av2/weights.json` | `local/paper_artifacts/av2/checkpoints/backbone.ckpt` |

Use the generated map with `--weights` and the corresponding backbone with
`extract_features.py --checkpoint`. The maps contain machine-local absolute
paths and stay outside Git. If `--output` is changed, use the maps below that
directory and pass its membership archive explicitly to split recovery:

```bash
python scripts/download_paper_artifacts.py --dataset all --output local/custom_assets
python scripts/restore_paper_splits.py womd \
  --memberships local/custom_assets/paper-split-hashes.json.gz \
  --training-index local/womd_training_raw.jsonl \
  --validation-index local/womd_validation_raw.jsonl --output local/splits
```

Full and matched NoRCIM share E-matched R1, but use different R2 weights.
`Full_fixed_OFF` reuses both Full heads without retraining. Native AV2 staging
accepts either an original checkpoint digest or its pinned inference-export digest.

## Export contents and terms

All model tensors, including normalization buffers, are unchanged from the frozen
checkpoints. Optimizers, callbacks, private paths and unused training metadata
are omitted. Required architecture and epoch/global-step fields remain. Exports
support inference, not an exact optimizer-state resume.

`configs/checkpoint_hashes.json` identifies the original experiment containers.
`configs/artifact_release.json` records export-file hashes, original-file hashes,
canonical tensor hashes and archive hashes separately. The two file hashes differ
because the containers contain different metadata.

Each model archive includes `MODEL_NOTICE.txt`, the Waymo Agreement and the
NDPNet/QCNet license copies. WOD models are subject to the Waymo non-commercial
terms, including downstream restrictions. AV2 attribution and dataset rights
remain separate. See [LICENSING.md](LICENSING.md). Checksums establish integrity,
not additional usage rights.

## Recover the original cohorts

`paper-split-hashes.json.gz` encodes the original seven membership documents as
ordered one-way SHA256 tokens. It contains no raw scene IDs, filenames, labels or
trajectories. Match the tokens against your legally obtained corpus:

```bash
python scripts/restore_paper_splits.py womd \
  --training-index local/womd_training_raw.jsonl \
  --validation-index local/womd_validation_raw.jsonl --output local/splits
python scripts/restore_paper_splits.py av2 \
  --processed-root data/av2 --output local/splits
export CCC_SPLIT_ROOT="$PWD/local/splits"
```

Use only the command for the dataset you prepared. WOMD needs complete v1.3.0
Training/Validation raw indexes; AV2 needs its complete processed train/val corpus.
See [data.md](data.md) for their preparation.

Restoration checks every member and its order against the original canonical
fingerprints before writing. It recovers WOMD calibration/audit, refiner
train/tune and final cohorts, plus AV2 ordered files and refiner train/tune.
Private exclusion, probe and qualification inventories are not required for this
route. The older `generate_splits.py` independently derives memberships when those
historical inputs are available. Both routes must match the same fingerprints.

Tokens are a matching mechanism, not access control or anonymization of a corpus
already held by the recipient. Restored identifiers remain local. The recovery
does not reconstruct the private exposure audit or establish backbone novelty.

## Verification scope

Local offline assembly and all 40 checkpoint-file checks have passed. Original
export tensor comparisons and seven-cohort recovery checks are recorded in
[validation.json](validation.json). Anonymous-host ZIP delivery, fresh environment
installation and a complete raw-data-to-table rerun are separate checks and are
not implied by local assembly.
