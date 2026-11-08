# Offline assets for anonymous review

Download the complete review repository ZIP and retain this directory. The
checksum-pinned hexadecimal chunks reconstruct the two frozen checkpoint
archives and the one-way cohort-membership asset. Each text chunk is smaller
than 8 MiB. Hexadecimal encoding avoids embedding personal hosting links and
does not contain author-name text. It is packaging, not a model transformation.

From the repository root:

```bash
python scripts/download_paper_artifacts.py --dataset all
```

Despite its historical filename, this entrypoint now assembles the assets
**offline**. It does not contact a personal GitHub repository or any other host.
It verifies each encoded chunk, the reconstructed archive, and every extracted
checkpoint. Missing/corrupted chunks cause an explicit failure. Do not rename
chunks, edit their hexadecimal text, or remove them from the downloaded archive.

All 40 model tensor files are byte-identical to the previously verified frozen
inference exports. Only the archive's original-author identification is withheld;
third-party notices and dataset/provider agreements are retained unchanged.
Archive filesystem owner/group names and original timestamps are neutralized.
No raw dataset, original scene-ID list, private path or credential is included.

Reconstruction needs about 100 MiB for compressed archives plus about 100 MiB
for extracted models, in addition to the encoded source files. Model use remains
subject to the included notices and provider agreements; see docs/checkpoints.md.
