# Release notes and verification scope

This review copy includes the core method, training/evaluation entrypoints,
recorded configurations and paper results. Author details and personal hosting
links are withheld; third-party notices and provider terms are retained.

## Current contents

- Forty frozen inference exports cover WOMD/AV2 backbones, RCIM42 and matched
  R1/R2 heads for refiner seeds 42/43/44.
- `assets/review` contains 34 encoded chunks, each below 8 MiB. The existing
  `scripts/download_paper_artifacts.py` entrypoint assembles them offline and
  verifies chunks, archives and checkpoint files.
- One-way membership tokens recover the seven original cohort documents from
  user-owned dataset metadata. Original membership/order fingerprints are checked.
- The QCNet AV2 parser is retained unmodified from commit
  `55cacb418cbbce3753119c1f157360e66993d0d0`, with its Apache-2.0 notice.
  `scripts/preprocess_av2.py` invokes it offline and rejects partial-output overwrite.

The inference exports retain every frozen model tensor and omit optimizers,
callbacks and private training metadata. Original-container and export hashes
are recorded separately. The formal scientific checkpoints and paper results
have not been changed.

## Historical checks

`docs/validation.json` keeps the original packaging checks under
`historical_packaging_audit`. Statements such as no weights distributed, pending
hosting and no Git push describe that earlier stage, not the current release.
Current distribution information is under `current_release`.

Those checks include source/weight integrity, imports and parameter counts in the
two existing model runtimes, argument checks, synthetic contracts, small frozen
CPU fit forwards and deterministic cohort reconstruction. They do not represent
a new training run or full benchmark. The older `docs/checks.log` referenced in
that record is an internal audit log and is not included in the review archive.

## Remaining limitations

- A clean resolver installation and complete raw-data-to-tables reproduction
  remain unverified; the WOMD reader SDK conflict is described in `environment.md`.
- Historical AV2 graphs do not embed a preprocessing-parser hash, so exact
  raw-to-archived-graph provenance is not retrospectively certified.
- SmartRefine/SRefiner neural and compatibility assets remain external. Table 11
  is not claimed to be independently reproduced from this archive alone.
- Local offline assembly has passed. Complete ZIP delivery and assembly through
  the anonymous hosting service, plus its expiry settings, remain UNVERIFIED
  because the review endpoint was inaccessible during the latest checks.
- Whole-scene bootstrap ordering and historical backbone exposure are documented
  in `reproduction.md`; neither is changed by this maintenance.

AI tools assisted repository packaging and documentation maintenance. This update
does not remove or alter the manuscript's disclosure of AI-assisted preparation.
