# Public candidate release boundary

This research-code repository is published publicly at
https://github.com/fuyankan-hue/constrained-candidate-coordination (main).
It was prepared independently under the project lead's express authorization and
does not inherit the Private research repository's Git history. Statements that no
push occurred in the initial preparation audit describe that earlier audit only. Root LICENSE is the unmodified Apache-2.0 text;
NOTICE and docs/LICENSING.md define third-party and dataset-term scope.

Copyright 2026 Yankan Fu. The copyright holder has explicitly confirmed the right
to release the original project code under Apache License 2.0. This confirmation
does not replace third-party licenses or the separate terms for WOMD architectures
and derivatives. Waymo's agreement and its copy remain included.

All seven historical scene/file inventories, fit smoke IDs, and the256-scene timing
list are omitted. Aggregate metrics and protocol scalar values are unchanged.
The deterministic split generator must reproduce expected historical fingerprints;
missing local exclusion/qualification metadata is explicitly a reproduction blocker.
Code for every core method, WOMD/AV2 processing/training/inference/evaluation and
adapted external interfaces is retained. The core algorithm files are checked
byte-for-byte against the audited Private candidate; only release I/O wrappers,
metadata generation and documentation are changed.

No checkpoints or raw/processed data are distributed. A future weight publication
requires its own provider/upstream permission and hosting decision. SmartRefine/
SRefiner neural sources and the original SRefiner compatibility archive remain
external dependencies; exact source hashes and acquisition links are retained.

Clean environment installation, raw-data-to-results reconstruction, fresh training,
CUDA bitwise reproduction and all Tables2-11 from scratch are not verified. Existing
runtime syntax/import/CLI/unit checks and deterministic metadata reconstruction are
reported separately. No training, held-out model evaluation or threshold search was
performed during this preparation. No new scientific result is asserted.

## Minimal documentation/AV2 entry repair (2026-11-06)

The missing AV2 parser is now supplied unmodified from QCNet commit
55cacb418cbbce3753119c1f157360e66993d0d0, matching the local QCNet preprocessing
environment file SHA256. Its original Apache-2.0 header remains. The new
scripts/preprocess_av2.py wrapper invokes that parser offline without changing its
agent/map conversion, checks the native50/60-step defaults, and refuses automatic
downloads or replacement of partial graph output. Historical preprocessed .pkl
files do not record a preprocessing-parser hash; this addition is not a claim that
every archived graph has retrospectively proven byte-identical provenance.
No raw-data processing, model inference, optimizer step or experiment rerun is part
of this repair. Existing audit evidence remains historical and is not relabelled
as a full fresh-environment reproduction.
