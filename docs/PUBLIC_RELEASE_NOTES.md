# Public candidate release boundary

This is an independent copy prepared under the project lead's express authorization
to publish original research code. It does not inherit Private Git history and was
not pushed or made Public. Root LICENSE is the unmodified Apache-2.0 text;
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
