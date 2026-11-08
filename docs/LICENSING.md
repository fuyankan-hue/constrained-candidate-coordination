# Licenses and attribution

Copyright 2026 the original authors. Author identification is withheld for
anonymous review; ownership and license terms are unchanged.

Original project code is licensed under Apache-2.0 (`LICENSE`). Third-party
copyrights, model terms and dataset agreements have their own scope; see `NOTICE`.

## Third-party code

| Component | License and source |
|---|---|
| NDPNet and QCNet snapshots | Original Apache-2.0 notices are retained in `third_party/`; adaptations and hashes are listed in `docs/source_manifest.json`. |
| RVO2-derived ORCA port | Apache-2.0; original header and modification notice retained. |
| SmartRefine | Archived upstream Apache-2.0 license retained. Neural source and weights are not bundled; obtain them from https://github.com/opendilab/SmartRefine and match `reproduction/external/SOURCE_MANIFEST.json`. |
| SRefiner | Archived MIT text is retained, including its upstream placeholders. Neural source, compatibility files and weights remain external; obtain them from https://github.com/Liwen-Xiao/SRefiner and clarify attribution before redistributing them. |

Original wrappers are identified separately in `PUBLIC_RELEASE_MANIFEST.json`.
Upstream licenses are not applied to unrelated project code or dataset content.

## Models and datasets

The [Waymo Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms/)
(March 2025) governs WOMD-derived models and related work. Its attribution,
downstream notice, agreement-copy and non-commercial conditions remain applicable
to the included inference exports. Each archive contains the Agreement and
`MODEL_NOTICE.txt`. Source-code Apache-2.0 does not replace these conditions or
grant commercial, production-system or vehicle-operation rights.

[Argoverse terms](https://www.argoverse.org/about.html#terms-of-use) provide data
and documentation under CC BY-NC-SA 4.0 and API code under MIT. AV2 attribution,
upstream code rights and applicable dataset-derived rights remain separate.

Raw samples, processed data, trajectories, maps, raw scene-ID/file inventories and
private paths are not distributed. The encoded review assets contain frozen
inference models and one-way tokens for research cohort choices. Tokens can be
matched against an authorized local corpus; they are not access control.
See `docs/checkpoints.md` for assembly and recovery.
