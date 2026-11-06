# License scope and copyright attribution

Copyright 2026 Yankan Fu.

The copyright holder has expressly confirmed ownership and authorization to release
the original project contributions under Apache License 2.0. `LICENSE` contains the
unmodified standard license text; `NOTICE` identifies its scope. Third-party code
and dataset-derived rights remain separately governed. This does not grant an
unrestricted commercial license for the whole research stack.

## Third-party components

* NDPNet and QCNet snapshots: original Apache-2.0 portions and copyright notices
  remain, with local adaptation/source hashes recorded in source_manifest.json.
* RVO2 ORCA port: original Apache-2.0 header and modification notice remain.
* SmartRefine: official archived license Apache-2.0; neural implementation and
  weights are not bundled. Obtain it from https://github.com/opendilab/SmartRefine
  and match reproduction/external/SOURCE_MANIFEST.json before native reproduction.
* SRefiner: archived MIT text has literal upstream copyright placeholders. It is
  preserved without inventing attribution. Neural and compatibility sources and
  weights are not bundled. Obtain the approved source from
  https://github.com/Liwen-Xiao/SRefiner and clarify attribution before vendoring.
* Original wrappers/integration changes are identified separately from upstream
  code in PUBLIC_RELEASE_MANIFEST.json. An upstream license is not falsely
  attributed to an entirely original algorithm or to dataset content.

## Dataset agreements are separate

Waymo's March 2025 Agreement defines Derivative IP broadly, including model
architectures/software/algorithms developed using its dataset. Sections 1, 2(c)
and 4 require attribution, prominent downstream non-commercial notice, a copy of
the Agreement for Distributed WOD Models, and restrictions on commercial,
production and vehicle-operation uses. We apply that notice to the WOMD-developed
architecture and related model-derived portions conservatively, even though no
weights are published. These conditions are not added to or erased from the Apache
text. See NOTICE and LICENSES/Waymo_Dataset_License_March_2025.html, and the
controlling source https://waymo.com/open/terms/.

The official FAQ https://waymo.com/open/faq/ describes public training-code release
for research reproduction as an acceptable example. This does not imply unrestricted
deployment or that no-weights means the Agreement is irrelevant.

AV2 data/documentation use CC BY-NC-SA 4.0; code/API use MIT, as stated at
https://www.argoverse.org/about.html#terms-of-use. Apache does not relicense AV2
data or grant commercial rights to dataset-derived artifacts. No raw samples,
processed tensors, trajectories, maps, checkpoints, scene lists or filename
inventories are included here. Rights for any future artifact publication must be
checked separately under its applicable dataset and upstream terms.

No general public-redistribution permission for individual scene-ID/filename
inventories was established in this audit. All seven historical lists are omitted,
including AV2 lists, rather than assuming identifiers are automatically unrestricted.
Only code, scalar protocols, aggregate paper results and fingerprints remain.
Deterministic scripts regenerate local metadata from authorized user-owned inputs.

The original-code copyright attribution is confirmed as Yankan Fu, 2026. The
separate dataset-term scope above continues to apply. Broader commercial
distribution would require the relevant provider permission; this release does
not grant it.
