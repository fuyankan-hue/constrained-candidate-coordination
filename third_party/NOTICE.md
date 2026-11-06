# Third-party provenance and release boundary

`ndpnet_womd/` is the exact source snapshot used by the frozen WOMD backbone.
`ndpnet_av2/` is the locally adapted source snapshot used by the frozen AV2 backbone.
The original implementation is [NDPNet](https://github.com/HuaHu-yizhou/NDPNet),
built on [QCNet](https://github.com/ZikangZhou/QCNet). Both upstream projects
provide Apache License 2.0. Their retained LICENSE files apply to the corresponding
upstream portions; they do not by themselves grant a license to all new project code.
The local experiment archives have no Git commit metadata. We therefore identify
these snapshots by file SHA256, not an invented upstream commit.

Copied native experiment functions are mapped in `docs/source_manifest.json`.
Import rewiring, output-path configuration and CLI adapters are packaging changes;
the network, objective, projection, solver, protection and metric functions are not
replacement implementations. Local AV2 changes include a 50/60-step adapter and
the native target/history interface. NDPNet internal Near/Far stages remain present.

No dataset, checkpoint, downloaded external implementation or private credential
is included. Third-party library dependencies remain separately licensed.
Original project additions: Copyright 2026 Yankan Fu, licensed under Apache-2.0
with the copyright holder's express authorization. This statement does not overwrite
upstream copyrights or licenses, or separate dataset terms. See root NOTICE and
docs/LICENSING.md. No weight release is implied.

The separate upstream QCNet license text is retained at
`licenses/QCNet_LICENSE`, retrieved from the official
https://raw.githubusercontent.com/ZikangZhou/QCNet/main/LICENSE on 2026-11-05.
Its SHA256 is c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4.
This upstream license does not substitute for approval of the new-code license.

The AV2 dataset parser at ndpnet_av2/datasets/argoverse_v2_dataset.py is additionally
retained unmodified from official QCNet commit55cacb418cbbce3753119c1f157360e66993d0d0.
Copyright2023 Zikang Zhou; Apache-2.0. See licenses/QCNet_LICENSE. Its file SHA256 is
851071ad9bd4fef01b2d7ba705b71ea2c1db6dc299a4ea2c1a18d9e756437399.
The release-only offline invocation wrapper does not change its conversion rules.
