# Reproduction environments

Model environments: Python3.8.10, Linux, single visible authorized CUDA device per trainer. WOMD raw indexing/graph processing uses a separate Python3.11.17 CPU-reader environment. The release
core is importable without PyG; only backbone extraction requires it. For a fresh
authorized environment, PyTorch CUDA wheels must be taken from PyTorch's official
wheel index; requirements do not install a private server environment.

* `requirements.txt`: Torch1.13.1+cu117 runtime, NumPy1.19.5, SciPy1.10.1.
* `requirements-scoring.txt`: Torch2.0.1+cu118, NumPy1.24.3, PyG2.3.0,
  Lightning2.0.4, AV2 API0.2.1. CUDA-specific torch-scatter/cluster/sparse wheels
  must match Torch/CUDA. Do not mix Torch1.13 extension binaries with Torch2.
* Native backbone snapshots include their original `environment.yml` as upstream
  dependency context. A current upstream conda file is not a replacement for the
  experiment's exact observed runtime. Observed package metadata is in
  `configs/runtime_versions.json` when available.

WOMD raw decoding uses `requirements-reader.txt`: TensorFlow-CPU2.13.0,
Torch2.0.1+cpu, PyG2.5.2, Waymo package1.6.7 and protobuf4.25.9.
Indexing and graph preprocessing are separate native processes, also avoiding
TensorFlow/PyTorch binary import-order conflicts. AV2 uses Torch2.0.1 throughout;
only WOMD continuous refinement uses Torch1.13.1. AV2 raw processing uses AV2 API and QCNet's
ArgoverseV2 dataset parser. PyG, AV2, protobuf, SciPy and NumPy retain their own
licenses. No dependency installation was performed as part of this task.

Preserve separate WOMD selection and refinement processes. Core CPU smoke checks
cannot certify CUDA bitwise equality across GPU/driver versions. Frozen weights,
source hashes and the official saved-result summaries remain the references.


## Installer validation boundary (2026-11-06 audit)

Do not install the three requirements files together. They are observed-version
records, not fully resolved environment locks. Relevant Torch/NumPy/SciPy/PyG/
Lightning/AV2 package constraints were checked in the existing model environments.
No package was installed or upgraded in this audit.

The installed Waymo1.6.7 SDK metadata requests NumPy1.23.5, pandas1.5.3 and the
`tensorflow==2.13` distribution. The observed reader instead has NumPy1.24.3,
TensorFlow-CPU2.13.0 and no pandas distribution. Its Scenario protobuf imports and
CPU reader interface work, but that does **not** demonstrate a clean resolver install.
The workflow uses Scenario protobufs, not the full SDK metrics stack. A deliberate
proto-only environment rebuild would need separate validation; no untested pip
bypass or silently changed dependency version is presented as an official recipe.
The upstream backbone environment.yml is retained as context, not the exact frozen
installation command. Python3.8 model environments and Python3.11 reader are separate.

## Public candidate dependency scope

All observed top-level runtime versions are in configs/runtime_versions.json.
requirements-scoring.txt includes the observed pandas/Shapely/SciPy/protobuf/
NetworkX/scikit-learn/tqdm and CUDA-specific PyG extension versions. Their wheel
availability is platform-specific; use the original CUDA/PyTorch ABI pairing, not
an automatic CPU/CUDA substitution. Official installation references:
https://pytorch.org/get-started/previous-versions/ and
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html .
These links do not constitute a clean-environment installation test.
Some optional upstream backbone dataset/submission imports need additional Waymo
SDK/proto support; the frozen reader and scoring processes must remain separated.
No missing wheel, reader resolver conflict or external compatibility dependency
was "fixed" by silently replacing the formal environment in this task.
