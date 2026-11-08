# Environments

The experiments used three Linux environments. Model stages used Python 3.8.10;
the WOMD raw-data reader used Python 3.11.17.

| Environment | Main dependencies | Requirements |
|---|---|---|
| WOMD refinement | PyTorch 1.13.1+cu117, NumPy 1.19.5, SciPy 1.10.1 | `requirements.txt` |
| WOMD backbone/scoring; AV2 preprocessing and all model stages | PyTorch 2.0.1+cu118, NumPy 1.24.3, PyG 2.3.0, Lightning 2.0.4, AV2 API 0.2.1 | `requirements-scoring.txt` |
| WOMD raw indexing/graphs | TensorFlow-CPU 2.13.0, PyTorch 2.0.1+cpu, PyG 2.5.2, Waymo package 1.6.7, protobuf 4.25.9 | `requirements-reader.txt` |

The requirements files record observed versions. They are not one combined
environment or resolver-tested locks. Exact recorded package versions are in
`configs/runtime_versions.json`; the retained backbone `environment.yml` files
provide upstream context rather than the frozen installation recipe.

## Model environments

Obtain CUDA-specific PyTorch wheels from the
[official previous-version instructions](https://pytorch.org/get-started/previous-versions/).
Use matching torch-scatter/cluster/sparse wheels from the
[PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
Do not mix extension binaries between Torch 1.13 and Torch 2.0.

Run WOMD selection and continuous refinement as separate processes in their
respective environments. AV2 uses Torch 2.0 throughout. Core modules do not
require PyG; backbone and dataset interfaces do. Some upstream WOMD imports also
need the Waymo SDK/protobuf dependencies. Trainers expect one visible CUDA device.

## WOMD reader limitation

The installed Waymo 1.6.7 SDK metadata requests NumPy 1.23.5, pandas 1.5.3 and
the `tensorflow==2.13` distribution. The observed reader instead used NumPy
1.24.3, TensorFlow-CPU 2.13.0 and no pandas distribution. Scenario protobuf
imports and the existing CPU reader worked in that environment, but a clean
`pip install -r requirements-reader.txt` has **not** been validated.

The reader uses Scenario protobufs, not the full SDK metrics stack. Indexing and
graph preprocessing run as separate native processes to avoid TensorFlow/PyTorch
import-order conflicts. A fresh proto-only environment needs its own validation;
no dependency override is prescribed here.

## What has been checked

The existing model environments passed the recorded import and small CPU forward
checks. This does not establish a fresh installation, a complete raw-data rebuild
or CUDA bitwise equality across drivers. See `docs/validation.json` for historical
checks and current release status. Optional external methods have their own
source, compatibility and environment dependencies in `reproduction/README.md`.
