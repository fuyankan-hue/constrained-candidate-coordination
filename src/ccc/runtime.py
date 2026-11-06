"""Runtime setup and native training I/O hooks; model functions are copied separately."""
import os,random
import numpy as np
import torch

def deterministic(seed=42,threads=1):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(threads);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)

def refiner_setup():
    deterministic(42,1)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1

def count(model):return sum(p.numel() for p in model.parameters())
