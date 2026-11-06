"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import torch

def admissible(pi,world,radius=20.):
    old=pi.argmax(-1);base=world.double()[torch.arange(len(pi)),old]
    allowed=(world.double()-base[:,None]).norm(dim=-1).amax(-1)<=radius
    assert allowed[torch.arange(len(pi)),old].all();return allowed

