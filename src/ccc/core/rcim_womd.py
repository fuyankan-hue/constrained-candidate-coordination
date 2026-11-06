"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import torch
from .motion_model import MotionNFI

class Model(MotionNFI):
    def __init__(self,arm):super().__init__(rich=arm!='padded');self.ablation=arm
    def forward(self,near,far,refine,world,edges):
        if self.ablation in ('no_near','geometry_only','refine_only'):near=torch.zeros_like(near)
        if self.ablation in ('no_far','geometry_only','refine_only'):far=torch.zeros_like(far)
        if self.ablation in ('no_refine','geometry_only'):refine=torch.zeros_like(refine)
        return super().forward(near,far,refine,world,edges)

