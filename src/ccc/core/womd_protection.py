"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import itertools
import numpy as np
from .geometry import selected_geometry

def pairs_for(meta):
    size=meta['size'].numpy().astype(float);ok=np.isfinite(size).all(-1)&(size>0).all(-1)
    return np.array([(i,j) for i,j in itertools.combinations(range(len(size)),2) if ok[i] and ok[j]],int).reshape(-1,2)


def geometry(path,meta,pairs):
    return selected_geometry(path,meta['current'].numpy(),meta['heading'].numpy(),meta['size'].numpy(),pairs)[0]


def safeguard(base,proposal,meta,pairs):
    # Common prediction-only protection, separate from frozenStageM and not novel.
    before=geometry(base,meta,pairs);identity=before.any(-1);budget=int(before.sum())
    for beta in (1.,.5,.25,.125,0.):
        candidate=base+beta*(proposal-base);overlap=geometry(candidate,meta,pairs)
        if not (overlap.any(-1)&~identity).any() and int(overlap.sum())<=budget:return candidate,beta
    raise AssertionError('Identity must be feasible')


def measure(path,base,meta,pairs):
    gt=meta['gt_world'].numpy().astype(float);valid=meta['valid'].numpy().astype(bool);good=valid[:,79]
    err=np.linalg.norm(path-gt,axis=-1);ade=(err*valid).sum(-1)/np.maximum(valid.sum(-1),1)
    overlap=geometry(path,meta,pairs);shift=np.linalg.norm(path-base,axis=-1)
    velocity=np.diff(np.concatenate([meta['current'].numpy()[:,None],path],1),axis=1)/.1
    accel=np.diff(velocity,axis=1)/.1;jerk=np.diff(accel,axis=1)/.1
    return {'targets':int(good.sum()),'fde_sum':float(err[good,-1].sum()),'ade_sum':float(ade[good].sum()),
        'overlap_pairs':int(overlap.any(-1).sum()),'eligible_pairs':len(pairs),'overlap_steps':int(overlap.sum()),
        'shift_sum':float(shift.sum()),'shift_steps':shift.size,'max_shift':float(shift.max()),
        'all_valid_FDE':err[good,-1].tolist()},[np.linalg.norm(z,axis=-1).ravel() for z in (velocity,accel,jerk)]

