"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import torch
import numpy as np
from types import SimpleNamespace
from . import solvers as P
from .solvers import select,geometry_arrays,solve
from .womd_geometry import tables
from .admissible import admissible
I=SimpleNamespace()

def selection(obs,model):
    a,b=obs['m1'],obs['m2'];n=len(a['pi'])
    edges=[(i,j) for i in range(n) for j in range(i+1,n) if float((b['current'][i]-b['current'][j]).norm())<75]
    with torch.inference_mode():C=model(*[b[k] for k in ('near','far','refine','world')],torch.tensor(edges,dtype=torch.long).reshape(-1,2))
    r=dict(base=a['pi'].argmax(-1).tolist(),pi=a['pi'].double().numpy(),edges=edges,
           allowed=admissible(a['pi'],a['world'],20.).numpy(),C_old=C)
    vp,rp,alpha=P.project(obs,r);rc,rd=select(vp,rp,'main');nc,nd=select(vp,rp,'NoRCIM')
    t=tables(obs);hard,G=geometry_arrays(t,edges,n)
    bm,bd=solve(r['pi'],.5*G,edges,r['allowed'],r['base'],hard=None,max_sweeps=8)
    return dict(projected_world=vp['m1']['world'],alpha=alpha,choices=dict(RCIM=rc,NoRCIM=nc,B2=bm),
                original_A=r['base'],search=dict(RCIM=rd,NoRCIM=nd,B2={k:v for k,v in bd.items() if k not in ('trace','blocked','updates','reject')}),
                prediction_only=True,requested=n)

