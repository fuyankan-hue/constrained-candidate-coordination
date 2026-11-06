"""Path, logging and native-prepared-row adapters only; the training loop is unmodified."""
from pathlib import Path
import json
import torch
from .io import save_json,sha

def configure_womd_i2(native,workdir,seed,arm,weights):
    native.SEED=seed;native.O=Path(workdir);native.ROOT=Path(workdir)/('seed%d'%seed)
    native.BRANCHES=({'B':'B2'}.get(arm,arm),)
    native.i1_path=lambda branch:Path(weights['heads'][str(seed)]['B2' if branch=='B2' else 'Full']['i1'])
    def check():
        cfg=json.loads((native.ROOT/'PROTOCOL_FROZEN.json').read_text())
        for p,h in cfg['frozen_SHA256'].items():assert sha(p)==h
        return cfg
    def load(split,smoke=False):
        rows=[]
        for p in sorted((native.ROOT/('smoke_data' if smoke else 'data')).glob(split+'_*.pt')):
            value=torch.load(str(p),map_location='cpu');rows.extend(value['rows'])
        return sorted(rows,key=lambda r:r['scene'])
    def state(stage,done,total,**kwargs):
        save_json(dict(stage=stage,done=done,total=total,seed=seed,**kwargs),native.ROOT/'progress.json')
    native.check=check;native.load=load;native.state=state
