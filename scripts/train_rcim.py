"""Invoke the preserved native Joint-CE recipe on user-prepared native records."""
import argparse,os,json
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from _bootstrap import ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[42],default=42)
    p.add_argument('--check-only',action='store_true',help='Validate frozen settings and arguments; no training/import of a trainer')
    a=p.parse_args()
    from ccc.io import config
    cfg=config(a.config)
    if a.check_only:
        print(json.dumps(dict(pass_=True,dataset=cfg['dataset'],seed=a.seed,training=False,RCIM=cfg['RCIM'])));return
    assert not (a.output/'refine_only42_best.pt').exists() and not (a.output/'best.pt').exists(),'Use a new output directory'
    if cfg['dataset']=='womd':
        os.environ['CCC_WOMD_WORK']=str(a.workdir.resolve());os.environ['CCC_MODEL_OUTPUT']=str(a.output.resolve())
        from ccc.recipes import womd_rcim as native
        native.main('refine_only',42,False)
    else:
        os.environ['CCC_AV2_WORK']=str(a.workdir.resolve())
        from ccc.recipes import av2_rcim as native
        expected=a.workdir/'models/rcim_seed42'
        assert a.output.resolve()==expected.resolve(),'Native AV2 output is workdir/models/rcim_seed42'
        native.train()

if __name__=='__main__':main()
