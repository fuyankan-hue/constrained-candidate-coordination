"""Use native training loops; no replacement objective or optimizer implementation."""
import argparse,json,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from _bootstrap import ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--stage',choices=['i1','i2'],required=True)
    p.add_argument('--arm',choices=['A','E','B','RCIM','NoRCIM'],required=True)
    p.add_argument('--seed',type=int,choices=[42,43,44],required=True)
    p.add_argument('--weights',type=Path,help='Native WOMD I2 needs local I1 checkpoint paths')
    p.add_argument('--check-only',action='store_true',help='Validate configuration and stage/arm arguments without starting training')
    a=p.parse_args()
    from ccc.io import config
    cfg=config(a.config)
    assert a.arm in (('A','E','B') if a.stage=='i1' else ('RCIM','NoRCIM','B'))
    if a.check_only:
        print(json.dumps(dict(pass_=True,dataset=cfg['dataset'],stage=a.stage,arm=a.arm,seed=a.seed,training=False,refinement=cfg['refinement'])));return
    if cfg['dataset']=='av2':
        os.environ['CCC_AV2_WORK']=str(a.workdir.resolve())
        from ccc.recipes.av2_refiners import train_one
        train_one(a.arm,a.seed,a.stage);return
    if a.stage=='i1':
        assert a.arm in ('A','E','B')
        from ccc.recipes import womd_i1 as native
        if a.arm=='A':
            from ccc.recipes.womd_a_bank import bankA
            load=native.load
            native.load=lambda root,split,label:[bankA(r) for r in load(root,split,'E42')]
        native.main(a.workdir,{'A':'A','E':'E42','B':'B05'}[a.arm],a.seed,False);return
    assert a.weights and a.arm in ('RCIM','NoRCIM','B')
    from ccc.recipes import womd_i2 as native
    from ccc.training_io import configure_womd_i2
    configure_womd_i2(native,a.workdir,a.seed,a.arm,json.loads(a.weights.read_text()))
    native.train()

if __name__=='__main__':main()
