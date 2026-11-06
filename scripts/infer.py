"""Two-phase frozen inference; never trains a model or selects a checkpoint."""
import argparse,json
from pathlib import Path
from _bootstrap import ROOT

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True,help='JSONL scene/path/split records')
    p.add_argument('--weights',type=Path,required=True,help='User-local checkpoint paths, never committed')
    p.add_argument('--split',choices=['fit','calibration','tune','audit','confirmation','validation'],required=True)
    p.add_argument('--phase',choices=['selection','refinement'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--methods',nargs='+',default=['A','A_I1','B2','NoRCIM','Full'])
    p.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--limit',type=int,default=None)
    return p

def main():
    args=parser().parse_args()
    import numpy as np
    import torch
    from ccc.io import config,read_manifest,selected_row,save_torch,save_json,sha
    from ccc.runtime import deterministic
    from ccc.inference import load_model,coordinate,predict
    from ccc.core import womd_i2_loss,av2_ops
    from ccc.core.womd_protection import pairs_for
    from ccc.core.womd_bank import make
    from ccc.core.womd_i2_loss import Choice
    cfg=config(args.config);dataset=cfg['dataset'];deterministic(42,1)
    weights=json.loads(args.weights.read_text());records=read_manifest(args.manifest,args.split)
    if args.limit is not None:records=records[:args.limit]
    assert records,'No requested scenes'
    contract=dict(config_SHA256=sha(args.config),weights_manifest_SHA256=sha(args.weights),dataset=dataset,
        scene_ids=[r['scene'] for r in records],source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/ccc').rglob('*.py')})
    marker=args.output/'INFERENCE_PROTOCOL.json'
    if marker.exists():assert json.loads(marker.read_text())==contract,'Inference assets/cohort changed; use a new output directory'
    else:save_json(contract,marker)
    if args.phase=='selection':
        scorer=load_model(weights['rcim'],'rcim',dataset,'cpu')
        for at,record in enumerate(records,1):
            row=selected_row(record,dataset);n=len(row['m1']['pi'] if dataset=='womd' else row['pi'])
            choices=coordinate(row,scorer,dataset) if n else {'empty':True}
            dest=args.output/'selection'/(record['scene']+'.pt')
            if dest.exists():
                old=torch.load(dest,map_location='cpu')
                assert old.get('empty',False)==choices.get('empty',False)
                if not choices.get('empty'):
                    for method in ('A','B','E','RCIM','NoRCIM'):assert old[method]['modes']==choices[method]['modes']
            else:save_torch(choices,dest)
            if at%100==0 or at==len(records):print('SELECTION',at,'/',len(records),flush=True)
        return
    heads={};seed=str(args.seed)
    assert not (args.output/('scene_metrics_seed'+seed+'.json')).exists(),'Never overwrite a completed inference run'
    for method in args.methods:
        if method in ('A','B','E','Full_coord'):continue
        key='Full' if method in ('Full_fixed_OFF','Full_I1','Full_noP2') else method
        spec=weights['heads'][seed][key]
        heads[method]=(load_model(spec['i1'],'refiner',dataset,args.device),
                       load_model(spec['i2'],'refiner',dataset,args.device) if 'i2' in spec else None)
    metrics=[]
    for at,record in enumerate(records,1):
        row=selected_row(record,dataset);choices=torch.load(str(args.output/'selection'/(record['scene']+'.pt')),map_location='cpu')
        if choices.get('empty'):
            metrics.append({'scene':record['scene'],'seed':args.seed,'methods':{m:dict(targets=0,pairs=0,ade_sum=0.,fde_sum=0.,overlap=0,overlap_steps=0,gt5=0) for m in args.methods}})
            continue
        paths={};values={}
        for method in args.methods:
            first,second=heads.get(method,(None,None))
            pred=predict(row,None,first,second,dataset,method,args.device,choices=choices)
            paths[method]=pred['path']
            if dataset=='av2':mt=av2_ops.measure(row,pred['path'],pred['anchor'])
            else:
                bank=make(row,Choice(pred['modes']));meta=bank['evaluation'];good=meta['valid'][:,79].numpy()
                helper=dict(base=pred['coordinated'],meta=meta,pairs=pairs_for(meta),
                            modes=pred['modes'],A_modes=choices['A']['modes'],original_A=pred['anchor'],
                            fdeA=np.linalg.norm(pred['anchor']-meta['gt_world'].double().numpy(),axis=-1)[:,-1])
                mt=womd_i2_loss.stats(helper,pred['path'])
            values[method]=mt
        save_torch({'scene':record['scene'],'seed':args.seed,'paths':paths},args.output/'predictions'/seed/(record['scene']+'.pt'))
        metrics.append({'scene':record['scene'],'seed':args.seed,'methods':values})
        if at%100==0 or at==len(records):print('REFINEMENT',at,'/',len(records),flush=True)
    save_json(metrics,args.output/('scene_metrics_seed'+seed+'.json'))

if __name__=='__main__':main()
