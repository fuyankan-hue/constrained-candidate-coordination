"""Assemble native prepared rows; forward only, no optimizer or weight updates."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from _bootstrap import ROOT,SPLIT_ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--stage',choices=['rcim','i1','i2'],required=True)
    p.add_argument('--weights',type=Path)
    p.add_argument('--selection-root',type=Path,help='Saved choices generated in the frozen scoring environment')
    p.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    a=p.parse_args()
    from ccc.io import config,read_manifest,selected_row,save_torch,save_json,sha
    from ccc.runtime import deterministic
    import numpy as np
    import torch
    cfg=config(a.config);dataset=cfg['dataset'];deterministic(42,1)
    records=read_manifest(a.manifest);a.workdir.mkdir(parents=True,exist_ok=True)
    ids={s:{r['scene'] for r in records if r['split']==s} for s in ('fit','train','calibration','tune','audit')}
    assert not (ids['fit']|ids['train'])&(ids['calibration']|ids['tune']|ids['audit'])
    if a.stage=='rcim' and dataset=='womd':
        from ccc.core.womd_labels import item
        # Same shard then record order as the frozen nfi_prepare.py, not scene sorting.
        records.sort(key=lambda r:(int(r['shard']),int(r['record_index'])))
        protocol=a.workdir/'experiment_protocol.json'
        if not protocol.exists():save_json(cfg,protocol)
        by={}
        for rec in records:by.setdefault(int(rec['shard']),[]).append(rec)
        counts=dict(fit=0,calibration=0,audit=0)
        for shard,rr in sorted(by.items()):
            dest=a.workdir/'nfi_data'/('%04d.pt'%shard);assert not dest.exists(),'Use a new preparation workdir'
            pack={k:[] for k in counts}
            for rec in rr:
                out=item(selected_row(rec,dataset))
                if out is not None:pack[rec['split']].append(out);counts[rec['split']]+=1
            save_torch(dict(protocol_sha256=sha(protocol),data=pack,source_scenes=len(rr)),dest)
        assert counts['fit']==327578 and counts['calibration']==1692,counts
        save_json(counts,a.workdir/'prepared_counts.json');return
    if a.stage=='rcim':
        # Preserve original numeric shard names, row order, and paths in AV2's native loop.
        # Features are prepared by extract_features.py directly into workdir/cache/train.
        from ccc.recipes import av2_rcim as recipe
        recipe.ROOT=a.workdir
        assert (a.workdir/'cache/train').is_dir()
        save_json(dict(cfg,RCIM=dict(cfg['RCIM'],loss='original_joint_ce'),
            splits=dict(fit_count=159753)),a.workdir/'AV2_MODULE_PROTOCOL.json')
        observed={r['scene'] for r in recipe.rows_for('fit')}
        assert len(observed)==159753
        # A forward/loss smoke, no backward; marker records precisely what was checked.
        model=recipe.RCIM().eval();row=next(recipe.rows_for('fit'))
        with torch.no_grad():value,_=recipe.loss(model,row,'cpu')
        assert value is not None and torch.isfinite(value)
        save_json(dict(passed=True,forward_loss_finite=True,backward_not_executed=True),a.workdir/'RCIM_AV2_TRAINING_SMOKE_PASS.json')
        save_json(dict(passed=True,fit_ids=len(observed)),a.workdir/'FEATURES_COMPLETE.json');return
    assert a.weights,'Frozen scorer/I1 weights required for training-row preparation'
    weights=json.loads(a.weights.read_text())
    from ccc.inference import load_model,coordinate
    from ccc.core.womd_bank import make
    from ccc.core.womd_i2_loss import Choice
    from ccc.core import av2_ops
    from ccc.core import womd_i2_loss as native
    from ccc.core.womd_native_row import native_row
    assert dataset!='womd' or a.selection_root,'WOMD choices must be saved by infer.py in the Torch2 scoring environment'
    scorer=load_model(weights['rcim'],'rcim',dataset,'cpu') if not a.selection_root else None
    if a.stage=='i2':
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        first={arm:load_model(weights['heads'][str(a.seed)][arm]['i1'],'refiner',dataset,'cuda') for arm in ('Full','B2')}
    splits=json.loads((SPLIT_ROOT/'womd_refiners.json').read_text()) if dataset=='womd' else {
        s:(SPLIT_ROOT/('av2_refiner_'+s+'.txt')).read_text().splitlines() for s in ('train','tune')}
    sets={s:set(v) for s,v in splits.items()};assert len(sets['train'])==8192 and len(sets['tune'])==1024 and not sets['train']&sets['tune']
    lookup={r['scene']:r for r in records};assert (sets['train']|sets['tune'])<=set(lookup),'Missing refiner scene cache records'
    for split in ('train','tune'):
        for chunk,at in enumerate(range(0,len(splits[split]),32)):
            pack=[]
            for sid in splits[split][at:at+32]:
                row=selected_row(lookup[sid],dataset)
                ch=torch.load(a.selection_root/(sid+'.pt'),map_location='cpu') if a.selection_root else coordinate(row,scorer,dataset)
                if dataset=='womd':
                    banks={label:make(row,Choice(ch[arm]['modes'])) for label,arm in [('E42','E'),('B05','B')]}
                    if a.stage=='i1':pack.append(dict(scene=sid,banks=banks));continue
                    vp=dict(row,m1=dict(row['m1'],world=ch['projected_world']))
                    recipe=dict(scene=sid,cache_SHA256=sha(lookup[sid]['path']))
                    variants={arm:native.row_for(vp,recipe,ch[arm]['modes'],first['Full']) for arm in ('RCIM','NoRCIM')}
                    variants['B2']=native_row(row,banks['B05'],first['B2'])
                    pack.append(dict(scene=sid,variants=variants))
                elif a.stage=='i1':
                    pack.append(dict(scene=sid,arms={arm:av2_ops.bank(row,ch[arm]['path'],ch[arm]['modes']) for arm in ('A','E','B')}))
                else:
                    arms={};anchor=ch['A']['path']
                    for arm in ('RCIM','NoRCIM','B'):
                        z=ch[arm];head=first['B2' if arm=='B' else 'Full']
                        from ccc.core.refiner_av2 import collate
                        path,beta,_=av2_ops.refine(row,z['path'],z['modes'],head,collate,'cuda')
                        radius=(np.asarray(z['modes'])==np.asarray(ch['A']['modes'])).astype(np.float32) if arm=='B' else np.clip(5.-1e-4-np.linalg.norm(path-anchor,axis=-1).max(1),0.,1.).astype(np.float32)
                        bank=av2_ops.bank(row,path,z['modes']);bank.update(radius=torch.as_tensor(radius),firstP_beta=beta);arms[arm]=bank
                    pack.append(dict(scene=sid,arms=arms))
            root=a.workdir/('data' if a.stage=='i1' else 'seed%d/data'%a.seed) if dataset=='womd' else a.workdir/'prepared'/a.stage/('seed%d'%a.seed if a.stage=='i2' else '')/split
            dest=root/('%s_%04d.pt'%(split,chunk));assert not dest.exists(),'Use a new preparation workdir'
            save_torch(dict(rows=pack) if a.stage=='i2' and dataset=='womd' else pack,dest)
            print('PREPARE',split,min(at+32,len(splits[split])),'/',len(splits[split]),flush=True)
    if dataset=='womd' and a.stage=='i1':save_json(dict(training=dict(cfg['refinement'],anchors='100% matched corresponding anchor')),a.workdir/'protocol.json')
    if dataset=='womd' and a.stage=='i2':
        import ccc.core.refiner_womd as refiner_module
        frozen_paths=[Path(refiner_module.__file__),a.weights,Path(weights['rcim']),
            Path(weights['heads'][str(a.seed)]['Full']['i1']),Path(weights['heads'][str(a.seed)]['B2']['i1'])]
        save_json(dict(frozen_SHA256={str(p.resolve()):sha(p) for p in frozen_paths},
            splits=splits,seed=a.seed,training=cfg['refinement']),a.workdir/('seed%d/PROTOCOL_FROZEN.json'%a.seed))
        save_json(dict(pass_=True,prepared_rows_verified=True,training_backward_not_executed=True),a.workdir/('seed%d/SMOKE.json'%a.seed))
    if dataset=='av2':
        for split in ('train','tune'):
            p=a.workdir/('manifests/refiner_'+split+'.txt');p.parent.mkdir(parents=True,exist_ok=True);p.write_text('\n'.join(splits[split])+'\n')

if __name__=='__main__':main()
