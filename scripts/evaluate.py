"""Aggregate saved scene metrics with fixed-model, whole-scene paired bootstrap."""
import argparse,json
from pathlib import Path
from _bootstrap import ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True,help='Directory containing scene_metrics_seed42/43/44.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--main',default='Full')
    p.add_argument('--references',nargs='+',default=['A','A_I1','B2','NoRCIM'])
    a=p.parse_args()
    import numpy as np
    from ccc.io import config,save_json
    cfg=config(a.config);seeds=[42,43,44];packs={}
    for seed in seeds:
        values=json.loads((a.input/('scene_metrics_seed%d.json'%seed)).read_text())
        assert len(values)==len({r['scene'] for r in values})
        packs[seed]={r['scene']:r['methods'] for r in values}
    scenes=sorted(packs[42]);assert scenes and all(set(packs[s])==set(scenes) for s in seeds)
    methods=list(packs[42][scenes[0]])
    assert set([a.main]+a.references)<=set(methods)
    fields=['ade_sum','fde_sum','overlap','gt5','overlap_steps']
    labels=['ADE','FDE','pair_rate','extra_gt5_rate','overlap_step_rate']
    if cfg['dataset']=='av2':
        fields+=['focal_ade_sum','focal_fde_sum'];labels+=['Focal_ADE6','Focal_FDE6']
    denominator=[];raw=[]
    for sid in scenes:
        ref=packs[42][sid][methods[0]];n=ref['targets'];pairs=ref['pairs']
        den=[n,n,pairs,n,pairs*cfg['future_steps']]
        if cfg['dataset']=='av2':den+=[ref.get('focal_targets',0)]*2
        denominator.append(den);scene=[]
        for seed in seeds:
            zz=[]
            for method in methods:
                r=packs[seed][sid][method];assert r['targets']==n and r['pairs']==pairs
                zz.append([r.get(f,0) for f in fields])
            scene.append(zz)
        raw.append(scene)
    den=np.asarray(denominator,float);raw=np.asarray(raw,float);ds=den.sum(0);assert (ds>0).all()
    point=raw.sum(0)/ds[None,None];table={}
    for i,seed in enumerate(seeds):
        table[str(seed)]={name:dict(zip(labels,point[i,j].tolist())) for j,name in enumerate(methods)}
    descriptive={name:{key:dict(mean=float(point[:,j,k].mean()),sample_SD=float(point[:,j,k].std(ddof=1)))
        for k,key in enumerate(labels)} for j,name in enumerate(methods)}
    # Identical scene draw across methods, metrics and all fixed refiner seeds.
    mean=raw.mean(1);diff=np.stack([mean[:,methods.index(a.main)]-mean[:,methods.index(ref)] for ref in a.references],1)
    bootstrap_seed=20261103 if 'Full_fixed_OFF' in a.references else cfg['evaluation']['bootstrap_seed']
    rng=np.random.default_rng(bootstrap_seed);draw=np.empty((5000,len(a.references),len(fields)))
    for rep in range(5000):
        ix=rng.integers(0,len(scenes),len(scenes));d=den[ix].sum(0)
        assert (d>0).all(),'Bootstrap denominator zero; cohort too small for this metric'
        draw[rep]=diff[ix].sum(0)/d
    delta=diff.sum(0)/ds;lo,hi=np.quantile(draw,[.025,.975],axis=0)
    ci={a.main+'_minus_'+ref:{key:dict(delta=float(delta[j,k]),CI95=[float(lo[j,k]),float(hi[j,k])])
        for k,key in enumerate(labels)} for j,ref in enumerate(a.references)}
    save_json(dict(scenes=len(scenes),scene_order='lexical scene ID',fixed_RCIM_seed=42,refiner_seeds=seeds,
        targets=int(ds[0]),pairs=int(ds[2]),per_seed=table,three_seed_descriptive=descriptive,
        paired_scene_CI=ci,bootstrap_seed=bootstrap_seed,resamples=5000,
        scope='Conditioned on these fixed models. No iid 3N claim. See archived official results for original serialized-order WOMD CIs.'),a.output)
    print(json.dumps(descriptive,indent=2))

if __name__=='__main__':main()
