"""Same-scene selected-trajectory metrics, protection and scene bootstrap.

Only the fixed native-Training development cohort is read. No validation access.
"""
import json,sys,time
import numpy as np
import torch
from baseline_data import ROOT,BASE,load_rows,batch,save,sha
from baseline_model import selected

def summarize(records):
    names=list(records[0]['methods']);table={}
    for name in names:
        rr=[r['methods'][name] for r in records]
        total=lambda k:sum(r[k] for r in rr)
        n=total('targets');pairs=total('pairs')
        table[name]=dict(ADE6=total('ade_sum')/n,FDE6=total('fde_sum')/n,
            close_center_pairs=total('overlap'),close_center_pair_rate=total('overlap')/max(1,pairs),
            close_center_steps=total('overlap_steps'),extra_gt5_vs_A=total('gt5'),
            valid6_targets=n,eligible_pairs=pairs,
            mean_acceleration=total('acceleration_sum')/max(1,total('acceleration_count')),
            mean_jerk=total('jerk_sum')/max(1,total('jerk_count')),
            median_scene_latency_ms=float(np.median([r.get('latency_ms',0) for r in rr])),
            mean_scene_latency_ms=float(np.mean([r.get('latency_ms',0) for r in rr])))
    ci={};rng=np.random.default_rng(420102)
    for new,old in [('external_native','NDPNet'),('external_common_P','NDPNet'),
                    ('external_native','frozen_full_method'),('external_common_P','frozen_full_method')]:
        ci[new+'_minus_'+old]={}
        for numerator,denominator,key in [('ade_sum','targets','ADE6'),('fde_sum','targets','FDE6'),
                                         ('overlap','pairs','close_center_rate'),('gt5','targets','extra_gt5_rate')]:
            delta=np.array([r['methods'][new][numerator]-r['methods'][old][numerator] for r in records])
            den=np.array([r['methods'][old][denominator] for r in records]);values=[]
            for _ in range(3000):
                ix=rng.integers(len(records),size=len(records));values.append(delta[ix].sum()/max(1,den[ix].sum()))
            ci[new+'_minus_'+old][key]=dict(delta=float(delta.sum()/max(1,den.sum())),CI95=np.quantile(values,[.025,.975]).tolist())
    return table,ci

@torch.no_grad()
def evaluate_comparison(model,seed):
    assert (ROOT/'models'/('%s_seed%d'%(model.method,seed))/'TRAINING_FROZEN.json').exists()
    sys.path.insert(0,str(BASE/'code'))
    from av2_pipeline import load_rcim,load_head
    from av2_core import choose,measure,protect,hits,refine
    from av2_refiner import collate
    model.eval();rows=load_rows('dev');scorer=load_rcim()
    heads={stage:load_head(seed,stage,'E' if stage=='I1' else 'RCIM','cuda') for stage in ('I1','I2')}
    records=[];begin=time.time();protection_violations=0
    for k,r in enumerate(rows):
        row=r['reference_row'];args,_,info=batch([r],model.method)
        torch.cuda.synchronize();tick=time.perf_counter()
        stages,hh=model(args,info);path,modes=selected(model,stages,hh,info)
        external=torch.einsum('ntc,ncd->ntd',path.double(),info['rotation'].double())+row['current'].double().to(path.device)[:,None]
        external=external.cpu().numpy();torch.cuda.synchronize();external_ms=(time.perf_counter()-tick)*1000
        a=row['world'][torch.arange(len(row['pi'])),row['pi'].argmax(-1)].double().numpy()
        tick=time.perf_counter();safe,beta=protect(a,external);protection_ms=(time.perf_counter()-tick)*1000
        _,before=hits(a);_,after=hits(safe)
        protection_violations+=int((after.any(-1)&~before.any(-1)).any() or after.sum()>before.sum())
        tick=time.perf_counter();ch=choose(row,scorer);z=ch['RCIM']
        first,_,_=refine(row,z['path'],z['modes'],heads['I1'],collate,'cuda')
        radius=np.clip(5.-1e-4-np.linalg.norm(first-a,axis=-1).max(1),0.,1.).astype(np.float32)
        full,_,_=refine(row,first,z['modes'],heads['I2'],collate,'cuda',radius)
        torch.cuda.synchronize();ours_ms=(time.perf_counter()-tick)*1000
        outputs={'NDPNet':a,'external_native':external,'external_common_P':safe,'frozen_full_method':full}
        methods={name:measure(row,p,a) for name,p in outputs.items()}
        methods['external_native']['latency_ms']=external_ms
        methods['external_common_P']['latency_ms']=external_ms+protection_ms
        methods['frozen_full_method']['latency_ms']=ours_ms
        records.append(dict(scene=r['scene'],agent_ids=r['agent_ids'],methods=methods,common_P_beta=beta))
        if (k+1)%128==0:print('DEVELOPMENT_COMPARISON',model.method,seed,k+1,'/',len(rows),flush=True)
    assert protection_violations==0
    out=ROOT/'models'/('%s_seed%d'%(model.method,seed))
    save(out/'development_records.pt',records,True)
    table,intervals=summarize(records)
    result=dict(passed=True,scenes=len(rows),table=table,paired_scene_CI95=intervals,
        common_P_violations=protection_violations,seconds=time.time()-begin,
        metric='Same frozen AV2 native6s protocol: selectedTop1; center-distance<1m proxy, not oriented-box overlap',
        fairness='Native external port and explicitly labelled +commonP are both reported; same backbone and scored/focal targets; native external training losses differ from our loss',
        exposure='Fixed native-Training development IDs; backbone has native-Training exposure; not independent confirmation',
        refiner_seed=seed,RCIM_seed=42,method=model.method,
        CI_unit='whole scene, fixed trained seed; never independent target bootstrap')
    save(out/'comparison.json',result);return result
