"""Read saved inference only; average fixed models within scene before bootstrap."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
V=Path('/workspace/validation');B=Path('/workspace/output')
sys.path.insert(0,'/workspace/external/code')
from baseline_data import save,sha
NAMES=['NDPNet','external_native','external_common_P','frozen_full_method']
METRICS=[('ade_sum','targets','ADE6'),('fde_sum','targets','FDE6'),
         ('focal_ade_sum','focal_targets','Focal_ADE6'),('focal_fde_sum','focal_targets','Focal_FDE6'),
         ('overlap','pairs','close_center_pair_rate'),('gt5','targets','extra_gt5_rate')]

def main():
    torch.set_num_threads(2);expected=[Path(p).stem for p in (B/'manifests/val_files.txt').read_text().splitlines()]
    protocol=json.loads((V/'PROTOCOL.json').read_text());ph=sha(V/'PROTOCOL.json');result={};bootstrap={}
    for method in ('SmartRefine','SRefiner'):
        records={};per_seed={}
        for seed in (42,43,44):
            out=V/'jobs'/('%s_seed%d'%(method,seed));assert json.loads((out/'complete.json').read_text())['protocol_sha256']==ph
            rows=[]
            for f in sorted((out/'shards').glob('*.pt')):
                z=torch.load(f,map_location='cpu');assert z['protocol_sha256']==ph
                for r in z['records']:
                    r.pop('predictions',None);rows.append(r)
            assert [r['scene'] for r in rows]==expected and len(rows)==24988
            for r in rows:
                assert len({(x['targets'],x['focal_targets'],x['pairs']) for x in r['methods'].values()})==1
            records[seed]=rows;table={}
            for name in NAMES:
                rr=[r['methods'][name] for r in rows];total=lambda k:sum(v[k] for v in rr)
                table[name]={label:total(num)/max(1,total(den)) for num,den,label in METRICS}
                for k in ('targets','focal_targets','pairs','overlap','overlap_steps','gt5','improved','degraded'):table[name][k]=total(k)
                for num,den,label in [('acceleration_sum','acceleration_count','mean_acceleration'),('jerk_sum','jerk_count','mean_jerk')]:table[name][label]=total(num)/max(1,total(den))
                if name.startswith('external'):table[name]['mean_scene_latency_ms']=float(np.mean([x['latency_ms'] for x in rr]))
            per_seed[str(seed)]=table
        means={name:{k:dict(mean=float(np.mean([per_seed[str(s)][name][k] for s in (42,43,44)])),sample_sd=float(np.std([per_seed[str(s)][name][k] for s in (42,43,44)],ddof=1))) for k in per_seed['42'][name]} for name in NAMES}
        result[method]=dict(per_seed=per_seed,three_seed=means);bootstrap[method]={}
        for new,old in [('external_native','NDPNet'),('external_common_P','NDPNet'),('external_native','frozen_full_method'),('external_common_P','frozen_full_method')]:
            intervals={}
            for num,den,label in METRICS:
                diff=np.mean([[r['methods'][new][num]-r['methods'][old][num] for r in records[s]] for s in (42,43,44)],axis=0)
                denominator=np.array([r['methods'][old][den] for r in records[42]])
                rng=np.random.default_rng(protocol['bootstrap_seed']);values=[]
                for _ in range(protocol['bootstrap_resamples']):
                    ix=rng.integers(len(expected),size=len(expected));values.append(float(diff[ix].sum()/max(1,denominator[ix].sum())))
                intervals[label]=dict(delta=float(diff.sum()/max(1,denominator.sum())),CI95=np.quantile(values,[.025,.975]).tolist())
            bootstrap[method][new+'_minus_'+old]=intervals
        del records
        print('VALIDATION_AGGREGATED',method,flush=True)
    save(V/'FINAL_RESULTS.json',dict(passed=True,scenes=len(expected),methods=result,paired_scene_bootstrap=bootstrap,
        protocol_sha256=ph,scope='Full AV2 Validation; historical backbone/our-method validation exposure disclosed; no validation-selected weights or parameters in this comparison.',
        CI_scope='Fixed three trained models averaged within scene, then whole-scene paired Bootstrap; not independent seeds times scenes.',
        implementation='Official-core NDPNet AV2 ports, not author released backbone/checkpoint reproduction. SRefiner native regression-only ranking remains fixed proposal rank0.',
        geometry='All requested-target center distance <1m proxy; not oriented-box overlap or actual collisions.',
        reference='Previously frozen full-method Validation paths and metrics reused, never overwritten.'))
    print('FINAL_RESULTS_READY',flush=True)

if __name__=='__main__':main()
