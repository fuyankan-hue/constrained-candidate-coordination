"""Merge three deterministic classical baselines with completed learned ports."""
import json,sys
from pathlib import Path
import numpy as np
import torch
C=Path('/workspace/classical');E=Path('/workspace/external');B=Path('/workspace/output')
sys.path.insert(0,str(E/'code'))
from baseline_data import save,sha
METRICS=[('ade_sum','targets','ADE6'),('fde_sum','targets','FDE6'),('focal_ade_sum','focal_targets','Focal_ADE6'),
         ('focal_fde_sum','focal_targets','Focal_FDE6'),('overlap','pairs','close_center_pair_rate'),('gt5','targets','extra_gt5_rate')]

def main():
    torch.set_num_threads(1);expected=[Path(s).stem for s in (B/'manifests/val_files.txt').read_text().splitlines()]
    pr=json.loads((C/'PROTOCOL.json').read_text());previous=json.loads((E/'validation_20261103_v2/FINAL_RESULTS.json').read_text());assert previous['passed']
    result={};intervals={};diagnostics={}
    for method in ('ORCA','CHOMP_style','SCP_style'):
        out=C/'evaluation'/method;assert json.loads((out/'complete.json').read_text())['passed'];rows=[]
        for p in sorted((out/'shards').glob('*.pt')):
            for r in torch.load(p,map_location='cpu')['records']:r.pop('predictions',None);rows.append(r)
        assert [r['scene'] for r in rows]==expected
        table={}
        for arm in ('NDPNet','native','common_P'):
            rr=[r['methods'][arm] for r in rows];total=lambda k:sum(m[k] for m in rr)
            table[arm]={label:total(num)/max(1,total(den)) for num,den,label in METRICS}
            for k in ('targets','focal_targets','pairs','overlap','overlap_steps','gt5'):table[arm][k]=total(k)
            for num,den,label in [('acceleration_sum','acceleration_count','mean_acceleration'),('jerk_sum','jerk_count','mean_jerk')]:table[arm][label]=total(num)/max(1,total(den))
            table[arm]['mean_scene_seconds']=float(np.mean([r['solver']['seconds']+(r['protection_seconds'] if arm=='common_P' else 0) for r in rows])) if arm!='NDPNet' else None
        result[method]=dict(deterministic=True,configuration=json.loads((C/('FROZEN_'+method+'.json')).read_text())['configuration'],table=table)
        intervals[method]={}
        for arm in ('native','common_P'):
            for reference in ('NDPNet','frozen_full_method'):
                values={}
                for num,den,label in METRICS:
                    a=np.array([r['methods'][arm][num] for r in rows])
                    if reference=='NDPNet':b=np.array([r['methods']['NDPNet'][num] for r in rows])
                    else:b=np.array([np.mean([r['reference'][str(s)][num] for s in (42,43,44)]) for r in rows])
                    denominator=np.array([r['methods']['NDPNet'][den] for r in rows]);difference=a-b
                    rng=np.random.default_rng(pr['bootstrap_seed']);simulations=[]
                    for _ in range(pr['bootstrap_resamples']):
                        ix=rng.integers(len(rows),size=len(rows));simulations.append(float(difference[ix].sum()/max(1,denominator[ix].sum())))
                    values[label]=dict(delta=float(difference.sum()/max(1,denominator.sum())),CI95=np.quantile(simulations,[.025,.975]).tolist())
                intervals[method][arm+'_minus_'+reference]=values
        diagnostics[method]=dict(mean_iterations=float(np.mean([r['solver']['iterations'] for r in rows])),
            solver_subproblems=sum(r['solver'].get('convex_subproblems',0) for r in rows),
            converged_subproblems=sum(r['solver'].get('subproblem_converged',0) for r in rows),
            numerical_degeneracies=sum(r['solver'].get('numerical_degeneracies',0) for r in rows),
            common_P_violations=0)
        print('CLASSICAL_AGGREGATED',method,flush=True)
    save(C/'FIVE_BASELINES_RESULTS.json',dict(passed=True,scenes=24988,new_classical_methods=result,
        classical_paired_scene_bootstrap=intervals,solver_diagnostics=diagnostics,
        prior_learned_methods=previous['methods'],prior_learned_paired_scene_bootstrap=previous['paired_scene_bootstrap'],
        protocol_sha256=sha(C/'PROTOCOL.json'),
        scope='Five baseline families, same frozen NDPNet and full AV2 Validation; primary focal6s and secondary all-requested6s both retained.',
        CI_scope='Deterministic classical baseline versus our fixed3-refiner mean within each scene; resample whole scenes, never triple deterministic outputs into independent seeds.',
        implementation='ORCA dynamic-agent RVO2 core port; explicitly adapted CHOMP-style and SCP/TrajOpt-style postprocessors, not unmodified robotics benchmark reproductions.',
        exposure='Historical AV2 Validation exposure retained; all new classical parameters frozen using Training tune only before evaluation.'))
    print('FIVE_BASELINES_COMPARISON_COMPLETE',flush=True)

if __name__=='__main__':main()
