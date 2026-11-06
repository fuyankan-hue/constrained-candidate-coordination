"""Smoke, tune-only parameter selection and frozen full-Validation evaluation."""
import argparse,copy,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy import optimize,sparse
C=Path('/workspace/classical');E=Path('/workspace/external');B=Path('/workspace/output');V=E/'validation_20261103_v2'
sys.path.insert(0,str(C));sys.path.insert(0,str(B/'code'));sys.path.insert(0,str(E/'code'))
from baseline_data import save,sha
from av2_core import measure,protect,hits
from classical import run,cost_grad,convex_step,orca_library
METHODS=('ORCA','CHOMP_style','SCP_style')
KEYS=('world','pi','current','history_velocity')

def verify():
    torch.set_num_threads(1)
    pr=json.loads((C/'PROTOCOL.json').read_text())
    for name,digest in pr['code_sha256'].items():assert sha(C/name)==digest,name
    native=json.loads((C/'ORCA_SOURCE.json').read_text())
    assert sha(C/'liborca_adapter.so')==native['library_sha256']
    assert sha(B/'MODELS_FROZEN.json')==pr['frozen_method_manifest_sha256']
    assert sha(B/'manifests/val_files.txt')==pr['validation_manifest_sha256']
    assert sha(B/'code/av2_core.py')==pr['metric_code_sha256']
    return pr

def setup_tune():
    pr=verify();expected=set((B/'manifests/refiner_tune.txt').read_text().splitlines());rows={}
    for p in sorted((B/'prepared/i1/tune').glob('*.pt')):
        for item in torch.load(p,map_location='cpu'):
            r=item['row'];assert r['scene'] in expected and r['scene'] not in rows;rows[r['scene']]=r
    assert len(rows)==1024 and set(rows)==expected
    assert not expected&{Path(s).stem for s in (B/'manifests/val_files.txt').read_text().splitlines()}
    save(C/'tune_rows.pt',[rows[s] for s in sorted(rows)],True)
    save(C/'TUNE_READY.json',dict(passed=True,scenes=1024,Validation_intersection=0,manifest_sha256=sha(B/'manifests/refiner_tune.txt')))
    print('TUNE_ROWS_READY 1024/1024',flush=True)

def anchor(row):
    return row['world'][torch.arange(len(row['pi'])),row['pi'].argmax(-1)].double().numpy()

def predict(method,row,cfg):return run(method,{k:row[k] for k in KEYS},cfg)

def check_P(a,x):
    _,before=hits(a);_,after=hits(x)
    assert not (after.any(-1)&~before.any(-1)).any() and after.sum()<=before.sum()

def optimization_checks(cfg):
    rng=np.random.default_rng(123);x=rng.normal(size=(2,6,2))*.15
    x[1,:,0]+=.65;current=np.array([[0.,0.],[.65,0.]])
    a=x+rng.normal(size=x.shape)*.03;f,g=cost_grad(x,a,current,cfg)
    numeric=np.zeros_like(x);eps=1e-6
    for index in np.ndindex(x.shape):
        plus=x.copy();minus=x.copy();plus[index]+=eps;minus[index]-=eps
        numeric[index]=(cost_grad(plus,a,current,cfg)[0]-cost_grad(minus,a,current,cfg)[0])/(2*eps)
    np.testing.assert_allclose(g,numeric,rtol=2e-5,atol=1e-6)
    Q=np.eye(6);q=rng.normal(size=6);A=rng.normal(size=(4,6));offset=np.ones(4)*.2
    qp_cfg=dict(admm_rho=10.,admm_max_iterations=500,admm_tolerance=1e-4)
    z,info=convex_step(sparse.csc_matrix(Q),q,sparse.csc_matrix(A),offset,1.,.4,qp_cfg)
    fun=lambda y:.5*y[:6]@Q@y[:6]+q@y[:6]+y[6:].sum()
    constraints=optimize.LinearConstraint(np.c_[A,np.eye(4)],offset,np.inf)
    sol=optimize.minimize(fun,np.r_[np.zeros(6),offset],method='SLSQP',constraints=constraints,
        bounds=optimize.Bounds(np.r_[np.full(6,-.4),np.zeros(4)],np.r_[np.full(6,.4),np.full(4,np.inf)]),
        options={'maxiter':1000,'ftol':1e-11})
    assert sol.success,sol.message
    ours=.5*z@Q@z+q@z+np.maximum(offset-A@z,0).sum()
    assert ours-sol.fun<.002,(ours,sol.fun,info)
    return dict(gradient_max_error=float(np.max(np.abs(g-numeric))),convex_objective_gap=float(ours-sol.fun),convex_solver=info)

def orca_checks():
    import ctypes
    lib=orca_library();ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    lib.orca_lp.argtypes=[ctypes.c_int,ptr,ctypes.c_double,ptr,ptr];lib.orca_lp.restype=ctypes.c_int
    rng=np.random.default_rng(123);maxgap=0.;maxviolation=0.
    for _ in range(100):
        theta=rng.uniform(-np.pi,np.pi,8);direction=np.stack([np.cos(theta),np.sin(theta)],-1)
        normals=np.stack([-direction[:,1],direction[:,0]],-1)
        points=-normals*rng.uniform(.1,1.,(8,1));lines=np.ascontiguousarray(np.c_[points,direction])
        pref=np.ascontiguousarray(rng.normal(size=2)*2);out=np.zeros(2)
        fail=lib.orca_lp(8,lines,2.,pref,out);assert fail==8
        low=(normals*points).sum(-1)
        # Independent double-precision enumeration of every possible optimum:
        # interior, one line, two lines, circle, or circle/line intersection.
        candidates=[np.zeros(2),pref,2*pref/np.linalg.norm(pref)]
        for i,n in enumerate(normals):
            candidates.append(pref+n*(low[i]-n@pref))
            tangent=np.array([-n[1],n[0]]);extent=np.sqrt(max(0.,4.-low[i]**2))
            candidates.extend((n*low[i]+tangent*extent,n*low[i]-tangent*extent))
            for j in range(i):
                matrix=np.array([n,normals[j]])
                if abs(np.linalg.det(matrix))>1e-10:candidates.append(np.linalg.solve(matrix,[low[i],low[j]]))
        feasible=[v for v in candidates if v@v<=4.+1e-9 and np.all(normals@v>=low-1e-9)]
        truth=min(.5*np.square(v-pref).sum() for v in feasible)
        gap=.5*np.square(out-pref).sum()-truth;violation=max(float(np.max(low-normals@out)),float(out@out-4.))
        maxgap=max(maxgap,abs(gap));maxviolation=max(maxviolation,violation)
        assert abs(gap)<5e-5 and violation<5e-5,(gap,violation)
    return dict(random_feasible_LP_cases=100,max_objective_gap=maxgap,max_constraint_violation=maxviolation)

def smoke(method):
    pr=verify();cfg=pr['configurations'][method][0]
    details=orca_checks() if method=='ORCA' else optimization_checks(cfg)
    rows=torch.load(C/'tune_rows.pt',map_location='cpu');chosen=[rows[0],max(rows[:64],key=lambda r:len(r['pi']))]
    measured=[]
    for row in chosen:
        path,d=predict(method,row,cfg);a=anchor(row)
        assert path.shape==a.shape and np.isfinite(path).all()
        poisoned=copy.deepcopy(row);poisoned['gt'].fill_(99999);poisoned['valid']=~poisoned['valid']
        pp,_=predict(method,poisoned,cfg);np.testing.assert_array_equal(path,pp)
        moved=copy.deepcopy(row);shift=torch.tensor([1000.,-400.],dtype=torch.float64)
        moved['current']=row['current'].double()+shift;moved['world']=row['world'].double()+shift
        mm,_=predict(method,moved,cfg);np.testing.assert_allclose(mm-path-np.array([1000.,-400.]),0.,atol=3e-5,rtol=0)
        safe,beta=protect(a,path);check_P(a,safe)
        denominator={(m['targets'],m['focal_targets'],m['pairs']) for m in [measure(row,z,a) for z in (a,path,safe)]};assert len(denominator)==1
        measured.append(dict(scene=row['scene'],targets=len(a),elapsed_seconds=d['seconds'],details=d,protection_beta=beta))
    save(C/('SMOKE_'+method+'.json'),dict(passed=True,method=method,numerical_checks=details,scenes=measured,protocol_sha256=sha(C/'PROTOCOL.json')))
    print('CLASSICAL_SMOKE_PASS',method,json.dumps(measured),flush=True)

def tune(method):
    pr=verify();assert json.loads((C/('SMOKE_'+method+'.json')).read_text())['passed'];rows=torch.load(C/'tune_rows.pt',map_location='cpu')
    scores=[];begun=time.time()
    for index,cfg in enumerate(pr['configurations'][method]):
        ade=fde=count=0.;statuses=[]
        for k,row in enumerate(rows):
            path,details=predict(method,row,cfg);m=measure(row,path,anchor(row));ade+=m['focal_ade_sum'];fde+=m['focal_fde_sum'];count+=m['focal_targets']
            statuses.append(details)
            if (k+1)%128==0:
                save(C/('tune_'+method+'_progress.json'),dict(configuration=index+1,total_configurations=len(pr['configurations'][method]),scenes=k+1,total=1024,seconds=time.time()-begun))
                print('CLASSICAL_TUNE',method,'config',index+1,'/',len(pr['configurations'][method]),k+1,'/1024',flush=True)
        score=(ade+.25*fde)/count
        scores.append(dict(index=index,configuration=cfg,Focal_ADE6=ade/count,Focal_FDE6=fde/count,criterion=score,
            mean_seconds=float(np.mean([z['seconds'] for z in statuses])),solver_statuses=statuses))
    best=min(range(len(scores)),key=lambda i:(scores[i]['criterion'],i))
    save(C/('FROZEN_'+method+'.json'),dict(passed=True,method=method,configuration=pr['configurations'][method][best],selected_index=best,
        selection='Original1024 Training tune scenes only; minimum focal ADE+.25FDE; deterministic index tie-break; no Validation selection',
        scores=scores,protocol_sha256=sha(C/'PROTOCOL.json'),seconds=time.time()-begun))
    print('CLASSICAL_CONFIGURATION_FROZEN',method,best,flush=True)

def evaluate(method):
    pr=verify();frozen=C/('FROZEN_'+method+'.json');cfg=json.loads(frozen.read_text())['configuration'];fh=sha(frozen)
    out=C/'evaluation'/method;out.mkdir(parents=True,exist_ok=True);expected=[Path(s).stem for s in (B/'manifests/val_files.txt').read_text().splitlines()]
    done=0;begun=time.time()
    for ix,p in enumerate(sorted((V/'prepared').glob('*.pt'))):
        dest=out/'shards'/p.name;wanted=expected[ix*64:(ix+1)*64]
        if dest.exists():
            cached=torch.load(dest,map_location='cpu');assert cached['configuration_sha256']==fh and [r['scene'] for r in cached['records']]==wanted
        else:
            records=[]
            for r in torch.load(p,map_location='cpu')['rows']:
                row=r['reference_row'];a=anchor(row);path,details=predict(method,row,cfg)
                tick=time.perf_counter();safe,beta=protect(a,path);pseconds=time.perf_counter()-tick;check_P(a,safe)
                methods={name:measure(row,z,a) for name,z in [('NDPNet',a),('native',path),('common_P',safe)]}
                assert len({(m['targets'],m['focal_targets'],m['pairs']) for m in methods.values()}|{(m['targets'],m['focal_targets'],m['pairs']) for m in r['reference'].values()})==1
                records.append(dict(scene=r['scene'],agent_ids=r['agent_ids'],methods=methods,reference=r['reference'],solver=details,
                    protection_beta=beta,protection_seconds=pseconds,predictions=dict(native=torch.as_tensor(path),common_P=torch.as_tensor(safe))))
            assert [r['scene'] for r in records]==wanted
            save(dest,dict(records=records,configuration_sha256=fh),True)
        done+=len(wanted);save(out/'progress.json',dict(scenes=done,total=24988,shards=ix+1,total_shards=391,seconds=time.time()-begun))
        print('CLASSICAL_VALIDATION',method,done,'/24988','seconds',round(time.time()-begun,1),flush=True)
    assert done==24988
    save(out/'complete.json',dict(passed=True,method=method,scenes=done,configuration_sha256=fh,seconds=time.time()-begun,common_P_violations=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare_tune','smoke','tune','evaluate']);p.add_argument('--method',choices=METHODS);a=p.parse_args()
    if a.mode=='prepare_tune':setup_tune()
    elif a.mode=='smoke':smoke(a.method)
    elif a.mode=='tune':tune(a.method)
    else:evaluate(a.method)
