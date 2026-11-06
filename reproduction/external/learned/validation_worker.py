"""Frozen AV2 Validation inference. Existing model/source mounts are read-only."""
import argparse,copy,json,os,pickle,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
V=Path('/workspace/validation');E=Path('/workspace/external');B=Path('/workspace/output')
if os.environ.get('EXTERNAL_BASELINE_METHOD')=='SRefiner':E=E/'srefiner_cuda_compat_v1'
sys.path.insert(0,str(E/'code'));sys.path.insert(1,str(B/'code'))
from baseline_data import context,batch,save,sha,verify_code
from baseline_model import Baseline,selected
from av2_core import measure,protect,hits

def protocol():return json.loads((V/'PROTOCOL.json').read_text())

def setup():
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def ids():return [Path(n).stem for n in (B/'manifests/val_files.txt').read_text().splitlines()]

def prepare(limit=None):
    setup();verify_code();pr=protocol();expected=ids();assert len(expected)==len(set(expected))==24988
    assert sha(B/'manifests/val_files.txt')==pr['validation_manifest_sha256']
    train={Path(x).stem for x in (B/'manifests/train_files.txt').read_text().splitlines()}
    assert not set(expected)&train
    frozen=json.loads((B/'MODELS_FROZEN.json').read_text())
    for name,digest in frozen['sha256'].items():assert sha(B/name)==digest,name
    files=sorted((B/'cache/val').glob('*.pt'));assert len(files)==391
    assert [p.name for p in files]==[p.name for p in sorted((B/'evaluation/shards').glob('*.pt'))]
    rawdir=Path('/workspace/QCNet/data/av2_qcnet/val/processed')
    assert all((rawdir/(sid+'.pkl')).is_file() for sid in expected)
    done=0;begun=time.time()
    for ix,p in enumerate(files):
        if limit is not None and ix>=limit:break
        wanted=expected[ix*64:(ix+1)*64];dest=V/'prepared'/p.name
        if dest.exists():
            z=torch.load(dest,map_location='cpu')
            assert z['protocol_sha256']==sha(V/'PROTOCOL.json') and [r['scene'] for r in z['rows']]==wanted
        else:
            rows=torch.load(p,map_location='cpu')['rows'];assert [r['scene'] for r in rows]==wanted
            ref=torch.load(B/'evaluation/shards'/p.name,map_location='cpu')
            assert ref['frozen_manifest_sha256']==sha(B/'MODELS_FROZEN.json')
            assert [r['scene'] for r in ref['records']]==wanted
            output=[]
            for row,rr in zip(rows,ref['records']):
                assert row['original_agent_id']==rr['original_agent_id']
                with (rawdir/(row['scene']+'.pkl')).open('rb') as f:raw=pickle.load(f)
                r=context(raw,row);r['reference_row']=row
                r['reference']={str(s):rr['methods'][str(s)]['FullRCIM'] for s in (42,43,44)}
                # Compare saved full-method metrics against saved paths, not new inference.
                if ix==0 and len(output)<2:
                    a=row['world'][torch.arange(len(row['pi'])),row['pi'].argmax(-1)].double().numpy()
                    for s in (42,43,44):
                        m=measure(row,rr['predictions'][str(s)]['FullRCIM'].numpy(),a)
                        for k,v in m.items():
                            if k=='delta_fde':np.testing.assert_allclose(v,r['reference'][str(s)][k],rtol=0,atol=1e-10)
                            else:assert abs(v-r['reference'][str(s)][k])<1e-9,(k,v)
                output.append(r)
            save(dest,dict(rows=output,protocol_sha256=sha(V/'PROTOCOL.json')),True)
        done+=len(wanted)
        save(V/'prepare_progress.json',dict(scenes=done,total=24988,shards=ix+1,total_shards=391,seconds=time.time()-begun))
        print('PREPARE_VALIDATION',done,'/24988','shards',ix+1,'/391','seconds',round(time.time()-begun,1),flush=True)
    if limit is None:
        assert done==24988
        save(V/'PREPARED_COMPLETE.json',dict(passed=True,scenes=done,shards=391,protocol_sha256=sha(V/'PROTOCOL.json'),seconds=time.time()-begun))

def load_model(method,seed):
    verify_code();out=E/'models'/('%s_seed%d'%(method,seed));z=json.loads((out/'TRAINING_FROZEN.json').read_text())
    assert z['passed'] and sha(out/'best.pt')==z['best_sha256']==protocol()['checkpoints'][method][str(seed)]['sha256']
    m=Baseline(method).cuda().eval();m.load_state_dict(torch.load(out/'best.pt',map_location='cpu')['state_dict']);m.requires_grad_(False)
    return m

@torch.inference_mode()
def predict(m,r):
    args,_,info=batch([r],m.method);torch.cuda.synchronize();tick=time.perf_counter()
    stages,h=m(args,info);path,modes=selected(m,stages,h,info)
    w=torch.einsum('ntc,ncd->ntd',path.double(),info['rotation'].double())+r['reference_row']['current'].double().cuda()[:,None]
    torch.cuda.synchronize();elapsed=(time.perf_counter()-tick)*1000
    return w.cpu().numpy(),modes.cpu().numpy(),elapsed

def check_protection(a,safe):
    _,old=hits(a);_,new=hits(safe)
    assert not (new.any(-1)&~old.any(-1)).any() and new.sum()<=old.sum()

@torch.inference_mode()
def smoke(method,seed):
    setup();m=load_model(method,seed);rows=torch.load(V/'prepared/00000.pt',map_location='cpu')['rows']
    picked=[rows[0],max(rows,key=lambda r:len(r['agent_ids']))];details=[]
    for r in picked:
        row=r['reference_row'];path,modes,ms=predict(m,r);repeat,_,_=predict(m,r)
        numerical=float(np.max(np.abs(path-repeat)))
        poison=copy.deepcopy(r)
        poison['labels']['gt_local'].fill_(99999);poison['labels']['gt_world'].fill_(-99999);poison['labels']['valid']=~poison['labels']['valid']
        poison['reference_row']['gt'].fill_(-99999);poison['reference_row']['valid']=~poison['reference_row']['valid']
        changed,cm,_=predict(m,poison)
        assert np.isfinite(path).all() and path.shape==(len(row['pi']),60,2)
        leak=float(np.max(np.abs(path-changed)));assert leak<=max(1e-5,4*numerical)
        np.testing.assert_array_equal(modes,cm)
        args,_,info=batch([r],method)
        a=row['world'][torch.arange(len(row['pi'])),row['pi'].argmax(-1)].double().numpy()
        inv=torch.einsum('ntc,ncd->ntd',info['local'][:,0].double(),info['rotation'].double())+row['current'].double().cuda()[:,None]
        np.testing.assert_allclose(inv.cpu().numpy(),a,rtol=1e-6,atol=2e-4)
        safe,beta=protect(a,path);check_protection(a,safe)
        mm=[measure(row,x,a) for x in (a,path,safe)]+[r['reference'][str(seed)]]
        assert len({(x['targets'],x['focal_targets'],x['pairs'],x['all_predicted_targets']) for x in mm})==1
        details.append(dict(scene=r['scene'],targets=len(row['pi']),focal_targets=mm[0]['focal_targets'],pairs=mm[0]['pairs'],latency_ms=ms,GT_poison_max_difference=leak,same_input_jitter=numerical,protection_beta=beta))
    result=dict(passed=True,method=method,seed=seed,device=torch.cuda.get_device_name(0),details=details,protocol_sha256=sha(V/'PROTOCOL.json'))
    save(V/('SMOKE_'+method+'.json'),result);print('VALIDATION_SMOKE_PASS',json.dumps(result),flush=True)

@torch.inference_mode()
def evaluate(method,seed):
    setup();m=load_model(method,seed);ph=sha(V/'PROTOCOL.json')
    assert json.loads((V/'PREPARED_COMPLETE.json').read_text())['protocol_sha256']==ph
    assert json.loads((V/('SMOKE_'+method+'.json')).read_text())['protocol_sha256']==ph
    out=V/'jobs'/('%s_seed%d'%(method,seed));out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():assert json.loads((out/'complete.json').read_text())['protocol_sha256']==ph;return
    done=0;begun=time.time();expected=ids()
    for ix,p in enumerate(sorted((V/'prepared').glob('*.pt'))):
        dest=out/'shards'/p.name;wanted=expected[ix*64:(ix+1)*64]
        if dest.exists():
            z=torch.load(dest,map_location='cpu');assert z['protocol_sha256']==ph and [r['scene'] for r in z['records']]==wanted
        else:
            rr=torch.load(p,map_location='cpu');assert rr['protocol_sha256']==ph
            records=[]
            for r in rr['rows']:
                row=r['reference_row'];path,modes,ms=predict(m,r)
                assert np.isfinite(path).all()
                a=row['world'][torch.arange(len(row['pi'])),row['pi'].argmax(-1)].double().numpy()
                tick=time.perf_counter();safe,beta=protect(a,path);pms=(time.perf_counter()-tick)*1000;check_protection(a,safe)
                methods={name:measure(row,x,a) for name,x in (('NDPNet',a),('external_native',path),('external_common_P',safe))}
                methods['frozen_full_method']=r['reference'][str(seed)]
                assert len({(x['targets'],x['focal_targets'],x['pairs'],x['all_predicted_targets']) for x in methods.values()})==1
                methods['external_native']['latency_ms']=ms;methods['external_common_P']['latency_ms']=ms+pms
                records.append(dict(scene=r['scene'],agent_ids=r['agent_ids'],methods=methods,common_P_beta=beta,
                    selected_modes=modes.tolist(),predictions={'native':torch.as_tensor(path),'common_P':torch.as_tensor(safe)}))
            assert [r['scene'] for r in records]==wanted
            save(dest,dict(records=records,protocol_sha256=ph),True)
        done+=len(wanted)
        save(out/'progress.json',dict(method=method,seed=seed,scenes=done,total=24988,shards=ix+1,total_shards=391,seconds=time.time()-begun,gpu=os.environ.get('EXTERNAL_GPU_INDEX'),stage='evaluation'))
        print('VALIDATION',method,seed,done,'/24988','seconds',round(time.time()-begun,1),flush=True)
    assert done==24988
    save(out/'complete.json',dict(passed=True,method=method,seed=seed,scenes=done,protocol_sha256=ph,seconds=time.time()-begun,GPU=os.environ.get('EXTERNAL_GPU_INDEX'),common_P_violations=0))
    print('VALIDATION_COMPLETE',method,seed,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','smoke','evaluate']);a.add_argument('--method');a.add_argument('--seed',type=int,default=42);a.add_argument('--limit',type=int);x=a.parse_args()
    if x.mode=='prepare':prepare(x.limit)
    elif x.mode=='smoke':smoke(x.method,x.seed)
    else:evaluate(x.method,x.seed)
