import argparse,copy,hashlib,json,os,random,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from baseline_data import ROOT,save,sha,load_rows,batch,verify_code
from baseline_model import Baseline,objective,selected,evaluate

def setup(seed,device='cuda'):
    torch.set_num_threads(2);random.seed(seed);np.random.seed(seed)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device=='cuda':assert torch.cuda.is_available() and torch.cuda.device_count()==1

def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch']);torch.cuda.set_rng_state_all(x['cuda'])

def smoke(method,device='cuda'):
    verify_code();setup(42,device);start=time.time();model=Baseline(method).to(device)
    ids=json.loads((ROOT/'manifests/train.json').read_text())
    rows=[]
    for sid in ids[:32]:
        p=ROOT/'prepared/train'/(sid+'.pt')
        if p.exists():rows.append(torch.load(p,map_location='cpu'))
    assert rows
    # Check both a normal mini-batch and the largest prepared mini-batch.
    groups=[rows[:min(4,len(rows))],sorted(rows,key=lambda r:len(r['prediction']['current'])*len(r['prediction']['lane_pos']),reverse=True)[:16]]
    details=[]
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    for group in groups:
        args,labels,info=batch(group,method,device);model.eval()
        from reference_loss_check import check as check_author_loss
        author_loss_check=check_author_loss(model,args,labels,info)
        with torch.no_grad():
            reference=model(args,info);repeat=model(args,info)
        numerical=max(float((a[0]-b[0]).abs().max()) for a,b in zip(reference[0],repeat[0]))
        poison=copy.deepcopy(group)
        for r in poison:
            r['labels']['gt_local'].fill_(99999);r['labels']['gt_world'].fill_(-99999)
            r['labels']['valid']=~r['labels']['valid']
        ap,yp,ip=batch(poison,method,device)
        with torch.no_grad():changed=model(ap,ip)
        maxdiff=max(float((a[0]-b[0]).abs().max()) for a,b in zip(reference[0],changed[0]))
        assert maxdiff<=max(1e-5,4*numerical),('future-label leak or numerical mismatch',maxdiff,numerical)
        model.train();optim=torch.optim.AdamW(model.parameters(),lr=1e-4)
        optim.zero_grad(set_to_none=True);loss=objective(model,args,labels,info)
        assert torch.isfinite(loss);loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        norm=sum(float(g.square().sum()) for g in grads)**.5;assert norm>0
        optim.step();model.eval()
        with torch.no_grad():stages,heads=model(args,info);out,modes=selected(model,stages,heads,info)
        assert out.shape==(sum(info['nlist']),60,2) and torch.isfinite(out).all()
        details.append(dict(scenes=len(group),targets=sum(info['nlist']),loss=float(loss),
                           gradient_norm=norm,future_label_perturbation_max_difference=maxdiff,
                           same_input_numerical_jitter=numerical,author_loss_check=author_loss_check))
    result=dict(passed=True,method=method,device=torch.cuda.get_device_name(0) if device=='cuda' else 'CPU',
        parameters=sum(p.numel() for p in model.parameters()),details=details,
        gpu_peak_MiB=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,seconds=time.time()-start,
        adapter_protocol_sha256=sha(ROOT/'ADAPTER_PROTOCOL.json'),
        scope='Official refinement cores ported to NDPNet AV2; fixed iteration implementation. Not authors released-model evaluation.')
    suffix='' if device=='cuda' else '_cpu'
    save(ROOT/('SMOKE_'+method+suffix+'.json'),result);print('BASELINE_SMOKE_PASS',json.dumps(result),flush=True)

def train(method,seed):
    code_digest=verify_code();setup(seed);out=ROOT/'models'/('%s_seed%d'%(method,seed));out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'ADAPTER_PROTOCOL.json').read_text());ph=sha(ROOT/'ADAPTER_PROTOCOL.json')
    assert json.loads((ROOT/'DATA_READY.json').read_text())['passed']
    assert json.loads((ROOT/('SMOKE_'+method+'.json')).read_text())['adapter_protocol_sha256']==ph
    if (out/'complete.json').exists():
        z=json.loads((out/'complete.json').read_text());assert sha(out/'best.pt')==z['best_sha256'];return
    train_rows=load_rows('train');tune_rows=load_rows('tune')
    assert not {r['scene'] for r in train_rows}&{r['scene'] for r in tune_rows}
    model=Baseline(method).cuda();lr=protocol[method]['lr'];optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=64,eta_min=0.)
    epoch=0;next_batch=0;steps=0;best=float('inf');history=[];begun=time.time();previous_seconds=0
    if (out/'last.pt').exists():
        z=torch.load(out/'last.pt',map_location='cpu');assert z['protocol_sha256']==ph
        model.load_state_dict(z['state_dict']);optimizer.load_state_dict(z['optimizer']);scheduler.load_state_dict(z['scheduler'])
        epoch=z['epoch'];next_batch=z['next_batch'];steps=z['steps'];best=z['best'];history=z['history'];previous_seconds=z['seconds']
        restore_rng(z['rng'])
    def state():
        return dict(state_dict=cpu_state(model),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    epoch=epoch,next_batch=next_batch,steps=steps,best=best,history=history,
                    rng=rng(),seconds=previous_seconds+time.time()-begun,protocol_sha256=ph,method=method,seed=seed,code_manifest_sha256=code_digest)
    batch_size=16;batches=(len(train_rows)+batch_size-1)//batch_size
    print('START',method,seed,'epochs',epoch,'/64','updates',steps,'/32768','GPU',os.environ.get('EXTERNAL_GPU_INDEX'),flush=True)
    while epoch<64:
        order=np.random.default_rng(seed+epoch).permutation(len(train_rows));model.train()
        for k in range(next_batch,batches):
            rows=[train_rows[int(i)] for i in order[k*batch_size:(k+1)*batch_size]]
            args,labels,info=batch(rows,method);optimizer.zero_grad(set_to_none=True)
            loss=objective(model,args,labels,info)
            assert torch.isfinite(loss),('nonfinite loss',method,seed,epoch,k)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step();steps+=1;next_batch=k+1
            progress=dict(method=method,seed=seed,stage='training',epoch=epoch+1,max_epochs=64,
                          scenes=min(next_batch*16,len(train_rows)),total_scenes=len(train_rows),
                          batches=next_batch,total_batches=batches,optimizer_steps=steps,total_steps=64*batches,
                          loss=float(loss),gpu=os.environ.get('EXTERNAL_GPU_INDEX'),
                          seconds=previous_seconds+time.time()-begun)
            save(out/'progress.json',progress)
            if steps%50==0:print('PROGRESS',json.dumps(progress),flush=True)
            if steps%128==0:save(out/'last.pt',state(),True)
        scheduler.step();metrics=evaluate(model,tune_rows,batch)
        metric=metrics[metrics['selection_metric']]
        history.append(dict(epoch=epoch+1,steps=steps,tune=metrics,lr=scheduler.get_last_lr()[0]))
        if metric<best:
            best=metric;save(out/'best.pt',dict(state_dict=cpu_state(model),epoch=epoch+1,steps=steps,
                tune=metrics,method=method,seed=seed,protocol_sha256=ph),True)
        epoch+=1;next_batch=0
        save(out/'history.json',history);save(out/'last.pt',state(),True)
        if epoch==10:save(out/'budget10.pt',torch.load(out/'best.pt',map_location='cpu'),True)
        print('EPOCH',method,seed,epoch,'/64',json.dumps(metrics),flush=True)
    # Freeze before reading the new Training development evaluation split.
    frozen=dict(passed=True,method=method,seed=seed,epochs=64,steps=steps,
                parameters=sum(p.numel() for p in model.parameters()),best_sha256=sha(out/'best.pt'),
                best_epoch=torch.load(out/'best.pt',map_location='cpu')['epoch'],
                protocol_sha256=ph,seconds=previous_seconds+time.time()-begun,
                GPU_index=os.environ.get('EXTERNAL_GPU_INDEX'),no_validation_access=True)
    save(out/'TRAINING_FROZEN.json',frozen)
    z=torch.load(out/'best.pt',map_location='cpu');model.load_state_dict(z['state_dict'])
    dev=evaluate(model,load_rows('dev'),batch)
    save(out/'development_metrics.json',dict(metrics=dev,interpretation='Training development only; no independent test claim'))
    from baseline_evaluate import evaluate_comparison
    comparison=evaluate_comparison(model,seed)
    save(out/'complete.json',dict(frozen,development=dev,comparison=comparison));save(out/'progress.json',dict(frozen,stage='complete'))
    print('BASELINE_TRAINING_COMPLETE',method,seed,json.dumps(dev),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['smoke','train'])
    ap.add_argument('--method',choices=['SmartRefine','SRefiner'],required=True);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=ap.parse_args()
    if a.mode=='smoke':smoke(a.method,a.device)
    else:
        # A scheduled job may start only after its real GPU forward/backward
        # smoke succeeds. CPU preflight never substitutes for this gate.
        smoke(a.method,'cuda')
        train(a.method,a.seed)
