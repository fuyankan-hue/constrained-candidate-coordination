"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import argparse,copy,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from ccc.core.refiner_womd import Refiner,collate,objective
from ccc.runtime import refiner_setup as setup
from ccc.io import save_torch as save


def load(root,split,label):
    out=[]
    for f in sorted((root/'data').glob(split+'_*.pt')):out.extend(r['banks'][label] for r in torch.load(str(f),map_location='cpu'))
    return sorted(out,key=lambda r:r['scene'])


def weight_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def development(model,rows):
    model.eval();ade=fde=count=0.
    for at in range(0,len(rows),16):
        batch=rows[at:at+16];x,y=collate(batch,[1]*len(batch),'cuda');pred=x['path_local']+model(x)
        err=(pred-y['gt']).norm(dim=-1);valid=y['valid'];good=valid[:,:,-1];aa=(err*valid).sum(-1)/valid.sum(-1).clamp_min(1)
        ade+=float(aa[good].sum());fde+=float(err[:,:,-1][good].sum());count+=int(good.sum())
    return {'ADE':ade/count,'FDE':fde/count,'targets':count,'score':(ade+.25*fde)/count}


def main(root,label,seed,smoke):
    setup();torch.set_num_threads(2);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    p=json.loads((root/'protocol.json').read_text());cfg=p['training'];rows=load(root,'train',label);dev=load(root,'tune',label)
    assert len(rows)==8192 and len(dev)==1024
    model=Refiner(False).cuda();assert sum(z.numel() for z in model.parameters())==76026
    init=weight_hash(model);x,y=collate(rows[:4],[1]*4,'cuda');assert torch.equal(model(x),torch.zeros_like(x['path_local']))
    poisoned=copy.deepcopy(rows[:4])
    for r in poisoned:r['targets']['gt_local'].fill_(9999);r['targets']['valid'].logical_not_()
    xp,_=collate(poisoned,[1]*4,'cuda');assert all(torch.equal(x[k],xp[k]) for k in x)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['max_epochs'],eta_min=cfg['lr']*.1)
    if smoke:
        for _ in range(3):opt.zero_grad(set_to_none=True);loss,delta=objective(model,x,y);assert torch.isfinite(loss);loss.backward();opt.step()
        assert torch.equal(model(x),model(xp)) and model(x).norm(dim=-1).max()<=1.00001
        (root/'training_smoke.json').write_text(json.dumps({'pass':True,'zero_initial_residual':True,'future_label_poison_invariant':True,'original_loss_backward_finite':True,'parameters':76026}))
        print('SM_TRAIN_SMOKE_PASS',flush=True);return
    out=root/'models'/label/('seed%d'%seed);out.mkdir(parents=True,exist_ok=True);bestfile=out/'best.pt';assert not bestfile.exists(),'Never overwrite a completed or partial training run'
    ph=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest();history=[];initial=development(model,dev);best=initial['score'];selected=0
    history.append({'epoch':0,'development':initial})
    save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':0,'label':label,'seed':seed,'protocol_SHA256':ph},bestfile)
    start=time.monotonic();steps=0;order_hash=hashlib.sha256()
    for epoch in range(1,cfg['max_epochs']+1):
        order=np.random.default_rng(seed+epoch).permutation(len(rows));order_hash.update(order.tobytes());model.train();loss_sum=0.
        for at in range(0,len(rows),cfg['batch_size']):
            batch=[rows[int(i)] for i in order[at:at+cfg['batch_size']]];x,y=collate(batch,[1]*len(batch),'cuda')
            opt.zero_grad(set_to_none=True);loss,delta=objective(model,x,y);assert torch.isfinite(loss) and delta.norm(dim=-1).max()<=1.00001
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm);opt.step();steps+=1;loss_sum+=float(loss)
        metric=development(model,dev);history.append({'epoch':epoch,'mean_training_loss':loss_sum/512,'development':metric,'steps':steps})
        if metric['score']<best-cfg['min_delta']:
            best=metric['score'];selected=epoch
            save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':epoch,'label':label,'seed':seed,'protocol_SHA256':ph},bestfile)
        sched.step();print('SM_EPOCH',label,seed,epoch,'score',metric['score'],'seconds',round(time.monotonic()-start,1),flush=True)
    assert steps==5120
    save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':10,'steps':steps,'protocol_SHA256':ph},out/'last.pt')
    result={'pass':True,'label':label,'seed':seed,'parameters':76026,'selected_epoch':selected,'epochs_completed':10,'steps':steps,
        'initial_state_SHA256':init,'scene_order_SHA256':order_hash.hexdigest(),'history':history,'seconds':time.monotonic()-start,
        'peak_GPU_MiB':torch.cuda.max_memory_allocated()/1024**2,'best_SHA256':hashlib.sha256(bestfile.read_bytes()).hexdigest()}
    (out/'complete.json').write_text(json.dumps(result,indent=2));print('SM_TRAIN_COMPLETE',label,seed,'seconds',result['seconds'],flush=True)

