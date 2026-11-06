"""Existing Refine-only RCIM architecture, with native 60-step AV2 supervision."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from motion_model import MotionNFI

ROOT=Path('/workspace/output')


def save(path,value,pt=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    if pt:
        with tmp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    else:tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
    os.replace(tmp,path)


class RCIM(MotionNFI):
    def forward(self,near,far,refine,world,edges):
        assert world.shape==(len(refine),6,60,2)
        near=torch.zeros_like(near);far=torch.zeros_like(far)
        if not len(edges):return world.new_zeros((0,6,6))
        i,j=edges[:,0],edges[:,1]
        wi,wj=world[i].unsqueeze(2),world[j].unsqueeze(1)
        d=torch.linalg.vector_norm(wi-wj,dim=-1)
        dn,df=d[...,:30],d[...,30:]
        ng=torch.stack([dn.amin(-1)/20,dn[...,-1]/20,(dn[...,0]-dn[...,-1])/20,dn[...,10]/20],-1)
        fg=torch.stack([df.amin(-1)/20,df[...,-1]/20,(df[...,0]-df[...,-1])/20,df[...,19]/20],-1)
        # Keep the original 2 s / 6 s / endpoint feature meanings. On AV2,
        # the 6 s feature coincides with the endpoint; no new time tuning.
        gg=torch.stack([d.amin(-1)/20,d[...,19]/20,d[...,59]/20,d[...,-1]/20],-1)
        ng=self.geometry(wi,wj,0,30,ng)
        fg=self.geometry(wi,wj,30,60,fg)
        gg=self.geometry(wi,wj,0,60,gg)
        zn=self.near_net(self.pair_embedding(near,edges,self.near_proj,ng))
        zf=self.far_net(torch.cat([self.pair_embedding(far,edges,self.far_proj,fg),zn],-1))
        zr=self.ref_net(torch.cat([self.pair_embedding(refine,edges,self.ref_proj,gg),zf],-1))
        return self.head(zr).squeeze(-1)


def inputs(row,edges,device):
    return [row[k].to(device) for k in ['near','far','refine','world']]+[edges.to(device)]


def labels(row):
    valid=row['valid'].bool();current=row['current']
    pairs=[(i,j) for i in range(len(current)) for j in range(i+1,len(current))
           if bool(valid[i,-1] and valid[j,-1]) and float((current[i]-current[j]).norm())<=75
           and int((valid[i]&valid[j]).sum())>=45]
    edges=torch.tensor(pairs,dtype=torch.long).reshape(-1,2)
    if not pairs:return edges,None
    rs=[]
    for i,j in pairs:
        mask=valid[i]&valid[j]
        pred=row['world'][i,:,None]-row['world'][j,None,:]
        gt=row['gt'][i]-row['gt'][j]
        er=(pred-gt).norm(dim=-1)
        rs.append(er[...,mask].mean(-1)+.25*er[...,-1])
    costs=torch.stack(rs)
    return edges,torch.softmax(-costs.reshape(-1,36)/3,dim=-1)


def loss(model,row,device):
    edges,q=labels(row)
    if q is None:return None,0
    c=model(*inputs(row,edges,device)).reshape(-1,36)
    value=-(q.to(device)*c.log_softmax(-1)).sum(-1).mean()
    return value,len(edges)


def smoke():
    torch.set_num_threads(1);torch.manual_seed(42)
    model=RCIM();assert sum(p.numel() for p in model.parameters())==44449
    rows=torch.load(ROOT/'probe_features.pt',map_location='cpu')['rows']
    row=next(r for r in rows if labels(r)[1] is not None)
    edges,q=labels(row);assert torch.allclose(q.sum(-1),torch.ones(len(edges)),atol=1e-6)
    c=model(*inputs(row,edges,'cpu'));reverse=model(*inputs(row,edges.flip(-1),'cpu')).transpose(-1,-2)
    assert torch.allclose(c,reverse,atol=1e-6,rtol=1e-6)
    poison=dict(row,gt=torch.full_like(row['gt'],1e4),valid=~row['valid'])
    assert torch.equal(c,model(*inputs(poison,edges,'cpu')))
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    values=[]
    for step in range(3):
        opt.zero_grad(set_to_none=True);v,n=loss(model,row,'cpu');assert torch.isfinite(v);v.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm)
        opt.step();values.append(float(v))
    save(ROOT/'RCIM_AV2_TRAINING_SMOKE_PASS.json',dict(passed=True,parameters=44449,
         original_joint_ce=True,bidirectional_pair_symmetry=True,prediction_inputs_gt_invariant=True,
         finite_loss_and_gradient=True,losses=values,eligible_edges=len(edges),
         world_shape=list(row['world'].shape),future_steps=60,minimum_common_future_steps=45,
         original_backbone_not_trained=True))
    print('AV2_RCIM_TRAINING_SMOKE_PASS',values,flush=True)


def manifest():
    cfg=ROOT/'AV2_MODULE_PROTOCOL.json'
    value=json.loads(cfg.read_text());assert value['RCIM']['loss']=='original_joint_ce'
    return value


def rows_for(split,epoch=0):
    files=sorted((ROOT/'cache/train').glob('*.pt'))
    if split=='fit':files=[files[int(i)] for i in np.random.default_rng(42+epoch).permutation(len(files))]
    for p in files:
        obj=torch.load(p,map_location='cpu');rows=obj['rows']
        if split=='fit':rows=[rows[int(i)] for i in np.random.default_rng(42+epoch+int(p.stem)).permutation(len(rows))]
        for row in rows:
            role=int(hashlib.sha256(row['scene'].encode()).hexdigest()[:8],16)%10
            target='calibration' if role==0 else 'development' if role==1 else 'fit'
            if target==split:yield row


@torch.no_grad()
def evaluate(model,device):
    model.eval();total=0.;count=0;edges_count=0
    for row in rows_for('calibration'):
        value,edges=loss(model,row,device)
        if value is not None:total+=float(value)*edges;count+=1;edges_count+=edges
    assert count>0
    return {'joint_ce':total/edges_count,'scenes_with_supervision':count,'edges':edges_count,
            'weighting':'original edge-weighted calibration CE; training loss remains mean over edges within each scene'}


def train():
    assert json.loads((ROOT/'RCIM_AV2_TRAINING_SMOKE_PASS.json').read_text())['passed']
    assert json.loads((ROOT/'FEATURES_COMPLETE.json').read_text())['passed']
    cfg=manifest();torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42)
    device='cuda';model=RCIM().to(device)
    assert sum(p.numel() for p in model.parameters())==44449
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    out=ROOT/'models/rcim_seed42';out.mkdir(parents=True,exist_ok=True)
    first=1;best=float('inf');patience=0;history=[];selected=None;steps=0
    if (out/'resume.pt').exists():
        z=torch.load(out/'resume.pt',map_location='cpu');model.load_state_dict(z['state_dict']);opt.load_state_dict(z['optimizer'])
        first=z['epoch']+1;best=z['best'];patience=z['patience'];history=z['history'];selected=z['selected'];steps=z['steps']
    started=time.time()
    for epoch in (range(first,9) if patience<3 else []):
        model.train();total=0.;scenes=0;all_scenes=0;edge_count=0;tick=time.time()
        for row in rows_for('fit',epoch):
            all_scenes+=1;opt.zero_grad(set_to_none=True);value,edges=loss(model,row,device)
            if value is not None:
                assert torch.isfinite(value);value.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                assert torch.isfinite(norm);opt.step();total+=float(value.detach());scenes+=1;edge_count+=edges;steps+=1
            if all_scenes%2000==0:
                z=dict(stage='rcim_training',epoch=epoch,max_epochs=8,scenes_done=all_scenes,
                       scenes_total=cfg['splits']['fit_count'],optimizer_steps=steps,seconds=time.time()-tick,
                       loss=total/max(scenes,1),selected_epoch=selected)
                save(ROOT/'module_state.json',z);print('AV2_RCIM_PROGRESS',json.dumps(z),flush=True)
        assert scenes>0
        cal=evaluate(model,device);rec={'epoch':epoch,'training_joint_ce':total/scenes,'calibration':cal,
                                      'training_scenes':all_scenes,'supervised_scenes':scenes,'edges':edge_count,
                                      'optimizer_steps':steps,'epoch_seconds':time.time()-tick}
        history.append(rec)
        if cal['joint_ce']<best-1e-4:
            best=cal['joint_ce'];selected=epoch;patience=0
            save(out/'best.pt',{'state_dict':{k:v.cpu().clone() for k,v in model.state_dict().items()},
                 'epoch':epoch,'calibration':cal,'seed':42,'parameters':44449,'protocol':cfg},True)
        else:patience+=1
        save(out/'resume.pt',{'state_dict':{k:v.cpu().clone() for k,v in model.state_dict().items()},
             'optimizer':opt.state_dict(),'epoch':epoch,'best':best,'patience':patience,'history':history,
             'selected':selected,'steps':steps},True)
        save(out/'history.json',history);print('AV2_RCIM_EPOCH',json.dumps(rec),flush=True)
        if patience>=3:break
    result=dict(passed=True,selected_epoch=selected,best_calibration_joint_ce=best,
                completed_epochs=history[-1]['epoch'],steps=steps,seconds=time.time()-started,
                history=history,backbone_frozen=True,validation_not_used_for_module_selection=True)
    save(ROOT/'RCIM_TRAINING_COMPLETE.json',result)
    print('AV2_RCIM_TRAINING_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','train']);a=p.parse_args()
    if a.mode=='smoke':smoke()
    else:train()
