"""Thin training/evaluation wrapper around the official refinement cores.

This is an NDPNet/AV2 port, not the authors' original backbone/checkpoint.
"""
import sys
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from baseline_data import ROOT

class Baseline(nn.Module):
    def __init__(self,method):
        super().__init__();self.method=method
        source=ROOT/'adapters'/method
        sys.path.insert(0,str(source))
        if method=='SmartRefine':
            from models.target_region import TargetRegion
            self.core=TargetRegion(future_steps=60,num_modes=6,node_dim=2,
                edge_dim=2,embed_dim=64,refine_num=5,seg_num=2,refine_radius=-1,r_lo=2,r_hi=10)
            self.iterations=5
        else:
            from refine.av2_target_region import TargetRegion_av2_multiagent
            self.core=TargetRegion_av2_multiagent(future_steps=60,num_modes=6,
                node_dim=2,edge_dim=2,embed_dim=64,refine_num=3,seg_num=4,local_radius=10)
            self.iterations=3

    def forward(self,args,info):
        deltas,heads=self.core(*args)
        positions=info['local'].transpose(0,1)
        stages=[]
        for delta in deltas:
            positions=positions+delta[...,:2]
            stages.append((positions,delta[...,2:]))
        return stages,heads

def nll(positions,scales,gt,valid):
    scale=scales.clamp_min(1e-6)
    error=(positions-gt).abs()
    loss=torch.log(2*scale)+error/scale
    return loss[valid].mean() if valid.any() else positions.sum()*0

def objective(model,args,labels,info):
    stages,heads=model(args,info)
    gt=labels['gt'];valid=labels['valid'];n=len(gt)
    idx=torch.arange(n,device=gt.device)
    eligible=valid.any(-1)
    if not eligible.any():return sum(s[0].sum()*0 for s in stages)
    if model.method=='SRefiner':
        losses=[]
        for positions,scale in stages:
            ade=((positions-gt).norm(dim=-1)*valid).sum(-1)/(valid.sum(-1)+1e-3)
            best=[];start=0
            for count in info['nlist']:
                mode=ade[:,start:start+count].sum(-1).argmin()
                best.append(mode.expand(count));start+=count
            mode=torch.cat(best)
            losses.append(nll(positions[mode,idx],scale[mode,idx],gt,valid))
        return torch.stack(losses).mean()
    # Author SmartRefine losses: per-target best-mode regression, soft CE,
    # and quality targets derived solely inside supervised training.
    pis,scores=heads
    traces=[info['local'].transpose(0,1)]+[s[0] for s in stages]
    l2=[((p-gt).norm(dim=-1)*valid).sum(-1) for p in traces]
    best_quality=torch.stack([v.min(0).values for v in l2])
    lo=best_quality.min(0).values;hi=best_quality.max(0).values
    reg=cls=quality=0.
    for j,error in enumerate(l2):
        best=error.argmin(0)
        q=((hi-error.min(0).values)/(hi-lo+1e-6)).clamp(0,1).detach()
        pred=scores[j].transpose(0,1)[best,idx]
        quality=quality+F.l1_loss(pred[eligible],q[eligible])
        if j:
            positions,scale=stages[j-1]
            reg=reg+nll(positions[best,idx],scale[best,idx],gt,valid)
            soft=F.softmax(-error[:,eligible]/valid[eligible].sum(-1).clamp_min(1),dim=0).t().detach()
            cls=cls+(-soft*F.log_softmax(pis[j-1][eligible],dim=-1)).sum(-1).mean()
    return reg/model.iterations+cls/model.iterations+.01*quality/(model.iterations+1)

def selected(model,stages,heads,info):
    path=stages[-1][0] # [6,N,60,2]
    if model.method=='SmartRefine':mode=heads[0][-1].argmax(-1)
    else:mode=torch.zeros(path.shape[1],device=path.device,dtype=torch.long)
    return path[mode,torch.arange(path.shape[1],device=path.device)],mode

@torch.no_grad()
def evaluate(model,rows,batch_fn,device='cuda',batch_size=16):
    model.eval();total={k:0. for k in ('ade','fde','targets','minfde','jointfde','joint_scenes')}
    for at in range(0,len(rows),batch_size):
        args,labels,info=batch_fn(rows[at:at+batch_size],model.method,device)
        stages,heads=model(args,info);path,_=selected(model,stages,heads,info)
        gt,valid=labels['gt'],labels['valid'];ok=valid[:,-1]
        error=(path-gt).norm(dim=-1)
        average=(error*valid).sum(-1)/valid.sum(-1).clamp_min(1)
        all_fde=(stages[-1][0][:,:,-1]-gt[None,:,-1]).norm(dim=-1)
        total['ade']+=float(average[ok].sum());total['fde']+=float(error[ok,-1].sum())
        total['targets']+=int(ok.sum());total['minfde']+=float(all_fde[:,ok].min(0).values.sum())
        start=0
        for count in info['nlist']:
            good=ok[start:start+count]
            if good.any():
                total['jointfde']+=float(all_fde[:,start:start+count][:,good].mean(-1).min())
                total['joint_scenes']+=1
            start+=count
    assert total['targets']>0
    return dict(ADE6=total['ade']/total['targets'],FDE6=total['fde']/total['targets'],
        minFDE6=total['minfde']/total['targets'],
        minJointFDE6=total['jointfde']/max(total['joint_scenes'],1),
        valid6_targets=int(total['targets']),scenes=len(rows),
        selection_metric='minFDE6' if model.method=='SmartRefine' else 'minJointFDE6')
