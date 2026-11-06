"""Bounded smooth residual refinement; equal-capacity independent/interactive arms.

References: SRefiner (ICCV2025) trajectory-motion-conditioned attention;
SmartRefine (CVPR2024) frozen proposal+embedding residual interface.
This is a new minimal adaptation, NOT a reproduction of their lane modules,
full soft-braid model, multimodal objective or iterative refinement system.
"""
import math
import numpy as np
import torch
from torch import nn

INPUTS=('path_local','path_scene','refine','history','prob','physical','current','rotation','pair_motion','agent_present')


def spline_basis():
    degree=3;n=12
    knots=np.r_[np.zeros(4),np.arange(1,9)*6/9,np.ones(4)*6]
    t=np.arange(1,61)/10.
    B=np.array([(t>=knots[i])&(t<knots[i+1]) for i in range(len(knots)-1)],float).T
    for d in range(1,degree+1):
        next_=np.zeros((60,B.shape[1]-1))
        for i in range(next_.shape[1]):
            if knots[i+d]>knots[i]:next_[:,i]+=(t-knots[i])/(knots[i+d]-knots[i])*B[:,i]
            if knots[i+d+1]>knots[i+1]:next_[:,i]+=(knots[i+d+1]-t)/(knots[i+d+1]-knots[i+1])*B[:,i+1]
        B=next_
    B[-1]=0;B[-1,-1]=1
    assert B.shape==(60,n) and np.allclose(B.sum(1),1) and B.min()>=-1e-12
    # First3 controls fixed0 -> zero initial displacement, velocity, acceleration.
    return torch.tensor(B[:,3:],dtype=torch.float32)



def sample_tokens(path):
    # Uniform 16-token physical-time grid; at 80 steps this equals 4::5 exactly.
    ix=torch.arange(1,17,device=path.device,dtype=path.dtype)*(path.shape[2]/16.)-1
    lo=ix.floor().long().clamp(0,path.shape[2]-1)
    hi=(lo+1).clamp(0,path.shape[2]-1)
    f=(ix-lo.to(ix.dtype))[None,None,:,None]
    return path[:,:,lo]*(1-f)+path[:,:,hi]*f

class Refiner(nn.Module):
    def __init__(self,interaction,max_shift=1.,dim=64,heads=4):
        super().__init__();self.interaction=bool(interaction);self.max_shift=max_shift;self.dim=dim;self.heads=heads
        self.hist=nn.Sequential(nn.Linear(77,64),nn.GELU(),nn.Linear(64,32))
        self.feat=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,64),nn.GELU(),nn.Linear(64,32))
        self.node=nn.Sequential(nn.Linear(83,dim),nn.LayerNorm(dim),nn.GELU())
        self.temporal=nn.Conv1d(dim,dim,3,padding=1,groups=dim)
        self.qkv=nn.Linear(dim,3*dim);self.motion=nn.Sequential(nn.Linear(12,32),nn.GELU(),nn.Linear(32,heads))
        self.relative=nn.Sequential(nn.Linear(3,16),nn.GELU(),nn.Linear(16,heads))
        self.out=nn.Linear(dim,dim);self.norm1=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,128),nn.GELU(),nn.Linear(128,dim));self.norm2=nn.LayerNorm(dim)
        self.head=nn.Linear(16*dim,9*2)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.register_buffer('basis',spline_basis())
        self.register_buffer('times',torch.arange(1,17,dtype=torch.float32)*(6./16))

    def forward(self,x):
        assert set(x)==set(INPUTS),'Model consumes prediction/history allowlist only'
        path=x['path_local'];B,N,_,_=path.shape;T=16;D=self.dim;H=self.heads
        pos=sample_tokens(path);past=torch.cat([torch.zeros_like(path[:,:,:1]),path[:,:,:-1]],2);vel=(path-past)/.1
        vel=sample_tokens(vel).clamp(-40,40)/10
        tau=self.times[None,None,:,None].expand(B,N,T,1)
        tf=torch.cat([tau/6,torch.sin(math.pi*tau/6),torch.cos(math.pi*tau/6)],-1)
        h=self.hist(x['history'])[:,:,None].expand(-1,-1,T,-1);f=self.feat(x['refine'])[:,:,None].expand(-1,-1,T,-1)
        p=x['prob'][:,:,None].expand(-1,-1,T,-1);physical=x['physical'][:,:,None].expand(-1,-1,T,-1)
        z=self.node(torch.cat([h,f,p,pos/20,vel,tf,physical],-1))
        z=z+self.temporal(z.reshape(B*N,T,D).transpose(1,2)).transpose(1,2).reshape(B,N,T,D)
        L=N*T;flat=z.reshape(B,L,D);q,k,v=self.qkv(flat).chunk(3,-1)
        q=q.reshape(B,L,H,D//H).transpose(1,2);k=k.reshape(B,L,H,D//H).transpose(1,2);v=v.reshape(B,L,H,D//H).transpose(1,2)
        dots=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(D//H)
        # Query-local relative positions and time offsets between future tokens.
        positions=sample_tokens(x['path_scene']).reshape(B,L,2)
        r=positions[:,None,:,:]-positions[:,:,None,:]
        rotation=x['rotation'][:,:,None].expand(-1,-1,T,-1,-1).reshape(B,L,2,2)
        local=torch.einsum('blkc,blcd->blkd',r,rotation)/20
        ts=self.times[None,None,:].expand(B,N,T).reshape(B,L)
        dt=ts[:,None,:]-ts[:,:,None]
        rel=self.relative(torch.cat([local,dt[...,None]],-1)).permute(0,3,1,2)
        motion=self.motion(x['pair_motion']).permute(0,3,1,2)
        motion=motion[:,:,:,None,:,None].expand(-1,-1,-1,T,-1,T).reshape(B,H,L,L)
        bias=-r.square().sum(-1)/(2*20.**2)-dt.square()/2
        current=x['current'];graph=(current[:,:,None]-current[:,None]).norm(dim=-1)<75.
        if not self.interaction:graph=torch.eye(N,dtype=torch.bool,device=path.device)[None].expand(B,-1,-1)
        graph=graph[:,:,None,:,None].expand(-1,-1,T,-1,T).reshape(B,L,L)
        present=x['agent_present'][:,:,None].expand(-1,-1,T).reshape(B,L)
        # Padded queries may attend to their own token only; they never affect real agents.
        mask=graph&(dt.abs()<=1.000001)&present[:,None,:]
        diag=torch.eye(L,dtype=torch.bool,device=path.device)[None]
        mask|=(~present)[:,:,None]&diag
        logits=(dots+rel+motion+bias[:,None]).masked_fill(~mask[:,None],-1e4)
        attention=logits.softmax(-1)
        msg=torch.matmul(attention,v).transpose(1,2).reshape(B,L,D)
        z=self.norm1(flat+self.out(msg));z=self.norm2(z+self.ff(z))
        control=self.head(z.reshape(B,N,T*D)).reshape(B,N,9,2)
        control=self.max_shift*control/torch.sqrt(1+control.square().sum(-1,keepdim=True))
        delta=torch.einsum('tc,bncd->bntd',self.basis,control)
        return delta*x['agent_present'][:,:,None,None]


def collate(rows,anchors,device='cpu'):
    B=len(rows);N=max(len(r['inputs']['current']) for r in rows);result={}
    for key in INPUTS:
        if key=='agent_present':
            v=torch.zeros(B,N,dtype=torch.bool)
            for i,r in enumerate(rows):v[i,:len(r['inputs']['current'])]=True
        else:
            values=[r['inputs'][key][a] if key in ('path_local','path_scene','refine','prob','pair_motion') else r['inputs'][key] for r,a in zip(rows,anchors)]
            if key=='pair_motion':
                v=torch.zeros(B,N,N,12)
                for i,a in enumerate(values):v[i,:len(a),:len(a)]=a
            else:
                v=torch.zeros((B,N)+values[0].shape[1:])
                for i,a in enumerate(values):v[i,:len(a)]=a
        result[key]=v.to(device)
    gt=torch.zeros(B,N,60,2);valid=torch.zeros(B,N,60,dtype=torch.bool)
    for i,r in enumerate(rows):
        n=len(r['inputs']['current']);gt[i,:n]=r['targets']['gt_local'];valid[i,:n]=r['targets']['valid']
    return result,{'gt':gt.to(device),'valid':valid.to(device)}


def objective(model,x,y):
    delta=model(x);pred=x['path_local']+delta;valid=y['valid'];gt=y['gt']
    loss=(torch.nn.functional.smooth_l1_loss(pred,gt,reduction='none').sum(-1)*valid).sum()/valid.sum().clamp_min(1)
    last=valid[:,:,-1];endpoint=(torch.nn.functional.smooth_l1_loss(pred[:,:,-1],gt[:,:,-1],reduction='none').sum(-1)*last).sum()/last.sum().clamp_min(1)
    reg=(delta.square().sum(-1)*x['agent_present'][:,:,None]).sum()/(x['agent_present'].sum().clamp_min(1)*60)
    accel=(delta[:,:,2:]-2*delta[:,:,1:-1]+delta[:,:,:-2])/.01
    smooth=(accel.square().sum(-1)*x['agent_present'][:,:,None]).sum()/(x['agent_present'].sum().clamp_min(1)*58)
    return loss+.25*endpoint+.05*reg+.002*smooth,delta
