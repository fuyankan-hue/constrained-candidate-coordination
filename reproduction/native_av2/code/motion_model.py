"""NFI motion-geometry ablation. All quantities are model predictions."""
import torch
from torch import nn
from nfi_model import NearFarInteraction

class MotionNFI(NearFarInteraction):
    def __init__(self,rich=True,hidden_dim=128,dim=32):
        super().__init__(hidden_dim,dim)
        self.rich=rich
        pair_dim=dim*3+16
        self.near_net[0]=nn.Linear(pair_dim,64)
        self.far_net[0]=nn.Linear(pair_dim+32,64)
        self.ref_net[0]=nn.Linear(pair_dim+32,64)

    def geometry(self,wi,wj,start,end,legacy):
        shape=legacy.shape[:-1]
        if not self.rich:return torch.cat([legacy,legacy.new_zeros((*shape,12))],-1)
        a=wi[...,start:end,:];b=wj[...,start:end,:]
        r=a-b;d=r.norm(dim=-1);r0=r[...,0,:];dr=r[...,-1,:]-r0
        d0=r0.norm(dim=-1).clamp_min(1.)
        va=a[...,-1,:]-a[...,0,:];vb=b[...,-1,:]-b[...,0,:]
        na=va.norm(dim=-1);nb=vb.norm(dim=-1)
        steps=end-start
        sa=(a[...,1:,:]-a[...,:-1,:]).norm(dim=-1).sum(-1)/(steps*.1)
        sb=(b[...,1:,:]-b[...,:-1,:]).norm(dim=-1).sum(-1)/(steps*.1)
        sa=sa.expand_as(d0);sb=sb.expand_as(d0);na=na.expand_as(d0);nb=nb.expand_as(d0)
        extra=torch.stack([
            d.mean(-1)/20,d.std(-1,unbiased=False)/20,d.argmin(-1).float()/max(1,steps-1),
            (r0*dr).sum(-1)/d0/20,
            (r0[...,0]*dr[...,1]-r0[...,1]*dr[...,0])/d0/20,
            dr.norm(dim=-1)/20,
            (sa+sb)/20,(sa-sb).abs()/20,
            (na+nb)/40,(na-nb).abs()/40,
            (va*vb).sum(-1)/(na*nb).clamp_min(1.),
            (r0*r[...,-1,:]).sum(-1)/(d0*r[...,-1,:].norm(dim=-1)).clamp_min(1.),
        ],-1)
        return torch.cat([legacy,extra],-1)

    def forward(self,near,far,refine,world,edges):
        assert near.shape==far.shape==refine.shape and world.shape==(len(near),6,80,2)
        if not len(edges):return world.new_zeros((0,6,6))
        wi=world[edges[:,0]].unsqueeze(2);wj=world[edges[:,1]].unsqueeze(1)
        distance=(wi-wj).norm(dim=-1);dn=distance[...,:30];df=distance[...,30:]
        ng=torch.stack([dn.amin(-1)/20,dn[...,-1]/20,(dn[...,0]-dn[...,-1])/20,dn[...,10]/20],-1)
        fg=torch.stack([df.amin(-1)/20,df[...,-1]/20,(df[...,0]-df[...,-1])/20,df[...,19]/20],-1)
        rg=torch.stack([distance.amin(-1)/20,distance[...,19]/20,distance[...,59]/20,distance[...,79]/20],-1)
        ng=self.geometry(wi,wj,0,30,ng);fg=self.geometry(wi,wj,30,80,fg);rg=self.geometry(wi,wj,0,80,rg)
        zn=self.near_net(self.pair_embedding(near,edges,self.near_proj,ng))
        zf=self.far_net(torch.cat([self.pair_embedding(far,edges,self.far_proj,fg),zn],-1))
        zr=self.ref_net(torch.cat([self.pair_embedding(refine,edges,self.ref_proj,rg),zf],-1))
        return self.head(zr).squeeze(-1)
