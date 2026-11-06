"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import torch
from torch import nn
from .motion_model import MotionNFI

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

