"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import torch

def item(v):
    a=v['m1'];b=v['m2'];selected=a['valid8'].nonzero().flatten().tolist();edges=[];labels=[]
    for x,i in enumerate(selected):
        for j in selected[x+1:]:
            if (b['current'][i]-b['current'][j]).norm()>75:continue
            both=a['valid'][i]&a['valid'][j]
            if int(both.sum())<60:continue
            pred=b['world'][i,:,None,both]-b['world'][j,None,:,both];gt=a['gt'][i,both]-a['gt'][j,both]
            ade=(pred-gt).norm(dim=-1).mean(-1);fde=(b['world'][i,:,None,79]-b['world'][j,None,:,79]-(a['gt'][i,79]-a['gt'][j,79])).norm(dim=-1)
            edges.append([i,j]);labels.append(ade+.25*fde)
    if not edges:return None
    e=torch.tensor(edges,dtype=torch.long);used,inverse=torch.unique(e,sorted=True,return_inverse=True)
    r={k:b[k][used].contiguous() for k in ('near','far','refine','world','pi')};r.update(scene=v['scene'],edges=inverse.reshape_as(e),rel=torch.stack(labels));assert torch.isfinite(r['rel']).all();return r

