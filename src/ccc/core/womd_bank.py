"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import torch
import numpy as np

def make(v,model):
    a,b=v['m1'],v['m2'];ids=a['original_agent_id'];n=len(ids);assert n>0
    assert torch.equal(ids,b['target_ids']) and torch.equal(ids,b['context_ids'])
    choices=[a['pi'].argmax(-1).tolist(),model.predict(v)['modes']]
    current=b['current'].double();h=v['history'];theta=h['heading'][ids,10].double();c,s=theta.cos(),theta.sin()
    rotation=torch.stack([c,-s,s,c],-1).reshape(n,2,2)
    chosen=torch.stack([a['world'][torch.arange(n),m] for m in choices]).double()
    local=torch.einsum('bntc,ncd->bntd',chosen-current[None,:,None],rotation)
    origin=current[0];scene=chosen-origin;current_scene=current-origin
    histpos=torch.einsum('ntc,ncd->ntd',h['position'][ids,:,:2].double()-current[:,None],rotation)/20
    histvel=torch.einsum('ntc,ncd->ntd',h['velocity'][ids,:,:2].double(),rotation)/10
    angle=h['heading'][ids].double()-theta[:,None];past=h['valid_mask'][ids].bool()
    hist=torch.cat([histpos,histvel,angle.sin()[...,None],angle.cos()[...,None],past.double()[...,None]],-1)
    hist[~past]=0;history=hist.reshape(n,77)
    p=a['pi'].double().softmax(-1);entropy=-(p*p.clamp_min(1e-12).log()).sum(-1);gap=a['pi'].double().sort(-1).values[:,-1]-a['pi'].double().sort(-1).values[:,-2]
    prob=torch.stack([torch.cat([p,p[torch.arange(n),m,None],entropy[:,None],gap[:,None]],-1) for m in choices])
    hidden=torch.stack([b['refine'][torch.arange(n),m] for m in choices])
    length=h['length'][ids,10] if h['length'].ndim==2 else h['length'][ids]
    width=h['width'][ids,10] if h['width'].ndim==2 else h['width'][ids]
    physical=torch.stack([length/5,width/3,h['type'][ids].float()/4],-1).float()
    velocity=(chosen-torch.cat([current[None,:,None].expand(2,-1,-1,-1),chosen[:,:,:-1]],2))/.1
    observed_v=h['velocity'][ids,10,:2].double()
    accel=(velocity-torch.cat([observed_v[None,:,None].expand(2,-1,-1,-1),velocity[:,:,:-1]],2))/.1
    pair=[]
    for anchor in range(2):
        data=torch.zeros(n,n,12,dtype=torch.float64)
        for i in range(n):
            for j in range(n):
                r=chosen[anchor,j]-chosen[anchor,i];dist=r.norm(dim=-1);k=int(dist.argmin())
                vi=(velocity[anchor,i,k]@rotation[i]).clamp(-40,40)/10;vj=(velocity[anchor,j,k]@rotation[i]).clamp(-40,40)/10
                ai=(accel[anchor,i,k]@rotation[i]).clamp(-20,20)/5;aj=(accel[anchor,j,k]@rotation[i]).clamp(-20,20)/5
                xy=(r[k]@rotation[i]).clamp(-100,100)/20
                data[i,j]=torch.cat([vi,vj,ai,aj,xy,torch.tensor([min(float(dist[k]),100)/20,(k+1)/80.],dtype=torch.float64)])
        pair.append(data)
    inputs=dict(path_local=local.float(),path_scene=scene.float(),refine=hidden.float(),history=history.float(),prob=prob.float(),
        physical=physical,current=current_scene.float(),rotation=rotation.float(),pair_motion=torch.stack(pair).float())
    for k,t in inputs.items():assert torch.isfinite(t).all(),k
    gt_local=torch.einsum('ntc,ncd->ntd',a['gt'].double()-current[:,None],rotation).float()
    size=torch.stack([length,width],-1).float()
    return {'scene':v['scene'],'inputs':inputs,'targets':{'gt_local':gt_local,'valid':a['valid'].bool()},
        'evaluation':{'base_world':chosen.float(),'gt_world':a['gt'].float(),'valid':a['valid'].bool(),
            'current':current.float(),'heading':theta.float(),'size':size,'type':h['type'][ids],
            'agent_ids':ids,'track_ids':v['track_ids'],'modes':choices},'source_checkpoint_SHA256':v['checkpoint_sha256']}

