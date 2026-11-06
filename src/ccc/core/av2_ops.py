"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import itertools,time
import numpy as np
import torch
from .solvers import project,select,solve,geometry_arrays

def point_tables(value):
    w=value['m1']['world'].double().numpy();n=len(w)
    pairs=np.asarray(list(itertools.combinations(range(n),2)),dtype=int).reshape(-1,2)
    hits=[];steps=[]
    for i,j in pairs:
        hit=np.linalg.norm(w[i,:,None]-w[j,None,:],axis=-1)<1.0
        hits.append(hit.any(-1));steps.append(hit.sum(-1))
    return {'pairs':pairs,'any':np.asarray(hits,dtype=bool).reshape(-1,6,6),
            'steps':np.asarray(steps,dtype=np.int16).reshape(-1,6,6)}


def choose(row,model):
    # No GT, FDE or future validity is passed to a model or geometry function.
    obs={k:row[k] for k in ['world','pi','current','near','far','refine']}
    n=len(obs['pi']);base=obs['pi'].argmax(-1).tolist();world=obs['world'].double()
    edges=[(i,j) for i in range(n) for j in range(i+1,n)
           if float((obs['current'][i]-obs['current'][j]).norm())<75.]
    allowed=(world-world[torch.arange(n),base,None]).norm(dim=-1).amax(-1)<=20.
    with torch.inference_mode():
        C=model(*[obs[k] for k in ['near','far','refine','world']],torch.tensor(edges,dtype=torch.long).reshape(-1,2)).cpu()
    r={'pi':obs['pi'].double().numpy(),'base':base,'edges':edges,'allowed':allowed.numpy(),'C_old':C}
    v={'m1':{'world':world}}
    raw=point_tables(v);hard,G=geometry_arrays(raw,edges,n)
    bm,bd=solve(r['pi'],.5*G,edges,r['allowed'],base,max_sweeps=8)
    em,ed=solve(r['pi'],C.double().numpy(),edges,r['allowed'],base,hard=hard,max_sweeps=8)
    # Inject only the geometry adapter into the unmodified projection routine.
    project.__globals__['tables']=point_tables
    vp,rp,alpha=project(v,r)
    nm,nd=select(vp,rp,'NoRCIM');rm,rd=select(vp,rp,'main')
    ii=torch.arange(n)
    return {'A':{'modes':base,'path':world[ii,base].numpy()},
            'E':{'modes':em,'path':world[ii,em].numpy(),'search':ed},
            'B':{'modes':bm,'path':world[ii,bm].numpy(),'search':bd},
            'NoRCIM':{'modes':nm,'path':vp['m1']['world'][ii,nm].numpy(),'search':nd},
            'RCIM':{'modes':rm,'path':vp['m1']['world'][ii,rm].numpy(),'search':rd},
            'alpha':alpha,'edges':np.asarray(edges,dtype=int).reshape(-1,2)}


def hits(path):
    n=len(path);pairs=np.asarray(list(itertools.combinations(range(n),2)),dtype=int).reshape(-1,2)
    if not len(pairs):return pairs,np.zeros((0,path.shape[1]),dtype=bool)
    return pairs,np.linalg.norm(path[pairs[:,0]]-path[pairs[:,1]],axis=-1)<1.0


def protect(base,proposal):
    _,old=hits(base);identity=old.any(-1);budget=int(old.sum())
    for beta in [1.,.5,.25,.125,0.]:
        p=base+beta*(proposal-base);_,h=hits(p)
        if not (h.any(-1)&~identity).any() and int(h.sum())<=budget:return p,beta
    raise AssertionError('The incoming path must be feasible')


def bank(row,path,modes):
    n=len(modes);current=row['current'].double();theta=row['heading'].double()
    c,s=theta.cos(),theta.sin();rotation=torch.stack([c,-s,s,c],-1).reshape(n,2,2)
    w=torch.as_tensor(path,dtype=torch.float64)
    local=torch.einsum('ntc,ncd->ntd',w-current[:,None],rotation)
    scene=w-current[0];current_scene=current-current[0]
    hp=torch.einsum('ntc,ncd->ntd',row['history_position'].double()-current[:,None],rotation)/20
    hv=torch.einsum('ntc,ncd->ntd',row['history_velocity'].double(),rotation)/10
    angle=row['history_heading'].double()-theta[:,None];past=row['history_valid'].bool()
    hist=torch.cat([hp,hv,angle.sin()[...,None],angle.cos()[...,None],past.double()[...,None]],-1)
    hist[~past]=0
    p=row['pi'].double().softmax(-1);entropy=-(p*p.clamp_min(1e-12).log()).sum(-1)
    gap=row['pi'].double().sort(-1).values[:,-1]-row['pi'].double().sort(-1).values[:,-2]
    prob=torch.cat([p,p[torch.arange(n),modes,None],entropy[:,None],gap[:,None]],-1)
    hidden=row['refine'][torch.arange(n),modes]
    # Explicit missing-size channels, not invented observed lengths/widths.
    physical=torch.stack([torch.zeros(n),torch.zeros(n),row['observed_type'].float()/10],-1)
    velocity=(w-torch.cat([current[:,None],w[:,:-1]],1))/.1
    acceleration=(velocity-torch.cat([row['history_velocity'][:,-1,:].double()[:,None],velocity[:,:-1]],1))/.1
    pair=torch.zeros(n,n,12,dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            r=w[j]-w[i];dist=r.norm(dim=-1);k=int(dist.argmin())
            vi=(velocity[i,k]@rotation[i]).clamp(-40,40)/10;vj=(velocity[j,k]@rotation[i]).clamp(-40,40)/10
            ai=(acceleration[i,k]@rotation[i]).clamp(-20,20)/5;aj=(acceleration[j,k]@rotation[i]).clamp(-20,20)/5
            xy=(r[k]@rotation[i]).clamp(-100,100)/20
            pair[i,j]=torch.cat([vi,vj,ai,aj,xy,torch.tensor([min(float(dist[k]),100)/20,(k+1)/60.])])
    inputs=dict(path_local=local.float()[None],path_scene=scene.float()[None],refine=hidden.float()[None],
                history=hist.reshape(n,77).float(),prob=prob.float()[None],physical=physical.float(),
                current=current_scene.float(),rotation=rotation.float(),pair_motion=pair.float()[None])
    assert all(torch.isfinite(v).all() for v in inputs.values())
    gt=torch.einsum('ntc,ncd->ntd',row['gt'].double()-current[:,None],rotation).float()
    return {'scene':row['scene'],'inputs':inputs,'targets':{'gt_local':gt,'valid':row['valid'].bool()},
            'path':np.asarray(path,dtype=np.float64),'rotation':rotation.numpy(),'modes':list(modes)}


@torch.inference_mode()
def refine(row,path,modes,model,collate,device='cuda',radius=None):
    b=bank(row,path,modes);x,_=collate([b],[0],device)
    delta=model(x)[0,:len(path)]
    if radius is not None:
        radius=torch.as_tensor(radius,device=device)
        assert torch.isfinite(radius).all() and (radius>=0).all() and (radius<=1).all()
        delta=delta*radius[:,None,None]
    world=np.einsum('ntc,ndc->ntd',delta.double().cpu().numpy(),b['rotation'])
    assert np.linalg.norm(world,axis=-1).max()<=1.00001
    result,beta=protect(path,path+world)
    return result,beta,path+world


def measure(row,path,original):
    path=np.asarray(path,dtype=np.float64);gt=row['gt'].double().numpy();valid=row['valid'].numpy().astype(bool)
    good=valid[:,-1];err=np.linalg.norm(path-gt,axis=-1);ae=(err*valid).sum(-1)/np.maximum(valid.sum(-1),1)
    original_err=np.linalg.norm(original[:,-1]-gt[:,-1],axis=-1);delta=err[:,-1]-original_err
    pairs,h=hits(path);actors=np.zeros(len(path),dtype=bool)
    if len(pairs):actors[pairs[h.any(-1)].ravel()]=True
    velocity=np.diff(np.concatenate([row['current'].double().numpy()[:,None],path],1),axis=1)/.1
    acceleration=np.diff(velocity,axis=1)/.1;jerk=np.diff(acceleration,axis=1)/.1
    focal=good&row['is_focal'].numpy()
    return {'targets':int(good.sum()),'all_predicted_targets':len(path),'pairs':len(pairs),
            'ade_sum':float(ae[good].sum()),'fde_sum':float(err[good,-1].sum()),
            'overlap':int(h.any(-1).sum()),'overlap_steps':int(h.sum()),
            'actors_with_conflict':int(actors.sum()),'gt5':int((delta[good]>5.000001).sum()),
            'improved':int((delta[good]<-1e-6).sum()),'degraded':int((delta[good]>1e-6).sum()),
            'delta_fde':delta[good].tolist(),'focal_targets':int(focal.sum()),
            'focal_ade_sum':float(ae[focal].sum()),'focal_fde_sum':float(err[focal,-1].sum()),
            'max_shift_from_A':float(np.linalg.norm(path-original,axis=-1).max()),
            'acceleration_sum':float(np.linalg.norm(acceleration,axis=-1).sum()),'acceleration_count':acceleration.shape[0]*acceleration.shape[1],
            'jerk_sum':float(np.linalg.norm(jerk,axis=-1).sum()),'jerk_count':jerk.shape[0]*jerk.shape[1]}

