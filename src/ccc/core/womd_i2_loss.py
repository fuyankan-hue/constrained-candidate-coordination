"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import numpy as np
import torch
from .womd_bank import make
from .refiner_womd import collate,INPUTS
from .womd_protection import pairs_for,safeguard,measure

class Choice:
    def __init__(self,m):self.m=m
    def predict(self,v):return {'modes':self.m}


def input_only(bank):
    indexed=('path_local','path_scene','refine','prob','pair_motion');n=len(bank['inputs']['current']);return {k:torch.ones(1,n,dtype=torch.bool,device='cuda') if k=='agent_present' else (bank['inputs'][k][1] if k in indexed else bank['inputs'][k])[None].cuda() for k in INPUTS}


def stats(row,path):
    mt,kin=measure(path,row['base'],row['meta'],row['pairs']);g=row['meta']['valid'][:,79].numpy().astype(bool);err=np.linalg.norm(path-row['meta']['gt_world'].double().numpy(),axis=-1);d=err[g,-1]-row['fdeA'][g];shiftA=np.linalg.norm(path-row['original_A'],axis=-1)
    return {'targets':mt['targets'],'pairs':mt['eligible_pairs'],'ade_sum':mt['ade_sum'],'fde_sum':mt['fde_sum'],'overlap':mt['overlap_pairs'],'overlap_steps':mt['overlap_steps'],'gt5':int((d>5.000001).sum()),'changed':int((np.asarray(row['modes'])[g]!=np.asarray(row['A_modes'])[g]).sum()),'improved':int((d< -1e-6).sum()),'degraded':int((d>1e-6).sum()),'deltas':d.tolist(),'shift_sum':mt['shift_sum'],'shift_steps':mt['shift_steps'],'max_shift_from_projected':mt['max_shift'],'max_shift_from_A':float(shiftA.max()),'accel_sum':float(kin[1].sum()),'accel_count':len(kin[1]),'jerk_sum':float(kin[2].sum()),'jerk_count':len(kin[2])}


def row_for(vp,recipe,modes,model):
    bank=make(vp,Choice(modes));n=len(modes);A=vp['m1']['pi'].argmax(-1).tolist();assert torch.equal(bank['evaluation']['agent_ids'],vp['m1']['original_agent_id']);w=vp['m1']['world'].double();bank['evaluation']['base_world']=torch.stack([w[torch.arange(n),A],w[torch.arange(n),modes]])
    base=bank['evaluation']['base_world'][1].numpy();originalA=bank['evaluation']['base_world'][0].numpy();assert np.linalg.norm(base-originalA,axis=-1).max()<=4.00000001
    with torch.no_grad():
        x=input_only(bank);local=model(x)[0].double().cpu().numpy();world=np.einsum('ntc,ndc->ntd',local,bank['inputs']['rotation'].double().numpy());assert np.linalg.norm(world,axis=-1).max()<=1.00001
    meta=bank['evaluation'];pairs=pairs_for(meta);firstsafe,beta=safeguard(base,base+world,meta,pairs);distance=np.linalg.norm(firstsafe-originalA,axis=-1).max(1);assert distance.max()<=5.000001;radius=np.clip(5.-1e-4-distance,0.,1.)
    vv=dict(vp);vv['m1']=dict(vp['m1'],world=vp['m1']['world'].double().clone());vv['m1']['world'][torch.arange(n),modes]=torch.as_tensor(firstsafe,dtype=torch.double);after=make(vv,Choice(modes));row={'scene':recipe['scene'],'bank_after':after,'base':base,'original_A':originalA,'firstsafe':firstsafe,'meta':meta,'pairs':pairs,'radius':torch.tensor(radius,dtype=torch.float32),'modes':list(modes),'A_modes':A,'firstP_beta':beta,'cache_SHA256':recipe['cache_SHA256']};row['fdeA']=np.linalg.norm(originalA-meta['gt_world'].double().numpy(),axis=-1)[:,-1];row['baseline']=stats(row,firstsafe);row['A_metrics']=stats(row,originalA);row['A_metrics'].update(changed=0,improved=0,degraded=0,shift_sum=0.,max_shift_from_projected=0.);assert row['baseline']['gt5']==0
    return row


def batch(rows,branch):
    selected=[r['variants'][branch] for r in rows];x,y=collate([r['bank_after'] for r in selected],[1]*len(rows),'cuda');radius=torch.zeros(x['current'].shape[:2],device='cuda')
    for i,r in enumerate(selected):radius[i,:len(r['radius'])]=r['radius'].cuda()
    return x,radius,y


def loss(model,x,radius,y):
    delta=model(x,radius);pred=x['path_local']+delta;valid=y['valid'];gt=y['gt'];good=valid[:,:,-1];present=x['agent_present'];pos=((pred-gt).norm(dim=-1)*valid).sum()/valid.sum().clamp_min(1);end=((pred[:,:,-1]-gt[:,:,-1]).norm(dim=-1)*good).sum()/good.sum().clamp_min(1);reg=(delta.square().sum(-1)*present[:,:,None]).sum()/(present.sum().clamp_min(1)*80);acc=(delta[:,:,2:]-2*delta[:,:,1:-1]+delta[:,:,:-2])/.01;smooth=(acc.square().sum(-1)*present[:,:,None]).sum()/(present.sum().clamp_min(1)*78);return pos+.25*end+.05*reg+.002*smooth

