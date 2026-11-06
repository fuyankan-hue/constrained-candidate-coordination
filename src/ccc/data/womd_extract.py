"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import torch
from pathlib import Path
from torch_geometric.data import HeteroData
from predictors import NDPNet
from transforms import TargetBuilder
from .womd_m1 import global_xy,candidate_features,interaction_features
from .womd_validate import validate_m1
HISTORY=('position','velocity','heading','valid_mask','length','width','height','type')
CHECKPOINT=None
CHECKPOINT_SHA=None
OUT=Path('.')

def validate(v,record):
    assert v['cache_contract']=='target-only-fp32-v2-official-target-coverage'
    assert v['scene']==record['scene'] and v['record_sha256']==record['sha256'] and v['checkpoint_sha256']==CHECKPOINT_SHA
    a,b=v['m1'],v['m2'];n=len(a['original_agent_id'])
    if n:validate_m1(a,v['scene'],b)
    else:assert a['world'].shape==(0,6,80,2) and b['near'].shape==(0,6,128)
    assert torch.equal(a['original_agent_id'],b['context_ids']) and torch.equal(a['original_agent_id'],b['target_ids'])
    assert v['official_track_ids']==v['predicted_track_ids'],(v['scene'],'missing official requested target',v['official_track_ids'],v['predicted_track_ids'])
    for key in HISTORY:assert torch.isfinite(v['history'][key]).all()
    return n


def setup():
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.manual_seed(42);model=NDPNet.load_from_checkpoint(str(CHECKPOINT),map_location='cpu',strict=True).cuda().eval()
    assert (model.num_historical_steps,model.num_future_steps,model.num_modes)==(11,80,6)
    captured={};handles=[]
    for name,attr in {'near':'m2m_propose_attn_layer_proximal','far':'m2m_propose_attn_layer_lat','refine':'m2m_refine_attn_layer'}.items():
        def hook(module,inputs,out,name=name):captured[name]=out.detach().clone()
        handles.append(getattr(model.decoder,attr).register_forward_hook(hook))
    return model,captured,handles


def make(model,captured,row,smoke=False):
    raw=row['raw'];record=row['record'];sid=record['scene'];data=TargetBuilder(11,80)(HeteroData(raw));agent=data['agent']
    # Vector-valid history requires the previous frame; it is NOT current presence.
    # Predict every official requested target, as in the repository's test split.
    now=agent['observed_current_valid'].bool();ids=torch.where(agent['target_mask'].bool())[0];n=int(agent['num_nodes'])
    predicted=sorted(int(x) for x in agent['id'][ids].tolist());official=sorted(row['official_track_ids'])
    assert predicted==official,(sid,'official targets not covered by original current-valid rule',predicted,official)
    history={key:(agent[key][:,:11].clone() if agent[key].ndim>1 else agent[key].clone()) for key in HISTORY}
    # Labels are detached before the model input is scrubbed.
    gt=agent['position'][ids,11:91,:2].clone().float();valid=(agent['valid_mask'][ids,11:91]&agent['predict_mask'][ids,11:91]).bool().clone()
    for name in ('agent','map_polygon','map_point'):data[name]['ptr']=torch.tensor([0,int(data[name]['num_nodes'])])
    data=data.cuda();clean=data.clone();ca=clean['agent']
    for key in ('position','heading','velocity','abs_vel','length','width','height'):
        if key in ca and ca[key].ndim>1 and ca[key].shape[1]>11:ca[key][:,11:]=0
    requests=now.cuda()|ca['target_mask'].bool()
    ca['valid_mask'][:,11:]=False;ca['predict_mask'][:,11:]=requests[:,None]
    if 'target' in ca:ca['target'].zero_()
    with torch.inference_mode():
        captured.clear();out=model(clean)
        hidden={key:captured[key].reshape(n,6,128)[ids].cpu().float().clone() for key in ('near','far','refine')}
        if smoke:
            reference=data.clone();reference['agent']['predict_mask'][:,11:]=requests[:,None]
            original=model(reference)
            for key in ('loc_refine_pos','pi'):
                assert torch.allclose(out[key][ids],original[key][ids],atol=1e-4,rtol=1e-5),(sid,'scrub changes historical predictions',key,float((out[key][ids]-original[key][ids]).abs().max()))
            poisoned=clean.clone()
            for key in ('position','heading','velocity'):
                poisoned['agent'][key][:,11:]=123.456
            if 'target' in poisoned['agent']:poisoned['agent']['target'].fill_(-987.654)
            repeat=model(poisoned)
            for key in ('loc_refine_pos','pi'):assert torch.allclose(out[key][ids],repeat[key][ids],atol=1e-4,rtol=1e-5),(sid,'GT leak',key)
        local=out['loc_refine_pos'][ids,...,:2];pi=out['pi'][ids];ag=clean['agent'];pos=ag['position'][ids,10,:2]
        world=global_xy(local,pos[:,None,None,:],ag['heading'][ids,10][:,None,None])
        a=dict(scene=sid,unary=candidate_features(local,pi).cpu().float(),pair=interaction_features(world,pi,pos,ids).cpu().float(),
          pi=pi.cpu().float(),world=world.cpu().float(),gt=gt,valid=valid,valid8=valid[:,79],original_agent_id=ids.cpu(),
          fde=(world.cpu()[:,:,79]-gt[:,None,79]).norm(dim=-1).float())
        b=dict(scene=sid,context_ids=ids.cpu(),target_ids=ids.cpu(),near=hidden['near'],far=hidden['far'],refine=hidden['refine'],
          world=a['world'],current=pos.cpu().float(),pi=a['pi'],is_target=torch.ones(len(ids),dtype=torch.bool))
    result=dict(scene=sid,record_sha256=record['sha256'],checkpoint_sha256=CHECKPOINT_SHA,role=record['role'],previous=bool(record['previous']),
      m1=a,m2=b,history=history,track_ids=agent['id'][ids].clone(),official_track_ids=official,predicted_track_ids=predicted,
      agent_count=n,map_polygons=row['num_map_polygons'],cache_contract='target-only-fp32-v2-official-target-coverage',objects_of_interest=row['objects_of_interest'])
    validate(result,record)
    if smoke:
        p=OUT/'cache/m1_train'/(sid+'.pt')
        if p.exists():
            legacy=torch.load(str(p),map_location='cpu');assert torch.equal(legacy['original_agent_id'],ids.cpu())
            for key in ('pi','world','unary'):assert torch.allclose(legacy[key],a[key],atol=1e-3,rtol=1e-4),(sid,'legacy replay',key)
    return result

