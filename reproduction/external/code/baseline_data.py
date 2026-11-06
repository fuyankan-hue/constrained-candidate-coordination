"""Prediction-only AV2 interfaces; labels remain in a separate dictionary."""
import copy,hashlib,json,os,pickle,time
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data

ROOT=Path('/workspace/external');BASE=Path('/workspace/output')
RAW=Path('/workspace/QCNet/data/av2_qcnet/train/processed')

def save(path,value,pt=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('wb' if pt else 'w') as f:
        if pt:torch.save(value,f)
        else:json.dump(value,f,indent=2,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_code():
    code=json.loads((ROOT/'DEPLOYED_CODE_SHA256.json').read_text())
    for name,digest in code.items():assert sha(ROOT/name)==digest,('code changed',name)
    protocol=json.loads((ROOT/'ADAPTER_PROTOCOL.json').read_text())
    assert sha(ROOT/'SOURCE_MANIFEST.json')==protocol['original_source_manifest_sha256']
    for name,digest in protocol['adapter_sha256'].items():assert sha(ROOT/name)==digest,('adapter changed',name)
    return sha(ROOT/'DEPLOYED_CODE_SHA256.json')

def scene_ids():
    train=(BASE/'manifests/refiner_train.txt').read_text().splitlines()
    tune=(BASE/'manifests/refiner_tune.txt').read_text().splitlines()
    assert len(train)==len(set(train))==8192 and len(tune)==len(set(tune))==1024
    assert not set(train)&set(tune)
    used=set(train)|set(tune)
    candidates=[Path(p).stem for p in (BASE/'manifests/train_files.txt').read_text().splitlines()]
    dev=sorted((p for p in candidates if p not in used),
               key=lambda p:hashlib.sha256(('external_baseline_dev_v1:'+p).encode()).hexdigest())[:1024]
    assert len(dev)==1024 and not used&set(dev)
    return dict(train=sorted(train),tune=sorted(tune),dev=sorted(dev))

def context(raw,row):
    assert raw['scenario_id']==row['scene']
    a=raw['agent'];ids=torch.as_tensor(row['context_ids']).long()
    assert [a['id'][int(i)] for i in ids]==row['original_agent_id']
    assert torch.allclose(a['position'][ids,49,:2],row['current'],atol=1e-4)
    assert (a['valid_mask'][ids,49] & (a['category'][ids]>=2)).all()
    origin=row['current'][0].float().clone()
    world=row['world'].float();heading=row['heading'].float()
    c,s=heading.cos(),heading.sin()
    rotation=torch.stack((torch.stack((c,s),-1),torch.stack((-s,c),-1)),-2)
    delta=world-row['current'][:,None,None]
    local=torch.einsum('nktc,ndc->nktd',delta,rotation)
    assert torch.allclose(torch.einsum('nktc,ncd->nktd',local,rotation)+row['current'][:,None,None],world,atol=2e-4)
    order=torch.argsort(row['pi'],dim=1,descending=True,stable=True)
    assert torch.equal(order[:,0],row['pi'].argmax(-1))
    local=torch.gather(local,1,order[:,:,None,None].expand(-1,-1,60,2))
    embed=torch.gather(row['refine'].float(),1,order[:,:,None].expand(-1,-1,128))
    logits=torch.gather(row['pi'].float(),1,order)
    p=raw['map_point'];polygon=raw['map_polygon']
    edge=raw[('map_point','to','map_polygon')]['edge_index'].long()
    polygon_of=torch.full((p['num_nodes'],),-1,dtype=torch.long)
    polygon_of[edge[0]]=edge[1];assert (polygon_of>=0).all()
    # QCNet enumerates point sides [LEFT,RIGHT,CENTER] and intersections
    # [True,False,None]. Only static observed map fields are used here.
    center=(p['side']==2);center_indices=torch.where(center)[0]
    assert len(center_indices)>0,'No observed map centerlines; stop rather than inventing a lane'
    lane_pos=p['position'][center,:2].float()-origin
    theta=p['orientation'][center].float();mag=p['magnitude'][center].float()
    lane_vec=torch.stack((theta.cos(),theta.sin()),-1)*mag[:,None]
    poly=polygon_of[center]
    intersection=(polygon['is_intersection'][poly]==0).long()
    width=torch.zeros(int(polygon['num_nodes']))
    for pi in torch.unique(poly).tolist():
        cc=p['position'][(polygon_of==pi)&center,:2].float()
        for side in (0,1):
            bb=p['position'][(polygon_of==pi)&(p['side']==side),:2].float()
            if len(bb):width[pi]+=torch.cdist(cc,bb).min(-1).values.median()
    attrs=torch.stack((polygon['type'][poly].float(),width[poly],intersection.float()),-1)
    # Store predicted/history/static inputs separately from future labels.
    pred=dict(local=local,embed=embed,logits=logits,current=row['current'].float()-origin,
              heading=heading,rotation=rotation,lane_pos=lane_pos,lane_vec=lane_vec,
              lane_intersection=intersection,lane_attr=attrs,
              historical_valid=a['valid_mask'][ids,:50].clone(),order=order,
              origin=origin,is_focal=row['is_focal'].clone())
    labels=dict(gt_local=torch.einsum('ntc,ndc->ntd',row['gt'].float()-row['current'][:,None],rotation),
                gt_world=row['gt'].float(),valid=row['valid'].bool())
    return dict(scene=row['scene'],agent_ids=row['original_agent_id'],prediction=pred,labels=labels)

def prepare(limit=None):
    torch.set_num_threads(2)
    splits=scene_ids();where={sid:sp for sp,ids in splits.items() for sid in ids}
    for sp,ids in splits.items():
        save(ROOT/'manifests'/(sp+'.json'),ids)
    wanted=set(where);done=set();start=time.time();seen=set()
    for split in ('train','tune'):
        for path in sorted((BASE/('prepared/i1/'+split)).glob('*.pt')):
            # Reuse frozen rows without rerunning NDPNet.
            for item in torch.load(path,map_location='cpu'):
                row=item['row'];sid=row['scene'];assert sid in wanted
                assert sid not in seen;seen.add(sid)
                dest=ROOT/'prepared'/where[sid]/(sid+'.pt')
                if dest.exists():
                    z=torch.load(dest,map_location='cpu');assert z['scene']==sid
                else:
                    with (RAW/(sid+'.pkl')).open('rb') as f:raw=pickle.load(f)
                    z=context(raw,row);save(dest,z,True)
                done.add(sid)
                if len(done)%128==0:print('PREPARE',len(done),'/',len(wanted),'seconds',round(time.time()-start,1),flush=True)
                if limit and len(done)>=limit:return
    # Development IDs are chosen from native Training, never Validation/audit.
    missing=set(splits['dev'])-done
    if missing:
        # Map fixed scene IDs to existing64-scene cache shards; no full cache scan.
        names=(BASE/'manifests/train_files.txt').read_text().splitlines()
        groups={}
        for idx,name in enumerate(names):
            sid=Path(name).stem
            if sid in missing:groups.setdefault(idx//64,set()).add(sid)
        for shard,sids in sorted(groups.items()):
            for row in torch.load(BASE/'cache/train'/('%05d.pt'%shard),map_location='cpu')['rows']:
                sid=row['scene']
                if sid not in sids:continue
                dest=ROOT/'prepared/dev'/(sid+'.pt')
                if not dest.exists():
                    with (RAW/(sid+'.pkl')).open('rb') as f:raw=pickle.load(f)
                    result=context(raw,row);result['reference_row']=row
                    save(dest,result,True)
                else:assert torch.load(dest,map_location='cpu')['scene']==sid
                done.add(sid)
    assert done==wanted,(len(done),len(wanted))
    save(ROOT/'DATA_READY.json',dict(passed=True,counts={s:len(v) for s,v in splits.items()},
         scene_intersections=0,seconds=time.time()-start,
         manifests={s:sha(ROOT/'manifests'/(s+'.json')) for s in splits},
         interpretation='Native AV2 Training development comparison; frozen backbone previously trained on native Training; no independent-test claim'))
    print('DATA_READY',len(done),'/',len(wanted),flush=True)

def load_rows(split):
    ids=json.loads((ROOT/'manifests'/(split+'.json')).read_text())
    return [torch.load(ROOT/'prepared'/split/(sid+'.pt'),map_location='cpu') for sid in ids]

def batch(rows,method,device='cuda'):
    predictions=[r['prediction'] for r in rows]
    nlist=[len(p['current']) for p in predictions];n=sum(nlist)
    local=torch.cat([p['local'] for p in predictions]).to(device)
    embed=torch.cat([p['embed'] for p in predictions]).to(device)
    rotation=torch.cat([p['rotation'] for p in predictions]).to(device)
    current=torch.cat([p['current'] for p in predictions]).to(device)
    labels=dict(gt=torch.cat([r['labels']['gt_local'] for r in rows]).to(device),
                valid=torch.cat([r['labels']['valid'] for r in rows]).to(device))
    counts=[len(p['lane_pos']) for p in predictions]
    if method=='SmartRefine':
        # Each requested target is an independent prediction with its own map.
        maps=[p for p in predictions for _ in range(len(p['current']))]
        positions=torch.zeros((n,50,2),device=device);positions[:,49]=current
        d=Data(positions=positions,agent_index=torch.arange(n,device=device),
            rotate_local=rotation,rotate_mat=rotation,rotate_angles=torch.zeros(n,device=device),
            tar_lane_positions=torch.cat([p['lane_pos'] for p in maps]).to(device),
            tar_lane_vectors=torch.cat([p['lane_vec'] for p in maps]).to(device),
            tar_is_intersections=torch.cat([p['lane_intersection'] for p in maps]).to(device),
            tar_turn_directions=torch.zeros(sum(len(p['lane_pos']) for p in maps),dtype=torch.long,device=device),
            tar_traffic_controls=torch.zeros(sum(len(p['lane_pos']) for p in maps),dtype=torch.long,device=device))
        d.tar_lane_points_num=[len(p['lane_pos']) for p in maps]
        args=(d,local.transpose(0,1).reshape(6*n,60,2),embed.transpose(0,1))
    else:
        history=torch.cat([p['historical_valid'] for p in predictions]).to(device)
        padding=torch.cat((~history,torch.ones((n,60),device=device,dtype=torch.bool)),-1)
        d=Data(x_centers=current,x_heading=torch.cat([p['heading'] for p in predictions]).to(device),
            rotate_agent2av=rotation,rotate_av2agent=rotation.transpose(1,2),
            tar_lane_positions=torch.cat([p['lane_pos'] for p in predictions]).to(device),
            tar_lane_vectors=torch.cat([p['lane_vec'] for p in predictions]).to(device),
            tar_lane_attr=torch.cat([p['lane_attr'] for p in predictions]).to(device),
            x_padding_mask=padding)
        d.tar_lane_points_num=counts;d.num_vehicles=nlist
        # Eligibility is entirely observed: every cached requested target.
        args=(local,embed,torch.ones(n,device=device,dtype=torch.bool),d)
    return args,labels,dict(nlist=nlist,local=local,current=current,rotation=rotation,
            logits=torch.cat([p['logits'] for p in predictions]).to(device))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);args=ap.parse_args();prepare(args.limit)
