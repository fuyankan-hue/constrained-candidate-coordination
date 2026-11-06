"""Frozen AV2 backbone interface, future-input audit, and resumable feature shards."""
import argparse,hashlib,json,os,pickle,sys,time
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import HeteroData

ROOT=Path('/workspace/output')
sys.path.insert(0,str(ROOT/'source/repo'))
from predictors.NDPNet import NDPNet
from transforms.av2_qcnet_adapter import AV2QCNetAdapter
from transforms.target_builder import TargetBuilder


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));os.replace(tmp,path)


class Extractor:
    def __init__(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        self.path=ROOT/'checkpoints/backbone_frozen.ckpt'
        self.checkpoint_sha=sha(self.path)
        c=torch.load(self.path,map_location='cpu')
        assert c['epoch']==4 and c['global_step']==124945
        self.model=NDPNet(**c['hyper_parameters']).cuda().eval()
        self.model.load_state_dict(c['state_dict'],strict=True)
        self.model.requires_grad_(False)
        self.adapter,self.builder=AV2QCNetAdapter(50,60),TargetBuilder(50,60)
        self.hidden={}
        for key,name in [('near','m2m_propose_attn_layer_proximal'),('far','m2m_propose_attn_layer_lat'),('refine','m2m_refine_attn_layer')]:
            def hook(module,inputs,output,key=key):self.hidden[key]=output.detach()
            getattr(self.model.decoder,name).register_forward_hook(hook)

    def load(self,split,name):
        with open('/workspace/QCNet/data/av2_qcnet/'+split+'/processed/'+name,'rb') as f:raw=pickle.load(f)
        assert raw['scenario_id']==Path(name).stem
        data=self.builder(self.adapter(HeteroData(raw)))
        a=data['agent'];assert len(set(a['id']))==a['num_nodes']
        # Official scored/focal categories and current observations specify targets.
        wanted=(a['category']>=2)&a['valid_mask'][:,49]
        ids=torch.where(wanted)[0]
        assert len(ids)>0
        target={
            'gt':a['position'][ids,50:,:2].float().clone(),
            'valid':a['valid_mask'][ids,50:].bool().clone(),
        }
        meta={'scene':str(raw['scenario_id']),'original_agent_id':[a['id'][int(i)] for i in ids],
              'context_ids':ids.clone(),'current':a['position'][ids,49,:2].float().clone(),
              'heading':a['heading'][ids,49].float().clone(),
              'history_position':a['position'][ids,39:50,:2].float().clone(),
              'history_velocity':a['velocity'][ids,39:50,:2].float().clone(),
              'history_heading':a['heading'][ids,39:50].float().clone(),
              'history_valid':a['valid_mask'][ids,39:50].bool().clone(),
              'observed_type':a['type'][ids].clone(),'is_focal':(a['category'][ids]==3).clone(),
              'sizes_available':False,'target_definition':'category>=2 and observed at step49; no future validity selection',
              'source_backbone_sha256':self.checkpoint_sha,'future_steps':60}
        a['target_mask']=wanted;a['interact_mask']=torch.zeros_like(wanted)
        a['predict_mask'][:,50:]=wanted[:,None]
        # All unavailable future fields are placeholders before forward inference.
        for key in ['position','velocity','heading','abs_vel']:a[key][:,50:]=0
        a['valid_mask'][:,50:]=False;a['target'].zero_()
        for key in ['agent','map_polygon','map_point']:
            data[key]['ptr']=torch.tensor([0,int(data[key]['num_nodes'])])
        return data.cuda(),ids.cuda(),meta,target

    @torch.inference_mode()
    def forward(self,data,ids):
        self.hidden.clear();p=self.model(data)
        n=int(data['agent']['num_nodes'])
        assert p['loc_refine_pos'].shape==(n,6,60,2)
        assert p['pi'].shape==(n,6)
        assert set(self.hidden)=={'near','far','refine'}
        h={k:v.reshape(n,6,128)[ids].cpu().float().clone() for k,v in self.hidden.items()}
        h['local']=p['loc_refine_pos'][ids].cpu().float().clone()
        h['pi']=p['pi'][ids].cpu().float().clone()
        assert all(torch.isfinite(v).all() for v in h.values())
        return h

    def row(self,split,name):
        data,ids,meta,target=self.load(split,name)
        h=self.forward(data,ids);theta=meta['heading'][:,None,None]
        xy=h.pop('local');c,s=theta.cos(),theta.sin()
        h['world']=torch.stack([xy[...,0]*c-xy[...,1]*s,xy[...,0]*s+xy[...,1]*c],-1)+meta['current'][:,None,None,:]
        row={**meta,**h,**target}
        assert row['world'].shape==(len(ids),6,60,2)
        return row


def validate(path,names,backbone_sha):
    try:
        obj=torch.load(path,map_location='cpu')
        assert obj['filenames']==names and obj['backbone_sha256']==backbone_sha
        assert len(obj['rows'])==len(names)
        for r,nm in zip(obj['rows'],names):
            n=len(r['original_agent_id']);assert n>0 and len(set(r['original_agent_id']))==n
            assert r['scene']==Path(nm).stem and r['world'].shape==(n,6,60,2)
            assert all(r[k].shape==(n,6,128) for k in ['near','far','refine'])
            assert r['gt'].shape==(n,60,2) and r['valid'].shape==(n,60)
            assert all(torch.isfinite(r[k]).all() for k in ['world','pi','near','far','refine','gt'])
        return True
    except (Exception,EOFError):return False


def probe(e):
    start=time.time();files=(ROOT/'manifests/train_files.txt').read_text().splitlines()
    selected=sorted(files,key=lambda n:hashlib.sha256(Path(n).stem.encode()).hexdigest())[:64]
    data,ids,meta,target=e.load('train',selected[0]);a=e.forward(data,ids)
    repeatability={k:0. for k in a}
    for _ in range(4):
        repeated=e.forward(data,ids)
        for k in a:repeatability[k]=max(repeatability[k],float((a[k]-repeated[k]).abs().max()))
    changed=data.clone()
    for k in ['position','velocity','heading','abs_vel']:changed['agent'][k][:,50:]=torch.randn_like(changed['agent'][k][:,50:])*100
    changed['agent']['target']=torch.randn_like(changed['agent']['target'])*100
    changed['agent']['valid_mask'][:,50:]=torch.rand_like(changed['agent']['valid_mask'][:,50:].float())>.5
    want=changed['agent']['target_mask'];changed['agent']['predict_mask'][:,50:]=False
    changed['agent']['predict_mask'][want,50]=True
    b=e.forward(changed,ids)
    differences={k:float((a[k]-b[k]).abs().max()) for k in a}
    # Scatter/reduction CUDA kernels have small same-input round-off jitter.
    # Compare future perturbations against that measured numerical control.
    assert all(differences[k]<=max(5e-6,4*repeatability[k]) for k in a),(differences,repeatability)
    rows=[];times=[];torch.cuda.reset_peak_memory_stats()
    for name in selected:
        torch.cuda.synchronize();t=time.perf_counter();r=e.row('train',name);torch.cuda.synchronize()
        times.append(time.perf_counter()-t);rows.append(r)
    metrics={};summed={k:0. for k in ['targets','focal_targets','ade_sum','fde_sum','minfde_sum','focal_fde_sum']}
    for r in rows:
        m=r['pi'].argmax(-1);w=r['world'][torch.arange(len(m)),m]
        err=(w-r['gt']).norm(dim=-1);ok=r['valid'][:,-1]
        ade=(err*r['valid']).sum(-1)/r['valid'].sum(-1).clamp_min(1)
        mine=(r['world'][:,:,-1]-r['gt'][:,None,-1]).norm(dim=-1).min(-1).values
        focal=ok&r['is_focal']
        for k,v in [('targets',ok.sum()),('focal_targets',focal.sum()),('ade_sum',ade[ok].sum()),('fde_sum',err[ok,-1].sum()),('minfde_sum',mine[ok].sum()),('focal_fde_sum',err[focal,-1].sum())]:summed[k]+=float(v)
    metrics={'valid6_targets':int(summed['targets']),'focal_targets':int(summed['focal_targets']),
             'top1_ADE6':summed['ade_sum']/max(summed['targets'],1),'top1_FDE6':summed['fde_sum']/max(summed['targets'],1),
             'oracle_minFDE6_diagnostic':summed['minfde_sum']/max(summed['targets'],1),
             'focal_top1_FDE6':summed['focal_fde_sum']/max(summed['focal_targets'],1)}
    torch.save({'rows':rows,'filenames':selected,'backbone_sha256':e.checkpoint_sha},ROOT/'probe_features.pt')
    result={'passed':True,'backbone_frozen':True,'epoch':4,'global_step':124945,'backbone_sha256':e.checkpoint_sha,
            'smoke_scene_ids':[Path(n).stem for n in selected],'scenes':64,
            'multiagent_scenes':sum(len(r['original_agent_id'])>1 for r in rows),
            'prediction_targets':sum(len(r['original_agent_id']) for r in rows),
            'future_label_perturbation_max_difference':differences,'metrics_development_probe_only':metrics,
            'same_input_repeatability_max_difference':repeatability,
            'seconds_per_scene_mean':float(np.mean(times[4:])),
            'estimated_all_features_four_gpu_hours':float(np.mean(times[4:]))*(199908+24988)/4/3600,
            'gpu_peak_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.time()-start,
            'no_backbone_training':True,'sizes_note':'AV2 motion caches have no measured length/width; downstream geometry needs an explicit AV2-native definition.'}
    assert sha(e.path)==e.checkpoint_sha
    save_json(ROOT/'FROZEN_BACKBONE_SMOKE_PASS.json',result)
    print('AV2_FROZEN_MULTIAGENT_SMOKE_PASS',json.dumps(result),flush=True)


def extract(e,worker,workers):
    assert json.loads((ROOT/'FROZEN_BACKBONE_SMOKE_PASS.json').read_text())['passed']
    lists={s:(ROOT/('manifests/'+s+'_files.txt')).read_text().splitlines() for s in ['train','val']}
    expected=sum(min(64,len(files)-offset) for files in lists.values()
                 for shard,offset in enumerate(range(0,len(files),64)) if shard%workers==worker)
    done=0;start=time.time()
    for split in ['train','val']:
        files=lists[split]
        for shard,offset in enumerate(range(0,len(files),64)):
            if shard%workers!=worker:continue
            names=files[offset:offset+64];dest=ROOT/'cache'/split/('%05d.pt'%shard)
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not (dest.exists() and validate(dest,names,e.checkpoint_sha)):
                rows=[e.row(split,n) for n in names]
                tmp=dest.with_name(dest.name+'.tmp.'+str(os.getpid()))
                with tmp.open('wb') as f:
                    torch.save({'rows':rows,'filenames':names,'backbone_sha256':e.checkpoint_sha},f);f.flush();os.fsync(f.fileno())
                os.replace(tmp,dest)
                assert validate(dest,names,e.checkpoint_sha)
            done+=len(names)
            z={'worker':worker,'workers':workers,'split':split,'shard':shard,'done':done,'total':expected,
               'elapsed_seconds':time.time()-start,'backbone_sha256':e.checkpoint_sha}
            save_json(ROOT/'worker_state'/('w%d.json'%worker),z)
            if shard%40==worker:print('AV2_FEATURE_PROGRESS',json.dumps(z),flush=True)
    save_json(ROOT/'worker_state'/('w%d_COMPLETE.json'%worker),{'worker':worker,'passed':True,'scenes':done,'seconds':time.time()-start})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['probe','extract']);ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    e=Extractor()
    if args.mode=='probe':probe(e)
    else:extract(e,args.worker,args.workers)
