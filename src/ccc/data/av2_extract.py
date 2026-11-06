"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import hashlib,os,pickle
from pathlib import Path
import torch
import numpy as np
from torch_geometric.data import HeteroData
from predictors.NDPNet import NDPNet
from transforms.av2_qcnet_adapter import AV2QCNetAdapter
from transforms.target_builder import TargetBuilder
ROOT=Path(os.environ.get('CCC_AV2_WORK','work/av2'))
PROCESSED=Path(os.environ.get('CCC_AV2_PROCESSED','data/av2/processed'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


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
        with open(str(PROCESSED/split/'processed'/name),'rb') as f:raw=pickle.load(f)
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

