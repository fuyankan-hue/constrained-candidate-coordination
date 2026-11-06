"""Prediction-side adapters around the frozen native functions, not a new solver."""
from pathlib import Path
import copy
import numpy as np
import torch
from .core import solvers,womd_selection,av2_ops
from .core.rcim_womd import Model as WOMDRCIM
from .core.rcim_av2 import RCIM as AV2RCIM
from .core.refiner_womd import Refiner as WOMDRefiner,collate as womd_collate
from .core.refiner_av2 import Refiner as AV2Refiner,collate as av2_collate
from .core.womd_bank import make
from .core.womd_geometry import tables
from .core.womd_protection import pairs_for,safeguard
from .core.womd_i2_loss import Choice
from .core.admissible import admissible

def load_model(path,kind,dataset,device='cpu'):
    if kind=='rcim':model=WOMDRCIM('refine_only') if dataset=='womd' else AV2RCIM()
    else:model=(WOMDRefiner if dataset=='womd' else AV2Refiner)(False)
    pack=torch.load(str(path),map_location='cpu');state=pack['state_dict']
    # WOMD's original Corrector owns the same frozen Refiner under "network".
    if kind=='refiner' and state and all(k.startswith('network.') for k in state):
        state={k[len('network.'):]:v for k,v in state.items()}
    model.load_state_dict(state,strict=True);model.to(device).eval();model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters())==(44449 if kind=='rcim' else 76026)
    return model

def prediction_observation(row):
    """I/O allowlist: future GT, validity and errors are not solver inputs."""
    a={k:row['m1'][k].detach().cpu().clone() for k in ('world','pi','original_agent_id')}
    b={k:row['m2'][k].detach().cpu().clone() for k in ('near','far','refine','world','current','target_ids','context_ids')}
    h={k:row['history'][k].detach().cpu().clone() for k in ('position','velocity','heading','valid_mask','length','width','type')}
    assert torch.equal(a['original_agent_id'],b['target_ids']) and torch.equal(b['target_ids'],b['context_ids'])
    return dict(scene=str(row['scene']),m1=a,m2=b,history=h,
                track_ids=copy.deepcopy(row['track_ids']),checkpoint_sha256=row['checkpoint_sha256'])

@torch.no_grad()
def coordinate(row,scorer,dataset):
    if dataset=='av2':return av2_ops.choose(row,scorer)
    solvers.project.__globals__['tables']=tables
    womd_selection.I.observations=prediction_observation
    chosen=womd_selection.selection(row,scorer)
    obs=prediction_observation(row);a,b=obs['m1'],obs['m2'];n=len(a['pi']);ii=np.arange(n)
    edges=[(i,j) for i in range(n) for j in range(i+1,n) if float((b['current'][i]-b['current'][j]).norm())<75]
    c=scorer(*[b[k] for k in ('near','far','refine','world')],torch.tensor(edges,dtype=torch.long).reshape(-1,2))
    hard,_=solvers.geometry_arrays(tables(obs),edges,n)
    em,ed=solvers.solve(a['pi'].double().numpy(),c.double().numpy(),edges,
                       admissible(a['pi'],a['world'],20.).numpy(),chosen['original_A'],hard=hard,max_sweeps=8)
    out={'A':{'modes':chosen['original_A'],'path':a['world'][ii,chosen['original_A']].double().numpy()},
         'E':{'modes':em,'path':a['world'][ii,em].double().numpy(),'search':ed},'alpha':chosen['alpha'],
         'projected_world':chosen['projected_world']}
    for native,public in [('RCIM','RCIM'),('NoRCIM','NoRCIM'),('B2','B')]:
        modes=chosen['choices'][native];world=chosen['projected_world'] if native!='B2' else a['world']
        out[public]={'modes':modes,'path':world[ii,modes].double().numpy(),'search':chosen['search'][native]}
    return out

def _bank(row,path,modes):
    """Use the copied native bank builder; labels are blank in prediction calls."""
    v=prediction_observation(row);n=len(modes)
    v['m1']['world']=v['m1']['world'].double().clone()
    v['m1']['world'][torch.arange(n),modes]=torch.as_tensor(path,dtype=torch.double)
    v['m1']['gt']=torch.zeros(n,80,2);v['m1']['valid']=torch.zeros(n,80,dtype=torch.bool)
    return make(v,Choice(modes))

@torch.no_grad()
def refine_once(row,path,modes,model,dataset,device='cpu',radius=None):
    if dataset=='av2':
        # The copied builder separates targets from model inputs. Future labels
        # are replaced before that builder is invoked during prediction.
        prediction=dict(row,gt=torch.zeros_like(row['world'][:,0]),
                        valid=torch.zeros(row['world'].shape[0],60,dtype=torch.bool))
        return av2_ops.refine(prediction,path,modes,model,av2_collate,device,radius)
    bank=_bank(row,path,modes);x,_=womd_collate([bank],[1],device)
    delta=model(x)[0,:len(path)]
    if radius is not None:delta=delta*torch.as_tensor(radius,device=device)[:,None,None]
    dw=np.einsum('ntc,ndc->ntd',delta.double().cpu().numpy(),bank['inputs']['rotation'].double().numpy())
    assert np.linalg.norm(dw,axis=-1).max()<=1.00001
    raw=path+dw;safe,beta=safeguard(path,raw,bank['evaluation'],pairs_for(bank['evaluation']))
    return safe,beta,raw

@torch.no_grad()
def predict(row,scorer,first,second,dataset,method='Full',device='cpu',choices=None):
    choices=coordinate(row,scorer,dataset) if choices is None else choices
    branch={'Full':'RCIM','Full_coord':'RCIM','Full_I1':'RCIM','Full_noP2':'RCIM',
            'Full_fixed_OFF':'NoRCIM','NoRCIM':'NoRCIM','B2':'B',
            'E_I1':'E','A_I1':'A','A':'A','E':'E','B':'B'}[method]
    selected=choices[branch];path=selected['path'];modes=selected['modes'];anchor=choices['A']['path']
    result=dict(scene=str(row['scene']),modes=modes,anchor=anchor,coordinated=path,
                search=selected.get('search',{}))
    if method in ('A','E','B','Full_coord'):
        result['path']=path;return result
    first_path,beta1,raw1=refine_once(row,path,modes,first,dataset,device)
    result.update(first_path=first_path,P1_beta=beta1)
    if method in ('A_I1','E_I1','Full_I1'):
        result['path']=first_path;return result
    if method=='B2':radius=(np.asarray(modes)==np.asarray(choices['A']['modes'])).astype(np.float32)
    else:radius=np.clip(5.-1e-4-np.linalg.norm(first_path-anchor,axis=-1).max(1),0.,1.).astype(np.float32)
    final,beta2,raw2=refine_once(row,first_path,modes,second,dataset,device,radius)
    if method!='B2':assert np.linalg.norm(final-anchor,axis=-1).max()<=5.00001
    result.update(path=raw2 if method=='Full_noP2' else final,P2_beta=beta2,radius=radius)
    return result
