"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
from pathlib import Path
import argparse,json,os,random,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
import torch.nn.functional as F
from ccc.core.motion_model import MotionNFI
from ccc.io import save_json as save, save_torch as atomic_torch, sha
RUN=Path(os.environ.get('CCC_WOMD_WORK','work/womd'))


class Model(MotionNFI):
    def __init__(self,arm):super().__init__(rich=arm!='padded');self.ablation=arm
    def forward(self,near,far,refine,world,edges):
        if self.ablation in ('no_near','geometry_only','refine_only'):near=torch.zeros_like(near)
        if self.ablation in ('no_far','geometry_only','refine_only'):far=torch.zeros_like(far)
        if self.ablation in ('no_refine','geometry_only'):refine=torch.zeros_like(refine)
        return super().forward(near,far,refine,world,edges)


def prediction(m,r):return m(*[r[k].cuda() for k in ('near','far','refine','world','edges')])


def loss_for(out,r):
    q=F.softmax(-r['rel'].cuda().flatten(1)/3.,-1);return -(q*F.log_softmax(out.flatten(1),-1)).sum(-1)


def read_data():
    parts={'fit':[],'calibration':[],'audit':[]};protocol=sha(RUN/'experiment_protocol.json')
    for p in sorted((RUN/'nfi_data').glob('*.pt')):
        value=torch.load(str(p),map_location='cpu');assert value['protocol_sha256']==protocol
        for k in parts:parts[k].extend(value['data'][k])
    assert parts['fit'] and parts['calibration'] and parts['audit'];assert not {r['scene'] for r in parts['fit']}&{r['scene'] for r in parts['calibration']+parts['audit']}
    return parts


@torch.no_grad()
def evaluate(m,rows):
    m.eval();num=den=indep=base=entropy=0.
    for r in rows:
        output=prediction(m,r).flatten(1);q=F.softmax(-r['rel'].cuda().flatten(1)/3.,-1);p=output.softmax(-1).reshape(-1,6,6)
        product=(p.sum(-1)[:,:,None]*p.sum(-2)[:,None,:]).reshape(-1,36)
        unary=(r['pi'][r['edges'][:,0]][:,:,None]+r['pi'][r['edges'][:,1]][:,None,:]).cuda().flatten(1)
        num+=float((-(q*output.log_softmax(-1)).sum(-1)).sum());indep+=float((-(q*product.clamp_min(1e-12).log()).sum(-1)).sum());base+=float((-(q*unary.log_softmax(-1)).sum(-1)).sum());entropy+=float((-(q*q.clamp_min(1e-12).log()).sum(-1)).sum());den+=len(output)
    return dict(CE=num/den,independent_marginal_CE=indep/den,conditional_gain=(indep-num)/den,original_unary_CE=base/den,target_entropy=entropy/den,pairs=int(den))


def main(arm,seed,smoke=False):
    root=Path(os.environ['CCC_MODEL_OUTPUT']);root.mkdir(exist_ok=True);name=arm+str(seed);protocol=sha(RUN/'experiment_protocol.json')
    if (root/(name+'_complete.json')).exists():assert json.loads((root/(name+'_complete.json')).read_text())['protocol_sha256']==protocol;return
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    data=read_data();m=Model(arm).cuda();opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4);initial={k:v.cpu().clone() for k,v in m.state_dict().items()}
    out=prediction(m,data['fit'][0]);loss=loss_for(out,data['fit'][0]).mean();assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());opt.zero_grad(set_to_none=True)
    if smoke:save({'pass':True,'model_params':sum(p.numel() for p in m.parameters()),'fit_scenes':len(data['fit']),'calibration_scenes':len(data['calibration']),'loss':float(loss)},Path(__file__).resolve().parent/'nfi_backward_smoke.json');return
    order=list(range(len(data['fit'])));best=float('inf');bad=0;history=[];first=1;last=root/(name+'_resume.pt')
    if last.exists():
        r=torch.load(str(last),map_location='cpu');assert r['protocol_sha256']==protocol;m.load_state_dict(r['state_dict']);opt.load_state_dict(r['optimizer']);order=r['order'];best=r['best'];bad=r['bad'];history=r['history'];first=r['epoch']+1;random.setstate(r['random']);torch.set_rng_state(r['torch_rng']);torch.cuda.set_rng_state_all(r['cuda_rng'])
    start=time.monotonic()
    for epoch in range(first,first if bad>=3 else 9):
        random.shuffle(order);m.train();total=pairs=0.
        for pos,index in enumerate(order,1):
            r=data['fit'][index];opt.zero_grad(set_to_none=True);out=prediction(m,r);loss=loss_for(out,r).mean();assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5.);assert torch.isfinite(norm);opt.step();total+=float(loss)*len(out);pairs+=len(out)
            if pos%10000==0:print('FULL_NFI_TRAIN',name,epoch,pos,'/',len(order),'elapsed',round(time.monotonic()-start,1),flush=True)
        cm=evaluate(m,data['calibration']);history.append({'epoch':epoch,'train_CE':total/pairs,'calibration':cm,'elapsed':time.monotonic()-start});print('FULL_NFI_EPOCH',name,history[-1],flush=True)
        if cm['CE']<best-1e-4:
            best=cm['CE'];bad=0;atomic_torch({'state_dict':{k:v.cpu().clone() for k,v in m.state_dict().items()},'arm':'padded' if arm=='padded' else 'rich','ablation':arm,'seed':seed,'epoch':epoch,'protocol_sha256':protocol,'calibration':cm},root/(name+'_best.pt'))
        else:bad+=1
        save(history,root/(name+'_history.json'));atomic_torch({'state_dict':{k:v.cpu().clone() for k,v in m.state_dict().items()},'optimizer':opt.state_dict(),'epoch':epoch,'order':order,'best':best,'bad':bad,'history':history,'random':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'protocol_sha256':protocol},last)
        if bad>=3:break
    ck=torch.load(str(root/(name+'_best.pt')),map_location='cpu');save({'pass':True,'checkpoint_sha256':sha(root/(name+'_best.pt')),'best_epoch':ck['epoch'],'audit_deferred_until_all_NFI_models_frozen':True,'protocol_sha256':protocol},root/(name+'_complete.json'))

