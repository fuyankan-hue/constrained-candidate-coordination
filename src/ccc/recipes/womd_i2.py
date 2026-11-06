"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import os,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ccc.core.refiner_womd import Refiner,collate
from ccc.core.womd_protection import safeguard,geometry
from ccc.core import womd_i2_loss as N
from ccc.core import aggregate as E
from ccc.io import sha,save_json,save_torch

def save(path,value):return save_json(value,path)
def savept(path,value):return save_torch(value,path)


def setup():
    assert torch.__version__.startswith('1.13.1')
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED); random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


class Corrector(nn.Module):
    def __init__(self, branch):
        super().__init__(); self.network = Refiner(False)
        self.network.load_state_dict(torch.load(str(i1_path(branch)), map_location='cpu')['state_dict'])
        nn.init.zeros_(self.network.head.weight); nn.init.zeros_(self.network.head.bias)

    def forward(self, x, radius):
        return self.network(x)*radius[:, :, None, None]


def check_contract(row, path, branch):
    g, old = geometry(path, row['meta'], row['pairs']), geometry(row['firstsafe'], row['meta'], row['pairs'])
    assert not (g.any(-1) & ~old.any(-1)).any() and g.sum() <= old.sum()
    mt = N.stats(row, path)
    if branch == 'B2':
        fixed = row['radius'].numpy() == 0
        assert np.array_equal(path[fixed], row['firstsafe'][fixed])
        assert mt['gt5'] == row['baseline']['gt5']
        assert np.linalg.norm(path-row['base'], axis=-1).max() <= 2.00002
    else:
        assert mt['gt5'] == 0 and mt['max_shift_from_A'] <= 5.000001
    return mt


@torch.no_grad()
def evaluate(model, rows, branch, protected=False, details=False):
    model.eval(); records=[]
    for at in range(0, len(rows), 16):
        rr = rows[at:at+16]; x, radius, y = N.batch(rr, branch)
        delta = model(x, radius).double().cpu().numpy()
        for j, z in enumerate(rr):
            row = z['variants'][branch]; n=len(row['radius'])
            dw = np.einsum('ntc,ndc->ntd', delta[j, :n], row['bank_after']['inputs']['rotation'].double().numpy())
            raw = row['firstsafe']+dw
            path, beta = safeguard(row['firstsafe'], raw, row['meta'], row['pairs']) if protected else (raw, None)
            mt = check_contract(row, path, branch) if protected else N.stats(row, path)
            rec = dict(scene=z['scene'], methods={'New':mt, 'Base_IP':row['baseline'], 'A':row['A_metrics']}, P2_beta=beta)
            if details:
                rec.update(path=torch.as_tensor(path), raw_metrics=N.stats(row, raw),
                           first_path=torch.as_tensor(row['firstsafe']), base_metrics=N.stats(row, row['base']))
            records.append(rec)
    table = {name:E.aggregate(records, name) for name in ('New', 'Base_IP', 'A')}
    table['New']['score'] = table['New']['ADE']+.25*table['New']['FDE']
    return table, records


def train():
    setup(); cfg=check(); assert json.loads((ROOT/'SMOKE.json').read_text())['pass_']
    rows, tune=load('train'), load('tune')
    assert len(rows)==8192 and len(tune)==1024
    assert not {r['scene'] for r in rows} & {r['scene'] for r in tune}
    for branch in BRANCHES:
        root=ROOT/'models'/branch
        if (root/'complete.json').exists():
            z=json.loads((root/'complete.json').read_text()); assert z['steps']==5120 and sha(root/'best.pt')==z['SHA256']; continue
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        m=Corrector(branch).cuda(); opt=torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=2e-5)
        resume=root/'resume.pt'; first=1; steps=0; started=time.time()
        if resume.exists():
            z=torch.load(str(resume), map_location='cpu'); assert z['source_SHA256']==sha(__file__)
            m.load_state_dict(z['state_dict']); opt.load_state_dict(z['optimizer']); scheduler.load_state_dict(z['scheduler'])
            history=z['history']; best=z['best']; selected=z['selected']; steps=z['steps']; first=z['epoch']+1
            torch.set_rng_state(z['torch_rng']); torch.cuda.set_rng_state(z['cuda_rng'])
        else:
            table,_=evaluate(m,tune,branch); met=table['New']; best=met['score']; selected=0; history=[dict(epoch=0,tune=met)]
            savept(root/'best.pt',dict(state_dict={k:v.cpu().clone() for k,v in m.state_dict().items()},epoch=0))
        for epoch in range(first,11):
            order=np.random.default_rng(SEED+epoch).permutation(len(rows)); m.train(); total=0.; batches=0
            for at in range(0,len(order),16):
                rr=[rows[int(k)] for k in order[at:at+16]]; x,radius,y=N.batch(rr,branch)
                opt.zero_grad(set_to_none=True); loss=N.loss(m,x,radius,y); assert torch.isfinite(loss)
                loss.backward(); grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.); assert torch.isfinite(grad); opt.step()
                steps+=1; batches+=1; total+=float(loss)
                if batches%128==0: state('training',steps,5120,branch=branch,epoch=epoch,batch=batches,batches_per_epoch=512)
            scheduler.step(); table,_=evaluate(m,tune,branch); met=table['New']
            history.append(dict(epoch=epoch,steps=steps,loss=total/batches,tune=met))
            if met['score']<best-1e-4:
                best=met['score']; selected=epoch
                savept(root/'best.pt',dict(state_dict={k:v.cpu().clone() for k,v in m.state_dict().items()},epoch=epoch))
            savept(resume,dict(state_dict={k:v.cpu().clone() for k,v in m.state_dict().items()},optimizer=opt.state_dict(),
                   scheduler=scheduler.state_dict(),history=history,best=best,selected=selected,epoch=epoch,steps=steps,
                   source_SHA256=sha(__file__),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
            save(root/'history.json',history); state('training',steps,5120,branch=branch,epoch=epoch)
            print('TRAIN',SEED,branch,epoch,'/10',steps,'/5120','tune',met['score'],'seconds',round(time.time()-started,1),flush=True)
        assert steps==5120
        save(root/'complete.json',dict(pass_=True,steps=steps,selected_epoch=selected,SHA256=sha(root/'best.pt'),seconds=time.time()-started,seed=SEED,branch=branch))
    state('training_complete',3,3)

