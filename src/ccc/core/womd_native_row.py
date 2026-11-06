"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import numpy as np
import torch
from .refiner_womd import collate
from .womd_protection import pairs_for,safeguard
from .womd_bank import make
from . import womd_i2_loss as N

@torch.no_grad()
def native_row(v, bank, model):
    meta = bank['evaluation']; modes, amodes = meta['modes'][1], meta['modes'][0]
    n = len(modes); pairs = pairs_for(meta)
    base = meta['base_world'][1].double().numpy(); aa = meta['base_world'][0].double().numpy()
    x, _ = collate([bank], [1], 'cuda')
    delta = model(x)[0].double().cpu().numpy()
    dw = np.einsum('ntc,ndc->ntd', delta, bank['inputs']['rotation'].double().numpy())
    first, beta = safeguard(base, base+dw, meta, pairs)
    w = v['m1']['world'].double().clone(); w[torch.arange(n), modes] = torch.as_tensor(first)
    vv = dict(v, m1=dict(v['m1'], world=w)); after = make(vv, N.Choice(modes))
    row = dict(scene=v['scene'], bank_after=after, base=base, original_A=aa, firstsafe=first,
               meta=meta, pairs=pairs, radius=torch.tensor(np.array(modes)==np.array(amodes), dtype=torch.float32),
               modes=modes, A_modes=amodes, firstP_beta=beta,
               fdeA=np.linalg.norm(aa-meta['gt_world'].double().numpy(), axis=-1)[:, -1])
    row['baseline'] = N.stats(row, first); row['A_metrics'] = N.stats(row, aa)
    return row

