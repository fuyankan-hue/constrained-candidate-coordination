"""Matched AV2 independent refiners; native 60-step version of frozen I1/I2.

Only the trajectory/time-grid adapters differ from the existing WOMD trainers.
I1 retains stage_SM_train SmoothL1, clip5; I2 retains trial9 Euclidean, clip1.
All arms receive 10 epochs / 5,120 updates, with tune-only checkpoint choice.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from av2_refiner import Refiner, collate, objective

ROOT = Path('/workspace/output')
CONFIG = {'train_scenes': 8192, 'tune_scenes': 1024, 'batch_size': 16,
          'max_epochs': 10, 'optimizer': 'AdamW', 'lr': 2e-4,
          'weight_decay': 1e-4, 'eta_min': 2e-5, 'min_delta': 1e-4,
          'I1_clip': 5., 'I2_clip': 1., 'future_steps': 60,
          'selection': 'unprotected tune valid6 ADE+.25FDE; identity epoch0 eligible'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(value, path, pt=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('wb' if pt else 'w') as f:
        if pt:
            torch.save(value, f)
        else:
            json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def weight_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def setup(seed):
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    print('AV2_REFINER_RUNTIME', json.dumps({'torch': torch.__version__,
          'numpy': np.__version__, 'device': torch.cuda.get_device_name(0),
          'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
          'seed': seed, 'pid': os.getpid()}), flush=True)


def load_rows(stage, arm, seed, split):
    directory = ROOT/'prepared'/stage
    if stage == 'i2':
        directory = directory/('seed%d' % seed)
    rows = []
    for path in sorted((directory/split).glob('*.pt')):
        pack = torch.load(path, map_location='cpu')
        records = pack['rows'] if isinstance(pack, dict) else pack
        for value in records:
            bank = value['arms'][arm]
            assert bank['scene'] == value['scene']
            rows.append(bank)
    rows.sort(key=lambda r: r['scene'])
    expected = CONFIG['train_scenes' if split == 'train' else 'tune_scenes']
    ids = [r['scene'] for r in rows]
    assert len(rows) == len(set(ids)) == expected, (stage, arm, split, len(rows))
    manifest = ROOT/'manifests'/('refiner_%s.txt' % split)
    assert set(ids) == set(manifest.read_text().splitlines())
    return rows


def batch(rows, device):
    x, y = collate(rows, [0]*len(rows), device)
    radius = torch.zeros(x['agent_present'].shape, device=device)
    for i, r in enumerate(rows):
        n = len(r['inputs']['current'])
        radius[i, :n] = torch.as_tensor(r.get('radius', np.ones(n)),
                                      dtype=radius.dtype, device=device)
    assert torch.isfinite(radius).all() and (radius >= 0).all() and (radius <= 1).all()
    return x, y, radius


def residual(model, x, radius, stage):
    delta = model(x)
    if stage == 'i2':
        delta = delta*radius[:, :, None, None]
    return delta


def loss_value(model, x, y, radius, stage):
    if stage == 'i1':
        return objective(model, x, y)
    delta = residual(model, x, radius, stage)
    predicted = x['path_local']+delta
    valid = y['valid']
    gt = y['gt']
    position = ((predicted-gt).norm(dim=-1)*valid).sum()/valid.sum().clamp_min(1)
    last = valid[:, :, -1]
    endpoint = ((predicted[:, :, -1]-gt[:, :, -1]).norm(dim=-1)*last).sum()/last.sum().clamp_min(1)
    present = x['agent_present']
    regularity = (delta.square().sum(-1)*present[:, :, None]).sum()/(present.sum().clamp_min(1)*60)
    acceleration = (delta[:, :, 2:]-2*delta[:, :, 1:-1]+delta[:, :, :-2])/.01
    smoothness = (acceleration.square().sum(-1)*present[:, :, None]).sum()/(present.sum().clamp_min(1)*58)
    return position+.25*endpoint+.05*regularity+.002*smoothness, delta


@torch.no_grad()
def development(model, rows, stage, device='cuda'):
    model.eval()
    ade = fde = count = 0
    for at in range(0, len(rows), CONFIG['batch_size']):
        x, y, radius = batch(rows[at:at+CONFIG['batch_size']], device)
        predicted = x['path_local']+residual(model, x, radius, stage)
        error = (predicted-y['gt']).norm(dim=-1)
        valid = y['valid']
        good = valid[:, :, -1]
        average = (error*valid).sum(-1)/valid.sum(-1).clamp_min(1)
        ade += float(average[good].sum())
        fde += float(error[:, :, -1][good].sum())
        count += int(good.sum())
    assert count > 0
    return {'ADE6': ade/count, 'FDE6': fde/count, 'targets': count,
            'score': (ade+.25*fde)/count, 'protection_applied': False}


def train_one(arm, seed, stage):
    stage = stage.lower()
    assert stage in ('i1', 'i2')
    assert arm in (('A', 'E', 'B') if stage == 'i1' else ('RCIM', 'NoRCIM', 'B'))
    out = ROOT/'models/refiners'/('seed%d' % seed)/(stage+'_'+arm)
    if (out/'complete.json').exists():
        result = json.loads((out/'complete.json').read_text())
        assert result['passed'] and sha(out/'best.pt') == result['best_SHA256']
        print('AV2_REFINER_ALREADY_COMPLETE', stage, arm, seed, flush=True)
        return result
    setup(seed)
    rows = load_rows(stage, arm, seed, 'train')
    tune = load_rows(stage, arm, seed, 'tune')
    assert not {r['scene'] for r in rows}&{r['scene'] for r in tune}
    model = Refiner(False).cuda()
    assert sum(p.numel() for p in model.parameters()) == 76026
    parent = None
    if stage == 'i2':
        parent_arm = 'B' if arm == 'B' else 'E'
        parent = ROOT/'models/refiners'/('seed%d' % seed)/('i1_'+parent_arm)/'best.pt'
        model.load_state_dict(torch.load(parent, map_location='cpu')['state_dict'])
        torch.nn.init.zeros_(model.head.weight)
        torch.nn.init.zeros_(model.head.bias)
    initial_hash = weight_hash(model)
    contract = {'config': CONFIG, 'seed': seed, 'stage': stage, 'arm': arm,
                'training_code_SHA256': sha(__file__),
                'model_code_SHA256': sha(Path(__file__).with_name('av2_refiner.py')),
                'train_scene_SHA256': sha(ROOT/'manifests/refiner_train.txt'),
                'tune_scene_SHA256': sha(ROOT/'manifests/refiner_tune.txt'),
                'parent_checkpoint_SHA256': sha(parent) if parent else None}
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=CONFIG['eta_min'])
    first = 1
    steps = 0
    selected = 0
    history = []
    resume = out/'resume.pt'
    if resume.exists():
        pack = torch.load(resume, map_location='cpu')
        assert pack['training_signature'] == signature
        model.load_state_dict(pack['state_dict'])
        opt.load_state_dict(pack['optimizer'])
        scheduler.load_state_dict(pack['scheduler'])
        first = pack['epoch']+1
        steps = pack['steps']
        best = pack['best']
        history = pack['history']
        selected = pack['selected']
        assert pack['initial_state_SHA256'] == initial_hash
    else:
        x, _, radius = batch(rows[:4], 'cuda')
        assert torch.equal(residual(model, x, radius, stage), torch.zeros_like(x['path_local']))
        initial = development(model, tune, stage)
        best = initial['score']
        history.append({'epoch': 0, 'tune': initial})
        save({'state_dict': state(model), 'epoch': 0, 'tune': initial,
              'training_signature': signature, 'contract': contract}, out/'best.pt', True)
        save({'state_dict': state(model), 'optimizer': opt.state_dict(),
              'scheduler': scheduler.state_dict(), 'epoch': 0, 'steps': 0,
              'best': best, 'history': history, 'selected': 0,
              'initial_state_SHA256': initial_hash, 'training_signature': signature}, resume, True)
        print('AV2_REFINER_IDENTITY_TUNE', stage, arm, seed, json.dumps(initial), flush=True)
    started = time.monotonic()
    for epoch in range(first, 11):
        order = np.random.default_rng(seed+epoch).permutation(len(rows))
        model.train()
        loss_sum = 0.
        tick = time.monotonic()
        for at in range(0, len(rows), 16):
            chosen = [rows[int(i)] for i in order[at:at+16]]
            x, y, radius = batch(chosen, 'cuda')
            opt.zero_grad(set_to_none=True)
            loss, delta = loss_value(model, x, y, radius, stage)
            assert torch.isfinite(loss) and float(delta.norm(dim=-1).max()) <= 1.00001
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['I1_clip' if stage == 'i1' else 'I2_clip'])
            assert torch.isfinite(norm)
            opt.step()
            steps += 1
            loss_sum += float(loss.detach())
            if (at//16+1) % 64 == 0:
                progress = {'stage': stage, 'arm': arm, 'seed': seed,
                            'epoch': epoch, 'max_epochs': 10,
                            'batches_done': at//16+1, 'batches_total': 512,
                            'scenes_done': at+len(chosen), 'scenes_total': 8192,
                            'steps_done': steps, 'steps_total': 5120,
                            'seconds_this_epoch': time.monotonic()-tick,
                            'mean_loss': loss_sum/(at//16+1)}
                save(progress, out/'progress.json')
                print('AV2_REFINER_PROGRESS', json.dumps(progress), flush=True)
        metric = development(model, tune, stage)
        history.append({'epoch': epoch, 'training_loss': loss_sum/512,
                        'tune': metric, 'steps': steps,
                        'epoch_seconds': time.monotonic()-tick})
        if metric['score'] < best-CONFIG['min_delta']:
            best = metric['score']
            selected = epoch
            save({'state_dict': state(model), 'epoch': epoch, 'tune': metric,
                  'training_signature': signature, 'contract': contract}, out/'best.pt', True)
        scheduler.step()
        save({'state_dict': state(model), 'optimizer': opt.state_dict(),
              'scheduler': scheduler.state_dict(), 'epoch': epoch, 'steps': steps,
              'best': best, 'history': history, 'selected': selected,
              'initial_state_SHA256': initial_hash, 'training_signature': signature}, resume, True)
        print('AV2_REFINER_EPOCH', stage, arm, seed, json.dumps(history[-1]), flush=True)
    assert steps == 5120 and history[-1]['epoch'] == 10
    order_hash = hashlib.sha256()
    for epoch in range(1, 11):
        order_hash.update(np.random.default_rng(seed+epoch).permutation(len(rows)).tobytes())
    best_pack = torch.load(out/'best.pt', map_location='cpu')
    result = {'passed': True, 'stage': stage, 'arm': arm, 'seed': seed,
              'parameters': 76026, 'epochs_completed': 10, 'optimizer_steps': steps,
              'selected_epoch': selected, 'best_tune': best_pack['tune'],
              'initial_state_SHA256': initial_hash, 'scene_order_SHA256': order_hash.hexdigest(),
              'best_SHA256': sha(out/'best.pt'), 'training_signature': signature,
              'seconds_this_session': time.monotonic()-started,
              'GPU_peak_MiB': torch.cuda.max_memory_allocated()/1024**2,
              'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'history': history, 'contract': contract,
              'resume_semantics': 'atomic completed-epoch checkpoints; interrupted epoch replayed deterministically'}
    save(result, out/'complete.json')
    print('AV2_REFINER_COMPLETE', json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['i1', 'i2'])
    parser.add_argument('arm')
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    args = parser.parse_args()
    train_one(args.arm, args.seed, args.stage)
