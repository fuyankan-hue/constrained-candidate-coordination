"""Native AV2 data preparation and frozen full-method evaluation.

All prediction functions receive prediction/history inputs only. Labels are
retained separately for training losses and for evaluation after selection.
"""
import argparse, copy, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from av2_rcim import RCIM, save
from av2_core import choose, bank, refine, protect, measure, hits
from av2_refiner import Refiner, collate, objective, sample_tokens

ROOT = Path('/workspace/output')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def load_rcim():
    p = ROOT / 'models/rcim_seed42/best.pt'
    model = RCIM().eval()
    model.load_state_dict(torch.load(p, map_location='cpu')['state_dict'])
    model.requires_grad_(False)
    return model


def load_head(seed, stage, arm, device):
    p = ROOT / ('models/refiners/seed%d/%s_%s/best.pt' % (seed, stage.lower(), arm))
    m = Refiner(False).to(device).eval()
    z = torch.load(p, map_location='cpu')
    m.load_state_dict(z['state_dict']); m.requires_grad_(False)
    return m


def setup(device='cpu'):
    torch.set_num_threads(1); torch.manual_seed(42); np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if device == 'cuda': assert torch.cuda.is_available()


def existing(path):
    if not path.exists(): return None
    try:
        z = torch.load(path, map_location='cpu')
        assert isinstance(z, list) and all('scene' in r and 'arms' in r for r in z)
        return z
    except Exception:
        # Recompute this derived shard atomically; leave other verified data intact.
        return None


def prepare_i1(worker, workers=4):
    setup(); model = load_rcim(); sets = {}
    for split in ('train', 'tune'):
        sets[split] = set((ROOT / ('manifests/refiner_%s.txt' % split)).read_text().splitlines())
    assert len(sets['train']) == 8192 and len(sets['tune']) == 1024 and not sets['train'] & sets['tune']
    counts = dict(train=0, tune=0); begun = time.time()
    for ix, p in enumerate(sorted((ROOT / 'cache/train').glob('*.pt'))):
        if ix % workers != worker: continue
        cached = {s: existing(ROOT / ('prepared/i1/%s/%s' % (s, p.name))) for s in sets}
        if all(v is not None for v in cached.values()):
            for s, v in cached.items(): counts[s] += len(v)
            continue
        rows = torch.load(p, map_location='cpu')['rows']
        output = {s: [] for s in sets}
        for row in rows:
            split = next((s for s in sets if row['scene'] in sets[s]), None)
            if split is None: continue
            selected = choose(row, model)
            arms = {a: bank(row, selected[a]['path'], selected[a]['modes']) for a in ('A', 'E', 'B')}
            output[split].append(dict(scene=row['scene'], arms=arms, row=row, choices=selected))
        for s, v in output.items():
            save(ROOT / ('prepared/i1/%s/%s' % (s, p.name)), v, True); counts[s] += len(v)
        if ix % (40 * workers) == worker:
            print('AV2_PREPARE_I1', worker, ix + 1, '/ 3124', counts, 'seconds', round(time.time()-begun, 1), flush=True)
    save(ROOT / ('prepared/i1/worker%d_complete.json' % worker),
         dict(passed=True, counts=counts, worker=worker, workers=workers, seconds=time.time()-begun))


def prepare_i2(seed, device='cuda'):
    setup(device); heads = {a: load_head(seed, 'I1', a, device) for a in ('E', 'B')}
    begun = time.time(); counts = {}
    for split in ('train', 'tune'):
        count = 0
        for p in sorted((ROOT / ('prepared/i1/%s' % split)).glob('*.pt')):
            dest = ROOT / ('prepared/i2/seed%d/%s/%s' % (seed, split, p.name))
            old = existing(dest)
            if old is not None: count += len(old); continue
            out = []
            for item in torch.load(p, map_location='cpu'):
                row, ch = item['row'], item['choices']; arms = {}
                original = ch['A']['path']
                for arm in ('RCIM', 'NoRCIM', 'B'):
                    a = ch[arm]; first = heads['B' if arm == 'B' else 'E']
                    path, beta, _ = refine(row, a['path'], a['modes'], first, collate, device)
                    if arm == 'B':
                        # Historical B two-stage control: only unchanged B targets get a second residual.
                        radius = (np.asarray(a['modes']) == np.asarray(ch['A']['modes'])).astype(np.float32)
                    else:
                        distance = np.linalg.norm(path-original, axis=-1).max(1)
                        assert distance.max() <= 5.00001
                        radius = np.clip(5.-1e-4-distance, 0., 1.).astype(np.float32)
                    b = bank(row, path, a['modes']); b['radius'] = torch.as_tensor(radius)
                    b['firstP_beta'] = beta; arms[arm] = b
                out.append(dict(scene=row['scene'], arms=arms))
            save(dest, out, True); count += len(out)
            if count and count % 512 < len(out): print('AV2_PREPARE_I2', seed, split, count, flush=True)
        assert count == (8192 if split == 'train' else 1024), (split, count)
        counts[split] = count
    save(ROOT / ('prepared/i2/seed%d/complete.json' % seed), dict(passed=True, counts=counts, seconds=time.time()-begun))


def preflight():
    from av2_refiner_train import batch as train_batch, loss_value
    setup(); torch.manual_seed(42); scorer = RCIM().eval(); model = Refiner(False).eval()
    assert sum(p.numel() for p in model.parameters()) == 76026
    assert sum(p.numel() for p in scorer.parameters()) == 44449
    rows = torch.load(ROOT / 'probe_features.pt', map_location='cpu')['rows'][:8]
    details = []; opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    for row in rows:
        poisoned = dict(row, gt=torch.full_like(row['gt'], 99999.), valid=~row['valid'])
        a, b = choose(row, scorer), choose(poisoned, scorer)
        for arm in ('A','E','B','NoRCIM','RCIM'):
            assert a[arm]['modes'] == b[arm]['modes'] and np.array_equal(a[arm]['path'], b[arm]['path'])
        bb = bank(row, a['RCIM']['path'], a['RCIM']['modes'])
        pp = bank(poisoned, a['RCIM']['path'], a['RCIM']['modes'])
        x,y = collate([bb], [0]); xp,_ = collate([pp], [0])
        assert all(torch.equal(x[k],xp[k]) for k in x)
        # New model identity is checked before the optimizer changes its zero head.
        fresh = Refiner(False).eval(); assert torch.equal(fresh(x), torch.zeros_like(x['path_local']))
        opt.zero_grad(set_to_none=True); loss, delta = objective(model, x, y)
        assert torch.isfinite(loss); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); assert torch.isfinite(gn); opt.step()
        assert torch.equal(model(x), model(xp))
        path, beta, raw = refine(row, a['RCIM']['path'], a['RCIM']['modes'], model, collate, 'cpu')
        _, oldhits = hits(a['RCIM']['path']); _, newhits = hits(path)
        assert not (newhits.any(-1) & ~oldhits.any(-1)).any() and newhits.sum() <= oldhits.sum()
        assert np.linalg.norm(a['RCIM']['path']-a['A']['path'], axis=-1).max() <= 4.0000001
        assert np.linalg.norm(path-a['A']['path'],axis=-1).max() <= 5.00001
        # Rotation must preserve local/world displacement norms and round-trip.
        rot = bb['rotation']; local = np.ones((len(rot),60,2))*.13
        world = np.einsum('ntc,ndc->ntd',local,rot)
        assert np.allclose(np.linalg.norm(local,axis=-1), np.linalg.norm(world,axis=-1),atol=1e-7)
        assert np.allclose(np.einsum('ntc,ncd->ntd',world,rot),local,atol=1e-7)
        details.append(dict(scene=row['scene'], targets=len(row['pi']), loss=float(loss), firstP_beta=beta))
        # Verify the actual second-stage loss and radius multiplication too.
        bb['radius']=torch.full((len(row['pi']),),.5)
        tx,ty,radius=train_batch([bb],'cpu')
        sm=Refiner(False).eval();so=torch.optim.AdamW(sm.parameters(),lr=2e-4,weight_decay=1e-4)
        for _ in range(2):
            so.zero_grad(set_to_none=True);sl,sd=loss_value(sm,tx,ty,radius,'i2')
            assert torch.isfinite(sl);sl.backward();sg=torch.nn.utils.clip_grad_norm_(sm.parameters(),1.)
            assert torch.isfinite(sg);so.step()
            assert sd.norm(dim=-1).max()<=.50001
    t = torch.randn(2,3,80,2); assert torch.equal(sample_tokens(t),t[:,:,4::5])
    # End-to-end method composition with smoke-only heads; production paths unchanged.
    toyheads={(stage,arm):Refiner(False).eval() for stage,arms in
              [('I1',('A','E','B')),('I2',('RCIM','NoRCIM','B'))] for arm in arms}
    rr=rows[0];cc=choose(rr,scorer);composed=final_paths(rr,cc,toyheads,42,'cpu')
    for name,path in composed.items():
        mt=measure(rr,path,cc['A']['path']);assert mt['targets']==int(rr['valid'][:,-1].sum())
        assert np.isfinite(mt['ade_sum']) and np.isfinite(mt['fde_sum'])
    # Full downstream controller must exist before this pass can release it.
    for name in ('method_controller.py','av2_refiner_train.py','av2_pipeline.py','av2_core.py'):
        compile((ROOT / 'code' / name).read_text(), name, 'exec')
    save(ROOT/'AV2_PIPELINE_PREFLIGHT_PASS.json',dict(passed=True,refiner_parameters=76026,
         rcim_parameters=44449,GT_poison_invariance=True,zero_residual_initialization=True,
         finite_training_gradient=True,projection_and_protection_contracts=True,local_world_roundtrip=True,
         native_future_steps=60,reference80_step_sampler_equivalent=True,details=details,
         code_sha256={p.name:sha(p) for p in sorted((ROOT/'code').glob('*.py'))}))
    print('AV2_FULL_PIPELINE_PREFLIGHT_PASS', flush=True)


def final_paths(row, choices, heads, seed, device):
    out = {'NDPNet':choices['A']['path'], 'RCIM_stage1':choices['RCIM']['path'],
           'OldRCIM_stage1':choices['E']['path'], 'GeometryB_stage1':choices['B']['path']}
    original = choices['A']['path']
    for arm in ('A','E','B'):
        z = choices[arm]; p,beta,raw = refine(row,z['path'],z['modes'],heads[('I1',arm)],collate,device)
        out[{'A':'NDPNet_refined','E':'OldRCIM_refined','B':'GeometryB_refined'}[arm]]=p
    for arm in ('RCIM','NoRCIM','B'):
        z=choices[arm]; i1=heads[('I1','B' if arm=='B' else 'E')]
        first,beta,_=refine(row,z['path'],z['modes'],i1,collate,device)
        radius=(np.asarray(z['modes'])==np.asarray(choices['A']['modes'])).astype(np.float32) if arm=='B' else np.clip(5.-1e-4-np.linalg.norm(first-original,axis=-1).max(1),0.,1.).astype(np.float32)
        last,beta2,raw=refine(row,first,z['modes'],heads[('I2',arm)],collate,device,radius)
        if arm in ('RCIM','NoRCIM'): assert np.linalg.norm(last-original,axis=-1).max()<=5.00001
        out[{'RCIM':'FullRCIM','NoRCIM':'Full_without_RCIM','B':'GeometryB_two_refiners'}[arm]]=last
    return out


def evaluate(worker,workers=4,device='cuda'):
    setup(device); frozen=json.loads((ROOT/'MODELS_FROZEN.json').read_text())
    for path,digest in frozen['sha256'].items(): assert sha(ROOT/path)==digest,path
    scorer=load_rcim();heads={s:{(stage,arm):load_head(s,stage,arm,device)
                       for stage,arms in [('I1',('A','E','B')),('I2',('RCIM','NoRCIM','B'))] for arm in arms}
                       for s in (42,43,44)}
    begin=time.time();done=0
    for ix,p in enumerate(sorted((ROOT/'cache/val').glob('*.pt'))):
        if ix%workers!=worker:continue
        dest=ROOT/('evaluation/shards/%s'%p.name)
        if dest.exists():
            try:
                old=torch.load(dest,map_location='cpu');assert old['frozen_manifest_sha256']==sha(ROOT/'MODELS_FROZEN.json')
                expected=torch.load(p,map_location='cpu')['rows']
                assert [r['scene'] for r in old['records']]==[r['scene'] for r in expected]
                done+=len(old['records']);continue
            except Exception as exc:
                print('RECOMPUTE_DAMAGED_EVALUATION_SHARD',str(dest),repr(exc),flush=True)
        records=[]
        for row in torch.load(p,map_location='cpu')['rows']:
            started=time.perf_counter();ch=choose(row,scorer);choice_seconds=time.perf_counter()-started
            methods={}; paths={}; n=len(row['pi'])
            for seed in (42,43,44):
                tick=time.perf_counter(); allpaths=final_paths(row,ch,heads[seed],seed,device)
                if device=='cuda':torch.cuda.synchronize()
                latency=time.perf_counter()-tick
                methods[str(seed)]={name:measure(row,path,ch['A']['path']) for name,path in allpaths.items()}
                for name,m in methods[str(seed)].items():
                    anchor='A' if name in ('NDPNet','NDPNet_refined') else 'E' if name.startswith('OldRCIM') else 'B' if name.startswith('GeometryB') else 'NoRCIM' if name=='Full_without_RCIM' else 'RCIM'
                    m['changed']=int(((np.asarray(ch[anchor]['modes'])!=np.asarray(ch['A']['modes'])) & row['valid'][:,-1].numpy()).sum())
                    m['mean_shift_from_A']=float(np.linalg.norm(allpaths[name]-ch['A']['path'],axis=-1).mean())
                paths[str(seed)]={k:torch.as_tensor(v,dtype=torch.float64) for k,v in allpaths.items()}
                methods[str(seed)]['timing']={'all_variants_seconds':latency}
            records.append(dict(scene=row['scene'],original_agent_id=row['original_agent_id'],methods=methods,
                  candidate_modes={k:v['modes'] for k,v in ch.items() if isinstance(v,dict) and 'modes' in v},
                  choice_seconds=choice_seconds,predictions=paths))
        save(dest,dict(records=records,frozen_manifest_sha256=sha(ROOT/'MODELS_FROZEN.json')),True)
        done+=len(records);save(ROOT/('evaluation/worker%d_state.json'%worker),dict(done=done,worker=worker,
             assigned_shards=len(range(worker,391,workers)),seconds=time.time()-begin))
        print('AV2_EVALUATION',worker,done,'scenes',round(time.time()-begin,1),'seconds',flush=True)
    save(ROOT/('evaluation/worker%d_complete.json'%worker),dict(passed=True,scenes=done,seconds=time.time()-begin))


def aggregate_rows(rows):
    names=[k for k in rows[0]['methods']['42'] if k!='timing'];result={}
    for seed in ('42','43','44'):
        result[seed]={}
        for name in names:
            rr=[r['methods'][seed][name] for r in rows];sumv=lambda key:sum(v[key] for v in rr)
            nt=sumv('targets');pair=sumv('pairs');actors=sumv('all_predicted_targets');focal=sumv('focal_targets')
            result[seed][name]=dict(ADE6=sumv('ade_sum')/nt,FDE6=sumv('fde_sum')/nt,
                center_conflict_pairs=sumv('overlap'),center_conflict_pair_rate=sumv('overlap')/max(1,pair),
                center_conflict_steps=sumv('overlap_steps'),actor_conflict_rate=sumv('actors_with_conflict')/max(1,actors),
                extra_gt5=sumv('gt5'),valid6_targets=nt,all_predicted_targets=actors,eligible_pairs=pair,
                focal_targets=focal,focal_ADE6=sumv('focal_ade_sum')/max(1,focal),focal_FDE6=sumv('focal_fde_sum')/max(1,focal),
                improved=sumv('improved'),degraded=sumv('degraded'),changed=sumv('changed'),
                mean_acceleration=sumv('acceleration_sum')/max(1,sumv('acceleration_count')),
                mean_jerk=sumv('jerk_sum')/max(1,sumv('jerk_count')),
                max_shift_from_A=max(v['max_shift_from_A'] for v in rr))
    means={name:{k:dict(mean=float(np.mean([result[s][name][k] for s in result])),
                          sample_sd=float(np.std([result[s][name][k] for s in result],ddof=1)))
                 for k in result['42'][name]} for name in names}
    return result,means


def summary():
    setup();rows=[]
    for p in sorted((ROOT/'evaluation/shards').glob('*.pt')):
        loaded=torch.load(p,map_location='cpu')['records']
        for r in loaded:r.pop('predictions',None)
        rows.extend(loaded)
    assert len(rows)==24988 and len({r['scene'] for r in rows})==24988
    rows.sort(key=lambda r:r['scene']);table,means=aggregate_rows(rows)
    comparisons=[('FullRCIM','Full_without_RCIM'),('FullRCIM','GeometryB_two_refiners'),
                 ('FullRCIM','OldRCIM_refined'),('FullRCIM','NDPNet_refined'),('FullRCIM','NDPNet')]
    rng=np.random.default_rng(4201);intervals={}
    # Fixed three models averaged within each scene, then resample whole scenes.
    for new,old in comparisons:
        metrics={}
        for numerator,denominator,label in [('ade_sum','targets','ADE6'),('fde_sum','targets','FDE6'),('overlap','pairs','center_conflict_rate')]:
            diff=np.array([np.mean([r['methods'][str(s)][new][numerator]-r['methods'][str(s)][old][numerator] for s in (42,43,44)]) for r in rows])
            den=np.array([r['methods']['42'][new][denominator] for r in rows]);sims=[]
            for _ in range(5000):
                ix=rng.integers(0,len(rows),len(rows));sims.append(float(diff[ix].sum()/max(1,den[ix].sum())))
            metrics[label]=dict(delta=float(diff.sum()/max(1,den.sum())),CI95=np.quantile(sims,[.025,.975]).tolist())
        intervals[new+'_minus_'+old]=metrics
    save(ROOT/'AV2_RESULTS.json',dict(passed=True,scenes=len(rows),per_seed=table,three_seed=means,
         paired_scene_bootstrap=intervals,bootstrap_scope='Conditioned on fixed RCIM42 and three fixed refiner seeds, not all training seeds.',
         metric_protocol='SelectedTop1 ADE/FDE over native6s;not official min-of6 metrics. Scored+focal valid endpoint, focal metrics separate; center-distance<1m conflict proxy,not WOMD oriented boxes.',
         measured_inference_cost=dict(selection_all_method_bundle_mean_ms=float(np.mean([r['choice_seconds'] for r in rows])*1000),
             refinement_all_variant_bundle_mean_ms=float(np.mean([r['methods'][str(s)]['timing']['all_variants_seconds'] for r in rows for s in (42,43,44)])*1000),
             scope='Measured sequential diagnostic bundle,not single-method production latency; excludes cached backbone extraction.'),
         modules_selected_on_train_calibration_only=True,backbone='Historical locally adapted AV2 NDPNet selected using AV2 validation; not official author weights or fully unexposed validation.',
         frozen_manifest_sha256=sha(ROOT/'MODELS_FROZEN.json')))
    print('AV2_FULL_METHOD_RESULTS_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','prepare_i1','prepare_i2','evaluate','summary']);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    if a.mode=='preflight':preflight()
    elif a.mode=='prepare_i1':prepare_i1(a.worker,a.workers)
    elif a.mode=='prepare_i2':prepare_i2(a.seed)
    elif a.mode=='evaluate':evaluate(a.worker,a.workers)
    else:summary()
