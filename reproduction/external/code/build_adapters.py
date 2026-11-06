"""Prepare explicitly labelled NDPNet/AV2 ports of the authors' refiners.

Original sources are immutable. No historical project source is changed.
"""
import hashlib,json,shutil
from pathlib import Path

ROOT=Path('/workspace/external')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not list((ROOT/'models').glob('*/last.pt')), 'Do not edit adapters after training begins'
    changes=[]
    for name in ('SmartRefine','SRefiner'):
        src=ROOT/'sources'/name;dst=ROOT/'adapters'/name
        shutil.copytree(src,dst,dirs_exist_ok=True,ignore=shutil.ignore_patterns(
            '__pycache__','*.pyc','ckpts','saved_ckpts','package','assets','*.png','*.pdf'))
        if name=='SmartRefine':
            p=dst/'models/target_region.py';s=p.read_text()
            for old,new in [('data.positions[data[\'agent_index\'], 19, :]',
                             'data.positions[data[\'agent_index\'], 49, :]'),
                            ('idx = [-16, -1]','idx = [-31, -1]')]:
                assert s.count(old)==1,(str(p),old)
                s=s.replace(old,new);changes.append(dict(method=name,file=str(p.relative_to(ROOT)),old=old,new=new))
            p.write_text(s)
        else:
            p=dst/'refine/av2_target_region.py';s=p.read_text()
            old='MLP(embed_dim=134, out_channels=64)';new='MLP(embed_dim=128, out_channels=64)'
            assert s.count(old)==1;s=s.replace(old,new);p.write_text(s)
            changes.append(dict(method=name,file=str(p.relative_to(ROOT)),old=old,new=new))
            # The official archive omits data_av2_refine. These are PyG data and
            # HiVT initialization utilities, not an alternative attention model.
            compat=dst/'data_av2_refine';compat.mkdir(exist_ok=True)
            shutil.copy2(ROOT/'sources/SmartRefine/utils.py',compat/'__init__.py')
            (compat/'argo_preprocess.py').write_text('from . import init_weights, TemporalData, DistanceDropEdge\n')
            changes.append(dict(method=name,compatibility='Missing data_av2_refine utilities supplied by official SmartRefine/HiVT utilities; neural attention source retained'))
    protocol={
        'status':'ports requiring runtime smoke verification; not original released checkpoints',
        'dataset':'AV2 native 50-history/60-future at10Hz; category>=2 and observed-current targets',
        'backbone':'same frozen local NDPNet AV2 checkpoint and six cached candidates as our method',
        'training':'same8192 train/1024 tune manifests as existing refiners; scene-disjoint',
        'evaluation':'1024 fixed-hash native-train development scenes disjoint from train/tune; no Validation in this launcher',
        'mode_interface':'all six candidates sorted per target by descending frozen unary; rank0 is exactly original top1; SRefiner ranks form six deterministic joint proposals',
        'SmartRefine':{'iterations':5,'segments':2,'embed_dim':64,'lr':5e-4,
            'input_feature_dim':128,'radius':-1,'r_lo':2,'r_hi':10,
            'loss':'author multi-iteration Laplace NLL + soft classification CE +0.01 quality L1',
            'top1':'maximum final learned classification logit; fixed-five-iteration port, adaptive stopping not claimed',
            'map':'observed AV2 centerline vectors/intersections; unavailable AV1 turn/control categories are constant0 and disclosed'},
        'SRefiner':{'iterations':3,'segments':4,'embed_dim':64,'lr':1.4e-4,
            'input_feature_dim':128,'local_radius':10,
            'loss':'author per-scene best joint ADE hypothesis + multi-iteration Laplace NLL',
            'top1':'fixed rank0 joint proposal; author released classification/quality heads are not trained by regression-only loss and are not used for top1',
            'eligibility':'all observed-current requested targets; future masks used only in loss/metrics',
            'map':'AV2 centerline vectors, observed lane type, geometric width proxy and intersection flag'},
        'budget':{'seeds':[42,43,44],'batch_scenes':16,'max_epochs':64,
                  'budget_matched_snapshot_epoch':10,'scheduler':'cosine64,eta_min0',
                  'checkpoint':'tune-only native oracle minFDE for SmartRefine; minJointFDE for SRefiner; no dev-selected epoch',
                  'no_budget10_conclusion_as_fully_converged':True},
        'resources':{'max_gpus':4,'gpu_selection':'any currently idle authorized card; no other process termination',
                     'cpu_per_worker':2,'memory_per_worker_GiB':12,'data_loader_workers':0},
        'historical_results_readonly':True,'no_backbone_training':True,
        'no_WOMD_or_sealed_confirmation_access':True,
        'changes':changes,
        'original_source_manifest_sha256':sha(ROOT/'SOURCE_MANIFEST.json'),
        'adapter_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'adapters').rglob('*.py')},
    }
    (ROOT/'ADAPTER_PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
    print('ADAPTERS_PREPARED_RUNTIME_SMOKE_STILL_REQUIRED',flush=True)
if __name__=='__main__':main()
