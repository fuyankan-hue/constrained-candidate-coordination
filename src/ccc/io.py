"""Portable file adapters; no experiment paths or credentials are embedded."""
from pathlib import Path
import hashlib,json,os

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def save_json(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');os.replace(str(tmp),str(path))

def save_torch(value,path):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    torch.save(value,str(tmp));os.replace(str(tmp),str(path))

def read_manifest(path,split=None):
    path=Path(path);rows=[]
    for line in path.read_text().splitlines():
        if not line.strip():continue
        r=json.loads(line)
        if split is not None and r['split']!=split:continue
        p=Path(r['path']);r['path']=str(p if p.is_absolute() else path.parent/p)
        rows.append(r)
    assert len(rows)==len({r['scene'] for r in rows}),'Duplicate scene ID'
    return rows

def cache_rows(path,dataset):
    import torch
    value=torch.load(str(path),map_location='cpu')
    if dataset=='womd':
        return [value] if 'm1' in value else value['rows']
    return value['rows'] if isinstance(value,dict) and 'rows' in value else value if isinstance(value,list) else [value]

def selected_row(record,dataset):
    rows=cache_rows(record['path'],dataset)
    match=[r for r in rows if str(r['scene'])==str(record['scene'])]
    assert len(match)==1,(record['scene'],'cache mapping mismatch')
    return match[0]

def config(path):
    p=Path(path);value=json.loads(p.read_text())
    assert value['dataset'] in ('womd','av2')
    assert value['RCIM']['parameters']==44449 and value['RCIM']['seed']==42
    assert value['coordination']=={'modes':6,'observed_graph_distance_m':75,'CJM_max_path_m':20,
        'projection_max_path_m':4,'projection_buffer_m':1e-8,'joint_score_weight':1,
        'exact_product_limit':65536,'coordinate_sweeps':8,'original_anchor':'independent_top1'}
    assert value['refinement']['parameters_each']==76026
    assert value['refinement']['seeds']==[42,43,44]
    r=value['RCIM'];f=value['refinement']
    for key,expected in dict(loss='original_joint_ce',soft_label_temperature=3,relative_FDE_weight=.25,
            lr=2e-4,weight_decay=1e-4,gradient_clip=5,max_epochs=8,patience=3,min_delta=1e-4).items():
        assert r[key]==expected,('Frozen RCIM setting changed',key)
    for key,expected in dict(train_scenes=8192,tune_scenes=1024,batch_size=16,max_epochs=10,
            optimizer_steps_each=5120,lr=2e-4,weight_decay=1e-4,eta_min=2e-5,min_delta=1e-4,
            total_path_shift_bound_m=5).items():
        assert f[key]==expected,('Frozen refiner setting changed',key)
    assert f['I1']['gradient_clip']==5 and f['I2']['gradient_clip']==1
    assert f['I1']['position_loss']=='SmoothL1' and f['I2']['position_loss']=='Euclidean'
    assert f['protection_betas']==[1,.5,.25,.125,0]
    assert r['supervision']['minimum_common_future_steps']==(60 if value['dataset']=='womd' else 45)
    assert value['evaluation']['extra_FDE_threshold_m']==5.000001
    assert value['evaluation']['bootstrap_resamples']==5000
    return value
