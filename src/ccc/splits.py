"""Historical deterministic metadata selection, without redistributed scene lists.

Only observed metadata, names and local exposure inventories are used. This module
does not import a model, open a trajectory tensor, or inspect future labels.
"""
from pathlib import Path
import hashlib,json


def canonical_sha(value):
    data=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
    return hashlib.sha256(data).hexdigest()


def rank(namespace,scene):
    return hashlib.sha256((namespace+':'+scene).encode()).hexdigest()


def womd_role(scene):
    if int(hashlib.sha256(scene.encode()).hexdigest()[:8],16)%5:
        return 'fit'
    return 'calibration' if int(hashlib.sha256(('full_interaction_cal:'+scene).encode()).hexdigest()[:8],16)%2 else 'audit'


def av2_role(scene):
    bucket=int(hashlib.sha256(scene.encode()).hexdigest()[:8],16)%10
    return 'calibration' if bucket==0 else 'development' if bucket==1 else 'fit'


def read_ids(path):
    path=Path(path)
    if path.suffix=='.json':
        value=json.loads(path.read_text())
        if isinstance(value,dict):
            key=next((k for k in ('scene_ids','records') if k in value),None)
            assert key,'Expected a list, scene_ids, or records in local metadata'
            value=value[key]
        assert isinstance(value,list)
        ids=[r['scene'] if isinstance(r,dict) else r for r in value]
    else:
        ids=path.read_text().splitlines()
    assert all(isinstance(s,str) and s and '\n' not in s for s in ids)
    assert len(ids)==len(set(ids)),'Duplicate local scene identifiers'
    return ids


def validate_document(name,value,rules):
    expected=rules['fingerprints'][name]
    assert canonical_sha(value)==expected['canonical_SHA256'],(
        'Generated membership/order differs from the historical paper protocol',name,
        'Check corpus version, historical exclusions and qualification metadata; do not substitute another split.')


def womd_documents(records,known,old_probe,rules,qualified=None,validation=None):
    known=set(known);old_probe=set(old_probe)
    assert len(old_probe)==64,'The exact 64 previously exposed probe IDs are required'
    ids=[r['scene'] for r in records]
    assert len(ids)==len(set(ids)) and len(ids)==486995,'Expected the original complete Training metadata index'
    assert all(r.get('split','training')=='training' for r in records)
    assert all(r.get('role',womd_role(r['scene']))==womd_role(r['scene']) for r in records)
    policy=rules['womd'];parts={}
    for role,count in [('calibration',2000),('audit',5000)]:
        pool=[s for s in ids if womd_role(s)==role and s not in known]
        parts[role]=sorted(pool,key=lambda s:rank(policy[role+'_namespace'],s))[:count]
        assert len(parts[role])==count
    parts.update(fit_rule='Training sha256(scene)[:8] mod5 !=0',
                 fixed_original_manifest_sha256=policy['fixed_original_manifest_sha256'])
    # Calibration/audit/confirmation are non-fit by construction. Historical R
    # explicitly excluded them too; no fit candidates can overlap those roles.
    excluded_fit=known|old_probe
    pool=[r['scene'] for r in records if womd_role(r['scene'])=='fit'
          and int(r['targets'])>0 and r['scene'] not in excluded_fit]
    chosen=sorted(pool,key=lambda s:rank(policy['refiner_namespace'],s))[:11264]
    assert len(chosen)==11264
    refiners=dict(train=chosen[:8192],tune=chosen[8192:9216])
    documents={'womd_rcim_development.json':parts,'womd_refiners.json':refiners}
    if qualified is not None or validation is not None:
        assert qualified is not None and validation is not None,'Both qualified confirmation metadata and Validation index are required for final cohorts'
        qualified=list(qualified)
        assert len(qualified)==len(set(qualified))==policy['qualified_confirmation_pool_count']
        assert set(qualified)<=set(ids)
        assert all(womd_role(s)!='fit' for s in qualified)
        assert not set(qualified)&(known|set(parts['calibration'])|set(parts['audit']))
        confirmation=sorted(qualified,key=lambda s:(rank(policy['confirmation_namespace'],s),s))[:20000]
        validation=list(validation)
        assert all(r.get('split','validation')=='validation' for r in validation)
        ordered=[r['scene'] for r in sorted(validation,key=lambda r:(int(r['shard']),int(r['record_index'])))]
        assert len(ordered)==len(set(ordered))==44097 and not set(ordered)&set(ids)
        documents['womd_final_cohorts.json']=dict(confirmation=confirmation,validation=ordered)
    for name,value in documents.items():validate_document(name,value,rules)
    return documents


def av2_documents(train_files,val_files,rules):
    train_files=sorted(train_files);val_files=sorted(val_files)
    assert len(train_files)==len(set(train_files))==199908
    assert len(val_files)==len(set(val_files))==24988
    assert not set(train_files)&set(val_files)
    assert all(Path(n).name==n and n.endswith('.pkl') for n in train_files+val_files)
    groups={k:[] for k in ('fit','calibration','development')}
    for name in train_files:
        scene=Path(name).stem;groups[av2_role(scene)].append(scene)
    p=rules['av2']
    train=sorted(sorted(groups['fit'],key=lambda s:rank(p['refiner_train_namespace'],s))[:8192])
    tune=sorted(sorted(groups['calibration'],key=lambda s:rank(p['refiner_tune_namespace'],s))[:1024])
    documents={'av2_train_files.txt':train_files,'av2_val_files.txt':val_files,
               'av2_refiner_train.txt':train,'av2_refiner_tune.txt':tune}
    for name,value in documents.items():validate_document(name,value,rules)
    return documents


def write_documents(documents,output,rules):
    output=Path(output)
    # Validate the whole bundle and all pre-existing files before the first write.
    for name,value in documents.items():
        validate_document(name,value,rules)
        assert not (output/name).exists(),'Never overwrite local split metadata; choose a new output directory'
    output.mkdir(parents=True,exist_ok=True)
    for name,value in documents.items():
        text=json.dumps(value,indent=2)+'\n' if name.endswith('.json') else '\n'.join(value)+'\n'
        (output/name).write_text(text,encoding='utf-8')

