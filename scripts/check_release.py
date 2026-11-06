"""Local checks only. Does not load data, train, use a GPU, or access a network."""
import argparse,hashlib,json,re
from pathlib import Path
from _bootstrap import ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--imports',action='store_true');a=p.parse_args()
    from ccc.io import config
    files=list(ROOT.rglob('*.py'))
    for file in files:compile(file.read_text(encoding='utf-8'),str(file),'exec')
    for file in (ROOT/'configs').glob('*.json'):
        json.loads(file.read_text())
    for name in ('womd','av2'):config(ROOT/('configs/'+name+'.json'))
    rules=json.loads((ROOT/'configs/split_rules.json').read_text())
    assert len(rules['fingerprints'])==7
    assert not (ROOT/'configs/splits').exists(),'Dataset-specific lists must stay in ignored local storage'
    assert len(json.loads((ROOT/'results/paper_tables.json').read_text())['tables'])==11
    for entry in json.loads((ROOT/'docs/source_manifest.json').read_text()):
        file=ROOT/entry['release_file']
        assert hashlib.sha256(file.read_bytes()).hexdigest()==entry['release_sha256'],('Packaged source changed',entry['release_file'])
    forbidden=re.compile(r'(?:\b(?:10\.101\.\d+\.\d+)\b|/data/(?:yanjinlong|datasets)/|[CD]:[\\/]Users[\\/]|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,})')
    for file in ROOT.rglob('*'):
        if not file.is_file() or '.git' in file.parts or '__pycache__' in file.parts:continue
        assert file.suffix.lower() not in ('.pt','.pth','.ckpt','.pkl','.npy','.npz','.tfrecord'),str(file)
        try:text=file.read_text(encoding='utf-8')
        except UnicodeDecodeError:continue
        # The scanner's own regular-expression source is not a disclosed address/path.
        if file.name!='check_release.py':assert not forbidden.search(text),str(file)
    counts={}
    if a.imports:
        import importlib
        modules=['ccc.inference','ccc.recipes.womd_rcim','ccc.recipes.womd_i1','ccc.recipes.womd_i2',
                 'ccc.recipes.av2_rcim','ccc.recipes.av2_refiners','ccc.core.womd_labels','ccc.core.womd_native_row']
        for name in modules:importlib.import_module(name)
        from ccc.core.rcim_womd import Model
        from ccc.core.rcim_av2 import RCIM
        from ccc.core.refiner_womd import Refiner
        counts=dict(RCIM_WOMD=sum(p.numel() for p in Model('refine_only').parameters()),
                    RCIM_AV2=sum(p.numel() for p in RCIM().parameters()),Refiner=sum(p.numel() for p in Refiner(False).parameters()))
        assert list(counts.values())==[44449,44449,76026]
    print(json.dumps(dict(pass_=True,python_files=len(files),paper_tables=11,parameter_counts=counts,no_training=True,local_scene_memberships='not bundled; separately regenerate and verify fingerprints')))

if __name__=='__main__':main()
