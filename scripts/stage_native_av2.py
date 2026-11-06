"""Stage the bundled native AV2 interfaces and user-owned assets for supplemental reproducers."""
import argparse,json,shutil
from pathlib import Path
from _bootstrap import ROOT,SPLIT_ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--backbone',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True,help='User native cache directory containing train/val shards')
    p.add_argument('--workdir',type=Path,required=True,help='New empty native workspace; never a formal result directory')
    a=p.parse_args()
    from ccc.io import sha,save_json
    weights=json.loads(a.weights.read_text());registry=json.loads((ROOT/'configs/checkpoint_hashes.json').read_text())['av2']
    assert a.cache_root.is_dir(),'The user-owned native cache directory is missing'
    assert not a.workdir.exists(),'Do not overwrite an existing native experiment workspace'
    exports=json.loads((ROOT/'configs/artifact_release.json').read_text())['checkpoints']
    def matches_original(path,original):
        actual=sha(path)
        return actual==original or any(x['sha256']==actual and x['source_sha256']==original for x in exports)
    assert matches_original(a.backbone,registry['backbone_sha256']) and matches_original(weights['rcim'],registry['RCIM42_sha256'])
    expected=registry['head_sha256'];a.workdir.mkdir(parents=True)
    def link(src,rel):
        dest=a.workdir/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(Path(src).resolve(),target_is_directory=Path(src).is_dir())
        if rel in expected:assert matches_original(dest,expected[rel]),('Frozen asset mismatch',rel)
    for file in (ROOT/'reproduction/native_av2/code').glob('*.py'):
        link(file,'code/'+file.name)
    link(ROOT/'third_party/ndpnet_av2','source/repo')
    link(a.backbone,'checkpoints/backbone_frozen.ckpt');link(weights['rcim'],'models/rcim_seed42/best.pt')
    for seed in ('42','43','44'):
        for method,first,second in [('A_I1','A',None),('B2','B','B'),('Full','E','RCIM'),('NoRCIM','E','NoRCIM')]:
            spec=weights['heads'][seed][method]
            dest='models/refiners/seed'+seed+'/i1_'+first+'/best.pt'
            if not (a.workdir/dest).exists():link(spec['i1'],dest)
            else:assert sha(spec['i1'])==sha(a.workdir/dest),'NoRCIM must share the same E-matched I1'
            if second:link(spec['i2'],'models/refiners/seed'+seed+'/i2_'+second+'/best.pt')
    for source,name in [('av2_train_files.txt','train_files.txt'),('av2_val_files.txt','val_files.txt'),
        ('av2_refiner_train.txt','refiner_train.txt'),('av2_refiner_tune.txt','refiner_tune.txt')]:
        link(SPLIT_ROOT/source,'manifests/'+name)
    link(a.cache_root,'cache')
    # A fresh local integrity manifest covers actual supplied code/model assets.
    # It does not impersonate the byte hash of the original private archive manifest.
    files={}
    for group in ('code','checkpoints','models','manifests'):
        for file in (a.workdir/group).rglob('*'):
            if file.is_file():files[str(file.relative_to(a.workdir))]=sha(file)
    for file in (ROOT/'third_party/ndpnet_av2').rglob('*.py'):
        files['source/repo/'+str(file.relative_to(ROOT/'third_party/ndpnet_av2'))]=sha(file)
    save_json(dict(passed=True,sha256=files,strategy='Frozen original AV2 weights and native algorithm sources; independently staged local I/O inventory',
        original_manifest_note='Controller-only files omitted. This local manifest SHA differs from archived original; do not use it to bypass archived-output checks.'),a.workdir/'MODELS_FROZEN.json')
    print('NATIVE_AV2_STAGED',a.workdir,'No model inference, training or dataset scan performed')

if __name__=='__main__':main()
