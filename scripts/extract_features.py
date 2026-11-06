"""Use the frozen native extraction functions. Run only in the backbone environment."""
import argparse,json,os,sys,pickle
from pathlib import Path
from _bootstrap import ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--preprocessed-root',type=Path,help='WOMD graphs produced separately in the native CPU reader environment')
    p.add_argument('--manifest',type=Path,required=True,help='WOMD raw record JSONL, or AV2 processed filenames')
    p.add_argument('--split',choices=['training','validation','train','val'],required=True)
    p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--worker',type=int,default=0)
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    from ccc.io import config,sha,save_torch
    cfg=config(a.config);dataset=cfg['dataset'];a.workdir.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'third_party'/('ndpnet_'+dataset)))
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count()==1,'Expose exactly one authorized GPU'
    assert 0<=a.worker<a.workers
    if dataset=='womd':
        from ccc.data import womd_extract as extract
        assert a.preprocessed_root,'WOMD raw parsing uses its separate Python3.11/TensorFlow CPU reader'
        extract.CHECKPOINT=a.checkpoint;extract.CHECKPOINT_SHA=sha(a.checkpoint);extract.OUT=a.workdir
        model,captured,handles=extract.setup()
        records=[json.loads(s) for s in a.manifest.read_text().splitlines() if s.strip()]
        for i,record in enumerate(records):
            if i%a.workers!=a.worker:continue
            assert record['split']==a.split
            dest=a.workdir/'cache'/a.split/('%04d'%record['shard'])/(record['scene']+'.pt')
            if dest.exists():extract.validate(torch.load(dest,map_location='cpu'),record);continue
            with (a.preprocessed_root/(record['scene']+'.pkl')).open('rb') as f:prepared=pickle.load(f)
            assert prepared['record']==record
            value=extract.make(model,captured,prepared,False);save_torch(value,dest)
            if i%100==0:print('FEATURES',i+1,'/',len(records),flush=True)
    else:
        os.environ['CCC_AV2_WORK']=str(a.workdir.resolve());os.environ['CCC_AV2_PROCESSED']=str(a.raw_root.resolve())
        from ccc.data import av2_extract as extract
        # The exact native Extractor checkpoint location is configured without copying a binary.
        expected=a.workdir/'checkpoints/backbone_frozen.ckpt';expected.parent.mkdir(exist_ok=True)
        if not expected.exists():expected.symlink_to(a.checkpoint.resolve())
        assert sha(expected)==sha(a.checkpoint)
        e=extract.Extractor();names=a.manifest.read_text().splitlines()
        for shard,start in enumerate(range(0,len(names),64)):
            if shard%a.workers!=a.worker:continue
            subset=names[start:start+64];dest=a.workdir/'cache'/a.split/('%05d.pt'%shard)
            if dest.exists():assert extract.validate(dest,subset,e.checkpoint_sha);continue
            value=dict(rows=[e.row(a.split,nm) for nm in subset],filenames=subset,backbone_sha256=e.checkpoint_sha)
            save_torch(value,dest);print('FEATURE_SHARDS',shard+1,'/',(len(names)+63)//64,flush=True)

if __name__=='__main__':main()
