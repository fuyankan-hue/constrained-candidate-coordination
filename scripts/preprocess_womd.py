"""Native CPU-reader graph construction; no backbone is loaded or trained."""
import argparse,json,os,pickle,sys
from pathlib import Path
from _bootstrap import ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--split',choices=['training','validation'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    sys.path.insert(0,str(ROOT/'third_party/ndpnet_womd'))
    from ccc.data import womd_raw as raw
    raw.RAW=a.raw_root;raw.wm.interp_arc=raw.interp_arc
    a.output.mkdir(parents=True,exist_ok=True);scratch=a.output/'scratch';scratch.mkdir(exist_ok=True)
    ds=raw.SafeDataset(root=str(scratch),split='train' if a.split=='training' else 'val',
        raw_dir=str(a.raw_root/a.split/'raw'),processed_dir=str(scratch),num_historical_steps=11,num_future_steps=80)
    records=[json.loads(s) for s in a.manifest.read_text().splitlines() if s.strip()]
    for i,record in enumerate(records):
        if i%a.workers!=a.worker:continue
        assert record['split']==a.split
        dest=a.output/(record['scene']+'.pkl')
        if dest.exists():
            with dest.open('rb') as f:assert pickle.load(f)['record']==record
            continue
        value=raw.make(ds,record);tmp=dest.with_name(dest.name+'.tmp.'+str(os.getpid()))
        with tmp.open('wb') as f:pickle.dump(value,f,protocol=4)
        os.replace(tmp,dest)
        if i%100==0:print('GRAPHS',i+1,'/',len(records),flush=True)

if __name__=='__main__':main()
