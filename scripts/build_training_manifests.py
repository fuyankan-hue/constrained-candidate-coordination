"""Map fixed WOMD scene membership to native cache paths using only the raw index."""
import argparse,json
from pathlib import Path
from _bootstrap import ROOT,SPLIT_ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-index',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True,help='Native cache/training directory')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    records=[json.loads(s) for s in a.raw_index.read_text().splitlines() if s.strip()]
    parts=json.loads((SPLIT_ROOT/'womd_rcim_development.json').read_text())
    heads=json.loads((SPLIT_ROOT/'womd_refiners.json').read_text())
    calibration=set(parts['calibration']);audit=set(parts['audit']);train=set(heads['train']);tune=set(heads['tune'])
    assert not calibration&audit and not train&tune
    head_ids=train|tune
    rcim=[];refiners=[]
    for r in records:
        sid=r['scene'];path=a.cache_root/('%04d'%r['shard'])/(sid+'.pt')
        rec=dict(scene=sid,path=str(path.resolve()),shard=r['shard'],record_index=r['record_index'])
        split='fit' if r['role']=='fit' else 'calibration' if sid in calibration else 'audit' if sid in audit else None
        if split:rcim.append(dict(rec,split=split))
        if sid in head_ids:refiners.append(dict(rec,split='fit' if sid in train else 'tune'))
    assert {r['scene'] for r in refiners}==head_ids
    assert {r['scene'] for r in rcim if r['split']=='calibration'}==calibration
    assert {r['scene'] for r in rcim if r['split']=='audit'}==audit
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for name,rr in [('womd_rcim_training',rcim),('refiner_train_tune',refiners)]:
        dest=a.output_dir/(name+'.jsonl');assert not dest.exists()
        dest.write_text(''.join(json.dumps(r)+'\n' for r in rr))
    print('MANIFESTS_WRITTEN',len(rcim),len(refiners),'No cache tensors were loaded')

if __name__=='__main__':main()
