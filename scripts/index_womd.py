"""Wrap the native CRC/protobuf indexer; only the explicitly requested split is read."""
import argparse,hashlib,json
from pathlib import Path
from _bootstrap import ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--split',choices=['training','validation'],required=True)
    p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from ccc.io import save_json
    from ccc.data import womd_index as native
    assert not a.output.exists();a.workdir.mkdir(parents=True,exist_ok=True)
    files=sorted((a.raw_root/a.split/'raw').glob(a.split+'.tfrecord-*'))
    assert len(files)==(1000 if a.split=='training' else 150),'Expected complete original raw split'
    inventory=dict(files=[dict(path=str(p.resolve()),split=a.split,shard=int(p.name.split('-')[1]),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in files])
    save_json(inventory,a.workdir/'inventory.json');native.RUN=a.workdir;native.main(0,1)
    records=[]
    for file in sorted((a.workdir/'index').glob(a.split+'_*.json')):
        d=json.loads(file.read_text())
        for r in d['records']:
            sid=r['scene'];bucket=int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)%5
            role='validation' if a.split=='validation' else ('fit' if bucket else (
                'calibration' if int(hashlib.sha256(('full_interaction_cal:'+sid).encode()).hexdigest()[:8],16)%2 else 'audit'))
            records.append(dict(r,shard=d['shard'],split=a.split,role=role,previous=1))
    assert len(records)==len({r['scene'] for r in records})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(r)+'\n' for r in records))
    print('RAW_RECORDS_INDEXED',len(records),'previous=1 is conservative unknown exposure, not a new unseen claim')

if __name__=='__main__':main()
