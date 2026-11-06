"""Index existing native feature caches by scene. Does not invoke a model."""
import argparse,json
from pathlib import Path
from _bootstrap import ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--split',choices=['fit','calibration','tune','audit','confirmation','validation'],required=True)
    p.add_argument('--scene-ids',type=Path,help='One scene per line; mandatory for non-validation WOMD')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from ccc.io import config,cache_rows
    cfg=config(a.config);wanted=set(a.scene_ids.read_text().splitlines()) if a.scene_ids else None
    assert wanted or a.split=='validation','Specify the frozen cohort; never guess a development cohort'
    assert not a.output.exists(),'Use a new manifest path'
    records=[]
    for file in sorted(a.cache_root.rglob('*.pt')):
        for row in cache_rows(file,cfg['dataset']):
            sid=str(row['scene'])
            if wanted is None or sid in wanted:records.append(dict(scene=sid,split=a.split,path=str(file.resolve())))
    assert len(records)==len({r['scene'] for r in records})
    if wanted is not None:assert {r['scene'] for r in records}==wanted,'Missing frozen scenes'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(r)+'\n' for r in sorted(records,key=lambda r:r['scene'])))
    print('INDEXED',len(records),'scenes; user-local manifest, do not commit')

if __name__=='__main__':main()
