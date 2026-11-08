"""Assemble checksum-pinned assets from the anonymous review archive; no network/model execution."""
import argparse,hashlib,json,os,tarfile
from pathlib import Path
from _bootstrap import ROOT


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def fetch(name,registry,output):
    spec=registry['assets'][name];dest=output/name
    if dest.exists():
        if sha(dest)!=spec['sha256']:raise ValueError('Existing asset has the wrong checksum: '+name)
        return dest
    tmp=dest.with_name(dest.name+'.download')
    try:
        if registry.get('asset_transport')!='repository_hex_chunks':
            raise ValueError('Expected the anonymous offline asset manifest')
        with tmp.open('wb') as f:
            for chunk in spec['chunks']:
                path=ROOT/chunk['path']
                if not path.is_file():
                    raise FileNotFoundError('Missing encoded review asset. Download the complete review ZIP, including assets/review, and preserve its layout.')
                encoded=path.read_bytes()
                if hashlib.sha256(encoded).hexdigest()!=chunk['sha256']:
                    raise ValueError('Encoded review chunk checksum mismatch: '+chunk['path'])
                block=bytes.fromhex(encoded.decode('ascii'))
                if len(block)!=chunk['decoded_bytes']:
                    raise ValueError('Encoded chunk length mismatch')
                f.write(block)
        if tmp.stat().st_size!=spec['bytes'] or sha(tmp)!=spec['sha256']:
            raise ValueError('Downloaded checksum mismatch: '+name)
        os.replace(str(tmp),str(dest))
    finally:
        if tmp.exists():tmp.unlink()
    return dest


def extract(archive,output,expected):
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();seen=set()
        for member in members:
            path=output/member.name
            if not member.isfile() or member.name in seen or output.resolve() not in path.resolve().parents:
                raise ValueError('Unsafe or duplicate artifact path')
            seen.add(member.name)
            if member.name not in expected:raise ValueError('Unexpected artifact file: '+member.name)
            if path.exists() and sha(path)!=expected[member.name]:raise ValueError('Refusing to overwrite mismatching file: '+member.name)
        if seen!=set(expected):raise ValueError('Incomplete artifact archive')
        for member in members:
            dest=output/member.name
            if dest.exists():continue
            data=tar.extractfile(member).read()
            if hashlib.sha256(data).hexdigest()!=expected[member.name]:raise ValueError('Export checksum mismatch')
            dest.parent.mkdir(parents=True,exist_ok=True)
            tmp=dest.with_name(dest.name+'.tmp');tmp.write_bytes(data);os.replace(str(tmp),str(dest))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['womd','av2','all'],default='all')
    p.add_argument('--output',type=Path,default=ROOT/'local/paper_artifacts')
    a=p.parse_args()
    registry=json.loads((ROOT/'configs/artifact_release.json').read_text(encoding='utf-8'))
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    # The notice is shown BEFORE downloading; all archives also include it.
    print('Research/non-commercial models. Waymo terms apply to WOD models; AV2 dataset terms and upstream rights remain separate. See docs/checkpoints.md.')
    fetch('paper-split-hashes.json.gz',registry,output)
    datasets=['womd','av2'] if a.dataset=='all' else [a.dataset]
    for dataset in datasets:
        archive=fetch(dataset+'-paper-checkpoints.tar.gz',registry,output)
        expected={x['path']:x['sha256'] for x in registry['checkpoints'] if x['path'].startswith(dataset+'/')}
        expected.update(registry['archive_notices'][dataset])
        extract(archive,output,expected)
        def ck(rel):return str(output/dataset/'checkpoints'/rel)
        heads={}
        for seed in (42,43,44):
            prefix='seed'+str(seed)+'/'
            heads[str(seed)]={
                'Full':dict(i1=ck(prefix+'i1_E.pt'),i2=ck(prefix+'i2_RCIM.pt')),
                'NoRCIM':dict(i1=ck(prefix+'i1_E.pt'),i2=ck(prefix+'i2_NoRCIM.pt')),
                'B2':dict(i1=ck(prefix+'i1_B.pt'),i2=ck(prefix+'i2_B.pt')),
                'A_I1':dict(i1=ck(prefix+'i1_A.pt')),
                'E_I1':dict(i1=ck(prefix+'i1_E.pt'))}
        value=dict(backbone=ck('backbone.ckpt'),rcim=ck('rcim42.pt'),heads=heads)
        target=output/dataset/'weights.json'
        if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=value:raise ValueError('Existing local weights map differs')
        target.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        print('VERIFIED',dataset,'weights:',target)
    print('Verified offline review assets; no network, model inference, training or dataset access.')


if __name__=='__main__':main()
