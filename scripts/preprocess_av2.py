"""Invoke the pinned QCNet AV2 parser offline; do not download or overwrite graphs."""
import argparse,hashlib,importlib.util,inspect,json,sys
from pathlib import Path
from _bootstrap import ROOT


def load_parser():
    source=ROOT/'third_party/ndpnet_av2/datasets/argoverse_v2_dataset.py'
    expected='851071ad9bd4fef01b2d7ba705b71ea2c1db6dc299a4ea2c1a18d9e756437399'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected,'Pinned QCNet parser changed'
    # Load the file directly: the NDPNet datasets package exports a WOMD class.
    # The parser's two geometry/list helpers use the retained QCNet-derived utils.
    sys.path.insert(0,str(ROOT/'third_party/ndpnet_av2'))
    spec=importlib.util.spec_from_file_location('ccc_qcnet_av2_dataset',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from av2.map.map_api import ArgoverseStaticMap
    from av2.utils.io import read_json_file
    import pyarrow
    parser=module.ArgoverseV2Dataset
    signature=inspect.signature(parser.__init__)
    for name,value in [('dim',3),('num_historical_steps',50),('num_future_steps',60),
                       ('predict_unseen_agents',False),('vector_repr',True)]:
        assert signature.parameters[name].default==value,(name,'Unexpected parser default')
    return parser,expected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True,help='AV2 root with SPLIT/raw/SCENE directories')
    p.add_argument('--split',choices=['train','val'],required=True)
    p.add_argument('--check-only',action='store_true',help='Check parser imports/defaults only; no dataset is read or processed')
    args=p.parse_args();parser,digest=load_parser()
    if args.check_only:
        print(json.dumps(dict(pass_=True,parser_SHA256=digest,history_steps=50,future_steps=60,
            parser_imports=True,dataset_read=False,preprocessing_executed=False,training=False)));return
    root=args.root.resolve();raw=root/args.split/'raw';processed=root/args.split/'processed'
    expected=199908 if args.split=='train' else 24988
    if processed.exists():
        entries=list(processed.iterdir())
        if len(entries)==expected and all(f.is_file() and f.suffix=='.pkl' for f in entries):
            print('PROCESSED_SPLIT_ALREADY_PRESENT',args.split,expected,'No files changed');return
        assert not entries,'Refusing to overwrite a partial processed directory; use a new empty output tree'
    assert raw.is_dir(),'Download AV2 from the official provider and unpack it into SPLIT/raw first'
    names=[f.name for f in raw.iterdir() if f.is_dir()]
    assert len(names)==len(set(names))==expected,'Expected the complete official split; no automatic download is permitted'
    class OfflineDataset(parser):
        def download(self):
            raise RuntimeError('Automatic download/extraction is disabled; prepare the official raw split yourself')
    dataset=OfflineDataset(root=str(root),split=args.split,dim=3,num_historical_steps=50,
        num_future_steps=60,predict_unseen_agents=False,vector_repr=True)
    assert len(dataset.processed_file_names)==expected
    print('AV2_GRAPHS_PREPARED',args.split,expected,'Native QCNet parser; no model inference or training')


if __name__=='__main__':main()
