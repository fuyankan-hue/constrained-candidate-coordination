"""Path-only wrapper for the exact native isolated AV2 benchmark; no training."""
import argparse,importlib.util,sys,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from _bootstrap import ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True,help='Created by stage_native_av2.py')
    p.add_argument('--processed-root',type=Path,required=True,help='AV2 root containing train/processed')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['prepare','verify','benchmark'],required=True)
    p.add_argument('--method',choices=['NDPNet','NDPNet_refined','OldRCIM_refined','GeometryB_two_refiners','Full_without_RCIM','FullRCIM'])
    a=p.parse_args()
    assert a.mode!='benchmark' or a.method
    a.output.mkdir(parents=True,exist_ok=True)
    work=a.workspace.resolve();sys.path[:0]=[str(work/'code'),str(work/'source/repo')]
    import av2_pipeline, av2_rcim
    av2_pipeline.ROOT=work;av2_rcim.ROOT=work
    from ccc.data import av2_extract
    av2_extract.ROOT=work;av2_extract.PROCESSED=a.processed_root.resolve()
    sys.modules['frozen_features']=av2_extract
    source=ROOT/'reproduction/av2_closing/inference_cost.py'
    spec=importlib.util.spec_from_file_location('native_isolated_av2_cost',source)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    native.ROOT=work;native.OUT=a.output.resolve()
    if a.mode=='prepare':native.prepare()
    elif a.mode=='verify':native.verify()
    else:native.benchmark(a.method)

if __name__=='__main__':main()
