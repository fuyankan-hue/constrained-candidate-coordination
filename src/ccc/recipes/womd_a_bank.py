"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import torch

def bankA(r):
 a=dict(r);a['inputs']=dict(r['inputs']);a['evaluation']=dict(r['evaluation'])
 for k in ('path_local','path_scene','refine','prob','pair_motion'):a['inputs'][k]=r['inputs'][k][[0,0]].clone()
 a['evaluation']['base_world']=r['evaluation']['base_world'][[0,0]].clone();a['evaluation']['modes']=[r['evaluation']['modes'][0]]*2
 return a

