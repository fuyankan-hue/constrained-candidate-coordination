"""Copied from the frozen experiment; source mapping: docs/source_manifest.json."""
import numpy as np

def aggregate(records,name):
    ss={k:sum(z['methods'][name][k] for z in records) for k in ('targets','pairs','ade_sum','fde_sum','overlap','overlap_steps','gt5','changed','improved','degraded')}
    ds=np.array([d for z in records for d in z['methods'][name]['deltas']]);ss.update(ADE=ss['ade_sum']/max(ss['targets'],1),FDE=ss['fde_sum']/max(ss['targets'],1),P99_delta=float(np.quantile(ds,.99)) if len(ds) else None,worst_delta=float(ds.max()) if len(ds) else None)
    return ss

