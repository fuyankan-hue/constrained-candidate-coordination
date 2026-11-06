import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from ccc.core.rcim_womd import Model
from ccc.core.rcim_av2 import RCIM
from ccc.core.refiner_womd import Refiner
from ccc.core import solvers,av2_ops
from ccc.io import config

class FrozenContracts(unittest.TestCase):
    def test_architecture_and_candidate_exchange(self):
        torch.manual_seed(42)
        for m,T in [(Model('refine_only'),80),(RCIM(),60)]:
            self.assertEqual(sum(p.numel() for p in m.parameters()),44449)
            features=[torch.randn(2,6,128) for _ in range(3)];world=torch.randn(2,6,T,2)
            with torch.no_grad():
                a=m(*features,world,torch.tensor([[0,1]]))
                b=m(*features,world,torch.tensor([[1,0]]))
            torch.testing.assert_close(a,b.transpose(1,2),rtol=1e-5,atol=1e-6)
        self.assertEqual(sum(p.numel() for p in Refiner(False).parameters()),76026)

    def test_projection_whole_path_and_score_anchors(self):
        torch.manual_seed(7)
        world=torch.randn(2,6,60,2)*20;world[:,0]=0
        pi=np.arange(12,dtype=float).reshape(2,6);C=torch.randn(1,6,6)
        row={'m1':{'world':world}}
        recipe=dict(base=[0,0],pi=pi,edges=[(0,1)],C_old=C,allowed=np.ones((2,6),bool))
        previous=solvers.project.__globals__['tables']
        solvers.project.__globals__['tables']=av2_ops.point_tables
        try:vp,rp,alpha=solvers.project(row,recipe)
        finally:solvers.project.__globals__['tables']=previous
        self.assertLessEqual(float(vp['m1']['world'].norm(dim=-1).max()),4.)
        np.testing.assert_array_equal(vp['m1']['world'][:,0],world[:,0])
        self.assertEqual(float(rp['C_old'][0,0,0]),float(C[0,0,0]))
        self.assertTrue(np.isfinite(rp['pi']).all())

    def test_configuration_and_split_isolation(self):
        root=Path(__file__).resolve().parents[1]
        for name in ('womd','av2'):
            cfg=config(root/('configs/'+name+'.json'))
            self.assertEqual(cfg['refinement']['optimizer_steps_each'],5120)
        import json
        from ccc.splits import canonical_sha,validate_document
        rules=json.loads((root/'configs/split_rules.json').read_text())
        self.assertEqual(rules['fingerprints']['womd_refiners.json']['counts'],{'train':8192,'tune':1024})
        self.assertFalse((root/'configs/splits').exists())
        synthetic={'train':['synthetic_scene_a'],'tune':['synthetic_scene_b']}
        local_rules={'fingerprints':{'example.json':{'canonical_SHA256':canonical_sha(synthetic)}}}
        validate_document('example.json',synthetic,local_rules)
        with self.assertRaises(AssertionError):
            validate_document('example.json',{'train':['synthetic_scene_b'],'tune':['synthetic_scene_a']},local_rules)


if __name__=='__main__':unittest.main()
