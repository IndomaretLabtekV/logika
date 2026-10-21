"""CPU checks of fixed rank fusion and its matched rank-normalized baseline."""
import importlib.util
import json
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

source=Path(__file__).with_name("189_fixed_kumo_union_rank.py");module=None
if source.exists():
    spec=importlib.util.spec_from_file_location("fixed_fusion",source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Checks(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(module,"Fixed outer evaluator missing")

    def test_exact_ties_and_matched_baseline(self):
        out=module.fuse(np.array([.2,.2,.4,.8],dtype=np.float32),np.array([.8,.4,.2,.2],dtype=np.float32))
        np.testing.assert_array_equal(out["baseline"],[.375,.375,.75,1.])
        np.testing.assert_array_equal(out["union_rank"],[1.,.75,.375,.375])
        np.testing.assert_array_equal(out["candidate"],[.6875,.5625,.5625,.6875])
        self.assertEqual(out["candidate"].dtype,np.float64)

    def test_row_permutation_does_not_change_scores(self):
        a=np.array([.1,.7,.4,.9]);b=np.array([.8,.1,.7,.4]);perm=np.array([2,0,3,1])
        original=module.fuse(a,b);changed=module.fuse(a[perm],b[perm])
        for key in original:np.testing.assert_array_equal(changed[key][np.argsort(perm)],original[key])

    def test_invalid_predictions_fail_before_fusion(self):
        for a,b in (([.1],[.1,.2]),([.1,np.nan],[.2,.3]),([-.1,.2],[.3,.4])):
            with self.assertRaises(ValueError):module.fuse(np.array(a),np.array(b))


if __name__=="__main__":
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out=Path(__file__).resolve().parent/"out"
    (out/"190_fixed_fusion_checks.json").write_text(json.dumps({"checks":r.testsRun,"failures":len(r.failures),"errors":len(r.errors),"model_executed":False},indent=2))
    fig,ax=plt.subplots(figsize=(7,3));ax.bar(["pass","fail/error"],[r.testsRun-len(r.failures)-len(r.errors),len(r.failures)+len(r.errors)])
    ax.set(ylabel="CPU checks",title="Fixed fusion: tie, calibration and alignment invariants")
    fig.tight_layout();fig.savefig(out/"190_fixed_fusion_checks.png",dpi=130)
    raise SystemExit(not r.wasSuccessful())
