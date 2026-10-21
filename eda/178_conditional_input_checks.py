"""CPU checks for conditional selection and canonical union inputs."""
import json
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:import conditional_input_pools as pools
except ModuleNotFoundError:pools=None


def fixture():
    rng=np.random.default_rng(178);n=512;y=np.tile([0,1],n//2)
    X=pd.DataFrame({c:["a","b"]*(n//2) for c in ("loan_purpose","region","employment_type","housing_status")})
    for i in range(20):X[f"x{i}"]=rng.normal(size=n)+(y if i==0 else 0)
    return X,y,np.arange(384)


class Checks(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(pools,"Conditional input adapter missing")

    def test_union_preserves_main_and_prefixes_all_friend_numeric(self):
        X,y,fit=fixture();other=X.copy();other.x0*=2
        union=pools.union_inputs(X,other)
        self.assertEqual(len(union.columns),44)
        np.testing.assert_array_equal(union.x0,X.x0)
        np.testing.assert_array_equal(union.teman__x0,other.x0)
        self.assertEqual(sum(c=="region" for c in union.columns),1)

    def test_union_rejects_category_misalignment(self):
        X,y,fit=fixture();other=X.copy();other.loc[0,"region"]="different"
        with self.assertRaises(ValueError):pools.union_inputs(X,other)
        with self.assertRaises(ValueError):pools.union_inputs(X,other.iloc[:-1])

    def test_conditional_selection_uses_only_real_context(self):
        X,y,fit=fixture();cols,gain,info=pools.gain_columns(X,y,fit,seed=4200,n_numeric=6)
        changed=X.copy();changed.loc[384:,[f"x{i}" for i in range(20)]]=1e12
        y2=y.copy();y2[384:]=1-y2[384:]
        cols2,gain2,info2=pools.gain_columns(changed,y2,fit,seed=4200,n_numeric=6)
        self.assertEqual(cols,cols2);np.testing.assert_array_equal(gain.to_numpy(),gain2.to_numpy())
        self.assertEqual(info["booster_sha256"],info2["booster_sha256"])
        self.assertEqual(len(cols),10);self.assertIn("x0",cols)

    def test_rejects_duplicate_or_out_of_bounds_context(self):
        X,y,fit=fixture()
        for rows in ([0,0,1],[0,512]):
            with self.assertRaises(ValueError):pools.gain_columns(X,y,rows,seed=4200,n_numeric=6)


if __name__=="__main__":
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out=Path(__file__).resolve().parent/"out";report={"checks":r.testsRun,"failures":len(r.failures),"errors":len(r.errors),"kumo_executed":False}
    (out/"178_conditional_input_checks.json").write_text(json.dumps(report,indent=2))
    fig,ax=plt.subplots(figsize=(7,3));ax.bar(["pass","fail/error"],[r.testsRun-len(r.failures)-len(r.errors),len(r.failures)+len(r.errors)])
    ax.set(ylabel="CPU checks",title="Conditional Kumo input preparation; no quality evidence")
    fig.tight_layout();fig.savefig(out/"178_conditional_input_checks.png",dpi=130)
    raise SystemExit(not r.wasSuccessful())
