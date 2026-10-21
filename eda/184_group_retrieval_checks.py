"""CPU checks that grouped neighborhoods use fitting covariates only."""
import json
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:import grouped_context as g
except ModuleNotFoundError:g=None


def fixture():
    rng=np.random.default_rng(184);X=pd.DataFrame({f"x{i}":rng.normal(size=128) for i in range(24)})
    for c in ("loan_purpose","region","employment_type","housing_status"):X[c]=["a","b"]*64
    X.loc[3,"x0"]=np.nan
    return X,np.arange(96),np.arange(96,128)


class Checks(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(g,"Grouped retrieval adapter missing")

    def test_contexts_unchanged_by_query_covariates_and_unknown_categories(self):
        X,fit,q=fixture();a=g.neighborhoods(X,fit,q,list(X.columns),seed=4200,n_clusters=4,n_context=32)
        changed=X.copy();changed.loc[q,[f"x{i}" for i in range(24)]]=1e12;changed.loc[q,"region"]="query_only"
        b=g.neighborhoods(changed,fit,q,list(X.columns),seed=4200,n_clusters=4,n_context=32)
        for key in a["contexts"]:np.testing.assert_array_equal(a["contexts"][key],b["contexts"][key])
        np.testing.assert_array_equal(a["centers"],b["centers"])
        self.assertEqual(a["geometry_sha256"],b["geometry_sha256"])
        self.assertEqual(b["unknown_query_categories"]["region"],len(q))

    def test_all_context_rows_are_fitting_and_query_coverage_is_exact(self):
        X,fit,q=fixture();a=g.neighborhoods(X,fit,q,list(X.columns),seed=4200,n_clusters=4,n_context=32)
        self.assertEqual(len(a["assignment"]),len(q));self.assertTrue(set(a["assignment"])<=set(a["contexts"]))
        for rows in a["contexts"].values():
            self.assertEqual(len(rows),32);self.assertEqual(len(set(rows)),32)
            self.assertTrue(set(rows)<=set(fit));self.assertTrue(set(rows).isdisjoint(q))
            np.testing.assert_array_equal(rows,np.sort(rows))

    def test_invalid_context_query_membership_is_rejected(self):
        X,fit,q=fixture()
        for badfit,badq in ((np.r_[fit,fit[0]],q),(fit,np.r_[q,fit[0]]),(fit,np.array([128]))):
            with self.assertRaises(ValueError):g.neighborhoods(X,badfit,badq,list(X.columns),seed=4200,n_clusters=4,n_context=32)


if __name__=="__main__":
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out=Path(__file__).resolve().parent/"out"
    (out/"184_group_retrieval_checks.json").write_text(json.dumps({"checks":r.testsRun,"failures":len(r.failures),"errors":len(r.errors),"model_executed":False},indent=2))
    fig,ax=plt.subplots(figsize=(7,3));ax.bar(["pass","fail/error"],[r.testsRun-len(r.failures)-len(r.errors),len(r.failures)+len(r.errors)])
    ax.set(ylabel="CPU checks",title="Grouped context: fitting-only geometry and exact query coverage")
    fig.tight_layout();fig.savefig(out/"184_group_retrieval_checks.png",dpi=130)
    raise SystemExit(not r.wasSuccessful())
