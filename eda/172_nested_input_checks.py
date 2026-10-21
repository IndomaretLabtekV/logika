"""CPU checks for bringing partition-specific teammate features into Kumo.

No model fitting, real-data feature build, GPU or outer-score access.
"""
import json
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import nested_input_pools as pools
except ModuleNotFoundError:pools=None


def fixture():
    rng=np.random.default_rng(172)
    ids=np.array([f"A{i:03}" for i in range(80)])
    y=np.tile([0,1],40)
    frame=pd.DataFrame({"application_id":ids,"customer_id":[f"C{i:03}" for i in range(80)],
                        "split":["train"]*48+["test"]*32,"target":np.r_[y[:48],np.full(32,np.nan)],
                        "decision_datetime":pd.to_datetime(["2025-06-01"]*80),
                        "loan_purpose":["consumer","business"]*40,"region":["east","west"]*40,
                        "employment_type":["employee","owner"]*40,"housing_status":["owned","rented"]*40,
                        "v2_prev_last_decision":["approved","rejected"]*40,
                        "v2_ecom_last_payment_method":["card","transfer"]*40})
    for i in range(24):frame[f"x{i}"]=rng.normal(size=80)
    frame.loc[2,"x3"]=np.nan
    return frame,ids,y,ids[:48].tolist()


class InputChecks(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(pools,"Missing nested Kumo pool adapter")

    def test_aligns_ids_and_excludes_metadata_and_extra_categories(self):
        frame,ids,y,fit=fixture()
        X=pools.teammate_inputs(frame.sample(frac=1,random_state=7),ids,y,fit)
        np.testing.assert_array_equal(X.x0.to_numpy(),frame.x0.to_numpy())
        self.assertEqual(set(X.columns),{f"x{i}" for i in range(24)}|set(pools.APPLICATION_CATEGORIES))
        self.assertNotIn("target",X);self.assertNotIn("customer_id",X)
        self.assertNotIn("v2_prev_last_decision",X);self.assertEqual(len(X),80)

    def test_rejects_duplicate_or_missing_query_ids(self):
        frame,ids,y,fit=fixture()
        for changed in (pd.concat([frame,frame.iloc[[0]]],ignore_index=True),frame.iloc[:-1]):
            with self.assertRaises(ValueError):pools.teammate_inputs(changed,ids,y,fit)

    def test_rejects_wrong_reference_membership_or_visible_query_targets(self):
        frame,ids,y,fit=fixture()
        changed=frame.copy();changed.loc[79,"target"]=1.
        with self.assertRaises(ValueError):pools.teammate_inputs(changed,ids,y,fit)
        with self.assertRaises(ValueError):pools.teammate_inputs(frame,ids,y,fit+[ids[-1]])
        with self.assertRaises(ValueError):pools.teammate_inputs(frame,ids,y,fit+[fit[0]])

    def test_rejects_reference_label_mismatch(self):
        frame,ids,y,fit=fixture();frame.loc[0,"target"]=1
        with self.assertRaises(ValueError):pools.teammate_inputs(frame,ids,y,fit)

    def test_selection_ignores_query_labels_and_covariates(self):
        frame,ids,y,fit=fixture();X=pools.teammate_inputs(frame,ids,y,fit)
        selected=pools.select_columns(X,y,np.arange(48),n_numeric=6)
        mutated=X.copy();mutated.loc[48:,[f"x{i}" for i in range(24)]]=1e12
        y2=y.copy();y2[48:]=1-y2[48:]
        self.assertEqual(selected,pools.select_columns(mutated,y2,np.arange(48),n_numeric=6))
        self.assertEqual(len(selected),10)
        self.assertEqual(selected[-4:],list(pools.APPLICATION_CATEGORIES))

    def test_stage_keeps_query_labels_out_and_fits_category_vocabulary_on_context(self):
        frame,ids,y,fit=fixture();X=pools.teammate_inputs(frame,ids,y,fit)
        X.loc[48,"region"]="query_only"
        cols=pools.select_columns(X,y,np.arange(48),n_numeric=6)
        stage=pools.prepare_stage(X,y,ids,np.arange(48),{"search":np.arange(48,64)},cols,"native")
        self.assertEqual(len(stage["frame"]),64)
        np.testing.assert_array_equal(stage["labels"][:48],y[:48])
        np.testing.assert_array_equal(stage["labels"][48:],np.zeros(16,dtype=int))
        self.assertTrue(pd.isna(stage["frame"].region.iloc[48]))
        self.assertNotIn("query_only",stage["frame"].region.cat.categories)
        self.assertEqual(stage["ids"].tolist(),ids[:64].tolist())
        self.assertEqual(stage["queries"]["search"].tolist(),list(range(48,64)))

    def test_stage_rejects_overlap_and_unknown_representation(self):
        frame,ids,y,fit=fixture();X=pools.teammate_inputs(frame,ids,y,fit)
        cols=pools.select_columns(X,y,np.arange(48),n_numeric=6)
        with self.assertRaises(ValueError):pools.prepare_stage(X,y,ids,np.arange(48),{"search":np.arange(40,50)},cols,"raw")
        with self.assertRaises(ValueError):pools.prepare_stage(X,y,ids,np.arange(48),{"search":np.arange(48,64)},cols,"mystery")

    def test_expanded_shortlist_reaches_budget_without_changing_filled_baseline(self):
        frame,ids,y,fit=fixture();X=pools.teammate_inputs(frame,ids,y,fit)
        # Exact correlated copies occupy the first shortlist. A weaker independent
        # feature is deliberately below it; extending candidates must recover it.
        rng=np.random.default_rng(1720)
        signal=y.astype(float)+rng.normal(0,.01,len(y))
        Z=X[list(pools.APPLICATION_CATEGORIES)].copy()
        for i in range(8):Z[f"duplicate{i}"]=signal
        for i in range(16):Z[f"other{i}"]=rng.normal(size=len(y))
        legacy=pools.select_columns(Z,y,np.arange(48),n_numeric=4)
        self.assertLess(len(legacy),8)
        expanded=pools.select_columns(Z,y,np.arange(48),n_numeric=4,extend_shortlist=True)
        self.assertEqual(len(expanded),8)
        normal=pools.select_columns(X,y,np.arange(48),n_numeric=6)
        self.assertEqual(normal,pools.select_columns(X,y,np.arange(48),n_numeric=6,extend_shortlist=True))


if __name__=="__main__":
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(InputChecks))
    out=Path(__file__).resolve().parent/"out"
    report={"checks":r.testsRun,"failures":len(r.failures),"errors":len(r.errors),"model_executed":False,"real_data_used":False}
    (out/"172_nested_input_checks.json").write_text(json.dumps(report,indent=2))
    fig,ax=plt.subplots(figsize=(7,3));ax.bar(["pass","fail/error"],[r.testsRun-len(r.failures)-len(r.errors),len(r.failures)+len(r.errors)])
    ax.set(ylabel="CPU checks",title="Nested input adapter; no model-quality claim");fig.tight_layout();fig.savefig(out/"172_nested_input_checks.png",dpi=130)
    raise SystemExit(not r.wasSuccessful())
