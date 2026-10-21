"""Synthetic checks of real teammate feature builders under nested reference IDs.

No ML fitting. Runs original V3/V4/V5 feature functions on temporary CSV/parquet
fixtures and tests whether query data/labels can alter context features.
"""
from pathlib import Path
import tempfile
import unittest
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import teammate_feature_adapter as adapter
except ModuleNotFoundError:
    adapter=None


def fixture(root,query_price=300.,query_dpd=90.,query_target=1.):
    (root/"data/processed").mkdir(parents=True)
    (root/"artifacts/optimization").mkdir(parents=True)
    base=pd.DataFrame({"application_id":["A_F","A_Q"],"customer_id":["F","Q"],
        "decision_datetime":pd.to_datetime(["2025-06-15"]*2),"split":["train"]*2,"target":[0.,query_target],
        "declared_monthly_income":[1000.,2000.],"proposed_monthly_installment":[100.,200.],"dependents":[1.,2.],
        "v2_account_3m_income_like_credits_mean":[1200.,2400.],"v2_account_3m_outflows_mean":[300.,600.],
        "v2_existing_active_installment":[50.,100.],"v2_account_3m_minimum_balance_min":[50.,100.],
        "v2_account_3m_average_balance_mean":[500.,1000.]})
    events=[];repay=[];bureau=[]
    for c,price,dpd in [("F",100.,0.),("Q",query_price,query_dpd)]:
        for j in range(3):
            events.append({"customer_id":c,"order_id":f"{c}_{j}","order_item_id":f"{c}_{j}_0",
                           "transaction_datetime":f"2025-0{j+1}-05","item_total":price*(j+1),"order_total":price*(j+1),
                           "quantity":1,"order_status":"completed","product_category":"groceries"})
            repay.append({"customer_id":c,"loan_id":f"{c}_L","due_date":f"2024-0{j+1}-01",
                          "last_payment_datetime":f"2024-0{j+1}-05","max_days_past_due":dpd,
                          "amount_due":100.,"amount_paid":100.})
        for provider,score in [("Bureau_A",600.),("Bureau_B",650.)]:
            bureau.append({"customer_id":c,"bureau_provider":provider,"score":score,
                           "report_datetime":"2025-01-01","available_datetime":"2025-01-02"})
    pd.DataFrame(events).to_csv(root/"data/ecommerce_transactions.csv",index=False)
    pd.DataFrame(repay).to_csv(root/"data/repayment_history.csv",index=False)
    b=pd.DataFrame(bureau)
    b.to_csv(root/"data/bureau_observations.csv",index=False)
    b.assign(report_datetime=pd.to_datetime(b.report_datetime),available_datetime=pd.to_datetime(b.available_datetime)).to_parquet(
        root/"data/processed/bureau_latest.parquet",index=False)
    return base


class ReferenceChecks(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(adapter,"Missing adapter for nested teammate features")

    def test_masks_query_labels_and_preserves_context_target(self):
        base=pd.DataFrame({"application_id":["A","B","T"],"customer_id":["a","b","t"],
                           "split":["train","train","test"],"target":[0.,1.,np.nan]})
        masked=adapter.mask_reference_population(base,["A"])
        self.assertEqual(masked.split.tolist(),["train","test","test"])
        self.assertEqual(masked.target.iloc[0],0.)
        self.assertTrue(masked.target.iloc[1:].isna().all())
        self.assertEqual(base.split.tolist(),["train","train","test"])

    def test_rejects_unknown_or_organizer_test_reference_ids(self):
        base=pd.DataFrame({"application_id":["A","T"],"customer_id":["a","t"],
                           "split":["train","test"],"target":[0.,np.nan]})
        for ids in (["MISSING"],["T"],["A","A"],[]):
            with self.assertRaises(ValueError):adapter.mask_reference_population(base,ids)

    def test_real_builders_ignore_query_values_when_fitting_reference(self):
        outputs=[]
        with tempfile.TemporaryDirectory(prefix="logika-reference-check-") as td:
            for i,(price,dpd,target) in enumerate([(300.,90.,1.),(300.,90.,0.),(30000.,0.,0.)]):
                work=Path(td)/str(i);base=fixture(work,price,dpd,target)
                outputs.append(adapter.run_reference_stages(work,base,["A_F"]))
        for frame in outputs:
            self.assertEqual(frame.application_id.tolist(),["A_F","A_Q"])
            self.assertEqual(frame.split.tolist(),["train","test"])
            self.assertTrue(np.isnan(frame.target.iloc[1]))
            # All late events of the sole reference customer are zero. Query's
            # late events MUST NOT influence the prior used for context or query.
            self.assertEqual(frame.v3_repay_all_late_1_posterior_24.iloc[0],0.)
        cols=[c for c in outputs[0] if c.startswith(("v3_","v4_","v5_"))]
        pd.testing.assert_frame_equal(outputs[0],outputs[1])
        pd.testing.assert_frame_equal(outputs[0].iloc[[0]][cols].reset_index(drop=True),
                                      outputs[2].iloc[[0]][cols].reset_index(drop=True))
        self.assertNotEqual(outputs[0].v5_price_pooled_all_mean.iloc[1],outputs[2].v5_price_pooled_all_mean.iloc[1])

    def test_price_reference_uses_fitting_customer_median(self):
        with tempfile.TemporaryDirectory(prefix="logika-reference-check-") as td:
            work=Path(td);base=fixture(work)
            frame=adapter.run_reference_stages(work,base,["A_F"])
        ref=np.log1p(200.)
        expected=float(np.mean(np.log1p([100.,200.,300.]))-ref)
        self.assertAlmostEqual(frame.v5_price_pooled_all_mean.iloc[0],expected,places=12)
        q_expected=float(np.mean(np.log1p([300.,600.,900.]))-ref)
        self.assertAlmostEqual(frame.v5_price_pooled_all_mean.iloc[1],q_expected,places=12)

    def test_real_builders_reject_observations_after_cutoff(self):
        with tempfile.TemporaryDirectory(prefix="logika-reference-check-") as td:
            work=Path(td);base=fixture(work)
            path=work/"data/ecommerce_transactions.csv"
            events=pd.read_csv(path);events.loc[0,"transaction_datetime"]="2025-07-01";events.to_csv(path,index=False)
            with self.assertRaises(AssertionError):adapter.run_reference_stages(work,base,["A_F"])

    def test_raw_tables_are_linked_and_conflicts_are_not_overwritten(self):
        self.assertTrue(callable(getattr(adapter,"prepare_workspace",None)),"Missing raw-table workspace resolver")
        names=("train_applications","test_applications","customer_profiles","previous_applications",
               "repayment_history","account_behavior","bureau_observations","ecommerce_transactions","sample_submission")
        with tempfile.TemporaryDirectory(prefix="logika-source-links-") as td:
            raw=Path(td)/"raw";raw.mkdir()
            for name in names:(raw/f"{name}.csv").write_text("fixture\n")
            work=Path(td)/"work"
            adapter.prepare_workspace(work,raw)
            for name in names:
                p=work/"data"/f"{name}.csv"
                self.assertTrue(p.is_symlink());self.assertEqual(p.resolve(),raw/f"{name}.csv")
            conflict=work/"data/train_applications.csv"
            conflict.unlink();conflict.write_text("keep-existing\n")
            with self.assertRaises(ValueError):adapter.prepare_workspace(work,raw)
            self.assertEqual(conflict.read_text(),"keep-existing\n")

    def test_schema_retains_history_categories_and_excludes_metadata(self):
        self.assertTrue(callable(getattr(adapter,"feature_schema",None)),"Missing teammate feature schema")
        frame=pd.DataFrame({"application_id":["A","B"],"customer_id":["a","b"],
                            "target":[0.,np.nan],"split":["train","test"],
                            "decision_datetime":pd.to_datetime(["2025-06-01"]*2),
                            "loan_purpose":["consumer","business"],"region":["X","Y"],
                            "employment_type":["employee",None],"housing_status":["owned","rented"],
                            "v2_prev_last_decision":["approved",None],
                            "v2_ecom_last_payment_method":pd.Categorical(["transfer","card"]),
                            "money":[10.,np.nan],"has_history":[True,False]})
        schema=adapter.feature_schema(frame)
        self.assertEqual(set(schema["categorical"]),set(frame.columns)-
                         {"application_id","customer_id","target","split","decision_datetime","money","has_history"})
        self.assertEqual(schema["numeric"],["money","has_history"])
        self.assertEqual(set(schema["excluded"]),{"application_id","customer_id","target","split","decision_datetime"})
        with self.assertRaises(ValueError):adapter.feature_schema(frame.assign(unexpected_text=["x","y"]))
        with self.assertRaises(ValueError):adapter.feature_schema(frame.assign(unexpected_date=pd.to_datetime(["2025-01-01"]*2)))


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceChecks))
    out=Path(__file__).resolve().parent/"out"
    report={"checks":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"real_data_used":False,"model_executed":False}
    (out/"165_teammate_reference_checks.json").write_text(json.dumps(report,indent=2))
    fig,ax=plt.subplots(figsize=(6,3));ax.bar(["passed","failed"],[result.testsRun-len(result.failures)-len(result.errors),len(result.failures)+len(result.errors)])
    ax.set(ylabel="Reference isolation checks",title="Original V3/V4/V5 builders on synthetic fixtures")
    fig.tight_layout();fig.savefig(out/"165_teammate_reference_checks.png",dpi=120);plt.close(fig)
    raise SystemExit(0 if result.wasSuccessful() else 1)
