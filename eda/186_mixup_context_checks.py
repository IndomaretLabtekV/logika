"""CPU invariants for labeled-training context augmentation, never query parents."""
import json
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:import mixup_context as m
except ModuleNotFoundError:m=None


def fixture():
    rng=np.random.default_rng(186);n=128
    X=pd.DataFrame({"amount":rng.normal(10,2,n),"ratio":rng.normal(size=n),"binary":np.tile([0,1],64),
                    "rp_frac_dpd1":np.r_[np.full(32,np.nan),rng.uniform(size=96)]})
    for c in ("loan_purpose","region","employment_type","housing_status"):X[c]="same"
    y=np.tile([0,0,1,0],32);ids=np.array([f"A{i:04}" for i in range(n)])
    return X,y,ids,np.arange(96)


class Checks(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(m,"Same-class mixup helper missing")

    def test_query_covariates_and_labels_cannot_change_augmentation(self):
        X,y,ids,fit=fixture();a,pa,ma=m.augment(X,y,ids,fit,list(X.columns),32,4200,"test")
        changed=X.copy();changed.loc[96:,["amount","ratio"]]=1e12;y2=y.copy();y2[96:]=1-y2[96:]
        b,pb,mb=m.augment(changed,y2,ids,fit,list(X.columns),32,4200,"test")
        pd.testing.assert_frame_equal(a,b);pd.testing.assert_frame_equal(pa,pb);self.assertEqual(ma,mb)

    def test_parents_classes_categories_masks_binary_and_class_count(self):
        X,y,ids,fit=fixture();out,parents,meta=m.augment(X,y,ids,fit,list(X.columns),32,4200,"test")
        lookup={key:i for i,key in enumerate(ids)}
        self.assertEqual(len(out),32);self.assertEqual(int(parents.label.sum()),8)
        self.assertEqual(parents.synthetic_id.nunique(),32);self.assertTrue(set(parents.synthetic_id).isdisjoint(ids))
        for i,row in parents.iterrows():
            a,b=lookup[row.parent_a],lookup[row.parent_b]
            self.assertIn(a,fit);self.assertIn(b,fit);self.assertEqual(y[a],y[b]);self.assertEqual(row.label,y[a])
            self.assertEqual(pd.isna(X.rp_frac_dpd1.iloc[a]),pd.isna(X.rp_frac_dpd1.iloc[b]))
            np.testing.assert_array_equal(out.iloc[i].isna(),X.iloc[a].isna())
            self.assertEqual(out.binary.iloc[i],X.binary.iloc[a])
            for c in m.CATEGORIES:self.assertEqual(out[c].iloc[i],X[c].iloc[a])

    def test_clone_control_is_exact_anchor_copy(self):
        X,y,ids,fit=fixture();out,parents,_=m.augment(X,y,ids,fit,list(X.columns),32,4200,"control",clone=True)
        expected=X.set_axis(ids).loc[parents.parent_a].reset_index(drop=True)
        pd.testing.assert_frame_equal(out,expected)

    def test_rejects_duplicate_fit_ids_or_excessive_cap(self):
        X,y,ids,fit=fixture()
        for rows,cap in ((np.r_[fit,fit[0]],32),(fit,100)):
            with self.assertRaises(ValueError):m.augment(X,y,ids,rows,list(X.columns),cap,4200,"bad")


if __name__=="__main__":
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out=Path(__file__).resolve().parent/"out"
    (out/"186_mixup_context_checks.json").write_text(json.dumps({"checks":r.testsRun,"failures":len(r.failures),"errors":len(r.errors),"model_executed":False},indent=2))
    fig,ax=plt.subplots(figsize=(7,3));ax.bar(["pass","fail/error"],[r.testsRun-len(r.failures)-len(r.errors),len(r.failures)+len(r.errors)])
    ax.set(ylabel="CPU checks",title="Same-class context augmentation invariants")
    fig.tight_layout();fig.savefig(out/"186_mixup_context_checks.png",dpi=130)
    raise SystemExit(not r.wasSuccessful())
