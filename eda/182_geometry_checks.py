"""CPU checks of monetary transformation and memory-preflight quality gating."""
import importlib.util
import json
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

source=Path(__file__).with_name("181_kumo_input_geometry.py");geometry=None
if source.exists():
    spec=importlib.util.spec_from_file_location("geometry",source)
    geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)


class Checks(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(geometry,"Geometry runner missing")

    def test_signed_log_preserves_missing_sign_and_unselected_columns(self):
        original=pd.DataFrame({"money":[-100.,0.,100.,np.nan],"ratio":[.1,.2,.3,.4],"category":pd.Categorical(["a","b","a","b"])})
        out=geometry.apply_money(original,["money"])
        np.testing.assert_allclose(out.money.to_numpy(),[-np.log(101),0,np.log(101),np.nan],equal_nan=True)
        pd.testing.assert_series_equal(out.ratio,original.ratio);pd.testing.assert_series_equal(out.category,original.category)
        self.assertEqual(original.money.iloc[2],100.)

    def test_unknown_monetary_column_is_rejected(self):
        with self.assertRaises(ValueError):geometry.apply_money(pd.DataFrame({"money":[1.]}),["other"])

    def test_only_terminal_same_run_preflight_enables_full_features(self):
        self.assertTrue(geometry.full_feasible({"run_sha256":"abc","status":"passed"},"abc"))
        self.assertFalse(geometry.full_feasible({"run_sha256":"abc","status":"oom"},"abc"))
        for report in (None,{"run_sha256":"abc","status":"running"},{"run_sha256":"different","status":"passed"}):
            with self.assertRaises(ValueError):geometry.full_feasible(report,"abc")


if __name__=="__main__":
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out=Path(__file__).resolve().parent/"out"
    (out/"182_geometry_checks.json").write_text(json.dumps({"checks":r.testsRun,"failures":len(r.failures),"errors":len(r.errors),"model_executed":False},indent=2))
    fig,ax=plt.subplots(figsize=(7,3));ax.bar(["pass","fail/error"],[r.testsRun-len(r.failures)-len(r.errors),len(r.failures)+len(r.errors)])
    ax.set(ylabel="CPU checks",title="Money transform and terminal memory-preflight gate")
    fig.tight_layout();fig.savefig(out/"182_geometry_checks.png",dpi=130)
    raise SystemExit(not r.wasSuccessful())
