"""CPU regression check: positive-class probability follows its column name."""
import importlib.util
import json
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sdm

source=Path(__file__).with_name("187_kumo_sgd_preflight.py");module=None
if source.exists():
    spec=importlib.util.spec_from_file_location("sgd_preflight",source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Checks(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(module,"SGD preflight module missing")

    def test_column_permutation_cannot_invert_positive_probability(self):
        for columns in (["0","1"],["1","0"]):
            frame=pd.DataFrame({"0":[.8,.3],"1":[.2,.7]})[columns]
            table=sdm.TableTensor.from_pandas(frame,stypes={c:"numerical" for c in columns},device="cpu")
            np.testing.assert_allclose(module.positive_probability(table).numpy(),[.2,.7])

    def test_missing_positive_column_is_rejected(self):
        table=sdm.TableTensor.from_pandas(pd.DataFrame({"0":[1.]}),stypes={"0":"numerical"},device="cpu")
        with self.assertRaises(ValueError):module.positive_probability(table)


if __name__=="__main__":
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out=Path(__file__).resolve().parent/"out"
    (out/"188_sgd_class_mapping_checks.json").write_text(json.dumps({"checks":r.testsRun,"failures":len(r.failures),"errors":len(r.errors),"gpu_used":False},indent=2))
    fig,ax=plt.subplots(figsize=(7,3));ax.bar(["pass","fail/error"],[r.testsRun-len(r.failures)-len(r.errors),len(r.failures)+len(r.errors)])
    ax.set(ylabel="CPU checks",title="Class-name mapping before Kumo gradient preflight")
    fig.tight_layout();fig.savefig(out/"188_sgd_class_mapping_checks.png",dpi=130)
    raise SystemExit(not r.wasSuccessful())
