"""CPU checks of search/confirmation gates and feature-cache provenance."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SOURCE=Path(__file__).with_name("173_nested_kumo_inputs.py")
runner=None
if SOURCE.exists():
    spec=importlib.util.spec_from_file_location("input_runner",SOURCE)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class RunnerChecks(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(runner,"Input runner has not been implemented")

    def test_search_gate_is_strict_and_stable_ties_prefer_baseline(self):
        base={"name":"utama_raw","ks":.5}
        self.assertIsNone(runner.challenger([base,{"name":"teman_raw","ks":.5}],.003))
        self.assertIsNone(runner.challenger([base,{"name":"teman_raw","ks":.503}],.003))
        self.assertEqual(runner.challenger([base,{"name":"teman_raw","ks":.504}],.003),"teman_raw")

    def test_confirmation_cannot_replace_search_challenger(self):
        search=[{"name":"utama_raw","ks":.5},{"name":"utama_native","ks":.51},
                {"name":"teman_raw","ks":.509},{"name":"teman_native","ks":.508}]
        confirmation=[{"name":"utama_raw","ks":.52},{"name":"utama_native","ks":.521}]
        self.assertEqual(runner.freeze_selection(search,confirmation,.003)["winner"],"utama_raw")
        with self.assertRaises(ValueError):
            runner.freeze_selection(search,[confirmation[0],{"name":"teman_raw","ks":.55}],.003)

    def test_only_both_independent_gates_allow_challenger(self):
        search=[{"name":"utama_raw","ks":.5},{"name":"teman_native","ks":.505}]
        confirmed=[{"name":"utama_raw","ks":.52},{"name":"teman_native","ks":.525}]
        self.assertEqual(runner.freeze_selection(search,confirmed,.003)["winner"],"teman_native")
        self.assertFalse(runner.freeze_selection(search,confirmed,.003)["outer_seen"])
        with self.assertRaises(ValueError):runner.freeze_selection(search,confirmed[:1],.003)

    def test_no_search_challenger_means_baseline_confirmation_only(self):
        search=[{"name":"utama_raw","ks":.5},{"name":"utama_native","ks":.499}]
        base=[{"name":"utama_raw","ks":.52}]
        self.assertEqual(runner.freeze_selection(search,base,.003)["winner"],"utama_raw")
        with self.assertRaises(ValueError):runner.freeze_selection(search,base+[{"name":"utama_native","ks":.6}],.003)

    def test_cache_rejects_wrong_population_and_modified_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory);output=work/"features.parquet"
            pd.DataFrame({"application_id":["A1","A2"]}).to_parquet(output,index=False)
            manifest={"status":"complete","inputs":{"partition":"inner","seed":42,"fold":0,
                      "fitting_application_ids":["A1"],"source_sha256":{}},
                      "output":"features.parquet","output_sha256":runner.file_hash(output)}
            manifest["inputs_sha256"]=runner.digest(manifest["inputs"])
            (work/"manifest.json").write_text(json.dumps(manifest))
            got,_=runner.read_pool(work,"inner",42,0,["A1"])
            self.assertEqual(got.application_id.tolist(),["A1","A2"])
            with self.assertRaises(ValueError):runner.read_pool(work,"inner",42,0,["A2"])
            with self.assertRaises(ValueError):runner.read_pool(work,"outer",42,0,["A1"])
            pd.DataFrame({"application_id":["changed"]}).to_parquet(output,index=False)
            with self.assertRaises(ValueError):runner.read_pool(work,"inner",42,0,["A1"])


if __name__=="__main__":
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RunnerChecks))
    out=Path(__file__).resolve().parent/"out"
    report={"checks":r.testsRun,"failures":len(r.failures),"errors":len(r.errors),"model_executed":False}
    (out/"174_input_runner_checks.json").write_text(json.dumps(report,indent=2))
    fig,ax=plt.subplots(figsize=(7,3));ax.bar(["pass","fail/error"],[r.testsRun-len(r.failures)-len(r.errors),len(r.failures)+len(r.errors)])
    ax.set(ylabel="CPU checks",title="Nested input policy gates and provenance")
    fig.tight_layout();fig.savefig(out/"174_input_runner_checks.png",dpi=130)
    raise SystemExit(not r.wasSuccessful())
