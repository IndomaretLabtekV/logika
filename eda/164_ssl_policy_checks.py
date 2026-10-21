"""CPU checks of SSL selection policies; no model execution or real pseudo labels."""
import ast
import json
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import ssl_policies as policy
except ModuleNotFoundError:
    policy=None


class PolicyChecks(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(policy,"Missing isolated SSL policy implementation")
        self.p=np.array([.001,.019,.021,.049,.05,.5,.699,.7,.9,.949,.95,.98,.999])

    def test_symmetric_legacy_threshold_preserved(self):
        selected=policy.pseudo_rows(self.p,{"kind":"confidence","threshold":.05,"cap":100},7)
        np.testing.assert_array_equal(selected,[0,1,2,3,4,10,11,12])

    def test_default_policies_match_archived_selector(self):
        archive=Path(__file__).resolve().parents[1]/"research/locked_recipes/156_kumo_ssl_context_pre_class_threshold.py"
        tree=ast.parse(archive.read_text())
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {"policy_name","group_ids","pseudo_rows"}]
        legacy={"np":np}
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(archive),"exec"),legacy)
        probs=np.random.default_rng(2718).uniform(size=1003)
        specs=[{"kind":"none","cap":0},
               {"kind":"confidence","threshold":.02,"cap":20},
               {"kind":"confidence","threshold":.05,"cap":2000},
               {"kind":"groups","groups":[0,2,3],"cap":30}]
        for spec in specs:
            for seed in (42,7,2026):
                np.testing.assert_array_equal(policy.pseudo_rows(probs,spec,seed),legacy["pseudo_rows"](probs,spec,seed))
                self.assertEqual(policy.policy_name(spec),legacy["policy_name"](spec))

    def test_asymmetric_threshold_accepts_positive_boundary(self):
        spec={"kind":"asymmetric","negative_threshold":.02,"positive_threshold":.7,"cap":100}
        selected=policy.pseudo_rows(self.p,spec,7)
        np.testing.assert_array_equal(selected,[0,1,7,8,9,10,11,12])
        self.assertEqual(policy.policy_name(spec),"asym_n0.02_p0.7_cap100")

    def test_seeded_cap_is_unique_sorted_and_from_eligible(self):
        spec={"kind":"asymmetric","negative_threshold":.02,"positive_threshold":.7,"cap":3}
        a=policy.pseudo_rows(self.p,spec,19);b=policy.pseudo_rows(self.p,spec,19)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(len(a),3);self.assertEqual(len(set(a)),3)
        self.assertTrue(np.all(np.diff(a)>0));self.assertTrue(set(a)<=set([0,1,7,8,9,10,11,12]))

    def test_no_ssl_and_empty_selection(self):
        self.assertEqual(len(policy.pseudo_rows(self.p,{"kind":"none","cap":0},0)),0)
        spec={"kind":"asymmetric","negative_threshold":.001,"positive_threshold":.999,"cap":30}
        self.assertEqual(len(policy.pseudo_rows(np.array([.2,.4,.6,.8]),spec,0)),0)

    def test_confidence_shells_legacy_boundaries(self):
        np.testing.assert_array_equal(policy.group_ids(np.array([.02,.05,.95,.98,.5])),[0,1,2,3,-1])
        selected=policy.pseudo_rows(self.p,{"kind":"groups","groups":[2,3],"cap":20},0)
        np.testing.assert_array_equal(selected,[10,11,12])

    def test_invalid_probability_and_overlapping_thresholds_rejected(self):
        spec={"kind":"asymmetric","negative_threshold":.8,"positive_threshold":.7,"cap":30}
        with self.assertRaises(ValueError):policy.pseudo_rows(self.p,spec,0)
        with self.assertRaises(ValueError):policy.pseudo_rows(np.array([np.nan]),{"kind":"none","cap":0},0)


if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PolicyChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=Path(__file__).resolve().parent/"out"
    report={"checks":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"model_executed":False}
    (out/"164_ssl_policy_checks.json").write_text(json.dumps(report,indent=2))
    fig,ax=plt.subplots(figsize=(6,3));ax.bar(["passed","failed"],[result.testsRun-len(result.failures)-len(result.errors),len(result.failures)+len(result.errors)])
    ax.set(ylabel="CPU policy checks",title="Synthetic fixture, not model-quality evidence")
    fig.tight_layout();fig.savefig(out/"164_ssl_policy_checks.png",dpi=120);plt.close(fig)
    raise SystemExit(0 if result.wasSuccessful() else 1)
