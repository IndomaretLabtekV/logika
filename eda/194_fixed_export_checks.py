"""CPU rejection checks for incomplete gates and malformed submission exports."""
import importlib.util
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

spec=importlib.util.spec_from_file_location("checked_export",Path(__file__).with_name("193_fixed_union_test.py"))
export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample=pl.DataFrame({"application_id":[f"A{i}" for i in range(30000)],"default_probability":np.zeros(30000)})

    def test_sample_order_restored(self):
        d=self.sample.with_columns(pl.Series("default_probability",np.linspace(0,1,30000))).reverse()
        out=export.align_submission(self.sample,d)
        self.assertEqual(out["application_id"].to_list(),self.sample["application_id"].to_list())
        np.testing.assert_array_equal(out["default_probability"].to_numpy(),np.linspace(0,1,30000))

    def test_duplicate_and_missing_ids_rejected(self):
        d=self.sample.with_columns(pl.when(pl.col("application_id")=="A1").then(pl.lit("A0")).otherwise(pl.col("application_id")).alias("application_id"))
        wrong_set=self.sample.with_columns(pl.when(pl.col("application_id")=="A1").then(pl.lit("unknown")).otherwise(pl.col("application_id")).alias("application_id"))
        for candidate in (d,wrong_set,self.sample.head(29999)):
            with self.assertRaises(ValueError):export.align_submission(self.sample,candidate)

    def test_nan_inf_and_out_of_range_rejected(self):
        for value in (np.nan,np.inf,-.01,1.01):
            d=self.sample.with_columns(pl.lit(value).alias("default_probability"))
            with self.assertRaises(ValueError):export.align_submission(self.sample,d)

    def test_wrong_columns_rejected(self):
        with self.assertRaises(ValueError):export.align_submission(self.sample,self.sample.rename({"default_probability":"prediction"}))

    def test_failed_or_partial_gate_rejected_before_audit_or_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/"validation_plan.json").write_text(json.dumps({"tag":"gate_fixture"}))
            report=root/"155_gate_fixture.json"
            for complete,passed in ((False,False),(False,True),(True,False)):
                report.write_text(json.dumps({"complete_5fold_3seed":complete,"passes_local_gate":passed,"rows":70000}))
                with patch.object(export,"OUT",root),self.assertRaisesRegex(ValueError,"Full local breakthrough gate"):
                    export.require_local_gate(root)


class ProvenanceChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.out=self.root/"out";self.out.mkdir();self.pq=self.root/"pq";self.pq.mkdir()
        (self.pq/"features.parquet").write_bytes(b"canonical fixture: no model or bootstrap computation")
        fits=[];audits=[]
        for seed in (42,7,2026):
            for fold in range(5):
                folder=self.root/"fixed_union_rank"/f"s{seed}_f{fold}";folder.mkdir(parents=True)
                manifest={"seed":seed,"fold":fold};(folder/"manifest.json").write_text(json.dumps(manifest))
                names=("manifest.json","input_audit.json","union_selector.json","union_gain.csv","splits.json","frozen_policy.json",
                       "outer_primary_fit.json","outer_primary_cache.json","outer_union_fit.json","outer_union_cache.json")
                for name in names[1:]:(folder/name).write_text("fixture metadata")
                reference=self.root/"teammate_features"/f"outer_s{seed}_f{fold}";reference.mkdir(parents=True)
                (reference/"manifest.json").write_text(json.dumps({"output":"features.parquet"}));(reference/"features.parquet").write_bytes(b"reference fixture")
                paths=[folder/name for name in names]+[reference/"manifest.json",reference/"features.parquet"]
                prediction=folder.parent/f"s{seed}_f{fold}_outer.parquet";prediction.write_bytes(f"prediction fixture {seed}/{fold}".encode())
                row={"seed":seed,"fold":fold,"rows":14000,"ks":.5,"path":str(prediction),"sha256":export.sha(prediction)}
                fits.append(row);audits.append({"seed":seed,"fold":fold,"run_sha256":export.digest(manifest),"outer_sha256":row["sha256"],
                    "artifacts_sha256":{str(p.relative_to(self.root)):export.sha(p) for p in paths},"passed":True})
        comparison={"candidate_ks":.54,"reference_ks":.536,"delta":.004,"bootstrap_delta_mean":.004,"bootstrap_sd":.001,
                    "p_gt_zero":.99,"fold_deltas_seed42_assignment":[.004]*5,"fold_wins":5,"fold_count":5}
        self.stats={"comparisons":{name:copy.deepcopy(comparison) for name in ("kumo_matched_baseline","v31_historical","v31_3seed_icl_reference")},
                    "passed":True,"candidate_fit_ks":[.5]*15,"baseline_fit_ks":[.5]*15}
        self.report={"complete_5fold_3seed":True,"passes_local_gate":True,"rows":70000,"seeds":[42,7,2026],"folds":list(range(5)),
                     "fit_provenance":fits,"baseline_provenance":copy.deepcopy(fits),"comparisons":copy.deepcopy(self.stats["comparisons"]),
                     "features_sha256":export.sha(self.pq/"features.parquet")}
        self.audit={"complete":True,"completed_audits":15,"fits":audits,"audit_source_sha256":export.sha(export.ROOT/"eda/192_fixed_union_provenance.py")}
        (self.root/"validation_plan.json").write_text(json.dumps({"tag":"gate_fixture","bootstrap":1000,
          "sources_sha256":{"155_kumo_nested_eval.py":export.sha(export.ROOT/"eda/155_kumo_nested_eval.py")}}))
        (self.root/"validation_progress.json").write_text(json.dumps({"status":"complete"}))
        self.publish()

    def publish(self):
        (self.out/"155_gate_fixture.json").write_text(json.dumps(self.report))
        (self.out/f"192_{self.root.name}_provenance_complete.json").write_text(json.dumps(self.audit))

    def gate(self):
        # Bootstrap math is155's existing contract. Mock it here to isolate
        #directory/metadata/role binding using real temporary file hashes.
        with patch.object(export,"OUT",self.out),patch.object(export,"PQ",self.pq),patch.object(export,"historical_identity",return_value={"fixture":"stable"}),patch.object(export,"compute_fixed_statistics",return_value=self.stats) as compute:
            result=export.require_local_gate(self.root)
            again=export.require_local_gate(self.root)
            self.assertEqual(result,again);self.assertEqual(compute.call_count,1)
            return result

    def test_passing_fifteen_fit_gate_is_bound_and_cached(self):
        result=self.gate();self.assertEqual(result["fixed_columns"],["candidate","baseline"]);self.assertEqual(len(result["fits"]),15)

    def test_incremental_policy_requires_verified_hybrid_and_complete_audit(self):
        self.report['passes_local_gate']=False;self.stats['passed']=False;self.publish()
        with patch.object(export,'OUT',self.out),patch.object(export,'PQ',self.pq),patch.object(export,'historical_identity',return_value={'fixture':'stable'}),patch.object(export,'compute_fixed_statistics',return_value=self.stats):
            with patch.object(export,'incremental_gate',side_effect=ValueError('Hybrid rejected')):
                with self.assertRaisesRegex(ValueError,'Hybrid rejected'):export.require_local_gate(self.root,'incremental_today')
            with patch.object(export,'incremental_gate',return_value={'fixture':'verified hybrid'}):
                result=export.require_local_gate(self.root,'incremental_today')
                self.assertEqual(result['release_policy'],'incremental_today')
                self.assertEqual(result['incremental_release'],{'fixture':'verified hybrid'})
                self.audit['complete']=False;self.publish()
                with self.assertRaisesRegex(ValueError,'provenance audit'):export.require_local_gate(self.root,'incremental_today')

    def test_incomplete_cached_comparison_set_is_rejected(self):
        self.gate();path=self.root/"fixed_union_gate_binding.json";binding=json.loads(path.read_text())
        binding["comparisons"]={};path.write_text(json.dumps(binding))
        with self.assertRaisesRegex(ValueError,"Invalid fixed-policy gate binding"):self.gate()

    def test_historical_dependency_change_invalidates_statistics_cache(self):
        self.gate()
        with patch.object(export,"OUT",self.out),patch.object(export,"PQ",self.pq),patch.object(export,"historical_identity",return_value={"fixture":"changed"}),patch.object(export,"compute_fixed_statistics",return_value=self.stats) as compute:
            export.require_local_gate(self.root);self.assertEqual(compute.call_count,1)

    def test_unrelated_report_path_rejected_even_with_matching_bytes(self):
        other=self.root/"unrelated.parquet";other.write_bytes(Path(self.report["fit_provenance"][0]["path"]).read_bytes())
        self.report["fit_provenance"][0]["path"]=str(other);self.publish()
        with self.assertRaisesRegex(ValueError,"different experiment"):self.gate()

    def test_missing_baseline_and_duplicate_audit_fits_rejected(self):
        self.report["baseline_provenance"]=[];self.publish()
        with self.assertRaisesRegex(ValueError,"exactly the 15"):self.gate()
        self.report["baseline_provenance"]=copy.deepcopy(self.report["fit_provenance"])
        self.audit["fits"][-1]=copy.deepcopy(self.audit["fits"][0]);self.publish()
        with self.assertRaisesRegex(ValueError,"duplicate gate fit"):self.gate()

    def test_different_column_statistics_rejected(self):
        self.report["comparisons"]["kumo_matched_baseline"]["candidate_ks"]=.55;self.publish()
        with self.assertRaisesRegex(ValueError,"statistics do not describe"):self.gate()

    def test_changed_selector_metadata_and_run_manifest_rejected(self):
        folder=self.root/"fixed_union_rank/s42_f0"
        (folder/"input_audit.json").write_text("changed column list")
        with self.assertRaisesRegex(ValueError,"Metadata or reference changed"):self.gate()
        (folder/"input_audit.json").write_text("fixture metadata")
        (folder/"manifest.json").write_text(json.dumps({"seed":42,"fold":0,"changed":True}))
        with self.assertRaisesRegex(ValueError,"current validation run"):self.gate()


if __name__=="__main__":
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (Checks,ProvenanceChecks))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=Path(__file__).resolve().parent/"out"
    (out/"194_fixed_export_checks.json").write_text(json.dumps({"checks":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"gpu_executed":False},indent=2))
    fig,ax=plt.subplots(figsize=(7,3));ax.bar(["pass","fail/error"],[result.testsRun-len(result.failures)-len(result.errors),len(result.failures)+len(result.errors)])
    ax.set(ylabel="CPU checks",title="Export gate and submission integrity")
    fig.tight_layout();fig.savefig(out/"194_fixed_export_checks.png",dpi=130);plt.close(fig)
    raise SystemExit(not result.wasSuccessful())
