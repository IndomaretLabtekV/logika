"""Predict official test applications with the frozen 15-context fixed recipe.

No submission API call. GPU inference requires complete passing local evaluation
and provenance audits. --prepare-only checks one existing context without GPU.
"""
import argparse
import gc
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

import valid2 as v
from common import ROOT,PQ,OUT,features
from nested_input_pools import teammate_inputs

spec=importlib.util.spec_from_file_location("test_fixed_union",Path(__file__).with_name("189_fixed_kumo_union_rank.py"))
fixed=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixed)
sha,digest,save=fixed.sha,fixed.digest,fixed.save


def read(path):return json.loads(Path(path).read_text())


def historical_identity():
    paths=[ROOT/"eda/screen2.py"]
    paths += [ROOT/"results/v22/v22/gbdt"/f"oof_{model}.parquet" for model in ("lgb","lgb_s","xgb","xgb_s","cat","cat_s")]
    paths += [ROOT/"results"/relative for relative in ("v22/v22/fm/oof_causilo.parquet","v28m/oof_kumo_large_c56000.parquet",
              "v38/oof_kumo_large_c56000.parquet","screen/seeds/s7/oof_causilo.parquet","screen/seeds/s2026/oof_causilo.parquet")]
    return {str(path.relative_to(ROOT)):sha(path) for path in paths}


def validate_fit_records(root,report,audit):
    expected={(seed,fold) for seed in (42,7,2026) for fold in range(5)}
    def records(rows):
        result={}
        for row in rows:
            key=(row["seed"],row["fold"])
            if key not in expected or key in result:raise ValueError("Unexpected or duplicate gate fit")
            result[key]=row
        if set(result)!=expected:raise ValueError("Gate must contain exactly the 15 fixed-policy fits")
        return result
    candidate=records(report["fit_provenance"]);baseline=records(report["baseline_provenance"]);audited=records(audit["fits"])
    for seed,fold in sorted(expected):
        key=(seed,fold);path=root/"fixed_union_rank"/f"s{seed}_f{fold}_outer.parquet";outer_hash=sha(path)
        for collection in (candidate,baseline):
            row=collection[key]
            if Path(row["path"]).resolve()!=path.resolve() or row["sha256"]!=outer_hash:
                raise ValueError("Gate report refers to a different experiment or changed prediction")
        row=audited[key];folder=root/"fixed_union_rank"/f"s{seed}_f{fold}"
        if row["outer_sha256"]!=outer_hash or row["run_sha256"]!=digest(read(folder/"manifest.json")):
            raise ValueError("Completed audit does not match current validation run")
        required={str((folder/name).relative_to(root)) for name in ("manifest.json","input_audit.json","union_selector.json","union_gain.csv","splits.json","frozen_policy.json",
                         "outer_primary_fit.json","outer_primary_cache.json","outer_union_fit.json","outer_union_cache.json")}
        reference=root/"teammate_features"/f"outer_s{seed}_f{fold}"
        rm=read(reference/"manifest.json");reference_output=(reference/rm["output"]).resolve()
        if not reference_output.is_relative_to(reference.resolve()):raise ValueError("Unsafe reference output path")
        required|={str((reference/"manifest.json").relative_to(root)),str(reference_output.relative_to(root))}
        if set(row["artifacts_sha256"])!=required:raise ValueError("Audit lacks the required metadata/reference fingerprints")
        for relative,expected_hash in row["artifacts_sha256"].items():
            if sha(root/relative)!=expected_hash:raise ValueError("Metadata or reference changed after the audit")
    return {f"{seed}/{fold}":audited[seed,fold]["outer_sha256"] for seed,fold in sorted(expected)}


def compute_fixed_statistics(root,bootstrap):
    #155 does not record selected column names. Recompute its deterministic
    #statistics from explicit fixed columns, rather than trusting a reused tag.
    path=ROOT/"eda/155_kumo_nested_eval.py"
    spec=importlib.util.spec_from_file_location("fixed_gate_evaluator",path)
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    canonical=pl.read_parquet(PQ/"features.parquet",columns=["application_id","is_train","target"]).filter(pl.col("is_train")==1).sort("application_id")
    ids=canonical["application_id"];y=canonical["target"].to_numpy().astype(int)
    p,_,details=evaluator.collect(root/"fixed_union_rank",ids,y,[42,7,2026],list(range(5)),"candidate")
    b,_,baseline_details=evaluator.collect(root/"fixed_union_rank",ids,y,[42,7,2026],list(range(5)),"baseline")
    references=evaluator.historical(ids,y);references["kumo_matched_baseline"]=b
    masks=[np.isin(np.arange(len(y)),va) for _,va in v.outer(y,42)]
    required=("kumo_matched_baseline","v31_historical","v31_3seed_icl_reference")
    comparisons={name:evaluator.compare(y,p,references[name],bootstrap,masks) for name in required}
    passed=all(c["delta"]>.003 and c["p_gt_zero"]>=.9 and c["fold_wins"]>=3 for c in comparisons.values())
    return {"comparisons":comparisons,"passed":passed,"candidate_fit_ks":[r["ks"] for r in details],"baseline_fit_ks":[r["ks"] for r in baseline_details]}


def incremental_gate(root):
    path=ROOT/'eda/203_fixed_hybrid_audit.py'
    spec=importlib.util.spec_from_file_location('incremental_hybrid_gate',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.require_incremental(root)


def require_local_gate(root,release_policy='breakthrough'):
    if release_policy not in ('breakthrough','incremental_today'):raise ValueError('Unknown release policy')
    plan=read(root/"validation_plan.json")
    report_path=OUT/f"155_{plan['tag']}.json"
    report=read(report_path)
    if not (report["complete_5fold_3seed"] and report["rows"]==70000 and
            (release_policy=='incremental_today' or report['passes_local_gate'])):
        raise ValueError("Full local breakthrough gate has not passed")
    if read(root/"validation_progress.json")["status"]!="complete":raise ValueError("Validation runner is not yet terminal")
    audit_path=OUT/f"192_{root.name}_provenance_complete.json";audit=read(audit_path)
    if not (audit["complete"] and audit["completed_audits"]==15 and all(r["passed"] for r in audit["fits"])):
        raise ValueError("Full provenance audit has not passed")
    assert audit["audit_source_sha256"]==sha(ROOT/"eda/192_fixed_union_provenance.py")
    assert report["seeds"]==[42,7,2026] and report["folds"]==[0,1,2,3,4]
    assert plan["sources_sha256"]["155_kumo_nested_eval.py"]==sha(ROOT/"eda/155_kumo_nested_eval.py")
    assert report["features_sha256"]==sha(PQ/"features.parquet")
    fits=validate_fit_records(root,report,audit)
    incremental=incremental_gate(root) if release_policy=='incremental_today' else None
    request={"report_sha256":sha(report_path),"audit_sha256":sha(audit_path),"validation_plan_sha256":sha(root/"validation_plan.json"),
             "validator_sha256":sha(__file__),"fixed_columns":["candidate","baseline"],"fits":fits,"bootstrap":plan["bootstrap"],
             "historical_reference_identity":historical_identity(),"release_policy":release_policy,"incremental_release":incremental}
    binding_path=root/"fixed_union_gate_binding.json"
    binding=read(binding_path) if binding_path.exists() else None
    if binding is None or binding["request"]!=request:
        print("Verifying actual fixed-policy paired statistics before test inference",flush=True)
        stats=compute_fixed_statistics(root,plan["bootstrap"])
        if release_policy=='breakthrough' and not stats["passed"]:raise ValueError("Actual fixed-policy statistics fail the local gate")
        for name,result in stats["comparisons"].items():
            if report["comparisons"][name]!=result:raise ValueError("Report statistics do not describe the fixed candidate/baseline columns")
        for role,key in (("candidate_fit_ks","fit_provenance"),("baseline_fit_ks","baseline_provenance")):
            rows=sorted(report[key],key=lambda r:([42,7,2026].index(r["seed"]),r["fold"]))
            if [r["ks"] for r in rows]!=stats[role]:raise ValueError("Report per-fit statistics describe different columns")
        binding={"request":request,"verified":True,"comparisons":stats["comparisons"]}
        save(binding_path,binding)
    required={"kumo_matched_baseline","v31_historical","v31_3seed_icl_reference"}
    if binding["verified"] is not True or set(binding["comparisons"])!=required or any(report["comparisons"][name]!=value for name,value in binding["comparisons"].items()):
        raise ValueError("Invalid fixed-policy gate binding")
    return {**request,"binding_sha256":sha(binding_path)}


def align_submission(sample,prediction):
    expected=["application_id","default_probability"]
    if sample.columns!=expected or prediction.columns!=expected:raise ValueError("Wrong submission columns")
    if sample.height!=30000 or prediction.height!=30000:raise ValueError("Expected exactly 30000 rows")
    if sample["application_id"].n_unique()!=30000 or prediction["application_id"].n_unique()!=30000:
        raise ValueError("Duplicate submission IDs")
    if set(sample["application_id"].to_list())!=set(prediction["application_id"].to_list()):raise ValueError("Submission ID set mismatch")
    result=sample.select("application_id").join(prediction,on="application_id",how="left",maintain_order="left")
    p=result["default_probability"].to_numpy()
    if not (np.isfinite(p).all() and ((p>=0)&(p<=1)).all()):raise ValueError("Invalid submission probability")
    assert result["application_id"].to_list()==sample["application_id"].to_list()
    return result


def predict_fit(root,work,seed,fold,prepare_only,release_policy='breakthrough'):
    gate=None if prepare_only else require_local_gate(root,release_policy)
    source=root/"fixed_union_rank"/f"s{seed}_f{fold}"
    vm=read(source/"manifest.json");validation_hash=digest(vm)
    assert (vm["seed"],vm["fold"])==(seed,fold) and vm["smoke_only"] is False
    for name,expected in vm["sources_sha256"].items():assert sha(ROOT/"eda"/name)==expected
    assert sha(PQ/"features.parquet")==vm["features_sha256"]
    assert fixed.base.checkpoint_identity()==vm["checkpoint"]
    for package,expected in vm["packages"].items():assert importlib.metadata.version(package)==expected
    audit=read(source/"input_audit.json");selector=source/"union_selector.json"
    assert audit["run_sha256"]==validation_hash and sha(selector)==audit["feature_selector_sha256"]
    assert audit["primary_columns"]==read(source/"outer_primary_fit.json")["columns"]
    assert audit["union_columns"]==read(source/"outer_union_fit.json")["columns"]==read(selector)["columns"]
    X,Xt,y,trids,teids=v.frames(features());train_ids=np.asarray(trids.to_list());test_ids=np.asarray(teids.to_list())
    assert len(train_ids)==70000 and len(test_ids)==30000 and set(train_ids).isdisjoint(test_ids)
    tr,va=v.outer(y,seed)[fold];splits=read(source/"splits.json")
    assert splits=={"run_sha256":validation_hash,"outer_train_ids":train_ids[tr].tolist(),"outer_validation_ids":train_ids[va].tolist()}
    cache=root/"teammate_features"/f"outer_s{seed}_f{fold}"
    frame,reference=fixed.base.read_pool(cache,"outer",seed,fold,train_ids[tr].tolist())
    assert reference==vm["teammate_reference"]
    friend=teammate_inputs(frame,train_ids,y,train_ids[tr])
    # Infer only the already selected columns. No selector, reference or vocabulary
    # is fitted on test covariates; the transform below learns vocab from tr only.
    primary_cols=audit["primary_columns"];union_cols=audit["union_columns"]
    friend_names=[c.removeprefix("teman__") for c in union_cols if c.startswith("teman__")]
    friend_test=frame.set_index("application_id").loc[test_ids,friend_names].reset_index(drop=True).copy()
    friend_all=pd.concat([friend[friend_names],friend_test],ignore_index=True).rename(columns=lambda c:"teman__"+c)
    main_all=pd.concat([X,Xt],ignore_index=True)
    union=pd.concat([main_all,friend_all],axis=1)
    for c in v.CAT_COLS:
        observed=frame.set_index("application_id").loc[test_ids,c].reset_index(drop=True).astype("string")
        assert observed.equals(Xt[c].reset_index(drop=True).astype("string")),"Test categorical alignment mismatch"
    del frame,friend,friend_test,friend_all,X,Xt;gc.collect()
    ids=np.r_[train_ids,test_ids];labels=np.r_[y,np.zeros(30000,dtype=int)];query=np.arange(70000,100000)
    manifest={"validation_manifest_sha256":sha(source/"manifest.json"),"validation_run_sha256":validation_hash,
              "validation_selector_sha256":sha(selector),"validation_audit_sha256":sha(source/"input_audit.json"),
              "test_script_sha256":sha(__file__),"test_sample_sha256":sha(ROOT/"data/sample_submission.csv"),
              "seed":seed,"fold":fold,"recipe":vm["policy"],"reference":reference,"test_ids":test_ids.tolist(),
              "gate":gate,"prepare_only":prepare_only,"submission_performed":False}
    folder=work/f"s{seed}_f{fold}";folder.mkdir(parents=True,exist_ok=True);mp=folder/"manifest.json"
    if mp.exists():assert read(mp)==manifest,"Changed source/config/gate; use a fresh output"
    elif any(folder.iterdir()):raise ValueError("Use fresh owned output")
    else:save(mp,manifest)
    run_hash=digest(manifest)
    if prepare_only:
        from nested_input_pools import prepare_stage
        for F,cols in ((main_all,primary_cols),(union,union_cols)):
            stage=prepare_stage(F,labels,ids,tr,{"test":query},cols,"raw")
            assert len(stage["frame"])==86000 and not stage["labels"][56000:].any()
            del stage;gc.collect()
        save(folder/"prepared.json",{"run_sha256":run_hash,"context_rows":56000,"test_rows":30000,"gpu_executed":False})
        values=[56000,30000];title="Prepared frozen context and official test; no inference"
    else:
        primary=fixed.PredictionCache(main_all,labels,ids,folder,run_hash,seed,fold)
        p=primary.predict("test_primary",tr,{"test":query},primary_cols,{"validation_run_sha256":validation_hash})["test"]
        second=fixed.PredictionCache(union,labels,ids,folder,run_hash,seed,fold);second.model=primary.model
        u=second.predict("test_union",tr,{"test":query},union_cols,{"reference":reference,"selector_sha256":sha(selector)})["test"]
        path=work/f"s{seed}_f{fold}_test.parquet";temporary=path.with_suffix(".parquet.tmp")
        pl.DataFrame({"application_id":test_ids,**fixed.fuse(p,u),"baseline_raw":p,"union_raw":u}).write_parquet(temporary);temporary.replace(path)
        save(folder/"complete.json",{"run_sha256":run_hash,"predictions_sha256":sha(path),"rows":30000,"submission_performed":False})
        values=[float(p.mean()),float(u.mean())];title="Official test mean probabilities; no labels or quality estimate"
    fig,ax=plt.subplots(figsize=(7,3));ax.bar(["primary/context","union/query"],values);ax.set(title=title)
    fig.tight_layout();fig.savefig(OUT/f"193_{work.name}_s{seed}_f{fold}.png",dpi=130);plt.close(fig)
    print("Prepared" if prepare_only else "Predicted",seed,fold,"context 56000 test 30000; no submission",flush=True)


def aggregate(root,work,release_policy='breakthrough'):
    gate=require_local_gate(root,release_policy);sample=pl.read_csv(ROOT/"data/sample_submission.csv")
    baseline=[];candidate=[];provenance=[]
    for seed in (42,7,2026):
        for fold in range(5):
            folder=work/f"s{seed}_f{fold}";manifest=read(folder/"manifest.json");complete=read(folder/"complete.json")
            assert manifest["gate"]==gate and manifest["prepare_only"] is False
            assert (manifest["seed"],manifest["fold"])==(seed,fold)
            source=root/"fixed_union_rank"/f"s{seed}_f{fold}"
            assert manifest["validation_manifest_sha256"]==sha(source/"manifest.json")
            assert manifest["validation_run_sha256"]==digest(read(source/"manifest.json"))
            assert manifest["validation_selector_sha256"]==sha(source/"union_selector.json")
            assert manifest["validation_audit_sha256"]==sha(source/"input_audit.json")
            assert manifest["test_script_sha256"]==sha(__file__) and complete["run_sha256"]==digest(manifest)
            path=work/f"s{seed}_f{fold}_test.parquet";assert complete["predictions_sha256"]==sha(path)
            d=pl.read_parquet(path)
            assert d["application_id"].to_list()==manifest["test_ids"]
            for col,p in fixed.fuse(d["baseline_raw"].to_numpy(),d["union_raw"].to_numpy()).items():
                assert np.array_equal(d[col].to_numpy(),p)
            for col,bucket in (("baseline",baseline),("candidate",candidate)):
                aligned=align_submission(sample,d.select("application_id",pl.col(col).alias("default_probability")))
                bucket.append(aligned["default_probability"].to_numpy())
            provenance.append({"seed":seed,"fold":fold,"path":str(path),"sha256":sha(path)})
    output=align_submission(sample,pl.DataFrame({"application_id":sample["application_id"],"default_probability":np.mean(candidate,axis=0)}))
    path=work/("kumo_component.csv" if release_policy=='incremental_today' else "submission.csv")
    tmp=path.with_suffix(".csv.tmp");output.write_csv(tmp);tmp.replace(path)
    # Re-read actual bytes as the submission validator sees them.
    align_submission(sample,pl.read_csv(path))
    baseline_output=pl.DataFrame({"application_id":sample["application_id"],"default_probability":np.mean(baseline,axis=0)})
    baseline_output.write_parquet(work/"matched_baseline_test.parquet")
    save(work/"export_manifest.json",{"gate":gate,"fits":provenance,"submission_sha256":sha(path),"rows":30000,
         "script_sha256":sha(__file__),"sample_sha256":sha(ROOT/"data/sample_submission.csv"),"submission_performed":False,
         "export_role":"Kumo component only; incremental hybrid must be assembled separately" if release_policy=='incremental_today' else "standalone fixed Kumo",
         "public_forecast":None,"submission_allowed_by_export":False})
    fig,ax=plt.subplots(figsize=(7,3));ax.hist(output["default_probability"].to_numpy(),bins=40);ax.set(xlabel="Fixed ensemble score",ylabel="Test rows",title="15-context export; no test quality estimate")
    fig.tight_layout();fig.savefig(OUT/f"193_{work.name}_export.png",dpi=130);plt.close(fig)
    print("Export validated",path,"rows 30000; forecast/notebook/reproduction still required; no submission",flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root",type=Path,default=ROOT/"results/v56")
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seed",type=int,choices=[42,7,2026],default=42)
    ap.add_argument("--fold",type=int,choices=range(5),default=0)
    ap.add_argument('--release-policy',choices=['breakthrough','incremental_today'],default='breakthrough',
                    help='Incremental export requires separately verified fixed-hybrid evidence under the latest user instruction')
    group=ap.add_mutually_exclusive_group();group.add_argument("--prepare-only",action="store_true");group.add_argument("--aggregate",action="store_true")
    a=ap.parse_args();root=a.run_root.resolve();work=a.out.resolve()
    if not work.is_relative_to(ROOT/"results") or work==root or work.is_relative_to(root/"fixed_union_rank") or work.is_relative_to(root/"teammate_features"):
        raise ValueError("Use a separate owned results export directory")
    work.mkdir(parents=True,exist_ok=True)
    if a.aggregate:aggregate(root,work,a.release_policy)
    else:predict_fit(root,work,a.seed,a.fold,a.prepare_only,a.release_policy)


if __name__=="__main__":main()
