"""Audit completed fixed-union fits without selecting a recipe or running models.

Partial audits describe completed fits only; --require-complete requires the
prescribed 15 fits. Reads only metadata/predictions and narrow reference columns.
"""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import valid2 as v
from common import ROOT, PQ, OUT

spec=importlib.util.spec_from_file_location("audit_fixed_union",Path(__file__).with_name("189_fixed_kumo_union_rank.py"))
fixed=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixed)
sha,digest=fixed.sha,fixed.digest


def read(path):return json.loads(Path(path).read_text())


def checked_prediction(folder,name,run_hash,columns,real_ids,labels,query_ids,seed,fold):
    fit=read(folder/f"{name}_fit.json")
    cache=read(folder/f"{name}_cache.json")
    path=folder/f"{name}_predictions.parquet"
    assert cache=={"fit_sha256":digest(fit),"predictions_sha256":sha(path)},"Prediction cache hash mismatch"
    assert fit["run_sha256"]==run_hash and fit["name"]==name
    assert fit["columns"]==columns and len(columns)==len(set(columns))==104
    assert fit["real_context_ids"]==real_ids and fit["real_labels"]==labels,"Context membership/labels changed"
    assert fit["query_ids"]==query_ids and fit["query_roles"]==["outer"]*len(query_ids)
    assert fit["representation"]=="raw" and fit["estimators"]==4 and fit["chunk"]==128
    assert fit["context_size"]==56000 and fit["generator_seed"]==100*seed+fold
    assert fit["query_labels_passed_to_model"] is False
    d=pl.read_parquet(path)
    assert d["application_id"].to_list()==query_ids and d["role"].to_list()==fit["query_roles"]
    p=d["prediction"].to_numpy()
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return p,fit


def audit_fit(root,seed,fold,ids,y,global_identity,hash_once):
    folder=root/"fixed_union_rank"/f"s{seed}_f{fold}"
    manifest=read(folder/"manifest.json");run_hash=digest(manifest)
    assert (manifest["seed"],manifest["fold"])==(seed,fold) and manifest["smoke_only"] is False
    stable={k:manifest[k] for k in ("policy","sources_sha256","features_sha256","checkpoint","packages","protocol_sha256","development_inner_choice_sha256")}
    assert stable==global_identity,"Recipe or environment differs between fits"
    for name,expected in manifest["sources_sha256"].items():assert hash_once(ROOT/"eda"/name)==expected
    tr,va=v.outer(y,seed)[fold];real_ids=ids[tr].tolist();query_ids=ids[va].tolist()
    assert len(real_ids)==len(set(real_ids))==56000 and len(query_ids)==len(set(query_ids))==14000
    assert set(real_ids).isdisjoint(query_ids)
    splits=read(folder/"splits.json")
    assert splits=={"run_sha256":run_hash,"outer_train_ids":real_ids,"outer_validation_ids":query_ids}
    frozen=read(folder/"frozen_policy.json")
    assert frozen=={"run_sha256":run_hash,"policy":manifest["policy"],"outer_scores_seen":False,"selection_type":"fixed before this outer evaluation"}
    audit=read(folder/"input_audit.json");selector=read(folder/"union_selector.json")
    assert audit["run_sha256"]==run_hash and selector["request"]["run_sha256"]==run_hash
    assert selector["request"]["fit_ids"]==real_ids and selector["request"]["fit_labels"]==y[tr].tolist()
    assert selector["columns"]==audit["union_columns"]
    assert sha(folder/"union_gain.csv")==selector["gain_sha256"]
    assert sha(folder/"union_selector.json")==audit["feature_selector_sha256"]
    assert sha(folder/"frozen_policy.json")==audit["frozen_policy_sha256"]
    forbidden=set(v.ID_COLS)|{"split"}
    for cols in (audit["primary_columns"],audit["union_columns"]):
        assert not set(cols)&forbidden and cols[-4:]==v.CAT_COLS
        assert len(cols)==len(set(cols))==104
    reference_folder=root/"teammate_features"/f"outer_s{seed}_f{fold}"
    reference=read(reference_folder/"manifest.json");inputs=reference["inputs"]
    assert reference["status"]=="complete" and digest(inputs)==reference["inputs_sha256"]
    assert (inputs["partition"],inputs["seed"],inputs["fold"])==("outer",seed,fold)
    assert inputs["fitting_application_ids"]==real_ids
    for package,expected in inputs["packages"].items():assert importlib.metadata.version(package)==expected
    for name,expected in inputs["source_sha256"].items():assert hash_once(ROOT/name)==expected
    for name,expected in inputs["raw_sha256"].items():assert hash_once(ROOT/"data"/f"{name}.csv")==expected
    assert inputs["id_split_cache_sha256"]==global_identity["features_sha256"]
    output=(reference_folder/reference["output"]).resolve()
    assert output.is_relative_to(reference_folder.resolve()) and sha(output)==reference["output_sha256"]
    expected_reference={"manifest_sha256":sha(reference_folder/"manifest.json"),"output_sha256":reference["output_sha256"],"inputs_sha256":reference["inputs_sha256"],"partition":"outer"}
    assert manifest["teammate_reference"]==expected_reference
    # Inspect actual narrow columns, rather than relying on the label-policy flag.
    r=pl.read_parquet(output,columns=["application_id","customer_id","split","target"])
    assert r.height==100000 and r["application_id"].n_unique()==r.height and r["customer_id"].n_unique()==r.height
    assert set(r["split"].unique().to_list())<={"train","test"}
    fitted=r.filter(pl.col("application_id").is_in(real_ids));unfitted=r.filter(~pl.col("application_id").is_in(real_ids))
    assert fitted.height==56000 and fitted["split"].to_list()==["train"]*56000
    assert unfitted["target"].null_count()==44000
    assert set(r.filter(pl.col("split")=="train")["application_id"].to_list())==set(real_ids)
    truth=dict(zip(real_ids,y[tr].tolist()))
    assert fitted["target"].to_list()==[truth[k] for k in fitted["application_id"]]
    primary,pfit=checked_prediction(folder,"outer_primary",run_hash,audit["primary_columns"],real_ids,y[tr].tolist(),query_ids,seed,fold)
    union,ufit=checked_prediction(folder,"outer_union",run_hash,audit["union_columns"],real_ids,y[tr].tolist(),query_ids,seed,fold)
    assert pfit["reference"]=={"family":"primary","features_sha256":manifest["features_sha256"]}
    assert ufit["reference"]=={"reference":expected_reference,"selector_sha256":sha(folder/"union_selector.json")}
    path=root/"fixed_union_rank"/f"s{seed}_f{fold}_outer.parquet";d=pl.read_parquet(path)
    assert d["application_id"].to_list()==query_ids and np.array_equal(d["target"].to_numpy(),y[va])
    assert np.array_equal(d["baseline_raw"].to_numpy(),primary) and np.array_equal(d["union_raw"].to_numpy(),union)
    fused=fixed.fuse(primary,union)
    for col,p in fused.items():assert np.array_equal(d[col].to_numpy(),p),"Saved fusion differs from fixed recipe"
    metrics=read(folder/"outer_metrics.json")
    assert metrics["run_sha256"]==run_hash and metrics["outer_predictions_sha256"]==sha(path)
    assert metrics["frozen_policy_sha256"]==audit["frozen_policy_sha256"]
    for label,p in (("baseline",fused["baseline"]),("candidate",fused["candidate"]),("union",union),("baseline_raw",primary)):
        assert abs(v.ks(y[va],p)-metrics[label]["ks"])<1e-12
    pinned=[folder/name for name in ("manifest.json","input_audit.json","union_selector.json","union_gain.csv","splits.json","frozen_policy.json",
                                    "outer_primary_fit.json","outer_primary_cache.json","outer_union_fit.json","outer_union_cache.json")]
    pinned += [reference_folder/"manifest.json",output]
    return {"seed":seed,"fold":fold,"run_sha256":run_hash,"outer_sha256":sha(path),
            "artifacts_sha256":{str(p.relative_to(root)):sha(p) for p in pinned},
            "delta_ks":metrics["delta_ks"],"passed":True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root",type=Path,default=ROOT/"results/v56")
    ap.add_argument("--require-complete",action="store_true")
    a=ap.parse_args();root=a.run_root.resolve();plan=read(root/"validation_plan.json")
    assert plan["seeds"]==[42,7,2026] and plan["folds"]==[0,1,2,3,4]
    first=read(root/"fixed_union_rank/s42_f0/manifest.json")
    identity={k:first[k] for k in ("policy","sources_sha256","features_sha256","checkpoint","packages","protocol_sha256","development_inner_choice_sha256")}
    hashes={}
    def hash_once(path):
        path=Path(path).resolve()
        if path not in hashes:hashes[path]=sha(path)
        return hashes[path]
    assert hash_once(PQ/"features.parquet")==identity["features_sha256"]
    assert hash_once(ROOT/"docs/integration/protokol_fixed_union_rank_outer.md")==identity["protocol_sha256"]
    assert hash_once(ROOT/"results/v51/conditional_blends_f0/selection.json")==identity["development_inner_choice_sha256"]
    assert fixed.base.checkpoint_identity()==identity["checkpoint"]
    assert {p:importlib.metadata.version(p) for p in identity["packages"]}==identity["packages"]
    canonical=pl.read_parquet(PQ/"features.parquet",columns=["application_id","target","is_train"]).filter(pl.col("is_train")==1).sort("application_id")
    ids=canonical["application_id"].to_numpy();y=canonical["target"].to_numpy().astype(int)
    results=[];missing=[]
    for seed in plan["seeds"]:
        for fold in plan["folds"]:
            marker=root/"fixed_union_rank"/f"s{seed}_f{fold}"/"outer_metrics.json"
            if not marker.exists():missing.append([seed,fold]);continue
            row=audit_fit(root,seed,fold,ids,y,identity,hash_once);results.append(row)
            print("AUDITED",seed,fold,"delta",row["delta_ks"],flush=True)
    complete=len(results)==15 and not missing
    # A whole seed covers all 70k rows. Partial seeds have different memberships
    # and must never be averaged as if they were complete OOF vectors.
    complete_seeds=[seed for seed in plan["seeds"] if sum(r["seed"]==seed for r in results)==5]
    seed_summaries=[];seed_predictions=[]
    for seed in complete_seeds:
        parts=[pl.read_parquet(root/"fixed_union_rank"/f"s{seed}_f{fold}_outer.parquet") for fold in range(5)]
        merged=pl.concat(parts).sort("application_id")
        assert merged["application_id"].to_list()==ids.tolist() and np.array_equal(merged["target"].to_numpy(),y)
        p=merged["candidate"].to_numpy();b=merged["baseline"].to_numpy()
        seed_predictions.append((p,b))
        row={"seed":seed,"candidate_ks":v.ks(y,p),"baseline_ks":v.ks(y,b)}
        row["delta_ks"]=row["candidate_ks"]-row["baseline_ks"];seed_summaries.append(row)
        print("COMPLETE-SEED DIAGNOSTIC",json.dumps(row),"not the full gate",flush=True)
    seed_mean=None
    if seed_predictions:
        p=np.mean([pair[0] for pair in seed_predictions],axis=0);b=np.mean([pair[1] for pair in seed_predictions],axis=0)
        deltas=[v.ks(y[va],p[va])-v.ks(y[va],b[va]) for _,va in v.outer(y,42)]
        seed_mean={"seeds":complete_seeds,"rows":len(y),"candidate_ks":v.ks(y,p),"baseline_ks":v.ks(y,b),
                   "delta_ks":v.ks(y,p)-v.ks(y,b),"fold_deltas_seed42_assignment":deltas,"fold_wins":sum(d>0 for d in deltas)}
        print("COMPLETE-SEED MEAN DIAGNOSTIC",json.dumps(seed_mean),"bootstrap/gate assessed separately",flush=True)
    report={"completed_audits":len(results),"expected":15,"complete":complete,"fits":results,"missing":missing,
            "complete_seed_diagnostics":seed_summaries,
            "complete_seed_mean_diagnostic":seed_mean,
            "audit_source_sha256":sha(__file__),"submission_allowed":False,
            "limits":["Hash and invariants verify recorded computations, not the statistical independence of research decisions.","Does not re-fit feature selectors or run models; selector fitting code is checked by source hash.","Requires a separate full paired evaluation and justified public forecast before submission."]}
    suffix="complete" if a.require_complete else "partial"
    target=OUT/f"192_{root.name}_provenance_{suffix}"
    fixed.save(target.with_suffix(".json"),report)
    fig,ax=plt.subplots(figsize=(8,3));ax.bar([f"{r['seed']}/{r['fold']}" for r in results],[r["delta_ks"] for r in results])
    ax.axhline(0,color="black",lw=.7);ax.set(ylabel="Recorded paired KS delta",title=f"Provenance audit: {len(results)}/15 fits; no selection")
    ax.tick_params(axis="x",rotation=45);fig.tight_layout();fig.savefig(target.with_suffix(".png"),dpi=130);plt.close(fig)
    print("Audit",len(results),"/15; complete",complete,"submission_allowed False",flush=True)
    if a.require_complete and not complete:raise SystemExit("Required 15 completed fits are not all available")


if __name__=="__main__":main()
