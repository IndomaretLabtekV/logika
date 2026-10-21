"""Nested Kumo monetary geometry and full-feature cap test, with memory preflight."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

import valid2 as v
from common import ROOT,PQ,OUT,features
from nested_input_pools import prepare_stage,select_columns

spec=importlib.util.spec_from_file_location("geometry_base",Path(__file__).with_name("173_nested_kumo_inputs.py"))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
sha,digest,save=base.file_hash,base.digest,base.save_json


def apply_money(frame,columns):
    if not set(columns)<=set(frame.columns):raise ValueError("Unknown monetary input")
    out=frame.copy()
    for c in columns:
        x=out[c].to_numpy(dtype=float);out[c]=np.sign(x)*np.log1p(np.abs(x))
    return out


def full_feasible(report,run_hash):
    if report is None or report.get("run_sha256")!=run_hash or report.get("status") not in {"passed","oom"}:
        raise ValueError("A terminal same-run memory preflight is required before quality inference")
    return report["status"]=="passed"


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-run",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--fold",type=int,choices=range(5),default=0)
    ap.add_argument("--preflight-all",action="store_true")
    ap.add_argument("--evaluate",action="store_true")
    a=ap.parse_args();old=a.baseline_run/f"s{a.seed}_f{a.fold}"
    bm=json.loads((old/"manifest.json").read_text());bh=digest(bm)
    assert not bm["smoke_only"] and (bm["config"]["seed"],bm["config"]["fold"])==(a.seed,a.fold)
    assert (bm["config"]["estimators"],bm["config"]["chunk"],bm["config"]["gpu_memory_fraction"])==(4,128,.7)
    assert sha(PQ/"features.parquet")==bm["canonical_features_sha256"] and base.checkpoint_identity()==bm["checkpoint"]
    for name,expected in bm["sources_sha256"].items():assert sha(ROOT/"eda"/name)==expected
    X,_,y,train_ids,_=v.frames(features());ids=np.asarray(train_ids.to_list())
    tr,va=v.outer(y,a.seed)[a.fold];fi,_=v.inner(y[tr],a.seed,a.fold);fi=tr[fi]
    splits=json.loads((old/"splits.json").read_text());assert splits["run_sha256"]==bh
    assert splits["ids"]["inner_fit"]==ids[fi].tolist() and splits["ids"]["outer_validation"]==ids[va].tolist()
    index={key:i for i,key in enumerate(ids)}
    qs,qc=[np.array([index[key] for key in splits["ids"][role]]) for role in ("search","confirmation")]
    assert len(fi)==48000 and len(qs)==len(qc)==4000
    assert len(np.unique(np.r_[qs,qc]))==8000 and set(np.r_[qs,qc])==set(tr)-set(fi)
    columns=select_columns(X,y,fi,extend_shortlist=True);assert len(columns)==104 and len(X.columns)==232
    money_path=OUT/"153_kumo_input_audit_money.json";money=json.loads(money_path.read_text())["columns"]
    assert len(money)==42 and set(money)<=set(X)-set(v.CAT_COLS)
    cached={};baseline_inputs={}
    for name in ("utama_raw","utama_native"):
        fp,pp,cp=[old/f"inner_{name}_{suffix}" for suffix in ("fit.json","predictions.parquet","cache.json")]
        fit=json.loads(fp.read_text());cache=json.loads(cp.read_text());p=pl.read_parquet(pp)
        assert cache=={"fit_sha256":digest(fit),"predictions_sha256":sha(pp)} and fit["run_sha256"]==bh
        assert fit["real_context_ids"]==ids[fi].tolist() and fit["real_labels"]==y[fi].tolist() and fit["columns"]==columns
        assert (fit["estimators"],fit["generator_seed"],fit["chunk"])==(4,100*a.seed+a.fold,128)
        assert fit["representation"]==name.split("_")[1]
        assert p["application_id"].to_list()==ids[np.r_[qs,qc]].tolist()==fit["query_ids"]
        assert p["role"].to_list()==["search"]*4000+["confirmation"]*4000==fit["query_roles"]
        cached[name]={role:p.filter(pl.col("role")==role)["prediction"].to_numpy() for role in ("search","confirmation")}
        baseline_inputs[name]={"fit_sha256":sha(fp),"predictions_sha256":sha(pp)}
    manifest={"baseline_run_sha256":bh,"baselines":baseline_inputs,"seed":a.seed,"fold":a.fold,"money_columns":money,
              "money_audit_sha256":sha(money_path),"checkpoint":bm["checkpoint"],"packages":bm["packages"],
              "source_sha256":{name:sha(ROOT/"eda"/name) for name in (Path(__file__).name,"nested_input_pools.py","173_nested_kumo_inputs.py","152_vipcop_kumo.py","valid2.py","common.py")},
              "protocol_sha256":sha(ROOT/"docs/integration/protokol_kumo_geometry.md"),"smoke_only":False,
              "estimators":4,"chunk":128,"gpu_memory_fraction":.7,"preflight_queries":128,"min_inner_gain":.003}
    work=a.out.resolve();work.mkdir(parents=True,exist_ok=True);folder=work/f"s{a.seed}_f{a.fold}";folder.mkdir(exist_ok=True)
    mp=folder/"manifest.json"
    if mp.exists():assert json.loads(mp.read_text())==manifest,"Changed inputs/source; use new directory"
    elif any(folder.iterdir()):raise ValueError("Use fresh owned output")
    else:save(mp,manifest)
    run_hash=digest(manifest);save(folder/"splits.json",{"run_sha256":run_hash,"ids":splits["ids"]})
    applied=[c for c in columns if c in money]
    save(folder/"input_audit.json",{"run_sha256":run_hash,"baseline_columns":columns,"money_applied_columns":applied,
             "money_applied_count":len(applied),"full_columns":X.columns.tolist(),"context_rows":len(fi),"search_rows":len(qs),"confirmation_rows":len(qc)})
    import torch
    import sdm
    import nested_input_pools as adapter
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.7);model=None

    def prediction(name,real,queries,chosen,representation,log_money):
        nonlocal model
        stage=prepare_stage(X,y,ids,real,queries,chosen,representation)
        transformed=[c for c in chosen if c in money] if log_money else []
        if transformed:stage["frame"]=apply_money(stage["frame"],transformed)
        query=np.concatenate(list(stage["queries"].values()))
        assert not stage["labels"][len(real):].any()
        provenance={"run_sha256":run_hash,"name":name,"columns":chosen,"representation":representation,"log_money_columns":transformed,
                    "real_context_ids":ids[real].tolist(),"real_labels":y[real].tolist(),"query_ids":stage["ids"][query].tolist(),
                    "query_roles":[role for role,rows in queries.items() for _ in rows],"estimators":4,"chunk":128,
                    "generator_seed":100*a.seed+a.fold,"all_real_context_retained":True,"query_labels_passed_to_model":False}
        fp,pp,cp=[folder/f"{name}_{suffix}" for suffix in ("fit.json","predictions.parquet","cache.json")]
        if cp.exists():
            assert json.loads(fp.read_text())==provenance
            assert json.loads(cp.read_text())=={"fit_sha256":digest(provenance),"predictions_sha256":sha(pp)}
            saved=pl.read_parquet(pp);assert saved["application_id"].to_list()==provenance["query_ids"] and saved["role"].to_list()==provenance["query_roles"]
            p=saved["prediction"].to_numpy();memory=json.loads((folder/f"{name}_memory.json").read_text());print("cached",name,flush=True)
        else:
            save(fp,provenance)
            if model is None:model=sdm.models.KumoTabular(task="classification",size="large",device="cuda");model.eval()
            torch.cuda.reset_peak_memory_stats();predictor=None
            try:
                predictor=adapter._vip.Predictor(stage["frame"],stage["labels"],model)
                p=predictor.predict_cached(stage["real"],query,chosen,4,100*a.seed+a.fold,128)
            finally:
                del predictor;gc.collect();torch.cuda.empty_cache()
            memory={"max_allocated_gib":torch.cuda.max_memory_allocated()/2**30,"max_reserved_gib":torch.cuda.max_memory_reserved()/2**30}
            save(folder/f"{name}_memory.json",memory)
            temporary=pp.with_suffix(".parquet.tmp")
            pl.DataFrame({"application_id":provenance["query_ids"],"role":provenance["query_roles"],"prediction":p}).write_parquet(temporary);temporary.replace(pp)
            save(cp,{"fit_sha256":digest(provenance),"predictions_sha256":sha(pp)})
            print("computed",name,"real",len(real),"query",len(query),"features",len(chosen),memory,flush=True)
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        result={};start=0
        for role,rows in queries.items():result[role]=p[start:start+len(rows)];start+=len(rows)
        return result,memory

    preflight_path=folder/"all_memory_preflight.json"
    plot=OUT/f"181_{work.name}_s{a.seed}_f{a.fold}_{run_hash[:8]}.png"
    if a.preflight_all:
        if preflight_path.exists():
            report=json.loads(preflight_path.read_text());full_feasible(report,run_hash);print("Existing terminal preflight",json.dumps(report),flush=True);return
        save(preflight_path,{"run_sha256":run_hash,"status":"running","quality_scores_used":False})
        try:
            _,memory=prediction("all_preflight",fi,{"memory_only":qs[:128]},X.columns.tolist(),"raw",False)
            report={"run_sha256":run_hash,"status":"passed","memory":memory,"real_context_rows":len(fi),"query_rows":128,"features":232,"quality_scores_used":False}
        except torch.cuda.OutOfMemoryError as exc:
            report={"run_sha256":run_hash,"status":"oom","quality_scores_used":False,"real_context_rows":len(fi),"features":232,"error":str(exc),"reason":"all_raw infeasible under fixed allocator; not a model-quality result"}
        save(preflight_path,report)
        fig,ax=plt.subplots(figsize=(8,3));ax.bar([report["status"]],[report.get("memory",{}).get("max_allocated_gib",0)])
        ax.set(ylabel="Peak allocated GiB",title="Full232 engineering preflight; no KS evaluated")
        fig.tight_layout();fig.savefig(plot,dpi=130);plt.close(fig)
        print("Engineering preflight",json.dumps(report),flush=True);return
    report=json.loads(preflight_path.read_text()) if preflight_path.exists() else None
    feasible=full_feasible(report,run_hash)
    policies={"money_raw":("raw",True,columns),"money_native":("native",True,columns)}
    if feasible:policies["all_raw"]=("raw",False,X.columns.tolist())
    def score(rows,p):return {"ks":v.ks(y[rows],p),"auc":float(roc_auc_score(y[rows],p))}
    selection_path=folder/"selection.json"
    if selection_path.exists():selection=json.loads(selection_path.read_text());assert selection["run_sha256"]==run_hash
    else:
        search=[{"name":name,**score(qs,p["search"])} for name,p in cached.items()];predictions=dict(cached)
        for name,(representation,log_money,chosen) in policies.items():
            predictions[name],_=prediction("inner_"+name,fi,{"search":qs,"confirmation":qc},chosen,representation,log_money)
            row={"name":name,**score(qs,predictions[name]["search"])};search.append(row);save(folder/"search.json",search);print("search",row,flush=True)
        best=max(search,key=lambda row:row["ks"]);winner="utama_raw";confirm=[{"name":winner,**score(qc,cached[winner]["confirmation"])}]
        if best["name"]!=winner and best["ks"]-search[0]["ks"]>.003+1e-12:
            confirm.append({"name":best["name"],**score(qc,predictions[best["name"]]["confirmation"])})
            if confirm[-1]["ks"]-confirm[0]["ks"]>.003+1e-12:winner=best["name"]
        selection={"run_sha256":run_hash,"winner":winner,"search":search,"confirmation":confirm,"outer_seen":False,"min_inner_gain":.003,"full_features_feasible":feasible,"confirmation_reuse":"same inner split as prior input projects, not independent evidence"}
        save(selection_path,selection);print("selection FROZEN before outer",winner,flush=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key in zip(axes,("search","confirmation")):
        rows=selection[key];ax.bar([r["name"] for r in rows],[r["ks"]-rows[0]["ks"] for r in rows]);ax.axhline(.003,color="red",linestyle=":")
        ax.tick_params(axis="x",rotation=40);ax.set(ylabel="Delta KS vs baseline",title="Inner "+key)
    fig.tight_layout();fig.savefig(plot,dpi=130);plt.close(fig)
    if a.evaluate:
        assert json.loads(selection_path.read_text())==selection
        outer_columns=select_columns(X,y,tr,extend_shortlist=True)
        baseline,_=prediction("outer_utama_raw",tr,{"outer":va},outer_columns,"raw",False);baseline=baseline["outer"]
        winner=selection["winner"]
        if winner=="utama_raw":candidate=baseline
        else:
            representation="native" if winner in {"utama_native","money_native"} else "raw"
            chosen=X.columns.tolist() if winner=="all_raw" else outer_columns
            p,_=prediction("outer_"+winner,tr,{"outer":va},chosen,representation,winner.startswith("money_"));candidate=p["outer"]
        op=work/f"s{a.seed}_f{a.fold}_outer.parquet";temporary=op.with_suffix(".parquet.tmp")
        pl.DataFrame({"application_id":ids[va],"target":y[va],"baseline":baseline,"candidate":candidate}).write_parquet(temporary);temporary.replace(op)
        metrics={"run_sha256":run_hash,"winner":winner,"baseline":score(va,baseline),"selected":score(va,candidate),"selection_sha256":sha(selection_path),"outer_rows":len(va),"outer_predictions_sha256":sha(op),"smoke_only":False}
        metrics["delta_ks"]=metrics["selected"]["ks"]-metrics["baseline"]["ks"];save(folder/"outer_metrics.json",metrics)
        print("outer FROZEN evaluation",json.dumps(metrics),flush=True)
    print("Artifacts",folder,"plot",plot,flush=True)


if __name__=="__main__":main()
