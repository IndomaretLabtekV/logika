"""Nested conditional feature selection for Kumo; preserve173's matched baseline.

Protocol: protokol_kumo_conditional_inputs.md. Three challengers, fixed CPU
selector budget, full real context/e4. Every selector/vocabulary is fit-only.
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
from sklearn.metrics import roc_auc_score

import valid2 as v
from common import ROOT,PQ,OUT,features
from conditional_input_pools import gain_columns,union_inputs
from nested_input_pools import teammate_inputs,prepare_stage,select_columns

spec=importlib.util.spec_from_file_location("conditional_base",Path(__file__).with_name("173_nested_kumo_inputs.py"))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
sha,digest,save=base.file_hash,base.digest,base.save_json
NAMES=("utama_gain","teman_gain","union_gain")
BASELINE="utama_raw"


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-run",type=Path,required=True)
    ap.add_argument("--inner-cache",type=Path,required=True)
    ap.add_argument("--outer-cache",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--fold",type=int,choices=range(5),default=0)
    ap.add_argument("--prepare-only",action="store_true",help="Fit/audit CPU selectors; no Kumo inference")
    ap.add_argument("--evaluate",action="store_true")
    a=ap.parse_args();old=a.baseline_run/f"s{a.seed}_f{a.fold}"
    bm=json.loads((old/"manifest.json").read_text());bh=digest(bm)
    assert not bm["smoke_only"] and (bm["config"]["seed"],bm["config"]["fold"])==(a.seed,a.fold)
    assert bm["config"]["estimators"]==4 and bm["config"]["chunk"]==128 and bm["config"]["gpu_memory_fraction"]==.7
    for name,h in bm["sources_sha256"].items():assert sha(ROOT/"eda"/name)==h
    assert sha(PQ/"features.parquet")==bm["canonical_features_sha256"]
    assert base.checkpoint_identity()==bm["checkpoint"]
    assert all(importlib.metadata.version(name)==version for name,version in bm["packages"].items())
    X,_,y,train_ids,_=v.frames(features());ids=np.asarray(train_ids.to_list())
    tr,va=v.outer(y,a.seed)[a.fold];fi,_=v.inner(y[tr],a.seed,a.fold);fi=tr[fi]
    splits=json.loads((old/"splits.json").read_text());assert splits["run_sha256"]==bh
    assert splits["ids"]["inner_fit"]==ids[fi].tolist()
    assert splits["ids"]["outer_train"]==ids[tr].tolist() and splits["ids"]["outer_validation"]==ids[va].tolist()
    index={key:i for i,key in enumerate(ids)}
    qs,qc=[np.array([index[key] for key in splits["ids"][role]]) for role in ("search","confirmation")]
    assert len(fi)==48000 and len(qs)==len(qc)==4000
    assert not np.intersect1d(fi,np.r_[qs,qc,va]).size and not np.intersect1d(qs,qc).size
    fp=old/"inner_utama_raw_fit.json";pp=old/"inner_utama_raw_predictions.parquet";cp=old/"inner_utama_raw_cache.json"
    fit=json.loads(fp.read_text());cached=json.loads(cp.read_text());baseline=pl.read_parquet(pp)
    assert fit["run_sha256"]==bh and fit["real_context_ids"]==ids[fi].tolist() and fit["real_labels"]==y[fi].tolist()
    assert fit["columns"]==select_columns(X,y,fi,extend_shortlist=True)
    assert (fit["representation"],fit["estimators"],fit["generator_seed"],fit["chunk"])==("raw",4,100*a.seed+a.fold,128)
    assert cached=={"fit_sha256":digest(fit),"predictions_sha256":sha(pp)}
    assert baseline["application_id"].to_list()==ids[np.r_[qs,qc]].tolist()==fit["query_ids"]
    assert baseline["role"].to_list()==["search"]*len(qs)+["confirmation"]*len(qc)==fit["query_roles"]
    baseline_p={role:baseline.filter(pl.col("role")==role)["prediction"].to_numpy() for role in ("search","confirmation")}
    assert all(np.isfinite(p).all() and ((p>=0)&(p<=1)).all() for p in baseline_p.values())
    frame,reference=base.read_pool(a.inner_cache,"inner",a.seed,a.fold,ids[fi].tolist())
    friend=teammate_inputs(frame,ids,y,ids[fi]);del frame
    pools={"utama_gain":X,"teman_gain":friend,"union_gain":union_inputs(X,friend)}
    manifest={"source_sha256":{name:sha(ROOT/"eda"/name) for name in
              (Path(__file__).name,"conditional_input_pools.py","nested_input_pools.py","173_nested_kumo_inputs.py","152_vipcop_kumo.py","valid2.py","common.py")},
              "baseline_run_sha256":bh,"baseline_fit_sha256":sha(fp),"baseline_predictions_sha256":sha(pp),
              "inner_cache":reference,"protocol_sha256":sha(ROOT/"docs/integration/protokol_kumo_conditional_inputs.md"),
              "packages":{**bm["packages"],"lightgbm":importlib.metadata.version("lightgbm")},"checkpoint":bm["checkpoint"],
              "seed":a.seed,"fold":a.fold,"smoke_only":False,"estimators":4,"chunk":128,"gpu_memory_fraction":.7,
              "selector_rounds":400,"policies":[BASELINE,*NAMES],"min_inner_gain":.003}
    work=a.out.resolve();work.mkdir(parents=True,exist_ok=True);folder=work/f"s{a.seed}_f{a.fold}";folder.mkdir(exist_ok=True)
    mp=folder/"manifest.json"
    if mp.exists():assert json.loads(mp.read_text())==manifest,"Changed inputs/source; use a new directory"
    elif any(folder.iterdir()):raise ValueError("Use fresh owned output")
    else:save(mp,manifest)
    run_hash=digest(manifest)
    save(folder/"splits.json",{"run_sha256":run_hash,"ids":splits["ids"]})

    def selector(name,pool,real):
        path=folder/f"{name}_selector.json";gain_path=folder/f"{name}_gain.csv"
        requested={"run_sha256":run_hash,"name":name,"fit_ids":ids[real].tolist(),"labels":y[real].tolist(),"pool_columns":pool.columns.tolist()}
        if path.exists():
            saved=json.loads(path.read_text());assert saved["request"]==requested and saved["gain_sha256"]==sha(gain_path)
            print("cached selector",name,"numeric",len(saved["columns"])-4,flush=True);return saved["columns"]
        cols,gain,metadata=gain_columns(pool,y,real,seed=100*a.seed+a.fold)
        gain.sort_values(ascending=False,kind="stable").rename_axis("feature").to_csv(gain_path)
        save(path,{"request":requested,"columns":cols,"gain_sha256":sha(gain_path),"selector":metadata})
        print("fit-only selector",name,"pool",metadata["numeric_pool"],"shortlist",metadata["shortlist"],"selected",len(cols),flush=True)
        return cols

    columns={name:selector("inner_"+name,pools[name],fi) for name in NAMES}
    for name,cols in columns.items():
        stage=prepare_stage(pools[name],y,ids,fi,{"search":qs,"confirmation":qc},cols,"raw")
        assert len(cols)==104 and not stage["labels"][len(fi):].any() and set(stage["ids"]).isdisjoint(ids[va]);del stage
    audit={"run_sha256":run_hash,"columns":columns,"baseline_column_count":len(fit["columns"]),
           "context_rows":len(fi),"search_rows":len(qs),"confirmation_rows":len(qc),"query_targets_erased":True,
           "selected_missing_fraction":{name:{c:float(pools[name].iloc[fi][c].isna().mean()) for c in cols} for name,cols in columns.items()}}
    save(folder/"input_audit.json",audit)
    plot=OUT/f"177_{work.name}_s{a.seed}_f{a.fold}_{run_hash[:8]}.png"
    if a.prepare_only:
        fig,ax=plt.subplots(figsize=(8,3));ax.bar(list(columns),[len(cols) for cols in columns.values()])
        ax.set(ylabel="Selected model columns",title="Conditional selectors fitted; no Kumo inference")
        fig.tight_layout();fig.savefig(plot,dpi=130);plt.close(fig)
        print("CPU selectors prepared; Kumo not evaluated",flush=True);return
    import torch
    import sdm
    import nested_input_pools as pool_adapter
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.7);model=None

    def predict(name,pool,real,queries,cols,identity):
        nonlocal model
        stage=prepare_stage(pool,y,ids,real,queries,cols,"raw");query=np.concatenate(list(stage["queries"].values()))
        provenance={"run_sha256":run_hash,"name":name,"columns":cols,"reference_cache":identity,"representation":"raw",
                    "real_context_ids":ids[real].tolist(),"real_labels":y[real].tolist(),"query_ids":stage["ids"][query].tolist(),
                    "query_roles":[role for role,rows in queries.items() for _ in rows],"estimators":4,"chunk":128,
                    "generator_seed":100*a.seed+a.fold,"all_real_context_retained":True,"query_labels_passed_to_model":False}
        fp,pp,cp=[folder/f"{name}_{suffix}" for suffix in ("fit.json","predictions.parquet","cache.json")]
        if cp.exists():
            assert json.loads(fp.read_text())==provenance
            assert json.loads(cp.read_text())=={"fit_sha256":digest(provenance),"predictions_sha256":sha(pp)}
            saved=pl.read_parquet(pp);assert saved["application_id"].to_list()==provenance["query_ids"] and saved["role"].to_list()==provenance["query_roles"]
            p=saved["prediction"].to_numpy();print("cached",name,flush=True)
        else:
            save(fp,provenance)
            if model is None:model=sdm.models.KumoTabular(task="classification",size="large",device="cuda");model.eval()
            torch.cuda.reset_peak_memory_stats();predictor=pool_adapter._vip.Predictor(stage["frame"],stage["labels"],model)
            try:p=predictor.predict_cached(stage["real"],query,cols,4,100*a.seed+a.fold,128)
            finally:del predictor;gc.collect();torch.cuda.empty_cache()
            save(folder/f"{name}_memory.json",{"max_allocated_gib":torch.cuda.max_memory_allocated()/2**30,"max_reserved_gib":torch.cuda.max_memory_reserved()/2**30})
            temporary=pp.with_suffix(".parquet.tmp")
            pl.DataFrame({"application_id":provenance["query_ids"],"role":provenance["query_roles"],"prediction":p}).write_parquet(temporary);temporary.replace(pp)
            save(cp,{"fit_sha256":digest(provenance),"predictions_sha256":sha(pp)})
            print("computed",name,"real",len(real),"query",len(query),flush=True)
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        out={};start=0
        for role,rows in queries.items():out[role]=p[start:start+len(rows)];start+=len(rows)
        return out

    def score(rows,p):return {"ks":v.ks(y[rows],p),"auc":float(roc_auc_score(y[rows],p))}
    selection_path=folder/"selection.json"
    if selection_path.exists():
        selection=json.loads(selection_path.read_text());assert selection["run_sha256"]==run_hash
    else:
        search=[{"name":BASELINE,**score(qs,baseline_p["search"])}];predictions={}
        for name in NAMES:
            identity=reference if name!="utama_gain" else {"canonical_features_sha256":bm["canonical_features_sha256"]}
            predictions[name]=predict("inner_"+name,pools[name],fi,{"search":qs,"confirmation":qc},columns[name],identity)
            row={"name":name,**score(qs,predictions[name]["search"])};search.append(row);save(folder/"search.json",search)
            print("search",row,flush=True)
        best=max(search,key=lambda row:row["ks"]);winner=BASELINE
        confirm=[{"name":BASELINE,**score(qc,baseline_p["confirmation"])}]
        if best["name"]!=BASELINE and best["ks"]-search[0]["ks"]>.003+1e-12:
            confirm.append({"name":best["name"],**score(qc,predictions[best["name"]]["confirmation"])})
            if confirm[-1]["ks"]-confirm[0]["ks"]>.003+1e-12:winner=best["name"]
        selection={"run_sha256":run_hash,"winner":winner,"search":search,"confirmation":confirm,"outer_seen":False,
                   "min_inner_gain":.003,"confirmation_reuse":"same inner split as173/176; not independent evidence across projects"}
        save(selection_path,selection);print("selection FROZEN before outer",winner,flush=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key in zip(axes,("search","confirmation")):
        rows=selection[key];ax.bar([r["name"] for r in rows],[r["ks"]-rows[0]["ks"] for r in rows]);ax.axhline(.003,color="red",linestyle=":")
        ax.tick_params(axis="x",rotation=40);ax.set(ylabel="Delta KS vs baseline",title="Inner "+key)
    fig.tight_layout();fig.savefig(plot,dpi=130);plt.close(fig)
    del pools,friend;gc.collect()
    if a.evaluate:
        assert json.loads(selection_path.read_text())==selection
        baseline=predict("outer_"+BASELINE,X,tr,{"outer":va},select_columns(X,y,tr,extend_shortlist=True),{"canonical_features_sha256":bm["canonical_features_sha256"]})["outer"]
        winner=selection["winner"]
        if winner==BASELINE:candidate=baseline
        else:
            if winner=="utama_gain":pool=X;identity={"canonical_features_sha256":bm["canonical_features_sha256"]}
            else:
                if a.outer_cache is None:raise ValueError("Rebuild outer references with166, then provide --outer-cache")
                frame,identity=base.read_pool(a.outer_cache,"outer",a.seed,a.fold,ids[tr].tolist())
                friend=teammate_inputs(frame,ids,y,ids[tr]);del frame
                pool=union_inputs(X,friend) if winner=="union_gain" else friend
            cols=selector("outer_"+winner,pool,tr)
            candidate=predict("outer_"+winner,pool,tr,{"outer":va},cols,identity)["outer"]
        op=work/f"s{a.seed}_f{a.fold}_outer.parquet";temporary=op.with_suffix(".parquet.tmp")
        pl.DataFrame({"application_id":ids[va],"target":y[va],"baseline":baseline,"candidate":candidate}).write_parquet(temporary);temporary.replace(op)
        metrics={"run_sha256":run_hash,"winner":winner,"baseline":score(va,baseline),"selected":score(va,candidate),
                 "selection_sha256":sha(selection_path),"outer_rows":len(va),"outer_predictions_sha256":sha(op),"smoke_only":False}
        metrics["delta_ks"]=metrics["selected"]["ks"]-metrics["baseline"]["ks"]
        save(folder/"outer_metrics.json",metrics);print("outer FROZEN evaluation",json.dumps(metrics),flush=True)
    print("Artifacts",folder,"plot",plot,flush=True)


if __name__=="__main__":main()
