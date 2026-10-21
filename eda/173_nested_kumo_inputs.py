"""Paired nested comparison of Kumo input families, with frozen inner selection.

Four policies are prescribed in protokol_fitur_teman_validation_v2.md. All real
context rows and e4 are retained. The teammate cache must have references fitted
on exactly the declared partition. No organizer test rows enter model fitting.
"""
import argparse
import gc
import hashlib
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
from sklearn.model_selection import train_test_split

import valid2 as v
from common import PQ,features
from nested_input_pools import teammate_inputs,select_columns,prepare_stage

ROOT=Path(__file__).resolve().parents[1]
POLICIES=("utama_raw","utama_native","teman_raw","teman_native")
BASELINE=POLICIES[0]


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def save_json(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def challenger(search,gate):
    if not search or search[0]["name"]!=BASELINE or gate<0:
        raise ValueError("Search must begin with matched baseline and use a nonnegative gate")
    names=[row["name"] for row in search]
    if len(names)!=len(set(names)) or not set(names)<=set(POLICIES):raise ValueError("Invalid search policies")
    if not all(np.isfinite(row["ks"]) for row in search):raise ValueError("Invalid search score")
    top=max(search,key=lambda row:row["ks"])
    # Ignore floating-point roundoff at the strict >0.003 boundary.
    return top["name"] if top["ks"]-search[0]["ks"]>gate+1e-12 else None


def freeze_selection(search,confirmation,gate):
    top=challenger(search,gate)
    expected=[BASELINE]+([top] if top is not None else [])
    if [row["name"] for row in confirmation]!=expected:
        raise ValueError("Confirmation must contain baseline and only the search-selected challenger")
    if not all(np.isfinite(row["ks"]) for row in confirmation):raise ValueError("Invalid confirmation score")
    passes=top is not None and confirmation[1]["ks"]-confirmation[0]["ks"]>gate+1e-12
    return {"winner":top if passes else BASELINE,"outer_seen":False,"min_inner_gain":gate,
            "search":search,"confirmation":confirmation,
            "reason":"both independent inner gates passed" if passes else "matched baseline retained: inner gates not both passed"}


def read_pool(work,partition,seed,fold,fit_ids):
    work=Path(work).resolve();mp=work/"manifest.json"
    manifest=json.loads(mp.read_text());inputs=manifest["inputs"]
    if manifest["status"]!="complete" or digest(inputs)!=manifest["inputs_sha256"]:
        raise ValueError("Feature cache is incomplete or has a changed manifest")
    if (inputs["partition"],inputs["seed"],inputs["fold"])!=(partition,seed,fold):
        raise ValueError("Feature cache partition/seed/fold mismatch")
    if inputs["fitting_application_ids"]!=list(fit_ids):raise ValueError("Feature reference population/order mismatch")
    for name,expected in inputs["source_sha256"].items():
        if file_hash(ROOT/name)!=expected:raise ValueError(f"Feature source changed: {name}")
    output=(work/manifest["output"]).resolve()
    if not output.is_relative_to(work) or file_hash(output)!=manifest["output_sha256"]:
        raise ValueError("Feature output path/hash mismatch")
    return pd.read_parquet(output),{"manifest_sha256":file_hash(mp),"output_sha256":manifest["output_sha256"],
                                    "inputs_sha256":manifest["inputs_sha256"],"partition":partition}


def checkpoint_identity():
    """Check cached bytes against the already audited loader/cache; no download."""
    from huggingface_hub.constants import HF_HUB_CACHE
    audit=json.loads((ROOT/"eda/out/168_kumo_checkpoint_audit.json").read_text())
    package=Path(importlib.util.find_spec("sdm").origin).parent
    cache=Path(HF_HUB_CACHE)/"models--nvidia--Kumo-Tabular"
    commit=(cache/"refs/v1.0.1").read_text().strip()
    weight=cache/"snapshots"/commit/"large/classifier.pt"
    if commit!=audit["snapshot_commit"] or file_hash(weight)!=audit["checkpoint_sha256"]:
        raise ValueError("Local checkpoint differs from the pinned audit")
    for relative,key in (("models/kumo/tabular/model.py","model_loader_sha256"),("models/_huggingface.py","downloader_sha256")):
        if file_hash(package/relative)!=audit[key]:raise ValueError("Model loader changed")
    return {key:audit[key] for key in ("repository","revision","snapshot_commit","checkpoint_sha256","model_loader_sha256","downloader_sha256")}


def score(y,p):return {"ks":v.ks(y,p),"auc":float(roc_auc_score(y,p))}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--inner-cache",type=Path,required=True)
    ap.add_argument("--outer-cache",type=Path)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--fold",type=int,choices=range(5),default=0)
    ap.add_argument("--chunk",type=int,default=128)
    ap.add_argument("--gpu-memory-fraction",type=float,default=.7)
    ap.add_argument("--min-inner-gain",type=float,default=.003)
    ap.add_argument("--prepare-only",action="store_true",help="Audit inner inputs without model/GPU inference")
    ap.add_argument("--evaluate",action="store_true",help="Evaluate outer after durable frozen selection")
    ap.add_argument("--smoke",action="store_true",help="Engineering only: retain full real context, reduce query counts")
    ap.add_argument("--smoke-query",type=int,default=64)
    a=ap.parse_args()
    if a.chunk<=0 or not 0<a.gpu_memory_fraction<1 or a.smoke_query<16:raise ValueError("Invalid engineering budget")
    if a.min_inner_gain<0:raise ValueError("Invalid gate")
    if not (PQ/"features.parquet").exists():raise ValueError("Canonical feature cache must already exist")
    X,_,y,train_ids,_=v.frames(features());ids=np.asarray(train_ids.to_list())
    tr,va=v.outer(y,a.seed)[a.fold]
    fi,stop=v.inner(y[tr],a.seed,a.fold);fi,stop=tr[fi],tr[stop]
    qs,qc=train_test_split(stop,test_size=.5,stratify=y[stop],random_state=a.seed+a.fold)
    qs,qc=np.sort(qs),np.sort(qc)
    if a.smoke:
        from sklearn.model_selection import StratifiedShuffleSplit
        def subset(rows,offset):
            if len(rows)<=a.smoke_query:return rows
            _,q=next(StratifiedShuffleSplit(n_splits=1,test_size=a.smoke_query,random_state=a.seed+offset).split(np.zeros(len(rows)),y[rows]))
            return np.sort(rows[q])
        qs,qc,va=subset(qs,13),subset(qc,14),subset(va,15)
    assert not np.intersect1d(fi,np.r_[qs,qc,va]).size and not np.intersect1d(qs,qc).size
    cache,inner_identity=read_pool(a.inner_cache,"inner",a.seed,a.fold,ids[fi].tolist())
    friend=teammate_inputs(cache,ids,y,ids[fi]);del cache;gc.collect()
    pools={"utama":X,"teman":friend}
    legacy_columns={name:select_columns(pool,y,fi) for name,pool in pools.items()}
    columns={name:select_columns(pool,y,fi,extend_shortlist=True) if len(legacy_columns[name])<104 else legacy_columns[name]
             for name,pool in pools.items()}
    for name,cols in columns.items():
        if len(cols)!=104:raise ValueError(f"{name}: selector returned {len(cols)} columns; 104-column protocol not met")
    config={k:value for k,value in vars(a).items() if k not in {"out","inner_cache","outer_cache","evaluate","prepare_only"}}
    config.update(estimators=4,model="KumoTabular-large",context="all fitting rows",policies=list(POLICIES),
                  selector="original KS/Spearman; double shortlist if needed to fill100")
    sources=[Path(__file__).name,"nested_input_pools.py","teammate_feature_adapter.py","166_teammate_nested_features.py",
             "152_vipcop_kumo.py","valid2.py","common.py"]
    manifest={"config":config,"sources_sha256":{name:file_hash(ROOT/"eda"/name) for name in sources},
              "packages":{name:importlib.metadata.version(name) for name in ("numpy","pandas","polars","torch","structured-data-models","scikit-learn")},
              "canonical_features_sha256":file_hash(PQ/"features.parquet"),"inner_cache":inner_identity,
              "checkpoint":checkpoint_identity(),
              "protocol_sha256":file_hash(ROOT/"docs/integration/protokol_fitur_teman_validation_v2.md"),
              "smoke_only":a.smoke,"test_rows_in_model":False}
    work=a.out.resolve();work.mkdir(parents=True,exist_ok=True);folder=work/f"s{a.seed}_f{a.fold}"
    folder.mkdir(exist_ok=True);mp=folder/"manifest.json"
    if mp.exists():
        if json.loads(mp.read_text())!=manifest:raise ValueError("Resume rejected: source, data, configuration or runtime changed")
    elif any(folder.iterdir()):raise ValueError("Use a fresh owned experiment directory")
    else:save_json(mp,manifest)
    run_hash=digest(manifest)
    split_ids={name:ids[rows].tolist() for name,rows in (("inner_fit",fi),("search",qs),("confirmation",qc),("outer_train",tr),("outer_validation",va))}
    save_json(folder/"splits.json",{"run_sha256":run_hash,"ids":split_ids})
    audit={"run_sha256":run_hash,"inner_columns":columns,"legacy_shortlist_column_counts":{name:len(cols) for name,cols in legacy_columns.items()},
           "filled_baseline_unchanged":columns["utama"]==legacy_columns["utama"],
           "context_rows":len(fi),"search_rows":len(qs),"confirmation_rows":len(qc),
           "smoke_only":a.smoke,"numeric_missingness":{name:{c:float(pools[name].iloc[fi][c].isna().mean()) for c in cols} for name,cols in columns.items()},
           "category_policy":"vocabulary fitted on real context only; unknown query categories become missing",
           "excluded_history_categories":["v2_prev_last_decision","v2_ecom_last_payment_method"]}
    # Validate both representations before any model is constructed.
    for name in POLICIES:
        family,representation=name.split("_")
        stage=prepare_stage(pools[family],y,ids,fi,{"search":qs,"confirmation":qc},columns[family],representation)
        assert len(stage["frame"])==len(fi)+len(qs)+len(qc)
        assert not stage["labels"][len(fi):].any()
        assert set(stage["ids"]).isdisjoint(ids[va])
        del stage
    save_json(folder/"input_audit.json",audit)
    plot=ROOT/"eda/out"/f"173_{work.name}_s{a.seed}_f{a.fold}_{run_hash[:8]}.png"
    if a.prepare_only:
        fig,ax=plt.subplots(figsize=(8,3));ax.bar(list(columns),[len(cols) for cols in columns.values()])
        ax.set(ylabel="Selected columns",title="Input preparation only; model not evaluated")
        fig.tight_layout();fig.savefig(plot,dpi=130);plt.close(fig)
        print("Prepared inner inputs; no model inference",json.dumps({"columns":{k:len(val) for k,val in columns.items()},"context":len(fi),"search":len(qs),"confirmation":len(qc)}),flush=True)
        return
    import torch
    import sdm
    import nested_input_pools as pool_adapter
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction)
    vip=pool_adapter._vip
    model=None

    def predict(name,pool,real,queries,cols,representation,reference_identity):
        nonlocal model
        stage=prepare_stage(pool,y,ids,real,queries,cols,representation)
        query=np.concatenate(list(stage["queries"].values()))
        fit={"run_sha256":run_hash,"name":name,"columns":cols,"representation":representation,
             "reference_cache":reference_identity,"real_context_ids":ids[real].tolist(),"real_labels":y[real].tolist(),
             "query_ids":stage["ids"][query].tolist(),"query_roles":[key for key,rows in queries.items() for _ in rows],
             "estimators":4,"generator_seed":100*a.seed+a.fold,"chunk":a.chunk,
             "all_real_context_retained":True,"query_labels_passed_to_model":False}
        fp,pp,cp=[folder/f"{name}_{suffix}" for suffix in ("fit.json","predictions.parquet","cache.json")]
        if cp.exists():
            if json.loads(fp.read_text())!=fit:raise ValueError("Cached fit provenance differs")
            if json.loads(cp.read_text())!={"fit_sha256":digest(fit),"predictions_sha256":file_hash(pp)}:raise ValueError("Prediction cache hash differs")
            saved=pl.read_parquet(pp)
            if saved["application_id"].to_list()!=fit["query_ids"] or saved["role"].to_list()!=fit["query_roles"]:raise ValueError("Cached query order/roles differ")
            p=saved["prediction"].to_numpy();print("cached",name,flush=True)
        else:
            save_json(fp,fit)
            if model is None:
                model=sdm.models.KumoTabular(task="classification",size="large",device="cuda");model.eval()
            torch.cuda.reset_peak_memory_stats()
            predictor=vip.Predictor(stage["frame"],stage["labels"],model)
            try:p=predictor.predict_cached(stage["real"],query,cols,4,100*a.seed+a.fold,a.chunk)
            finally:del predictor;gc.collect();torch.cuda.empty_cache()
            save_json(folder/f"{name}_memory.json",{"max_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
                                                   "max_reserved_gib":torch.cuda.max_memory_reserved()/2**30})
            temporary=pp.with_suffix(".parquet.tmp")
            pl.DataFrame({"application_id":fit["query_ids"],"role":fit["query_roles"],"prediction":p}).write_parquet(temporary);temporary.replace(pp)
            save_json(cp,{"fit_sha256":digest(fit),"predictions_sha256":file_hash(pp)})
            print(f"computed {name}: context={len(real)} query={len(query)} representation={representation}",flush=True)
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        result={};start=0
        for role,rows in queries.items():result[role]=p[start:start+len(rows)];start+=len(rows)
        return result

    selection_path=folder/"selection.json"
    if selection_path.exists():
        selection=json.loads(selection_path.read_text())
        if selection["run_sha256"]!=run_hash:raise ValueError("Frozen selection belongs to another run")
    else:
        predictions={};search=[]
        for name in POLICIES:
            family,representation=name.split("_")
            predictions[name]=predict("inner_"+name,pools[family],fi,{"search":qs,"confirmation":qc},columns[family],representation,
                                      inner_identity if family=="teman" else {"canonical_features_sha256":manifest["canonical_features_sha256"]})
            row={"name":name,**score(y[qs],predictions[name]["search"])};search.append(row)
            save_json(folder/"search.json",search);print("search",row,flush=True)
        top=challenger(search,a.min_inner_gain)
        # Confirmation predictions exist, but labels are scored only for baseline
        # and the single search-selected challenger. No confirmation re-selection.
        confirm=[{"name":BASELINE,**score(y[qc],predictions[BASELINE]["confirmation"])}]
        if top is not None:confirm.append({"name":top,**score(y[qc],predictions[top]["confirmation"])})
        selection=freeze_selection(search,confirm,a.min_inner_gain)
        selection.update(run_sha256=run_hash,inner_columns=columns,smoke_only=a.smoke)
        save_json(selection_path,selection)
        print("selection FROZEN before outer",selection["winner"],selection["reason"],flush=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for axis,key in zip(axes,("search","confirmation")):
        rows=selection[key];axis.bar([r["name"] for r in rows],[r["ks"] for r in rows]);axis.tick_params(axis="x",rotation=35)
        axis.set(ylabel="KS",title=f"{'SMOKE ' if a.smoke else ''}Inner {key}; outer not used")
    fig.tight_layout();fig.savefig(plot,dpi=130);plt.close(fig)
    del friend,pools;gc.collect()
    if a.evaluate:
        if json.loads(selection_path.read_text())!=selection:raise ValueError("Frozen selection changed")
        cols=select_columns(X,y,tr,extend_shortlist=True)
        if len(cols)!=104:raise ValueError("Outer baseline violates 104-column protocol")
        baseline=predict("outer_"+BASELINE,X,tr,{"outer":va},cols,"raw",{"canonical_features_sha256":manifest["canonical_features_sha256"]})["outer"]
        winner=selection["winner"]
        if winner==BASELINE:candidate=baseline
        else:
            family,representation=winner.split("_");reference={"canonical_features_sha256":manifest["canonical_features_sha256"]}
            if family=="teman":
                if a.outer_cache is None:raise ValueError("Frozen teammate winner needs a separately rebuilt outer reference cache; provide --outer-cache")
                cache,reference=read_pool(a.outer_cache,"outer",a.seed,a.fold,ids[tr].tolist())
                pool=teammate_inputs(cache,ids,y,ids[tr]);del cache
                cols=select_columns(pool,y,tr,extend_shortlist=True)
                if len(cols)!=104:raise ValueError("Outer teammate pool violates 104-column protocol")
            else:pool=X
            candidate=predict("outer_"+winner,pool,tr,{"outer":va},cols,representation,reference)["outer"]
        output=work/f"s{a.seed}_f{a.fold}_outer.parquet"
        temporary=output.with_suffix(".parquet.tmp")
        pl.DataFrame({"application_id":ids[va],"target":y[va],"baseline":baseline,"candidate":candidate}).write_parquet(temporary);temporary.replace(output)
        metrics={"run_sha256":run_hash,"winner":winner,"smoke_only":a.smoke,"baseline":score(y[va],baseline),"selected":score(y[va],candidate),
                 "selection_sha256":file_hash(selection_path),"outer_rows":len(va),"outer_predictions_sha256":file_hash(output)}
        metrics["delta_ks"]=metrics["selected"]["ks"]-metrics["baseline"]["ks"]
        save_json(folder/"outer_metrics.json",metrics);print("outer FROZEN evaluation",json.dumps(metrics),flush=True)
    print("Artifacts",folder,"plot",plot,flush=True)


if __name__=="__main__":main()
