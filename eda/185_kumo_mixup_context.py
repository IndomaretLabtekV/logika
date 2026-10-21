"""Nested same-class context augmentation with an anchor-duplication control."""
import argparse
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
from nested_input_pools import select_columns
from mixup_context import augment

spec=importlib.util.spec_from_file_location("mixup_runtime",Path(__file__).with_name("183_kumo_group_retrieval.py"))
runtime_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime_module)
base=runtime_module.base
sha,digest,save=base.file_hash,base.digest,base.save_json
PredictionCache=runtime_module.PredictionCache


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-run",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--fold",type=int,choices=range(5),default=0)
    ap.add_argument("--prepare-only",action="store_true")
    ap.add_argument("--evaluate",action="store_true")
    a=ap.parse_args();old=a.baseline_run/f"s{a.seed}_f{a.fold}"
    bm=json.loads((old/"manifest.json").read_text());bh=digest(bm)
    assert not bm["smoke_only"] and (bm["config"]["seed"],bm["config"]["fold"])==(a.seed,a.fold)
    assert sha(PQ/"features.parquet")==bm["canonical_features_sha256"] and base.checkpoint_identity()==bm["checkpoint"]
    for name,expected in bm["sources_sha256"].items():assert sha(ROOT/"eda"/name)==expected
    X,_,y,train_ids,_=v.frames(features());ids=np.asarray(train_ids.to_list())
    tr,va=v.outer(y,a.seed)[a.fold];fi,_=v.inner(y[tr],a.seed,a.fold);fi=tr[fi]
    splits=json.loads((old/"splits.json").read_text());assert splits["run_sha256"]==bh
    assert splits["ids"]["inner_fit"]==ids[fi].tolist() and splits["ids"]["outer_validation"]==ids[va].tolist()
    index={key:i for i,key in enumerate(ids)}
    qs,qc=[np.array([index[key] for key in splits["ids"][role]]) for role in ("search","confirmation")]
    assert len(fi)==48000 and len(qs)==len(qc)==4000 and set(np.r_[qs,qc])==set(tr)-set(fi)
    cols=select_columns(X,y,fi,extend_shortlist=True);assert len(cols)==104
    fp,pp,cp=[old/f"inner_utama_raw_{suffix}" for suffix in ("fit.json","predictions.parquet","cache.json")]
    fit=json.loads(fp.read_text());p=pl.read_parquet(pp)
    assert json.loads(cp.read_text())=={"fit_sha256":digest(fit),"predictions_sha256":sha(pp)}
    assert fit["run_sha256"]==bh and fit["real_context_ids"]==ids[fi].tolist() and fit["real_labels"]==y[fi].tolist() and fit["columns"]==cols
    assert (fit["representation"],fit["estimators"],fit["generator_seed"],fit["chunk"])==("raw",4,100*a.seed+a.fold,128)
    assert p["application_id"].to_list()==ids[np.r_[qs,qc]].tolist()==fit["query_ids"]
    assert p["role"].to_list()==["search"]*4000+["confirmation"]*4000==fit["query_roles"]
    baseline={role:p.filter(pl.col("role")==role)["prediction"].to_numpy() for role in ("search","confirmation")}
    manifest={"baseline_run_sha256":bh,"baseline_fit_sha256":sha(fp),"baseline_predictions_sha256":sha(pp),
              "source_sha256":{name:sha(ROOT/"eda"/name) for name in (Path(__file__).name,"mixup_context.py","183_kumo_group_retrieval.py","grouped_context.py","nested_input_pools.py","173_nested_kumo_inputs.py","152_vipcop_kumo.py","valid2.py","common.py")},
              "protocol_sha256":sha(ROOT/"docs/integration/protokol_kumo_sameclass_mixup.md"),"packages":bm["packages"],"checkpoint":bm["checkpoint"],
              "seed":a.seed,"fold":a.fold,"caps":[2000,8000],"estimators":4,"chunk":128,"gpu_memory_fraction":.7,"smoke_only":False,"min_inner_gain":.003}
    work=a.out.resolve();work.mkdir(parents=True,exist_ok=True);folder=work/f"s{a.seed}_f{a.fold}";folder.mkdir(exist_ok=True)
    mp=folder/"manifest.json"
    if mp.exists():assert json.loads(mp.read_text())==manifest,"Changed source/data/configuration; use new directory"
    elif any(folder.iterdir()):raise ValueError("Use fresh owned output")
    else:save(mp,manifest)
    run_hash=digest(manifest);save(folder/"splits.json",{"run_sha256":run_hash,"ids":splits["ids"]})

    def augmented(name,real,columns,cap,clone=False):
        seed=100*a.seed+a.fold+701+cap
        request={"run_sha256":run_hash,"name":name,"real_ids":ids[real].tolist(),"real_labels":y[real].tolist(),
                 "columns":columns,"cap":cap,"seed":seed,"clone":clone}
        meta_path=folder/f"{name}_augmentation.json";data_path=folder/f"{name}_features.parquet";parents_path=folder/f"{name}_parents.parquet"
        if meta_path.exists():
            saved=json.loads(meta_path.read_text());assert saved["request"]==request
            assert sha(data_path)==saved["features_sha256"] and sha(parents_path)==saved["parents_sha256"]
            synthetic=pd.read_parquet(data_path);parents=pd.read_parquet(parents_path)
        else:
            prefix=f"SYNTH_s{a.seed}_f{a.fold}_{name}"
            synthetic,parents,meta=augment(X,y,ids,real,columns,cap,seed,prefix,clone=clone)
            for path,frame in ((data_path,synthetic),(parents_path,parents)):
                temporary=path.with_suffix(".parquet.tmp");frame.to_parquet(temporary,index=False);temporary.replace(path)
            saved={"request":request,"features_sha256":sha(data_path),"parents_sha256":sha(parents_path),"augmentation":meta,
                   "label_status":"same-class augmentation assumption; not newly observed targets"}
            save(meta_path,saved)
        assert len(synthetic)==len(parents)==cap and parents.synthetic_id.is_unique
        assert set(parents.parent_a)<=set(ids[real]) and set(parents.parent_b)<=set(ids[real])
        assert set(parents.synthetic_id).isdisjoint(ids)
        truth=dict(zip(ids[real],y[real]))
        assert parents.label.tolist()==[truth[key] for key in parents.parent_a]==[truth[key] for key in parents.parent_b]
        return synthetic,parents,saved,meta_path

    specs={"clone8000":(8000,True),"mix2000":(2000,False),"mix8000":(8000,False)}
    frames=[X];targets=[y];all_ids=[ids];contexts={};details={};parents_all={};offset=len(X)
    for name,(cap,clone) in specs.items():
        frame,parents,meta,path=augmented("inner_"+name,fi,cols,cap,clone)
        frames.append(frame);targets.append(parents.label.to_numpy());all_ids.append(parents.synthetic_id.to_numpy())
        contexts[name]=np.r_[fi,np.arange(offset,offset+cap)];offset+=cap
        details[name]={"augmentation_sha256":sha(path),"metadata":meta["augmentation"],"total_context":len(contexts[name])}
        parents_all[name]=parents
    pool=pd.concat(frames,ignore_index=True);labels=np.concatenate(targets);pool_ids=np.concatenate(all_ids)
    assert len(pool_ids)==len(set(pool_ids))==len(pool) and len(pool)==offset
    for name,rows in contexts.items():
        assert np.array_equal(rows[:len(fi)],fi) and set(pool_ids[rows]).isdisjoint(ids[np.r_[qs,qc,va]])
        assert np.array_equal(labels[rows[:len(fi)]],y[fi])
    audit={"run_sha256":run_hash,"real_context":len(fi),"columns":cols,"candidates":details,"query_rows":len(qs)+len(qc),
           "outer_ids_in_inner_context":False,"all_real_context_retained":True,"synthetic_labels_independent_observations":False}
    save(folder/"input_audit.json",audit)
    plot=OUT/f"185_{work.name}_s{a.seed}_f{a.fold}_{run_hash[:8]}.png"
    if a.prepare_only:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for name in ("mix2000","mix8000"):axes[0].hist(parents_all[name]["lambda"],bins=20,alpha=.5,density=True,label=name)
        axes[0].set(xlabel="Mixing coefficient",ylabel="Density",title="Same-class Beta(0.2,0.2) augmentation");axes[0].legend()
        axes[1].bar(["real",*specs],[y[fi].mean(),*[parents_all[name].label.mean() for name in specs]])
        axes[1].set(ylabel="Positive fraction",title="Synthetic prior follows real context; no class balance")
        fig.tight_layout();fig.savefig(plot,dpi=130);plt.close(fig)
        print("Prepared augmentation; no Kumo inference",json.dumps(details),flush=True);return
    runtime=PredictionCache(pool,labels,pool_ids,folder,run_hash,a.seed,a.fold)
    def score(rows,p):return {"ks":v.ks(y[rows],p),"auc":float(roc_auc_score(y[rows],p))}
    selection_path=folder/"selection.json"
    if selection_path.exists():selection=json.loads(selection_path.read_text());assert selection["run_sha256"]==run_hash
    else:
        p=runtime.predict("inner_clone_control",contexts["clone8000"],{"search":qs},cols,details["clone8000"])["search"]
        save(folder/"clone_control.json",{"score":score(qs,p),"context":56000,"synthetic":8000,"promotion_candidate":False})
        print("clone control search",score(qs,p),flush=True)
        predictions={};search=[{"name":"utama_raw",**score(qs,baseline["search"])}]
        for name in ("mix2000","mix8000"):
            predictions[name]=runtime.predict("inner_"+name,contexts[name],{"search":qs,"confirmation":qc},cols,details[name])
            row={"name":name,**score(qs,predictions[name]["search"])};search.append(row);save(folder/"search.json",search);print("search",row,flush=True)
        predictions["probability_blend"]={role:.5*(baseline[role]+predictions["mix8000"][role]) for role in baseline}
        search.append({"name":"probability_blend",**score(qs,predictions["probability_blend"]["search"])})
        save(folder/"search.json",search);print("search",search[-1],flush=True)
        best=max(search,key=lambda row:row["ks"]);winner="utama_raw";confirm=[{"name":winner,**score(qc,baseline["confirmation"])}]
        if best["name"]!=winner and best["ks"]-search[0]["ks"]>.003+1e-12:
            confirm.append({"name":best["name"],**score(qc,predictions[best["name"]]["confirmation"])})
            if confirm[-1]["ks"]-confirm[0]["ks"]>.003+1e-12:winner=best["name"]
        selection={"run_sha256":run_hash,"winner":winner,"search":search,"confirmation":confirm,"outer_seen":False,"min_inner_gain":.003,
                   "confirmation_reuse":"same inner split as prior research; not independent project evidence","all_real_context_retained":True}
        save(selection_path,selection);print("selection FROZEN before outer",winner,flush=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,role in zip(axes,("search","confirmation")):
        rows=selection[role];ax.bar([r["name"] for r in rows],[r["ks"]-rows[0]["ks"] for r in rows]);ax.axhline(.003,color="red",linestyle=":")
        ax.tick_params(axis="x",rotation=40);ax.set(ylabel="Delta KS vs real-only context",title="Inner "+role)
    fig.tight_layout();fig.savefig(plot,dpi=130);plt.close(fig)
    if a.evaluate:
        assert json.loads(selection_path.read_text())==selection
        outer_cols=select_columns(X,y,tr,extend_shortlist=True)
        op=runtime.predict("outer_utama_raw",tr,{"outer":va},outer_cols,{"kind":"real-only-baseline"})["outer"]
        winner=selection["winner"]
        if winner=="utama_raw":candidate=op
        else:
            cap=2000 if winner=="mix2000" else 8000
            frame,parents,meta,path=augmented(f"outer_mix{cap}",tr,outer_cols,cap)
            outer_pool=pd.concat([X,frame],ignore_index=True);outer_y=np.r_[y,parents.label.to_numpy()];outer_ids=np.r_[ids,parents.synthetic_id.to_numpy()]
            outer_runtime=PredictionCache(outer_pool,outer_y,outer_ids,folder,run_hash,a.seed,a.fold);outer_runtime.model=runtime.model
            real=np.r_[tr,np.arange(len(X),len(outer_pool))]
            p=outer_runtime.predict("outer_"+winner,real,{"outer":va},outer_cols,{"augmentation_sha256":sha(path),"metadata":meta["augmentation"]})["outer"]
            candidate=.5*(op+p) if winner=="probability_blend" else p
        output=work/f"s{a.seed}_f{a.fold}_outer.parquet";temporary=output.with_suffix(".parquet.tmp")
        pl.DataFrame({"application_id":ids[va],"target":y[va],"baseline":op,"candidate":candidate}).write_parquet(temporary);temporary.replace(output)
        metrics={"run_sha256":run_hash,"winner":winner,"baseline":score(va,op),"selected":score(va,candidate),"selection_sha256":sha(selection_path),"outer_rows":len(va),"outer_predictions_sha256":sha(output),"smoke_only":False}
        metrics["delta_ks"]=metrics["selected"]["ks"]-metrics["baseline"]["ks"];save(folder/"outer_metrics.json",metrics)
        print("outer FROZEN evaluation",json.dumps(metrics),flush=True)
    print("Artifacts",folder,"plot",plot,flush=True)


if __name__=="__main__":main()
