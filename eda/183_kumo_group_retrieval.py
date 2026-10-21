"""Nested cluster-shared retrieval context for Kumo, with an explicit size control."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import valid2 as v
from common import ROOT,PQ,OUT,features
from nested_input_pools import prepare_stage,select_columns
from grouped_context import neighborhoods

spec=importlib.util.spec_from_file_location("retrieval_base",Path(__file__).with_name("173_nested_kumo_inputs.py"))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
sha,digest,save=base.file_hash,base.digest,base.save_json


class PredictionCache:
    """ID-bearing fit/prediction cache; real context is explicitly supplied."""
    def __init__(self,X,y,ids,folder,run_hash,seed,fold):
        self.X,self.y,self.ids=X,np.asarray(y),np.asarray(ids)
        self.folder,self.run_hash,self.seed,self.fold=folder,run_hash,seed,fold
        self.model=None

    def predict(self,name,real,queries,columns,reference,representation="raw"):
        stage=prepare_stage(self.X,self.y,self.ids,real,queries,columns,representation)
        query=np.concatenate(list(stage["queries"].values()))
        assert not stage["labels"][len(real):].any()
        fit={"run_sha256":self.run_hash,"name":name,"columns":columns,"representation":representation,
             "reference":reference,"real_context_ids":self.ids[real].tolist(),"real_labels":self.y[real].tolist(),
             "query_ids":stage["ids"][query].tolist(),"query_roles":[role for role,rows in queries.items() for _ in rows],
             "estimators":4,"generator_seed":100*self.seed+self.fold,"chunk":128,
             "query_labels_passed_to_model":False,"context_size":len(real)}
        fp,pp,cp=[self.folder/f"{name}_{suffix}" for suffix in ("fit.json","predictions.parquet","cache.json")]
        if cp.exists():
            assert json.loads(fp.read_text())==fit
            assert json.loads(cp.read_text())=={"fit_sha256":digest(fit),"predictions_sha256":sha(pp)}
            saved=pl.read_parquet(pp);assert saved["application_id"].to_list()==fit["query_ids"] and saved["role"].to_list()==fit["query_roles"]
            p=saved["prediction"].to_numpy();print("cached",name,flush=True)
        else:
            import torch
            import sdm
            import nested_input_pools as adapter
            save(fp,fit)
            if self.model is None:
                torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.7)
                self.model=sdm.models.KumoTabular(task="classification",size="large",device="cuda");self.model.eval()
            torch.cuda.reset_peak_memory_stats();predictor=None
            try:
                predictor=adapter._vip.Predictor(stage["frame"],stage["labels"],self.model)
                p=predictor.predict_cached(stage["real"],query,columns,4,100*self.seed+self.fold,128)
            finally:del predictor;gc.collect();torch.cuda.empty_cache()
            save(self.folder/f"{name}_memory.json",{"max_allocated_gib":torch.cuda.max_memory_allocated()/2**30,"max_reserved_gib":torch.cuda.max_memory_reserved()/2**30})
            temporary=pp.with_suffix(".parquet.tmp")
            pl.DataFrame({"application_id":fit["query_ids"],"role":fit["query_roles"],"prediction":p}).write_parquet(temporary);temporary.replace(pp)
            save(cp,{"fit_sha256":digest(fit),"predictions_sha256":sha(pp)})
            print("computed",name,"context",len(real),"query",len(query),flush=True)
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        result={};start=0
        for role,rows in queries.items():result[role]=p[start:start+len(rows)];start+=len(rows)
        return result


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
    X,_,y,train_ids,_=v.frames(features());ids=np.asarray(train_ids.to_list());index={key:i for i,key in enumerate(ids)}
    tr,va=v.outer(y,a.seed)[a.fold];fi,_=v.inner(y[tr],a.seed,a.fold);fi=tr[fi]
    splits=json.loads((old/"splits.json").read_text());assert splits["run_sha256"]==bh
    assert splits["ids"]["inner_fit"]==ids[fi].tolist() and splits["ids"]["outer_validation"]==ids[va].tolist()
    qs,qc=[np.array([index[key] for key in splits["ids"][role]]) for role in ("search","confirmation")]
    assert len(fi)==48000 and len(qs)==len(qc)==4000 and set(np.r_[qs,qc])==set(tr)-set(fi)
    columns=select_columns(X,y,fi,extend_shortlist=True);assert len(columns)==104
    fp,pp,cp=[old/f"inner_utama_raw_{suffix}" for suffix in ("fit.json","predictions.parquet","cache.json")]
    fit=json.loads(fp.read_text());p=pl.read_parquet(pp)
    assert json.loads(cp.read_text())=={"fit_sha256":digest(fit),"predictions_sha256":sha(pp)}
    assert fit["run_sha256"]==bh and fit["real_context_ids"]==ids[fi].tolist() and fit["real_labels"]==y[fi].tolist() and fit["columns"]==columns
    assert (fit["representation"],fit["estimators"],fit["generator_seed"],fit["chunk"])==("raw",4,100*a.seed+a.fold,128)
    assert p["application_id"].to_list()==ids[np.r_[qs,qc]].tolist()==fit["query_ids"]
    assert p["role"].to_list()==["search"]*4000+["confirmation"]*4000==fit["query_roles"]
    baseline={role:p.filter(pl.col("role")==role)["prediction"].to_numpy() for role in ("search","confirmation")}
    manifest={"baseline_run_sha256":bh,"baseline_fit_sha256":sha(fp),"baseline_predictions_sha256":sha(pp),
              "source_sha256":{name:sha(ROOT/"eda"/name) for name in (Path(__file__).name,"grouped_context.py","nested_input_pools.py","173_nested_kumo_inputs.py","152_vipcop_kumo.py","valid2.py","common.py")},
              "protocol_sha256":sha(ROOT/"docs/integration/protokol_kumo_group_retrieval.md"),"packages":bm["packages"],"checkpoint":bm["checkpoint"],
              "seed":a.seed,"fold":a.fold,"clusters":8,"context_per_group":12000,"estimators":4,"chunk":128,
              "gpu_memory_fraction":.7,"smoke_only":False,"min_inner_gain":.003}
    work=a.out.resolve();work.mkdir(parents=True,exist_ok=True);folder=work/f"s{a.seed}_f{a.fold}";folder.mkdir(exist_ok=True)
    mp=folder/"manifest.json"
    if mp.exists():assert json.loads(mp.read_text())==manifest,"Changed inputs/source; use new directory"
    elif any(folder.iterdir()):raise ValueError("Use fresh owned output")
    else:save(mp,manifest)
    run_hash=digest(manifest);save(folder/"splits.json",{"run_sha256":run_hash,"ids":splits["ids"]})

    def plan(name,real,queries,cols):
        path=folder/f"{name}_plan.json";rows=np.concatenate(list(queries.values()))
        request={"run_sha256":run_hash,"real_ids":ids[real].tolist(),"query_ids":ids[rows].tolist(),"columns":cols}
        if path.exists():
            saved=json.loads(path.read_text());assert saved["request"]==request
            assert digest(saved["geometry"])==saved["geometry_sha256"]
            contexts={int(key):np.array([index[item] for item in items]) for key,items in saved["context_ids"].items()}
            assignment=np.asarray(saved["assignment"],dtype=int)
        else:
            result=neighborhoods(X,real,rows,cols,seed=100*a.seed+a.fold)
            contexts,assignment=result["contexts"],result["assignment"]
            saved={"request":request,"geometry":result["geometry"],"geometry_sha256":result["geometry_sha256"],
                   "context_ids":{str(key):ids[value].tolist() for key,value in contexts.items()},"assignment":assignment.tolist(),
                   "unknown_query_categories":result["unknown_query_categories"],"fit_cluster_counts":result["fit_cluster_counts"],
                   "query_cluster_counts":result["query_cluster_counts"]}
            save(path,saved)
        assert len(assignment)==len(rows) and set(assignment)<=set(contexts)
        for ctx in contexts.values():
            assert len(ctx)==len(set(ctx))==12000 and set(ctx)<=set(real) and set(ctx).isdisjoint(rows) and len(set(y[ctx]))==2
        return contexts,assignment,saved,path

    contexts,assignment,geometry,plan_path=plan("inner",fi,{"search":qs,"confirmation":qc},columns)
    audit={"run_sha256":run_hash,"geometry_sha256":geometry["geometry_sha256"],"columns":columns,
           "context_size":12000,"full_baseline_context_size":len(fi),"fit_cluster_counts":geometry["fit_cluster_counts"],
           "query_cluster_counts":geometry["query_cluster_counts"],"context_positive_rates":{str(key):float(y[ctx].mean()) for key,ctx in contexts.items()},
           "unknown_retrieval_categories":geometry["unknown_query_categories"],"inner_outer_disjoint":True,"geometry_query_assignment_exactly_once":True}
    save(folder/"input_audit.json",audit)
    plot=OUT/f"183_{work.name}_s{a.seed}_f{a.fold}_{run_hash[:8]}.png"
    if a.prepare_only:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].bar(np.arange(8),geometry["query_cluster_counts"]);axes[0].set(xlabel="Retrieval group",ylabel="Queries",title="Covariate-only assignment")
        axes[1].bar(np.arange(8),[y[contexts[key]].mean() for key in range(8)]);axes[1].set(xlabel="Retrieval group",ylabel="Context positive rate",title="12k nearest contexts; no class balancing")
        fig.tight_layout();fig.savefig(plot,dpi=130);plt.close(fig)
        print("Prepared retrieval geometry; no Kumo inference",json.dumps(audit),flush=True);return
    runtime=PredictionCache(X,y,ids,folder,run_hash,a.seed,a.fold)
    def score(rows,p):return {"ks":v.ks(y[rows],p),"auc":float(roc_auc_score(y[rows],p))}
    def retrieve(name,queries,cols,contexts,assignment,identity):
        out={role:np.full(len(rows),np.nan) for role,rows in queries.items()};coverage={role:np.zeros(len(rows),dtype=int) for role,rows in queries.items()}
        offset={};start=0
        for role,rows in queries.items():offset[role]=assignment[start:start+len(rows)];start+=len(rows)
        for key,ctx in contexts.items():
            group={role:rows[offset[role]==key] for role,rows in queries.items() if np.any(offset[role]==key)}
            if not group:continue
            result=runtime.predict(f"{name}_group{key}",ctx,group,cols,identity)
            for role,p in result.items():
                mask=offset[role]==key;assert not coverage[role][mask].any();out[role][mask]=p;coverage[role][mask]+=1
        assert all((counts==1).all() for counts in coverage.values()) and all(np.isfinite(p).all() for p in out.values())
        return out
    def combine(role,retrieved,name,base_prediction=baseline):
        if name=="retrieval":return retrieved[role]
        if name=="probability_blend":return .5*(base_prediction[role]+retrieved[role])
        if name=="rank_blend":return (rankdata(base_prediction[role],method="average")+rankdata(retrieved[role],method="average"))/(2*len(retrieved[role]))
        return base_prediction[role]
    selection_path=folder/"selection.json"
    if selection_path.exists():selection=json.loads(selection_path.read_text());assert selection["run_sha256"]==run_hash
    else:
        control=np.sort(np.random.default_rng(100*a.seed+a.fold+501).choice(fi,12000,replace=False))
        p=runtime.predict("inner_uniform_control",control,{"search":qs},columns,{"kind":"uniform-size-control","promotion_candidate":False})["search"]
        save(folder/"size_control.json",{"name":"uniform12000","context_ids":ids[control].tolist(),"score":score(qs,p),"promotion_candidate":False})
        print("size control search",score(qs,p),flush=True)
        retrieved=retrieve("inner",{"search":qs,"confirmation":qc},columns,contexts,assignment,{"plan_sha256":sha(plan_path)})
        search=[{"name":"utama_raw",**score(qs,baseline["search"])}]
        search.extend({"name":name,**score(qs,combine("search",retrieved,name))} for name in ("retrieval","probability_blend","rank_blend"))
        for row in search:print("search",row,flush=True)
        winner="utama_raw";confirm=[{"name":winner,**score(qc,baseline["confirmation"])}];best=max(search,key=lambda row:row["ks"])
        if best["name"]!=winner and best["ks"]-search[0]["ks"]>.003+1e-12:
            confirm.append({"name":best["name"],**score(qc,combine("confirmation",retrieved,best["name"]))})
            if confirm[-1]["ks"]-confirm[0]["ks"]>.003+1e-12:winner=best["name"]
        selection={"run_sha256":run_hash,"winner":winner,"search":search,"confirmation":confirm,"outer_seen":False,"min_inner_gain":.003,
                   "confirmation_reuse":"same inner split as prior research; not independent project evidence","context_per_group":12000,"clusters":8}
        save(selection_path,selection);print("selection FROZEN before outer",winner,flush=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,role in zip(axes,("search","confirmation")):
        rows=selection[role];ax.bar([r["name"] for r in rows],[r["ks"]-rows[0]["ks"] for r in rows]);ax.axhline(.003,color="red",linestyle=":")
        ax.tick_params(axis="x",rotation=40);ax.set(ylabel="Delta KS vs full context",title="Inner "+role)
    fig.tight_layout();fig.savefig(plot,dpi=130);plt.close(fig)
    if a.evaluate:
        assert json.loads(selection_path.read_text())==selection
        cols=select_columns(X,y,tr,extend_shortlist=True)
        op=runtime.predict("outer_utama_raw",tr,{"outer":va},cols,{"kind":"full-context-baseline"})["outer"]
        if selection["winner"]=="utama_raw":candidate=op
        else:
            ctx,assignment,_,path=plan("outer",tr,{"outer":va},cols)
            retrieved=retrieve("outer",{"outer":va},cols,ctx,assignment,{"plan_sha256":sha(path)})
            candidate=combine("outer",retrieved,selection["winner"],{"outer":op})
        output=work/f"s{a.seed}_f{a.fold}_outer.parquet";temporary=output.with_suffix(".parquet.tmp")
        pl.DataFrame({"application_id":ids[va],"target":y[va],"baseline":op,"candidate":candidate}).write_parquet(temporary);temporary.replace(output)
        metrics={"run_sha256":run_hash,"winner":selection["winner"],"baseline":score(va,op),"selected":score(va,candidate),"selection_sha256":sha(selection_path),"outer_rows":len(va),"outer_predictions_sha256":sha(output),"smoke_only":False}
        metrics["delta_ks"]=metrics["selected"]["ks"]-metrics["baseline"]["ks"];save(folder/"outer_metrics.json",metrics)
        print("outer FROZEN evaluation",json.dumps(metrics),flush=True)
    print("Artifacts",folder,"plot",plot,flush=True)


if __name__=="__main__":main()
