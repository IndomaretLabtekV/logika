"""Evaluate a prescribed fixed Kumo input blend on untouched outer applications.

No inner gain gate or weight search: this evaluates a NEW fixed policy. Old
173/177 selection fallbacks remain unchanged. The full user breakthrough gate
is assessed separately with155 after five folds and at least three seeds.
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
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import valid2 as v
from common import ROOT,PQ,OUT,features
from conditional_input_pools import gain_columns,union_inputs
from nested_input_pools import teammate_inputs,select_columns

spec=importlib.util.spec_from_file_location("fixed_runtime",Path(__file__).with_name("183_kumo_group_retrieval.py"))
runtime_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime_module)
base=runtime_module.base
sha,digest,save=base.file_hash,base.digest,base.save_json
PredictionCache=runtime_module.PredictionCache


def fuse(primary,union):
    primary,union=np.asarray(primary),np.asarray(union)
    if primary.ndim!=1 or union.ndim!=1 or len(primary)!=len(union) or len(primary)<2:
        raise ValueError("Unaligned/empty constituent predictions")
    if not all(np.isfinite(p).all() and ((p>=0)&(p<=1)).all() for p in (primary,union)):
        raise ValueError("Invalid constituent probabilities")
    a=rankdata(primary,method="average").astype(np.float64)
    b=rankdata(union,method="average").astype(np.float64)
    n=len(primary)
    return {"baseline":a/n,"candidate":(a+b)/(2*n),"union_rank":b/n}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature-cache",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--fold",type=int,choices=range(5),default=0)
    ap.add_argument("--prepare-only",action="store_true",help="Fit only the prescribed training feature selectors")
    a=ap.parse_args()
    assert (PQ/"features.parquet").exists()
    X,_,y,train_ids,_=v.frames(features());ids=np.asarray(train_ids.to_list())
    tr,va=v.outer(y,a.seed)[a.fold]
    assert len(tr)==56000 and len(va)==14000 and set(tr).isdisjoint(va)
    frame,reference=base.read_pool(a.feature_cache,"outer",a.seed,a.fold,ids[tr].tolist())
    friend=teammate_inputs(frame,ids,y,ids[tr]);del frame
    U=union_inputs(X,friend)
    policy={"mode":"fixed_policy","primary":"KS/Spearman100+4raw","union":"LGBgain400+Spearman100+4raw",
            "fusion":"0.5rank(primary)+0.5rank(union); cohort average-ties float64",
            "matched_baseline":"rank(primary) on the same cohort; raw probability retained separately",
            "estimators":4,"real_context":"all outer training rows","chunk":128,"gpu_memory_fraction":.7,
            "generator":"100*seed+fold","outer_used_for_policy_choice":False,"inner_gate_not_applied":True}
    sources=(Path(__file__).name,"conditional_input_pools.py","nested_input_pools.py","teammate_feature_adapter.py",
             "166_teammate_nested_features.py","183_kumo_group_retrieval.py","grouped_context.py","173_nested_kumo_inputs.py",
             "152_vipcop_kumo.py","valid2.py","common.py")
    manifest={"policy":policy,"seed":a.seed,"fold":a.fold,"sources_sha256":{name:sha(ROOT/"eda"/name) for name in sources},
              "features_sha256":sha(PQ/"features.parquet"),"teammate_reference":reference,"checkpoint":base.checkpoint_identity(),
              "packages":{p:importlib.metadata.version(p) for p in ("numpy","pandas","polars","torch","structured-data-models","scikit-learn","lightgbm")},
              "protocol_sha256":sha(ROOT/"docs/integration/protokol_fixed_union_rank_outer.md"),
              "development_inner_choice_sha256":sha(ROOT/"results/v51/conditional_blends_f0/selection.json"),"smoke_only":False}
    work=a.out.resolve();work.mkdir(parents=True,exist_ok=True);folder=work/f"s{a.seed}_f{a.fold}";folder.mkdir(exist_ok=True)
    mp=folder/"manifest.json"
    if mp.exists():assert json.loads(mp.read_text())==manifest,"Changed policy/source/data; use a fresh experiment directory"
    elif any(folder.iterdir()):raise ValueError("Use fresh owned output")
    else:save(mp,manifest)
    run_hash=digest(manifest)
    frozen=folder/"frozen_policy.json"
    frozen_value={"run_sha256":run_hash,"policy":policy,"outer_scores_seen":False,"selection_type":"fixed before this outer evaluation"}
    if frozen.exists():assert json.loads(frozen.read_text())==frozen_value
    else:save(frozen,frozen_value)
    save(folder/"splits.json",{"run_sha256":run_hash,"outer_train_ids":ids[tr].tolist(),"outer_validation_ids":ids[va].tolist()})
    primary_columns=select_columns(X,y,tr,extend_shortlist=True);assert len(primary_columns)==104
    selector_path=folder/"union_selector.json";gain_path=folder/"union_gain.csv"
    request={"run_sha256":run_hash,"fit_ids":ids[tr].tolist(),"fit_labels":y[tr].tolist(),"pool_columns":U.columns.tolist()}
    if selector_path.exists():
        selector=json.loads(selector_path.read_text());assert selector["request"]==request and selector["gain_sha256"]==sha(gain_path)
        union_columns=selector["columns"]
    else:
        union_columns,gain,metadata=gain_columns(U,y,tr,seed=100*a.seed+a.fold)
        gain.sort_values(ascending=False,kind="stable").rename_axis("feature").to_csv(gain_path)
        selector={"request":request,"columns":union_columns,"selector":metadata,"gain_sha256":sha(gain_path)}
        save(selector_path,selector)
    assert len(union_columns)==104
    audit={"run_sha256":run_hash,"primary_columns":primary_columns,"union_columns":union_columns,
           "context_rows":len(tr),"outer_query_rows":len(va),"union_numeric_pool":U.shape[1]-4,
           "selected_teammate_numeric":sum(c.startswith("teman__") for c in union_columns),
           "outer_training_reference_only":True,"frozen_policy_sha256":sha(frozen),"feature_selector_sha256":sha(selector_path)}
    save(folder/"input_audit.json",audit)
    plot=OUT/f"189_{work.name}_s{a.seed}_f{a.fold}_{run_hash[:8]}.png"
    if a.prepare_only:
        fig,ax=plt.subplots(figsize=(8,3));ax.bar(["primary","union"],[len(primary_columns),len(union_columns)])
        ax.set(ylabel="Selected columns",title="Fixed policy; selectors fit on outer training only")
        fig.tight_layout();fig.savefig(plot,dpi=130);plt.close(fig)
        print("Prepared fixed-policy selectors; no outer predictions",json.dumps(audit),flush=True);return
    assert json.loads(frozen.read_text())==frozen_value
    runtime=PredictionCache(X,y,ids,folder,run_hash,a.seed,a.fold)
    primary=runtime.predict("outer_primary",tr,{"outer":va},primary_columns,{"family":"primary","features_sha256":manifest["features_sha256"]})["outer"]
    union_runtime=PredictionCache(U,y,ids,folder,run_hash,a.seed,a.fold);union_runtime.model=runtime.model
    union=union_runtime.predict("outer_union",tr,{"outer":va},union_columns,{"reference":reference,"selector_sha256":sha(selector_path)})["outer"]
    fused=fuse(primary,union)
    def score(p):return {"ks":v.ks(y[va],p),"auc":float(roc_auc_score(y[va],p))}
    baseline_raw=score(primary);baseline_rank=score(fused["baseline"])
    assert abs(baseline_raw["ks"]-baseline_rank["ks"])<1e-6 and abs(baseline_raw["auc"]-baseline_rank["auc"])<1e-9
    output=work/f"s{a.seed}_f{a.fold}_outer.parquet";temporary=output.with_suffix(".parquet.tmp")
    pl.DataFrame({"application_id":ids[va],"target":y[va],**fused,"baseline_raw":primary,"union_raw":union}).write_parquet(temporary);temporary.replace(output)
    metrics={"run_sha256":run_hash,"mode":"fixed_policy","seed":a.seed,"fold":a.fold,"rows":len(va),
             "baseline":baseline_rank,"baseline_raw":baseline_raw,"union":score(union),"candidate":score(fused["candidate"]),
             "outer_predictions_sha256":sha(output),"frozen_policy_sha256":sha(frozen),"smoke_only":False,
             "complete_5fold_3seed":False,"submission_allowed":False}
    metrics["delta_ks"]=metrics["candidate"]["ks"]-metrics["baseline"]["ks"]
    save(folder/"outer_metrics.json",metrics)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(["primary","union","fixed blend"],[baseline_rank["ks"],metrics["union"]["ks"],metrics["candidate"]["ks"]]);axes[0].set(ylabel="Outer KS",title="Fixed policy; no weight selection")
    sample=np.random.default_rng(189).choice(len(va),min(1500,len(va)),replace=False)
    axes[1].scatter(fused["baseline"][sample],fused["union_rank"][sample],s=3,alpha=.2);axes[1].set(xlabel="Primary rank",ylabel="Union rank",title="Paired outer constituent predictions")
    fig.tight_layout();fig.savefig(plot,dpi=130);plt.close(fig)
    print("Fixed outer evaluation",json.dumps(metrics),flush=True);print("Artifacts",folder,"plot",plot,flush=True)


if __name__=="__main__":main()
