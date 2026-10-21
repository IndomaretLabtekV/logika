"""Prescribed six-weight rank fusion, selected inner; no model fitting or outer."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from common import ROOT,PQ,OUT
import valid2 as v


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--baseline-run",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--fold",type=int,default=0)
    a=ap.parse_args();folder=a.run/f"s{a.seed}_f{a.fold}";old=a.baseline_run/f"s{a.seed}_f{a.fold}"
    manifest=json.loads((folder/"manifest.json").read_text());run_hash=digest(manifest)
    bm=json.loads((old/"manifest.json").read_text());bh=digest(bm)
    assert not manifest["smoke_only"] and not bm["smoke_only"] and manifest["baseline_run_sha256"]==bh
    assert (manifest["seed"],manifest["fold"])==(a.seed,a.fold)
    for name,expected in manifest["source_sha256"].items():assert sha(ROOT/"eda"/name)==expected
    assert sha(PQ/"features.parquet")==bm["canonical_features_sha256"]
    splits=json.loads((folder/"splits.json").read_text());assert splits["run_sha256"]==run_hash
    ids=splits["ids"]
    truth_frame=pl.read_parquet(PQ/"features.parquet",columns=["application_id","target","is_train"]).filter(pl.col("is_train")==1)
    truth=dict(zip(truth_frame["application_id"],truth_frame["target"]))
    values={};inputs={}
    for name,source,expected_run in (("utama_raw",old,bh),("union_gain",folder,run_hash)):
        fp,pp,cp=[source/f"inner_{name}_{suffix}" for suffix in ("fit.json","predictions.parquet","cache.json")]
        fit=json.loads(fp.read_text());cache=json.loads(cp.read_text());p=pl.read_parquet(pp)
        assert cache=={"fit_sha256":digest(fit),"predictions_sha256":sha(pp)} and fit["run_sha256"]==expected_run
        assert fit["real_context_ids"]==ids["inner_fit"] and fit["real_labels"]==[truth[key] for key in ids["inner_fit"]]
        assert (fit["representation"],fit["estimators"],fit["generator_seed"],fit["chunk"])==("raw",4,100*a.seed+a.fold,128)
        assert p["application_id"].to_list()==fit["query_ids"] and p["role"].to_list()==fit["query_roles"]
        assert p.height==p["application_id"].n_unique() and set(p["application_id"]).isdisjoint(ids["inner_fit"]+ids["outer_validation"])
        values[name]={}
        for role in ("search","confirmation"):
            q=p.filter(pl.col("role")==role);assert q["application_id"].to_list()==ids[role]
            values[name][role]=q["prediction"].to_numpy()
            assert np.isfinite(values[name][role]).all() and ((values[name][role]>=0)&(values[name][role]<=1)).all()
        inputs[name]={"fit_sha256":sha(fp),"predictions_sha256":sha(pp)}
    weights=[0.,.1,.2,.3,.4,.5]
    metadata={"source_run_sha256":run_hash,"baseline_run_sha256":bh,"inputs":inputs,"weights":weights,"min_inner_gain":.003,
              "source_sha256":sha(Path(__file__).resolve()),"protocol_sha256":sha(ROOT/"docs/integration/protokol_rankfusion_conditional.md")}
    a.out.mkdir(parents=True,exist_ok=True);mp=a.out/"manifest.json"
    if mp.exists():assert json.loads(mp.read_text())==metadata
    elif any(a.out.iterdir()):raise ValueError("Use fresh owned output")
    else:mp.write_text(json.dumps(metadata,indent=2))
    y={role:np.array([truth[key] for key in ids[role]],dtype=int) for role in ("search","confirmation")}
    rank={name:{role:rankdata(p,method="average")/len(p) for role,p in roles.items()} for name,roles in values.items()}
    def score(role,weight):
        # Alpha0 uses the exact baseline probabilities, preserving original ties.
        p=values["utama_raw"][role] if weight==0 else (1-weight)*rank["utama_raw"][role]+weight*rank["union_gain"][role]
        return {"union_weight":weight,"ks":v.ks(y[role],p),"auc":float(roc_auc_score(y[role],p))}
    search=[score("search",weight) for weight in weights];best=max(search,key=lambda row:row["ks"])
    confirmation=[score("confirmation",0.)];winner=0.
    if best["union_weight"]>0 and best["ks"]-search[0]["ks"]>.003+1e-12:
        confirmation.append(score("confirmation",best["union_weight"]))
        if confirmation[-1]["ks"]-confirmation[0]["ks"]>.003+1e-12:winner=best["union_weight"]
    selection={"inputs_sha256":digest(metadata),"winner_union_weight":winner,"search":search,"confirmation":confirmation,
               "outer_seen":False,"smoke_only":False,"min_inner_gain":.003,
               "confirmation_reuse":"same inner split as prior input experiments; not independent project evidence",
               "outer_status":"No outer predictions read/materialized; nested refit required for any promotion"}
    sp=a.out/"selection.json"
    if sp.exists():assert json.loads(sp.read_text())==selection
    else:sp.write_text(json.dumps(selection,indent=2))
    pd.DataFrame([dict(role=role,**row) for role,rows in (("search",search),("confirmation",confirmation)) for row in rows]).to_csv(a.out/"scores.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(weights,[r["ks"]-search[0]["ks"] for r in search],"o-");axes[0].axhline(.003,color="red",linestyle=":")
    axes[0].set(xlabel="Union rank weight",ylabel="Delta KS vs baseline",title="Inner-search grid fixed before evaluation")
    axes[1].bar([str(r["union_weight"]) for r in confirmation],[r["ks"]-confirmation[0]["ks"] for r in confirmation]);axes[1].axhline(.003,color="red",linestyle=":")
    axes[1].set(xlabel="Union rank weight",ylabel="Delta KS vs baseline",title="Only search-selected weight confirmed")
    fig.tight_layout();plot=OUT/f"180_{a.out.name}_s{a.seed}_f{a.fold}.png";fig.savefig(plot,dpi=130);plt.close(fig)
    print(json.dumps(selection,indent=2),flush=True);print("Saved",a.out,"plot",plot,flush=True)


if __name__=="__main__":main()
