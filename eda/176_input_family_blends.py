"""Evaluate prescribed fixed input ensembles on cached inner predictions only.

No fitting or outer-score access. This reuses173's inner splits, so confirmation
is not a new independent holdout across experiments. Outer selection stays shut.
"""
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

SINGLES=("utama_raw","utama_native","teman_raw","teman_native")
BASELINE=SINGLES[0]


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def fixed_predictions(single):
    result={name:single[name] for name in SINGLES}
    result["pair_probability"]=.5*(single[BASELINE]+single["teman_native"])
    result["all4_probability"]=np.mean([single[name] for name in SINGLES],axis=0)
    result["pair_rank"]=.5*(rankdata(single[BASELINE],method="average")+rankdata(single["teman_native"],method="average"))/len(single[BASELINE])
    assert all(np.isfinite(p).all() and ((p>=0)&(p<=1)).all() for p in result.values())
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--fold",type=int,default=0)
    ap.add_argument("--allow-smoke",action="store_true")
    a=ap.parse_args();folder=a.run/f"s{a.seed}_f{a.fold}"
    manifest=json.loads((folder/"manifest.json").read_text());run_hash=digest(manifest)
    if manifest["smoke_only"] and not a.allow_smoke:raise ValueError("Smoke is not quality evidence")
    if manifest["config"]["seed"]!=a.seed or manifest["config"]["fold"]!=a.fold:raise ValueError("Run split mismatch")
    assert sha(PQ/"features.parquet")==manifest["canonical_features_sha256"]
    for name,expected in manifest["sources_sha256"].items():assert sha(ROOT/"eda"/name)==expected
    splits=json.loads((folder/"splits.json").read_text())
    assert splits["run_sha256"]==run_hash
    split_ids=splits["ids"]
    canonical=(pl.read_parquet(PQ/"features.parquet",columns=["application_id","target","is_train"])
               .filter(pl.col("is_train")==1))
    truth=dict(zip(canonical["application_id"],canonical["target"]))
    fits={};predictions={role:{} for role in ("search","confirmation")}
    for name in SINGLES:
        prefix=folder/f"inner_{name}"
        fp=prefix.with_name(prefix.name+"_fit.json");pp=prefix.with_name(prefix.name+"_predictions.parquet")
        cp=prefix.with_name(prefix.name+"_cache.json")
        fit=json.loads(fp.read_text());cache=json.loads(cp.read_text());p=pl.read_parquet(pp)
        assert fit["run_sha256"]==run_hash
        assert cache=={"fit_sha256":digest(fit),"predictions_sha256":sha(pp)}
        assert fit["real_context_ids"]==split_ids["inner_fit"]
        assert fit["real_labels"]==[truth[key] for key in split_ids["inner_fit"]]
        assert p["application_id"].to_list()==fit["query_ids"]
        assert p["role"].to_list()==fit["query_roles"] and p["application_id"].n_unique()==p.height
        assert set(fit["query_ids"]).isdisjoint(split_ids["inner_fit"]+split_ids["outer_validation"])
        for role in predictions:
            q=p.filter(pl.col("role")==role)
            assert q["application_id"].to_list()==split_ids[role]
            predictions[role][name]=q["prediction"].to_numpy()
        fits[name]={"fit_sha256":sha(fp),"predictions_sha256":sha(pp)}
    # Confirm tie treatment against a hand-computed case before real scores.
    np.testing.assert_array_equal(rankdata([.4,.4,.8],method="average")/3,[.5,.5,1.])
    combined={role:fixed_predictions(single) for role,single in predictions.items()}
    y={role:np.array([truth[key] for key in split_ids[role]],dtype=int) for role in predictions}
    protocol=ROOT/"docs/integration/protokol_blend_input_kumo.md"
    inputs={"source_run_sha256":run_hash,"source_sha256":sha(Path(__file__).resolve()),"fits":fits,
            "protocol_sha256":sha(protocol),"min_inner_gain":.003,"smoke_only":manifest["smoke_only"]}
    a.out.mkdir(parents=True,exist_ok=True);mp=a.out/"manifest.json"
    if mp.exists():assert json.loads(mp.read_text())==inputs,"Changed source or predictions; use a fresh output"
    elif any(a.out.iterdir()):raise ValueError("Use a fresh owned directory")
    else:mp.write_text(json.dumps(inputs,indent=2))
    def score(role,name):
        p=combined[role][name]
        return {"name":name,"ks":v.ks(y[role],p),"auc":float(roc_auc_score(y[role],p))}
    search=[score("search",name) for name in combined["search"]]
    best=max(search,key=lambda row:row["ks"])
    top=best["name"] if best["ks"]-search[0]["ks"]>.003+1e-12 else None
    confirmation=[score("confirmation",BASELINE)]
    winner=BASELINE
    if top is not None:
        confirmation.append(score("confirmation",top))
        if confirmation[-1]["ks"]-confirmation[0]["ks"]>.003+1e-12:winner=top
    selection={"inputs_sha256":digest(inputs),"winner":winner,"search":search,"confirmation":confirmation,
               "outer_seen":False,"smoke_only":manifest["smoke_only"],"min_inner_gain":.003,
               "confirmation_reuse":"same inner split as173; not independent evidence across projects",
               "outer_status":"No outer predictions read/materialized by176; refit required if promoted"}
    sp=a.out/"selection.json"
    if sp.exists():assert json.loads(sp.read_text())==selection
    else:sp.write_text(json.dumps(selection,indent=2))
    pd.DataFrame([dict(role=role,**row) for role,rows in (("search",search),("confirmation",confirmation)) for row in rows]).to_csv(a.out/"scores.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,role,rows in zip(axes,("search","confirmation"),(search,confirmation)):
        ax.bar([r["name"] for r in rows],[r["ks"]-rows[0]["ks"] for r in rows]);ax.axhline(.003,color="red",linestyle=":",label="strict gain gate")
        ax.tick_params(axis="x",rotation=65);ax.set(ylabel="Delta KS vs matched baseline",title=f"{'SMOKE ' if manifest['smoke_only'] else ''}Inner {role}");ax.legend(fontsize=8)
    fig.tight_layout();plot=OUT/f"176_{a.out.name}_s{a.seed}_f{a.fold}.png";fig.savefig(plot,dpi=130);plt.close(fig)
    print(json.dumps(selection,indent=2),flush=True);print("Saved",a.out,"plot",plot,flush=True)


if __name__=="__main__":main()
