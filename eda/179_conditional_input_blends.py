"""Prescribed conditional-input blends; cached inner predictions, no outer use."""
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

BASELINE="utama_raw"
SINGLES=(BASELINE,"utama_gain","teman_gain","union_gain")


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def predictions(single):
    out=dict(single)
    for partner in ("teman_gain","union_gain"):
        out[partner+"_probability"]=.5*(single[BASELINE]+single[partner])
        out[partner+"_rank"]=.5*(rankdata(single[BASELINE],method="average")+rankdata(single[partner],method="average"))/len(single[partner])
    assert all(np.isfinite(p).all() and ((p>=0)&(p<=1)).all() for p in out.values())
    return out


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
    assert not manifest["smoke_only"] and not bm["smoke_only"]
    assert manifest["baseline_run_sha256"]==bh and (manifest["seed"],manifest["fold"])==(a.seed,a.fold)
    assert sha(PQ/"features.parquet")==bm["canonical_features_sha256"]
    for name,expected in manifest["source_sha256"].items():assert sha(ROOT/"eda"/name)==expected
    splits=json.loads((folder/"splits.json").read_text());assert splits["run_sha256"]==run_hash
    split_ids=splits["ids"]
    canonical=pl.read_parquet(PQ/"features.parquet",columns=["application_id","target","is_train"]).filter(pl.col("is_train")==1)
    truth=dict(zip(canonical["application_id"],canonical["target"]))
    single={role:{} for role in ("search","confirmation")};inputs={}
    for name in SINGLES:
        source=old if name==BASELINE else folder;expected_run=bh if name==BASELINE else run_hash
        fp,pp,cp=[source/f"inner_{name}_{suffix}" for suffix in ("fit.json","predictions.parquet","cache.json")]
        fit=json.loads(fp.read_text());cache=json.loads(cp.read_text());frame=pl.read_parquet(pp)
        assert fit["run_sha256"]==expected_run and fit["real_context_ids"]==split_ids["inner_fit"]
        assert fit["real_labels"]==[truth[key] for key in split_ids["inner_fit"]]
        assert (fit["representation"],fit["estimators"],fit["generator_seed"],fit["chunk"])==("raw",4,100*a.seed+a.fold,128)
        assert cache=={"fit_sha256":digest(fit),"predictions_sha256":sha(pp)}
        assert frame["application_id"].to_list()==fit["query_ids"] and frame["role"].to_list()==fit["query_roles"]
        assert frame["application_id"].n_unique()==frame.height
        assert set(frame["application_id"]).isdisjoint(split_ids["inner_fit"]+split_ids["outer_validation"])
        for role in single:
            part=frame.filter(pl.col("role")==role);assert part["application_id"].to_list()==split_ids[role]
            single[role][name]=part["prediction"].to_numpy()
        inputs[name]={"fit_sha256":sha(fp),"predictions_sha256":sha(pp)}
    combined={role:predictions(values) for role,values in single.items()}
    y={role:np.array([truth[key] for key in split_ids[role]],dtype=int) for role in single}
    provenance={"run_sha256":run_hash,"baseline_run_sha256":bh,"inputs":inputs,"source_sha256":sha(Path(__file__).resolve()),
                "protocol_sha256":sha(ROOT/"docs/integration/protokol_blend_kumo_conditional.md"),"min_inner_gain":.003,"smoke_only":False}
    a.out.mkdir(parents=True,exist_ok=True);mp=a.out/"manifest.json"
    if mp.exists():assert json.loads(mp.read_text())==provenance
    elif any(a.out.iterdir()):raise ValueError("Use a fresh owned directory")
    else:mp.write_text(json.dumps(provenance,indent=2))
    def score(role,name):
        p=combined[role][name];return {"name":name,"ks":v.ks(y[role],p),"auc":float(roc_auc_score(y[role],p))}
    search=[score("search",name) for name in combined["search"]]
    best=max(search,key=lambda row:row["ks"])
    top=best["name"] if best["ks"]-search[0]["ks"]>.003+1e-12 else None
    winner=BASELINE;confirmation=[score("confirmation",BASELINE)]
    if top is not None:
        confirmation.append(score("confirmation",top))
        if confirmation[-1]["ks"]-confirmation[0]["ks"]>.003+1e-12:winner=top
    selection={"inputs_sha256":digest(provenance),"winner":winner,"search":search,"confirmation":confirmation,
               "outer_seen":False,"smoke_only":False,"min_inner_gain":.003,
               "confirmation_reuse":"same inner split as173/176/177; not independent evidence across projects",
               "outer_status":"No outer predictions read/materialized; refit required for any promoted constituents"}
    sp=a.out/"selection.json"
    if sp.exists():assert json.loads(sp.read_text())==selection
    else:sp.write_text(json.dumps(selection,indent=2))
    pd.DataFrame([dict(role=role,**row) for role,rows in (("search",search),("confirmation",confirmation)) for row in rows]).to_csv(a.out/"scores.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,role,rows in zip(axes,("search","confirmation"),(search,confirmation)):
        ax.bar([r["name"] for r in rows],[r["ks"]-rows[0]["ks"] for r in rows]);ax.axhline(.003,color="red",linestyle=":")
        ax.tick_params(axis="x",rotation=65);ax.set(ylabel="Delta KS vs matched baseline",title="Inner "+role)
    fig.tight_layout();plot=OUT/f"179_{a.out.name}_s{a.seed}_f{a.fold}.png";fig.savefig(plot,dpi=130);plt.close(fig)
    print(json.dumps(selection,indent=2),flush=True);print("Saved",a.out,"plot",plot,flush=True)


if __name__=="__main__":main()
