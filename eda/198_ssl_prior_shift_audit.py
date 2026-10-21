"""Audit class mix in completed hard-pseudo-label experiments, no new fits.

A changed context prior is a hypothesis about input geometry, not proof of why
KS fell. Constant monotone prior recalibration alone cannot improve KS ranks.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def main():
    folder=ROOT/"results/v49/ssl_asymmetric_safe_f0/s42_f0"
    read=lambda name:json.loads((folder/name).read_text())
    teacher=read("inner_teacher_fit.json");cache=read("inner_teacher_cache.json")
    teacher_predictions=folder/"inner_teacher_predictions.parquet"
    assert cache=={"fit_sha256":digest(teacher),"predictions_sha256":sha(teacher_predictions)}
    labels=np.asarray(teacher["real_labels"]);real_ids=teacher["real_context_ids"]
    assert len(labels)==len(real_ids)==len(set(real_ids))==48000 and set(labels)=={0,1}
    n_real=len(labels);positives=int(labels.sum());prior=float(labels.mean())
    search=read("search.json");assert search[0]["name"]=="noSSL" and len(search)==13
    baseline=search[0]["ks"];rows=[]
    for row in search[1:]:
        name=row["name"];fit=read(f"inner_search_{name}_fit.json")
        assert fit["real_context_ids"]==real_ids and fit["real_labels"]==teacher["real_labels"]
        assert fit["all_real_context_retained"] is True
        assert len(fit["pseudo_context_ids"])==len(fit["pseudo_labels"])==row["n_pseudo"]
        assert sum(fit["pseudo_labels"])==row["pseudo_positive"]
        assert set(fit["pseudo_context_ids"]).isdisjoint(real_ids) and set(fit["pseudo_context_ids"]).isdisjoint(fit["query_ids"])
        total=n_real+row["n_pseudo"];combined=(positives+row["pseudo_positive"])/total
        item={"name":name,"real_rows":n_real,"real_positive":positives,"real_prior":prior,
              "pseudo_rows":row["n_pseudo"],"pseudo_positive":row["pseudo_positive"],
              "pseudo_prior":row["pseudo_positive"]/row["n_pseudo"],"combined_prior":combined,
              "prior_shift_percentage_points":100*(combined-prior),"ks":row["ks"],"delta_ks":row["ks"]-baseline}
        rows.append(item)
        print(name,"pseudo positive%",round(100*item["pseudo_prior"],3),"combined%",round(100*combined,3),"deltaKS",item["delta_ks"],flush=True)
    best=max(rows,key=lambda row:row["ks"])
    p=pl.read_parquet(teacher_predictions).filter(pl.col("role")=="test")["prediction"].to_numpy().astype(np.float64)
    assert len(p)==30000 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    multiplier=(best["combined_prior"]/(1-best["combined_prior"]))/ (prior/(1-prior))
    adjusted=multiplier*p/(1-p+multiplier*p)
    same_order=bool(np.array_equal(rankdata(p),rankdata(adjusted)));assert same_order
    available=pl.read_parquet(folder/"inner_teacher_pseudo_candidates.parquet")
    assert available["application_id"].n_unique()==30000
    tp=available["teacher_probability"].to_numpy()
    pos=int((tp>=.7).sum());neg=int((tp<=.02).sum());proposals=[]
    for cap in (2000,10000):
        total=min(cap,int(np.floor(pos/prior)),int(np.floor(neg/(1-prior))))
        positive=int(round(total*prior));negative=total-positive
        assert positive<=pos and negative<=neg and total>0
        proposals.append({"cap":cap,"actual_rows":total,"pseudo_positive":positive,"pseudo_negative":negative,
                          "pseudo_prior":positive/total,"class_prior_from":"real fitting labels only",
                          "model_executed":False,"quality_known":False})
    result={"teacher_fit_sha256":sha(folder/"inner_teacher_fit.json"),"search_sha256":sha(folder/"search.json"),
            "real_rows":n_real,"real_positive":positives,"real_prior":prior,"baseline_ks":baseline,"policies":rows,"best":best,
            "constant_odds_recalibration_preserves_test_order":same_order,"unexecuted_prior_preserving_count_proposals":proposals,
            "limits":["Descriptive audit of the existing inner-search experiment; no confirmation/outer labels used.",
                      "Class-mix drift alone is not a causal explanation for the observed KS losses.",
                      "A monotone prior-odds correction preserves ranking and cannot improve KS by itself.",
                      "New pseudo sampling could change context geometry, but its quality and label errors remain untested.",
                      "Teacher confidence is not ground-truth accuracy on test; no new model or submission."]}
    output=ROOT/"eda/out/198_ssl_prior_shift_audit"
    output.with_suffix(".json").write_text(json.dumps(result,indent=2))
    pl.DataFrame(rows).write_csv(output.with_suffix(".csv"))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].barh([r["name"] for r in rows],[100*r["combined_prior"] for r in rows]);axes[0].axvline(100*prior,color="black",ls="--",label="Real context")
    axes[0].set(xlabel="Positive fraction of combined context (%)",title="All real rows retained; pseudo class mix differs");axes[0].legend(fontsize=8);axes[0].tick_params(axis="y",labelsize=7)
    axes[1].scatter([r["prior_shift_percentage_points"] for r in rows],[r["delta_ks"] for r in rows]);axes[1].axhline(0,color="black",lw=.7)
    axes[1].set(xlabel="Combined prior shift (percentage points)",ylabel="Observed inner-search KS delta",title="Descriptive association; not a causal test")
    fig.tight_layout();fig.savefig(output.with_suffix(".png"),dpi=130);plt.close(fig)
    print("Real prior",prior,"best pseudo prior",best["pseudo_prior"],"best delta",best["delta_ks"],"monotone control preserves order",same_order,flush=True)
    print("Count proposals only; no new inference",json.dumps(proposals),flush=True)


if __name__=="__main__":main()
