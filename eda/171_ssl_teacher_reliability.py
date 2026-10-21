"""Describe the locked teacher on inner-search labels, never outer/test labels.

No calibration fit, threshold choice or modification of the running SSL policy.
Counts on organizer test are eligibility only, not correctness measurements.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from common import ROOT,PQ,OUT


def wilson(correct,n):
    if not n:return None,None
    z=1.959963984540054;p=correct/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-half),min(1.,center+half)


def main():
    folder=ROOT/"results/v49/ssl_asymmetric_safe_f0/s42_f0"
    predpath=folder/"inner_teacher_predictions.parquet"
    d=pl.read_parquet(predpath)
    splits=json.loads((folder/"splits.json").read_text())["ids"]
    fit=json.loads((folder/"inner_teacher_fit.json").read_text())
    assert len(fit["real_context_ids"])==48000
    assert d.height==d["application_id"].n_unique()==38000
    assert not set(fit["real_context_ids"])&set(d["application_id"].to_list())
    q=d.filter(pl.col("role")=="search")
    assert q.height==4000 and set(q["application_id"].to_list())==set(splits["inner_search"])
    test=d.filter(pl.col("role")=="test")
    assert test.height==30000 and set(test["application_id"].to_list())==set(splits["organizer_test"])
    truth=(pl.read_parquet(PQ/"features.parquet",columns=["application_id","target","is_train"])
           .filter(pl.col("is_train")==1).select("application_id","target"))
    q=q.join(truth,on="application_id",how="left",validate="1:1")
    assert q["target"].null_count()==0
    p,y=q["prediction"].to_numpy(),q["target"].to_numpy().astype(int)
    pt=test["prediction"].to_numpy()
    assert np.isfinite(p).all() and np.isfinite(pt).all() and ((p>=0)&(p<=1)).all()
    rows=[]
    for label,thresholds in [(0,(.02,.05)),(1,(.7,.9,.95,.98))]:
        for threshold in thresholds:
            mask=p<=threshold if label==0 else p>=threshold
            eligible=pt<=threshold if label==0 else pt>=threshold
            n=int(mask.sum());correct=int((y[mask]==label).sum());lo,hi=wilson(correct,n)
            mean_probability=float(np.mean(1-p[mask] if label==0 else p[mask])) if n else None
            row={"pseudo_label":label,"threshold":threshold,"inner_search_rows":n,
                 "inner_correct":correct,"inner_incorrect":n-correct,
                 "inner_precision":correct/n if n else None,"precision_wilson_lo":lo,"precision_wilson_hi":hi,
                 "mean_teacher_correct_class_probability":mean_probability,"test_eligible_rows":int(eligible.sum())}
            rows.append(row)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/"171_ssl_teacher_reliability.csv",index=False)
    report={"inner_search_rows":len(y),"test_rows":len(pt),"inner_search_positive_rate":float(y.mean()),
            "rows":rows,"teacher_predictions_sha256":hashlib.sha256(predpath.read_bytes()).hexdigest(),
            "source_sha256":hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
            "outer_labels_used":False,"confirmation_labels_used":False,"thresholds_changed":False,
            "limits":["Search labels describe teacher reliability; the running selection uses its locked thresholds.",
                      "Search is not a fresh confirmation of a chosen policy or an estimate of test correctness.",
                      "Wilson intervals treat unique customer rows as independent and do not include teacher-training variance.",
                      "Unlabeled test counts are eligibility only, not precision."]}
    (OUT/"171_ssl_teacher_reliability.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for row in rows:
        if row["inner_search_rows"]:
            x=row["mean_teacher_correct_class_probability"];z=row["inner_precision"]
            axes[0].errorbar(x,z,yerr=[[z-row["precision_wilson_lo"]],[row["precision_wilson_hi"]-z]],fmt="o",capsize=3,
                            label=f"y{row['pseudo_label']} threshold{row['threshold']} (n={row['inner_search_rows']})")
    axes[0].plot([.5,1],[.5,1],"k:");axes[0].set(xlabel="Mean teacher probability for assigned class",ylabel="Inner-search empirical precision",title="Descriptive only; 95% Wilson intervals")
    axes[0].legend(loc="lower left",fontsize=7)
    axes[1].bar([f"y{r['pseudo_label']} t{r['threshold']}" for r in rows],[r["test_eligible_rows"] for r in rows])
    axes[1].set(yscale="symlog",ylabel="Eligible organizer-test rows",title="Eligibility, not correctness");axes[1].tick_params(axis="x",rotation=45)
    fig.tight_layout();fig.savefig(OUT/"171_ssl_teacher_reliability.png",dpi=130);plt.close(fig)
    print(frame.to_string(index=False));print("No outer/confirmation label use; no policy changes")


if __name__=="__main__":main()
