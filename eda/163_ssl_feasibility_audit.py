"""Unlabeled feasibility audit of predetermined Kumo pseudo-context policies.

Historical test predictions only: no labels loaded, no teacher trained, no policy
selected by KS. These predictions must NEVER become an inner-fold SSL teacher.
New teachers must be fitted inside the corresponding inner/outer training split.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"eda/out"


def read_test(path,ids):
    frame=pl.read_parquet(path)
    assert frame["application_id"].n_unique()==len(ids)==frame.height
    assert set(frame["application_id"])==set(ids)
    columns=[c for c in frame.columns if c!="application_id"]
    assert len(columns)==1 and "target" not in frame.columns
    aligned=pl.DataFrame({"application_id":ids}).join(frame,on="application_id",how="left",maintain_order="left")
    p=aligned[columns[0]].to_numpy()
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return p


def main():
    ids=pl.read_csv(ROOT/"data/sample_submission.csv",columns=["application_id"])["application_id"]
    sources={"kumo":ROOT/"results/v28m/test_kumo_large_c56000.parquet",
             "lgb":ROOT/"results/v22/v22/gbdt/test_lgb.parquet",
             "cat":ROOT/"results/v22/v22/gbdt/test_cat.parquet"}
    ps={name:read_test(path,ids) for name,path in sources.items()}
    p=ps["kumo"]
    rows=[]
    # Policies .02/.05 are fixed in existing script156; do not tune on labels here.
    for t in (.02,.05):
        neg,pos=p<=t,p>=1-t
        eligible=neg|pos
        rows.append({"threshold":t,"negative_candidates":int(neg.sum()),"positive_candidates":int(pos.sum()),
                     "eligible":int(eligible.sum()),
                     "teacher_positive_fraction_in_tail":float(pos.sum()/max(eligible.sum(),1)),
                     "lgb_hard_agreement_in_tail":float(np.mean((p[eligible]>=.5)==(ps["lgb"][eligible]>=.5))) if eligible.any() else None,
                     "cat_hard_agreement_in_tail":float(np.mean((p[eligible]>=.5)==(ps["cat"][eligible]>=.5))) if eligible.any() else None})
    groups={"p<=.02":p<=.02,".02<p<=.05":(p>.02)&(p<=.05),
            ".95<=p<.98":(p>=.95)&(p<.98),"p>=.98":p>=.98}
    report={"rows":len(p),"predictions_provenance":"historical fold-averaged test predictions; descriptive only",
            "probability_quantiles":{str(q):float(np.quantile(p,q)) for q in (0,.01,.05,.1,.5,.9,.95,.99,1)},
            "fixed_confidence_policies":rows,"shell_counts":{name:int(mask.sum()) for name,mask in groups.items()},
            "source_sha256":{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sources.items()},
            "labels_loaded":False,"model_executed":False,"policy_selected":False,
            "limitations":["Agreement is not correctness; no test labels exist.",
                           "Historical test averages are smoother than a new split-specific teacher.",
                           "No historical predictor in this audit is eligible as a nested teacher.",
                           "Do not claim high-confidence positive pseudo labels when the shells contain none."]}
    (OUT/"163_ssl_feasibility_audit.json").write_text(json.dumps(report,indent=2))
    pl.DataFrame(rows).write_csv(OUT/"163_ssl_feasibility_policies.csv")
    print(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,q in ps.items():axes[0].hist(q,bins=60,histtype="step",label=name)
    axes[0].legend();axes[0].set(xlabel="Historical test probability",ylabel="Rows",title="Unlabeled probability distributions")
    axes[1].bar(list(groups),[mask.sum() for mask in groups.values()]);axes[1].tick_params(axis="x",rotation=25)
    axes[1].set(ylabel="Rows",title="Fixed SSL156 confidence shells (not correctness)")
    fig.tight_layout();fig.savefig(OUT/"163_ssl_feasibility_audit.png",dpi=130);plt.close(fig)


if __name__=="__main__":main()
