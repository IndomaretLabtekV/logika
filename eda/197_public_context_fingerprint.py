"""Identify the actual historical Kumo-only public file by prediction ranks.

Read-only artifact audit, no model inference, test labels or leaderboard probe.
Context-size differences cannot explain a score for a file that did not use them.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata,spearmanr

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    sample=pl.read_csv(ROOT/"data/sample_submission.csv").select("application_id")
    submitted_path=ROOT/"results/v31/kumo_only.csv";submitted=pl.read_csv(submitted_path)
    assert submitted.columns==["application_id","default_probability"] and submitted.height==30000
    assert submitted["application_id"].n_unique()==30000
    assert set(submitted["application_id"].to_list())==set(sample["application_id"].to_list())
    submitted=sample.join(submitted,on="application_id",how="left",maintain_order="left")
    score=submitted["default_probability"].to_numpy();submitted_rank=rankdata(score)
    sources={"56k fold ensemble":"results/v28m/test_kumo_large_c56000.parquet",
             "70k single fit v30":"results/v30/v30/test_kumo_full70k.parquet",
             "70k production v34":"results/v34/test_kumo_full70k.parquet"}
    rows=[];rank_differences={}
    for label,relative in sources.items():
        path=ROOT/relative;d=pl.read_parquet(path)
        assert d.height==d["application_id"].n_unique()==30000
        assert set(d["application_id"].to_list())==set(sample["application_id"].to_list())
        columns=[c for c in d.columns if c!="application_id"];assert len(columns)==1
        p=sample.join(d,on="application_id",how="left",maintain_order="left")[columns[0]].to_numpy()
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        rank=rankdata(p);diff=(rank-submitted_rank)/30000;rank_differences[label]=diff
        row={"recipe":label,"path":relative,"sha256":sha(path),"spearman":float(spearmanr(p,score).statistic),
             "rank_order_exact_match":bool(np.array_equal(rank,submitted_rank)),
             "max_percentile_rank_difference":float(np.abs(diff).max()),
             "max_raw_score_difference":float(np.abs(p-score).max()),
             "max_rank_score_difference":float(np.abs(rank/30000-score).max())}
        rows.append(row);print(json.dumps(row),flush=True)
    result={"submitted_file":"results/v31/kumo_only.csv","submitted_sha256":sha(submitted_path),
            "historical_public_score":.56056,"public_score_source":"handoff/04_riwayat_versi/v31_v35_submission.md (historical, not a new submission)",
            "comparisons":rows,"uses_56k_fold_ensemble_ranks":rows[0]["rank_order_exact_match"],
            "limits":["File equality/order supports recipe identity, not a causal explanation of public labels.",
                      "A pretraining context limit is not evidence that a particular competition score was caused by it.",
                      "No private/public target values are available; scores for other hypothetical exports are unknown."]}
    out=ROOT/"eda/out/197_public_context_fingerprint"
    out.with_suffix(".json").write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    indices=np.random.default_rng(197).choice(30000,1000,replace=False)
    for label,diff in rank_differences.items():
        axes[0].scatter(submitted_rank[indices]/30000,diff[indices],s=3,alpha=.35,label=label)
    axes[0].set(xlabel="Historical submitted percentile rank",ylabel="Artifact minus submitted rank",title="Which prediction file generated the Kumo-only order?")
    axes[0].legend(fontsize=7)
    axes[1].barh([r["recipe"] for r in rows],[r["max_percentile_rank_difference"] for r in rows])
    axes[1].set(xlabel="Max absolute percentile rank difference",title="Zero = identical historical order")
    fig.tight_layout();fig.savefig(out.with_suffix(".png"),dpi=130);plt.close(fig)
    print("56k exact historical rank match:",result["uses_56k_fold_ensemble_ranks"],"no new submission",flush=True)


if __name__=="__main__":main()
