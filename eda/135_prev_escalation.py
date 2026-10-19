"""Previous-application dynamics not in features.py: loan-size escalation (slope of log requested amount over time,
last / first, last / mean), application acceleration (mean gap of the last 3 vs earlier gaps), re-application speed
after the latest rejection, new installment vs the latest scheduled installment, share of requests above the latest
approved amount. All rows with decision_datetime <= cutoff. Residual test vs the clean TREES-ONLY OOF (the family whose
CV transfers to public), same tests as eda/134 (O/E chi2/df per safe bin, cross-fit dKS, permuted null).
Usage: cd eda && ../.venv/bin/python 135_prev_escalation.py"""
import io
import contextlib

import numpy as np
import polars as pl

import blend2
import valid2 as v
from common import OUT, features, load
from common9 import calibrate, crossfit_dks, oe_test, safe_bins

DAY = 86400.0
f = features()
cut = f.select("customer_id", "application_id", "is_train", cutoff="decision_datetime",
               app_dt=pl.col("decision_datetime"), inst_new=pl.col("proposed_monthly_installment"), amt_new=pl.col("loan_amount"))
apps = pl.concat([load("train_applications"), load("test_applications")], how="diagonal_relaxed").select(
    "application_id", app_time="application_datetime")
cut = cut.join(apps, on="application_id", how="left")
p = (load("previous_applications").join(cut.select("customer_id", "cutoff", "app_time", "inst_new", "amt_new"), on="customer_id")
     .filter(pl.col("decision_datetime") <= pl.col("cutoff")).sort("customer_id", "application_datetime", "application_id")
     .with_columns(t=(pl.col("application_datetime") - pl.col("cutoff")).dt.total_seconds() / DAY / 365,
                   lr=pl.col("requested_amount").log(),
                   gap=(pl.col("application_datetime").diff().over("customer_id")).dt.total_seconds() / DAY))
P = p.group_by("customer_id").agg(
    E_req_slope=pl.cov("t", "lr") / pl.col("t").var(),
    E_req_last_first=pl.col("lr").last() - pl.col("lr").first(),
    E_req_last_mean=pl.col("lr").last() - pl.col("lr").mean(),
    E_new_vs_req_mean=(pl.col("amt_new").first().log() - pl.col("lr").mean()),
    E_gap_accel=pl.col("gap").drop_nulls().tail(3).mean() / (pl.col("gap").drop_nulls().head(pl.len() - 4).mean() + 1),
    E_days_since_rej=((pl.col("app_time").first() - pl.col("decision_datetime").filter(pl.col("decision") == "rejected").max())
                      .dt.total_seconds() / DAY),
    E_inst_vs_last_sched=pl.col("inst_new").first() / pl.col("scheduled_monthly_installment").drop_nulls().last(),
    E_share_req_above_appr=(pl.col("requested_amount") > pl.col("approved_amount").drop_nulls().last()).mean(),
    E_n_canc_365=((pl.col("decision") == "cancelled") & (pl.col("t") > -1)).sum(),
)
cand = cut.select("customer_id", "application_id", "is_train").join(P, on="customer_id", how="left")
cand.drop("customer_id", "is_train").write_parquet(OUT / "135_prev_escalation.parquet")
tr = cand.filter(pl.col("is_train") == 1).sort("application_id")
with contextlib.redirect_stdout(io.StringIO()):
    oof, _, _, _, _, res = blend2.main(["../results/v22/v22"])
assert (oof["application_id"] == tr["application_id"]).all()
y = oof["target"].to_numpy()
pc = calibrate(y, res["equal_trees"][0])
rng = np.random.default_rng(0)
x0 = f.filter(pl.col("is_train") == 1).sort("application_id")["pv_n"].cast(pl.Float64).to_numpy()
null = [(oe_test(y, pc, safe_bins(z, 10))[0], crossfit_dks(y, pc, safe_bins(z, 10)))
        for z in (rng.permutation(x0) for _ in range(30))]
c95, d95 = np.quantile([a for a, _ in null], .95), np.quantile([b for _, b in null], .95)
print(f"null (permuted pv_n, keeps the 30% missing block): chi2/df p95 {c95:.2f}, dKS p95 {d95:+.5f}")
rows = []
for c in [c for c in tr.columns if c.startswith("E_")]:
    x = tr[c].cast(pl.Float64).to_numpy()
    b = safe_bins(x, 10)
    chi, tab = oe_test(y, pc, b)
    rows.append(dict(feature=c, cover=float(np.isfinite(x).mean()), ks_uni=v.ks_nan(y, x), chi2=chi, dks=crossfit_dks(y, pc, b),
                     oe=" ".join(f"{t[4]:.2f}" for t in tab)))
with pl.Config(tbl_rows=20, tbl_cols=8, fmt_str_lengths=80, tbl_width_chars=200):
    print(pl.DataFrame(rows).sort("chi2", descending=True))
pl.DataFrame(rows).write_csv(OUT / "135_prev_escalation.csv")
