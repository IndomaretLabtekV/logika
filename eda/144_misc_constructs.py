"""Constructs not in features.py or the catalogue (checked 19 Okt): outflow volatility (cv of monthly outflows), liquidity
buffer shape (minimum / average balance per month), volatility of the non-income inflow share, per-provider active
accounts and revolving limits on the selected pulls, previous-request LTI (requested / declared income), and the
bureau score residual (score minus a fitted function of the report's own delinquency / utilisation / inquiries /
accounts; fitted on train customers' reports only, no target) — a latent analogous to pp_mean. All rows <= cutoff.
Residual test vs the clean TREES-ONLY OOF (eda/134 setup). Writes eda/out/144_misc_constructs.parquet.
Usage: cd eda && ../.venv/bin/python 144_misc_constructs.py"""
import io
import contextlib

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

import blend2
import valid2 as v
from common import OUT, features, load
from common9 import calibrate, crossfit_dks, oe_test, safe_bins

f = features().sort("application_id")
cut = f.select("customer_id", "application_id", "is_train", "declared_monthly_income", cutoff="decision_datetime")
acc = (load("account_behavior").join(cut.select("customer_id", "cutoff"), on="customer_id")
       .filter(pl.col("month").dt.offset_by("1mo").cast(pl.Datetime) <= pl.col("cutoff")))
A = acc.group_by("customer_id").agg(
    G_out_cv=pl.col("outflows").std() / (pl.col("outflows").mean() + 1),
    G_minavg_mean=(pl.col("minimum_balance") / (pl.col("average_balance").abs() + 1e4)).mean(),
    G_minavg_l3=(pl.col("minimum_balance") / (pl.col("average_balance").abs() + 1e4)).sort_by("month").tail(3).mean(),
    G_nonincome_sd=(1 - pl.col("income_like_credits") / (pl.col("inflows") + 1)).std(),
    G_out_max_over_mean=pl.col("outflows").max() / (pl.col("outflows").mean() + 1),
)
b = (load("bureau_observations").join(cut.select("customer_id", "cutoff", sel="application_id"), on="customer_id")
     .filter(pl.col("available_datetime") <= pl.col("cutoff")))
# bureau score residual: score ~ report's own risk fields, fitted on train customers' reports only
trc = set(f.filter(pl.col("is_train") == 1)["customer_id"].to_list())
bs = b.filter(pl.col("score").is_not_null())
Z = bs.select("active_accounts", "outstanding_balance", "revolving_credit_limit", "revolving_outstanding_balance",
              "utilization", "inquiries_last_6m", "delinquent_accounts_last_12m").to_numpy().astype(float)
m_tr = bs["customer_id"].is_in(list(trc)).to_numpy()
reg = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Z[m_tr], bs["score"].to_numpy()[m_tr])
bs = bs.with_columns(res=pl.Series(bs["score"].to_numpy() - reg.predict(Z)))
B = bs.group_by("customer_id").agg(G_score_res_mean=pl.col("res").mean(),
                                   G_score_res_sel=pl.col("res").filter(pl.col("application_id") == pl.col("sel")).mean())
sel = b.filter(pl.col("application_id") == pl.col("sel")).with_columns(prov=pl.col("bureau_provider").str.slice(-1))
P = sel.pivot(on="prov", index="customer_id", values=["active_accounts", "revolving_credit_limit"], aggregate_function="first")
P = P.rename({c: "G_" + c for c in P.columns if c != "customer_id"})
pv = (load("previous_applications").join(cut.select("customer_id", "cutoff", "declared_monthly_income"), on="customer_id")
      .filter(pl.col("decision_datetime") <= pl.col("cutoff")))
V = pv.group_by("customer_id").agg(G_prev_lti_mean=(pl.col("requested_amount") / pl.col("declared_monthly_income")).mean(),
                                   G_prev_lti_max=(pl.col("requested_amount") / pl.col("declared_monthly_income")).max())
cand = cut.select("customer_id", "application_id", "is_train")
for part in (A, B, P, V):
    cand = cand.join(part, on="customer_id", how="left")
cand.drop("customer_id", "is_train").write_parquet(OUT / "144_misc_constructs.parquet")
tr = cand.filter(pl.col("is_train") == 1).sort("application_id")
with contextlib.redirect_stdout(io.StringIO()):
    oof, _, _, _, _, res = blend2.main(["../results/v22/v22"])
assert (oof["application_id"] == tr["application_id"]).all()
y = oof["target"].to_numpy()
pc = calibrate(y, res["equal_trees"][0])
rng = np.random.default_rng(0)
x0 = tr["G_out_cv"].to_numpy().astype(float)
null = [(oe_test(y, pc, safe_bins(z, 10))[0], crossfit_dks(y, pc, safe_bins(z, 10))) for z in (rng.permutation(x0) for _ in range(30))]
c95, d95 = np.quantile([a for a, _ in null], .95), np.quantile([q for _, q in null], .95)
print(f"null: chi2/df p95 {c95:.2f}, dKS p95 {d95:+.5f}")
rows = []
for c in [c for c in tr.columns if c.startswith("G_")]:
    x = tr[c].cast(pl.Float64).to_numpy()
    x = np.where(np.isfinite(x), x, np.nan)
    bb = safe_bins(x, 10)
    rows.append(dict(feature=c, cover=float(np.isfinite(x).mean()), ks_uni=v.ks_nan(y, x), chi2=oe_test(y, pc, bb)[0], dks=crossfit_dks(y, pc, bb)))
with pl.Config(tbl_rows=20):
    print(pl.DataFrame(rows).sort("chi2", descending=True))
