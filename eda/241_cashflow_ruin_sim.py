"""Structural, label-free cash-flow ruin simulation. Hypothesis: the generator simulates monthly cash flow
and the new 12-month loan defaults when the balance cannot cover installments for long enough
(DPD>=90 ~ three missed installments). Ratios (cf_inst_to_free, inst_to_avgbal) are static; ruin is
path-dependent on starting balance, mean AND volatility of net flow and the installment size.

Per customer: bootstrap 12 future months from the customer's own pre-decision monthly net flows
(inflows - outflows), start at the last closing balance, subtract the new installment monthly, 400 paths.
Features: P(>=3 consecutive months below zero), P(cumulative shortfall >= 3 installments),
P(any month below zero), quantiles of min balance / installment, months-to-first-negative median.
Written for all 100k applications (NaN without account history) in 237's per-fold format.
Usage: cd eda && ../.venv/bin/python 241_cashflow_ruin_sim.py
"""
import os

import numpy as np
import polars as pl

from common import ROOT

PATHS, H = 400, 12
rng = np.random.default_rng(241)
apps = pl.concat([pl.read_csv(ROOT / f"data/{s}_applications.csv").select("application_id", "customer_id", "proposed_monthly_installment")
                  for s in ("train", "test")])
acc = pl.read_csv(ROOT / "data/account_behavior.csv", try_parse_dates=True).sort("customer_id", "month")
acc = acc.with_columns(net=pl.col("inflows") - pl.col("outflows"))
g = acc.group_by("customer_id", maintain_order=True).agg(pl.col("net"), pl.col("closing_balance").last().alias("bal0"))
d = apps.join(g, on="customer_id", how="left")
inst = d["proposed_monthly_installment"].to_numpy().astype(float)
out = {k: np.full(d.height, np.nan) for k in ["p_run3", "p_short3", "p_neg", "minbal_q10", "minbal_q50", "t_neg_med"]}
nets, bal0 = d["net"].to_list(), d["bal0"].to_numpy()
for i in range(d.height):
    if nets[i] is None:
        continue
    net = np.asarray(nets[i], float)
    draws = net[rng.integers(0, len(net), (PATHS, H))] - inst[i]     # monthly change after paying the new installment
    bal = bal0[i] + np.cumsum(draws, axis=1)                          # (PATHS, H) end-of-month balances
    neg = bal < 0
    run = np.zeros(PATHS, int); best = np.zeros(PATHS, int)
    for t in range(H):                                                # longest run of negative months
        run = np.where(neg[:, t], run + 1, 0); best = np.maximum(best, run)
    out["p_run3"][i] = (best >= 3).mean()
    out["p_short3"][i] = (bal.min(1) <= -3 * inst[i]).mean()
    out["p_neg"][i] = neg.any(1).mean()
    mb = bal.min(1) / inst[i]
    out["minbal_q10"][i], out["minbal_q50"][i] = np.quantile(mb, [.1, .5])
    first = np.where(neg.any(1), neg.argmax(1), H)
    out["t_neg_med"][i] = np.median(first)
df = pl.DataFrame({"application_id": d["application_id"], **{f"ruin_{k}": v for k, v in out.items()}})
print(df.describe())
for k in range(5):  # label-free: identical for every outer fold
    path = ROOT / f"results/v66/series_oof/ruin_f{k}.parquet"
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    tmp.replace(path)
train = pl.read_csv(ROOT / "data/train_applications.csv").select("application_id", "target").join(df, on="application_id")
import valid2 as v
for c in [c for c in df.columns if c.startswith("ruin_")]:
    x, y = train[c].to_numpy(), train["target"].to_numpy()
    print(f"{c:18s} univariate KS {v.ks_nan(y, x):.4f} (descriptive; not used for selection)")
