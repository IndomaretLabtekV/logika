"""Is there a label-noise floor in train? Under symmetric flip noise eps the observed bad rate cannot fall below eps for
the safest customers nor exceed 1 - eps for the riskiest, and the lowest-risk bins flatten instead of decreasing.
Clean trees OOF (v22) and v31 OOF: bad rate in fine quantile bins at both extremes (0-0.5%, 0.5-1%, ... 0-10%), with
binomial CIs, and the slope of log bad rate over the lowest 10 bins of 1%. Also the same for a 'cleanest' signal
subset (history customers with long on-time repayment). Saves eda/out/148_label_floor.png.
Usage: cd eda && ../.venv/bin/python 148_label_floor.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import beta, rankdata

from common import OUT, ROOT, features

R = ROOT / "results"
f = features().filter(pl.col("is_train") == 1).sort("application_id")
y = f["target"].to_numpy()


def raw(p):
    d = pl.read_parquet(p).sort("application_id")
    return d[[c for c in d.columns if c not in ("application_id", "target")][0]].to_numpy()


rk = lambda x: rankdata(x) / len(x)  # noqa: E731
trees = rk(np.mean([rk(raw(R / f"v22/v22/gbdt/oof_{m}.parquet")) for m in ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]], axis=0))
edges = [0, .0025, .005, .01, .02, .03, .05, .1, .2, .5, .8, .9, .95, .97, .98, .99, .995, .9975, 1]
fig, ax = plt.subplots(figsize=(6, 4))
mids, rates = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (trees > lo) & (trees <= hi)
    k, n = y[m].sum(), m.sum()
    ci = beta.ppf([0.025, 0.975], k + 0.5, n - k + 0.5)
    print(f"trees rank ({lo:.4f}, {hi:.4f}]: n {n:6d} bad {k:5d} rate {k / n:.4f} 95% [{ci[0]:.4f}, {ci[1]:.4f}]")
    mids.append((lo + hi) / 2)
    rates.append(k / n)
ax.semilogy(mids, rates, "o-")
low = [(y[(trees > i / 100) & (trees <= (i + 1) / 100)].mean()) for i in range(10)]
print("lowest 10 percentiles bad rate:", [round(x, 4) for x in low])
print(f"log-slope over lowest 10%: {np.polyfit(np.arange(10), np.log(np.array(low) + 1e-4), 1)[0]:+.3f} per percentile")
safe = (f["rp_frac_dpd1"].to_numpy() == 0) & (f["rp_n_inst"].to_numpy() >= 24)
print(f"history customers always on time with >= 24 installments: n {safe.sum()}, bad rate {y[safe].mean():.4f}; "
      f"their lowest-decile-by-trees bad rate {y[safe & (trees <= np.quantile(trees[safe], .1))].mean():.4f}")
ax.set(xlabel="trees OOF rank", ylabel="bad rate (log)", title="Bad rate at the extremes")
fig.tight_layout()
fig.savefig(OUT / "148_label_floor.png", dpi=110)
