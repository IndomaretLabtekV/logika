"""Bureau: current pulls per provider, provider agreement, history deltas."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, analyze, features, ks

f = features()
cols = [c for c in f.columns if c.startswith(("bc_", "bp_", "bh_"))] + ["debt_to_income"]
analyze(f, cols, "04_bureau", top_plot=16)
tr = f.filter(pl.col("is_train") == 1)
y = tr["target"].to_numpy()
# do providers measure the same thing? correlation of scores and per-provider KS
print("score corr between providers:\n", tr.select("bp_score_a", "bp_score_b", "bp_score_c").drop_nulls().corr())
for p in "abc":
    print(p, "KS", round(ks(y, tr[f"bp_score_{p}"].to_numpy()), 4), "coverage", tr[f"bp_score_{p}"].is_not_null().mean())
print("bad rate by n_reports:", tr.group_by("bc_n_reports").agg(pl.len(), pl.col("target").mean()).sort("bc_n_reports"))
print("bad rate when no score at all:", tr.filter(pl.col("bc_score_mean").is_null())["target"].mean())
print("bad rate by delinq_max:", tr.group_by("bc_delinq_max").agg(pl.len(), pl.col("target").mean()).sort("bc_delinq_max"))
print("bad rate by inq_max:", tr.group_by("bc_inq_max").agg(pl.len(), pl.col("target").mean()).sort("bc_inq_max"))
# score by rate tier: is pricing derived from score?
print(tr.group_by("annual_interest_rate").agg(pl.col("bc_score_mean").mean(), pl.col("bc_score_mean").std().alias("sd")).sort("annual_interest_rate"))
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for k, lab in [(0, "good"), (1, "bad")]:
    ax[0].hist(tr.filter(pl.col("target") == k)["bc_score_mean"].drop_nulls(), bins=60, density=True, alpha=.5, label=lab)
ax[0].legend(); ax[0].set_title("mean bureau score by class")
s = tr.select("bp_score_a", "bp_score_b").drop_nulls().sample(5000, seed=0)
ax[1].scatter(s["bp_score_a"], s["bp_score_b"], s=2, alpha=.3); ax[1].set_title("provider A vs B score")
ax[2].hist(tr["bh_score_delta"].drop_nulls().clip(-150, 150), bins=60); ax[2].set_title("score now - last historical score")
plt.tight_layout(); plt.savefig(OUT / "04_bureau_extra.png", dpi=80)
