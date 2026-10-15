"""Account behaviour: cash-flow health and income verification."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from common import OUT, analyze, features

f = features()
cols = [c for c in f.columns if c.startswith("ac_")] + ["dti_verified", "income_verif_ratio", "inst_to_avgbal"]
analyze(f, cols, "06_account", top_plot=16)
tr = f.filter(pl.col("is_train") == 1).drop_nulls("ac_income_mean")
print("income_verif_ratio quantiles:", tr["income_verif_ratio"].quantile(0.01), tr["income_verif_ratio"].median(), tr["income_verif_ratio"].quantile(0.99))
print(tr.with_columns(q=pl.col("income_verif_ratio").qcut(5)).group_by("q").agg(pl.len(), pl.col("target").mean()).sort("q"))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
s = tr.sample(8000, seed=0)
ax[0].scatter(s["declared_monthly_income"].log10(), s["ac_income_mean"].log10(), s=2, c=s["target"], cmap="coolwarm", alpha=.4)
ax[0].set_xlabel("log10 declared"); ax[0].set_ylabel("log10 observed income-like credits"); ax[0].set_title("income verification")
for k in (0, 1):
    ax[1].hist(tr.filter(pl.col("target") == k)["ac_lowdays_l3"], bins=31, density=True, alpha=.5, label=f"target={k}")
ax[1].legend(); ax[1].set_title("low-balance days, last 3 months")
plt.tight_layout(); plt.savefig(OUT / "06_account_extra.png", dpi=80)
