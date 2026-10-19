"""Systematic 'relative to peers' scan (the construction that produced pp_mean): every numeric feature x as
(a) x - median(x | group) / IQR(x | group) and (b) within-group percentile, for 7 categorical peer groups (purpose,
region, employment, housing, rate tier, has_history, decision month). Group statistics use x only (train + test rows,
no labels; every feature is already cutoff-safe). Residual test vs the clean TREES-ONLY OOF (eda/134 setup): O/E
chi2/df per safe bin and cross-fit dKS; the top candidates by chi2 are re-scored with 3 extra half-split seeds.
Multiple testing: Bonferroni-style threshold from the permuted-feature null max over the same number of tests.
Saves eda/out/137_peer_relative.csv / .png.
Usage: cd eda && ../.venv/bin/python 137_peer_relative.py"""
import io
import contextlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import blend2
import valid2 as v
from common import OUT, features
from common9 import calibrate, crossfit_dks, oe_test, safe_bins

f = features().sort("application_id")
f = f.with_columns(rate_tier=(pl.col("annual_interest_rate") * 100).round().cast(pl.Int32))
groups = ["loan_purpose", "region", "employment_type", "housing_status", "rate_tier", "has_history", "dec_month"]
skip = set(v.ID_COLS) | set(groups) | {"annual_interest_rate"}
nums = [c for c in f.columns if c not in skip and f[c].dtype not in (pl.String, pl.Categorical)]
is_tr = f["is_train"].to_numpy() == 1
with contextlib.redirect_stdout(io.StringIO()):
    oof, _, _, _, _, res = blend2.main(["../results/v22/v22"])
assert (oof["application_id"] == f.filter(pl.col("is_train") == 1)["application_id"]).all()
y = oof["target"].to_numpy()
pc = calibrate(y, res["equal_trees"][0])

rows = []
for g in groups:
    exprs = []
    for c in nums:
        x = pl.col(c).cast(pl.Float64).fill_nan(None)
        med = x.median().over(g)
        iqr = (x.quantile(0.75) - x.quantile(0.25)).over(g)
        exprs += [((x - med) / (iqr + 1e-9)).alias(f"{c}|z|{g}"), (x.rank() / x.count()).over(g).alias(f"{c}|r|{g}")]
    F = f.select(exprs).filter(pl.Series(is_tr))
    for name in F.columns:
        xx = F[name].to_numpy().astype(float)
        if np.isfinite(xx).sum() < 5000 or np.nanstd(xx) == 0:
            continue
        rows.append(dict(feature=name, chi2=oe_test(y, pc, safe_bins(xx, 10))[0]))
    print(f"group {g}: {len(rows)} candidates so far", flush=True)
d = pl.DataFrame(rows).sort("chi2", descending=True)
# null: max chi2 over the same number of tests, permuted copies of random candidates
rng = np.random.default_rng(0)
top = d.head(40)["feature"].to_list()
fe_all = {}
for name in top:
    c, kind, g = name.split("|")
    x = pl.col(c).cast(pl.Float64).fill_nan(None)
    e = ((x - x.median().over(g)) / ((x.quantile(0.75) - x.quantile(0.25)).over(g) + 1e-9)) if kind == "z" else (x.rank() / x.count()).over(g)
    fe_all[name] = f.select(e.alias("v")).filter(pl.Series(is_tr))["v"].to_numpy().astype(float)
null_chi = []
for _ in range(200):
    z = rng.permutation(fe_all[top[rng.integers(len(top))]])
    null_chi.append(oe_test(y, pc, safe_bins(z, 10))[0])
null_chi = np.array(null_chi)
# Bonferroni over n tests: per-test p = 0.05 / n -> use the normal-approx tail of the permuted chi2/df distribution
n = d.height
thr = np.quantile(null_chi, 0.95) + (np.std(null_chi) * (np.sqrt(2 * np.log(n)) - 1.645))
print(f"{n} candidates; permuted chi2/df mean {null_chi.mean():.2f} p95 {np.quantile(null_chi, .95):.2f}; "
      f"approx family-wise threshold {thr:.2f}")
out = []
for name in top[:20]:
    b = safe_bins(fe_all[name], 10)
    dk = [crossfit_dks(y, pc, b, seed=s) for s in range(4)]
    out.append(dict(feature=name, chi2=d.filter(pl.col("feature") == name)["chi2"][0], dks_mean=float(np.mean(dk)), dks_min=float(np.min(dk))))
o = pl.DataFrame(out)
with pl.Config(tbl_rows=25, fmt_str_lengths=60, tbl_width_chars=160):
    print(o)
d.write_csv(OUT / "137_peer_relative.csv")
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(d["chi2"].to_numpy(), 80, label="candidates")
ax.hist(null_chi, 40, alpha=.6, label="permuted")
ax.axvline(thr, color="r", ls="--", label="family-wise thr")
ax.set(xlabel="chi2/df vs trees-only OOF", yscale="log", title="Peer-relative features")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "137_peer_relative.png", dpi=110)
