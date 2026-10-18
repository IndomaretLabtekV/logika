"""Residuals against the clean validation_v2 reference (v22: cross-fitted trees 50% + causilo 50%, bin-calibrated).
1) KS per segment for every clean model (where does Causilo beat the trees?).
2) Every existing feature: observed vs expected per safe bin (chi2/df) and cross-fit dKS, with a permutation null.
   A feature with residual signal is under-used by the models (misfit), not missing information.
Prints numbers, saves eda/out/103_clean_residual.png and .csv."""
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
from common9 import calibrate, safe_bins, oe_test, crossfit_dks

with contextlib.redirect_stdout(io.StringIO()):
    oof, test, names, R, Rt, res = blend2.main(["../results/v22/v22"])
y = oof["target"].to_numpy()
ref = res["trees_causilo_50"][0]
p = calibrate(y, ref)
f = features().filter(pl.col("is_train") == 1).sort("application_id")
assert (f["application_id"] == oof["application_id"]).all()

segs = {"thin": f["has_history"].to_numpy() == 0, "history": f["has_history"].to_numpy() == 1}
for t in (16, 20, 24, 28, 32):
    segs[f"rate{t}"] = np.isclose(f["annual_interest_rate"].to_numpy(), t / 100)
print(f"{'segment':10s} {'n':>6s} " + " ".join(f"{n:>8s}" for n in names + ['ref']))
for s, m in segs.items():
    print(f"{s:10s} {m.sum():6d} " + " ".join(f"{v.ks(y[m], R[m, j]):8.4f}" for j in range(len(names))) + f" {v.ks(y[m], ref[m]):8.4f}")

rng = np.random.default_rng(0)
null = [(oe_test(y, p, safe_bins(rng.permutation(f["pp_mean"].to_numpy()), 10))[0],
         crossfit_dks(y, p, safe_bins(rng.permutation(f["pp_mean"].to_numpy()), 10))) for _ in range(30)]
c95, d95 = np.quantile([a for a, _ in null], .95), np.quantile([b for _, b in null], .95)
print(f"\nnull (permuted feature): chi2/df p95 {c95:.2f}, dKS p95 {d95:+.5f}")
rows = []
for c in f.columns:
    if c in v.ID_COLS or f[c].dtype == pl.String:
        continue
    x = f[c].cast(pl.Float64).to_numpy()
    b = safe_bins(x, 10)
    rows.append(dict(feature=c, chi2=oe_test(y, p, b)[0], dks=crossfit_dks(y, p, b)))
d = pl.DataFrame(rows).sort("chi2", descending=True)
d.write_csv(OUT / "103_clean_residual.csv")
print(d.head(20))
print("features above both nulls:", d.filter((pl.col("chi2") > c95) & (pl.col("dks") > d95)).height, "of", d.height,
      f"(expected by chance ~{0.05 * 0.05 * d.height:.1f} if independent tests)")
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(d["chi2"], d["dks"], s=8)
ax.axvline(c95, color="r", lw=.5)
ax.axhline(d95, color="r", lw=.5)
for r in d.head(8).iter_rows(named=True):
    ax.annotate(r["feature"], (r["chi2"], r["dks"]), fontsize=6)
ax.set(xlabel="chi2/df vs clean reference", ylabel="cross-fit dKS", title="Residual signal of existing features")
fig.tight_layout()
fig.savefig(OUT / "103_clean_residual.png", dpi=110)
