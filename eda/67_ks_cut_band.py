"""KS = TPR - FPR at ONE cut. Only the side of the cut each row lands on matters, so what counts is how well models
order the rows near the cut, not the global AUC. This script (saved OOF only, no training):
 1) locates the KS cut of the v6 blend and the share of rows in bands around it;
 2) measures each model's discrimination (AUC) inside the band, where the KS is decided;
 3) tests a band-specific blend (weights chosen on one half inside the band, scored on the other half) against the
    global blend, so any gain is cross-fitted.
If one model is clearly better inside the band, a KS-targeted blend or a band-focused second-stage model is a real
lever; if all models are equal there, the KS is limited by information, not by the blend."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata
from common import OUT, ROOT, features, ks, v6_oof

f = features(); tr = f.filter(pl.col("is_train") == 1); y = tr["target"].to_numpy()
s = v6_oof(); n = len(y)
names = ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s", "tabpfn", "tabm", "tabm_p"]
R = {m: rankdata(np.load(ROOT / "results" / "v6" / f"oof_{m}.npy")) / n for m in names}
r = rankdata(s) / n
o = np.argsort(s); F1 = np.cumsum(y[o]) / y.sum(); F0 = np.cumsum(1 - y[o]) / (n - y.sum())
i = np.argmax(np.abs(F1 - F0)); q_cut = (i + 1) / n
print(f"v6 blend KS {ks(y, s):.4f}; cut at score quantile {q_cut:.3f} (top {1 - q_cut:.1%} flagged), "
      f"TPR {1 - F1[i]:.3f} FPR {1 - F0[i]:.3f}")


def auc(yy, x):
    rk = rankdata(x); p = yy.sum(); return (rk[yy == 1].sum() - p * (p + 1) / 2) / (p * (len(yy) - p))


rows = []
for w in (0.05, 0.10, 0.20):
    band = np.abs(r - q_cut) <= w
    a = {m: auc(y[band], R[m][band]) for m in names}
    a["blend"] = auc(y[band], s[band])
    rows.append((w, band.mean(), y[band].mean(), *[a[m] for m in names + ["blend"]]))
B = pl.DataFrame(rows, schema=["half_width", "share", "bad_rate", *names, "blend"], orient="row")
with pl.Config(float_precision=4, tbl_cols=20):
    print("AUC inside the band around the KS cut (rows whose blend quantile is within +-half_width of the cut):")
    print(B)

# cross-fitted band blend: inside band reorder with the best single model or a greedy blend fit on the other half
rng = np.random.default_rng(1); half = rng.random(n) < 0.5
res = []
for w in (0.05, 0.10, 0.20):
    band = np.abs(r - q_cut) <= w
    for fit, ev in ((half, ~half), (~half, half)):
        bf = band & fit
        best = max(names, key=lambda m: auc(y[bf], R[m][bf]))
        for alpha in (0.5, 1.0):
            new = r.copy()
            # inside the band keep the band's rank range but reorder by (1-alpha)*blend + alpha*best model
            idx = np.where(band)[0]
            mix = (1 - alpha) * rankdata(s[idx]) + alpha * rankdata(R[best][idx])
            new[idx] = np.sort(r[idx])[np.argsort(np.argsort(mix))]
            res.append((w, alpha, best, ks(y[ev], r[ev]), ks(y[ev], new[ev])))
E = pl.DataFrame(res, schema=["half_width", "alpha", "best_in_band", "ks_global", "ks_band_reordered"], orient="row").with_columns(
    gain=pl.col("ks_band_reordered") - pl.col("ks_global"))
with pl.Config(float_precision=4, tbl_rows=20):
    print(E)
    print(E.group_by("half_width", "alpha").agg(pl.col("gain").mean()).sort("half_width", "alpha"))
E.write_csv(OUT / "67_ks_cut_band.csv"); B.write_csv(OUT / "67_band_auc.csv")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(np.arange(n) / n, F0 - F1); ax[0].axvline(q_cut, color="r", ls="--"); ax[0].set_xlabel("blend score quantile")
ax[0].set_title("F0 - F1 along the score (KS = peak)")
ax[1].bar(names + ["blend"], B.row(1)[3:]); ax[1].set_ylim(0.5, 0.75); ax[1].tick_params(axis="x", rotation=45)
ax[1].set_title("AUC inside +-10% band around the KS cut")
plt.tight_layout(); plt.savefig(OUT / "67_ks_cut_band.png", dpi=80)
