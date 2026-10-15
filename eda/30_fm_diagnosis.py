"""Why do the foundation models (TabPFN-3.5, Causilo) lag the GBDTs, and do they carry anything the GBDTs miss?
Uses only saved v2 OOF predictions: segment KS per model, residual test of FM scores against the GBDT blend,
disagreement analysis, and the feature budget the FMs saw (64 of 214)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata
from common import OUT, ROOT, calibrate, features, ks, resid_test

f = features()
tr = f.filter(pl.col("is_train") == 1)
y = tr["target"].to_numpy()
R = ROOT / "results" / "v2"
N = ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s", "tabpfn", "causilo"]
O = {n: rankdata(np.load(R / f"oof_{n}.npy")) / len(y) for n in N}
gb = np.mean([O[n] for n in N[:6]], 0)
fm = (O["tabpfn"] + O["causilo"]) / 2
print(f"GBDT equal blend KS={ks(y, gb):.4f}   FM pair KS={ks(y, fm):.4f}   rank corr={np.corrcoef(gb, fm)[0, 1]:.4f}")

# 1) segment KS per model
seg = {"thin-file": tr["has_history"].to_numpy() == 0, "history": tr["has_history"].to_numpy() == 1}
nrep = tr["rp_n_inst"].fill_null(0).to_numpy()
seg["history, <=12 inst"] = (nrep > 0) & (nrep <= 12)
seg["history, >12 inst"] = nrep > 12
print("\n=== KS by segment")
print(f"{'model':<10}" + "".join(f"{k:>22}" for k in seg))
for n, s in list(O.items()) + [("GBDT", gb), ("FM", fm)]:
    print(f"{n:<10}" + "".join(f"{ks(y[m], s[m]):>22.4f}" for m in seg.values()))

# 2) does the FM score explain residual of the GBDT blend?
p = calibrate(y, gb)
for n in ["tabpfn", "causilo"]:
    chi, lift, dks = resid_test(y, p, O[n])
    print(f"\nresidual test of {n} vs GBDT blend: chi2/df={chi:.2f}  dKS_cf={dks:+.4f} (noise p95 ~1.2 / +0.002)")
for w in (0.1, 0.2, 0.3):
    print(f"  blend GBDT*(1-{w}) + FM*{w}: KS={ks(y, (1 - w) * gb + w * fm):.4f}"
          f"  thin={ks(y[seg['thin-file']], ((1 - w) * gb + w * fm)[seg['thin-file']]):.4f}"
          f"  hist={ks(y[seg['history']], ((1 - w) * gb + w * fm)[seg['history']]):.4f}")

# 3) cross-segment calibration: is the pooled ranking hurt by segment offsets?
for name, s in [("GBDT", gb), ("FM", fm)]:
    pooled = ks(y, s)
    # perfect within-segment ranks re-pooled by within-segment calibration (bad rate per score bin)
    cal = np.zeros(len(y))
    for m in seg["thin-file"], seg["history"]:
        cal[m] = calibrate(y[m], s[m], bins=20)
    print(f"{name}: pooled KS {pooled:.4f} vs segment-recalibrated pooled KS {ks(y, cal):.4f}")

# 4) disagreement: where FM ranks a bad much higher than GBDT does
d = fm - gb
q = np.quantile(d, [0.05, 0.95])
for lab, m in [("FM >> GBDT", d > q[1]), ("FM << GBDT", d < q[0]), ("agree", (d >= q[0]) & (d <= q[1]))]:
    print(f"{lab:<11} n={m.sum():>6} bad rate={y[m].mean():.3f} thin-file share={seg['thin-file'][m].mean():.2f}")

# 5) the FM feature budget: how much univariate signal sits outside the top-64-by-KS
ks_csv = OUT / "27_v2_screen.csv"
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for n, s in [("GBDT", gb), ("TabPFN", O["tabpfn"]), ("Causilo", O["causilo"])]:
    ax[0].plot([ks(y[m], s[m]) for m in seg.values()], "o-", label=n)
ax[0].set_xticks(range(len(seg))); ax[0].set_xticklabels(list(seg), rotation=15, fontsize=8); ax[0].set_title("KS by segment"); ax[0].legend()
ax[1].hist(d[y == 1], bins=60, alpha=.5, density=True, label="bad"); ax[1].hist(d[y == 0], bins=60, alpha=.5, density=True, label="good")
ax[1].set_title("FM rank - GBDT rank"); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT / "30_fm_diagnosis.png", dpi=80)
