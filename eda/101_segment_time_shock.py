"""H12: a macro shock that hits one segment (region, employment type, purpose, housing, rate tier) during part of the
outcome window would show as observed-minus-expected bad counts that drift with decision month inside that segment.
eda/64 only tested the overall weekly total. Reference = calibrated v8 blend OOF. A permutation of the month labels
within segment gives the null for the max |z| and chi2. Prints numbers, saves eda/out/101_segment_time_shock.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT
from common9 import v8_ref

tr, y, p = v8_ref()
month = tr["decision_datetime"].dt.month().to_numpy()
quarter = (month - 1) // 3
segs = {c: tr[c].cast(pl.String).to_numpy() for c in ["region", "employment_type", "loan_purpose", "housing_status"]}
segs["rate"] = tr["annual_interest_rate"].round(2).cast(pl.String).to_numpy()
segs["history"] = tr["has_history"].cast(pl.String).to_numpy()


def table(seg, t):
    """chi2 over segment x time cells of (O-E)/sd after removing each segment's overall O/E (main effect)."""
    chi, zs = 0.0, []
    for s in np.unique(seg):
        m = seg == s
        scale = y[m].sum() / p[m].sum()
        for k in np.unique(t[m]):
            c = m & (t == k)
            e, var = p[c].sum() * scale, (p[c] * (1 - p[c])).sum()
            z = (y[c].sum() - e) / np.sqrt(var)
            zs.append((s, k, c.sum(), y[c].sum() / max(e, 1e-9), z))
            chi += z * z
    return chi / len(zs), zs


rng = np.random.default_rng(0)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (name, seg) in zip(axes.ravel(), segs.items()):
    for tname, t in [("month", month), ("quarter", quarter)]:
        chi, zs = table(seg, t)
        null = []
        for _ in range(100):
            perm = t.copy()
            for s in np.unique(seg):
                m = seg == s
                perm[m] = rng.permutation(t[m])
            null.append(table(seg, perm)[0])
        zmax = max(abs(r[4]) for r in zs)
        print(f"{name:16s} x {tname:7s} cells {len(zs):3d}  chi2/cell {chi:5.2f}  null p95 {np.quantile(null, .95):5.2f}  "
              f"p={np.mean(np.array(null) >= chi):.2f}  max|z| {zmax:.2f}")
        if tname == "month":
            for s in np.unique(seg):
                rows = [r for r in zs if r[0] == s]
                ax.plot([r[1] for r in rows], [r[3] for r in rows], "o-", label=s)
    ax.axhline(1, color="k", lw=.5)
    ax.set(title=f"O/E by decision month: {name}", xlabel="month")
    ax.legend(fontsize=6)
fig.tight_layout()
fig.savefig(OUT / "101_segment_time_shock.png", dpi=100)
