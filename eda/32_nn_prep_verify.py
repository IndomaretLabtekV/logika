"""Verify the TabM input preparation (eda/nn_prep.py) does what it should:
(a) finite output, (b) order preserved -> univariate KS identical on non-missing rows, (c) roughly N(0,1) per
column, ties mapped to the middle of their block, (d) no train->test drift added, (e) missing-indicator groups,
(f) before/after figures by class, and a raw-vs-prepared view of one thin-file and one history customer."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, features, ks, psi
from nn_prep import nn_prep

f = features()
ID = ["application_id", "customer_id", "decision_datetime", "target", "is_train"]
CAT = ["loan_purpose", "region", "employment_type", "housing_status"]
FEATURES = sorted(c for c in f.columns if c not in ID)
NUM = [c for c in FEATURES if c not in CAT]
tr, te = f.filter(pl.col("is_train") == 1), f.filter(pl.col("is_train") == 0)
y = tr["target"].to_numpy()
A_tr = tr.select(NUM).cast(pl.Float64).fill_nan(None).to_numpy()
A_te = te.select(NUM).cast(pl.Float64).fill_nan(None).to_numpy()
Xtr, Xte, info = nn_prep(A_tr, A_te)
kept, ind = info["kept"], info["indicator_of"]
print(f"(a) input {A_tr.shape} -> output {Xtr.shape}: {len(kept)} numeric kept, {len(NUM) - len(kept)} constant dropped "
      f"({[NUM[j] for j in range(len(NUM)) if j not in kept]}), {len(ind)} missing-indicator groups")
print(f"    finite train/test: {np.isfinite(Xtr).all()} / {np.isfinite(Xte).all()}  dtype {Xtr.dtype}")
assert np.isfinite(Xtr).all() and np.isfinite(Xte).all()

# (b) order preservation: KS on non-missing rows must be identical
worst = 0.0
for i, j in enumerate(kept):
    m = ~np.isnan(A_tr[:, j])
    if m.sum() > 100 and y[m].min() != y[m].max():
        worst = max(worst, abs(ks(y[m], A_tr[m, j]) - ks(y[m], Xtr[m, i])))
print(f"(b) max |KS(raw) - KS(prepared)| over non-missing rows: {worst:.2e}")
assert worst < 1e-9

# (c) scale and ties
mu = np.array([Xtr[~np.isnan(A_tr[:, j]), i].mean() for i, j in enumerate(kept)])
sd = np.array([Xtr[~np.isnan(A_tr[:, j]), i].std() for i, j in enumerate(kept)])
print(f"(c) per-column mean over observed rows: median {np.median(mu):+.3f}, range [{mu.min():+.2f}, {mu.max():+.2f}]; "
      f"std median {np.median(sd):.3f}, range [{sd.min():.2f}, {sd.max():.2f}]")
for c in ["rp_frac_dpd1", "ec_orders_7d", "rp_n_dpd90", "ec2_cancel_28"]:
    j = NUM.index(c); i = kept.index(j)
    x = A_tr[:, j]; m = ~np.isnan(x)
    share0 = (x[m] == 0).mean()
    print(f"    {c:<16} share of zeros {share0:.2f} -> zero maps to {Xtr[m & (x == 0), i][0]:+.3f} "
          f"(middle of tie block = {float(__import__('scipy.special', fromlist=['ndtri']).ndtri(share0 / 2)):+.3f})")

# (d) drift added by the transform
ps = [psi(Xtr[:, i], Xte[:, i]) for i in range(len(kept))]
print(f"(d) max PSI train->test after transform: {max(ps):.5f} (raw max ~0.0012)")

# (e) indicator groups
groups = {}
for i in ind:
    groups[i] = [NUM[kept[k]] for k in range(len(kept)) if (np.isnan(A_tr[:, kept[k]]) == np.isnan(A_tr[:, kept[i]])).all()]
print("(e) missing-indicator groups (first member, size, missing share, bad rate when missing):")
for i, members in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:10]:
    miss = np.isnan(A_tr[:, kept[i]])
    print(f"    {members[0]:<26} cols={len(members):>3} missing={miss.mean():.3f} bad|missing={y[miss].mean():.3f} bad|present={y[~miss].mean():.3f}")

# (f) figures + raw vs prepared rows
show = ["rp_frac_dpd1", "ec_accel_30d", "bc_score_mean", "declared_monthly_income", "ac_out_in_delta", "sv_maxdpd_mean_late"]
fig, ax = plt.subplots(2, len(show), figsize=(3.2 * len(show), 6))
for k, c in enumerate(show):
    j = NUM.index(c); i = kept.index(j); m = ~np.isnan(A_tr[:, j])
    for t in (0, 1):
        mm = m & (y == t)
        ax[0, k].hist(np.clip(A_tr[mm, j], *np.nanpercentile(A_tr[:, j], [1, 99])), bins=40, alpha=.5, density=True, label=f"y={t}")
        ax[1, k].hist(Xtr[mm, i], bins=40, alpha=.5, density=True)
    ax[0, k].set_title(f"{c}\nraw (1-99 pct)", fontsize=8); ax[1, k].set_title("prepared", fontsize=8)
ax[0, 0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "32_nn_prep.png", dpi=80)
hist = tr["has_history"].to_numpy()
for lab, r in [("thin-file", int(np.argmax(hist == 0))), ("history", int(np.argmax(hist == 1)))]:
    cols = ["rp_frac_dpd1", "ec_accel_30d", "bc_score_mean", "ac_out_in_delta"]
    raw = {c: A_tr[r, NUM.index(c)] for c in cols}
    prep = {c: float(Xtr[r, kept.index(NUM.index(c))]) for c in cols}
    print(f"(f) {lab} row {r}: raw {({k: round(float(v), 3) for k, v in raw.items()})}\n    prepared {({k: round(v, 3) for k, v in prep.items()})}"
          f"  indicators on: {int(Xtr[r, len(kept):].sum())}/{len(ind)}")
