"""Does the v8 blend lose KS because its score scale is not consistent across segments?

KS uses ONE threshold for the whole population, so the best pooled score is a calibrated probability. A rank average
can map the same score to different bad rates in different segments (thin-file vs history, months, rate tiers,
purposes). Tests, all cross-fitted (fit on one half, score the other, both directions, 3 repeats):
  1) per-segment recalibration of the blend: bad rate in score bins estimated within each segment;
  2) logit-space stacking of the saved model predictions (Newton logistic regression, numpy only) instead of the
     rank average, optionally with segment interactions.
These are fits on 16 saved prediction columns, not model training on features."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata
from common import ROOT, features, ks

R8 = ROOT / "results" / "v8"
ids = np.load(R8 / "train_ids.npy", allow_pickle=True).astype(str)
f = pl.DataFrame({"application_id": ids}).join(features().filter(pl.col("is_train") == 1), on="application_id", maintain_order="left")
y = f["target"].to_numpy()
blend = pl.DataFrame({"application_id": ids}).join(
    pl.read_parquet(ROOT / "eda" / "v8_review" / "out" / "00_bound_oof.parquet"), on="application_id", maintain_order="left")["blend"].to_numpy()
names = sorted(p.stem[4:] for p in R8.glob("oof_*.npy"))
P = np.column_stack([np.load(R8 / f"oof_{n}.npy") for n in names])
L = np.log(np.clip(P, 1e-5, 1 - 1e-5) / (1 - np.clip(P, 1e-5, 1 - 1e-5)))
SEG = {
    "history": f["has_history"].to_numpy(),
    "month": f["decision_datetime"].dt.month().to_numpy(),
    "rate": rankdata(f["annual_interest_rate"].to_numpy(), method="dense"),
    "purpose": rankdata(f["loan_purpose"].to_numpy(), method="dense"),
    "employment": rankdata(f["employment_type"].to_numpy(), method="dense"),
}


def bin_cal(s_fit, y_fit, s_ev, bins=20):
    q = np.quantile(s_fit, np.linspace(0, 1, bins + 1)[1:-1])
    bf, be = np.searchsorted(q, s_fit), np.searchsorted(q, s_ev)
    rate = (np.bincount(bf, weights=y_fit, minlength=bins) + 1) / (np.bincount(bf, minlength=bins) + 1 / y_fit.mean())
    # monotone in the score: cumulative max keeps the ranking inside the segment
    return np.maximum.accumulate(rate)[be] + 1e-6 * rankdata(s_ev) / len(s_ev)


def logreg(X, yy, l2=1.0, it=25):
    X = np.column_stack([np.ones(len(X)), X]); w = np.zeros(X.shape[1])
    for _ in range(it):
        pr = 1 / (1 + np.exp(-X @ w)); g = X.T @ (pr - yy) + l2 * np.r_[0, w[1:]]
        H = (X * (pr * (1 - pr))[:, None]).T @ X + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(H, g)
    return w


def apply(w, X):
    return np.column_stack([np.ones(len(X)), X]) @ w


rng = np.random.default_rng(96)
res = {}
for rep in range(3):
    half = rng.random(len(y)) < 0.5
    for fit, ev in ((half, ~half), (~half, half)):
        res.setdefault("v8 blend (rank average)", []).append(ks(y[ev], blend[ev]))
        for sname, seg in SEG.items():
            s = np.zeros(ev.sum())
            for v in np.unique(seg):
                mf, me = fit & (seg == v), (seg[ev] == v)
                s[me] = bin_cal(blend[mf], y[mf], blend[ev][me])
            res.setdefault(f"recalibrate within {sname}", []).append(ks(y[ev], s))
        w = logreg(L[fit], y[fit]); res.setdefault("logit stacking", []).append(ks(y[ev], apply(w, L[ev])))
        Lm = (L - L.mean(1, keepdims=True))
        Xh = np.column_stack([L.mean(1), SEG["history"], L.mean(1) * SEG["history"]])
        w = logreg(Xh[fit], y[fit]); res.setdefault("mean logit x history", []).append(ks(y[ev], apply(w, Xh[ev])))
        Xs = np.column_stack([L, L * SEG["history"][:, None], SEG["history"]])
        w = logreg(Xs[fit], y[fit], l2=10); res.setdefault("logit stacking x history", []).append(ks(y[ev], apply(w, Xs[ev])))
base = np.array(res["v8 blend (rank average)"])
rows = [(k, float(np.mean(v)), float(np.mean(np.array(v) - base)), int((np.array(v) > base).sum())) for k, v in res.items()]
R = pl.DataFrame(rows, schema=["score", "KS_heldout_half", "gain_vs_v8", "halves_up_of_6"], orient="row")
with pl.Config(float_precision=4, fmt_str_lengths=40):
    print(R)
R.write_csv(OUT_ := ROOT / "eda" / "out" / "96_segment_calibration.csv")
# calibration curves of the blend per history segment
fig, ax = plt.subplots(figsize=(6, 4))
for v, lab in ((0, "thin-file"), (1, "history")):
    m = SEG["history"] == v
    q = np.quantile(blend, np.linspace(0, 1, 21))
    b = np.clip(np.searchsorted(q, blend[m]) - 1, 0, 19)
    ax.plot(range(20), [y[m][b == k].mean() for k in range(20)], "o-", label=lab)
ax.set_xlabel("global blend score bin (20 quantiles)"); ax.set_ylabel("bad rate"); ax.legend(); ax.set_title("same score, different bad rate?")
plt.tight_layout(); plt.savefig(ROOT / "eda" / "out" / "96_segment_calibration.png", dpi=80)
print("saved", ROOT / "eda" / "out" / "96_segment_calibration.png")
