"""Where would historical rows help, and how should they enter? Same counting-only additive score as eda/82,
same 3 x 2 held-out halves, now reporting KS on the held-out half overall and per segment (thin-file vs history),
for these training sets:
  main      : half A of the selected loans
  +hist     : A + all historical rows
  +hist w.5 : A + historical rows with weight 0.5
  +hist k1  : A + only order-1 snapshots (the loan without any earlier repayment history: thin-file at that time)
  +hist noec: A + historical rows, but e-commerce / price features are blanked on them (their ec mapping differs)
A difference is credible only if it is consistent over the six halves."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, PQ, features, ks

f = features()
M = f.filter(pl.col("is_train") == 1)
H = pl.read_parquet(PQ / "hist_feats.parquet")
K = pl.read_csv(OUT / "82_feature_ks.csv")
cols = K.filter(pl.col("ks_main") > 0.05)["feature"].to_list()
EC = [c for c in cols if c.startswith(("ec_", "ec2_", "pp_", "ecspend"))]
y_m, y_h = M["target"].to_numpy(), H["target"].to_numpy()
w_h = np.ones(len(y_h))
arr = lambda d: {c: d[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float) for c in cols}  # noqa: E731
XM, XH = arr(M), arr(H)
XH_noec = {c: (np.full(len(y_h), np.nan) if c in EC else XH[c]) for c in cols}
k1 = H["snap_order"].to_numpy() == 1
hist = M["has_history"].to_numpy() == 1
print(f"{len(cols)} features ({len(EC)} e-commerce/price); historical rows {len(y_h):,} ({k1.sum():,} order-1)")


def edges(x):
    return np.unique(np.nanquantile(x, np.linspace(0, 1, 11)[1:-1])) if np.isfinite(x).any() else np.array([])


def bins(x, e):
    b = np.searchsorted(e, x, side="right"); b[~np.isfinite(x)] = len(e) + 1
    return b


def additive(Xf, yf, wf, Xe, rounds=4, shrink=0.5, prior=50.0):
    pr = np.average(yf, weights=wf); base = np.log(pr / (1 - pr))
    E = {c: edges(Xf[c][wf > 0]) for c in cols}
    Bf, Be = {c: bins(Xf[c], E[c]) for c in cols}, {c: bins(Xe[c], E[c]) for c in cols}
    adj = {c: np.zeros(len(E[c]) + 2) for c in cols}
    lp = np.full(len(yf), base)
    for _ in range(rounds):
        for c in cols:
            lp_c = lp - adj[c][Bf[c]]
            p = 1 / (1 + np.exp(-lp_c))
            o = np.bincount(Bf[c], weights=wf * yf, minlength=len(adj[c]))
            e = np.bincount(Bf[c], weights=wf * p, minlength=len(adj[c]))
            adj[c] = shrink * np.log((o + prior * pr) / (e + prior * pr))
            lp = lp_c + adj[c][Bf[c]]
    return base + sum(adj[c][Be[c]] for c in cols)


def cat(Xa, Xb):
    return {c: np.r_[Xa[c], Xb[c]] for c in cols}


rng = np.random.default_rng(83)
rows = []
for rep in range(3):
    half = rng.random(len(y_m)) < 0.5
    for a, b in ((half, ~half), (~half, half)):
        A = {c: XM[c][a] for c in cols}; B = {c: XM[c][b] for c in cols}
        wa = np.ones(a.sum())
        variants = {
            "main": (A, y_m[a], wa),
            "+hist": (cat(A, XH), np.r_[y_m[a], y_h], np.r_[wa, w_h]),
            "+hist w.5": (cat(A, XH), np.r_[y_m[a], y_h], np.r_[wa, 0.5 * w_h]),
            "+hist k1": (cat(A, {c: XH[c][k1] for c in cols}), np.r_[y_m[a], y_h[k1]], np.r_[wa, w_h[k1]]),
            "+hist noec": (cat(A, XH_noec), np.r_[y_m[a], y_h], np.r_[wa, w_h]),
        }
        for name, (Xf, yf, wf) in variants.items():
            s = additive(Xf, yf, wf, B)
            hb = hist[b]
            rows.append((rep, name, ks(y_m[b], s), ks(y_m[b][~hb], s[~hb]), ks(y_m[b][hb], s[hb])))
R = pl.DataFrame(rows, schema=["rep", "variant", "ks_all", "ks_thin", "ks_hist"], orient="row")
R = R.with_columns(half=pl.int_range(pl.len()).over("variant"))
base = R.filter(pl.col("variant") == "main").select("half", b_all="ks_all", b_thin="ks_thin", b_hist="ks_hist")
D = R.join(base, on="half").with_columns(d_all=pl.col("ks_all") - pl.col("b_all"), d_thin=pl.col("ks_thin") - pl.col("b_thin"),
                                         d_hist=pl.col("ks_hist") - pl.col("b_hist"))
S = D.group_by("variant", maintain_order=True).agg(
    ks_all=pl.col("ks_all").mean(), d_all=pl.col("d_all").mean(), up_all=(pl.col("d_all") > 0).sum(),
    d_thin=pl.col("d_thin").mean(), up_thin=(pl.col("d_thin") > 0).sum(), d_hist=pl.col("d_hist").mean(), up_hist=(pl.col("d_hist") > 0).sum())
with pl.Config(float_precision=4):
    print(S)
S.write_csv(OUT / "83_augment_segments.csv")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, seg in enumerate(["d_all", "d_thin", "d_hist"]):
    for v in S["variant"].to_list()[1:]:
        ax[i].scatter([v] * 6, D.filter(pl.col("variant") == v)[seg], s=15)
    ax[i].axhline(0, color="k", lw=0.5); ax[i].set_title(f"held-out KS change vs main only: {seg[2:]}"); ax[i].tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig(OUT / "83_augment_segments.png", dpi=80)
print("saved", OUT / "83_augment_segments.png")
