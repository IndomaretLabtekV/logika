"""Helpers for the v9 analyses (eda/90+): the v8 blend OOF as the reference score, and residual tests that keep
zero / rare values in their own bins (the quantile-bin collapse found in eda/v8_review/04 is avoided)."""
import numpy as np
import polars as pl
from common import OUT, PQ, ROOT, features, ks, load  # noqa: F401


def v8_ref():
    """Train rows of features() with the v8 blend OOF (rank score) and its bin-calibrated probability."""
    s = pl.read_parquet(ROOT / "eda" / "v8_review" / "out" / "00_bound_oof.parquet").select("application_id", "blend")
    tr = features().filter(pl.col("is_train") == 1).join(s, on="application_id", how="left", maintain_order="left")
    assert tr["blend"].null_count() == 0
    y = tr["target"].to_numpy()
    return tr, y, calibrate(y, tr["blend"].to_numpy())


def calibrate(y, s, bins=50):
    b = np.minimum((np.argsort(np.argsort(s)) * bins) // len(s), bins - 1)
    rate = np.bincount(b, weights=y, minlength=bins) / np.bincount(b, minlength=bins)
    return rate[b]


def safe_bins(x, n=8):
    """Bins for a numeric candidate: NaN own bin, exact zero own bin, few distinct values kept as levels,
    otherwise quantile bins on the non-zero values."""
    x = np.asarray(x, float)
    out = np.full(len(x), -1)
    m = np.isfinite(x)
    vals = np.unique(x[m])
    if len(vals) <= n:
        out[m] = np.searchsorted(vals, x[m])
        return out
    z = m & (x == 0)
    out[z] = 0
    nz = m & (x != 0)
    q = np.unique(np.quantile(x[nz], np.linspace(0, 1, n + 1))[1:-1])
    out[nz] = 1 + np.searchsorted(q, x[nz], side="right")
    return out


def oe_test(y, p, b):
    """Observed vs expected bad count per bin of b given calibrated p. Returns (chi2/df, table rows)."""
    rows, chi, k = [], 0.0, 0
    for v in np.unique(b):
        m = b == v
        o, e, var = y[m].sum(), p[m].sum(), (p[m] * (1 - p[m])).sum()
        rows.append((int(v), int(m.sum()), float(o), float(e), float(o / max(e, 1e-9)), float((o - e) / np.sqrt(max(var, 1e-9)))))
        chi += (o - e) ** 2 / max(var, 1e-9); k += 1
    return chi / max(k - 1, 1), rows


def crossfit_dks(y, p, b, seed=0, reps=3):
    """KS gain of logit(p) + log(O/E) per bin, bin effects fitted on one half and applied to the other."""
    lp = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    _, inv = np.unique(b, return_inverse=True)
    K = inv.max() + 1
    gains = []
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        half = rng.random(len(y)) < 0.5
        adj = np.zeros(len(y))
        for fit in (half, ~half):
            o = np.bincount(inv[fit], weights=y[fit], minlength=K) + 5
            e = np.bincount(inv[fit], weights=p[fit], minlength=K) + 5
            adj[~fit] = np.log(o / e)[inv[~fit]]
        gains.append(ks(y, lp + adj) - ks(y, lp))
    return float(np.mean(gains))
