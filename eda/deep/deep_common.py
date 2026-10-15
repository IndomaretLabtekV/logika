"""Shared I/O and diagnostics. No model fitting, inference, or network access."""
import os
os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import polars as pl
from scipy.stats import rankdata, chi2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "mpl_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(HERE.parent))
from common import ks, ks_nan, psi
IDS = {"application_id", "customer_id", "decision_datetime", "target", "is_train"}
CATS = ["loan_purpose", "region", "employment_type", "housing_status"]
TABLES = ["train_applications", "test_applications", "customer_profiles", "previous_applications",
          "repayment_history", "account_behavior", "bureau_observations", "ecommerce_transactions"]
TREES = ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]

def table(name):
    p = ROOT / "data_pq" / (name + ".parquet")
    if p.exists():
        return pl.read_parquet(p)
    return pl.read_csv(ROOT / "data" / (name + ".csv"), try_parse_dates=True, infer_schema_length=100000)

def features():
    return pl.read_parquet(ROOT / "data_pq/features.parquet")

def apps():
    return pl.concat([table("train_applications"), table("test_applications")], how="diagonal_relaxed")

def save_table(rows, name):
    df = rows if isinstance(rows, pl.DataFrame) else pl.DataFrame(rows)
    df.write_csv(OUT / (name + ".csv"))
    with pl.Config(tbl_rows=16, tbl_cols=12, fmt_str_lengths=42, float_precision=5):
        print(name, df.shape)
        print(df.head(16))
    return df

def save_json(obj, name):
    (OUT / (name + ".json")).write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    print(name, json.dumps(obj, default=str))

def figsave(fig, name):
    fig.tight_layout()
    p = OUT / (name + ".png")
    fig.savefig(p, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print("FIGURE", p.relative_to(ROOT))

def folds(y, seed=42, k=5):
    rng = np.random.default_rng(seed)
    f = np.empty(len(y), np.int8)
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        f[idx] = np.arange(len(idx)) % k
    return f

def v3():
    f = features().filter(pl.col("is_train") == 1)
    y = f["target"].to_numpy()
    names = TREES + ["tabpfn", "tabm"]
    raw = {n: np.load(ROOT / "results/v3" / ("oof_" + n + ".npy")) for n in names}
    assert all(p.shape == y.shape and np.isfinite(p).all() for p in raw.values())
    # Legacy npy files lack IDs: bind to cached feature order only if recorded metrics and CSV agree.
    assert abs(ks(y, raw["lgb"]) - 0.5117563863351741) < 1e-10, "OOF row-order mismatch"
    R = np.column_stack([rankdata(raw[n]) / len(y) for n in names])
    w = np.array([8, 3, 1, 3, 2, 0, 0, 13]) / 30
    score = R @ w
    assert abs(ks(y, score) - 0.514227) < 1e-6
    te = features().filter(pl.col("is_train") == 0)
    rt = np.column_stack([rankdata(np.load(ROOT / "results/v3" / ("test_" + n + ".npy"))) / te.height for n in names])
    pred = rt @ w
    pred = (pred - pred.min()) / np.ptp(pred)
    sub = pl.read_csv(ROOT / "results/v3/submission.csv")
    ck = te.select("application_id").with_columns(expected=pl.Series(pred)).join(sub, on="application_id", validate="1:1")
    assert ck.height == te.height and np.allclose(ck["expected"], ck["default_probability"], atol=1e-12)
    return f, y, raw, R, score

def signed_ks(y, score):
    from scipy.stats import ks_2samp
    # Positive statistic means increasing score corresponds to increasing risk.
    return float(ks_2samp(score[y == 0], score[y == 1], alternative="greater").statistic)

def qbins(x, reference, n=6):
    x, reference = np.asarray(x, float), np.asarray(reference, float)
    r = reference[np.isfinite(reference)]
    cuts = np.unique(np.quantile(r, np.linspace(0, 1, n + 1)[1:-1])) if len(r) else np.array([])
    return np.where(np.isfinite(x), np.searchsorted(cuts, x, side="right"), len(cuts) + 1).astype(int)

def bh(p):
    p = np.asarray(p, float)
    ix = np.argsort(p)
    q = np.minimum.accumulate((p[ix] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty(len(p)); out[ix] = np.minimum(q, 1)
    return out

def conditional_test(y, bins, strata):
    """Generalized stratified contingency score test; diagnostic, not a trained predictor."""
    y = np.asarray(y)
    _, s = np.unique(strata, return_inverse=True)
    _, g = np.unique(bins, return_inverse=True)
    ns, ng = s.max() + 1, g.max() + 1
    n = np.bincount(s * ng + g, minlength=ns * ng).reshape(ns, ng).astype(float)
    obs = np.bincount(s * ng + g, weights=y, minlength=ns * ng).reshape(ns, ng)
    total = n.sum(1)
    p = obs.sum(1) / total
    U = (obs - n * p[:, None]).sum(0)
    V = np.zeros((ng, ng))
    for ni, pi, ti in zip(n, p, total):
        if ti > 1:
            V += pi * (1 - pi) * ti / (ti - 1) * (np.diag(ni) - np.outer(ni, ni) / ti)
    ev, vec = np.linalg.eigh(V)
    keep = ev > max(ev.max(initial=0) * 1e-9, 1e-10)
    df = int(keep.sum())
    stat = float(np.sum((vec[:, keep].T @ U) ** 2 / ev[keep])) if df else 0.
    return stat, df, float(chi2.sf(stat, df)) if df else 1.

def self_check():
    y = np.array([0, 0, 1, 1])
    assert ks(y, np.arange(4)) == 1 and signed_ks(y, np.arange(4)) == 1
    assert ks(y, -np.arange(4)) == 1 and signed_ks(y, -np.arange(4)) == 0
    assert ks(y, np.ones(4)) == 0
    assert np.array_equal(qbins([np.nan, 0, 1], [0, 1], 2), [2, 0, 1])
    assert np.allclose(bh([.01, .04, .03]), [.03, .04, .04])
    a = conditional_test(np.tile([0, 1], 100), np.tile([0, 1], 100), np.zeros(200))
    assert a[2] < 1e-10
