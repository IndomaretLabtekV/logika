"""Read-only inputs; CPU diagnostics only. Outputs isolated under this directory."""
import os
os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
import sys, json, hashlib
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "mpl_cache"))
sys.path.insert(0, str(HERE.parent / "deep"))
from deep_common import table, features, folds, qbins, bh, conditional_test, ks, signed_ks, CATS, IDS, TREES
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def csv(rows, name):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(OUT / (name + ".csv"), index=False)
    print(name, df.shape, "\n", df.iloc[:20, :12].to_string(index=False), flush=True)
    return df

def js(obj, name):
    s = json.dumps(obj, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    (OUT / (name + ".json")).write_text(s + "\n")
    print(name, s, flush=True)

def figsave(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / (name + ".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("FIGURE", name, flush=True)

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for b in iter(lambda: fp.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def data():
    f = features()
    assert f["customer_id"].n_unique() == f.height
    return f, f.filter(pl.col("is_train") == 1), f.filter(pl.col("is_train") == 0)

def predictions():
    f, tr, te = data()
    y = tr["target"].to_numpy()
    names = TREES + ["tabpfn_rel", "tabm"]
    raw = {n: np.load(ROOT / "results/v5" / ("oof_" + n + ".npy")) for n in names}
    assert all(a.shape == y.shape and np.isfinite(a).all() for a in raw.values())
    R = np.column_stack([rankdata(raw[n]) / len(y) for n in names])
    w = np.array([4, 0, 0, 1, 9, 3, 0, 13]) / 30
    s = R @ w
    assert abs(ks(y, raw["lgb"]) - .5241) < .00005, "Cached train order inconsistent with notebook"
    assert abs(ks(y, s) - .5274) < .00005
    rt = np.column_stack([rankdata(np.load(ROOT / "results/v5" / ("test_" + n + ".npy"))) / te.height for n in names])
    t = rt @ w
    t = (t - t.min()) / np.ptp(t)
    sub = pl.read_csv(ROOT / "results/v5/submission.csv")
    check = te.select("application_id").with_columns(expected=pl.Series(t)).join(sub, on="application_id", validate="1:1")
    assert check.height == te.height and np.allclose(check["expected"], check["default_probability"], atol=1e-12)
    # Legacy NPY has no IDs: agreement with logs and submission supports, but cannot prove, train row binding.
    return f, tr, te, y, raw, R, s

def items(f):
    e = table("ecommerce_transactions")
    cut = f.select("customer_id", cutoff="decision_datetime")
    e = e.join(cut, on="customer_id", validate="m:1").filter(pl.col("transaction_datetime") <= pl.col("cutoff"))
    assert e.filter((pl.col("quantity") <= 0) | (pl.col("item_total") <= 0)).height == 0
    return e.with_columns(
        lu=(pl.col("item_total") / pl.col("quantity")).log(),
        age=(pl.col("cutoff") - pl.col("transaction_datetime")).dt.total_seconds() / 86400,
        event_month=pl.col("transaction_datetime").dt.truncate("1mo"),
        query_month=pl.col("cutoff").dt.truncate("1mo"))

def self_check():
    assert ks(np.array([0, 0, 1, 1]), np.arange(4)) == 1
    assert signed_ks(np.array([0, 0, 1, 1]), -np.arange(4)) == 0
    assert np.allclose(bh([.01, .04, .03]), [.03, .04, .04])
