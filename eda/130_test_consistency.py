"""Public says Causilo-only 0.55333, Kumo-only 0.56056, trees-only 0.56337 while CV ranks them the other way. Are the
ICL TEST predictions damaged (as in the Kumo class-order bug, where each chunk looked fine)? Compare, for every model,
the Spearman with the tree family on OOF vs on test, per test chunk (Causilo predicts test in chunks of 10,000 rows
in valid2.icl_fit), and the score distribution of each chunk. Saves eda/out/130_test_consistency.png.
Usage: cd eda && ../.venv/bin/python 130_test_consistency.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr

from common import OUT, ROOT

R = ROOT / "results"


def col(p, key):
    d = pl.read_parquet(p).sort("application_id")
    return d["application_id"], d[[c for c in d.columns if c not in ("application_id", "target")][0]].to_numpy()


rk = lambda x: rankdata(x) / len(x)  # noqa: E731
G = ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]
oof_ids, _ = col(R / "v22/v22/gbdt/oof_lgb.parquet", 0)
tst_ids, _ = col(R / "v22/v22/gbdt/test_lgb.parquet", 0)
t_oof = rk(np.mean([rk(col(R / f"v22/v22/gbdt/oof_{m}.parquet", 0)[1]) for m in G], axis=0))
t_tst = rk(np.mean([rk(col(R / f"v22/v22/gbdt/test_{m}.parquet", 0)[1]) for m in G], axis=0))
models = {"causilo_v22": ("v22/v22/fm/oof_causilo.parquet", "v22/v22/fm/test_causilo.parquet"),
          "causilo_v37": ("v37/oof_causilo.parquet", "v37/test_causilo.parquet"),
          "kumo_v28m": ("v28m/oof_kumo_large_c56000.parquet", "v28m/test_kumo_large_c56000.parquet"),
          "tabm_v22": ("v22/v22/nn/oof_tabm.parquet", "v22/v22/nn/test_tabm.parquet"),
          "lgb_v22": ("v22/v22/gbdt/oof_lgb.parquet", "v22/v22/gbdt/test_lgb.parquet")}
# test rows are predicted in file order of Xt (sorted by application_id in valid2.frames): chunk k = rows k*10000..
chunk = np.arange(len(tst_ids)) // 10000
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, (o, t) in models.items():
    io, xo = col(R / o, 0)
    it, xt = col(R / t, 0)
    assert (io == oof_ids).all() and (it == tst_ids).all()
    so, st = spearmanr(xo, t_oof).statistic, spearmanr(xt, t_tst).statistic
    per = [spearmanr(xt[chunk == k], t_tst[chunk == k]).statistic for k in range(3)]
    med = [np.median(xt[chunk == k]) for k in range(3)]
    print(f"{name:12s} spearman vs trees OOF {so:.4f} test {st:.4f}; per test chunk {[round(p, 4) for p in per]}; "
          f"chunk medians {[round(m, 4) for m in med]}; OOF mean {xo.mean():.4f} test mean {xt.mean():.4f}")
    axes[0].hist(xo, 100, histtype="step", density=True, label=f"{name} OOF")
    axes[1].hist(xt, 100, histtype="step", density=True, label=f"{name} test")
for a in axes:
    a.set_xlim(0, 0.6)
    a.legend(fontsize=6)
fig.tight_layout()
fig.savefig(OUT / "130_test_consistency.png", dpi=110)
