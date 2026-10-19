"""Public: Causilo-only is 0.010 below trees-only (z -3.3), Kumo 0.003 below; CV says Causilo ~ trees. Is there a
natural subset of rows (decision month, segment, rate tier, region, purpose, employment, housing) where, on clean OOF,
Causilo / Kumo lose ~0.01 KS to trees? If a non-random public subset were such a group, it would explain the gap.
Prints KS(model) - KS(trees) per subset with a bootstrap sd. Saves eda/out/132_icl_gap_by_subset.png.
Usage: cd eda && ../.venv/bin/python 132_icl_gap_by_subset.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata

import valid2 as v
from common import OUT, ROOT, features

R = ROOT / "results"
f = features().filter(pl.col("is_train") == 1).sort("application_id")
ids, y = f["application_id"], f["target"].to_numpy()


def raw(p):
    d = pl.read_parquet(p).sort("application_id")
    assert (d["application_id"] == ids).all()
    return d[[c for c in d.columns if c not in ("application_id", "target")][0]].to_numpy()


rk = lambda x: rankdata(x) / len(x)  # noqa: E731
trees = rk(np.mean([rk(raw(R / f"v22/v22/gbdt/oof_{m}.parquet")) for m in ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]], axis=0))
M = {"causilo": rk((raw(R / "v22/v22/fm/oof_causilo.parquet") + 2 * raw(R / "v37/oof_causilo.parquet")) / 3),
     "kumo": rk(raw(R / "v28m/oof_kumo_large_c56000.parquet")),
     "lr_q": rk(raw(R / "screen/lin_5f/oof_lr_q.parquet"))}
groups = {
    "dec_month": f["dec_month"].to_numpy(),
    "has_history": f["has_history"].to_numpy(),
    "rate": np.round(f["annual_interest_rate"].to_numpy() * 100).astype(int),
    "region": f["region"].to_numpy(),
    "purpose": f["loan_purpose"].to_numpy(),
    "employment": f["employment_type"].to_numpy(),
    "housing": f["housing_status"].to_numpy(),
}
rng = np.random.default_rng(0)
worst = []
for gname, g in groups.items():
    print(f"\n{gname}:")
    for val in sorted(set(g.tolist())):
        m = g == val
        if m.sum() < 1500:
            continue
        idx = np.where(m)[0]
        line = f"  {str(val):16s} n {m.sum():6d} bad {y[m].mean():.3f} trees {v.ks(y[m], trees[m]):.4f}"
        for name, s in M.items():
            d = v.ks(y[m], s[m]) - v.ks(y[m], trees[m])
            b = [(lambda i: v.ks(y[i], s[i]) - v.ks(y[i], trees[i]))(rng.choice(idx, len(idx))) for _ in range(100)]
            line += f" | {name} {d:+.4f}±{np.std(b):.4f}"
            worst.append((name, gname, val, d, np.std(b), m.sum()))
        print(line)
for name in M:
    w = sorted([x for x in worst if x[0] == name], key=lambda x: x[3])[:3]
    print(f"\nworst subsets for {name}:", [(g, v_, round(d, 4), n) for _, g, v_, d, _, n in w])
fig, ax = plt.subplots(figsize=(6, 4))
for name in M:
    ax.scatter([x[5] for x in worst if x[0] == name], [x[3] for x in worst if x[0] == name], s=10, label=name)
ax.axhline(-0.010, color="r", ls="--", lw=.6, label="public Causilo gap")
ax.axhline(0, color="k", lw=.5)
ax.set(xlabel="subset size", ylabel="KS(model) - KS(trees) on OOF", title="ICL gap to trees by natural subset")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "132_icl_gap_by_subset.png", dpi=110)
