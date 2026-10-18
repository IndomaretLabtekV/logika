"""Learning curve on outer fold 0 (validation_v2 protocol): LightGBM and shallow CatBoost trained on 40/60/80/100% of
the training fold (same inner stop split rule inside the subsample), scored on the untouched fold-0 rows.
A steep curve means more training rows (full-train refit for the test predictions, extra labelled rows) still pay;
a flat one means the models sit at the information ceiling of the features. Saves eda/out/102_learning_curve.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import valid2 as v
from common import OUT, features

X, Xt, y, ids, tids = v.frames(features())
tr, va = v.outer(y, 42)[0]
fits = {"lgb": v.lgb_fit(v.LGB), "cat_s": v.cat_fit(v.CAT_S)}
rows = []
for frac in (0.4, 0.6, 0.8, 1.0):
    for rep in (0, 1):
        rng = np.random.default_rng(100 * rep + int(frac * 10))
        sub = np.sort(rng.choice(tr, int(len(tr) * frac), replace=False)) if frac < 1 else tr
        fi, st = v.inner(y[sub], 42, rep)
        for name, fit in fits.items():
            (pv,), n = fit(X, y, sub, fi, st, [X.iloc[va]], 42 + rep)
            rows.append(dict(model=name, frac=frac, rep=rep, n_train=len(sub), ks=v.ks(y[va], pv), budget=n))
            print(rows[-1], flush=True)
d = pd.DataFrame(rows)
d.to_csv(OUT / "102_learning_curve.csv", index=False)
g = d.groupby(["model", "frac"]).ks.mean().unstack(0)
print(g)
for m in g:
    slope = np.polyfit(np.log2(g.index * len(tr)), g[m], 1)[0]
    print(f"{m}: KS gain per doubling of training rows {slope:+.4f}; +25% rows (full-train refit) ~ {slope * np.log2(1.25):+.4f}")
fig, ax = plt.subplots(figsize=(6, 4))
for m in g:
    ax.plot(g.index * len(tr), g[m], "o-", label=m)
ax.set(xscale="log", xlabel="training rows", ylabel="KS on outer fold 0", title="Learning curve (fold 0, 2 reps)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "102_learning_curve.png", dpi=110)
