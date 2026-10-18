"""Which validation scheme fits this competition, and what does it say about training size and drift?
A) StratifiedKFold 5 vs 10 (seed 42), LightGBM v8 params, inner 1/7 stopping + refit (validation_v2): pooled OOF KS.
   The gap measures the value of +12.5% training rows per model (10-fold models see 90% instead of 80%).
B) Forward-in-time: train on decision months <= m, score month m+1 (m = 6..11) vs a random split of the same size.
   A drop beyond noise would mean drift that the random split hides.
C) Adversarial validation: LightGBM separating train from test rows (5-fold AUC). ~0.5 = interleaved random split.
CPU only (LightGBM). Prints numbers, saves eda/out/104_cv_schemes.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import valid2 as v
from common import OUT, features

X, Xt, y, ids, tids = v.frames(features())
fit = v.lgb_fit(dict(v.LGB, n_jobs=12))
month = features().filter(features()["is_train"] == 1).sort("application_id")["decision_datetime"].dt.month().to_numpy()

print("== A) 5 vs 10 folds (seed 42) ==")
res = {}
for k in (5, 10):
    oof = np.zeros(len(y))
    fks = []
    for f, (tr, va) in enumerate(StratifiedKFold(k, shuffle=True, random_state=42).split(X, y)):
        fi, st = v.inner(y[tr], 42, f)
        (pv,), n = fit(X, y, tr, fi, st, [X.iloc[va]], 42)
        oof[va] = pv
        fks.append(v.ks(y[va], pv))
    res[k] = oof
    print(f"{k:2d}-fold: pooled OOF KS {v.ks(y, oof):.5f}  mean fold KS {np.mean(fks):.5f} sd {np.std(fks):.4f}")
rng = np.random.default_rng(0)
d = [v.ks(y[i], res[10][i]) - v.ks(y[i], res[5][i]) for i in (rng.choice(len(y), len(y)) for _ in range(300))]
print(f"10-fold minus 5-fold: {np.mean(d):+.5f} sd {np.std(d):.5f} P>0 {np.mean(np.array(d) > 0):.2f}")

print("\n== B) forward-in-time vs random of the same size ==")
rows = []
for m in range(6, 12):
    tr, va = np.flatnonzero(month <= m), np.flatnonzero(month == m + 1)
    fi, st = v.inner(y[tr], 42, m)
    (pv,), _ = fit(X, y, tr, fi, st, [X.iloc[va]], 42)
    ks_fwd = v.ks(y[va], pv)
    perm = rng.permutation(np.r_[tr, va])
    tr2, va2 = np.sort(perm[:len(tr)]), np.sort(perm[len(tr):])
    fi2, st2 = v.inner(y[tr2], 42, m)
    (pv2,), _ = fit(X, y, tr2, fi2, st2, [X.iloc[va2]], 42)
    rows.append((m + 1, len(tr), ks_fwd, v.ks(y[va2], pv2)))
    print(f"score month {m + 1:2d}: train {len(tr):5d}  forward KS {ks_fwd:.4f}  random KS {rows[-1][3]:.4f}")
r = np.array(rows)
print(f"mean forward - random: {np.mean(r[:, 2] - r[:, 3]):+.4f} (sd of one-month KS ~0.02)")

print("\n== C) adversarial validation train vs test ==")
A = np.r_[np.zeros(len(X)), np.ones(len(Xt))]
XA = __import__("pandas").concat([X, Xt], ignore_index=True)
pa = np.zeros(len(A))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(XA, A):
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=12, feature_fraction=0.5),
                  lgb.Dataset(XA.iloc[tr], A[tr]), 300)
    pa[va] = m.predict(XA.iloc[va])
print(f"adversarial AUC {roc_auc_score(A, pa):.4f}")
imp = sorted(zip(m.feature_importance("gain"), XA.columns), reverse=True)[:8]
print("top adversarial features:", [(c, round(g)) for g, c in imp])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(r[:, 0], r[:, 2], "o-", label="forward")
ax[0].plot(r[:, 0], r[:, 3], "s--", label="random")
ax[0].set(xlabel="scored month", ylabel="KS", title="Forward-in-time vs random split")
ax[0].legend()
ax[1].hist([pa[A == 0], pa[A == 1]], bins=50, label=["train", "test"], density=True)
ax[1].set(title="Adversarial P(test)")
ax[1].legend()
fig.tight_layout()
fig.savefig(OUT / "104_cv_schemes.png", dpi=110)
