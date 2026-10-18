"""Why did the ICL-only blend lose 0.007 on public vs the trees+ICL blend when clean CV predicted +0.0002 (4 sd)?
Hypothesis: training rows have closer neighbours among other training rows than test rows do (generator structure
within train), which in-context learners (attention ~ soft nearest neighbours) exploit in CV but not on test.
Fold 0: nearest-neighbour distances from (a) fold-0 validation rows and (b) a same-size sample of test rows to the
same 56k training-fold rows, in rank-gaussian space of the 104 fm_select features. Also KS of a 50-NN label average
on the validation rows. Saves eda/out/110_neighbor_gap.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer
import valid2 as v
from common import OUT, features

X, Xt, y, ids, tids = v.frames(features())
tr, va = v.outer(y, 42)[0]
cols = [c for c in v.fm_select(X, y, tr, 100) if c not in v.CAT_COLS]
qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0).fit(X.iloc[tr][cols])
med = np.nanmedian(qt.transform(X.iloc[tr][cols]), axis=0)


def emb(d):
    z = qt.transform(d[cols])
    miss = np.isnan(z)
    z[miss] = np.take(med, np.where(miss)[1])
    return np.hstack([z, miss.astype(float)]).astype(np.float32)


E_tr = emb(X.iloc[tr])
rng = np.random.default_rng(0)
E_va, E_te = emb(X.iloc[va]), emb(Xt.iloc[rng.choice(len(Xt), len(va), replace=False)])
nn = NearestNeighbors(n_neighbors=50, n_jobs=6).fit(E_tr)
d_va, i_va = nn.kneighbors(E_va)
d_te, _ = nn.kneighbors(E_te)
for k in (1, 10, 50):
    a, b = d_va[:, :k].mean(1), d_te[:, :k].mean(1)
    print(f"mean dist to {k}-NN: validation {a.mean():.4f} (median {np.median(a):.4f})  test {b.mean():.4f} (median {np.median(b):.4f})  "
          f"ratio {b.mean() / a.mean():.4f}")
print("share of rows whose 1-NN distance < 5th pct of validation 1-NN:",
      f"val {np.mean(d_va[:, 0] < np.quantile(d_va[:, 0], .05)):.3f} test {np.mean(d_te[:, 0] < np.quantile(d_va[:, 0], .05)):.3f}")
knn_score = y[tr][i_va].mean(1)
print(f"50-NN label average KS on validation: {v.ks(y[va], knn_score):.4f}")
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(d_va[:, 0], 80, alpha=.5, density=True, label="validation -> train fold")
ax.hist(d_te[:, 0], 80, alpha=.5, density=True, label="test -> train fold")
ax.set(xlabel="1-NN distance (rank-gaussian space)", title="Neighbour gap: validation vs test")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "110_neighbor_gap.png", dpi=110)
