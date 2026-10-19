"""What do the ICL models (Kumo ctx full, Causilo) know that the clean GBDT ensemble does not?
On clean validation_v2 OOF (v22 trees = rank mean of 6 GBDT, v22 Causilo, v28m Kumo):
1. correlation matrix of the three scores (all rows, within each class);
2. ICL residual r = z(icl) - b z(trees) (z = normal scores of ranks): its own KS, and the LR chi2 of y ~ z(trees) + r;
3. which features drive r (Spearman with every feature) -> where trees under-use information;
4. per segment (has_history, e-commerce item count tertile) KS of trees / icl / v31 blend.
Usage: cd eda && ../.venv/bin/python 113_icl_vs_trees.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression

import valid2 as v
from common import OUT, ROOT, features

R = ROOT / "results"
X, Xt, y, ids, tids = v.frames(features())


def load(path, col):
    d = pl.read_parquet(path).to_pandas().set_index("application_id")[col]
    return d.loc[ids].to_numpy()


z = lambda s: norm.ppf((rankdata(s) - 0.5) / len(s))  # noqa: E731
trees = np.mean([rankdata(load(R / f"v22/v22/gbdt/oof_{m}.parquet", m)) for m in
                 ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]], axis=0)
S = pd.DataFrame({"trees": z(trees), "causilo": z(load(R / "v22/v22/fm/oof_causilo.parquet", "causilo")),
                  "kumo": z(load(R / "v28m/oof_kumo_large_c56000.parquet", "kumo_large_c56000"))})
S["v31"] = z(S[["trees", "causilo", "kumo"]].apply(rankdata).mean(axis=1))
print("KS:", {c: round(v.ks(y, S[c].to_numpy()), 5) for c in S})
print("spearman all\n", S.corr("spearman").round(4))
for c in (0, 1):
    print(f"spearman within y={c}\n", S[y == c].corr("spearman").round(4))


def lr_chi2(cols):
    """In-sample LR deviance of y ~ cols vs y ~ trees (df = len(cols) - 1)."""
    def dev(F):
        m = LogisticRegression(C=1e6, max_iter=1000).fit(F, y)
        p = m.predict_proba(F)[:, 1]
        return -2 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return dev(S[["trees"]]) - dev(S[cols])


num = [c for c in X.columns if c not in v.CAT_COLS]
rows = []
for icl in ("kumo", "causilo"):
    b = np.polyfit(S["trees"], S[icl], 1)
    r = S[icl] - np.polyval(b, S["trees"])
    S[f"r_{icl}"] = r
    print(f"\n{icl}: slope on trees {b[0]:.4f}, resid sd {r.std():.4f}, KS(resid) {v.ks(y, r.to_numpy()):.5f}, "
          f"LR chi2 y~trees+resid vs y~trees {lr_chi2(['trees', f'r_{icl}']):.1f} (df 1)")
    for c in num:
        x = X[c].to_numpy()
        m = ~np.isnan(x)
        if m.sum() > 1000:
            rows.append(dict(icl=icl, feature=c, rho=spearmanr(x[m], r[m])[0], cover=m.mean()))
print(f"\nLR chi2 y~trees+r_kumo+r_causilo: {lr_chi2(['trees', 'r_kumo', 'r_causilo']):.1f} (df 2)")
F = pd.DataFrame(rows)
F["abs"] = F["rho"].abs()
for icl in ("kumo", "causilo"):
    print(f"\ntop features driving {icl} residual (Spearman):")
    print(F[F.icl == icl].nlargest(20, "abs")[["feature", "rho", "cover"]].to_string(index=False))
F.to_csv(OUT / "113_resid_feature_rho.csv", index=False)

# segments
seg = {"history": X["has_history"].to_numpy() == 1, "thin": X["has_history"].to_numpy() == 0}
n_items = X["pp_qty_item"].notna().to_numpy()
ec_n = X["ec_n_orders"].to_numpy()
q = np.nanquantile(ec_n, [1 / 3, 2 / 3])
seg |= {"ec_low": ec_n <= q[0], "ec_mid": (ec_n > q[0]) & (ec_n <= q[1]), "ec_high": ec_n > q[1]}
print("\nsegment KS (n, bad rate, trees, causilo, kumo, v31):")
for k, m in seg.items():
    print(f"  {k:8s} {m.sum():6d} {y[m].mean():.3f} " + " ".join(f"{v.ks(y[m], S[c].to_numpy()[m]):.4f}"
                                                            for c in ("trees", "causilo", "kumo", "v31")))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, icl in zip(ax, ("kumo", "causilo")):
    r = S[f"r_{icl}"].to_numpy()
    bins = np.quantile(r, np.linspace(0, 1, 11))
    k = np.clip(np.searchsorted(bins, r, side="right") - 1, 0, 9)
    a.plot(range(10), [y[k == i].mean() for i in range(10)], "o-")
    a.set_title(f"bad rate by decile of {icl} residual (vs trees)")
    a.set_xlabel("decile")
fig.tight_layout()
fig.savefig(OUT / "113_icl_resid.png", dpi=110)
