"""Do the model families win in different segments? Thin-file (no previous/repayment/account history, 30%) vs history.
Clean OOF (validation_v2, seed 42). 1) per-segment KS of each family; 2) blend weights over 4 family scores
(trees mean rank, causilo, kumo, tabm) on a 0.1 simplex grid, chosen on 4 outer folds and applied to the 5th,
either one global weight vector or one per segment. Pooled cross-fitted KS + paired bootstrap vs the global choice."""
import itertools
import numpy as np
import polars as pl
import blend2 as b
import valid2 as v

oof, _ = b.load(["../results/v22/v22", "../results/v26_local"])
seg = pl.read_parquet("../data_pq/features.parquet", columns=["application_id", "has_history"])
oof = oof.join(seg, on="application_id", validate="1:1").sort("application_id")
y, h = oof["target"].to_numpy(), oof["has_history"].to_numpy().astype(int)
names = [c for c in oof.columns if c not in ("application_id", "target", "has_history")]
R = b.ranks(oof, names)
col = {n: R[:, j] for j, n in enumerate(names)}
trees = [n for n in names if n.split("_")[0] in ("lgb", "xgb", "cat")]
F = np.column_stack([np.mean([col[n] for n in trees], 0), col["causilo"], col["kumo_large_c20000_b2"], col["tabm"]])
fam = ["trees", "causilo", "kumo", "tabm"]
for s in (0, 1):
    m = h == s
    print(f"has_history={s} n={m.sum()} bad={y[m].mean():.3f} " + " ".join(f"{f}:{v.ks(y[m], F[m, j]):.4f}" for j, f in enumerate(fam)))
grid = [np.array(w) / 10 for w in itertools.product(range(11), repeat=4) if sum(w) == 10]
folds = v.outer(y, 42)


def best(idx):
    return max(grid, key=lambda w: v.ks(y[idx], F[idx] @ w))


glob, perseg = np.zeros(len(y)), np.zeros(len(y))
for tr, va in folds:
    wg = best(tr)
    glob[va] = F[va] @ wg
    for s in (0, 1):
        ws = best(tr[h[tr] == s])
        vs = va[h[va] == s]
        perseg[vs] = F[vs] @ ws
        print(f"fold weights seg{s}: {dict(zip(fam, ws.round(1)))}  global {dict(zip(fam, wg.round(1)))}")
m, sd, p = b.boot(y, perseg, glob)
print(f"cross-fit KS global {v.ks(y, glob):.5f}  per-segment {v.ks(y, perseg):.5f}  diff {m:+.5f} sd {sd:.5f} P>0 {p:.2f}")
print("folds won", int((b.fold_ks(y, perseg, folds) > b.fold_ks(y, glob, folds)).sum()), "/5")
