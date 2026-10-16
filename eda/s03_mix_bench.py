"""Fixed-weight mix of v8-stack logit score with a candidate (z-scored logit). Weight chosen on one half, scored on
the other (20 x 2). Saved predictions only."""
import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s01_stack_bench import BASE, ROOT, fit_lr, ks, load_base, logit  # noqa: E402

df = load_base()
cands = {}
for arg in sys.argv[1:]:
    name, spec = arg.split("=", 1)
    path, col = spec.rsplit(":", 1)
    df = df.join(pl.read_parquet(ROOT / path).select("application_id", pl.col(col).alias(name)), on="application_id", validate="1:1")
    cands[name] = name
y = df["target"].to_numpy()
X = np.column_stack([logit(df[c].to_numpy()) for c in BASE])
X = (X - X.mean(0)) / X.std(0)
grid = np.array([0, .05, .1, .15, .2, .3, .4])
rng = np.random.default_rng(0)
for name in cands:
    z = logit(df[name].to_numpy()); z = (z - z.mean()) / z.std()
    d, full = [], np.zeros(len(grid))
    for r in range(20):
        half = np.zeros(len(y), bool)
        for c in (0, 1):
            idx = np.flatnonzero(y == c); half[rng.choice(idx, len(idx) // 2, replace=False)] = True
        for a, b in ((half, ~half), (~half, half)):
            w = fit_lr(X[a], y[a]); s = X @ w[1:]; s = s / s[a].std()
            ka = [ks(y[a], s[a] + g * z[a]) for g in grid]
            g = grid[int(np.argmax(ka))]
            d.append(ks(y[b], s[b] + g * z[b]) - ks(y[b], s[b]))
            full += [ks(y[b], s[b] + gg * z[b]) for gg in grid]
    d = np.array(d)
    print(f"{name:10s} chosen-weight delta {d.mean():+.5f} wins {(d > 0).sum()}/40 | mean held-out KS by weight",
          dict(zip(grid, np.round(full / 40, 5))))
