"""How much can one public-LB score move by chance? Clean cross-fitted OOF blends (validation_v2, seed-42 folds) are
scored on random 30% subsets (public = 9k of 30k test rows) to get the sampling distribution of
KS(candidate) - KS(v8 recipe). Also the trees-only vs ICL/NN-only pair, the clean analogue of probes S4/S5
(public: S5 - S4 = -0.0027 while biased OOF said +0.0040). Saves eda/out/107_public_noise.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import blend2 as b
import valid2 as v

oof, _ = b.load(["../results/v22/v22", "../results/screen/lin_5f", "../results/v28m"])
y = oof["target"].to_numpy()
names = [c for c in oof.columns if c not in ("application_id", "target")]
R = b.ranks(oof, names)
col = {n: R[:, j] for j, n in enumerate(names)}
trees = [n for n in names if n.split("_")[0] in ("lgb", "xgb", "cat")]
mean = lambda w: sum(col[n] * x for n, x in w.items()) / sum(w.values())  # noqa: E731
S = {
    "v8_recipe": mean(b.V8),
    "trees_causilo_kumo": mean({**{n: 1 / 3 / len(trees) for n in trees}, "causilo": 1 / 3, "kumo_large_c56000": 1 / 3}),
    "trees_only": mean({n: 1 for n in trees}),
    "icl_nn_only": mean({"tabm": 1, "causilo": 1, "kumo_large_c56000": 1}),
}
rng = np.random.default_rng(0)
n_pub = int(0.3 * len(y))
pairs = [("trees_causilo_kumo", "v8_recipe"), ("icl_nn_only", "trees_only")]
d = {p: [] for p in pairs}
for _ in range(2000):
    i = rng.choice(len(y), n_pub, replace=False)
    for a, c in pairs:
        d[(a, c)].append(v.ks(y[i], S[a][i]) - v.ks(y[i], S[c][i]))
fig, ax = plt.subplots(figsize=(6, 4))
for (a, c), x in d.items():
    x = np.array(x)
    full = v.ks(y, S[a]) - v.ks(y, S[c])
    print(f"{a} - {c}: full OOF {full:+.5f}; 30% subset mean {x.mean():+.5f} sd {x.std():.5f}; "
          f"P(subset diff < 0) {(x < 0).mean():.2f}; 5-95% [{np.quantile(x, .05):+.4f}, {np.quantile(x, .95):+.4f}]")
    ax.hist(x, 60, alpha=.5, label=f"{a} - {c}")
ax.axvline(-0.0027, color="k", ls="--", label="public S5 - S4")
ax.set(xlabel="KS difference on a random 30% subset", title="Public-LB sampling noise (clean OOF)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("out/107_public_noise.png", dpi=110)
