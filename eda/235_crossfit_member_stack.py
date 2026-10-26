"""Cross-fitted stacking of all complete strict Kumo members + trees6 + Causilo (seed-42 folds).
Per outer fold k: L2 logistic regression on per-cohort logit-ranks of members, fit on the other 4
folds' OOF, applied to fold k. Caveat (validation_v2): base models of other folds saw fold-k labels,
so this is not end-to-end nested. Compared with K64 2-seed (equal weights) and REF.
Figure: eda/out/235_member_stack.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import valid2 as v
from common import ROOT

R = ROOT / "results"
MEMBERS = {"cap500_8k_s2026": "v60/strict_tfm::pool_cap500_s2026", "nested_small_s2026": "v60/strict_tfm::nested_small_s2026",
           "cap500_16k_s2026": "v62/strict_variant::cap500_ctx16k_s2026", "main_small_s2026": "v62/strict_variant::main_small_s2026",
           "main_small_s2027": "v62/strict_variant::main_small_s2027", "cap300_28k_s2026": "v64/strict_variant::cap300_ctx28k_s2026",
           "cap300_28k_s2027": "v64/strict_variant::cap300_ctx28k_s2027", "main150_s2026": "v65/strict_variant::main150_s2026",
           "main150_s2027": "v65/strict_variant::main150_s2027"}
train = pl.read_csv(ROOT / "data/train_applications.csv").sort("application_id")
ids, y = train["application_id"].to_numpy(), train["target"].to_numpy().astype(int)
folds = v.outer(y, 42)


def rank(x):
    return rankdata(x) / len(x)


cols = {}
for name, ref in MEMBERS.items():
    root, member = ref.split("::")
    x = np.full(len(y), np.nan)
    for f, (_, va) in enumerate(folds):
        d = pl.read_parquet(R / root / f"outer_f{f}/{member}.parquet")
        assert d["application_id"].to_list()[:len(va)] == ids[va].tolist()
        x[va] = rank(d["prediction"].to_numpy()[:len(va)])
    cols[name] = x


def load(path):
    d = pl.read_parquet(path).sort("application_id")
    assert d["application_id"].to_list() == ids.tolist()
    return d[[c for c in d.columns if c not in ("application_id", "target")][0]].to_numpy()


trees = np.mean([rank(load(R / f"v22/v22/gbdt/oof_{m}.parquet")) for m in ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]], axis=0)
cau = sum(w * load(p) for w, p in [(1, R / "v22/v22/fm/oof_causilo.parquet"), (2, R / "v37/oof_causilo.parquet"), (2, R / "v41/causilo_s12/oof_causilo.parquet")]) / 5
for name, x in [("trees6", trees), ("causilo", cau)]:
    z = np.full(len(y), np.nan)
    for _, va in folds:
        z[va] = rank(x[va])
    cols[name] = z
names = list(cols)
Z = logit(np.clip(np.column_stack([cols[n] for n in names]), 1e-4, 1 - 1e-4))
k64 = pl.read_parquet(R / "v60/candidates/K__K64_2seed/oof.parquet").sort("application_id")["K"].to_numpy()
ref_oof = np.full(len(y), np.nan)
for _, va in folds:
    ref_oof[va] = (rank(cau[va]) + rank(trees[va]) + rank(pl.read_parquet(R / "v28m/oof_kumo_large_c56000.parquet").sort("application_id")[:, -1].to_numpy()[va])) / 3
rng = np.random.default_rng(235)
pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
boots = [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(300)]
report = {}
for subset_name, subset in [("kumo_only", [n for n in names if n not in ("trees6", "causilo")]), ("all", names)]:
    idx = [names.index(n) for n in subset]
    for C in [0.01, 0.1, 1.0]:
        out, coefs = np.empty(len(y)), []
        for tr, va in folds:
            m = LogisticRegression(C=C, max_iter=2000).fit(Z[tr][:, idx], y[tr])
            out[va] = m.decision_function(Z[va][:, idx])
            coefs.append(m.coef_[0].round(3).tolist())
        res = {"ks": v.ks(y, out), "auc": float(roc_auc_score(y, out))}
        for bname, b in [("K64_2seed", k64), ("REF", ref_oof)]:
            dk = np.array([v.ks(y[r], out[r]) - v.ks(y[r], b[r]) for r in boots])
            res[f"delta_{bname}"] = v.ks(y, out) - v.ks(y, b)
            res[f"p_gt0_{bname}"] = float((dk > 0).mean())
        res["folds_won_vs_K64"] = int(sum(v.ks(y[va], out[va]) > v.ks(y[va], k64[va]) for _, va in folds))
        res["mean_coef"] = dict(zip(subset, np.mean(coefs, axis=0).round(3).tolist()))
        report[f"{subset_name}|C{C}"] = res
        print(subset_name, C, json.dumps({k: (round(x, 5) if isinstance(x, float) else x) for k, x in res.items() if k != "mean_coef"}), flush=True)
print("K64 2-seed", round(v.ks(y, k64), 5), "REF", round(v.ks(y, ref_oof), 5))
best = max(report, key=lambda k: report[k]["ks"])
print("best", best, json.dumps(report[best]["mean_coef"]))
(ROOT / "eda/out/235_member_stack.json").write_text(json.dumps(report, indent=2))
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(list(report), [report[k]["delta_K64_2seed"] for k in report]); ax.axvline(0, color="k", lw=.8)
ax.set_xlabel("Δ pooled KS vs K64 2-seed"); ax.set_title("235 cross-fit member stacking"); fig.tight_layout()
fig.savefig(ROOT / "eda/out/235_member_stack.png", dpi=120)
