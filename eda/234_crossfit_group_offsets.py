"""Cross-fitted group logit offsets on top of a fixed OOF score (diagnosis of 233's overdispersion).
For each outer fold k: isotonic calibration and per-group ridge intercepts (Newton, penalty lam) are fit
on the other 4 folds' OOF predictions + labels, then applied to fold k. Groups are decision-time
segments (month, week, employment×month, housing×month, ...). Pooled KS/AUC vs the unadjusted score,
paired stratified bootstrap. Figure: eda/out/234_group_offsets.png
Usage: cd eda && ../.venv/bin/python 234_crossfit_group_offsets.py [candidate_dir]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

import valid2 as v
from common import ROOT

cand = sys.argv[1] if len(sys.argv) > 1 else "K__K64_2seed"
apps = pl.read_csv(ROOT / "data/train_applications.csv", try_parse_dates=True).sort("application_id")
prof = pl.read_csv(ROOT / "data/customer_profiles.csv")
oof = pl.read_parquet(ROOT / f"results/v60/candidates/{cand}/oof.parquet")
score_col = [c for c in oof.columns if c not in ("application_id", "target")][0]
d = apps.join(prof, on="customer_id", how="left").join(oof.select("application_id", score_col), on="application_id").sort("application_id")
assert d.height == 70000
y = d["target"].to_numpy().astype(int)
s = d[score_col].to_numpy()
d = d.with_columns(pl.col("decision_datetime").dt.strftime("%m").alias("month"),
                   pl.col("decision_datetime").dt.strftime("%W").alias("week"))
GROUPS = {"month": ["month"], "week": ["week"], "employment×month": ["employment_type", "month"],
          "housing×month": ["housing_status", "month"], "purpose×month": ["loan_purpose", "month"],
          "region×month": ["region", "month"]}
COMBOS = {"month+emp×m+hous×m": ["month", "employment×month", "housing×month"],
          "week+emp×m+hous×m": ["week", "employment×month", "housing×month"]}
LAMS = [5.0, 20.0, 80.0]


def codes(keys):
    return d.select(pl.concat_str([pl.col(k).cast(pl.String) for k in keys], separator="|")).to_series().cast(pl.Categorical).to_physical().to_numpy()


def fit_offsets(o, yy, g, lam, iters=8):
    b = np.zeros(g.max() + 1)
    for _ in range(iters):
        p = expit(o + b[g])
        grad = np.bincount(g, p - yy, len(b)) + 2 * lam * b
        hess = np.bincount(g, p * (1 - p), len(b)) + 2 * lam
        b -= grad / hess
    return b


def adjust(group_names, lam):
    out = np.empty(len(y))
    for tr, va in v.outer(y, 42):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(s[tr], y[tr])
        o_tr, o_va = logit(iso.predict(s[tr])), logit(iso.predict(s[va]))
        for name in group_names:  # backfitting: each grouping fitted on the current offset
            g = codes(GROUPS[name])
            b = fit_offsets(o_tr, y[tr], g[tr], lam)
            o_tr, o_va = o_tr + b[g[tr]], o_va + b[g[va]]
        out[va] = o_va
    return out


base_ks = v.ks(y, s)
rng = np.random.default_rng(234)
pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
boots = [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(300)]
print(f"{cand}: base pooled KS {base_ks:.5f} AUC {roc_auc_score(y, s):.5f}")
results = {}
for name, members in [*[(k, [k]) for k in GROUPS], *COMBOS.items()]:
    for lam in LAMS:
        a = adjust(members, lam)
        dk = np.array([v.ks(y[r], a[r]) - v.ks(y[r], s[r]) for r in boots])
        results[f"{name}|lam{lam}"] = {"ks": v.ks(y, a), "auc": float(roc_auc_score(y, a)), "delta": v.ks(y, a) - base_ks,
                                       "p_gt0": float((dk > 0).mean()),
                                       "folds_won": int(sum(v.ks(y[va], a[va]) > v.ks(y[va], s[va]) for _, va in v.outer(y, 42)))}
        r = results[f"{name}|lam{lam}"]
        print(f"{name:22s} lam={lam:5.1f} KS {r['ks']:.5f} Δ {r['delta']:+.5f} P>0 {r['p_gt0']:.2f} folds {r['folds_won']}/5 AUC {r['auc']:.5f}", flush=True)
(ROOT / "eda/out/234_group_offsets.json").write_text(json.dumps({"candidate": cand, "base_ks": base_ks, "results": results}, indent=2))
fig, ax = plt.subplots(figsize=(8, 6))
keys = list(results)
ax.barh(keys, [results[k]["delta"] for k in keys]); ax.axvline(0, color="k", lw=.8)
ax.set_xlabel("Δ pooled KS vs unadjusted (cross-fitted)"); ax.set_title(f"234 group offsets on {cand}")
fig.tight_layout(); fig.savefig(ROOT / "eda/out/234_group_offsets.png", dpi=120)
