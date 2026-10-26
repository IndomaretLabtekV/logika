"""Compare a fresh fold-0 rebuild (results/v71/fresh_check) with the cached family outputs used by merge_R2DK.

Reports byte equality, max |diff|, Spearman and fold-0 KS for: re_* features, Kumo V5 raw, R recipe, K64 member
cap300_ctx28k_s2026. Usage: cd eda && ../.venv/bin/python 265_fresh_check.py
"""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import valid2 as v
from common import ROOT

F, C = ROOT / "results/v71/fresh_check", ROOT / "results/v71"
lab = pd.read_csv(ROOT / "data/train_applications.csv", usecols=["application_id", "target"]).set_index("application_id").target


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compare(name, new, old, key, col, val_mask=None):
    a, b = pd.read_parquet(new), pd.read_parquet(old)
    m = a.merge(b, on=key, suffixes=("_new", "_old"))
    x, y = m[f"{col}_new"].to_numpy(float), m[f"{col}_old"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    r = {"rows": len(m), "bytes_identical": sha(new) == sha(old), "max_abs_diff": float(np.abs(x[ok] - y[ok]).max()),
         "spearman": float(spearmanr(x[ok], y[ok]).statistic)}
    if val_mask is not None:
        vv = m[val_mask(m)]
        yy = lab.loc[vv.application_id].to_numpy()
        r["fold0_ks_new"], r["fold0_ks_old"] = v.ks(yy, vv[f"{col}_new"].to_numpy()), v.ks(yy, vv[f"{col}_old"].to_numpy())
    print(name, {k: (round(x, 6) if isinstance(x, float) else x) for k, x in r.items()}, flush=True)
    return r


rep = {"reverse_re_price_log_posterior": compare("reverse", F / "teammate_s42/f0/reverse.parquet", C / "teammate_s42/f0/reverse.parquet",
                                                  "application_id", "re_price_log_posterior"),
       "kumo_v5": compare("kumo", F / "teammate_s42/f0/kumo.parquet", C / "teammate_s42/f0/kumo.parquet", ["application_id", "part"], "kumo",
                          lambda m: m.part.eq("val")),
       "recipe_R": compare("R", F / "teammate_s42/f0/post.parquet", C / "teammate_s42/f0/post.parquet", ["application_id", "part"], "R",
                           lambda m: m.part.eq("val"))}
k_new, k_old = F / "strict_variant/outer_f0/cap300_ctx28k_s2026.parquet", ROOT / "results/v64/strict_variant/outer_f0/cap300_ctx28k_s2026.parquet"
if k_new.exists():
    rep["k64_cap300_s2026"] = compare("K64 member", k_new, k_old, "application_id", "prediction", lambda m: m.application_id.isin(lab.index))
(ROOT / "eda/out/265_fresh_check.json").write_text(json.dumps(rep, indent=1))
