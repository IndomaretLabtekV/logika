"""Audit the DSC0152 population-level (label-free) references against per-partition fitting, seed-42 outer folds.

Recomputes the two reference families that D/KD features use with outer-train customers only and compares them to
the population version: (1) category median unit price behind ec_price_rel / ec lrel, (2) gradenorm group mean/std.
Reports max |delta| and Spearman of the resulting features. Usage: cd eda && ../.venv/bin/python 263_dsc_reference_audit.py
"""
import importlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import valid2 as v
from common import ROOT

t251 = importlib.import_module("251_teammate_recipe_s42")
DSC = ROOT / "scripts/dsc0152"
GN = ["ec_price_rel", "dti_total", "bu_score_mean", "rh_rate1_smooth", "declared_monthly_income", "loan_to_income"]

lab = t251.labels()
folds = v.outer(lab.target.to_numpy(), 42)
apps = pd.concat([pd.read_csv(ROOT / "data/train_applications.csv", usecols=["application_id", "customer_id"]),
                  pd.read_csv(ROOT / "data/test_applications.csv", usecols=["application_id", "customer_id"])])
cust = apps.set_index("application_id").customer_id
ec = pd.read_csv(ROOT / "data/ecommerce_transactions.csv", usecols=["customer_id", "product_category", "quantity", "item_total"])
ec = ec[ec.quantity > 0]
unit = np.log(ec.item_total / ec.quantity)
glob = unit.groupby(ec.product_category).median()
base = pd.read_pickle(DSC / "agas/shared/cache/features_cache.pkl")
rep = {"price_median": [], "price_rel_spearman": [], "gradenorm_max_abs_dz": [], "gradenorm_min_spearman": []}
for k, (tr, _) in enumerate(folds):
    fit_c = set(cust.loc[lab.application_id.iloc[tr]])
    m = ec.customer_id.isin(fit_c).to_numpy()
    part = unit[m].groupby(ec.product_category[m]).median()
    rep["price_median"].append(float((part - glob).abs().max()))
    rel_g = (unit - ec.product_category.map(glob)).groupby(ec.customer_id).mean()
    rel_p = (unit - ec.product_category.map(part)).groupby(ec.customer_id).mean()
    rep["price_rel_spearman"].append(float(spearmanr(rel_g, rel_p.loc[rel_g.index]).statistic))
    fitrow = base.application_id.isin(lab.application_id.iloc[tr]).to_numpy()
    dz, rho = [], []
    for grp in ("interest_rate", "employment_type", "region"):
        g = base[grp].astype(str)
        for c in [c for c in GN if c in base.columns]:
            x = pd.to_numeric(base[c], errors="coerce")
            zg = (x - x.groupby(g).transform("mean")) / x.groupby(g).transform("std")
            mp, sp = x[fitrow].groupby(g[fitrow]).mean(), x[fitrow].groupby(g[fitrow]).std()
            zp = (x - g.map(mp)) / g.map(sp)
            ok = zg.notna() & zp.notna()
            dz.append(float((zg[ok] - zp[ok]).abs().max()))
            rho.append(float(spearmanr(zg[ok], zp[ok]).statistic))
    rep["gradenorm_max_abs_dz"].append(max(dz))
    rep["gradenorm_min_spearman"].append(min(rho))
    print(k, {key: round(val[-1], 6) for key, val in rep.items()}, flush=True)
(ROOT / "eda/out/263_dsc_reference_audit.json").write_text(json.dumps(rep, indent=1))
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(range(5), rep["gradenorm_max_abs_dz"], label="gradenorm max |dz|")
ax.bar(np.arange(5) + .3, rep["price_median"], width=.3, label="max |d log median price|")
ax.set_xlabel("outer fold"); ax.legend(); ax.set_title("263 population vs per-partition references")
fig.tight_layout(); fig.savefig(ROOT / "eda/out/263_dsc_reference_audit.png", dpi=120)
