"""Relative-measure scan: the strongest planted signal (pp_mean) is *relative* (log unit price vs the category norm),
and eda/91 only tested counts/sums per time window. Here the item price z (same expanding category-month reference
as features.v5_features) is averaged per customer within subgroups the feature set never split by: payment method,
cancelled vs completed, night/day, weekend, multi-quantity, basket size, plus per-customer z dispersion and trend.
Each candidate is tested for information beyond the clean v31 blend (validation_v2 OOF, trees+Causilo+Kumo):
likelihood-ratio chi2 of adding [x, missing(x)] to logit(target) ~ logit(p_oof). Existing pp_* features are scored
the same way as a calibration (they are already inside the models, so they should be ~null).
Saves eda/out/111_relative_scan.csv."""
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2, rankdata

import blend2 as b
from common import OUT, features, load

oof, _ = b.load(["../results/v22/v22", "../results/v28m"])
names = [c for c in oof.columns if c not in ("application_id", "target")]
R = {n: rankdata(oof[n].to_numpy()) / oof.height for n in names}
trees = [n for n in names if n.split("_")[0] in ("lgb", "xgb", "cat")]
s = (np.mean([R[n] for n in trees], 0) + R["causilo"] + R["kumo_large_c56000"]) / 3
y = oof["target"].to_numpy()


def llf(X):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum()), m.coef_[0]


base_X = np.log(s / (1 - s + 1e-9) + 1e-9)[:, None]
base_llf, _ = llf(base_X)
print(f"base logit fit: llf {base_llf:.1f}")

apps = load("train_applications")
cut = apps.select("application_id", "customer_id", cutoff=pl.col("decision_datetime"), dm=pl.col("decision_datetime").dt.truncate("1mo"))
raw = load("ecommerce_transactions").with_columns(lu=(pl.col("item_total") / pl.col("quantity")).log(),
                                                  ym=pl.col("transaction_datetime").dt.truncate("1mo"))
ms = raw.group_by("product_category", "ym").agg(n=pl.len(), s=pl.col("lu").sum(), ss=(pl.col("lu") ** 2).sum())
ref = (cut.select("dm").unique().join(ms, how="cross").filter(pl.col("ym") < pl.col("dm"))
       .group_by("product_category", "dm").agg(pl.col("n", "s", "ss").sum())
       .with_columns(mu=pl.col("s") / pl.col("n"), sd=((pl.col("ss") - pl.col("s") ** 2 / pl.col("n")) / (pl.col("n") - 1)).sqrt()))
e = raw.join(cut, on="customer_id").filter(pl.col("transaction_datetime") <= pl.col("cutoff"))
e = e.join(ref.select("product_category", "dm", "mu", "sd"), on=["product_category", "dm"], how="left")
e = e.with_columns(z=(pl.col("lu") - pl.col("mu")) / pl.col("sd"),
                   age_d=(pl.col("cutoff") - pl.col("transaction_datetime")).dt.total_seconds() / 86400,
                   hour=pl.col("transaction_datetime").dt.hour(), wd=pl.col("transaction_datetime").dt.weekday(),
                   n_items=pl.len().over("order_id"))
zf = lambda cond: pl.col("z").filter(cond).mean()  # noqa: E731
agg = {f"z_pm_{m}": zf(pl.col("payment_method") == m) for m in ["COD", "e_wallet", "bank_transfer", "debit_card", "credit_card"]}
agg |= {
    "z_cancelled": zf(pl.col("order_status") == "cancelled"),
    "z_night": zf((pl.col("hour") >= 22) | (pl.col("hour") < 5)),
    "z_day": zf((pl.col("hour") >= 9) & (pl.col("hour") < 18)),
    "z_weekend": zf(pl.col("wd") >= 6),
    "z_qty_gt1": zf(pl.col("quantity") > 1),
    "z_basket1": zf(pl.col("n_items") == 1),
    "z_basket3p": zf(pl.col("n_items") >= 3),
    "z_sd": pl.col("z").std(),
    "z_frac_gt1": (pl.col("z") > 1).mean(),
    "z_frac_lt_m1": (pl.col("z") < -1).mean(),
    "z_slope": pl.cov("z", "age_d") / pl.col("age_d").var(),
    "z_last10": pl.col("z").sort_by("age_d").head(10).mean(),
    "credit_spend_share": pl.col("item_total").filter(pl.col("payment_method") == "credit_card").sum() / pl.col("item_total").sum(),
    "cod_spend_share": pl.col("item_total").filter(pl.col("payment_method") == "COD").sum() / pl.col("item_total").sum(),
}
cand = e.group_by("application_id").agg(**agg)
cand = cand.with_columns(
    z_credit_minus_debit=pl.col("z_pm_credit_card") - pl.col("z_pm_debit_card"),
    z_cod_minus_ewallet=pl.col("z_pm_COD") - pl.col("z_pm_e_wallet"),
    z_cancel_minus_all=pl.col("z_cancelled") - pl.col("z_day"),
    z_night_minus_day=pl.col("z_night") - pl.col("z_day"),
)
feats = features().filter(pl.col("is_train") == 1).select("application_id", pl.col("^pp_.*$"))
d = oof.select("application_id").join(cand, on="application_id", how="left").join(feats, on="application_id", how="left")
rows = []
for c in [c for c in d.columns if c != "application_id"]:
    x = d[c].to_numpy().astype(float)
    miss = np.isnan(x)
    if miss.all():
        continue
    xr = np.where(miss, 0.0, (rankdata(np.where(miss, np.nanmedian(x), x)) / len(x) - 0.5))
    cols = [base_X, xr[:, None]] + ([miss[:, None].astype(float)] if 0 < miss.sum() < len(x) else [])
    ll, coef = llf(np.hstack(cols))
    lr = 2 * (ll - base_llf)
    dfree = len(cols) - 1
    rows.append(dict(feature=c, chi2=lr, df=dfree, p=chi2.sf(lr, dfree), coef=coef[1], coverage=1 - miss.mean(),
                     existing=c.startswith("pp_")))
R_ = pd.DataFrame(rows).sort_values("chi2", ascending=False)
n_new = (~R_.existing).sum()
print(f"{n_new} new candidates; Bonferroni chi2(1) bar at 0.05 ~ {chi2.isf(0.05 / n_new, 1):.1f}")
print(R_.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
R_.to_csv(OUT / "111_relative_scan.csv", index=False)
