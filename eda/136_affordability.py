"""Three-way affordability constructs (trees approximate ratios of sums poorly): total monthly debt service after the
new loan (new installment + active previous installments) over verified recent income (account income_like_credits,
last 3 complete months), free cash after the new installment, buffer months of balance, and the same against declared
income and against outflows. Inputs are existing leak-safe features (all <= cutoff). Residual test vs the clean
TREES-ONLY OOF as eda/134/135.
Usage: cd eda && ../.venv/bin/python 136_affordability.py"""
import io
import contextlib

import numpy as np
import polars as pl

import blend2
import valid2 as v
from common import OUT, features
from common9 import calibrate, crossfit_dks, oe_test, safe_bins

f = features()
inst = pl.col("proposed_monthly_installment")
act = pl.col("rp_active_inst").fill_null(0)
c = f.select(
    "application_id", "is_train",
    F_dsr_verified_l3=(inst + act) / (pl.col("ac_income_l3") + 1e5),
    F_dsr_inflow=(inst + act) / (pl.col("ac_inflow_mean") + 1e5),
    F_free_after_new=(pl.col("ac_income_l3") - pl.col("cf_out_l3") - inst) / (pl.col("ac_income_l3") + 1e5),
    F_buffer_months=pl.col("ac_avgbal_l3") / (inst + act + 1e5),
    F_minbal_months=pl.col("cf_minbal_l3") / (inst + act + 1e5),
    F_dsr_declared_total=(inst + act) / pl.col("declared_monthly_income"),
    F_dsr_bureau=(inst + act + 0.05 * pl.col("bc_outstanding_max").fill_null(0)) / pl.col("declared_monthly_income"),
    F_out_plus_new=(pl.col("cf_out_l3") + inst) / (pl.col("ac_inflow_mean") + 1e5),
)
c.drop("is_train").write_parquet(OUT / "136_affordability.parquet")
tr = c.filter(pl.col("is_train") == 1).sort("application_id")
with contextlib.redirect_stdout(io.StringIO()):
    oof, _, _, _, _, res = blend2.main(["../results/v22/v22"])
assert (oof["application_id"] == tr["application_id"]).all()
y = oof["target"].to_numpy()
pc = calibrate(y, res["equal_trees"][0])
rng = np.random.default_rng(0)
x0 = f.filter(pl.col("is_train") == 1).sort("application_id")["ac_income_l3"].cast(pl.Float64).to_numpy()
null = [(oe_test(y, pc, safe_bins(z, 10))[0], crossfit_dks(y, pc, safe_bins(z, 10))) for z in (rng.permutation(x0) for _ in range(30))]
c95, d95 = np.quantile([a for a, _ in null], .95), np.quantile([b for _, b in null], .95)
print(f"null (permuted ac_income_l3, keeps missing block): chi2/df p95 {c95:.2f}, dKS p95 {d95:+.5f}")
rows = []
for col in [k for k in tr.columns if k.startswith("F_")]:
    x = tr[col].cast(pl.Float64).to_numpy()
    x = np.where(np.isfinite(x), x, np.nan)
    b = safe_bins(x, 10)
    chi, tab = oe_test(y, pc, b)
    rows.append(dict(feature=col, cover=float(np.isfinite(x).mean()), ks_uni=v.ks_nan(y, x), chi2=chi, dks=crossfit_dks(y, pc, b),
                     oe=" ".join(f"{t[4]:.2f}" for t in tab)))
with pl.Config(tbl_rows=20, tbl_cols=8, fmt_str_lengths=80, tbl_width_chars=200):
    print(pl.DataFrame(rows).sort("chi2", descending=True))
pl.DataFrame(rows).write_csv(OUT / "136_affordability.csv")
