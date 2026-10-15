"""Macro/time structure: every loan is observed for 12 months after origination, so a simulated economy-wide shock
would make the bad rate depend smoothly on the decision DATE (not only the month the models see as dec_month), and
possibly differently per employment type / region / income source. Label use: bad rates per bin (counting only).
Tests the v6 residual against week-of-year, day-of-month, weekday and date x segment interactions."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, calibrate, features, ks, resid_test, v6_oof

f = features(); tr = f.filter(pl.col("is_train") == 1); y = tr["target"].to_numpy()
s = v6_oof(); p = calibrate(y, s)
rng = np.random.default_rng(9)
null = np.array([resid_test(y, p, rng.random(len(y)), bins=20) for _ in range(30)])
CHI, DKS = np.percentile(null[:, 0], 95), np.percentile(null[:, 2], 95)
print(f"noise (20 bins) chi2/df p95={CHI:.2f} dKS p95={DKS:+.4f}")

d = tr["decision_datetime"]
t = tr.select("target", "employment_type", "region", "has_history").with_columns(
    doy=d.dt.ordinal_day(), wk=d.dt.week(), dom=d.dt.day(), wd=d.dt.weekday(), hr=d.dt.hour(), p=pl.Series(p))
g = t.group_by("wk").agg(n=pl.len(), bad=pl.col("target").mean(), exp=pl.col("p").mean()).sort("wk")
print("bad rate vs v6-expected per decision week: sd(O-E) =", float((g["bad"] - g["exp"]).std()),
      " binomial sd expected ~", float(np.sqrt((g["exp"] * (1 - g["exp"]) / g["n"]).mean())))
rows = []
for name, x in {"doy": t["doy"], "week": t["wk"], "dom": t["dom"], "weekday": t["wd"], "dec_hour": t["hr"]}.items():
    rows.append((name, *resid_test(y, p, x.to_numpy().astype(float), bins=20)))
# date x segment: residual of (date bin, segment) cells
for seg in ("employment_type", "region", "has_history"):
    code = t[seg].cast(pl.String).rank("dense").to_numpy()
    x = (t["doy"].to_numpy() // 61) * 10 + code  # 6 two-month periods x segment
    rows.append((f"bimonth_x_{seg}", *resid_test(y, p, x.astype(float), bins=20)))
R = pl.DataFrame(rows, schema=["candidate", "chi2_df", "max_lift", "dKS_cf"], orient="row").with_columns(
    above=(pl.col("chi2_df") > CHI) & (pl.col("dKS_cf") > DKS))
with pl.Config(float_precision=4):
    print(R)
R.write_csv(OUT / "64_macro_time.csv")

# per-month O/E by employment type (is there a shock that only hits variable income?)
m = t.with_columns(mon=d.dt.month()).group_by("mon", "employment_type").agg(oe=pl.col("target").sum() / pl.col("p").sum(), n=pl.len()).sort("mon")
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(g["wk"], g["bad"], "o-", label="observed"); ax[0].plot(g["wk"], g["exp"], "-", label="v6 calibrated")
ax[0].set_title("bad rate by decision week"); ax[0].legend()
for et in m["employment_type"].unique().sort():
    q = m.filter(pl.col("employment_type") == et); ax[1].plot(q["mon"], q["oe"], "o-", label=et)
ax[1].axhline(1, color="k", lw=0.5); ax[1].set_title("observed/expected by month x employment"); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT / "64_macro_time.png", dpi=80)
