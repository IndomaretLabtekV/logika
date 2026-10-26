"""Is there group/time structure the models miss? Residual overdispersion of the best OOF (K64 2-seed,
isotonic-calibrated on OOF for diagnosis only) per grouping: phi = sum_g (y_g - p_g)^2 / sum_g var_g
divided by G (1 = binomial noise). Also lag-1 residual autocorrelation in decision-time order and a
permutation control (groups shuffled). Train labels only; nothing here is a model input.
Figure: eda/out/233_group_overdispersion.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import chi2
from sklearn.isotonic import IsotonicRegression

from common import ROOT

apps = pl.read_csv(ROOT / "data/train_applications.csv", try_parse_dates=True)
prof = pl.read_csv(ROOT / "data/customer_profiles.csv")
oof = pl.read_parquet(ROOT / "results/v60/candidates/K__K64_2seed/oof.parquet")
d = apps.join(prof, on="customer_id").join(oof.select("application_id", "K"), on="application_id")
assert d.height == 70000
y = d["target"].to_numpy().astype(float)
p = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(d["K"].to_numpy(), y).predict(d["K"].to_numpy())
d = d.with_columns(pl.Series("p", p), pl.Series("r", y - p),
                   pl.col("decision_datetime").dt.strftime("%Y-%m").alias("month"),
                   pl.col("decision_datetime").dt.strftime("%Y-%W").alias("week"),
                   pl.col("decision_datetime").dt.date().cast(pl.String).alias("day"),
                   pl.col("decision_datetime").dt.hour().cast(pl.String).alias("hour"),
                   pl.col("decision_datetime").dt.weekday().cast(pl.String).alias("weekday"),
                   (pl.col("loan_amount") // 10_000_000).cast(pl.String).alias("amount_bin"),
                   pl.col("term_months").cast(pl.String).alias("term"),
                   pl.col("annual_interest_rate").cast(pl.String).alias("rate"),
                   pl.col("dependents").cast(pl.String).alias("deps"))
groupings = {"month": ["month"], "week": ["week"], "day": ["day"], "hour": ["hour"], "weekday": ["weekday"],
             "region": ["region"], "region×month": ["region", "month"], "region×week": ["region", "week"],
             "purpose×month": ["loan_purpose", "month"], "employment×month": ["employment_type", "month"],
             "housing×month": ["housing_status", "month"], "term×rate": ["term", "rate"],
             "purpose×region": ["loan_purpose", "region"], "amount_bin×term": ["amount_bin", "term"],
             "deps×employment": ["deps", "employment_type"]}


def phi(frame, keys):
    g = frame.group_by(keys).agg(pl.col("r").sum().alias("rs"), (pl.col("p") * (1 - pl.col("p"))).sum().alias("v"), pl.len().alias("n"))
    g = g.filter(pl.col("v") > 1)
    stat = float((g["rs"] ** 2 / g["v"]).sum())
    return stat / g.height, g.height, float(chi2.sf(stat, g.height))


rng = np.random.default_rng(233)
rows = []
for name, keys in groupings.items():
    ph, G, pv = phi(d, keys)
    perm = d.with_columns([pl.Series(k, rng.permutation(d[k].to_numpy())) for k in keys])
    ph0, _, _ = phi(perm, keys)
    rows.append((name, G, ph, pv, ph0))
    print(f"{name:18s} G={G:5d} phi={ph:.3f} p={pv:.2e} perm_phi={ph0:.3f}", flush=True)
order = d.sort("decision_datetime")["r"].to_numpy()
ac = np.corrcoef(order[:-1], order[1:])[0, 1]
print(f"lag-1 residual autocorrelation in decision-time order: {ac:.4f} (se ~ {1/np.sqrt(len(order)):.4f})")
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh([r[0] for r in rows], [r[2] for r in rows], label="observed")
ax.scatter([r[4] for r in rows], [r[0] for r in rows], color="k", s=12, label="permuted control", zorder=3)
ax.axvline(1, color="grey", lw=.8, ls="--"); ax.set_xlabel("dispersion phi (1 = binomial)"); ax.legend()
ax.set_title("233 residual overdispersion of K64 2-seed OOF"); fig.tight_layout()
fig.savefig(ROOT / "eda/out/233_group_overdispersion.png", dpi=120)
