"""Model-free scorecard (rank average of orthogonal signals; only the sign is taken from train) ->
KS floor, and simulated public/private KS gap for a 30k test set."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, ROOT, features, ks

f = features()
# one representative per redundancy cluster (11_correlation_conditional.py); NaN -> neutral rank 0.5
comp = {"bc_score_mean": -1, "rp_frac_dpd1": +1, "ec_accel_30d": +1, "annual_interest_rate": +1,
        "rp_cur_dpd_max": +1, "ac_out_in_delta": +1, "debt_to_income": +1, "ec_aov": +1, "pv_rej_rate": +1}
def score(df, w=None):
    w = w or {c: 1.0 for c in comp}
    parts = [(pl.col(c).rank() / pl.col(c).count()).fill_null(0.5) * s * w[c] for c, s in comp.items()]
    return df.select(sum(parts).alias("s"))["s"].to_numpy()
tr = f.filter(pl.col("is_train") == 1); y = tr["target"].to_numpy()
s = score(tr)
print(f"scorecard KS (equal weights): {ks(y, s):.4f}")
for drop in comp:
    w = {c: (0.0 if c == drop else 1.0) for c in comp}
    print(f"  without {drop:<22} KS={ks(y, score(tr, w)):.4f}")

# public/private gap simulation: test=30k, public share unknown -> try 30/50 %
rng = np.random.default_rng(0)
for pub in (0.3, 0.5):
    gaps, pubs = [], []
    for _ in range(300):
        idx = rng.choice(len(y), 30000, replace=False); cut = int(pub * 30000)
        a, b = idx[:cut], idx[cut:]
        ka, kb = ks(y[a], s[a]), ks(y[b], s[b]); gaps.append(ka - kb); pubs.append(ka)
    gaps = np.array(gaps)
    print(f"public={pub:.0%}: KS_pub sd={np.std(pubs):.4f}  |pub-priv| p50={np.median(np.abs(gaps)):.4f} p95={np.percentile(np.abs(gaps), 95):.4f}")
plt.hist(gaps, bins=40); plt.title("simulated KS_public - KS_private (public=50%)"); plt.savefig(OUT / "13_ks_gap.png", dpi=80)

sub = f.filter(pl.col("is_train") == 0)
st = score(sub); st = (st - st.min()) / (st.max() - st.min())
(ROOT / "submissions").mkdir(exist_ok=True)
pl.DataFrame({"application_id": sub["application_id"], "default_probability": st}).write_csv(ROOT / "submissions" / "00_scorecard_baseline.csv")
print("wrote submissions/00_scorecard_baseline.csv")
