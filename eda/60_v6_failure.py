"""Why did v6 (public 0.56123) not beat v5 (0.56145), and how far is top-1 (0.57184) in real terms?
Only saved OOF/test predictions are used (no model). Answers:
 1) how noisy is a public KS on a random 30%/50% slice of 30k rows (the sd of what we are reading on the LB);
 2) is the stable public-OOF offset (+0.033) explained by subset size (KS is a max -> biased up on small samples);
 3) where v5 and v6 test predictions disagree and how big a public difference that disagreement can make;
 4) per month / segment KS of v6 (stability, the public-private risk)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr
from common import OUT, ROOT, features, ks, v5_oof, v6_oof

f = features(); tr = f.filter(pl.col("is_train") == 1); te = f.filter(pl.col("is_train") == 0)
y = tr["target"].to_numpy(); n_te = te.height
s5, s6 = v5_oof(), v6_oof()
rng = np.random.default_rng(0)
print(f"full OOF KS v5 {ks(y, s5):.4f}  v6 {ks(y, s6):.4f}  diff {ks(y, s6) - ks(y, s5):+.4f}")

# 1+2) KS on random subsets of the size of a public slice
rows = []
for share in (0.3, 0.5):
    m = int(share * n_te)
    d = []
    for _ in range(400):
        i = rng.choice(len(y), m, replace=False)
        d.append((ks(y[i], s5[i]), ks(y[i], s6[i])))
    d = np.array(d)
    rows.append((share, m, d[:, 1].mean() - ks(y, s6), d[:, 1].std(), (d[:, 1] - d[:, 0]).std(), np.mean(d[:, 1] < d[:, 0])))
    print(f"public share {share:.0%} ({m} rows): KS bias {rows[-1][2]:+.4f}  sd {rows[-1][3]:.4f}  "
          f"sd(v6-v5) {rows[-1][4]:.4f}  P(v6<v5) {rows[-1][5]:.2f}")
pl.DataFrame(rows, schema=["share", "rows", "ks_bias", "ks_sd", "sd_v6_minus_v5", "p_v6_below_v5"], orient="row").write_csv(OUT / "60_public_noise.csv")
lb = {"v1": (0.531, 0.5051), "v2": (0.54192, 0.5134), "v3": (0.54545, 0.5142), "v5": (0.56145, 0.5274), "v6": (0.56123, 0.5283)}
off = np.array([p - o for p, o in lb.values()])
print("public - OOF per version:", {k: round(p - o, 4) for k, (p, o) in lb.items()}, f"mean {off.mean():+.4f} sd {off.std():.4f}")
print(f"top-1 0.57184 -> OOF-equivalent ~{0.57184 - off.mean():.4f} (gap to v6 {0.57184 - off.mean() - ks(y, s6):+.4f})")

# 3) disagreement between v5 and v6 test predictions
R = ROOT / "results"
t5 = pl.read_csv(R / "v5" / "submission.csv"); t6 = pl.read_csv(R / "v6" / "submission.csv")
j = t5.join(t6, on="application_id", suffix="_6")
r5, r6 = rankdata(j["default_probability"]) / n_te, rankdata(j["default_probability_6"]) / n_te
print(f"test Spearman v5-v6 {spearmanr(r5, r6)[0]:.4f}; |rank change| p50 {np.median(abs(r5 - r6)):.4f} p95 {np.percentile(abs(r5 - r6), 95):.4f}")
o5, o6 = rankdata(s5) / len(y), rankdata(s6) / len(y)
print(f"OOF  Spearman v5-v6 {spearmanr(o5, o6)[0]:.4f}; |rank change| p50 {np.median(abs(o5 - o6)):.4f} p95 {np.percentile(abs(o5 - o6), 95):.4f}")
# rows where v6 moved a lot: did the move go the right way on OOF?
mv = o6 - o5; big = abs(mv) > np.percentile(abs(mv), 90)
print(f"OOF rows with top-10% rank moves: n={big.sum()} bad-rate {y[big].mean():.3f}; "
      f"moved up & bad {np.mean(y[big & (mv > 0)]):.3f}, moved down & bad {np.mean(y[big & (mv < 0)]):.3f} (base {y.mean():.3f})")

# 4) per-month and segment KS
mon = tr["decision_datetime"].dt.month().to_numpy()
hist = tr["has_history"].to_numpy() == 1
mk = [(m, (mon == m).sum(), ks(y[mon == m], s6[mon == m]), ks(y[mon == m], s5[mon == m])) for m in np.unique(mon)]
for m, n, a, b in mk:
    print(f"month {m:2d}: n={n:5d}  v6 {a:.4f}  v5 {b:.4f}  diff {a - b:+.4f}")
for name, m in (("thin", ~hist), ("history", hist)):
    print(f"{name:8s}: n={m.sum()} v6 {ks(y[m], s6[m]):.4f} v5 {ks(y[m], s5[m]):.4f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
mk = np.array(mk)
ax[0].bar(mk[:, 0] - 0.2, mk[:, 2], 0.4, label="v6"); ax[0].bar(mk[:, 0] + 0.2, mk[:, 3], 0.4, label="v5")
ax[0].set_ylim(0.45, 0.6); ax[0].set_title("OOF KS by decision month"); ax[0].legend()
ax[1].hist(r6 - r5, 100); ax[1].set_title("test rank change v6 - v5")
ax[2].bar(list(lb), off); ax[2].set_title("public - OOF offset per version"); ax[2].set_ylim(0, 0.04)
plt.tight_layout(); plt.savefig(OUT / "60_v6_failure.png", dpi=80)
