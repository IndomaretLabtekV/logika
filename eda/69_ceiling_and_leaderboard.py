"""Two questions that decide where to spend GPU time:
 1) Self-consistency / Bayes view: if the calibrated v6 probability were the true default probability, what KS would
    labels drawn from it give? If that equals the observed 0.528, the errors are Bernoulli noise around our p, and a
    higher KS needs NEW information (more extreme p), not a better learner on the same inputs.
    A second probe: sharpen p (temperature on the logit) and see how much KS a hypothetically better-informed model
    gets -> how much extra information 0.541 (top-1 equivalent) would need.
 2) Leaderboard selection: simulate many teams whose models are as good as ours but rank differently (rank correlation
    0.95 with us, the typical spread between independent pipelines) and who each pick their best of K near-identical
    submissions on the public slice. How far above our public score does the top public score land by luck alone?"""
import functools
import matplotlib; matplotlib.use("Agg")
print = functools.partial(print, flush=True)  # noqa: A001
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.special import expit, logit
from scipy.stats import rankdata
from common import OUT, calibrate, features, v6_oof


def ks(yy, x):
    """Fast exact two-sample KS (same value as ks_2samp for continuous scores)."""
    yy = np.asarray(yy, bool); o = np.argsort(x, kind="mergesort"); t = yy[o]
    return float(np.max(np.abs(np.cumsum(t) / t.sum() - np.cumsum(~t) / (~t).sum())))


f = features(); y = f.filter(pl.col("is_train") == 1)["target"].to_numpy(); n = len(y)
s = v6_oof(); p = np.clip(calibrate(y, s, bins=100), 1e-4, 1 - 1e-4)
rng = np.random.default_rng(69)
sim = [ks(rng.random(n) < p, s) for _ in range(100)]
print(f"observed KS {ks(y, s):.4f}; KS if calibrated v6 p were the truth: {np.mean(sim):.4f} +- {np.std(sim):.4f}")
print("-> errors are consistent with Bernoulli noise around our p" if abs(np.mean(sim) - ks(y, s)) < 2 * np.std(sim) else
      "-> observed KS differs from the self-consistent KS")
rows = []
for t in (1.0, 1.05, 1.1, 1.2, 1.3, 1.5):
    q = expit(logit(p) * t)  # a model whose log-odds are t times more spread (more information), same ranking
    k = np.mean([ks(rng.random(n) < q, q) for _ in range(30)])
    rows.append((t, k, float(np.std(logit(q)))))
T = pl.DataFrame(rows, schema=["logit_scale", "expected_KS", "sd_logodds"], orient="row")
with pl.Config(float_precision=4):
    print("expected KS of a model whose log-odds spread is scaled (truth = that model):"); print(T)

# 2) leaderboard in a synthetic world with EQUAL-quality teams: true log-odds L = a * logit(p_cal); every team sees
#    L + independent noise (sd sigma), tuned so each team's KS equals ours (~0.528); labels y ~ Bernoulli(sigmoid(L)).
#    Each team submits 10 near-identical variants (rank corr ~0.998) and keeps its best public score.
from scipy.special import expit as sig
lp = logit(p)
res = []
for a_ in (1.05, 1.15, 1.3):
    sd_l = np.std(a_ * lp)
    sig_ = None
    for cand in np.linspace(0.05, 2.0, 79):
        yy = rng.random(n) < sig(a_ * lp)
        if ks(yy, a_ * lp + cand * rng.standard_normal(n)) < ks(y, s):
            sig_ = cand; break
    team_corr = sd_l ** 2 / (sd_l ** 2 + sig_ ** 2)
    for share in (0.3, 0.5):
        m = int(share * 30000); d = []
        for it in range(40):
            idx = rng.choice(n, m, replace=False)
            L_ = a_ * lp[idx]; yy = rng.random(m) < sig(L_)
            ours = ks(yy, L_ + sig_ * rng.standard_normal(m))
            best = max(max(ks(yy, (base := L_ + sig_ * rng.standard_normal(m)) + 0.06 * sig_ * rng.standard_normal(m)) for _ in range(10))
                       for _ in range(20))
            d.append(best - ours)
        d = np.array(d)
        res.append((a_, sig_, team_corr, share, d.mean(), np.percentile(d, 10), np.percentile(d, 90)))
        print(f"truth spread x{a_}, team noise sd {sig_:.2f} (team-to-team corr {team_corr:.2f}), public {share:.0%}: "
              f"top-of-20 teams (best of 10 subs) minus an equal team: mean {d.mean():+.4f} [p10 {np.percentile(d, 10):+.4f}, p90 {np.percentile(d, 90):+.4f}]")
L = pl.DataFrame(res, schema=["truth_scale", "team_noise_sd", "team_corr", "public_share", "top_minus_equal_mean", "p10", "p90"], orient="row")
L.write_csv(OUT / "69_leaderboard.csv"); T.write_csv(OUT / "69_ceiling.csv")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(sim, 30); ax[0].axvline(ks(y, s), color="r", label="observed"); ax[0].legend(); ax[0].set_title("KS if calibrated v6 p were the truth")
ax[1].plot(T["logit_scale"], T["expected_KS"], "o-"); ax[1].axhline(0.541, color="r", ls="--", label="top-1 OOF-equivalent")
ax[1].set_xlabel("log-odds spread x"); ax[1].set_title("KS needs more spread (information)"); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT / "69_ceiling_and_leaderboard.png", dpi=80)
