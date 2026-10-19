"""VIP-COP (Chen, Ding & Akoglu, arXiv 2605.12904) for Kumo-Tabular large, feature-level hard context optimisation,
validation_v2-clean fold-0 screen (seed 42).
Items = the 228 numeric features (the 4 categoricals always kept); the context gets d_C = 100 of them. Each round
samples B feature sets with P(i) = softmax(z(phi)/tau) (without replacement), scores each with ONE frozen-Kumo forward
pass (search context = 20k stratified inner-fit rows, query = 4,000 fixed inner-stop rows, AUROC), and refits phi by
ridge regression of AUROC on the inclusion indicators over all coalitions so far (closed-form KernelSHAP-style
surrogate). Several temperature chains run; each returns its top-100 by phi; the chain whose selection scores best on
the inner-stop query wins. Only training-fold labels are used. The outer fold is scored ONCE with the whole training
fold as context (e4, generator seed 0) and compared paired with the same Kumo on fm_select features
(eda/out/kumo_finetune_f0_base.parquet, same context / generator / estimators).
Usage: cd eda && ../.venv/bin/python 152_vipcop_kumo.py [--rounds 40 --batch 8 --taus 1,0.5,0.25]"""
import argparse
import time

import numpy as np
import polars as pl
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

import sdm
import valid2 as v
from common import OUT, features

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=40)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--taus", default="1,0.5,0.25")
ap.add_argument("--d", type=int, default=100)
ap.add_argument("--nctx", type=int, default=20000)
ap.add_argument("--nq", type=int, default=4000)
a = ap.parse_args()

X, Xt, y, ids, tids = v.frames(features())
tr, va = v.outer(y, 42)[0]
fi, st = v.inner(y[tr], 42, 0)
fi, st = tr[fi], tr[st]
nums = [c for c in X.columns if c not in v.CAT_COLS]
F = X[nums + v.CAT_COLS].copy()
for c in v.CAT_COLS:
    F[c] = F[c].cat.codes.astype(float).replace(-1, np.nan)
F = F.astype(float).reset_index(drop=True).assign(target=y.astype(str))
stypes = sdm.infer_stypes(F, overrides={"target": "categorical", **{c: "numerical" for c in nums + v.CAT_COLS}})
dev = torch.device("cuda")
T = sdm.TableTensor.from_pandas(df=F, stypes=stypes, device=dev)
model = sdm.models.KumoTabular(task="classification", size="large", device=dev)
rng = np.random.default_rng(0)
strat = lambda pool, n, r: np.sort(np.concatenate([r.choice(pool[y[pool] == c], int(round(n * (y[pool] == c).mean())), replace=False) for c in (0, 1)]))  # noqa: E731
ctx_search = strat(fi, a.nctx, np.random.default_rng(1))
q_search = np.sort(np.random.default_rng(2).choice(st, a.nq, replace=False))
C0, Q0 = T[torch.as_tensor(ctx_search, device=dev)], T[torch.as_tensor(q_search, device=dev)]


def predict(C, Q, cols, n_est=1, seed=0, chunk=2000):
    xc, yc = C.select_columns(cols), C[:, "target"]
    g = torch.Generator(device=dev).manual_seed(seed)
    out = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for i in range(0, Q.size(0), chunk):
            r = model(x_context=xc, y_context=yc, x_query=Q[i:i + chunk].select_columns(cols), num_estimators=n_est, generator=g)
            out.append(r.to_pandas()["1"].to_numpy())  # by label name (class order varies per call, eda/112)
    return np.concatenate(out)


def score(idx):
    cols = [nums[i] for i in idx] + v.CAT_COLS
    return roc_auc_score(y[q_search], predict(C0, Q0, cols))


S = len(nums)
t0 = time.time()
fm = [c for c in v.fm_select(X, y, tr, a.d) if c not in v.CAT_COLS]
auc_fm = score([nums.index(c) for c in fm])
print(f"fm_select (KS-ranked) features on the search query: AUROC {auc_fm:.5f}", flush=True)
chains = []
for tau in [float(t) for t in a.taus.split(",")]:
    Cm, P = [], []
    phi = np.full(S, 1.0 / S)
    for t in range(a.rounds):
        z = (phi - phi.mean()) / (phi.std() + 1e-12)
        pr = np.exp(z / tau)
        pr /= pr.sum()
        for _ in range(a.batch):
            idx = rng.choice(S, a.d, replace=False, p=pr)
            c = np.zeros(S)
            c[idx] = 1
            Cm.append(c)
            P.append(score(idx))
        phi = Ridge(alpha=1.0).fit(np.array(Cm), np.array(P)).coef_
        if t % 10 == 9:
            print(f"tau {tau} round {t + 1}: coalitions {len(P)}, mean AUROC {np.mean(P[-a.batch:]):.5f}, best so far "
                  f"{np.max(P):.5f}, {time.time() - t0:.0f} s", flush=True)
    sel = np.argsort(-phi)[: a.d]
    chains.append((score(sel), tau, sel, phi))
    print(f"tau {tau}: selected top-{a.d} AUROC {chains[-1][0]:.5f} (fm_select {auc_fm:.5f}); overlap with fm_select "
          f"{len(set(nums[i] for i in sel) & set(fm))}", flush=True)
best = max(chains, key=lambda c: c[0])
cols = [nums[i] for i in best[2]] + v.CAT_COLS
pl.DataFrame({"feature": nums, "phi": best[3]}).sort("phi", descending=True).write_csv(OUT / "152_vipcop_phi.csv")
print("top 15 by VIP:", [nums[i] for i in np.argsort(-best[3])[:15]])
# outer fold, scored once: whole training fold as context, e4, generator seed 0 (same as the cached baseline)
Ctr, Qva = T[torch.as_tensor(tr, device=dev)], T[torch.as_tensor(va, device=dev)]
p_vip = predict(Ctr, Qva, cols, n_est=4, seed=0, chunk=1000)
base = pl.read_parquet(OUT / "kumo_finetune_f0_base.parquet").sort("application_id")
assert (base["application_id"] == pl.Series(ids.to_numpy()[va]).sort()).all()
order = np.argsort(ids.to_numpy()[va])
b = base["base"].to_numpy()[np.argsort(order)]
r = np.random.default_rng(0)
d = np.array([(lambda i: v.ks(y[va][i], p_vip[i]) - v.ks(y[va][i], b[i]))(r.choice(len(va), len(va))) for _ in range(1000)])
print(f"outer fold 0: VIP-COP features KS {v.ks(y[va], p_vip):.5f} vs fm_select Kumo {v.ks(y[va], b):.5f}  "
      f"d {d.mean():+.5f} sd {d.std():.5f} P>0 {(d > 0).mean():.2f}  ({time.time() - t0:.0f} s total)")
oof = np.full(len(y), np.nan)
oof[va] = p_vip
pl.DataFrame({"application_id": ids, "target": y, "kumo_vipcop": oof}).write_parquet(OUT / "152_oof_kumo_vipcop_f0.parquet")
