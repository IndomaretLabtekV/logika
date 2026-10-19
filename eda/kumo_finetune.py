"""Fine-tune Kumo-Tabular large on this competition's training fold (fold-0 screen, validation_v2-clean).
Episodes: a stratified context of C rows and Q query rows, both drawn from the inner-fit rows only; loss = BCE of the
query predictions. Early stopping on the inner stop rows (context = a fixed inner-fit subsample). The outer fold is
scored only twice, before and after fine-tuning, with the whole training fold as context (paired comparison).
Usage: python kumo_finetune.py [--steps 400] [--lr 1e-5] [--ctx 2048] [--q 512] [--train icl|all]"""
import argparse
import copy
import time

import numpy as np
import polars as pl
import torch

import sdm
import valid2 as v
from common import OUT, features

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--q", type=int, default=512)
ap.add_argument("--train", default="all", choices=["all", "icl"])
ap.add_argument("--eval_every", type=int, default=25)
ap.add_argument("--patience", type=int, default=6)
ap.add_argument("--skip_base", action="store_true")
a = ap.parse_args()

X, Xt, y, ids, tids = v.frames(features())
tr, va = v.outer(y, 42)[0]
fi, st = v.inner(y[tr], 42, 0)
fi, st = tr[fi], tr[st]
cols = v.fm_select(X, y, tr, 100)
dev = torch.device("cuda")
F = X[cols].copy()
for c in v.CAT_COLS:
    F[c] = F[c].cat.codes.astype(float).replace(-1, np.nan)
F = F.astype(float).reset_index(drop=True).assign(target=y.astype(str))
stypes = sdm.infer_stypes(F, overrides={"target": "categorical", **{c: "numerical" for c in cols}})
T = sdm.TableTensor.from_pandas(df=F, stypes=stypes, device=dev)
model = sdm.models.KumoTabular(task="classification", size="large", device=dev)
rng = np.random.default_rng(0)


def strat(pool, n, r):
    return np.sort(np.concatenate([r.choice(pool[y[pool] == c], int(round(n * (y[pool] == c).mean())), replace=False) for c in (0, 1)]))


def rows(idx):
    return T[torch.as_tensor(idx, device=dev)]


def p1(out):
    """P(target == "1"). The class column order of out.numerical changes between calls (Kumo draws a class order per
    call), so index by label name: numerical[..., 1] mixed P(0) and P(1) across chunks -> pooled KS 0.298 (bug 18 Okt)."""
    return out.numerical[..., out.columns[sdm.Stype.numerical].index("1")]


def predict(ctx, q, n_est, seed=0, chunk=1000):
    model.eval()
    c = rows(ctx)
    xc, yc = c.drop_columns("target"), c[:, "target"]
    g = torch.Generator(device=dev).manual_seed(seed)
    out = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(q), chunk):
            r = model(x_context=xc, y_context=yc, x_query=rows(q[i:i + chunk]).drop_columns("target"), num_estimators=n_est, generator=g)
            out.append(p1(r).float().cpu().numpy())
    return np.concatenate(out)


t0 = time.time()
BASE = OUT / "kumo_finetune_f0_base.parquet"  # untouched model, context = whole training fold, e4, generator seed 0
if a.skip_base:
    base = None
elif BASE.exists():
    base = pl.read_parquet(BASE)["base"].to_numpy()
else:
    base = predict(tr, va, 4)
    pl.DataFrame({"application_id": ids[va], "base": base}).write_parquet(BASE)
if base is not None:
    print(f"before fine-tuning: outer fold-0 KS {v.ks(y[va], base):.5f} ({time.time() - t0:.0f} s)", flush=True)
mon_ctx = strat(fi, 16000, np.random.default_rng(1))
mon = lambda: v.ks(y[st], predict(mon_ctx, st, 1, seed=1))  # noqa: E731
best_ks, best_state, bad = mon(), None, 0
print(f"step 0 inner-stop KS {best_ks:.5f}", flush=True)

params = [p for n, p in model.named_parameters() if a.train == "all" or "icl_block" in n]
for n, p in model.named_parameters():
    p.requires_grad_(a.train == "all" or "icl_block" in n)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda")
gen = torch.Generator(device=dev).manual_seed(0)
losses = []
for step in range(1, a.steps + 1):
    model.train()
    ctx = strat(fi, a.ctx, rng)
    q = rng.choice(np.setdiff1d(fi, ctx), a.q, replace=False)
    c = rows(ctx)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = model(x_context=c.drop_columns("target"), y_context=c[:, "target"], x_query=rows(q).drop_columns("target"),
                    num_estimators=1, generator=gen)
    pr = p1(out).float().clamp(1e-6, 1 - 1e-6)
    yt = torch.as_tensor(y[q], device=dev, dtype=torch.float32)
    loss = torch.nn.functional.binary_cross_entropy(pr, yt)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt)
    scaler.update()
    losses.append(loss.item())
    if step % a.eval_every == 0:
        ks = mon()
        print(f"step {step} loss {np.mean(losses[-a.eval_every:]):.4f} inner-stop KS {ks:.5f} "
              f"VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB {time.time() - t0:.0f} s", flush=True)
        if ks > best_ks:
            best_ks, bad = ks, 0
            best_state = {k: t.detach().to("cpu", copy=True) for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
if best_state is None:
    print("no improvement on the inner stop rows; outer fold not re-scored")
else:
    model.load_state_dict(best_state)
    after = predict(tr, va, 4)
    print(f"after fine-tuning (best inner-stop KS {best_ks:.5f}): outer fold-0 KS {v.ks(y[va], after):.5f}", flush=True)
    if base is not None:
        r = np.random.default_rng(0)
        d = []
        for _ in range(400):
            i = r.choice(len(va), len(va))
            d.append(v.ks(y[va][i], after[i]) - v.ks(y[va][i], base[i]))
        d = np.array(d)
        print(f"after - before: {d.mean():+.5f} sd {d.std():.5f} P>0 {(d > 0).mean():.2f}")
    pl.DataFrame({"application_id": ids[va], "before": base if base is not None else np.nan, "after": after}).write_parquet(
        OUT / f"kumo_finetune_f0_{a.train}_lr{a.lr:g}.parquet")
