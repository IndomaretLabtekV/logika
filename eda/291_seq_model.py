"""H2: event-set model on raw sequences (290 cache) — a member that sees events, not aggregates.

Per table: event MLP (d -> 64 -> 64), pooled by masked mean, masked max and attention, plus has-events and log count.
Five tables + static MLP -> head MLP -> logit. BCE, AdamW, early stopping on the valid2 inner split (AUC), no refit.
Folds = valid2.outer(y, seed); test = mean over folds. Then: KS alone, Spearman with eight_seg, rank blend with
eight_seg at several weights judged per fold over 3 split seeds and by paired bootstrap.
Usage: cd eda && ../.venv/bin/python 291_seq_model.py [seed] [epochs] [full]
"full" (H2b): the static input also gets every engineered feature (features + f251), rank-gauss over train + test
(label-free) with a missing flag per column that has missing values.
"""
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import valid2 as v

R = Path(__file__).resolve().parent.parent
OUT = R / "results/v73/seq_model"
TABLES = ["ec", "rp", "bu", "ac", "pv"]
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class TableEnc(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.att = nn.Linear(h, 1)
        self.out = 3 * h + 2

    def forward(self, x, m):
        e = self.mlp(x)
        mf = m.unsqueeze(-1).float()
        cnt = mf.sum(1).clamp(min=1)
        mean = (e * mf).sum(1) / cnt
        mx = e.masked_fill(~m.unsqueeze(-1), -1e4).max(1).values
        has = m.any(1, keepdim=True)
        mx = torch.where(has, mx, torch.zeros_like(mx))
        a = self.att(e).squeeze(-1).masked_fill(~m, -1e4)
        w = torch.softmax(a, 1) * m.float()
        w = w / (w.sum(1, keepdim=True) + 1e-9)
        att = (e * w.unsqueeze(-1)).sum(1)
        return torch.cat([mean, mx, att, has.float(), torch.log1p(m.sum(1, keepdim=True).float())], -1)


class Net(nn.Module):
    def __init__(self, dims, d_static):
        super().__init__()
        self.enc = nn.ModuleDict({t: TableEnc(d) for t, d in dims.items()})
        self.st = nn.Sequential(nn.Linear(d_static, 64), nn.GELU())
        d = sum(e.out for e in self.enc.values()) + 64
        self.head = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, batch):
        z = [self.enc[t](batch[t], batch[t + "_m"]) for t in TABLES] + [self.st(batch["st"])]
        return self.head(torch.cat(z, -1)).squeeze(-1)


def engineered(app_ids):
    from scipy.special import erfinv
    import common as c
    f = (c.features().join(pl.read_parquet(c.PQ / "f251_repayment_markov.parquet"), on="application_id", how="left")
         .sort("application_id"))
    assert (f["application_id"].to_numpy().astype(str) == app_ids).all()
    cols = [k for k, d in f.schema.items() if d.is_numeric() and k not in v.ID_COLS]
    out = []
    for k in cols:
        x = f[k].cast(pl.Float64).to_numpy()
        ok = np.isfinite(x)
        if ok.sum() < 100 or np.unique(x[ok]).size < 2:
            continue
        g = np.zeros(len(x))
        g[ok] = np.sqrt(2) * erfinv(2 * (rankdata(x[ok]) - 0.5) / ok.sum() - 1)
        out.append(g)
        if not ok.all():
            out.append((~ok).astype(float))
    return np.column_stack(out).astype(np.float32)


def take(data, idx):
    return {k: val[idx] for k, val in data.items()}


def predict(net, data, idx, bs=4096):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            out.append(net(take(data, idx[i:i + bs])).float().cpu().numpy())
    return np.concatenate(out)


def fit_fold(data, y_t, tr_idx, st_idx, seed, epochs, dims, d_static):
    torch.manual_seed(seed)
    net = Net(dims, d_static).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    y_st = y_t[st_idx].cpu().numpy()
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tr_idx)
        for i in range(0, len(perm), 1024):
            b = perm[i:i + 1024]
            opt.zero_grad()
            loss = lossf(net(take(data, b)), y_t[b])
            loss.backward()
            opt.step()
        sched.step()
        auc = roc_auc_score(y_st, predict(net, data, st_idx))
        if auc > best:
            best, bad = auc, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                break
    net.load_state_dict(best_state)
    return net, best, ep + 1


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    full = len(sys.argv) > 3 and sys.argv[3] == "full"
    z = np.load(R / "results/v73/seq_cache/seq.npz")
    is_tr = z["is_train"]
    data_np = {k: z[k] for k in TABLES + [t + "_m" for t in TABLES] + ["st"]}
    if full:
        data_np["st"] = np.hstack([data_np["st"], engineered(z["application_id"])])
        print("static input with engineered features:", data_np["st"].shape, flush=True)
    tag = "seqfull" if full else "seq"
    data = {k: torch.from_numpy(val).to(DEV) for k, val in data_np.items()}
    dims = {t: data_np[t].shape[2] for t in TABLES}
    tr_rows, te_rows = np.where(is_tr)[0], np.where(~is_tr)[0]
    ids = z["application_id"][tr_rows]
    yd = pl.read_parquet(R / "results/v72k_out/v72/preds/oof_lgb.parquet")
    assert (yd["application_id"].to_numpy().astype(str) == ids).all()
    y = yd["target"].to_numpy()
    y_all = np.zeros(len(is_tr), dtype=np.float32)
    y_all[tr_rows] = y
    y_t = torch.from_numpy(y_all).to(DEV)
    oof, test = np.zeros(len(y)), np.zeros(len(te_rows))
    for k, (tr, va) in enumerate(v.outer(y, seed)):
        t0 = time.time()
        fi, st = v.inner(y[tr], seed, k)
        net, auc, ep = fit_fold(data, y_t, tr_rows[tr[fi]], tr_rows[tr[st]], seed * 10 + k, epochs, dims, data_np["st"].shape[1])
        oof[va] = predict(net, data, tr_rows[va])
        test += rankdata(predict(net, data, te_rows)) / len(te_rows) / 5
        print(f"fold {k}: inner AUC {auc:.4f} epochs {ep} KS {v.ks(y[va], oof[va]):.5f} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"application_id": ids, "target": y, tag: oof}).write_parquet(OUT / f"oof_{tag}_s{seed}.parquet")
    pl.DataFrame({"application_id": z["application_id"][te_rows], tag: test}).write_parquet(OUT / f"test_{tag}_s{seed}.parquet")
    print(f"{tag} model seed {seed}: OOF KS {v.ks(y, oof):.5f}")

    fb = importlib.import_module("281_final_blend")
    bs = importlib.import_module("275_blend_scheme")
    ss = pl.read_csv(R / "data/sample_submission.csv").select("application_id")
    e8s = np.mean([rankdata(fb.member(m, yd["application_id"], ss)[0]) / len(y) for m in fb.SIX + ["lgb_x", "xgb_d8", "lgb_seg", "xgb_seg"]], 0)
    r = rankdata(oof) / len(y)
    print(f"spearman seq vs eight_seg {np.corrcoef(rankdata(r), rankdata(e8s))[0, 1]:.4f}")
    for w in (0.1, 0.2, 0.3):
        b = (1 - w) * e8s + w * r
        mu, pp = bs.boot(y, b, e8s, n=500, seed=1)
        d = [v.ks(y[va], b[va]) - v.ks(y[va], e8s[va]) for sd in (1, 2, 3) for _, va in v.outer(y, sd)]
        print(f"   w {w:.1f}: KS {v.ks(y, b):.5f} vs eight_seg {v.ks(y, e8s):.5f} boot {mu:+.5f} P>0 {pp:.2f} folds won {sum(x > 0 for x in d)}/15")


if __name__ == "__main__":
    main()
