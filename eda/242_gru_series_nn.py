"""AmEx 1st-place style NN: GRU over each customer's pre-decision event sequences (ecommerce items, repayment
installments; inputs from 240's label-free prep) + MLP over the 232 main features (quantile-normal, fitted
on outer-train), concatenated head, BCE. Per outer fold (seed 42): early stopping on the inner 1/7 stop split
(validation_v2 inner), the stopped model scores the outer fold (no refit; declared).
Locked candidate before any score: per-fold 0.75*rank(lgb_base4) + 0.25*rank(GRU-NN) vs lgb_base4.
Usage: cd eda && ../.venv/bin/python 242_gru_series_nn.py
"""
import importlib.util
import os

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer

import valid2 as v
from common import ROOT

spec = importlib.util.spec_from_file_location("c240", ROOT / "eda/240_coles_embeddings.py")
c240 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c240)
OUT = ROOT / "results/v68/gru_nn"
torch.manual_seed(242)


class Net(nn.Module):
    def __init__(self, n_tab, n_e, n_r):
        super().__init__()
        self.enc_e, self.enc_r = c240.Encoder(n_e), c240.Encoder(n_r)
        self.tab = nn.Sequential(nn.Linear(n_tab, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Dropout(.3),
                                 nn.Linear(256, 128), nn.BatchNorm1d(128), nn.LeakyReLU())
        self.head = nn.Sequential(nn.Linear(128 + 2 * c240.HID, 128), nn.LeakyReLU(), nn.Dropout(.2), nn.Linear(128, 1))

    def seq(self, enc, x, c, lens):
        out = torch.zeros(len(lens), c240.HID, device=x.device)
        have = lens > 0
        if have.any():
            out[have] = enc(x[have], c[have], lens[have])
        return out

    def forward(self, t, xe, ce, le, xr, cr, lr):
        return self.head(torch.cat([self.tab(t), self.seq(self.enc_e, xe, ce, le), self.seq(self.enc_r, xr, cr, lr)], 1)).squeeze(1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    a = c240.apps()
    feats = pl.read_parquet(ROOT / "data_pq/features.parquet")
    X, _, y, ids, _ = v.frames(feats)
    order = {k: i for i, k in enumerate(a["application_id"].to_list())}
    cust = [a["customer_id"][order[k]] for k in ids.to_list()]
    customers = a["customer_id"].to_list()
    pos_of = {c: i for i, c in enumerate(customers)}
    rows = np.array([pos_of[c] for c in cust])
    seqs = {}
    for s in ("ecom", "repay"):
        e, num = c240.events(s, a)
        Xs, Cs, Ls = c240.pack(e, num, customers)
        seqs[s] = (Xs[rows], Cs[rows], Ls[rows])
    num_cols = [c for c in X.columns if c not in v.CAT_COLS]
    T = X[num_cols].to_numpy(float)
    oof = np.full(len(y), np.nan)
    for k, (tr, va) in enumerate(v.outer(y, 42)):
        path = OUT / f"oof_f{k}.npy"
        if path.exists():
            oof[va] = np.load(path)
            continue
        fi, st = v.inner(y[tr], 42, k)
        qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=242).fit(T[tr])
        Tq = np.nan_to_num(qt.transform(T), nan=0.0).astype(np.float32)
        miss = np.isnan(T).astype(np.float32)
        tab = np.concatenate([Tq, miss], 1)

        def batch(idx):
            out = [torch.from_numpy(tab[idx]).to(dev)]
            for s in ("ecom", "repay"):
                xs, cs, ls = seqs[s]
                m = max(int(ls[idx].max()), 1)
                out += [torch.from_numpy(xs[idx, :m]).to(dev), torch.from_numpy(cs[idx, :m]).to(dev), torch.from_numpy(ls[idx])]
            return out

        def predict(model, idx):
            model.eval(); res = []
            with torch.no_grad():
                for b in range(0, len(idx), 2048):
                    res.append(torch.sigmoid(model(*batch(idx[b:b + 2048]))).cpu().numpy())
            return np.concatenate(res)

        model = Net(tab.shape[1], seqs["ecom"][0].shape[2], seqs["repay"][0].shape[2]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        fit_idx, stop_idx = tr[fi], tr[st]
        best, best_state, bad = -1, None, 0
        rng = np.random.default_rng(242 + k)
        for ep in range(25):
            model.train()
            perm = rng.permutation(fit_idx)
            for b in range(0, len(perm), 512):
                idx = perm[b:b + 512]
                loss = nn.functional.binary_cross_entropy_with_logits(model(*batch(idx)), torch.from_numpy(y[idx].astype(np.float32)).to(dev))
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            ks = v.ks(y[stop_idx], predict(model, stop_idx))
            print(f"fold {k} epoch {ep} inner-stop KS {ks:.5f}", flush=True)
            if ks > best:
                best, bad = ks, 0
                best_state = {n: p.detach().clone() for n, p in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 4:
                    break
        model.load_state_dict(best_state)
        p = predict(model, va)
        tmp = path.with_name(path.name + ".tmp.npy")
        np.save(tmp, p)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        tmp.replace(path)
        oof[va] = p
        print(f"fold {k} outer KS {v.ks(y[va], p):.5f} (inner best {best:.5f})", flush=True)
    base = pl.read_parquet(ROOT / "results/v66/series_oof_eval/oof_lgb_base4.parquet")["lgb_base4"].to_numpy()
    blend = np.empty(len(y))
    for _, va in v.outer(y, 42):
        blend[va] = .75 * rankdata(base[va]) / len(va) + .25 * rankdata(oof[va]) / len(va)
    rng = np.random.default_rng(242)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = [v.ks(y[r], blend[r]) - v.ks(y[r], base[r]) for r in (np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(500))]
    print(f"GRU-NN pooled KS {v.ks(y, oof):.5f} AUC {roc_auc_score(y, oof):.5f}")
    print(f"blend 0.75 LGB + 0.25 NN: {v.ks(y, blend):.5f} vs LGB {v.ks(y, base):.5f} delta {v.ks(y, blend) - v.ks(y, base):+.5f} "
          f"ci95 {np.quantile(d, [.025, .975]).round(5).tolist()} P>0 {np.mean(np.array(d) > 0):.3f} "
          f"folds won {sum(v.ks(y[va], blend[va]) > v.ks(y[va], base[va]) for _, va in v.outer(y, 42))}/5")
    pl.DataFrame({"application_id": ids, "target": y, "gru_nn": oof}).write_parquet(OUT / "oof_gru_nn.parquet")


if __name__ == "__main__":
    main()
