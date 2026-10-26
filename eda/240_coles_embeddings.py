"""CoLES-style self-supervised event-sequence embeddings (Babaev et al., SIGMOD'22, arXiv 2002.08232).
A GRU encodes each customer's pre-decision event sequence; training uses no labels: 5 random contiguous
slices per customer, supervised-contrastive loss with slices of the same customer as positives.
Trained on all 100k customers' covariates (train + test, label-free SSL, allowed by the organisers).
Output (atomic): results/v67/coles/<source>_emb.parquet = application_id + PCA-32 of the full-sequence
embedding (NaN for customers without events); one identical copy per outer fold as
results/v66/series_oof/coles_<source>_f<k>.parquet so 238 can evaluate it.
Usage: cd eda && ../.venv/bin/python 240_coles_embeddings.py <ecom|repay> [epochs]
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from common import ROOT

OUT = ROOT / "results/v67/coles"
L, HID, SLICES, BATCH, TEMP = 256, 128, 5, 256, 0.1
torch.manual_seed(240)
rng = np.random.default_rng(240)


def apps():
    cols = ["application_id", "customer_id", "decision_datetime"]
    return pl.concat([pl.read_csv(ROOT / f"data/{s}_applications.csv", try_parse_dates=True).select(cols) for s in ("train", "test")])


def events(source, a):
    if source == "ecom":
        e = pl.read_csv(ROOT / "data/ecommerce_transactions.csv", try_parse_dates=True).rename({"transaction_datetime": "t"})
        e = e.join(a.select("customer_id", "decision_datetime"), on="customer_id").filter(pl.col("t") < pl.col("decision_datetime"))
        e = e.with_columns(unit=(pl.col("item_total") / pl.col("quantity").clip(1)).log(), month=pl.col("t").dt.strftime("%Y-%m"))
        e = e.with_columns(rel=pl.col("unit") - pl.col("unit").median().over("product_category", "month"),
                           items=pl.len().over("order_id").log1p(), qty=pl.col("quantity").log1p(),
                           tot=pl.col("order_total").clip(1).log(), cancel=(pl.col("order_status") == "cancelled").cast(pl.Float32),
                           c1=pl.col("product_category").cast(pl.Categorical).to_physical().cast(pl.Int64) + 1,
                           c2=pl.col("payment_method").cast(pl.Categorical).to_physical().cast(pl.Int64) + 1)
        num = ["unit", "rel", "items", "qty", "tot", "cancel"]
    else:
        e = pl.read_csv(ROOT / "data/repayment_history.csv", try_parse_dates=True).rename({"due_date": "t"})
        e = e.join(a.select("customer_id", "decision_datetime"), on="customer_id").filter(pl.col("t") < pl.col("decision_datetime"))
        e = e.sort("loan_id", "t").with_columns(
            amt=pl.col("amount_due").clip(1).log(), paid=(pl.col("amount_paid") / pl.col("amount_due").clip(1)).clip(0, 2),
            dpd=pl.col("days_past_due").clip(0).log1p(), mdpd=pl.col("max_days_past_due").clip(0).log1p(),
            delay=((pl.col("last_payment_datetime") - pl.col("t")).dt.total_seconds() / 86400).fill_null(0).clip(-60, 120),
            ino=pl.int_range(pl.len()).over("loan_id").cast(pl.Float32),
            c1=pl.col("loan_id").rank("dense").over("customer_id").clip(0, 15).cast(pl.Int64), c2=pl.lit(1, pl.Int64))
        num = ["amt", "paid", "dpd", "mdpd", "delay", "ino"]
    e = e.sort("customer_id", "t").with_columns(
        gap=(pl.col("t").diff().over("customer_id").dt.total_seconds() / 86400).fill_null(0).clip(0).log1p(),
        age=((pl.col("decision_datetime") - pl.col("t")).dt.total_seconds() / 86400).clip(0).log1p())
    num = num + ["gap", "age"]
    for c in num:  # global standardisation of covariates (label-free)
        e = e.with_columns(((pl.col(c) - pl.col(c).mean()) / (pl.col(c).std() + 1e-6)).fill_null(0).fill_nan(0).alias(c))
    return e, num


def pack(e, num, customers):
    """Last L events per customer -> dense arrays; lengths 0 for customers without events."""
    e = e.with_columns(rk=pl.int_range(pl.len()).over("customer_id").cast(pl.Int64),  # signed: n - L may be < 0
                       n=pl.len().over("customer_id").cast(pl.Int64))
    e = e.filter(pl.col("rk") >= pl.col("n") - L).with_columns(pos=pl.col("rk") - (pl.col("n") - L).clip(0))
    idx = {c: i for i, c in enumerate(customers)}
    ci = np.array([idx[c] for c in e["customer_id"].to_list()])
    pos = e["pos"].to_numpy()
    X = np.zeros((len(customers), L, len(num)), np.float32)
    C = np.zeros((len(customers), L, 2), np.int64)
    X[ci, pos] = e.select(num).to_numpy().astype(np.float32)
    C[ci, pos] = e.select("c1", "c2").to_numpy()
    lens = np.bincount(ci, minlength=len(customers)).clip(0, L)
    return X, C, lens


class Encoder(nn.Module):
    def __init__(self, n_num):
        super().__init__()
        self.e1, self.e2 = nn.Embedding(32, 8), nn.Embedding(16, 4)
        self.proj = nn.Sequential(nn.Linear(n_num + 12, HID), nn.LayerNorm(HID))
        self.gru = nn.GRU(HID, HID, batch_first=True)

    def forward(self, x, c, lens):
        h = self.proj(torch.cat([x, self.e1(c[..., 0]), self.e2(c[..., 1])], -1))
        packed = nn.utils.rnn.pack_padded_sequence(h, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, last = self.gru(packed)
        return last[-1]


def slices(X, C, lens, ids):
    xs, cs, ls, owner = [], [], [], []
    for j, i in enumerate(ids):
        n = lens[i]
        for _ in range(SLICES):
            m = rng.integers(max(1, min(5, n)), n + 1)
            s = rng.integers(0, n - m + 1)
            xs.append(X[i, s:s + m]); cs.append(C[i, s:s + m]); ls.append(m); owner.append(j)
    mx = max(ls)
    xb = np.zeros((len(xs), mx, X.shape[2]), np.float32); cb = np.zeros((len(xs), mx, 2), np.int64)
    for k, (a, b) in enumerate(zip(xs, cs)):
        xb[k, :len(a)], cb[k, :len(b)] = a, b
    return torch.from_numpy(xb), torch.from_numpy(cb), torch.tensor(ls), torch.tensor(owner)


def supcon(z, owner):
    z = nn.functional.normalize(z, dim=-1)
    sim = z @ z.T / TEMP
    eye = torch.eye(len(z), device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)
    pos = (owner[:, None] == owner[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    return -(logp * pos).sum(1).div(pos.sum(1)).mean()


def main():
    source, epochs = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12
    OUT.mkdir(parents=True, exist_ok=True)
    a = apps()
    customers = a["customer_id"].to_list()
    t0 = time.time()
    e, num = events(source, a)
    X, C, lens = pack(e, num, customers)
    print(source, "events", e.height, "customers with events", int((lens > 0).sum()), "prep", round(time.time() - t0), "s", flush=True)
    dev = torch.device("cuda")
    model = Encoder(len(num)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_ids = np.flatnonzero(lens >= 2)
    for ep in range(epochs):
        rng.shuffle(train_ids)
        tot, nb = 0.0, 0
        for b in range(0, len(train_ids) - BATCH + 1, BATCH):
            xb, cb, lb, ob = slices(X, C, lens, train_ids[b:b + BATCH])
            loss = supcon(model(xb.to(dev), cb.to(dev), lb), ob.to(dev))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        print(f"epoch {ep} loss {tot / nb:.4f} {round(time.time() - t0)} s", flush=True)
    model.eval()
    emb = np.full((len(customers), HID), np.nan, np.float32)
    with torch.no_grad():
        have = np.flatnonzero(lens > 0)
        for b in range(0, len(have), 2048):
            i = have[b:b + 2048]
            emb[i] = model(torch.from_numpy(X[i]).to(dev), torch.from_numpy(C[i]).to(dev), torch.from_numpy(lens[i])).cpu().numpy()
    ok = np.isfinite(emb).all(1)
    pc = np.full((len(customers), 32), np.nan, np.float32)
    pc[ok] = PCA(32, random_state=240).fit_transform(emb[ok])
    df = pl.DataFrame({"application_id": a["application_id"], **{f"coles_{source}_{j}": pc[:, j] for j in range(32)}})
    paths = [OUT / f"{source}_emb.parquet"] + [ROOT / f"results/v66/series_oof/coles_{source}_f{k}.parquet" for k in range(5)]
    for path in paths:  # unsupervised: identical features for every outer fold
        tmp = path.with_name(path.name + ".tmp")
        df.write_parquet(tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        tmp.replace(path)
    print("saved", [p.name for p in paths], round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
