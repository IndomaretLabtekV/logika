"""Hybrid event-set network: hand features plus attention-pooled encoders over raw table rows.
Execution is restricted to Kaggle CUDA; the event builder is pure polars/numpy."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

SEQ = dict(ec=128, rp=64, ac=24, bu=16, pv=16)
EC_CATS = ["groceries", "fashion", "electronics", "home", "health", "education", "personal_care"]
PAY = ["COD", "e_wallet", "bank_transfer", "debit_card", "credit_card"]
PROV = ["Bureau_A", "Bureau_B", "Bureau_C"]
DEC = ["approved", "rejected", "cancelled"]
SEEDS = (42, 7, 2026)
VARIANTS = ("hybrid", "events")


def ks(y, p):
    assert len(y) == len(p) and np.isfinite(p).all()
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def splits(y, n, seed):
    rng = np.random.default_rng(seed)
    parts = [np.array_split(rng.permutation(np.flatnonzero(y == c)), n) for c in (0, 1)]
    for k in range(n):
        valid = np.sort(np.concatenate([p[k] for p in parts]))
        train = np.setdiff1d(np.arange(len(y)), valid)
        yield train, valid


def slog(c):
    return pl.col(c).sign() * pl.col(c).abs().log1p()


def days(a, b):
    return (pl.col(a) - pl.col(b)).dt.total_seconds() / 86400.0


def pad(df, order, L, num_cols, cat_cols):
    """df rows already filtered and sorted newest first per customer -> [N, L, *] arrays + mask."""
    df = df.with_columns(pos=pl.int_range(pl.len()).over("customer_id")).filter(pl.col("pos") < L)
    df = df.join(order, on="customer_id")
    r, p = df["row"].to_numpy(), df["pos"].to_numpy()
    N = order.height
    num = np.zeros((N, L, len(num_cols)), np.float32)
    cat = np.zeros((N, L, max(len(cat_cols), 1)), np.int64)
    mask = np.zeros((N, L), bool)
    vals = df.select(num_cols).fill_nan(0).fill_null(0).to_numpy().astype(np.float32)
    assert np.isfinite(vals).all()
    # unsupervised per-column standardisation over all observed rows (no labels involved)
    vals = np.clip((vals - vals.mean(0)) / (vals.std(0) + 1e-6), -6, 6)
    num[r, p] = vals
    if cat_cols:
        cat[r, p] = df.select(cat_cols).to_numpy()
    mask[r, p] = True
    return num, cat, mask


def build_events(tables, apps):
    """apps: customer_id, application_id, decision_datetime in model row order. Same cutoff rules as features.py."""
    order = apps.select("customer_id", "application_id").with_row_index("row").with_columns(pl.col("row").cast(pl.Int64))
    cut = apps.select("customer_id", sel_app="application_id", cutoff="decision_datetime",
                      dm=pl.col("decision_datetime").dt.truncate("1mo"))
    out = {}
    # e-commerce items, price relative to an expanding monthly category reference (as v5_features)
    raw = tables["ecommerce_transactions"].filter(pl.col("quantity") > 0).with_columns(
        lu=(pl.col("item_total") / pl.col("quantity")).log(), ym=pl.col("transaction_datetime").dt.truncate("1mo"))
    ms = raw.group_by("product_category", "ym").agg(n=pl.len(), s=pl.col("lu").sum(), ss=(pl.col("lu") ** 2).sum())
    ref = (cut.select("dm").unique().join(ms, how="cross").filter(pl.col("ym") < pl.col("dm"))
           .group_by("product_category", "dm").agg(pl.col("n", "s", "ss").sum())
           .with_columns(mu=pl.col("s") / pl.col("n"),
                         sd=((pl.col("ss") - pl.col("s") ** 2 / pl.col("n")) / (pl.col("n") - 1)).sqrt()))
    e = raw.join(cut, on="customer_id").filter(pl.col("transaction_datetime") <= pl.col("cutoff"))
    e = e.join(ref.select("product_category", "dm", "mu", "sd"), on=["product_category", "dm"], how="left")
    e = e.with_columns(
        z=((pl.col("lu") - pl.col("mu")) / pl.col("sd")).fill_null(0),
        age=days("cutoff", "transaction_datetime"),
        lit=(pl.col("item_total").log1p() - 11) / 2, lot=(pl.col("order_total").log1p() - 13) / 2,
        hr=pl.col("transaction_datetime").dt.hour() / 23, dw=pl.col("transaction_datetime").dt.weekday() / 7,
        q=pl.col("quantity").cast(pl.Float64),
        canc=(pl.col("order_status") == "cancelled").cast(pl.Float64),
        c_cat=pl.col("product_category").replace_strict(EC_CATS, range(1, 8), default=0),
        c_pay=pl.col("payment_method").replace_strict(PAY, range(1, 6), default=0),
    ).with_columns(la=pl.col("age").log1p() / 6, a1=pl.col("age") / 365)
    e = e.sort(["customer_id", "transaction_datetime", "order_item_id"], descending=[False, True, False])
    out["ec"] = pad(e, order, SEQ["ec"], ["z", "lit", "lot", "hr", "dw", "q", "canc", "la", "a1"], ["c_cat", "c_pay"]), [8, 6]
    # repayment installments
    r = tables["repayment_history"].join(cut, on="customer_id").filter(pl.col("due_date") <= pl.col("cutoff"))
    r = r.sort(["customer_id", "loan_id", "due_date"]).with_columns(
        k=pl.int_range(pl.len()).over("loan_id"), nl=pl.len().over("loan_id"))
    r = r.with_columns(
        dpd=pl.col("days_past_due").log1p() / 4, mdpd=pl.col("max_days_past_due").log1p() / 4,
        pf=(pl.col("amount_paid") / pl.col("amount_due")).clip(0, 1.5),
        unpaid=(pl.col("amount_paid") < pl.col("amount_due")).cast(pl.Float64),
        late=(days("last_payment_datetime", "due_date") / 30).clip(-3, 6),
        nolp=pl.col("last_payment_datetime").is_null().cast(pl.Float64),
        age=days("cutoff", "due_date"), amt=(pl.col("amount_due").log1p() - 14) / 2,
        kk=pl.col("k") / 12, last=(pl.col("k") == pl.col("nl") - 1).cast(pl.Float64),
    ).with_columns(la=pl.col("age").log1p() / 6, a1=pl.col("age") / 365)
    r = r.sort(["customer_id", "due_date", "loan_id"], descending=[False, True, False])
    out["rp"] = pad(r, order, SEQ["rp"], ["dpd", "mdpd", "pf", "unpaid", "late", "nolp", "amt", "kk", "last", "la", "a1"], []), []
    # account months (complete months only)
    a = tables["account_behavior"].with_columns(m=pl.col("month").cast(pl.Datetime("us"))).join(cut, on="customer_id")
    a = a.filter(pl.col("m").dt.offset_by("1mo") <= pl.col("cutoff"))
    mon = ["opening_balance", "closing_balance", "inflows", "outflows", "income_like_credits", "average_balance", "minimum_balance"]
    a = a.with_columns(*[(slog(c) / 16).alias(f"s_{c}") for c in mon],
                       lbd=pl.col("low_balance_days") / 30, am=days("cutoff", "m") / 365,
                       net=slog("inflows") / 16 - slog("outflows") / 16)
    a = a.sort(["customer_id", "m"], descending=[False, True])
    out["ac"] = pad(a, order, SEQ["ac"], [f"s_{c}" for c in mon] + ["lbd", "am", "net"], []), []
    # bureau observations available at the cutoff
    b = tables["bureau_observations"].join(cut, on="customer_id").filter(pl.col("available_datetime") <= pl.col("cutoff"))
    b = b.with_columns(
        sc=((pl.col("score") - 650) / 100).fill_null(0), nosc=pl.col("score").is_null().cast(pl.Float64),
        act=pl.col("active_accounts") / 5, out=(pl.col("outstanding_balance").log1p() - 10) / 6,
        rl=(pl.col("revolving_credit_limit").log1p() - 8) / 8, ro=(pl.col("revolving_outstanding_balance").log1p() - 8) / 8,
        ut=pl.col("utilization").fill_null(0).clip(0, 3), nout=pl.col("utilization").is_null().cast(pl.Float64),
        inq=pl.col("inquiries_last_6m") / 5, dlq=pl.col("delinquent_accounts_last_12m") / 3,
        age=days("cutoff", "available_datetime"), dly=days("available_datetime", "report_datetime") / 30,
        own=(pl.col("application_id") == pl.col("sel_app")).cast(pl.Float64),
        c_prov=pl.col("bureau_provider").replace_strict(PROV, range(1, 4), default=0),
    ).with_columns(la=pl.col("age").log1p() / 6)
    b = b.sort(["customer_id", "available_datetime", "bureau_observation_id"], descending=[False, True, False])
    out["bu"] = pad(b, order, SEQ["bu"], ["sc", "nosc", "act", "out", "rl", "ro", "ut", "nout", "inq", "dlq", "la", "dly", "own"], ["c_prov"]), [4]
    # previous applications decided before the cutoff
    p = tables["previous_applications"].join(cut, on="customer_id").filter(pl.col("decision_datetime") <= pl.col("cutoff"))
    p = p.with_columns(
        req=(pl.col("requested_amount").log1p() - 16) / 2,
        apr=(pl.col("approved_amount") / pl.col("requested_amount")).fill_null(0),
        rate=pl.col("annual_interest_rate").fill_null(0) * 3, term=pl.col("term_months") / 12,
        inst=((pl.col("scheduled_monthly_installment").log1p() - 14) / 2).fill_null(0),
        age=days("cutoff", "decision_datetime"), lag=days("decision_datetime", "application_datetime") / 3,
        c_dec=pl.col("decision").replace_strict(DEC, range(1, 4), default=0),
    ).with_columns(la=pl.col("age").log1p() / 6, a1=pl.col("age") / 365)
    p = p.sort(["customer_id", "decision_datetime", "application_id"], descending=[False, True, False])
    out["pv"] = pad(p, order, SEQ["pv"], ["req", "apr", "rate", "term", "inst", "la", "a1", "lag"], ["c_dec"]), [4]
    return out


class SetEncoder(nn.Module):
    def __init__(self, n_num, cards, d=64):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(c, 8) for c in cards)
        self.row = nn.Sequential(nn.Linear(n_num + 8 * len(cards), d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.att = nn.Linear(d, 4)
        self.d_out = d * 4 + d + d + 1

    def forward(self, num, cat, mask):
        x = torch.cat([num] + [e(cat[..., i]) for i, e in enumerate(self.embs)], -1)
        h = self.row(x)
        m = mask.unsqueeze(-1)
        a = self.att(h).masked_fill(~m, -1e4)
        w = torch.softmax(a, 1) * m
        att = torch.einsum("blk,bld->bkd", w, h).flatten(1)
        n = mask.sum(1, keepdim=True).float()
        mean = (h * m).sum(1) / n.clamp(min=1)
        mx = h.masked_fill(~m, -1e4).max(1).values * (n > 0)
        return torch.cat([att, mean, mx, torch.log1p(n) / 5], -1)


class Hybrid(nn.Module):
    def __init__(self, n_hand, hand_cards, ev_specs, use_hand=True, d=256, drop=.2):
        super().__init__()
        self.use_hand = use_hand
        self.encs = nn.ModuleDict({k: SetEncoder(n, cards) for k, (n, cards) in ev_specs.items()})
        width = sum(e.d_out for e in self.encs.values())
        if use_hand:
            self.hembs = nn.ModuleList(nn.Embedding(c, 8) for c in hand_cards)
            self.hand = nn.Sequential(nn.Linear(n_hand + 8 * len(hand_cards), d), nn.GELU(), nn.Dropout(drop))
            width += d
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Dropout(drop), nn.Linear(width, d), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

    def forward(self, hn, hc, ev):
        parts = [self.encs[k](*ev[k]) for k in self.encs]
        if self.use_hand:
            parts.append(self.hand(torch.cat([hn] + [e(hc[:, i]) for i, e in enumerate(self.hembs)], -1)))
        return self.head(torch.cat(parts, -1)).squeeze(-1)


def fit_predict(variant, seed, hand_n, hand_c, ev, y, fit, stop, pred_sets, out, tag, epochs=40, patience=6):
    """Train on fit rows, early stop on stop rows (AUC), return predictions for each index array in pred_sets."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda"
    specs = {k: (v[0].shape[-1], cards) for k, (v, cards) in ev.items()}
    hand_cards = [int(hand_c[:, i].max()) + 1 for i in range(hand_c.shape[1])]
    model = Hybrid(hand_n.shape[1], hand_cards, specs, use_hand=variant == "hybrid").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)

    def batch(idx):
        i = torch.as_tensor(idx, device=dev)
        return (hand_n[i], hand_c[i], {k: tuple(a[i] for a in v) for k, (v, _) in ev.items()}), i

    def predict(idx):
        model.eval()
        res = []
        with torch.no_grad():
            for s in range(0, len(idx), 2048):
                xb, _ = batch(idx[s:s + 2048])
                res.append(torch.sigmoid(model(*xb)).float().cpu().numpy())
        return np.concatenate(res)

    best, best_state, wait, curve = -1.0, None, 0, []
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(fit)
        tot = 0.0
        for s in range(0, len(perm), 512):
            xb, i = batch(perm[s:s + 512])
            yb = yt[i]
            loss = F.binary_cross_entropy_with_logits(model(*xb), yb)
            assert torch.isfinite(loss)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(yb)
        ps = predict(stop)
        auc = roc_auc_score(y[stop], ps)
        curve.append(dict(epoch=ep, loss=tot / len(fit), auc=auc, ks=ks(y[stop], ps)))
        print(tag, curve[-1], flush=True)
        if auc > best:
            best, wait = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    pd.DataFrame(curve).to_csv(out / f"curve_{tag}.csv", index=False)
    model.load_state_dict(best_state)
    preds = [predict(ix) for ix in pred_sets]
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return preds


def main_v14(build_all, nn_prep):
    assert Path("/kaggle/input").exists() and torch.cuda.is_available(), "Kaggle CUDA only"
    data = next(Path("/kaggle/input").rglob("train_applications.csv")).parent
    out = Path("/kaggle/working/v14")
    out.mkdir(exist_ok=True)
    names = ["train_applications", "test_applications", "customer_profiles", "previous_applications",
             "repayment_history", "account_behavior", "bureau_observations", "ecommerce_transactions"]
    tables = {n: pl.read_csv(data / f"{n}.csv", try_parse_dates=True, infer_schema_length=100000) for n in names}
    features = build_all(tables).sort("application_id")
    ntr = int((features["is_train"] == 1).sum())
    features = pl.concat([features.filter(pl.col("is_train") == 1), features.filter(pl.col("is_train") == 0)])
    t0 = time.monotonic()
    ev = build_events(tables, features.select("customer_id", "application_id", "decision_datetime"))
    print("events built in", round(time.monotonic() - t0), "s;",
          {k: (v[0][0].shape, float(v[0][2].sum(1).mean())) for k, v in ev.items()}, flush=True)
    del tables
    gc.collect()
    ev = {k: (tuple(torch.as_tensor(a).cuda() for a in v), cards) for k, (v, cards) in ev.items()}
    y = features["target"].to_numpy()[:ntr].astype(int)
    cols = [c for c in features.columns if c not in {"application_id", "customer_id", "decision_datetime", "target", "is_train"}]
    cats = [c for c in cols if features[c].dtype == pl.String]
    nums = [c for c in cols if c not in cats]
    Xnum = features.select(nums).to_numpy().astype(float)
    hc = np.column_stack([features[c].fill_null("__na").cast(pl.Categorical).to_physical().to_numpy().astype(np.int64) + 1
                          for c in cats])
    N = features.height
    test_idx = np.arange(ntr, N)
    oof = {v: np.zeros(ntr) for v in VARIANTS}
    tst = {v: np.zeros(N - ntr) for v in VARIANTS}
    folds = np.full(ntr, -1)
    deadline = time.monotonic() + 10.5 * 3600
    for fold, (train, valid) in enumerate(splits(y, 5, 42)):
        # nn_prep fitted on the training rows of this fold only
        a, b, _ = nn_prep(Xnum[train], Xnum[np.r_[valid, test_idx]])
        hn = np.zeros((N, a.shape[1]), np.float32)
        hn[train], hn[np.r_[valid, test_idx]] = a, b
        hn = torch.as_tensor(hn).cuda()
        hct = torch.as_tensor(hc).cuda()
        for variant in VARIANTS:
            for si, seed in enumerate(SEEDS):
                if time.monotonic() > deadline:
                    raise TimeoutError("deadline")
                fit, stop = next(splits(y[train], 10, seed + fold))
                pv, pt = fit_predict(variant, seed, hn, hct, ev, y, train[fit], train[stop], [valid, test_idx],
                                     out, f"{variant}_f{fold}_s{seed}")
                oof[variant][valid] += pv / len(SEEDS)
                tst[variant] += pt / len(SEEDS) / 5
            print(f"fold {fold} {variant} KS {ks(y[valid], oof[variant][valid]):.5f}", flush=True)
        folds[valid] = fold
        np.savez(out / f"fold{fold}.npz", valid=valid, **{v: oof[v][valid] for v in VARIANTS})
    ids = features["application_id"]
    pl.DataFrame({"application_id": ids[:ntr], "target": y, "fold": folds, **oof}).write_parquet(out / "oof.parquet")
    pl.DataFrame({"application_id": ids[ntr:], **tst}).write_parquet(out / "test.parquet")
    res = {v: ks(y, oof[v]) for v in VARIANTS}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(res)
    for v in VARIANTS:
        pl.DataFrame({"application_id": ids[ntr:], "default_probability": tst[v]}).write_csv(out / f"submission_{v}.csv")
