"""Audit native categorical preprocessing and cached global-context Kumo large.

All recipe choices use disjoint inner search/confirmation partitions. Outer is only
scored after the recipe is frozen. No test predictions or test-fitted statistics.
The cached path holds one estimator ensemble fixed across query chunks; uncached
baseline re-draws preprocessing per call. That distinction is reported explicitly.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import sdm
import valid2 as v
from common import OUT, PQ, features

spec = importlib.util.spec_from_file_location("vipcop", Path(__file__).with_name("152_vipcop_kumo.py"))
vip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vip)




def representation(X, fit_rows, name, money_cols):
    mode = "raw" if name == "raw" else "categorical"
    F = vip.prepare(X, fit_rows, mode)
    if name == "money":
        for c in money_cols:
            x = F[c].to_numpy(dtype=float)
            F[c] = np.sign(x) * np.log1p(np.abs(x))
    return F


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", default="raw,categorical,gain,all,money")
    ap.add_argument("--money-cols", default="", help="Explicit audited column names, comma separated")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gpu-memory-fraction", type=float, default=.9)
    ap.add_argument("--nctx", type=int, default=20000)
    ap.add_argument("--nq", type=int, default=4000)
    ap.add_argument("--est", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--confirm-nctx", type=int, default=56000)
    ap.add_argument("--confirm-nq", type=int, default=4000)
    ap.add_argument("--confirm-est", type=int, default=4)
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    assert 1 <= a.threads <= 8
    torch.set_num_threads(a.threads)
    assert 0 < a.gpu_memory_fraction < 1
    torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction)
    v.LGB = dict(v.LGB, n_jobs=a.threads)
    run_root = a.out
    a.out = run_root / f"s{a.seed}_f{a.fold}"
    a.out.mkdir(parents=True, exist_ok=True)
    assert not (a.out / "selection.json").exists(), "Use a fresh directory; never overwrite a selected experiment"
    variants = a.variants.split(",")
    assert all(name in {"raw", "categorical", "gain", "all", "money"} for name in variants)
    money_cols = a.money_cols.split(",") if a.money_cols else []
    assert "money" not in variants or money_cols, "Money transform requires an explicitly audited feature list"
    X, _, y, ids, _ = v.frames(features())
    assert set(money_cols) <= set(X.columns) - set(v.CAT_COLS)
    tr, va = v.outer(y, a.seed)[a.fold]
    fi, st = v.inner(y[tr], a.seed, a.fold)
    fi, st = tr[fi], tr[st]
    qs, qc = train_test_split(st, train_size=min(a.nq, len(st) // 2), stratify=y[st], random_state=a.seed + a.fold)
    qs = np.sort(qs)
    qc = vip.strat(qc, y, a.confirm_nq, np.random.default_rng(a.seed + a.fold + 3))
    ctx = vip.strat(fi, y, a.nctx, np.random.default_rng(a.seed + a.fold + 1))
    cc = vip.strat(fi, y, a.confirm_nctx, np.random.default_rng(a.seed + a.fold + 4))
    assert not np.intersect1d(qs, qc).size and not np.intersect1d(tr, va).size
    cols = v.fm_select(X, y, fi, 100)
    manifest = {"config": {**vars(a), "out": str(a.out)},
                "packages": {p: importlib.metadata.version(p) for p in ("torch", "structured-data-models", "polars", "scikit-learn")},
                "features_sha256": hashlib.sha256((PQ / "features.parquet").read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "search_ids": ids[qs].to_list(), "confirmation_ids": ids[qc].to_list(),
                "context_ids": ids[ctx].to_list(), "confirmation_context_ids": ids[cc].to_list()}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    model = sdm.models.KumoTabular(task="classification", size="large", device="cuda")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    # Verify cached vs direct on one identical call before using the alternative engine.
    pr = vip.Predictor(vip.prepare(X, fi, "raw"), y, model)
    small_c, small_q = ctx[:min(1024, len(ctx))], qs[:min(128, len(qs))]
    # Context prefix can contain both classes here; assert rather than silently repair.
    assert len(np.unique(y[small_c])) == 2
    direct = pr.predict(small_c, small_q, cols, n_est=a.est, seed=a.seed * 100, chunk=len(small_q))
    cp = pr.predict_cached(small_c, small_q, cols, a.est, a.seed * 100, len(small_q))
    split = pr.predict_cached(small_c, small_q, cols, a.est, a.seed * 100, max(1, len(small_q) // 2))
    proof = {"max_abs_direct_cache": float(np.max(np.abs(direct - cp))),
             "max_abs_cache_chunk": float(np.max(np.abs(cp - split)))}
    print("cached engine parity", proof, flush=True)
    # Small fused attention/half-precision discrepancies are permitted, not flipped classes or misaligned rows.
    assert np.allclose(direct, cp, atol=0.002, rtol=0.005), proof
    assert np.allclose(cp, split, atol=0.002, rtol=0.005), proof
    (a.out / "cache_parity.json").write_text(json.dumps(proof, indent=2))
    del pr
    rows, sets = [], {}
    for name in variants:
        chosen = list(X.columns) if name == "all" else v.fm_select(X, y, fi, 100, rand=-1) if name == "gain" else cols
        sets[name] = chosen
        pr = vip.Predictor(representation(X, fi, name, money_cols), y, model)
        ts = time.time()
        p = pr.predict_cached(ctx, qs, chosen, a.est, a.seed * 100, a.chunk)
        row = {"name": name, "ks": v.ks(y[qs], p), "auc": float(roc_auc_score(y[qs], p)),
               "features": len(chosen), "seconds": time.time() - ts,
               "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
        rows.append(row)
        print("search", row, flush=True)
        pl.DataFrame({"application_id": ids[qs], "target": y[qs], "prediction": p}).write_parquet(a.out / f"search_{name}.parquet")
        pd.DataFrame(rows).to_csv(a.out / "search.csv", index=False)
        del pr
        torch.cuda.empty_cache()
    # Confirm only baseline plus one search winner. No outer-label or outer-score feedback.
    top = max(rows, key=lambda row: row["ks"])["name"]
    confirm = []
    for name in dict.fromkeys(["raw", top]):
        chosen = sets.get(name, cols)
        pr = vip.Predictor(representation(X, fi, name, money_cols), y, model)
        p = pr.predict_cached(cc, qc, chosen, a.confirm_est, a.seed * 100, a.chunk)
        row = {"name": name, "ks": v.ks(y[qc], p), "auc": float(roc_auc_score(y[qc], p)), "cols": chosen}
        confirm.append(row)
        print("confirm", {k: val for k, val in row.items() if k != "cols"}, flush=True)
        pl.DataFrame({"application_id": ids[qc], "target": y[qc], "prediction": p}).write_parquet(a.out / f"confirm_{name}.parquet")
        del pr
        torch.cuda.empty_cache()
    winner = max(confirm, key=lambda row: row["ks"])
    (a.out / "selection.json").write_text(json.dumps({"winner": winner["name"], "cols": winner["cols"],
        "confirmation": confirm, "search": rows, "seconds": time.time() - t0}, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar([r["name"] for r in rows], [r["ks"] for r in rows])
    axes[0].set(title="Inner search KS, not outer evidence", ylabel="KS")
    axes[1].bar([r["name"] for r in confirm], [r["ks"] for r in confirm])
    axes[1].set(title="Disjoint inner confirmation", ylabel="KS")
    fig.tight_layout()
    fig.savefig(OUT / f"154_{run_root.name}_s{a.seed}_f{a.fold}.png", dpi=120)
    plt.close(fig)
    if a.evaluate:
        name = winner["name"]
        # Freeze the representation/ranking POLICY in inner; refit its feature
        # ranking and vocabulary on outer-train, with no outer-label feedback.
        base_cols = v.fm_select(X, y, tr, 100)
        pr = vip.Predictor(representation(X, tr, "raw", money_cols), y, model)
        baseline = pr.predict_cached(tr, va, base_cols, 4, a.seed * 100, a.chunk)
        del pr
        torch.cuda.empty_cache()
        chosen = list(X.columns) if name == "all" else v.fm_select(X, y, tr, 100, rand=-1) if name == "gain" else base_cols
        if name == "raw":
            p = baseline
        else:
            pr = vip.Predictor(representation(X, tr, name, money_cols), y, model)
            p = pr.predict_cached(tr, va, chosen, 4, a.seed * 100, a.chunk)
            del pr
            torch.cuda.empty_cache()
        pl.DataFrame({"application_id": ids[va], "target": y[va], "candidate": p, "baseline": baseline}).write_parquet(
            run_root / f"s{a.seed}_f{a.fold}_outer.parquet")
        print(f"outer KS={v.ks(y[va], p):.6f} baseline={v.ks(y[va], baseline):.6f} winner={name} "
              f"elapsed={time.time()-t0:.0f}s peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)


if __name__ == "__main__":
    main()
