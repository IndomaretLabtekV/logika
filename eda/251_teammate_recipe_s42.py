"""Reproduce teammate goal054 (Kumo V5 + latent offset + spline) on StratifiedKFold 5 seed 42.

Protocol: artifacts/optimization/continuation_25okt/protocol_251_teammate_s42.json (locked before scores).
Stages per fold (each cached atomically, one heavy unit per process):
  reverse  re_* latent features from outer-train customers only (label-free)       [CPU]
  kumo     Kumo large, 90 V5 cols, context 50k (teammate sampler), e8, seed 2026+fold [GPU]
  post     offset correction on the 6k holdout + spline on 56k outer-train          [CPU]
  eval     pooled OOF of K5, K5o, S, R0, R over the 5 folds
Usage: cd eda && ../.venv/bin/python 251_teammate_recipe_s42.py --stage {reverse,kumo,post} --fold k | --stage eval
"""
import argparse
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import valid2 as v
from common import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
TEAM = ROOT / "results/v56/teammate_features"
OUT = Path(os.environ.get("LOGIKA_MERGE_OUT", ROOT / "results/v71/teammate_s42"))  # override = fresh reproduction check
DEF = ROOT / "artifacts/optimization/goal054/cv5_seed31027/kumo_offset_l20p003_unweighted_d0p5/definition.json"
SPEC = dict(penalty=0.003, balance="unweighted", correction_strength=0.5)
SEED = 2026  # base of the context/model seed (seed + fold); --seed 2027 = second seed (protocol 256)


def sfx():
    return "" if SEED == 2026 else f"_s{SEED}"


def atomic_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_json(value, path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(value, indent=2, default=str))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def labels():
    train = pd.read_csv(ROOT / "data/train_applications.csv", usecols=["application_id", "target"])
    return train.sort_values("application_id").reset_index(drop=True)


def fold_frame(fold):
    """V5 (+ re_* when built) for the fold; val/test rows carry no label. Returns frame, fit/val/test ids."""
    v5 = pd.read_parquet(TEAM / f"outer_s42_f{fold}/data/processed/application_features_v5.parquet")
    lab = labels()
    folds = v.outer(lab.target.to_numpy(), 42)
    tr, va = folds[fold]
    fit_ids, val_ids = set(lab.application_id.iloc[tr]), set(lab.application_id.iloc[va])
    assert set(v5.application_id[v5.split.eq("train")]) == fit_ids, "fold dir fitting ids != valid2.outer seed 42"
    assert v5.target[~v5.split.eq("train")].isna().all()
    rev = OUT / f"f{fold}/reverse.parquet"
    if rev.exists():
        v5 = v5.merge(pd.read_parquet(rev), on="application_id", how="left", validate="one_to_one")
    return v5, fit_ids, val_ids


def stage_reverse(fold):
    import reverse_synthetic_patterns as rsp

    v5, _, _ = fold_frame(fold)
    rsp.ROOT = TEAM / f"outer_s42_f{fold}"  # raw csv symlinks there; split=='train' = outer-train customers
    rsp.distribution_summary = lambda values: {}  # audit-only summary; its scipy kstest call breaks on scipy here
    base = v5.set_index("customer_id")
    parts = [rsp.fit_price_latent(base)[0], rsp.fit_bureau_latent(base)[0], rsp.fit_account_latent(base)[0],
             rsp.fit_repayment_latent(base)[0]]
    new = pd.concat(parts, axis=1)
    assert new.index.equals(base.index) and new.columns.is_unique and not any("target" in c for c in new.columns)
    new = new.replace([np.inf, -np.inf], np.nan)
    new.insert(0, "application_id", base.application_id.to_numpy())
    atomic_parquet(new.reset_index(drop=True), OUT / f"f{fold}/reverse.parquet")
    print("reverse", fold, new.shape, flush=True)


def split_rows(frame, fit_ids, val_ids, fold):
    from kumo_tabular import context_indices

    lab = labels().set_index("application_id").target
    is_fit = frame.application_id.isin(fit_ids).to_numpy()
    ti = np.flatnonzero(is_fit)
    y = np.zeros(len(frame), dtype=int)
    y[ti] = lab.loc[frame.application_id.iloc[ti]].to_numpy()
    ci = np.sort(context_indices(ti, y, 50000, SEED + fold))
    hi = np.setdiff1d(ti, ci)
    vi = np.flatnonzero(frame.application_id.isin(val_ids).to_numpy())
    te = np.flatnonzero(~is_fit & ~frame.application_id.isin(val_ids).to_numpy())
    assert len(ci) == 50000 and len(hi) == 6000 and len(vi) == 14000 and len(te) == 30000
    return y, ti, ci, hi, vi, te


def stage_kumo(fold):
    import strict_kumo_engine as eng

    frame, fit_ids, val_ids = fold_frame(fold)
    y, ti, ci, hi, vi, te = split_rows(frame, fit_ids, val_ids, fold)
    columns = json.loads(DEF.read_text())["context_columns"]
    query = np.r_[vi, hi, te]
    p, meta = eng.predict(frame, ci, query, y, columns, SEED + fold, n_estimators=8, column_cap=500, chunk=128,
                          memory_fraction=.70)
    part = np.r_[["val"] * len(vi), ["holdout"] * len(hi), ["test"] * len(te)]
    out = pd.DataFrame({"application_id": frame.application_id.iloc[query].to_numpy(), "part": part, "kumo": p})
    atomic_parquet(out, OUT / f"f{fold}/kumo{sfx()}.parquet")
    meta.pop("context_rows", None)
    atomic_json(meta, OUT / f"f{fold}/kumo{sfx()}_meta.json")
    print("kumo", fold, "peak GiB", meta.get("max_allocated_gib"), flush=True)


def stage_post(fold):
    import hybrid_kumo as hk
    import research_context_correction054 as cc

    frame, fit_ids, val_ids = fold_frame(fold)
    y, ti, ci, hi, vi, te = split_rows(frame, fit_ids, val_ids, fold)
    frame = frame.assign(target=np.where(np.isin(np.arange(len(frame)), ti), y, np.nan))
    k = pd.read_parquet(OUT / f"f{fold}/kumo{sfx()}.parquet").set_index("application_id").kumo
    kh, kv, kt = (k.loc[frame.application_id.iloc[ix]].to_numpy() for ix in (hi, vi, te))
    d = json.loads(DEF.read_text())
    spec = {**SPEC, "correction_columns": d["correction_columns"]}
    bundle = cc.fit_correction(spec, frame, hi, kh)
    cv_, _ = cc.predict_correction(bundle, frame.iloc[vi], kv)
    ct_, _ = cc.predict_correction(bundle, frame.iloc[te], kt)
    spline_spec, _ = hk.specifications()
    sb = hk.fit_spline(spline_spec, frame, ti)
    sv, st = hk.predict_spline(sb, frame.iloc[vi]), hk.predict_spline(sb, frame.iloc[te])
    rows = []
    for part, ix, kk, kc, ss in (("val", vi, kv, cv_, sv), ("test", te, kt, ct_, st)):
        rows.append(pd.DataFrame({"application_id": frame.application_id.iloc[ix].to_numpy(), "part": part,
                                  "K5": kk, "K5o": kc, "S": ss, "R0": .75 * kk + .25 * ss, "R": .75 * kc + .25 * ss}))
    atomic_parquet(pd.concat(rows, ignore_index=True), OUT / f"f{fold}/post{sfx()}.parquet")
    print("post", fold, "correction iters", bundle["iterations"], flush=True)


def stage_eval():
    lab = labels()
    folds = v.outer(lab.target.to_numpy(), 42)
    names = ["K5", "K5o", "S", "R0", "R"]
    oof = pd.DataFrame({"application_id": lab.application_id})
    fold_ks = {n: [] for n in names}
    for f, (_, va) in enumerate(folds):
        p = pd.read_parquet(OUT / f"f{f}/post{sfx()}.parquet").query("part == 'val'").set_index("application_id")
        ids = lab.application_id.iloc[va]
        for n in names:
            oof.loc[va, n] = p.loc[ids, n].to_numpy()
            fold_ks[n].append(v.ks(lab.target.iloc[va].to_numpy(), oof.loc[va, n].to_numpy()))
    y = lab.target.to_numpy()
    rep = {n: {"pooled_ks": v.ks(y, oof[n].to_numpy()), "auc": float(roc_auc_score(y, oof[n])),
               "fold_ks": fold_ks[n], "mean_fold_ks": float(np.mean(fold_ks[n]))} for n in names}
    for n, r in rep.items():
        print(f"{n:4s} pooled {r['pooled_ks']:.5f} auc {r['auc']:.5f} mean-fold {r['mean_fold_ks']:.5f}",
              [round(x, 4) for x in r["fold_ks"]])
    atomic_parquet(oof.assign(target=y), OUT / f"oof{sfx()}.parquet")
    test = pd.concat([pd.read_parquet(OUT / f"f{f}/post{sfx()}.parquet").query("part == 'test'") for f in range(5)])
    atomic_parquet(test.groupby("application_id")[names].mean().reset_index(), OUT / f"test{sfx()}.parquet")
    atomic_json(rep, ROOT / f"eda/out/251_teammate_s42{sfx()}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["reverse", "kumo", "post", "eval"])
    ap.add_argument("--fold", type=int)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    SEED = a.seed
    {"reverse": stage_reverse, "kumo": stage_kumo, "post": stage_post}.get(a.stage, lambda f: stage_eval())(a.fold)
