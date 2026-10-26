"""Evaluate locked candidates of amendment_01 from cached strict members (complete 5-fold members only).

K   = equal rank mean of strict Kumo members (per outer cohort, test per fold then mean over folds)
KT  = 0.5 rank(K) + 0.5 rank(trees6)            (v22 GBDT family)
KTC = thirds rank(trees6), rank(Causilo), rank(K)  (v59 structure, legacy Kumo replaced)
REF = thirds trees6 / Causilo / legacy Kumo v28m (OOF analogue of v59; public 0.56296)
Weights are fixed by amendment_01; nothing is tuned on OOF. Prints pooled/fold KS+AUC, paired
stratified bootstrap deltas, writes oof/test/submission per candidate and one figure.
Usage: cd eda && ../.venv/bin/python 221_evaluate_recipe_candidates.py [member ...]
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import valid2 as v
from common import ROOT

STRICT = ROOT / "results/v60/strict_tfm"
OUT = ROOT / "results/v60/candidates"
R = ROOT / "results"
TREES = [R / f"v22/v22/gbdt/{{}}_{m}.parquet" for m in ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]]
CAUSILO = [(1, R / "v22/v22/fm/{}_causilo.parquet"), (2, R / "v37/{}_causilo.parquet"), (2, R / "v41/causilo_s12/{}_causilo.parquet")]
DEFAULT_MEMBERS = [f"{view}_s{seed}" for view in ("pool_cap500", "nested_small", "pool_uncapped") for seed in (2026, 2027)]


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rank(x):
    return rankdata(x, method="average") / len(x)


def ks(y, p):
    return v.ks(y, p)


def fast_ks(y, p, rows_list):
    """KS on bootstrap index sets via sorted order (ties broken consistently by stable sort)."""
    out = []
    for rows in rows_list:
        yy, pp = y[rows], p[rows]
        order = np.argsort(-pp, kind="stable")
        ys, ps = yy[order], pp[order]
        tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
        last = np.r_[ps[1:] != ps[:-1], True]  # evaluate only at distinct-score boundaries
        out.append(np.max(np.abs(tp[last] / tp[-1] - fp[last] / fp[-1])))
    return np.array(out)


def load(path, ids):
    d = pl.read_parquet(path).sort("application_id")
    assert d["application_id"].to_list() == sorted(ids), path
    x = d[[c for c in d.columns if c not in ("application_id", "target")][0]].to_numpy()
    assert np.isfinite(x).all(), path
    return x


def main():
    args = sys.argv[1:]
    tag = args.pop(args.index("--tag") + 1) if "--tag" in args else None
    if tag:
        args.remove("--tag")
    members = args or DEFAULT_MEMBERS
    train = pl.read_csv(ROOT / "data/train_applications.csv").sort("application_id")
    ids, y = train["application_id"].to_numpy(), train["target"].to_numpy().astype(int)
    tids = pl.read_csv(ROOT / "data/sample_submission.csv")["application_id"].to_numpy()
    stids = np.sort(tids)
    folds = v.outer(y, 42)
    available = []
    roots = {}
    for i, m in enumerate(members):  # "results/v62/strict_variant::member" reads another root
        root, m = (ROOT / m.split("::")[0], m.split("::")[1]) if "::" in m else (STRICT, m)
        members[i] = m
        roots[m] = root
        ok = all((root / f"outer_f{f}/{m}.json").exists() for f in range(5))
        print(f"member {m}: {'complete' if ok else 'INCOMPLETE -> excluded'}")
        if ok:
            available.append(m)
    assert available, "no complete member"
    # legacy/historical families (OOF averaged over outer seeds, already OOF per row)
    trees_oof = np.mean([rank(load(str(p).format("oof"), ids)) for p in TREES], axis=0)
    trees_test = np.mean([rank(load(str(p).format("test"), tids)) for p in TREES], axis=0)
    cau_oof = sum(w * load(str(p).format("oof"), ids) for w, p in CAUSILO) / 5
    cau_test = sum(w * load(str(p).format("test"), tids) for w, p in CAUSILO) / 5
    kumo_legacy_oof = load(R / "v28m/oof_kumo_large_c56000.parquet", ids)
    member_oof = {m: np.full(len(y), np.nan) for m in available}
    names = ["K", "KT", "KTC", "REF"]
    oof = {n: np.full(len(y), np.nan) for n in names}
    test_parts = {n: [] for n in ["K", "KT", "KTC"]}
    test_pos = {t: i for i, t in enumerate(stids)}
    evidence = []
    for f, (tr, va) in enumerate(folds):
        val_ranks, test_ranks = [], []
        for m in available:
            folder = roots[m] / f"outer_f{f}"
            manifest = json.loads((folder / "manifest.json").read_text())
            assert manifest["fit_ids"] == ids[tr].tolist() and manifest["query_ids"] == ids[va].tolist()
            assert not set(manifest["fit_ids"]) & set(manifest["query_ids"])
            meta = json.loads((folder / f"{m}.json").read_text())
            pp = folder / f"{m}.parquet"
            assert meta["manifest_sha256"] == sha(folder / "manifest.json") and meta["prediction_sha256"] == sha(pp)
            assert meta["metadata"]["query_labels_accessed"] is False
            d = pl.read_parquet(pp)
            assert d["application_id"].to_list() == ids[va].tolist() + tids.tolist()
            p = d["prediction"].to_numpy()
            member_oof[m][va] = p[:len(va)]
            val_ranks.append(rank(p[:len(va)]))
            test_ranks.append(rank(p[len(va):]))
            evidence.append({"fold": f, "member": m, "prediction_sha256": meta["prediction_sha256"],
                             "context": meta["metadata"]["context_count"],
                             "processed_columns": [c["columns"] for c in meta["metadata"]["processed_contexts"]][:1],
                             "memory_fraction": meta["metadata"]["memory_fraction"]})
        k_val, k_test = np.mean(val_ranks, axis=0), np.mean(test_ranks, axis=0)
        t_val, c_val, l_val = rank(trees_oof[va]), rank(cau_oof[va]), rank(kumo_legacy_oof[va])
        oof["K"][va] = k_val
        oof["KT"][va] = (rank(k_val) + t_val) / 2
        oof["KTC"][va] = (rank(k_val) + t_val + c_val) / 3
        oof["REF"][va] = (l_val + t_val + c_val) / 3
        # test ids in sample order -> map trees/causilo (sorted order) into sample order
        order = np.array([test_pos[t] for t in tids])
        kt = rank(k_test)
        test_parts["K"].append(k_test)
        test_parts["KT"].append((kt + rank(trees_test[order])) / 2)
        test_parts["KTC"].append((kt + rank(trees_test[order]) + rank(cau_test[order])) / 3)
    assert all(np.isfinite(o).all() for o in oof.values())
    rng = np.random.default_rng(221)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boots = [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(1000)]
    bks = {n: fast_ks(y, oof[n], boots) for n in names}
    report = {"members_used": available, "members_requested": members, "evidence": evidence, "candidates": {}}
    for m in available:
        report.setdefault("member_pooled_ks", {})[m] = ks(y, member_oof[m])
    for n in names:
        r = {"pooled_ks": ks(y, oof[n]), "auc": float(roc_auc_score(y, oof[n])),
             "fold_ks": [ks(y[va], oof[n][va]) for _, va in folds]}
        r["mean_fold_ks"] = float(np.mean(r["fold_ks"]))
        for base in ["REF", "K"]:
            if base != n:
                d = bks[n] - bks[base]
                r[f"delta_vs_{base}"] = {"observed": r["pooled_ks"] - ks(y, oof[base]),
                                        "ci95": np.quantile(d, [.025, .975]).tolist(), "sd": float(d.std(ddof=1)),
                                        "p_gt0": float((d > 0).mean()),
                                        "folds_won": int(sum(ks(y[va], oof[n][va]) > ks(y[va], oof[base][va]) for _, va in folds))}
        report["candidates"][n] = r
        print(n, json.dumps({k: (round(x, 5) if isinstance(x, float) else x) for k, x in r.items() if k != "fold_ks"}),
              "folds", [round(x, 4) for x in r["fold_ks"]])
    for m, s in report.get("member_pooled_ks", {}).items():
        print(f"  member {m}: pooled KS {s:.5f}")
    tag = tag or f"{len(available)}m"
    names_used = "+".join(available)
    for n in ["K", "KT", "KTC"]:
        folder = OUT / f"{n}__{tag}"
        folder.mkdir(parents=True, exist_ok=True)
        pred = np.mean(test_parts[n], axis=0)
        sub = pl.DataFrame({"application_id": tids, "default_probability": pred})
        assert sub.height == 30000 and sub["application_id"].n_unique() == 30000 and set(sub["application_id"]) == set(tids)
        assert np.isfinite(pred).all() and pred.min() >= 0 and pred.max() <= 1
        tmp = folder / "submission.csv.tmp"
        sub.write_csv(tmp)
        with open(tmp, "rb") as s:
            os.fsync(s.fileno())
        tmp.replace(folder / "submission.csv")
        pl.DataFrame({"application_id": ids, "target": y, n: oof[n]}).write_parquet(folder / "oof.parquet")
        (folder / "report.json").write_text(json.dumps({"members": names_used, "candidate": n, **report["candidates"][n],
                                                        "submission_sha256": sha(folder / "submission.csv")}, indent=2))
    (OUT / f"report__{tag}.json").write_text(json.dumps(report, indent=2))
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in names:
        ax.plot(range(5), report["candidates"][n]["fold_ks"], marker="o", label=f"{n} {report['candidates'][n]['pooled_ks']:.4f}")
    ax.set_xlabel("outer fold (seed 42)"); ax.set_ylabel("KS"); ax.legend(); ax.set_title(f"221 candidates, members: {len(available)}")
    fig.tight_layout(); fig.savefig(ROOT / f"eda/out/221_candidates_{tag}.png", dpi=120)


if __name__ == "__main__":
    main()
