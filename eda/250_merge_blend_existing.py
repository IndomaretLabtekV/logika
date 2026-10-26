"""Merge step (i): equal-weight rank blends of existing OOF families across the three team pipelines.

Families (protocol artifacts/optimization/continuation_25okt/protocol_250_blend_existing.json, locked before scores):
T trees6 v22, C Causilo, KL Kumo legacy v28m, K64 Kumo strict 2-seed, D = DSC0152 v66 ext0.5|tm30|lgb_lean.
D is OOF averaged over 10-fold x 6 fold seeds -> exploratory until reproduced on seed-42 5-fold.
Prints pooled/fold KS+AUC, bootstrap deltas vs REF and vs best component, test Spearman vs scored submissions.
Usage: cd eda && ../.venv/bin/python 250_merge_blend_existing.py [--d-npz path] [--tag name]
"""
import importlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

import valid2 as v
from common import ROOT

e221 = importlib.import_module("221_evaluate_recipe_candidates")
rank, load, fast_ks, ks, R = e221.rank, e221.load, e221.fast_ks, e221.ks, e221.R
PROTO = ROOT / "artifacts/optimization/continuation_25okt/protocol_250_blend_existing.json"
SCORED = {"goal054_0.56470": R / "v70/goal054_kumo_offset/submission.csv", "v31_0.56405": R / "v31/submission.csv",
          "K64_0.56168": R / "v63/v70b_k64_2seed/submission.csv"}


def pd_read(path):
    return pl.read_parquet(path)


def dsc_recipe(z, prefix):
    """v66 make_submission formula: ext0.5 of tm30 away from lgb_lean (ranks over all rows of the array)."""
    r = {k[len(prefix):]: rank(z[k]) for k in z.files if k.startswith(prefix)}
    trees = (r["lgb_full"] + r["lgb_lean"] + r["cat_full"] + r["xgb_full"]) / 4
    tm30 = 0.7 * (trees * 2 + r["lgbx_full"]) / 3 + 0.3 * r["linear"]
    return rank(tm30 + 0.5 * (tm30 - r["lgb_lean"]))


def by_id(values, src_ids, dst_ids):
    pos = {t: i for i, t in enumerate(src_ids)}
    return values[np.array([pos[t] for t in dst_ids])]


def families(d_npz):
    """All seed-42 family OOF (by sorted id) and test vectors (sample order) + the DSC npz."""
    train = pl.read_csv(ROOT / "data/train_applications.csv").sort("application_id")
    ids, y = train["application_id"].to_numpy(), train["target"].to_numpy().astype(int)
    tids = pl.read_csv(ROOT / "data/sample_submission.csv")["application_id"].to_numpy()  # sample order
    stids = np.sort(tids)
    folds = v.outer(y, 42)
    fam_oof, fam_test = {}, {}
    fam_oof["T"] = np.mean([rank(load(str(p).format("oof"), ids)) for p in e221.TREES], axis=0)
    fam_test["T"] = by_id(np.mean([rank(load(str(p).format("test"), stids)) for p in e221.TREES], axis=0), stids, tids)
    fam_oof["C"] = sum(w * load(str(p).format("oof"), ids) for w, p in e221.CAUSILO) / 5
    fam_test["C"] = by_id(sum(w * load(str(p).format("test"), stids) for w, p in e221.CAUSILO) / 5, stids, tids)
    fam_oof["KL"] = load(R / "v28m/oof_kumo_large_c56000.parquet", ids)
    k64 = pl.read_parquet(R / "v60/candidates/K__K64_2seed/oof.parquet").sort("application_id")
    assert k64["application_id"].to_list() == ids.tolist()
    fam_oof["K64"] = k64["K"].to_numpy()
    k64t = pl.read_csv(R / "v60/candidates/K__K64_2seed/submission.csv")
    fam_test["K64"] = by_id(k64t["default_probability"].to_numpy(), k64t["application_id"].to_numpy(), tids)
    z = np.load(d_npz, allow_pickle=True)
    fam_oof["D"] = by_id(dsc_recipe(z, "oof_"), z["train_ids"], ids)
    fam_test["D"] = by_id(dsc_recipe(z, "test_"), z["test_ids"], tids)
    r_dir = R / "v71/teammate_s42"
    for fam, stem in [("R", ""), ("KF", "_kf"), ("KD", "_kd"), ("T5", "_t5")]:  # teammate recipe on seed 42 (eda/251), Kumo V5+factor (eda/253)
        if (r_dir / f"oof{stem}.parquet").exists():
            ro, rt = pd_read(r_dir / f"oof{stem}.parquet"), pd_read(r_dir / f"test{stem}.parquet")
            fam_oof[fam] = by_id(ro[fam].to_numpy(), ro["application_id"].to_numpy(), ids)
            fam_test[fam] = by_id(rt[fam].to_numpy(), rt["application_id"].to_numpy(), tids)
    if "R" in fam_oof and (r_dir / "oof_s2027.parquet").exists():  # R2 = mean of two model seeds (protocol 256)
        ro, rt = pd_read(r_dir / "oof_s2027.parquet"), pd_read(r_dir / "test_s2027.parquet")
        fam_oof["R2"] = (fam_oof["R"] + by_id(ro["R"].to_numpy(), ro["application_id"].to_numpy(), ids)) / 2
        fam_test["R2"] = (fam_test["R"] + by_id(rt["R"].to_numpy(), rt["application_id"].to_numpy(), tids)) / 2
    if "KD" in fam_oof and (r_dir / "oof_kd_s2027.parquet").exists():  # KD2 = mean of two model seeds (protocol 257)
        ro, rt = pd_read(r_dir / "oof_kd_s2027.parquet"), pd_read(r_dir / "test_kd_s2027.parquet")
        fam_oof["KD2"] = (fam_oof["KD"] + by_id(ro["KD"].to_numpy(), ro["application_id"].to_numpy(), ids)) / 2
        fam_test["KD2"] = (fam_test["KD"] + by_id(rt["KD"].to_numpy(), rt["application_id"].to_numpy(), tids)) / 2
    for k in ["lgb_lean", "xgb_full", "fac_full"]:
        fam_oof[f"D_{k}"] = by_id(z[f"oof_{k}"], z["train_ids"], ids)
    return ids, y, tids, folds, fam_oof, fam_test, z


def main():
    args = sys.argv[1:]
    d_npz = args[args.index("--d-npz") + 1] if "--d-npz" in args else str(R / "dsc0152/agas/v66/output/preds_v66.npz")
    tag = args[args.index("--tag") + 1] if "--tag" in args else "v66"
    proto_path = ROOT / args[args.index("--protocol") + 1] if "--protocol" in args else PROTO
    proto = json.loads(proto_path.read_text())
    ids, y, tids, folds, fam_oof, fam_test, z = families(d_npz)
    cands = {"REF": ["T", "C", "KL"], **proto.get("candidates_equal_weight_rank", proto.get("candidates", {}))}
    cands = {n: ms for n, ms in cands.items() if all(m in fam_oof for m in ms)}  # skip families not built yet
    oof = {}
    for n, members in cands.items():
        oof[n] = np.full(len(y), np.nan)
        for tr, va in folds:  # per-cohort rank inside each outer fold
            oof[n][va] = np.mean([rank(fam_oof[m][va]) for m in members], axis=0)
    test = {n: np.mean([rank(fam_test[m]) for m in ms], axis=0) for n, ms in cands.items() if "KL" not in ms}
    rng = np.random.default_rng(250)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boots = [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(1000)]
    comp = {}  # components ranked per cohort, like the blends, so deltas compare like with like
    for m in [m for m in ["T", "C", "KL", "K64", "D", "R", "R2", "KF", "KD", "KD2", "T5"] if m in fam_oof]:
        comp[m] = np.full(len(y), np.nan)
        for tr, va in folds:
            comp[m][va] = rank(fam_oof[m][va])
    print("components pooled KS / AUC (raw, global):")
    for m, p in fam_oof.items():
        print(f"  {m:12s} {ks(y, p):.5f} {roc_auc_score(y, p):.5f} folds",
              [round(ks(y[va], p[va]), 4) for _, va in folds])
    bks = {n: fast_ks(y, p, boots) for n, p in {**oof, **comp}.items()}
    report = {"protocol": proto["id"], "d_npz": d_npz, "candidates": {}}
    for n, members in cands.items():
        r = {"members": members, "pooled_ks": ks(y, oof[n]), "auc": float(roc_auc_score(y, oof[n])),
             "fold_ks": [ks(y[va], oof[n][va]) for _, va in folds]}
        best = max(members, key=lambda m: ks(y, comp[m]))
        for base_name, base in [("REF", oof["REF"]), (f"best_{best}", comp[best])]:
            if n == "REF":
                continue
            key = "REF" if base_name == "REF" else best
            d = bks[n] - bks[key]
            r[f"delta_vs_{base_name}"] = {"observed": r["pooled_ks"] - ks(y, base),
                                          "ci95": np.quantile(d, [.025, .975]).tolist(), "p_gt0": float((d > 0).mean()),
                                          "folds_won": int(sum(ks(y[va], oof[n][va]) > ks(y[va], base[va]) for _, va in folds))}
        if n in test:
            r["test_spearman"] = {}
            for s, path in SCORED.items():
                if path.exists():
                    sub = pl.read_csv(path)
                    r["test_spearman"][s] = float(spearmanr(test[n], by_id(sub["default_probability"].to_numpy(),
                                                                         sub["application_id"].to_numpy(), tids))[0])
        report["candidates"][n] = r
        print(n, json.dumps({k: (round(x, 5) if isinstance(x, float) else x) for k, x in r.items()
                             if k not in ("fold_ks", "members")}, default=str),
              "folds", [round(x, 4) for x in r["fold_ks"]])
    out = ROOT / f"eda/out/250_blend_existing_{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    folder = R / f"v71/merge_blend/{tag}"
    folder.mkdir(parents=True, exist_ok=True)
    for n, t in test.items():
        sub = pl.DataFrame({"application_id": tids, "default_probability": t})
        assert sub.height == 30000 and sub["application_id"].n_unique() == 30000 and t.min() >= 0 and t.max() <= 1
        sub.write_csv(folder / f"{n}.csv")
    pl.DataFrame({"application_id": ids, "target": y, **oof}).write_parquet(folder / "oof.parquet")
    fig, ax = plt.subplots(figsize=(8, 4))
    for n, r in report["candidates"].items():
        ax.plot(range(5), r["fold_ks"], marker="o", label=f"{n} {r['pooled_ks']:.4f}")
    ax.set_xlabel("outer fold (seed 42)"); ax.set_ylabel("KS"); ax.legend(fontsize=7); ax.set_title(f"250 merge blends ({tag})")
    fig.tight_layout(); fig.savefig(ROOT / f"eda/out/250_blend_existing_{tag}.png", dpi=120)


if __name__ == "__main__":
    main()
