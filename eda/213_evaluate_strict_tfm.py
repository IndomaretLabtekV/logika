"""Verify all five strict TFM folds, report pooled metrics, and export if eligible."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import ks_2samp, rankdata
from sklearn.metrics import roc_auc_score, roc_curve

import valid2 as v
from common import ROOT
from strict_feature_pool import build_pool


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def rank(p):
    return rankdata(p, method="average") / len(p)


def score(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    result = {"pooled_ks": ks(y, p), "auc": float(roc_auc_score(y, p))}
    assert abs(result["pooled_ks"] - float(np.abs(tpr - fpr).max())) < 1e-12
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/v60/strict_tfm")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--protocol", type=Path, default=ROOT / "artifacts/optimization/continuation_23okt/protocol.json")
    a = ap.parse_args()
    assert a.bootstrap >= 100
    root = a.out.resolve()
    protocol_path = a.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    train = pl.read_csv(ROOT / "data/train_applications.csv").sort("application_id")
    ids, y = train["application_id"].to_numpy(), train["target"].to_numpy().astype(int)
    sample = pl.read_csv(ROOT / "data/sample_submission.csv")
    assert len(ids) == len(set(ids)) == 70000 and sample.height == 30000
    candidate, baseline = np.full(len(y), np.nan), np.full(len(y), np.nan)
    test_parts, fold_metrics, evidence = [], [], []
    for fold, (tr, va) in enumerate(v.outer(y, protocol["outer_seed"])):
        folder = root / f"outer_f{fold}"
        mp = folder / "manifest.json"
        manifest = json.loads(mp.read_text())
        assert manifest["stage"] == "outer" and manifest["fold"] == fold
        assert manifest["protocol_sha256"] == sha(protocol_path)
        assert manifest["fit_ids"] == ids[tr].tolist()
        assert manifest["query_ids"] == ids[va].tolist()
        assert manifest["test_ids"] == sample["application_id"].to_list()
        assert set(manifest["fit_ids"]).isdisjoint(manifest["query_ids"])
        for path, expected in manifest["sources"].items():
            assert sha(ROOT / path) == expected, f"Changed source {path}"
        pool = build_pool(ids[tr].tolist(), Path(manifest["pool_root"]) / f"pool_outer_f{fold}")
        assert sha(pool["manifest_path"]) == manifest["pool_manifest_sha256"]
        member_predictions = {}
        expected_query_ids = ids[va].tolist() + sample["application_id"].to_list()
        for spec in manifest["specs"]:
            name = spec["name"]
            pp, cp = folder / f"{name}.parquet", folder / f"{name}.json"
            cache = json.loads(cp.read_text())
            assert cache["manifest_sha256"] == sha(mp) and cache["prediction_sha256"] == sha(pp)
            p = pl.read_parquet(pp)
            assert p["application_id"].to_list() == expected_query_ids
            values = p["prediction"].to_numpy()
            assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
            member_predictions[name] = values
        names = [s["name"] for s in manifest["specs"] if s["name"] != "baseline"]
        assert len(names) == 6
        cpred = np.mean([rank(member_predictions[name][:len(va)]) for name in names], axis=0)
        bpred = rank(member_predictions["baseline"][:len(va)])
        test = np.mean([rank(member_predictions[name][len(va):]) for name in names], axis=0)
        d = pl.read_parquet(folder / "evaluation.parquet")
        assert d["application_id"].to_list() == ids[va].tolist()
        assert np.array_equal(d["target"].to_numpy(), y[va])
        assert np.array_equal(d["candidate"].to_numpy(), cpred)
        assert np.array_equal(d["baseline"].to_numpy(), bpred)
        td = pl.read_parquet(folder / "test.parquet")
        assert td["application_id"].to_list() == sample["application_id"].to_list()
        assert np.array_equal(td["candidate"].to_numpy(), test)
        assert np.isnan(candidate[va]).all()
        candidate[va], baseline[va] = cpred, bpred
        test_parts.append(test)
        fold_metrics.append({"fold": fold, "rows": len(va), "candidate_ks": ks(y[va], cpred),
                             "baseline_ks": ks(y[va], bpred), "candidate_auc": float(roc_auc_score(y[va], cpred)),
                             "baseline_auc": float(roc_auc_score(y[va], bpred))})
        evidence.append({"fold": fold, "manifest_sha256": sha(mp),
                         "evaluation_sha256": sha(folder / "evaluation.parquet"),
                         "test_sha256": sha(folder / "test.parquet")})
    assert np.isfinite(candidate).all() and np.isfinite(baseline).all()
    c, b = score(y, candidate), score(y, baseline)
    c["mean_fold_ks"] = float(np.mean([r["candidate_ks"] for r in fold_metrics]))
    b["mean_fold_ks"] = float(np.mean([r["baseline_ks"] for r in fold_metrics]))
    rng = np.random.default_rng(213)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    samples = np.empty((a.bootstrap, 3))
    for j in range(a.bootstrap):
        rows = np.r_[rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
        ck, bk = ks(y[rows], candidate[rows]), ks(y[rows], baseline[rows])
        samples[j] = ck, bk, ck - bk
    uncertainty = {"resamples": a.bootstrap, "seed": 213, "unit": "customer/application; stratified paired bootstrap",
                   "candidate_ks_95ci": np.quantile(samples[:, 0], [.025, .975]).tolist(),
                   "baseline_ks_95ci": np.quantile(samples[:, 1], [.025, .975]).tolist(),
                   "delta_95ci": np.quantile(samples[:, 2], [.025, .975]).tolist(),
                   "delta_sd": float(samples[:, 2].std(ddof=1)), "p_delta_positive": float((samples[:, 2] > 0).mean()),
                   "limits": "Conditional on fixed OOF predictions; does not correct repeated research selection or retraining variation."}
    eligible = c["pooled_ks"] >= .53
    report = {"rows": len(y), "complete": True, "audit_passed": True, "outer_seed": protocol["outer_seed"],
              "model_seeds": protocol["model_seeds"], "candidate": c, "baseline": b,
              "delta_baseline": c["pooled_ks"] - b["pooled_ks"], "fold_metrics": fold_metrics,
              "uncertainty": uncertainty, "evidence": evidence, "submission_eligible": eligible,
              "submission_performed": False, "protocol_sha256": sha(protocol_path)}
    pl.DataFrame({"application_id": ids, "target": y, "candidate": candidate, "baseline": baseline}).write_parquet(root / "complete_oof.parquet")
    if eligible:
        prediction = np.mean(test_parts, axis=0)
        assert np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
        submission = sample.select("application_id").with_columns(pl.Series("default_probability", prediction))
        assert submission["application_id"].n_unique() == 30000
        path = root / "submission.csv"
        submission.write_csv(path)
        report["submission_csv_sha256"] = sha(path)
    (root / "complete_evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
