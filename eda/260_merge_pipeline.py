"""Single entry point of the merged team pipeline (utama + teman goal054 + DSC0152) on StratifiedKFold 5 seed 42.

Recipe: artifacts/optimization/continuation_25okt/recipe_merge.json (families, their build commands/outputs, blend).
  --stage plan            list every build command and whether its outputs are cached
  --stage build [--family X]  run missing commands in order, one process at a time (start eda/thermal_guard.py first)
  --stage blend           locked equal-weight per-cohort rank blend -> OOF report, submission.csv, manifest (SHA256)
  --stage verify          re-run blend in memory and assert submission/OOF bytes match the stored manifest
Usage: cd eda && ../.venv/bin/python 260_merge_pipeline.py --stage plan
"""
import argparse
import hashlib
import importlib
import io
import json
import os
import subprocess

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from common import ROOT

RECIPE = ROOT / "artifacts/optimization/continuation_25okt/recipe_merge.json"
e250 = importlib.import_module("250_merge_blend_existing")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_atomic(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def plan(recipe, family=None):
    for name, fam in recipe["families"].items():
        if family and name != family:
            continue
        for step in fam["steps"]:
            done = all((ROOT / p).exists() for p in step["outputs"])
            yield name, step, done


def blend(recipe):
    ids, y, tids, folds, fam_oof, fam_test, _ = e250.families(str(ROOT / recipe["dsc_npz"]))
    members = recipe["blend"]["members"]
    oof = np.zeros(len(y))
    for _, va in folds:  # per-cohort rank inside each outer fold, equal weights (locked)
        oof[va] = np.mean([e250.rank(fam_oof[m][va]) for m in members], axis=0)
    test = np.mean([e250.rank(fam_test[m]) for m in members], axis=0)
    sub = pl.DataFrame({"application_id": tids, "default_probability": test})
    assert sub.height == 30000 and sub["application_id"].n_unique() == 30000
    assert np.isfinite(test).all() and test.min() >= 0 and test.max() <= 1
    report = {"members": members, "pooled_ks": e250.ks(y, oof), "auc": float(roc_auc_score(y, oof)),
              "fold_ks": [e250.ks(y[va], oof[va]) for _, va in folds]}
    return sub, pl.DataFrame({"application_id": ids, "target": y, "oof": oof}), report


def to_bytes(frame, kind):
    buf = io.BytesIO()
    frame.write_csv(buf) if kind == "csv" else frame.write_parquet(buf)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["plan", "build", "blend", "verify"])
    ap.add_argument("--family")
    ap.add_argument("--recipe", default=str(RECIPE))
    ap.add_argument("--out", help="output folder (default: recipe output_dir)")
    a = ap.parse_args()
    recipe = json.loads(open(a.recipe).read())
    out = ROOT / (a.out or recipe["output_dir"])
    if a.stage == "plan":
        for name, step, done in plan(recipe, a.family):
            print(f"[{'cached' if done else 'MISSING'}] {name}: {step['cmd']}")
    elif a.stage == "build":
        for name, step, done in plan(recipe, a.family):
            if not done:
                print(f"=== {name}: {step['cmd']}", flush=True)
                subprocess.run(step["cmd"], shell=True, check=True, cwd=ROOT / step.get("cwd", "eda"))
    else:
        missing = [n for n, s, d in plan(recipe) if not d and n in recipe["blend"]["members"]]
        assert not missing, f"families not built: {missing}"
        sub, oof, report = blend(recipe)
        files = {"submission.csv": to_bytes(sub, "csv"), "oof.parquet": to_bytes(oof, "parquet")}
        if a.stage == "verify":
            manifest = json.loads((out / "manifest.json").read_text())
            for fname, data in files.items():
                assert hashlib.sha256(data).hexdigest() == manifest["outputs"][fname], fname
            print("verify OK", manifest["outputs"]["submission.csv"])
            return
        out.mkdir(parents=True, exist_ok=True)
        for fname, data in files.items():
            write_atomic(out / fname, data)
        inputs = sorted({p for _, s, _ in plan(recipe) for p in s["outputs"]
                         if any(p.startswith(x) for x in recipe["blend"]["input_prefixes"])})
        manifest = {"recipe": recipe, "report": report, "outputs": {f: sha(out / f) for f in files},
                    "inputs": {p: sha(ROOT / p) for p in inputs if (ROOT / p).is_file()}}
        write_atomic(out / "manifest.json", json.dumps(manifest, indent=2).encode())
        print(json.dumps(report), "\nsubmission", manifest["outputs"]["submission.csv"])


if __name__ == "__main__":
    main()
