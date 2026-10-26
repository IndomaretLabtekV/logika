"""Copy a candidate submission to results/v63/<name>/, validate it, and write the public forecast
BEFORE upload (amendment_01 rule: public ~ pooled OOF + offset; sd from 9k-row sampling + seed).
Usage: cd eda && ../.venv/bin/python 225_prepare_submission.py <candidate_dir> <name> <offset_kind: blend|kumo>
"""
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from common import ROOT

OFFSET = {"blend": .0373, "kumo": .0339}
SD = float(np.hypot(.0022, .0015))
src, name, kind = ROOT / sys.argv[1], sys.argv[2], sys.argv[3]
report = json.loads((src / "report.json").read_text())
sub = pl.read_csv(src / "submission.csv")
sample = pl.read_csv(ROOT / "data/sample_submission.csv")
assert sub.columns == ["application_id", "default_probability"]
assert sub.height == 30000 and sub["application_id"].n_unique() == 30000
assert set(sub["application_id"]) == set(sample["application_id"])
p = sub["default_probability"].to_numpy()
assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
out = ROOT / "results/v63" / name
out.mkdir(parents=True, exist_ok=True)
shutil.copyfile(src / "submission.csv", out / "submission.csv")
best = pl.read_csv(ROOT / "results/v31/submission.csv").join(sub, on="application_id", suffix="_new")
forecast = report["pooled_ks"] + OFFSET[kind]
pre = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "source": str(src.relative_to(ROOT)),
       "members": report["members"], "candidate": report["candidate"], "pooled_oof_ks": report["pooled_ks"],
       "oof_auc": report["auc"], "fold_ks": report["fold_ks"], "delta_vs_REF": report.get("delta_vs_REF"),
       "public_forecast_mean": forecast, "public_forecast_sd": SD, "offset_kind": kind,
       "spearman_vs_v31": float(spearmanr(best["default_probability"], best["default_probability_new"]).statistic),
       "csv_sha256": hashlib.sha256((out / "submission.csv").read_bytes()).hexdigest(), "submitted": False}
(out / "submission_preflight.json").write_text(json.dumps(pre, indent=2))
print(json.dumps(pre, indent=2))
