"""Independent integration checks: hashes, executable cells, CSV IDs and source inventory.

No model training/inference, no network, no prediction selection. Outputs evidence
to eda/out and docs/integration. Historical reported metrics stay source-tagged.
"""
import ast
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/integration"
OUT = ROOT / "eda/out"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    mapping = list(csv.DictReader((AUDIT / "pemetaan_file.csv").open()))
    original = {r["path"]: r for r in csv.DictReader((AUDIT / "inventaris_awal.csv").open())}
    errors, checked, sources, notebooks, submissions = [], [], [], [], []
    for row in mapping:
        p = ROOT / row["destination"]
        actual = digest(p) if p.is_file() else None
        okay = actual == row["new_sha256"]
        if not okay:
            errors.append(f"Destination hash mismatch: {p}")
        checked.append({"source": row["source"], "destination": row["destination"], "hash_ok": okay,
                        "adapted": row["original_sha256"] != row["new_sha256"]})
        if p.suffix == ".py":
            tree = ast.parse(p.read_text(), filename=str(p))
            sources.append({"path": row["destination"],
                            "functions": [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))],
                            "imports": [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))],
                            "path_literals": sorted({n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                                                     and isinstance(n.value, str) and len(n.value)<250
                                                     and any(s in n.value for s in ("data/", "artifacts/", "notebook/", "submission"))})})
            if "Logika-main" in p.read_text() or "data/raw" in p.read_text() or "notebooks/" in p.read_text():
                errors.append(f"Obsolete active path: {p}")
        if p.suffix == ".ipynb":
            nb = nbformat.read(p, as_version=4)
            nbformat.validate(nb)
            count = 0
            for i, cell in enumerate(nb.cells):
                if cell.cell_type != "code":
                    continue
                try:
                    code = TransformerManager().transform_cell(cell.source)
                    compile(code, f"{p}:cell{i}", "exec")
                    if any(s in cell.source for s in ("Logika-main", "data/raw", "notebooks/")):
                        errors.append(f"Obsolete notebook code path: {p}:cell{i}")
                except SyntaxError as exc:
                    errors.append(f"Notebook syntax: {p}:cell{i}: {exc}")
                count += 1
            notebooks.append({"path": row["destination"], "code_cells_compiled": count})
    # Baseline code and existing submission remain byte-identical after structural work.
    baseline_paths = ["eda/common.py", "eda/features.py", "eda/valid2.py", "eda/run_valid2.py", "eda/blend2.py",
                      "eda/152_vipcop_kumo.py", "eda/153_kumo_input_audit.py", "eda/154_kumo_recipe_audit.py",
                      "eda/155_kumo_nested_eval.py", "eda/157_kumo_precision_audit.py",
                      "results/v31/submission.csv", "data_pq/features.parquet"]
    baselines = {p: digest(ROOT/p)==original[p]["sha256"] for p in baseline_paths}
    # Authorized production recipe update after the user's request to submit
    # today. Preserve the original builder/notebook bytes in an explicit archive.
    production_archive={"notebook/build_final.py":"notebook/archive/production_v34_21okt/build_production_v34.py",
                        "notebook/final.ipynb":"notebook/archive/production_v34_21okt/production_v34.ipynb"}
    for old,archived in production_archive.items():baselines[archived]=digest(ROOT/archived)==original[old]["sha256"]
    production=json.loads((ROOT/"results/v59/v41_lowvar_today/notebook_verification.json").read_text())
    production_checks={"builder_current":production["builder_sha256"]==digest(ROOT/"notebook/build_final.py"),
                       "notebook_current":production["notebook_sha256"]==digest(ROOT/"notebook/final.ipynb"),
                       "all_code_cells_executed":production["all_code_cells_executed"] is True,
                       "csv_current":production["csv_sha256"]==digest(ROOT/"results/v59/v41_lowvar_today/submission.csv"),
                       "embedded_sources_current":all(digest(ROOT/"eda"/n)==s for n,s in production["source_sha256"].items())}
    errors.extend(f"Production reproduction evidence mismatch: {name}" for name,okay in production_checks.items() if not okay)
    archived_ssl="research/locked_recipes/156_kumo_ssl_context_pre_class_threshold.py"
    baselines[archived_ssl]=digest(ROOT/archived_ssl)==original["eda/156_kumo_ssl_context.py"]["sha256"]
    errors.extend(f"Changed baseline: {p}" for p, okay in baselines.items() if not okay)
    sample = pd.read_csv(ROOT/"data/sample_submission.csv")
    assert len(sample)==30000 and sample.application_id.is_unique
    for p in sorted((ROOT/"results/teman").glob("*.csv")):
        frame = pd.read_csv(p)
        prob = frame.default_probability.to_numpy(float)
        assert list(frame)==["application_id", "default_probability"], p
        assert len(frame)==30000 and frame.application_id.is_unique, p
        assert set(frame.application_id)==set(sample.application_id), p
        assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all(), p
        submissions.append({"path": str(p.relative_to(ROOT)), "rows": len(frame), "sha256": digest(p),
                            "sample_order": frame.application_id.equals(sample.application_id),
                            "minimum": prob.min(), "maximum": prob.max()})
    # Validate the two upstream model/license files against their original bytes.
    licenses = [r for r in mapping if Path(r["destination"]).name in {"NOTICE", "LICENSE"}]
    assert len(licenses)>=2
    for r in licenses:
        assert digest(ROOT/r["destination"])==r["original_sha256"], r
    with (OUT/"159_migration_hashes.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(checked[0])); w.writeheader(); w.writerows(checked)
    (AUDIT/"indeks_modul_teman.json").write_text(json.dumps(sources, indent=2))
    result={"files_checked": len(checked), "adapted_files": sum(r["adapted"] for r in checked),
            "sources_parsed": len(sources), "notebooks": notebooks, "submissions": submissions,
            "baseline_byte_identity": baselines, "licenses_preserved": [r["destination"] for r in licenses],
            "errors": errors, "model_executed": False,
            "intentional_research_extension": {"path":"eda/156_kumo_ssl_context.py",
                "reason":"Optional asymmetric class thresholds after user reauthorized SSL; original source archived byte-identical",
                "current_sha256":digest(ROOT/"eda/156_kumo_ssl_context.py"),"original_archive":archived_ssl},
            "intentional_production_update":{"original_archive":production_archive,"current_recipe":"v41 low-variance released as v59",
                "reproduction_checks":production_checks,"verification_mode":"actual local replay; no fresh GPU training today"},
            "limits": ["Notebook cells compiled, not trained or fully executed.",
                       "Snapshot lacks OOF/test per-fold predictions, model files and feature cache.",
                       "Historical experiment JSON metrics are reports, not rerun validation_v2 results."]}
    (OUT/"159_integration_verify.json").write_text(json.dumps(result, indent=2))
    fig, ax=plt.subplots(figsize=(8,3))
    labels=["Files SHA256", "Python parsed", "Notebook code cells", "Submission CSVs", "Baseline unchanged"]
    values=[len(checked),len(sources),sum(r["code_cells_compiled"] for r in notebooks),len(submissions),sum(baselines.values())]
    ax.barh(labels, values); ax.set(xlabel="Checks (different units)", title="Integration evidence; no model quality claim")
    for i,v in enumerate(values): ax.text(v,i,f" {v}",va="center")
    fig.tight_layout(); fig.savefig(OUT/"159_integration_verify.png",dpi=130); plt.close(fig)
    print(json.dumps({"files":len(checked),"sources":len(sources),"notebooks":len(notebooks),
                      "cells":sum(r["code_cells_compiled"] for r in notebooks),"csvs":len(submissions),
                      "baseline_checks":baselines,"errors":errors},indent=2))
    assert not errors, errors


if __name__ == "__main__":
    main()
