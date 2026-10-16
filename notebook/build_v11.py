"""Assemble v11 without running any notebook or model code."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def md(s):
    return dict(cell_type="markdown", metadata={}, source=s)


def code(s):
    return dict(cell_type="code", metadata={}, source=s, outputs=[], execution_count=None)


def section(title, anchor):
    return md(f"---\n\n# {title} <a name=\"{anchor}\"></a>\n\n---")


def clean(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                n.body.pop(0)
    return ast.unparse(tree)


feature = (ROOT.parent / "eda/features.py").read_text()
prep = (ROOT.parent / "eda/nn_prep.py").read_text()
runner = (ROOT / "v11_experiment.py").read_text()
helpers = "\n\n".join(ast.unparse(n) for n in ast.parse((ROOT / "v10_experiment.py").read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name in {"ks", "splits", "evaluate"})
imports = "\n".join(ast.unparse(n) for n in ast.parse(runner).body if isinstance(n, (ast.Import, ast.ImportFrom)))
cells = [
    md("---\n\n# DSC LOGIKA UI 2026: TabM Tuning\n\n*Nested validation of learning rate, regularization and numerical bins.*"),
    md("---\n\n## TeamName (DSCxxxx)\n\n- Name 1\n- Name 2\n- Name 3"),
    md("---\n\n## Table of Contents\n\n1. [Introduction](#1)\n2. [Initialization](#2)\n3. [Features](#3)\n4. [Experiment](#4)"),
    section("Introduction", "1"),
    md("## Overview and Aim\n\nThe experiment tests four fixed TabM configurations on the competition data. "
       "The headline metric is pooled out-of-fold KS. Control and tuned policies share five outer folds. "
       "Inside each outer training partition, an 80/20 split selects epochs using AUC and configuration using KS. "
       "Both policies are refitted on the outer training rows for the selected number of epochs. "
       "Outer labels never enter fitting, stopping or selection. Prior feature exploration means these are not pristine holdouts. "
       "This single-seed experiment is not an exact replication of the five-seed v9 model."),
    section("Initialization", "2"),
    md("Kaggle GPU and Internet must be enabled, with LOGIKA competition data attached. Execution is refused outside Kaggle CUDA."),
    code("%pip install -q polars==2.0.0 tabm rtdl_num_embeddings scikit-learn pandas pyarrow scipy matplotlib"),
    md("## Libraries\n\nPackage versions and source fingerprints are saved with the results."),
    code(imports + "\nFEATURE_SHA256 = " + repr(hashlib.sha256(feature.encode()).hexdigest()) +
         "\nNN_PREP_SHA256 = " + repr(hashlib.sha256(prep.encode()).hexdigest())),
    section("Features", "3"),
    md("The existing cutoff-aware features and neural input preparation are reused. Numerical transforms, missingness indicators "
       "and category vocabularies are fitted only on the current training rows. Unseen categories receive a reserved index."),
    code(clean(feature) + "\n\n" + clean(prep)),
    section("Experiment", "4"),
    md("Control uses learning rate 0.002, weight decay 0.0003 and 48 bins. The alternatives change one setting each: "
       "learning rate 0.0005, weight decay 0.01 or 16 bins. Training curves are saved after every epoch; "
       "online member BCE and validation ensemble BCE have different aggregation and must not be compared as identical losses. "
       "A candidate warrants larger replication only if pooled KS gains more than 0.002 and at least four outer folds improve. "
       "Bootstrap intervals are conditional on fitted models. A scored submission is required to establish public KS."),
    code(helpers + "\n\n" + clean(runner)),
    code("main_tabm()"),
    md("#### Insights\n\n> Results are not known before execution. Inspect metrics.json, comparison.png and learning_curves.png. "
       "The target public KS of 0.57768 remains unverified until an actual submission is scored."),
]
for i, cell in enumerate(cells):
    cell["id"] = f"v11-{i:03d}"
    if cell["cell_type"] == "code" and not cell["source"].startswith("%"):
        compile(cell["source"], f"cell-{i}", "exec")
nb = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
    kernelspec=dict(display_name="Python 3", language="python", name="python3"),
    language_info=dict(name="python", version="3.13.0")))
out = ROOT / "v11.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {len(cells)} cells -> {out}")
