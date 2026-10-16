"""Reuse the v10 data and evaluation harness for an EBM experiment."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
nb = json.loads((ROOT / "v10.ipynb").read_text())
replacement = ast.parse((ROOT / "v13_experiment.py").read_text())
replacements = {}
for n in replacement.body:
    if isinstance(n, ast.Assign):
        replacements[n.targets[0].id] = n
    elif isinstance(n, ast.FunctionDef):
        replacements[n.name] = n
changed = dict(BASE=0, CANDIDATES=0, run=0, output=0)
for cell in nb["cells"]:
    s = "".join(cell["source"])
    if cell["cell_type"] != "code" or s.startswith("%"):
        continue
    tree = ast.parse(s)
    for i, n in enumerate(tree.body):
        key = n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else None
        if key in replacements:
            tree.body[i] = replacements[key]
            changed[key] += 1
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id == "lgb":
            n.id = "interpret"
        elif isinstance(n, ast.arg) and n.arg == "lgb":
            n.arg = "interpret"
        elif isinstance(n, ast.keyword) and n.arg == "lightgbm":
            n.arg = "interpret"
        elif isinstance(n, ast.Constant) and n.value == "/kaggle/working/v10":
            n.value = "/kaggle/working/v13"
            changed["output"] += 1
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Import) and any(a.name == "lightgbm" for a in n.names))]
    ast.fix_missing_locations(tree)
    cell["source"] = ast.unparse(tree)
    cell["outputs"] = []
    cell["execution_count"] = None
assert changed == dict(BASE=1, CANDIDATES=1, run=1, output=1), changed
nb["cells"][0]["source"] = "---\n\n# DSC LOGIKA UI 2026: Explainable Boosting\n\n*Additive risk functions and a limited set of pairwise interactions.*"
nb["cells"][4]["source"] = (
    "The experiment tests whether a smooth additive model generalizes better than the existing ensembles. "
    "Two fixed EBM specifications are compared across five outer folds. The additive specification uses no interactions; "
    "the pairwise specification selects 20 interactions within each training fold. Four internal bags and internal "
    "15% validation support smoothing and early stopping. No outer validation labels enter fitting or stopping. "
    "For compatibility with the existing evaluator, output keys control and tuned denote additive and pairwise respectively; "
    "there is no hyperparameter search or automatic winner selection. Pooled OOF KS is the headline metric. "
    "This dataset has already been used for feature exploration, so these folds are not pristine holdouts."
)
nb["cells"][6]["source"] = "The notebook runs on Kaggle CPU with competition data attached and Internet enabled."
nb["cells"][7]["source"] = "%pip install -q interpret-core polars==2.0.0 pandas pyarrow scipy matplotlib"
nb["cells"][9]["source"] += "\nimport interpret\nfrom interpret.glassbox import ExplainableBoostingClassifier"
nb["cells"][14]["source"] = (
    "The input features and outer splits are unchanged from v10. Numerical and categorical bins are learned by EBM "
    "using the training partition only. Both candidates use 128 main bins, 16 interaction bins, four bags, "
    "learning rate 0.03 and min_samples_leaf 50. Each fit saves iteration counts, runtime and term importance. "
    "Settings follow the [official EBM API](https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html). "
    "The existing paired bootstrap is conditional on fitted models. A larger repeat requires delta KS above 0.002 "
    "and gains on at least four folds; actual public performance still requires a scored submission."
)
nb["cells"][17]["source"] = "#### Insights\n\n> Inspect metrics.json, comparison.png and fit_stats.json after execution. No improvement is claimed in advance. Public KS 0.57768 remains unverified."
for i, c in enumerate(nb["cells"]):
    c["id"] = f"v13-{i:03d}"
    if c["cell_type"] == "code" and not c["source"].startswith("%"):
        compile(c["source"], f"v13-cell-{i}", "exec")
out = ROOT / "v13.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("Wrote", len(nb["cells"]), "cells ->", out, "; four source changes verified")
