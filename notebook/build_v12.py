"""Reuse the v11 notebook and change only the predeclared TabM capacity experiment."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
nb = json.loads((ROOT / "v11.ipynb").read_text())
settings = dict(lr=.002, wd=.0003, bins=48)
candidates = {
    "control": settings,
    "width128": dict(settings, model=dict(d_block=128)),
    "one_block": dict(settings, model=dict(n_blocks=1)),
    "dropout30": dict(settings, model=dict(dropout=.3)),
}
changed = dict(grid=0, model=0, output=0)
for cell in nb["cells"]:
    source = "".join(cell["source"])
    if cell["cell_type"] != "code" or source.startswith("%"):
        continue
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CANDIDATES" for t in node.targets):
            node.value = ast.parse(repr(candidates), mode="eval").body
            changed["grid"] += 1
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "tabm.TabM.make":
            node.keywords.append(ast.keyword(arg=None, value=ast.parse("settings.get(\"model\", {})", mode="eval").body))
            changed["model"] += 1
        if isinstance(node, ast.Constant) and node.value == "/kaggle/working/v11":
            node.value = "/kaggle/working/v12"
            changed["output"] += 1
    ast.fix_missing_locations(tree)
    cell["source"] = ast.unparse(tree)
    compile(cell["source"], "v12-cell", "exec")
    cell["outputs"] = []
    cell["execution_count"] = None
assert changed == dict(grid=1, model=1, output=1), changed
nb["cells"][0]["source"] = "---\n\n# DSC LOGIKA UI 2026: TabM Capacity\n\n*Reducing overfitting with a smaller or more regularized network.*"
nb["cells"][7]["source"] = "%pip install -q polars==2.0.0 tabm==0.0.3 rtdl_num_embeddings==0.0.12 scikit-learn==1.6.1 pandas pyarrow scipy matplotlib"
nb["cells"][14]["source"] = (
    "The v11 curves show early validation peaks while online training loss continues decreasing. "
    "This experiment changes only network capacity or dropout: width 128, one block, or dropout 0.3, "
    "compared with default TabM (width 512, two blocks with numerical embeddings, dropout 0.1). "
    "All candidates retain learning rate 0.002, weight decay 0.0003 and 48 bins. "
    "The same nested holdout protocol and five outer folds are reused for paired comparison. "
    "Repeated experimentation makes the outer folds exploratory rather than pristine. "
    "Parameter definitions follow the [official TabM implementation](https://github.com/yandex-research/tabm). "
    "No public gain is claimed before scoring."
)
for i, cell in enumerate(nb["cells"]):
    cell["id"] = f"v12-{i:03d}"
out = ROOT / "v12.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("Wrote", len(nb["cells"]), "cells ->", out, "; all three source changes verified")
