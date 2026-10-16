"""v20 notebook: v10 feature cell (eda/features.py) + ranking / level-2 stacker experiment."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
nb = json.loads((ROOT / "v10.ipynb").read_text())
md = lambda s: dict(cell_type="markdown", metadata={}, source=s)  # noqa: E731
code = lambda s: dict(cell_type="code", metadata={}, source=s, outputs=[], execution_count=None)  # noqa: E731
cells = [md("---\n\n# DSC LOGIKA UI 2026: Causilo and TabPFN with 16 Estimators\n\n*v20 experiment.*"),
         md("Causilo and TabPFN with 16 estimators (v8 used 4) on the full training fold (~56k rows), top 100 / 200 numeric "
            "features by training-fold KS plus categoricals. Query rows are scored in chunks of 5000. "
            "Five outer folds (seed 42), folds spread over both T4s."),
         code("%pip install -q tabpfn polars==2.0.0 pyarrow scipy\n%pip install -q --no-deps causilo"),
         code("import hashlib\nimport json\nimport os\nimport time\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport polars as pl"),
         nb["cells"][12],
         code((ROOT / "v20_experiment.py").read_text()),
         code("main_v20(build_all)")]
for i, c in enumerate(cells):
    c["id"] = f"v20-{i:03d}"
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
        s = "".join(c["source"])
        if not s.startswith("%"):
            compile(s, f"v20-{i}", "exec")
nb["cells"] = cells
(ROOT / "v20.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote v20.ipynb", len(cells))
