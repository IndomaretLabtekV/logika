"""v16 notebook: v10 feature cell (eda/features.py) + ranking / level-2 stacker experiment."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
nb = json.loads((ROOT / "v10.ipynb").read_text())
md = lambda s: dict(cell_type="markdown", metadata={}, source=s)  # noqa: E731
code = lambda s: dict(cell_type="code", metadata={}, source=s, outputs=[], execution_count=None)  # noqa: E731
cells = [md("---\n\n# DSC LOGIKA UI 2026: TabICLv2 with Chunked Queries\n\n*v16 experiment.*"),
         md("TabICLv2 (tabicl-classifier-v2) on the top 100 features by training-fold KS plus missingness-group flags. "
            "Context is a stratified 40k sample of the training fold; query rows (validation fold and test) are scored "
            "in chunks of 2000 so memory is set by the context. Five outer folds (seed 42), folds spread over both T4s."),
         code("%pip install -q tabicl polars==2.0.0 pyarrow scipy"),
         code("import hashlib\nimport json\nimport os\nimport time\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport polars as pl"),
         nb["cells"][12],
         code((ROOT / "v16_experiment.py").read_text()),
         code("main_v16(build_all)")]
for i, c in enumerate(cells):
    c["id"] = f"v16-{i:03d}"
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
        s = "".join(c["source"])
        if not s.startswith("%"):
            compile(s, f"v16-{i}", "exec")
nb["cells"] = cells
(ROOT / "v16.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote v16.ipynb", len(cells))
