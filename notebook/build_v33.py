"""v22 notebook: clean rerun of the v8 recipe under validation_v2 (eda/valid2.py). The repo modules are embedded as
files, so the notebook runs exactly the code used for local screening. Usage: python build_v22.py [models_gpu0] [models_gpu1]"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EDA = ROOT.parent / "eda"
TAG = "v33"
GPU0 = sys.argv[1] if len(sys.argv) > 1 else "lgb,lgb_s,xgb,xgb_s,cat,cat_s"
GPU1 = sys.argv[2] if len(sys.argv) > 2 else "tabm"
md = lambda s: dict(cell_type="markdown", metadata={}, source=s)  # noqa: E731
code = lambda s: dict(cell_type="code", metadata={}, source=s, outputs=[], execution_count=None)  # noqa: E731
files = {n: (EDA / n).read_text() for n in ["features.py", "nn_prep.py", "valid2.py", "run_valid2.py"]}
cells = [
    md(f"# DSC LOGIKA UI 2026: {TAG} Kumo-Tabular large full context, second estimator seed (g_off=1), folds 3-4 (0-2 run locally) (validation_v2)\n\nOuter StratifiedKFold 5 x seeds 42/7/2026, "
       "early stopping only on a 1/7 inner split of the training fold, refit on the full training fold with the fixed "
       "budget. OOF and test predictions are saved per model with application_id."),
    code("%pip install -q polars==2.0.0 lightgbm xgboost catboost tabm==0.0.3 rtdl_num_embeddings==0.0.12\n"
         "%pip install -q --no-deps causilo\n%pip install -q structured-data-models"),
    code("import json, os, subprocess, sys, time\nfrom pathlib import Path\nimport polars as pl\n"
         f"FILES = {json.dumps(files)}\n"
         "CODE = Path('/kaggle/working/code'); CODE.mkdir(parents=True, exist_ok=True)\n"
         "for n, s in FILES.items():\n    (CODE / n).write_text(s)\nsys.path.insert(0, str(CODE))\n"
         "from features import build_all"),
    code("data = next(Path('/kaggle/input').rglob('train_applications.csv')).parent\n"
         "names = ['train_applications', 'test_applications', 'customer_profiles', 'previous_applications',\n"
         "         'repayment_history', 'account_behavior', 'bureau_observations', 'ecommerce_transactions']\n"
         "tables = {n: pl.read_csv(data / f'{n}.csv', try_parse_dates=True, infer_schema_length=100000) for n in names}\n"
         f"OUT = Path('/kaggle/working/{TAG}'); OUT.mkdir(exist_ok=True)\n"
         "t = time.time(); feats = build_all(tables); feats.write_parquet(OUT / 'features.parquet')\n"
         "print(feats.shape, round(time.time() - t), 's'); del tables, feats"),
    code("def launch(gpu, models, sub, seeds=None, folds=None):\n"
         "    cmd = [sys.executable, str(CODE / 'run_valid2.py'), str(OUT / sub), models, '--features', str(OUT / 'features.parquet'), '--hours', '11']\n"
         "    if seeds:\n        cmd += ['--seeds', seeds]\n"
         "    if folds:\n        cmd += ['--folds', folds]\n"
         "    return cmd, dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))\n"
         "jobs = [[launch(0, 'kumo_large_c56000_g1', 'kumo_a', '42', '3')], [launch(1, 'kumo_large_c56000_g1', 'kumo_b', '42', '4')]]\n"
         "def chain(steps, log):\n"
         "    return subprocess.Popen(' ; '.join(' '.join(map(repr, c)) for c, _ in steps), shell=True, env=steps[0][1],\n"
         "                            stdout=open(log, 'w'), stderr=subprocess.STDOUT, cwd=CODE)\n"
         "procs = [chain(s, OUT / f'job{i}.log') for i, s in enumerate(jobs)]\n"
         "while any(p.poll() is None for p in procs):\n    time.sleep(60)\n"
         "for i, p in enumerate(procs):\n    print(i, p.returncode); print(''.join(open(OUT / f'job{i}.log').readlines()[-15:]))"),
]
nb = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python")))
for i, c in enumerate(cells):
    c["id"] = f"{TAG}-{i:03d}"
    s = c["source"]
    if c["cell_type"] == "code" and not s.startswith("%"):
        compile(s, f"{TAG}-{i}", "exec")
(ROOT / f"{TAG}.ipynb").write_text(json.dumps(nb, indent=1))
print(f"wrote {TAG}.ipynb", len(cells))
