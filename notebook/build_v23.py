"""v23 notebook: TabArena zoo fold-0 screen (eda/zoo_screen.py) for models that do not fit the local 8 GB GPU.
Two job chains, one per T4. Usage: python build_v23.py  (edit JOBS)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EDA = ROOT.parent / "eda"
TAG = "v25"
JOBS = [  # (gpu, models, extra args)
    [(0, "LimiXModel", ["--max-train", "15000"]), (0, "TabICLv2Model", ["--max-train", "30000"]),
     (0, "MitraModel", ["--max-train", "8000", "--hp", '{"ag.max_memory_usage_ratio": 10}'])],
    [(1, "TabDPTModel TabDPTTurboModel NoriModel Nori30MModel", ["--hp", '{"ag.max_memory_usage_ratio": 4}'])],
]
md = lambda s: dict(cell_type="markdown", metadata={}, source=s)  # noqa: E731
code = lambda s: dict(cell_type="code", metadata={}, source=s, outputs=[], execution_count=None)  # noqa: E731
files = {n: (EDA / n).read_text() for n in ["features.py", "nn_prep.py", "valid2.py", "zoo_screen.py"]}
cells = [
    md(f"# DSC LOGIKA UI 2026: {TAG} TabArena zoo screen\n\nvalidation_v2 fold 0 (seed 42): each model is fitted by "
       "AutoGluon on the inner-fit rows with the inner 1/7 stop rows as tuning data, default TabArena hyperparameters; "
       "the outer fold is only predicted."),
    code("%pip install -q polars==2.0.0 'tabarena[tabicl,tabdpt,limix,iltm,nori]==0.1.0' 'autogluon.tabular[mitra]>=1.6,<1.7'"),
    code("import json, os, subprocess, sys, time\nfrom pathlib import Path\nimport polars as pl\n"
         f"FILES = {json.dumps(files)}\n"
         "CODE = Path('/kaggle/working/code'); CODE.mkdir(parents=True, exist_ok=True)\n"
         "for n, s in FILES.items():\n    (CODE / n).write_text(s)\nsys.path.insert(0, str(CODE))\n"
         "from features import build_all\nimport torch; print(torch.__version__, torch.cuda.device_count())"),
    code("data = next(Path('/kaggle/input').rglob('train_applications.csv')).parent\n"
         "names = ['train_applications', 'test_applications', 'customer_profiles', 'previous_applications',\n"
         "         'repayment_history', 'account_behavior', 'bureau_observations', 'ecommerce_transactions']\n"
         "tables = {n: pl.read_csv(data / f'{n}.csv', try_parse_dates=True, infer_schema_length=100000) for n in names}\n"
         f"OUT = Path('/kaggle/working/{TAG}'); OUT.mkdir(exist_ok=True)\n"
         "feats = build_all(tables); feats.write_parquet(OUT / 'features.parquet'); print(feats.shape); del tables, feats"),
    code(f"JOBS = {json.dumps(JOBS)}\n"
         "def cmd(models, extra, i):\n"
         "    return ' '.join(map(repr, [sys.executable, str(CODE / 'zoo_screen.py'), *models.split(), '--minutes', '50',\n"
         "                                '--features', str(OUT / 'features.parquet'), '--out', str(OUT / f'zoo{i}'), *extra]))\n"
         "procs = []\n"
         "for i, chain in enumerate(JOBS):\n"
         "    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(chain[0][0]))\n"
         "    procs.append(subprocess.Popen(' ; '.join(cmd(m, e, i) for _, m, e in chain), shell=True, env=env, cwd=CODE,\n"
         "                                  stdout=open(OUT / f'job{i}.log', 'w'), stderr=subprocess.STDOUT))\n"
         "while any(p.poll() is None for p in procs):\n    time.sleep(60)\n"
         "for i in range(len(JOBS)):\n    print(open(OUT / f'zoo{i}' / 'log.csv').read() if (OUT / f'zoo{i}' / 'log.csv').exists() else f'job {i}: no log')"),
]
nb = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python")))
for i, c in enumerate(cells):
    c["id"] = f"{TAG}-{i:03d}"
    if c["cell_type"] == "code" and not c["source"].startswith("%"):
        compile(c["source"], f"{TAG}-{i}", "exec")
(ROOT / f"{TAG}.ipynb").write_text(json.dumps(nb, indent=1))
print(f"wrote {TAG}.ipynb", len(cells))
