"""Builds notebook/v72.ipynb: Kaggle kernel for 9 GBDT + repayment Markov chain, 3 seeds, GPU T4 x2.
Repo files are embedded as strings, written to /kaggle/working/code/eda and run there, so Kaggle runs the local code.
Usage: .venv/bin/python notebook/build_v72.py  ->  notebook/v72.ipynb and results/v72k/
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EDA = ROOT / "eda"
FILES = ["common.py", "features.py", "valid2.py", "251_repayment_markov.py", "kaggle_v72_run.py"]
SLUG = "logika-v72-trees-markov"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    compile(s, "<cell>", "exec") if not s.lstrip().startswith(("!", "%")) else None
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def main():
    cells = [
        md("# LOGIKA v72: 9 GBDT + repayment Markov chain\n\n"
           "Main features plus a per-customer repayment Markov chain (f251), then 9 gradient boosting models "
           "(LightGBM, XGBoost, CatBoost, three depths each) with validation_v2 over seeds 42, 7 and 2026. "
           "Submissions are equal-weight rank means."),
        code("%pip install -q polars==2.0.0 lightgbm==4.7.0 xgboost==3.4.1 catboost==1.2.10"),
    ]
    files = {f: (EDA / f).read_text() for f in FILES}
    files["common.py"] = files["common.py"].replace('DATA = ROOT / "data"', 'DATA = Path(INPUT)')
    files["common.py"] = files["common.py"].replace("import polars as pl\n", "import polars as pl\n\nINPUT = next(p.parent for p in Path('/kaggle/input').rglob('train_applications.csv'))\n", 1)
    assert "Path(INPUT)" in files["common.py"]
    cells.append(code("import os, pathlib\nCODE = pathlib.Path('/kaggle/working/code/eda')\nCODE.mkdir(parents=True, exist_ok=True)\n"
                      f"FILES = {json.dumps(files)}\n"
                      "for name, text in FILES.items():\n    (CODE / name).write_text(text)\nprint(sorted(os.listdir(CODE)))"))
    cells.append(code("import subprocess, sys\n"
                      "def sh(args, log=None):\n"
                      "    r = subprocess.run(args, cwd=CODE, capture_output=True, text=True)\n"
                      "    print(r.stdout[-4000:], r.stderr[-4000:])\n"
                      "    assert r.returncode == 0\n"
                      "sh([sys.executable, '-c', 'import common as c; f = c.features(); print(f.shape)'])\n"
                      "sh([sys.executable, '251_repayment_markov.py'])"))
    cells.append(code("import polars as pl\n"
                      "f = pl.read_parquet('/kaggle/working/code/data_pq/features.parquet').sort('application_id')\n"
                      "num = [c for c, d in f.schema.items() if d.is_numeric()]\n"
                      "sums = f.select([pl.col(c).cast(pl.Float64).fill_nan(None).sum().alias(c) for c in num])\n"
                      "os.makedirs('/kaggle/working/v72', exist_ok=True)\n"
                      "sums.write_parquet('/kaggle/working/v72/feature_sums.parquet')\nprint(f.shape)"))
    cells.append(code("import time\nenv = dict(os.environ)\nprocs = {}\n"
                      "for g, gpu in (('lgb', ''), ('xgb', '0'), ('cat', '1')):\n"
                      "    e = dict(env, CUDA_VISIBLE_DEVICES=gpu)\n"
                      "    procs[g] = subprocess.Popen([sys.executable, 'kaggle_v72_run.py', g], cwd=CODE, env=e,\n"
                      "                                stdout=open(f'/kaggle/working/v72/run_{g}.log', 'w'), stderr=subprocess.STDOUT)\n"
                      "t0 = time.time()\n"
                      "while any(p.poll() is None for p in procs.values()):\n"
                      "    time.sleep(60)\n"
                      "    print(f'{(time.time() - t0) / 60:.0f} min', {g: p.poll() for g, p in procs.items()}, flush=True)\n"
                      "for g in procs:\n    print(g, procs[g].returncode)\n"
                      "    print(open(f'/kaggle/working/v72/run_{g}.log').read()[-1500:])"))
    cells.append(code("import numpy as np\nfrom scipy.stats import rankdata, ks_2samp\n"
                      "OUT = pathlib.Path('/kaggle/working/code/v72')\n"
                      "ss = pl.read_csv(next(pathlib.Path('/kaggle/input').rglob('sample_submission.csv')))\n"
                      "def blend(models, fname):\n"
                      "    oof = [pl.read_parquet(OUT / f'oof_{m}.parquet') for m in models]\n"
                      "    y = oof[0]['target'].to_numpy()\n"
                      "    o = np.mean([rankdata(d[m].to_numpy()) for d, m in zip(oof, models)], 0)\n"
                      "    ks = ks_2samp(o[y == 1], o[y == 0]).statistic\n"
                      "    t = ss.select('application_id')\n"
                      "    for m in models:\n        t = t.join(pl.read_parquet(OUT / f'test_{m}.parquet'), on='application_id', how='left')\n"
                      "    r = np.mean([rankdata(t[m].to_numpy()) / t.height for m in models], 0)\n"
                      "    t.select('application_id').with_columns(default_probability=pl.Series(r)).write_csv(f'/kaggle/working/{fname}')\n"
                      "    print(fname, 'OOF KS', round(ks, 5), [round(ks_2samp(d[m].to_numpy()[y == 1], d[m].to_numpy()[y == 0]).statistic, 5) for d, m in zip(oof, models)])\n"
                      "six = ['lgb', 'lgb_s', 'xgb', 'xgb_s', 'cat', 'cat_s']\n"
                      "blend(six, 'submission_six.csv')\nblend(six + ['lgb_x', 'xgb_d8', 'cat_d8'], 'submission_nine.csv')\n"
                      "import shutil\nshutil.copytree(OUT, '/kaggle/working/v72/preds', dirs_exist_ok=True)"))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (ROOT / "notebook/v72.ipynb").write_text(json.dumps(nb, indent=1))
    k = ROOT / "results/v72k"
    k.mkdir(parents=True, exist_ok=True)
    (k / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"athillazaidan/{SLUG}", "title": "Logika v72 trees markov", "code_file": f"{SLUG}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "keywords": [], "dataset_sources": [], "kernel_sources": [],
        "competition_sources": ["data-science-competition-logika-ui-2026"], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"}, indent=1))
    print("wrote", k)


if __name__ == "__main__":
    main()
