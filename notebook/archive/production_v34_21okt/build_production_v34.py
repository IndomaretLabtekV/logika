"""Final competition notebook (ref/notebook/SKILL.md layout). Reproduces results/v34/submission.csv on Kaggle T4x2:
6 GBDT (validation_v2, 3 seeds, fold-averaged test) 1/3 + Causilo (5 folds) 1/3 + Kumo-Tabular large with the whole
training set as context 1/3, rank-averaged. The repo modules are written into the notebook with %%writefile so the
notebook runs exactly the code used for validation. Usage: python notebook/build_final.py"""
import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
EDA = ROOT.parent / "eda"
OUT = ROOT / "final.ipynb"
HR = "---"
MODULES = ["features.py", "nn_prep.py", "valid2.py", "run_valid2.py", "kumo_fulltrain.py"]
TREES = ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid4().hex[:8], "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": uuid4().hex[:8], "metadata": {}, "outputs": [], "source": source}


def section(title: str, anchor: str) -> dict:
    return md(f"{HR}\n\n# {title} <a name=\"{anchor}\"></a>\n\n{HR}")


cells = []

# ---------------------------------------------------------------------------
# Banner (3 cells) + TOC
# ---------------------------------------------------------------------------
cells.append(md(
    f"{HR}\n\n"
    "# Data Science Competition LOGIKA UI 2026\n\n"
    "*Prediksi risiko gagal bayar serius (DPD 90+ dalam 12 bulan) untuk pinjaman tenor 12 bulan, dinilai dengan "
    "statistik Kolmogorov-Smirnov.*"
))
cells.append(md(
    f"{HR}\n\n"
    "## DSCxxxx - lagi mager lomba bentar\n\n"
    "- Nama Anggota 1 (Ketua)\n"
    "- Nama Anggota 2\n"
    "- Nama Anggota 3"
))
cells.append(md(
    f"{HR}\n\n"
    "## Table of Contents\n\n"
    "1. [**Introduction**](#1)\n"
    "2. [**Initialization**](#2)\n"
    "3. [**Pipeline Modules**](#3)\n"
    "4. [**Feature Engineering**](#4)\n"
    "5. [**Model Training**](#5)\n"
    "6. [**Blending and Submission**](#6)\n"
    "7. [**Conclusion**](#7)\n"
))

# ---------------------------------------------------------------------------
# Section 1: Introduction
# ---------------------------------------------------------------------------
cells.append(section("Introduction", "1"))
cells.append(md(
    "## Overview\n\n"
    "*Lender perlu memisahkan pemohon yang akan mengalami kesulitan bayar serius dari yang tidak, memakai data yang "
    "tersedia sampai saat keputusan kredit. Data kompetisi berisi aplikasi pinjaman terpilih, profil customer, riwayat "
    "aplikasi dan cicilan sebelumnya, perilaku rekening bulanan, laporan biro kredit, dan transaksi e-commerce.*"
))
cells.append(md(
    "## Aim\n\n"
    "*Notebook ini membangun skor risiko untuk 30.000 aplikasi test dan menulis `submission.csv` "
    "(`application_id`, `default_probability`). Skor berasal dari rata-rata rank tiga keluarga model yang "
    "dipilih dengan validasi bersih (validation_v2).*"
))
cells.append(md(
    "## Metric\n\n"
    "***Kolmogorov-Smirnov (KS)** mengukur jarak maksimum antara distribusi kumulatif skor kelas gagal bayar dan "
    "kelas lancar. KS hanya bergantung pada urutan skor, jadi rata-rata rank antar model aman dipakai.*\n\n"
    "$$\\text{KS} = \\max_t \\left| F_1(t) - F_0(t) \\right|$$"
))
cells.append(md(
    "## Dataset\n\n"
    "*Data berasal dari kompetisi Kaggle Data Science Competition LOGIKA UI 2026 "
    "(`data-science-competition-logika-ui-2026`).*\n\n"
    "*Train berisi 70.000 aplikasi (bad rate sekitar 13,5%) dan test 30.000 aplikasi, satu aplikasi terpilih per "
    "customer, dengan periode keputusan yang sama sepanjang 2025. Semua tabel bantu sudah dipotong di "
    "`decision_datetime`; kode fitur tetap memfilter cutoff ini secara eksplisit.*\n\n"
    "**Attributes**\n\n"
    "1.  `train_applications` / `test_applications` : pinjaman terpilih (jumlah, bunga, cicilan, tujuan, waktu)\n"
    "2.  `customer_profiles`                         : demografi dan pendapatan yang dideklarasikan\n"
    "3.  `previous_applications`                     : aplikasi historis (disetujui, ditolak, dibatalkan)\n"
    "4.  `repayment_history`                         : snapshot cicilan pinjaman lama pada cutoff\n"
    "5.  `account_behavior`                          : arus kas rekening per bulan lengkap\n"
    "6.  `bureau_observations`                       : laporan tiga biro kredit (skor, utang, inquiry)\n"
    "7.  `ecommerce_transactions`                    : item belanja online (kategori, harga, status)\n\n"
    "**Target**\n"
    " `target` : 1 jika DPD akhir hari mencapai 90 dalam 12 bulan, 0 jika tidak"
))
cells.append(md(
    "## Approach: Rank Blend of Gradient Boosting and Tabular Foundation Models\n\n"
    "*Sekitar 230 fitur level customer dibangun dari semua tabel (agregat repayment, burst belanja, premium harga per "
    "kategori, biro, arus kas). Tiga keluarga model dilatih di atasnya: enam GBDT (LightGBM, XGBoost, CatBoost, "
    "masing-masing versi dalam dan dangkal), Causilo, dan NVIDIA Kumo-Tabular large. Dua yang terakhir adalah model "
    "in-context learning pretrained yang membaca baris train sebagai context tanpa pelatihan ulang bobot.*\n\n"
    "*GBDT dan Causilo memakai validation_v2: StratifiedKFold 5 fold, early stopping hanya di inner split 1/7 dari "
    "train fold, lalu refit di seluruh train fold dengan jumlah ronde tetap; prediksi test adalah rata-rata model "
    "fold. Kumo memakai seluruh 70.000 baris train sebagai context dan memprediksi test sekali.*\n\n"
    "```\n"
    "features (~230 kolom)\n"
    "    |\n"
    "    +-- 6 GBDT x 3 seed x 5 fold  -> rank rata-rata  (bobot 1/3)\n"
    "    +-- Causilo x 5 fold          -> rank            (bobot 1/3)\n"
    "    +-- Kumo-Tabular, context 70k -> rank            (bobot 1/3)\n"
    "    |\n"
    "rata-rata rank -> default_probability\n"
    "```\n\n"
    "*Di validasi bersih (OOF, seed 42) blend ini unggul +0,0022 KS atas resep blend v8 (paired bootstrap "
    "P>0 = 0,93, menang di 4 dari 5 fold).*"
))

# ---------------------------------------------------------------------------
# Section 2: Initialization
# ---------------------------------------------------------------------------
cells.append(section("Initialization", "2"))
cells.append(md(
    "## Environment Setup\n\n"
    "Notebook ini dirancang untuk Kaggle dengan akselerator GPU T4 x2 dan internet aktif (unduh bobot pretrained "
    "Causilo dan Kumo dari Hugging Face). Waktu jalan tercatat sekitar 5 jam: Kumo di GPU 0 (sekitar 4,5 jam) "
    "berjalan paralel dengan GBDT dan Causilo di GPU 1 (sekitar 2 jam)."
))
cells.append(code("!nvidia-smi"))
cells.append(md("The following cell installs all libraries used in this notebook."))
cells.append(code(
    "import subprocess, sys\n"
    "from pathlib import Path\n"
    "if Path('/kaggle/input').exists():  # pinned versions of the validated run; a local venv already has them\n"
    "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'polars==2.0.0', 'lightgbm==4.6.0', 'xgboost==3.4.1',\n"
    "                    'catboost==1.2.10', 'structured-data-models==0.2.1'], check=True)\n"
    "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'causilo==1.0.3'], check=True)"
))
cells.append(md("## Import Libraries"))
cells.append(code(
    "import os\n"
    "import random\n"
    "import subprocess\n"
    "import sys\n"
    "import time\n"
    "from pathlib import Path\n"
    "\n"
    "import numpy as np\n"
    "import polars as pl\n"
    "from scipy.stats import rankdata"
))
cells.append(md("## Seed Everything"))
cells.append(code(
    "def seed_everything(seed: int = 42):\n"
    "    random.seed(seed)\n"
    "    os.environ['PYTHONHASHSEED'] = str(seed)\n"
    "    np.random.seed(seed)\n"
    "\n"
    "seed_everything(42)"
))
cells.append(md(
    "## Settings\n\n"
    "Semua path, daftar model, dan bobot blend dipusatkan di sini. Seed outer fold untuk GBDT adalah 42, 7, dan 2026; "
    "Causilo dan Kumo memakai seed 42."
))
cells.append(code(
    "class Settings:\n"
    "    SEED       = 42\n"
    "    _ON_KAGGLE = Path('/kaggle/input').exists()\n"
    "\n"
    "    DATA_DIR   = next(Path('/kaggle/input' if _ON_KAGGLE else '../data').rglob('train_applications.csv')).parent\n"
    "    OUTPUT_DIR = Path('/kaggle/working') if _ON_KAGGLE else Path('../results/final').resolve()  # absolute: subprocesses run in CODE_DIR\n"
    "    CODE_DIR   = OUTPUT_DIR / 'pipeline'\n"
    "    RUN_DIR    = OUTPUT_DIR / 'run'\n"
    "\n"
    "    TABLES = ['train_applications', 'test_applications', 'customer_profiles', 'previous_applications',\n"
    "              'repayment_history', 'account_behavior', 'bureau_observations', 'ecommerce_transactions']\n"
    f"    TREES  = {TREES!r}\n"
    "    TREE_SEEDS = '42,7,2026'\n"
    "    ICL_SEED   = '42'\n"
    "\n"
    "    WEIGHTS = {**{m: 1 / 18 for m in TREES}, 'causilo': 1 / 3, 'kumo_full70k': 1 / 3}\n"
    "\n"
    "CFG = Settings()\n"
    "for d in (CFG.OUTPUT_DIR, CFG.CODE_DIR, CFG.RUN_DIR):\n"
    "    d.mkdir(parents=True, exist_ok=True)\n"
    "print(CFG.DATA_DIR)"
))
cells.append(md(
    "## Load Dataset\n\n"
    "Delapan tabel dibaca dengan parsing tanggal otomatis. Tidak ada visualisasi di sini."
))
cells.append(code(
    "tables = {n: pl.read_csv(CFG.DATA_DIR / f'{n}.csv', try_parse_dates=True, infer_schema_length=100000)\n"
    "          for n in CFG.TABLES}\n"
    "for n, t in tables.items():\n"
    "    print(f'{n:24s} {t.shape}')"
))

# ---------------------------------------------------------------------------
# Section 3: Pipeline modules
# ---------------------------------------------------------------------------
cells.append(section("Pipeline Modules", "3"))
cells.append(md(
    "Kode pipeline ditulis ke folder `pipeline/` dengan `%%writefile`, persis sama dengan modul yang dipakai saat "
    "validasi. `features.py` membangun fitur, `nn_prep.py` berisi transformasi rank-gaussian, `valid2.py` berisi "
    "protokol validation_v2 dan semua pembungkus model, `run_valid2.py` menjalankan model per fold dan menyimpan "
    "prediksi OOF/test per `application_id`, dan `kumo_fulltrain.py` menjalankan Kumo dengan seluruh train sebagai "
    "context. Import di dalam modul ini dieksekusi oleh subprocess, bukan oleh kernel notebook."
))
cells.append(code("sys.path.insert(0, str(CFG.CODE_DIR))\nprint(CFG.CODE_DIR)"))
for name in MODULES:
    cells.append(code(f"%%writefile {{CFG.CODE_DIR}}/{name}\n" + (EDA / name).read_text().rstrip("\n")))

# ---------------------------------------------------------------------------
# Section 4: Feature engineering
# ---------------------------------------------------------------------------
cells.append(section("Feature Engineering", "4"))
cells.append(md(
    "`build_all` menggabungkan fitur aplikasi, biro, repayment, rekening, e-commerce (termasuk premium harga per "
    "kategori dengan referensi expanding per bulan keputusan), dan aplikasi sebelumnya menjadi satu baris per "
    "aplikasi. Setiap builder memfilter baris tabel bantu di `decision_datetime`."
))
cells.append(code(
    "build_all = __import__('features').build_all\n"
    "t0 = time.time()\n"
    "feats = build_all(tables)\n"
    "feats.write_parquet(CFG.RUN_DIR / 'features.parquet')\n"
    "print(feats.shape, f'{time.time() - t0:.0f} s')\n"
    "print(feats.group_by('is_train').agg(pl.len(), pl.col('target').mean().alias('bad_rate')))\n"
    "del tables"
))
cells.append(md(
    "#### Insights\n\n"
    "> Train dan test memakai builder yang sama dalam satu pass, sehingga tidak ada perbedaan definisi fitur antar "
    "split. Label hanya dipakai di langkah model, di dalam train fold."
))

# ---------------------------------------------------------------------------
# Section 5: Model training
# ---------------------------------------------------------------------------
cells.append(section("Model Training", "5"))
cells.append(md(
    "Dua rantai proses berjalan paralel, satu per GPU. GPU 0 menjalankan Kumo-Tabular large dengan 70.000 baris "
    "train sebagai context. GPU 1 menjalankan enam GBDT (3 seed outer x 5 fold) lalu Causilo (5 fold). Setiap model "
    "menulis `oof_<model>.parquet`, `test_<model>.parquet`, dan `log_<model>.csv` ke `run/`."
))
cells.append(code(
    "def cmd(script, *args):\n"
    "    return ' '.join(map(repr, [sys.executable, str(CFG.CODE_DIR / script), *args]))\n"
    "\n"
    "feat_path = str(CFG.RUN_DIR / 'features.parquet')\n"
    "chains = {\n"
    "    0: [cmd('kumo_fulltrain.py', str(CFG.RUN_DIR), '--features', feat_path)],\n"
    "    1: [cmd('run_valid2.py', str(CFG.RUN_DIR / 'gbdt'), ','.join(CFG.TREES), '--seeds', CFG.TREE_SEEDS,\n"
    "            '--features', feat_path, '--hours', '11'),\n"
    "        cmd('run_valid2.py', str(CFG.RUN_DIR / 'fm'), 'causilo', '--seeds', CFG.ICL_SEED,\n"
    "            '--features', feat_path, '--hours', '11')],\n"
    "}\n"
    "import torch\n"
    "if torch.cuda.device_count() < 2:  # one GPU (local / single T4): run both chains one after the other on GPU 0\n"
    "    chains = {0: chains[0] + chains[1]}\n"
    "procs = {}\n"
    "for gpu, steps in chains.items():\n"
    "    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')\n"
    "    procs[gpu] = subprocess.Popen(' && '.join(steps), shell=True, env=env, cwd=CFG.CODE_DIR,\n"
    "                                  stdout=open(CFG.RUN_DIR / f'gpu{gpu}.log', 'w'), stderr=subprocess.STDOUT)\n"
    "t0 = time.time()\n"
    "while any(p.poll() is None for p in procs.values()):\n"
    "    time.sleep(120)\n"
    "for gpu, p in procs.items():\n"
    "    print(f'GPU {gpu}: exit {p.returncode}, {(time.time() - t0) / 3600:.1f} h')\n"
    "    print(''.join(open(CFG.RUN_DIR / f'gpu{gpu}.log').readlines()[-6:]))\n"
    "assert all(p.returncode == 0 for p in procs.values())"
))
cells.append(md(
    "Skor KS OOF per model (validation_v2) dibaca dari log. Kumo tidak punya OOF di notebook ini karena dijalankan "
    "sekali di seluruh train; skor OOF-nya diukur terpisah saat validasi (0,52665, context penuh per fold)."
))
cells.append(code(
    "logs = pl.concat([pl.read_csv(p) for p in sorted(CFG.RUN_DIR.rglob('log_*.csv'))], how='diagonal')\n"
    "print(logs.group_by('model').agg(pl.col('ks').mean().alias('mean_fold_ks'), pl.len().alias('fits'),\n"
    "                                 (pl.col('sec').sum() / 60).alias('minutes')).sort('model'))"
))

# ---------------------------------------------------------------------------
# Section 6: Blending and submission
# ---------------------------------------------------------------------------
cells.append(section("Blending and Submission", "6"))
cells.append(md(
    "Prediksi test setiap model diubah menjadi rank ternormalisasi lalu dirata-rata dengan bobot tetap: 1/18 untuk "
    "setiap GBDT (total 1/3), 1/3 Causilo, dan 1/3 Kumo. Bobot ini ditetapkan sebelum evaluasi akhir dan diuji "
    "dengan cross-fit di OOF, bukan dioptimasi di leaderboard."
))
cells.append(code(
    "test = None\n"
    "for p in sorted(CFG.RUN_DIR.rglob('test_*.parquet')):\n"
    "    t = pl.read_parquet(p)\n"
    "    test = t if test is None else test.join(t, on='application_id', how='inner', validate='1:1')\n"
    "test = test.sort('application_id')\n"
    "score = sum(w * rankdata(test[m].to_numpy()) / test.height for m, w in CFG.WEIGHTS.items()) / sum(CFG.WEIGHTS.values())\n"
    "sub = pl.DataFrame({'application_id': test['application_id'], 'default_probability': score})\n"
    "\n"
    "ref = pl.read_csv(CFG.DATA_DIR / 'sample_submission.csv')\n"
    "assert sub.height == ref.height == 30000 and sub['application_id'].n_unique() == 30000\n"
    "assert set(sub['application_id']) == set(ref['application_id'])\n"
    "assert np.isfinite(score).all() and score.min() >= 0 and score.max() <= 1\n"
    "sub.write_csv(CFG.OUTPUT_DIR / 'submission.csv')\n"
    "print(sub.head(), sub['default_probability'].describe(), sep='\\n')"
))
cells.append(md(
    "#### Insights\n\n"
    "> File `submission.csv` berisi 30.000 baris dengan ID unik yang sama persis dengan `sample_submission.csv` dan "
    "skor di [0, 1].\n\n"
    "> Karena KS hanya bergantung pada urutan, skor rank-average tidak perlu dikalibrasi menjadi probabilitas."
))

# ---------------------------------------------------------------------------
# Section 7: Conclusion
# ---------------------------------------------------------------------------
cells.append(section("Conclusion", "7"))
cells.append(md(
    "Blend rank enam GBDT, Causilo, dan Kumo-Tabular memberi KS OOF bersih 0,526 (resep blend lama 0,524) dan "
    "public leaderboard sekitar 0,563-0,564. Dua pelajaran utama dari validasi: early stopping di fold OOF membuat "
    "skor validasi lama terlalu optimis, dan model in-context learning hanya membantu bila dicampur dengan GBDT; "
    "blend tanpa GBDT turun tajam di leaderboard walaupun OOF-nya setara."
))

# ---------------------------------------------------------------------------
# Notebook writer
# ---------------------------------------------------------------------------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "cells": cells,
}
for c in cells:
    s = c["source"]
    if c["cell_type"] == "code" and not s.startswith(("%", "!")):
        compile(s, "cell", "exec")
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(cells)} cells -> {OUT}")
