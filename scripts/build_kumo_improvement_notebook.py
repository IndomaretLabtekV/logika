"""Build notebook 09 with visible EDA, feature, inference and export code."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import pprint
from pathlib import Path

import nbformat as nbf

from build_kumo_notebook import function

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/optimization/kumo_improvement"


def assignments(filename, names):
    source = (ROOT / "scripts" / filename).read_text()
    return "\n".join(
        ast.get_source_segment(source, n)
        for n in ast.parse(source).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
    )


def build():
    lock = json.loads((DEST / "locked_confirmation_recipe.json").read_text())
    notebook_lock = {
        key: lock[key]
        for key in ["screen_stage", "confirmation_stage", "rule", "warning"]
    }
    notebook_lock["selected"] = {
        key: lock["selected"][key] for key in ["name", "components"]
    }
    for key in ["reason", "prior_failed_confirmation"]:
        if key in lock:
            notebook_lock[key] = lock[key]
    previous = nbf.read(ROOT / "notebook/08_kumo_tabular.ipynb", as_version=4)
    cells = copy.deepcopy(previous.cells[:67])
    cells[
        0
    ].source = """# 09 — Improvement Kumo: EDA hingga submission kandidat

Notebook ini memperlihatkan eksperimen setelah checkpoint Kumo public **0,56262**.
Semua kode fitur, konteks model, validasi, dan ekspor tersedia di sel.
Gunakan Python 3.11, `environments/teman/requirements-kumo.txt`, GPU CUDA, dan data mentah kompetisi.

Enam kandidat pertama diuji pada tiga fold seed 8247. Resep dikunci sebelum
konfirmasi pada lima fold seed 31027, melawan baseline pada **fold yang sama**.
Pergantian split memakai label training yang sama; ini pemeriksaan konsistensi,
bukan holdout yang belum pernah digunakan. Angka KS lokal tidak menjamin skor Kaggle.
Public score kandidat baru **belum diketahui**.
"""
    prior_file = DEST / "confirmation_verdict_blend_failed.json"
    prior_failure = json.loads(prior_file.read_text()) if prior_file.exists() else None
    if prior_failure:
        cells[
            0
        ].source += "\nCampuran 75/25 hasil seleksi awal gagal pada pooled KS lima fold. Uji konteks penuh\nditetapkan setelah kegagalan tersebut, memakai lima split yang sama. Ini menambah\nseleksi model berulang; angka validasi tetap dapat optimistis.\n"
    cells[2].source = (
        cells[2]
        .source.replace('RUN_NAME = "08_kumo"', 'RUN_NAME = "09_kumo_improvement"')
        .replace(
            'CANONICAL_FILE = "submission_kumo.csv"',
            'CANONICAL_FILE = "submission_kumo_v2.csv"',
        )
        .replace("PUBLIC_SCORE = 0.56262", "PUBLIC_SCORE = None")
    )
    cells[2].source += "\nimport re\nfrom scipy import optimize, special, stats\n"

    def md(source):
        cells.append(nbf.v4.new_markdown_cell(source))

    def code(source):
        ast.parse(source)
        cells.append(nbf.v4.new_code_cell(source))

    md(
        """## Fitur estimasi noise: eksperimen yang ikut diuji

Harga, biro, pendapatan, dan keterlambatan diperkirakan tanpa target. Parameter
unsupervised memakai fitur seluruh train sebagaimana audit versi sebelumnya;
preprocessing model tetap dipelajari hanya dari konteks training fold.
Fitur ini tidak membuktikan bahwa seed RNG asli sudah ditemukan.
Cache dibangun dari data mentah bila belum ada. Pengulangan pada lingkungan Kumo
cocok dalam toleransi `rtol=1e-6, atol=1e-7`; selisih terbesar sekitar 4e-7.
"""
    )
    code(
        "DESTINATION = OUT / 'latent_audit'\nDESTINATION.mkdir(parents=True, exist_ok=True)\n"
        + "\n\n".join(
            function("reverse_synthetic_patterns.py", n)
            for n in [
                "distribution_summary",
                "aggregate_ratio_grid",
                "fit_price_latent",
                "fit_bureau_latent",
                "fit_account_latent",
                "beta_binomial_prior",
                "fit_repayment_latent",
            ]
        )
    )
    code(
        """reverse_file = PROCESSED / 'application_features_reverse.parquet'
if FORCE_REBUILD_FEATURES or not reverse_file.exists():
    base = pd.read_parquet(PROCESSED / 'application_features_v5.parquet').set_index('customer_id')
    latent_parts = [fn(base)[0] for fn in
        [fit_price_latent, fit_bureau_latent, fit_account_latent, fit_repayment_latent]]
    latent = pd.concat(latent_parts, axis=1)
    assert latent.index.equals(base.index) and latent.columns.is_unique
    assert not any('target' in c for c in latent.columns)
    base.join(latent).replace([np.inf, -np.inf], np.nan).reset_index().to_parquet(reverse_file, index=False)
    del base, latent, latent_parts
    gc.collect()
print('Fitur estimasi noise:', reverse_file)
"""
    )
    md(
        """## Definisi kandidat dan konteks

Kumo Large memakai bobot pretrained `v1.0.1` yang dibekukan. TabICLv2 memakai
checkpoint resmi `jingang/TabICL`, commit `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`.
Tidak ada label validation dalam konteks prediksinya. ID dan target dikeluarkan dari X.
Baseline memakai maksimum 50.000 baris konteks; kandidat konteks penuh memakai seluruh
training fold. Opsi balanced mempertahankan semua positif dan mengambil negatif secara
acak per anggota; koreksi odds mengembalikan prior populasi sebelum pencampuran.
"""
    )
    code(
        "DEST = ROOT / 'artifacts/optimization/kumo_improvement'\nDEST.mkdir(parents=True, exist_ok=True)\n"
        + assignments("kumo_tabular.py", ["CORE_COLUMNS", "EXCLUDED"])
        + "\n"
        + assignments("improve_kumo.py", ["LATENT_COLUMNS", "DERIVED_COLUMNS"])
    )
    for name in ["ks", "metrics", "make_tables", "context_indices", "predict_batches"]:
        code(function("kumo_tabular.py", name))
    for name in [
        "definitions",
        "load_frame",
        "fingerprint",
        "fit_context",
        "restore_population_prior",
        "predict_context_batches",
        "evaluate",
    ]:
        code(function("improve_kumo.py", name))
    code(
        "display(pd.DataFrame([dict(name=s['name'], features=len(s['columns']), members=s['estimators'], context_limit=s['context_limit']) for s in definitions()]))"
    )
    md(
        """## Seleksi tiga fold: semua percobaan dan bobot tetap

`RUN_SCREEN=False` membaca laporan historis yang tersimpan. Ubah menjadi `True`
untuk menghitung seluruh eksperimen dan prediksi test dari konteks masing-masing fold.
Campuran hanya memakai bobot kandidat 25% dan 50%; tidak ada pencarian bobot halus.
Syarat lolos: pooled KS dan rata-rata KS fold naik, dengan minimal 2 dari 3 fold membaik.
"""
    )
    for name in ["read_oof", "summarize", "export"]:
        code(function("summarize_kumo_improvement.py", name))
    code(
        """RUN_SCREEN = False
FORCE_RERUN_IMPROVEMENT = False
def experiment_args(names, splits, seed, estimator_batch=4):
    return argparse.Namespace(candidates=names, splits=splits, fold_seed=seed,
        folds=None, batch_size=8192, estimator_batch_size=estimator_batch,
        test=True, force=FORCE_RERUN_IMPROVEMENT)

if RUN_SCREEN:
    evaluate(experiment_args(['baseline_e8'], 3, 8247))
    evaluate(experiment_args(['core37_e8','denoised_e8','v5_latent100_e8','v5_e16'], 3, 8247, 1))
    evaluate(experiment_args(['tabicl_v5_e8','tabicl_denoised_e8'], 3, 8247))
    secondary = DEST / 'secondary_screen_plan.json'
    if secondary.exists():
        evaluate(experiment_args(json.loads(secondary.read_text())['candidates'], 3, 8247))
    screening = summarize('cv3_seed8247')
else:
    screening = json.loads((DEST / 'cv3_seed8247/comparison.json').read_text())
with pd.option_context('display.max_rows', None):
    display(pd.DataFrame(screening['comparisons']).drop(columns=['components','fold_ks','fold_deltas']))
"""
    )
    if prior_failure:
        md(
            """## Konfirmasi campuran yang tidak lolos

Campuran 75% baseline + 25% konteks seimbang membaik pada 3/5 fold dan menaikkan
rata-rata KS, tetapi pooled KS turun. Karena itu resep tersebut tidak diekspor.
Uji lanjutan memakai seluruh 56.000 baris training fold, dibanding batas 50.000
pada baseline. Pada tiga fold awal, kedua batas menghasilkan konteks yang sama
(46.666–46.667 baris), sehingga efek penambahan ini baru dapat diuji pada lima fold.
"""
        )
        code(
            "prior_failed_confirmation = "
            + pprint.pformat(prior_failure["selected"], width=90, sort_dicts=False)
            + "\ndisplay(pd.DataFrame([prior_failed_confirmation]).drop(columns=['components','fold_ks','fold_deltas']))"
        )
    md(
        """## Resep yang dikunci dan konfirmasi lima fold

Lima fold baru memakai seed 31027. Baseline dihitung ulang dengan daftar fitur,
jumlah anggota, dan batas konteks yang sama dengan checkpoint public. Perbandingan
berikut selalu antarmodel pada split yang sama. Jangan membandingkan angka pooled
tiga fold dan lima fold sebagai selisih improvement.
"""
    )
    code(
        "LOCK = "
        + pprint.pformat(notebook_lock, width=90, sort_dicts=False)
        + "\nprint(json.dumps(LOCK, indent=2))\n"
        + """confirmation_names = list(dict.fromkeys(['baseline_e8'] +
    [c['name'] for c in LOCK['selected']['components']]))
evaluate(experiment_args(confirmation_names, 5, 31027))
confirmation = summarize('cv5_seed31027')
comparison = pd.DataFrame(confirmation['comparisons'])
display(comparison.drop(columns=['components','fold_ks','fold_deltas']))
selected = next(r for r in confirmation['comparisons'] if r['name'] == LOCK['selected']['name'])
display(pd.DataFrame({'fold':range(1,6), 'candidate_ks':selected['fold_ks'],
    'delta_vs_baseline':selected['fold_deltas']}))
"""
    )
    md(
        """## Prediksi test dan ekspor submission.csv

CSV adalah rata-rata prediksi test dari lima konteks fold, lalu campuran sesuai resep
yang dikunci. Join ke sample submission memeriksa urutan dan kelengkapan 30.000 ID.
Ekspor hanya dilakukan jika kedua ringkasan KS naik dan minimal tiga fold membaik.
`submission_kumo.csv` dengan public 0,56262 dipertahankan. Kandidat baru belum memiliki
skor public dan tidak diunggah otomatis.
"""
    )
    code(
        """passed = selected['delta_ks'] > 0 and selected['delta_mean_fold_ks'] > 0 and selected['fold_wins'] >= 3
if passed:
    metadata = export('cv5_seed31027', LOCK['selected']['name'])
    candidate = ROOT / metadata['file']
    local_csv = OUT / 'submission.csv'
    local_csv.write_bytes(candidate.read_bytes())
    assert hashlib.sha256(local_csv.read_bytes()).hexdigest() == metadata['sha256']
    print('File untuk Kaggle:', candidate)
    print('Salinan notebook:', local_csv)
    print('SHA-256:', metadata['sha256'])
    print('Public score: belum diketahui')
    display(pd.read_csv(local_csv).head())
else:
    print('Konfirmasi belum lolos; checkpoint public 0,56262 tetap pilihan utama.')
"""
    )
    for cell in cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3.11 (LOGIKA Kumo)",
                "language": "python",
                "name": "logika-kumo",
            },
            "language_info": {"name": "python", "version": "3.11.13"},
        },
    )
    nbf.validate(nb)
    path = ROOT / "notebook/09_kumo_improvement.ipynb"
    nbf.write(nb, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    path = build()
    if args.execute:
        from nbclient import NotebookClient

        preserved_files = [
            "analisis_logika.ipynb",
            "submission.csv",
            "submission_v3.csv",
            "submission_v4.csv",
            "submission_v5.csv",
            "submission_kumo.csv",
            "submission_kumo_blend.csv",
            "submission_kumo_v2.csv",
        ]
        before = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in preserved_files
            if (ROOT / name).exists()
        }
        nb = nbf.read(path, as_version=4)
        NotebookClient(
            nb,
            timeout=3600,
            kernel_name="logika-kumo",
            resources={"metadata": {"path": str(ROOT)}},
            on_cell_start=lambda cell, cell_index: print(
                f"Notebook cell {cell_index}", flush=True
            ),
        ).execute()
        nbf.write(nb, path)
        after = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in before
        }
        assert before == after, "Notebook changed a pre-existing CSV or user notebook."
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert all(c.execution_count is not None for c in code_cells)
        assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
        candidate = ROOT / "results/teman/submission_kumo_v2.csv"
        exported = ROOT / "artifacts/notebook_runs/09_kumo_improvement/submission.csv"
        if exported.exists():
            assert candidate.read_bytes() == exported.read_bytes()
        verification = dict(
            notebook=str(path.relative_to(ROOT)),
            executed_code_cells=len(code_cells),
            errors=0,
            notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            existing_files_preserved=after,
            exported_csv_matches_candidate=exported.exists()
            and exported.read_bytes() == candidate.read_bytes(),
        )
        (DEST / "notebook_verification.json").write_text(
            json.dumps(verification, indent=2) + "\n"
        )
    print(path)


if __name__ == "__main__":
    main()
