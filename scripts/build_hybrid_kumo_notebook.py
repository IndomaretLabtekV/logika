"""Build the visible EDA-to-submission pipeline for the Kumo/spline hybrid."""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

import nbformat as nbf
from build_kumo_notebook import function

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/optimization/kumo_hybrid"


def build():
    previous = nbf.read(ROOT / "notebook/09_kumo_improvement.ipynb", as_version=4)
    cutoff = next(
        i
        for i, c in enumerate(previous.cells)
        if c.cell_type == "markdown" and c.source.startswith("## Definisi kandidat")
    )
    cells = copy.deepcopy(previous.cells[:cutoff])
    cells[
        0
    ].source = """# 10 — Kumo + spline: EDA hingga submission.csv

Resep tetap: **75% Kumo Large + 25% spline dengan fitur laten**.
Kumo memakai bobot pretrained; spline mempelajari hubungan aditif nonlinier dari
training fold. Tiga fold screening memakai seed 8247, lima fold berikutnya seed 31027.
Setiap perbandingan memakai pembagian baris yang sama untuk baseline dan gabungan.

Label serta lima split sudah pernah dipakai untuk eksperimen sebelumnya. Ini adalah
pemeriksaan konsistensi dengan pemilihan model berulang, bukan holdout yang belum
pernah dilihat. Skor public kandidat baru **belum diketahui**. Checkpoint Kumo
`submission_kumo.csv` dengan public 0,56262 tetap disimpan.

Gunakan Python 3.11, environments/teman/requirements-kumo.txt, GPU CUDA dan data mentah kompetisi.
Semua kode fitur, konteks Kumo, fitting spline, prediksi dan ekspor terlihat di sel.
"""
    cells[2].source = cells[2].source.replace(
        'RUN_NAME = "09_kumo_improvement"', 'RUN_NAME = "10_kumo_spline"'
    )
    cells[2].source += "\nimport joblib\n"

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))

    def code(text):
        ast.parse(text)
        cells.append(nbf.v4.new_code_cell(text))

    md(
        """## Resep model dan fungsi umum

Kumo: 90 fitur, delapan anggota ensemble, maksimum 50.000 konteks berlabel.
Spline: basis kubik lima knot, regularisasi C=0,05, tanpa interaksi/pembobotan kelas.
Daftar fitur berasal dari resep reverse_spline di notebook 07. ID dan target tidak
masuk X. Missing value dan kategori diproses hanya berdasarkan training fold.
"""
    )
    code(
        "DEST = ROOT / 'artifacts/optimization/kumo_hybrid'\nDEST.mkdir(parents=True, exist_ok=True)\nWEIGHT = 0.25\nFORCE_RERUN_HYBRID = False\n"
        + function("improve_kumo.py", "fingerprint")
        + "\n\n"
        + function("summarize_kumo_improvement.py", "read_oof")
    )
    for name in ["ks", "metrics", "make_tables", "context_indices", "predict_batches"]:
        code(function("kumo_tabular.py", name))
    for name in ["specifications", "read_data", "ids_hash", "baseline_directory"]:
        code(function("hybrid_kumo.py", name))
    code(
        "data_hybrid, test_hybrid, spline_spec, kumo_columns = read_data()\nprint('Train/test:', len(data_hybrid), len(test_hybrid))\nprint('Fitur Kumo/spline:', len(kumo_columns), len(spline_spec['columns']))\ndisplay(pd.DataFrame({'spline_feature':spline_spec['columns']}))"
    )
    md(
        """## Kumo baseline pada fold yang sama

Kode berikut menyiapkan konteks dari outer-training, membangun preprocessing dan
cache Kumo, lalu memprediksi validation/test. Tidak ada pembaruan bobot Kumo.
OOF historis dipakai jika tersedia; pada clone baru, baseline dihitung otomatis.
Konteks tidak pernah berisi baris validation yang sedang diprediksi.
"""
    )
    code(function("hybrid_kumo.py", "ensure_baseline"))
    md(
        """## Spline: log bertanda, basis nonlinier dan regresi logistik

Numerik diubah menjadi sign(x) × log(1+|x|). Fitur kontinu memakai basis spline;
fitur diskret memakai bagian linier. Indikator missing dan one-hot kategori ikut
masuk model. Median, knot, standardisasi dan koefisien hanya dipelajari pada train.
Tidak ada pemilihan iterasi dari validation fold: parameternya sudah tetap.
"""
    )
    for name in ["prepare", "fit_spline", "predict_spline"]:
        code(function("hybrid_kumo.py", name))
    md(
        """## Jalankan tiga fold dan lima fold

Bobot spline tetap 25% selama kedua tahap. Cache diperiksa terhadap definisi, data,
dan versi paket. FORCE_RERUN_HYBRID=True melatih ulang spline; baseline Kumo memakai
konteks/resep yang sama. Kode fit dan predict tersedia langsung pada fungsi berikut.
"""
    )
    code(function("hybrid_kumo.py", "evaluate"))
    code(
        """screening = evaluate(3, 8247, FORCE_RERUN_HYBRID)
confirmation = evaluate(5, 31027, FORCE_RERUN_HYBRID)
for name, result in [('Screening 3 fold', screening), ('Pemeriksaan 5 fold', confirmation)]:
    print(name)
    display(pd.DataFrame(result['comparisons']).drop(columns=['fold_ks','fold_deltas']))
selected = confirmation['comparisons'][-1]
display(pd.DataFrame({'fold':range(1,6), 'hybrid_ks':selected['fold_ks'], 'delta_vs_kumo':selected['fold_deltas']}))
"""
    )
    md(
        """## Pemeriksaan prediksi spline dari model tersimpan

Sel ini menghitung ulang probabilitas test dari lima model spline dan membandingkannya
dengan cache. File OOF/test serta model menyimpan checksum pada metadata fold.
"""
    )
    code(
        """rebuild_checks = []
for fold in range(5):
    directory = DEST / 'cv5_seed31027/reverse_spline'
    model_file = directory / f'fold{fold}_model.joblib'
    meta = json.loads((directory / f'fold{fold}_metadata.json').read_text())
    assert hashlib.sha256(model_file.read_bytes()).hexdigest() == meta['model_sha256']
    bundle = joblib.load(model_file)
    rebuilt = predict_spline(bundle, test_hybrid)
    cached = pd.read_parquet(directory / f'fold{fold}_test.parquet')
    assert cached.application_id.equals(test_hybrid.application_id)
    difference = float(np.max(np.abs(rebuilt-cached.prediction)))
    assert difference < 5e-12
    rebuild_checks.append({'fold':fold+1, 'test_prediction_max_abs_difference':difference})
display(pd.DataFrame(rebuild_checks))
"""
    )
    md(
        """## Ekspor kandidat ke submission.csv

Pooled KS dan rata-rata KS harus naik, dengan minimal 3/5 fold membaik. Prediksi
test lima fold dirata-ratakan per komponen, kemudian dicampur 75/25. Join ke sample
submission memastikan urutan 30.000 application_id. Skor Kaggle belum diketahui.

Jika hanya pooled KS yang naik, EXPORT_EXPLORATORY=True membuat file terpisah
untuk uji Kaggle. File ini belum memenuhi aturan promosi; penurunan rata-rata
KS dan jumlah fold membaik tetap ditampilkan. Checkpoint public dipertahankan.
"""
    )
    code(function("hybrid_kumo.py", "export"))
    code(
        """EXPORT_EXPLORATORY = True
passed = selected['delta_ks'] > 0 and selected['delta_mean_fold_ks'] > 0 and selected['fold_wins'] >= 3
metadata = None
if passed:
    metadata = export()
elif EXPORT_EXPLORATORY and selected['delta_ks'] > 0:
    metadata = export(exploratory=True)
    print('Kandidat EKSPLORATIF: pooled KS naik, tetapi konsistensi fold belum lolos.')
if metadata is not None:
    candidate = ROOT / metadata['file']
    local_csv = OUT / 'submission.csv'
    local_csv.write_bytes(candidate.read_bytes())
    assert hashlib.sha256(local_csv.read_bytes()).hexdigest() == metadata['sha256']
    print('File kandidat:', candidate)
    print('Salinan notebook:', local_csv)
    print('SHA-256:', metadata['sha256'])
    print('Public score: belum diketahui')
    display(pd.read_csv(local_csv).head())
else:
    print('Kandidat belum lolos; checkpoint public 0,56262 tetap tersedia.')
"""
    )
    md(
        """## Screening tambahan dari OOF historis

Tabel berikut mendokumentasikan pencarian awal spline dan CatBoost sebagai tambahan
Kumo, masing-masing 25%/50%. CatBoost memakai koreksi probabilitas untuk bobot kelas
positif enam. Model asal direview di notebook 07. RUN_HISTORICAL_SCREEN=True
menghitung ulang campuran jika OOF historis tersedia; bagian ini tidak diperlukan
untuk membuat submission dari resep final di atas.
"""
    )
    code(
        """RUN_HISTORICAL_SCREEN = False
historical = json.loads((DEST / 'screening_comparison.json').read_text())
display(pd.DataFrame(historical['comparisons']).drop(columns=['fold_ks']))
if RUN_HISTORICAL_SCREEN:
    prior = pd.read_parquet(ROOT / 'artifacts/reverse_engineering/validation_oof.parquet')
    base = read_oof(baseline_directory(3,8247), prior[['application_id','target']], 3)
    yy = prior.target.astype(int)
    ff = list(StratifiedKFold(3,shuffle=True,random_state=8247).split(prior,yy))
    rows = []
    corrected_cat = prior.reverse_cat5.to_numpy() / (6-5*prior.reverse_cat5.to_numpy())
    for family, probability in [('reverse_spline',prior.reverse_spline.to_numpy()), ('reverse_cat5_prior_corrected',corrected_cat)]:
        for weight in [0.25,0.5]:
            p = (1-weight)*base+weight*probability
            rows.append({'family':family, 'weight':weight, **metrics(yy,p),
                'mean_fold_ks':float(np.mean([ks(yy.iloc[vi],p[vi]) for _,vi in ff]))})
    display(pd.DataFrame(rows))
"""
    )
    md(
        """## Screening pasangan TabM

TabM dari notebook 06 juga diuji langsung sebagai tambahan Kumo dengan bobot
25%/50%. Versi loss berbobot enam memakai koreksi p/(6−5p). Tidak ada campuran
yang memenuhi syarat pooled/rata-rata KS naik dan dua dari tiga fold membaik.
Tabel ini memakai OOF historis; model baru tidak dilatih pada eksperimen ini.
"""
    )
    code(
        """RUN_TABM_SCREEN = False
tabm_screen = json.loads((ROOT / 'artifacts/optimization/kumo_tabm/screening_comparison.json').read_text())
display(pd.DataFrame(tabm_screen['comparisons']).drop(columns=['fold_ks']))
if RUN_TABM_SCREEN:
    prior = pd.read_parquet(ROOT / 'artifacts/optimization/v5/selection_oof_tabm.parquet')
    base = read_oof(baseline_directory(3,8247),prior[['application_id','target']],3).astype('float64')
    yy = prior.target.astype(int)
    ff = list(StratifiedKFold(3,shuffle=True,random_state=8247).split(prior,yy))
    rows = []
    p1 = prior.v5_tabm_weight1.to_numpy(dtype='float64')
    p6 = prior.v5_tabm_weight6.to_numpy(dtype='float64')
    for family,probability in [('TabM weight 1',p1),('TabM weight 6 corrected',p6/(6-5*p6))]:
        for weight in [0.25,0.5]:
            p = (1-weight)*base+weight*probability
            rows.append({'family':family,'weight':weight,**metrics(yy,p),
                'mean_fold_ks':float(np.mean([ks(yy.iloc[vi],p[vi]) for _,vi in ff]))})
    display(pd.DataFrame(rows))
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
    path = ROOT / "notebook/10_kumo_spline.ipynb"
    nbf.write(nb, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    path = build()
    if args.execute:
        from nbclient import NotebookClient

        preserved = [
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
            for name in preserved
            if (ROOT / name).exists()
        }
        nb = nbf.read(path, as_version=4)
        NotebookClient(
            nb,
            timeout=3600,
            kernel_name="logika-kumo",
            resources={"metadata": {"path": str(ROOT)}},
            on_cell_start=lambda cell, cell_index: print(
                "Notebook cell", cell_index, flush=True
            ),
        ).execute()
        nbf.write(nb, path)
        after = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in before
        }
        assert before == after
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert all(c.execution_count is not None for c in code_cells)
        assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
        metadata_path = DEST / "submission_metadata.json"
        if not metadata_path.exists():
            metadata_path = DEST / "exploratory_submission_metadata.json"
        export_metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else None
        )
        candidate = ROOT / export_metadata["file"] if export_metadata else None
        local_csv = ROOT / "artifacts/notebook_runs/10_kumo_spline/submission.csv"
        if local_csv.exists():
            assert (
                candidate is not None
                and candidate.read_bytes() == local_csv.read_bytes()
            )
        result = dict(
            notebook=str(path.relative_to(ROOT)),
            executed_code_cells=len(code_cells),
            errors=0,
            notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            existing_files_preserved=after,
            exported_csv_matches_candidate=local_csv.exists()
            and candidate.read_bytes() == local_csv.read_bytes(),
            submission_file=export_metadata["file"] if export_metadata else None,
            export_mode=export_metadata.get("export_mode") if export_metadata else None,
        )
        (DEST / "notebook_verification.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(path, flush=True)


if __name__ == "__main__":
    main()
