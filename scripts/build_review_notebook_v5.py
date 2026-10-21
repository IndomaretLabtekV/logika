"""Append readable TabM cells to the complete version-4 review pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json

import nbformat as nbf

from build_review_notebooks import (
    ROOT,
    DEST,
    source,
    body,
    md,
    code,
    section,
    literal,
    validation_plot,
    EXPORT,
)


def build():
    public_file = ROOT / "artifacts/optimization/v5/public_leaderboard_result.json"
    public_result = (
        json.loads(public_file.read_text()) if public_file.exists() else None
    )
    canonical_file = ROOT / "results/teman/submission_v5.csv"
    if public_result and (
        not canonical_file.exists()
        or public_result.get("sha256")
        != hashlib.sha256(canonical_file.read_bytes()).hexdigest()
    ):
        public_result = None
    public_score = public_result["public_score"] if public_result else None
    notebook = nbf.read(DEST / "05_submission_v4.ipynb", as_version=4)
    # Keep EDA, all earlier feature stages, and every component of version 4.
    cells = notebook.cells[:-3]
    cells[
        0
    ].source = """# 06 — Submission versi 5: versi 4 + TabM

Notebook ini mereproduksi **submission_v5.csv**, mulai dari EDA, seluruh agregasi riwayat,
model versi sebelumnya, sampai gabungan 75% versi 4 dan 25% TabM.

TabM dilatih dari data kompetisi. Early stopping berada di dalam training setiap fold.
Bobot akhir dikunci sebelum validasi sepuluh fold. Skor public versi 5 belum diketahui.

Instal `environments/teman/requirements-v5.txt` dan `environments/teman/requirements-notebook.txt` dengan Python 3.10.
GPU mempercepat training; CPU juga didukung. Default memakai cache jika tersedia,
tetapi probabilitas test tetap dihitung kembali dari model.
"""
    if public_result:
        cells[0].source = cells[0].source.replace(
            "Skor public versi 5 belum diketahui.",
            f"Skor public versi 5 **{public_score:.5f}**, dilaporkan pengguna pada "
            f"{public_result['reported_on']}; versi 3 memperoleh 0,55845. "
            "Kenaikan validasi lokal belum diikuti kenaikan public leaderboard.",
        )
    for cell in cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.source = cell.source.replace('"05_v4"', '"06_v5"').replace(
                "'05_v4'", "'06_v5'"
            )
            cell.source = cell.source.replace(
                '"submission_v4.csv"', '"submission_v5.csv"'
            ).replace("'submission_v4.csv'", "'submission_v5.csv'")
            if public_score is not None:
                cell.source = cell.source.replace(
                    "PUBLIC_SCORE = None", f"PUBLIC_SCORE = {public_score!r}"
                )
    feature_body = body("build_features_v5.py", "main")
    for quote in ["'", '"']:
        for old, new in [
            (
                "ROOT / {q}data/processed/application_features_v4.parquet{q}",
                "PROCESSED / 'application_features_v4.parquet'",
            ),
            (
                "ROOT / {q}data/ecommerce_transactions.csv{q}",
                "RAW / 'ecommerce_transactions.csv'",
            ),
            (
                "ROOT / {q}data/processed/application_features_v5.parquet{q}",
                "feature_file_v5",
            ),
            ("OUT / {q}v5{q}", "OUT / 'feature_audits'"),
        ]:
            feature_body = feature_body.replace(old.format(q=quote), new)
    section(
        cells,
        "Tambahan fitur versi 5: pilihan harga pada semua basket",
        "Pesanan cancelled memberikan informasi barang yang dipilih, tetapi tidak menjadi belanja. "
        "Referensi kategori hanya memakai riwayat customer training tanpa label target. "
        "Kode berikut menambahkan 28 fitur; TabM final memakai empat ringkasan harga pooled di antaranya.",
        "feature_file_v5 = PROCESSED / 'application_features_v5.parquet'\n"
        "if FORCE_REBUILD_FEATURES or not feature_file_v5.exists():\n"
        + "\n".join("    " + line for line in feature_body.splitlines())
        + "\nfeatures_v5 = pd.read_parquet(feature_file_v5)\n"
        "assert features_v5.application_id.equals(features_v4.application_id)\n"
        "assert features_v5.target.equals(features_v4.target)\n"
        "print('Fitur versi 5:', features_v5.shape)\n"
        "display(features_v5[['v4_price_all_mean', 'v5_price_pooled_all_mean']].describe())",
    )
    section(
        cells,
        "Review eksperimen sebelum kandidat akhir",
        "Tabel menampilkan kandidat spline, EBM, XGBoost, CatBoost, specialist, dan TabM. "
        "File ringkasan historis disertakan di GitHub. Kandidat final hanya menunjukkan kenaikan kecil "
        "pada tiga fold dan meningkatkan satu fold; pengujian sepuluh fold adalah konfirmasi eksploratif. "
        "Pemilihan berulang dari label training menyebabkan optimisme seleksi.",
        """
rows = []
for file in sorted((ROOT / 'artifacts/optimization/v5').glob('summary_*.json')):
    for row in json.loads(file.read_text()):
        if row['name'] != 'version4':
            rows.append({'kelompok': file.stem.replace('summary_', ''), **row})
display(pd.DataFrame(rows).sort_values('ks', ascending=False))
selection_report = ROOT / 'artifacts/optimization/v5/submission_v5_metadata.json'
if selection_report.exists():
    display(pd.Series(json.loads(selection_report.read_text())['selection_3fold']))
""",
    )
    md(
        cells,
        "## TabM: preprocessing, early stopping, dan prediksi\n\n"
        "Numerik memakai signed log, median, dan skala yang dipelajari pada training. "
        "Indikator missing dan one-hot kategori ditambahkan. Bin piecewise-linear juga dipelajari pada training. "
        "Model berisi 16 anggota dengan dua hidden layer (128 unit, dropout 0,15). "
        "Setiap anggota memiliki loss sendiri; probabilitasnya dirata-ratakan saat prediksi. "
        "Early stopping memakai validasi internal 12,5%, kemudian model dilatih ulang pada seluruh training outer fold. "
        "Label fold luar tidak dipakai untuk menentukan epoch. Implementasi TabM memakai paket "
        "[resmi penulis](https://github.com/yandex-research/tabm); kode pipeline proyek terlihat di bawah.",
    )
    code(cells, "import sys, warnings\n" + source("neural_v5.py", "neural_imports"))
    for name in [
        "make_neural_encoder",
        "make_neural_model",
        "train_neural",
        "fit_neural",
        "predict_neural",
    ]:
        code(cells, source("neural_v5.py", name))
    metadata = json.loads(
        (
            ROOT / "artifacts/optimization/v5/confirmation_recipe_v5_tabm_weight1.json"
        ).read_text()
    )
    section(
        cells,
        "Training sepuluh fold dengan resep yang dikunci",
        "Pembagian fold sama dengan versi 4: StratifiedKFold dengan seed 9173. "
        "Model fold hanya dilatih pada ti dan menghasilkan OOF pada vi. "
        "Cache divalidasi terhadap kolom, parameter, dan seed. Probabilitas test dihitung ulang dari setiap model.",
        "V5_SPEC = "
        + literal(metadata["spec"])
        + "\nV5_RECIPE = "
        + literal(metadata["recipe"])
        + """
data_v5 = features_v5[features_v5.split.eq('train')].reset_index(drop=True)
test_v5 = features_v5[features_v5.split.eq('test')].reset_index(drop=True)
assert data_v5.application_id.equals(v4_oof.application_id)
assert test_v5.application_id.equals(v4_test.application_id)
assert data_v5.target.astype(int).equals(v4_oof.target.astype(int))
assert set(V5_SPEC['columns']).issubset(data_v5.columns)
assert not set(V5_SPEC['columns']) & EXCLUDED
folds = list(StratifiedKFold(10, shuffle=True, random_state=9173).split(data_v5, data_v5.target))
v5_oof = v4_oof[['application_id', 'target', 'version4']].copy()
v5_test = v4_test[['application_id', 'version4']].copy()
name = V5_SPEC['name']
v5_oof[name] = np.nan
v5_test[name] = 0.0
historical_oof = ROOT / f'artifacts/optimization/v5/final_oof_{name}.parquet'
cached = None
if historical_oof.exists() and not (FORCE_RETRAIN_MODELS or FORCE_RECOMPUTE_VALIDATION):
    cached = pd.read_parquet(historical_oof)
    assert cached.application_id.equals(data_v5.application_id)
    assert cached.target.astype(int).equals(data_v5.target.astype(int))
for fold, (ti, vi) in enumerate(folds, 1):
    filename = f'fold{fold}_{name}.joblib'
    local = OUT / 'models' / filename
    historical = ROOT / 'artifacts/optimization/v5/final_models' / filename
    path = local if local.exists() else historical
    if path.exists() and not FORCE_RETRAIN_MODELS:
        bundle = joblib.load(path)
        assert bundle['columns'] == V5_SPEC['columns']
        assert bundle['params'] == V5_SPEC['params']
        assert bundle['seed'] == 2026 + fold
    else:
        bundle = fit_neural(V5_SPEC, data_v5, ti, 2026 + fold)
        joblib.dump(bundle, local, compress=3)
    if cached is not None:
        v5_oof.loc[vi, name] = cached[name].iloc[vi].to_numpy()
    else:
        v5_oof.loc[vi, name] = predict_neural(bundle, data_v5.iloc[vi])
    v5_test[name] += predict_neural(bundle, test_v5) / 10
    print(f'Fold {fold}/10 selesai; epoch {bundle["epochs"]}')
assert v5_oof.notna().all().all()
v5_oof['version5'] = sum(w * v5_oof[n] for n, w in V5_RECIPE.items())
v5_test['version5'] = sum(w * v5_test[n] for n, w in V5_RECIPE.items())
display(pd.DataFrame([{'versi': name, **metrics(v5_oof.target, v5_oof[name])},
                      {'versi': 'v4', **metrics(v5_oof.target, v5_oof.version4)},
                      {'versi': 'v5', **metrics(v5_oof.target, v5_oof.version5)}]))
comparison = pd.DataFrame([
    {'fold': fold, 'KS_v4': ks(v5_oof.target.iloc[vi], v5_oof.version4.iloc[vi]),
     'KS_v5': ks(v5_oof.target.iloc[vi], v5_oof.version5.iloc[vi])}
    for fold, (_, vi) in enumerate(folds, 1)])
comparison['selisih'] = comparison.KS_v5 - comparison.KS_v4
display(comparison)
print('Rata-rata KS per fold:', comparison[['KS_v4','KS_v5']].mean().to_dict())
print('Fold yang membaik:', comparison.selisih.gt(0).sum(), '/ 10')
test_probability = v5_test.version5.to_numpy()
""",
    )
    validation_plot(cells, "Versi 5 OOF", "v5_oof", "version5")
    section(
        cells,
        "Terakhir: hasilkan submission.csv",
        "CSV mengikuti header dan urutan sample submission. Output disimpan di folder notebook "
        "dan dibandingkan dengan checkpoint versi 5. File root tidak ditimpa dan tidak ada submit otomatis.",
        EXPORT,
    )
    md(
        cells,
        "## Hal yang perlu diperhatikan saat review\n\n"
        "Periksa cutoff, pemisahan ti/vi, validasi internal untuk epoch, dan bobot 75%/25%. "
        "Skor lokal dapat mengandung optimisme seleksi. Skor public perlu diperiksa setelah upload CSV. "
        "Jika mengubah data atau parameter, aktifkan flag rebuild/retrain yang sesuai pada bagian persiapan.",
    )
    notebook.cells = cells
    notebook.metadata.logika = {
        "version": 6,
        "self_contained_code": True,
        "canonical_file": "submission_v5.csv",
    }
    nbf.validate(notebook)
    path = DEST / "06_submission_v5.ipynb"
    nbf.write(notebook, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    path = build()
    print("Created:", path, flush=True)
    if args.execute:
        from nbclient import NotebookClient

        notebook = nbf.read(path, as_version=4)
        try:
            NotebookClient(
                notebook,
                timeout=1800,
                kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()
        finally:
            nbf.write(notebook, path)
        print("Passed:", path, flush=True)


if __name__ == "__main__":
    main()
