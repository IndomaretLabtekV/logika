"""Build an Indonesian EDA-to-submission notebook with visible Kumo code."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def function(filename, name):
    source = (ROOT / "scripts" / filename).read_text()
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])


def build(variant):
    public_path = ROOT / "artifacts/optimization/kumo/public_leaderboard_result.json"
    public_result = (
        json.loads(public_path.read_text()) if public_path.exists() else None
    )
    canonical = ROOT / "results/teman/submission_kumo.csv"
    if public_result and (
        public_result.get("variant") != variant
        or not canonical.exists()
        or public_result.get("sha256")
        != hashlib.sha256(canonical.read_bytes()).hexdigest()
    ):
        public_result = None
    size, features, context, estimators = variant.split("_")
    context, estimators = int(context.removeprefix("ctx")), int(
        estimators.removeprefix("e")
    )
    previous = nbf.read(ROOT / "notebook/06_submission_v5.ipynb", as_version=4)
    cells = [
        nbf.v4.new_markdown_cell(
            """# 08 — Kumo Tabular: EDA hingga submission

Kumo adalah foundation model pretrained. Pada eksperimen ini bobotnya **dibekukan**:
`fit` menyiapkan preprocessing dan cache konteks berlabel, tanpa gradient training.
Validasi memakai tiga fold yang sama dengan seleksi V5 (seed 8247).

Gunakan lingkungan **Python 3.11** terpisah dengan `environments/teman/requirements-kumo.txt` dan GPU CUDA.
Model lokal: [NVIDIA Kumo Tabular](https://huggingface.co/nvidia/Kumo-Tabular),
checkpoint `v1.0.1`, paket `structured-data-models==0.2.1`.
Data kompetisi tetap diproses di mesin lokal.

Sel berikut memuat EDA dan seluruh kode fitur dari notebook 06, tanpa melatih model
lama. Kode Kumo dan ekspor berada di sel notebook, sehingga dapat direview langsung.
Submission Kumo disimpan terpisah sebagai `submission_kumo.csv`; public score belum diketahui.
"""
        )
    ]
    if public_result:
        cells[0].source = cells[0].source.replace(
            "public score belum diketahui.",
            f"public score **{public_result['public_score']:.5f}**, dilaporkan pada "
            f"{public_result['reported_on']} dan dikonfirmasi melalui API Kaggle. "
            "Skor public campuran belum diketahui.",
        )
    setup = copy.deepcopy(previous.cells[2])
    setup.source = setup.source.replace('RUN_NAME = "06_v5"', 'RUN_NAME = "08_kumo"')
    setup.source = setup.source.replace(
        'CANONICAL_FILE = "submission_v5.csv"', 'CANONICAL_FILE = "submission_kumo.csv"'
    )
    public_score = public_result["public_score"] if public_result else None
    setup.source = setup.source.replace(
        "PUBLIC_SCORE = 0.5568", f"PUBLIC_SCORE = {public_score!r}"
    )
    setup.source = "\n".join(
        line
        for line in setup.source.splitlines()
        if not any(
            token in line
            for token in [
                "import joblib",
                "from xgboost",
                "from lightgbm",
                "from catboost",
            ]
        )
    )
    setup.source += "\nimport argparse, importlib.metadata\nimport torch, sdm\nassert torch.cuda.is_available()\nprint('Torch:', torch.__version__, '; SDM:', sdm.__version__)"
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Persiapan\n\nFitur dibangun dari data mentah jika cache belum tersedia."
        )
    )
    cells.append(setup)
    cells.extend(copy.deepcopy(previous.cells[3:65]))
    cells.extend(copy.deepcopy(previous.cells[97:99]))

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))

    def code(text):
        ast.parse(text)
        cells.append(nbf.v4.new_code_cell(text))

    md(
        """## Konfigurasi dan fitur Kumo

Daftar fitur sama untuk seluruh fold. Numerik mempertahankan missing value; kategori
ditangani oleh recipe bawaan SDM. ID, waktu mentah, split, dan target dilarang masuk X.
Seluruh preprocessing dipelajari dari konteks training fold, lalu diterapkan ke query.
"""
    )
    script = (ROOT / "scripts/kumo_tabular.py").read_text()
    tree = ast.parse(script)
    assignments = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in ["CORE_COLUMNS", "EXCLUDED"]
            for t in n.targets
        )
    ]
    code(
        "OUT = ROOT / 'artifacts/optimization/kumo'\nOUT.mkdir(parents=True, exist_ok=True)\n"
        + "\n".join(ast.get_source_segment(script, n) for n in assignments)
        + "\n\n"
        + "\n\n".join(
            function("kumo_tabular.py", n) for n in ["ks", "metrics", "read_data"]
        )
    )
    code(
        f"""VARIANT = {variant!r}
FORCE_RERUN_KUMO = False
RUN_ALL_VARIANTS = True
args = argparse.Namespace(size={size!r}, features={features!r}, context={context},
    estimators={estimators}, batch_size=8192, folds=[0,1,2], pilot=0, test=True,
    force=FORCE_RERUN_KUMO)
train_kumo, test_kumo, kumo_columns = read_data(args.features)
display(train_kumo[kumo_columns].isna().mean().sort_values(ascending=False).head(15))
print('Train / test / fitur:', len(train_kumo), len(test_kumo), len(kumo_columns))
print('Rasio positif:', train_kumo.target.mean())
"""
    )
    md(
        """## Konteks berlabel dan batching query

Setiap fold memakai subset stratified training dengan seed tetap. Baris validation tidak
pernah menjadi konteks bagi prediksinya sendiri. Proporsi positif konteks mengikuti data asli.
Kumo melakukan ensemble transformasi fitur dan kategori dengan recipe default.
Prediksi kelas positif diambil dari kolom bernama `1`, karena urutan kategori dapat berubah.
"""
    )
    for name in ["make_tables", "context_indices", "predict_batches"]:
        code(function("kumo_tabular.py", name))
    md(
        """## Jalankan tiga fold dan bandingkan pada baris yang sama

Kode ini menampilkan seluruh proses `model.fit` dan `model.predict`. Cache prediksi
divalidasi menggunakan resep dan fingerprint data. Aktifkan `FORCE_RERUN_KUMO`
untuk menjalankan ulang inferensi dari bobot pretrained dan konteks yang sama.

Jika tersedia, baseline berasal dari prediksi **OOF tiga fold** V5, bukan OOF sepuluh
fold atau prediksi in-sample. Tanpa `selection_oof_tabm.parquet`, Kumo tetap berjalan
hingga ekspor; perbandingan V5 dilewati. Untuk mereproduksi perbandingan historis,
hasilkan OOF V4 dan jalankan evaluasi TabM V5 melalui pipeline sebelumnya.
"""
    )
    code(function("kumo_tabular.py", "run"))
    code(
        """configs = [
    dict(size='medium', features='v5', context=32768, estimators=4),
    dict(size='large', features='v5', context=50000, estimators=8),
]
for config in configs if RUN_ALL_VARIANTS else [vars(args)]:
    current = argparse.Namespace(**(vars(args) | config))
    run(current)
"""
    )
    md(
        """## Membaca selisih skor

KS pooled dihitung dari semua 70.000 prediksi OOF. Rata-rata KS fold dihitung terpisah;
masing-masing dibandingkan dengan baseline pada fold yang sama. Dua bobot campuran
kasar ditampilkan sebagai diagnosis. Pemilihan model dari OOF mengandung optimisme
seleksi dan tidak menjamin kenaikan public leaderboard.
"""
    )
    code(function("report_kumo.py", "compare"))
    code("summary, oof = compare()\ndisplay(pd.DataFrame(summary['comparisons']))")
    md(
        """## Ekspor submission Kumo

Probabilitas test adalah rata-rata tiga model konteks fold yang sama. Urutan CSV
ditetapkan melalui join `application_id` terhadap sample submission, disertai pemeriksaan
keunikan ID, jumlah baris, dan probabilitas. Ini merupakan prediksi **Kumo murni**.
Tidak ada pengiriman otomatis ke Kaggle.
"""
    )
    code(function("report_kumo.py", "export"))
    code(
        "metadata = export(VARIANT)\ndisplay(metadata)\ndisplay(pd.read_csv(ROOT / 'results/teman/submission_kumo.csv').head())"
    )
    md(
        """## Kandidat campuran dengan V5

Bobot Kumo 25% ditampilkan terpisah agar kontribusi model mudah dinilai. Campuran
memakai prediksi test dari **model seleksi V5 tiga fold yang sama dengan validasi**;
berkas ini berbeda dari V5 public yang memakai sepuluh fold. Sel dilewati bila baseline
test V5 tidak tersedia. Validasi lokal dan skor public harus dibedakan.
"""
    )
    code(function("report_kumo.py", "export_blend"))
    code(
        """if all((ROOT/'artifacts/optimization/v5'/f).exists() for f in ['selection_test_tabm.parquet','selection_oof_tabm.parquet']):
    blend_metadata = export_blend(VARIANT, weight=0.25)
    display(blend_metadata)
    display(pd.read_csv(ROOT / 'results/teman/submission_kumo_blend.csv').head())
else:
    print('Kumo murni selesai; baseline V5 tidak tersedia untuk membuat campuran.')
"""
    )
    for c in cells:
        if c.cell_type == "code":
            c.outputs = []
            c.execution_count = None
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
    path = ROOT / "notebook/08_kumo_tabular.ipynb"
    nbf.write(nb, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="large_v5_ctx50000_e8")
    args = parser.parse_args()
    print(build(args.variant))


if __name__ == "__main__":
    main()
