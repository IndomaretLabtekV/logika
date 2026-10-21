"""Build a readable EDA, model-research, independent-audit and submission notebook.

Copied functions receive simple prefixes so the independent model families do
not overwrite each other's names. The generated notebook contains ordinary
Python definitions, not hidden script execution.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/optimization/goal054"


class Rename(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        node.id = self.mapping.get(node.id, node.id)
        return node

    def visit_FunctionDef(self, node):
        node.name = self.mapping.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module in {
            "report_goal054",
            "research_context_correction054",
            "research_calendar054",
            "research_relational054",
            "blend_relational054",
        }:
            return None
        return node


def module_cells(filename, prefix, extra=None, selected=None):
    source = (ROOT / "scripts" / filename).read_text()
    nodes = ast.parse(source).body
    definitions = [
        n
        for n in nodes
        if isinstance(n, ast.FunctionDef)
        and (n.name != "main" or selected is not None and "main" in selected)
    ]
    if selected is not None:
        definitions = [n for n in definitions if n.name in selected]
    assignments = [n for n in nodes if isinstance(n, ast.Assign)]
    if selected is not None:
        assignments = []
    names = [n.name for n in definitions]
    names += [t.id for n in assignments for t in n.targets if isinstance(t, ast.Name)]
    mapping = {name: prefix + name for name in names}
    mapping.update(extra or {})
    cells = []
    for node in assignments + definitions:
        rendered = ast.unparse(
            ast.fix_missing_locations(Rename(mapping).visit(copy.deepcopy(node)))
        )
        ast.parse(rendered)
        cells.append(nbf.v4.new_code_cell(rendered))
    return cells, mapping


def build():
    previous = nbf.read(ROOT / "notebook/10_kumo_spline.ipynb", as_version=4)
    cutoff = next(
        i
        for i, c in enumerate(previous.cells)
        if c.cell_type == "markdown" and c.source.startswith("## Ekspor kandidat")
    )
    cells = copy.deepcopy(previous.cells[:cutoff])
    cells[
        0
    ].source = """# 12 — Riset KS OOF 0,54: EDA, model, audit dan submission

Target: **KS gabungan seluruh OOF ≥ 0,54**, memakai lima fold stratified seed
31027. Baseline Kumo–spline memiliki pooled KS **0,52851** dan rata-rata KS fold
**0,53095**. Pengguna melaporkan public **0,56393** untuk CSV baseline pada
18 Oktober 2026. Skor public setiap kandidat yang berubah masih belum diketahui.

Bagian awal membangun fitur dan baseline dari data mentah. Bagian riset memuat
kode TabPFN 3.5, EBM, koreksi Kumo/kalender, Kumo Relational, audit, dan ekspor. Mode review
default membaca hasil eksperimen yang selesai. Ubah flag training untuk mengulang
model baru pada clone tanpa cache. Eksperimen yang belum lengkap tidak mempunyai
skor pooled. Notebook hanya mengekspor kandidat goal jika target telah tercapai.

Pemilihan berulang memakai training labels dan split yang pernah digunakan.
Angka OOF tetap mengandung optimisme seleksi; ini bukan untouched holdout.
Gunakan Python 3.11, environments/teman/requirements-goal054.txt dan GPU CUDA untuk training.
"""
    cells[2].source = cells[2].source.replace(
        'RUN_NAME = "10_kumo_spline"', 'RUN_NAME = "12_goal054"'
    )
    for c in cells:
        if c.cell_type == "code" and c.source.startswith("screening = evaluate(3,"):
            c.source = """baseline_confirmation = evaluate(5, 31027, FORCE_RERUN_HYBRID)
display(pd.DataFrame(baseline_confirmation['comparisons']).drop(columns=['fold_ks','fold_deltas']))
assert abs(baseline_confirmation['comparisons'][-1]['ks'] - 0.5285099762252743) < 1e-12
"""

    def md(source):
        cells.append(nbf.v4.new_markdown_cell(source))

    def code(source):
        ast.parse(source)
        cells.append(nbf.v4.new_code_cell(source))

    md(
        """## Protokol riset dan lingkungan

Semua model baru dibandingkan dengan baseline pada baris validation yang sama.
Bobot model baru ditetapkan 25%/50%/75%, serta model tunggal. TabPFN memakai delapan
anggota dan 50.000 konteks. EBM mempelajari bin, interaksi dan early stopping hanya
dari outer-training. Dataset tidak dikirim ke layanan inferensi eksternal.

Referensi: [TabPFN](https://github.com/PriorLabs/TabPFN),
[bobot 3.5](https://huggingface.co/Prior-Labs/tabpfn_3_5),
[EBM](https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html).
"""
    )
    code(
        """import argparse, gc, hashlib, importlib.metadata, json, os, shutil, time
from datetime import datetime, timezone
from types import SimpleNamespace
from scipy.stats import ks_2samp
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits
display(pd.DataFrame([{'package':p,'version':importlib.metadata.version(p)} for p in
 ['torch','structured-data-models','tabpfn','interpret-core','numpy','pandas','scikit-learn']]))
"""
    )
    # The two deterministic feature helpers need their explicit feature lists.
    for filename, names in [
        ("kumo_tabular.py", {"CORE_COLUMNS", "EXCLUDED"}),
        ("improve_kumo.py", {"LATENT_COLUMNS", "DERIVED_COLUMNS"}),
    ]:
        source = (ROOT / "scripts" / filename).read_text()
        for n in ast.parse(source).body:
            if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in names for t in n.targets
            ):
                code(ast.get_source_segment(source, n))
    shared, _ = module_cells(
        "improve_kumo.py", "shared_", selected={"definitions", "load_frame"}
    )
    cells.extend(shared)
    md(
        """## TabPFN 3.5: data, cache dan inferensi

Bobot diikat ke commit Hugging Face. SafeTensors dimaterialisasi dengan suffix yang
benar. Cache konteks disimpan di CPU di antara query; setiap fold membangun cache
baru hanya dari konteksnya. Prefix `g_` membedakan fungsi riset ini dari baseline.
"""
    )
    goal, mapping = module_cells(
        "research_goal054.py",
        "g_",
        extra={"definitions": "shared_definitions", "load_frame": "shared_load_frame"},
    )
    cells.extend(goal)
    exported = [
        "write_json",
        "checksum",
        "specifications",
        "scores",
        "inputs",
        "model_features",
        "checkpoint_path",
        "classifier",
        "predict",
        "run",
        "read_oof",
        "compare",
    ]
    attrs = {name: "g_" + name for name in exported}
    attrs.update(
        {
            name: "g_" + name
            for name in [
                "DEST",
                "FOLD_SEED",
                "SPLITS",
                "PUBLIC_CHECKPOINT",
                "PUBLIC_SHA256",
            ]
        }
    )
    attrs.update(
        {
            name: name
            for name in [
                "ROOT",
                "EXCLUDED",
                "context_indices",
                "baseline_directory",
                "ids_hash",
                "fingerprint",
            ]
        }
    )
    code(
        "# Namespace helper mengikat fungsi yang terlihat di atas.\nr = SimpleNamespace(\n"
        + ",\n".join(f"    {key}={value}" for key, value in attrs.items())
        + "\n)"
    )
    md(
        """## EBM: loss biasa dan seimbang

EBM mempelajari model aditif dan 12 interaksi dari training fold, memakai delapan
bag internal. Bobot positif versi seimbang dihitung dari outer-training. Prediksi
dikonversi kembali ke prior populasi sebelum dicampur dengan baseline.
"""
    )
    ebm, _ = module_cells("research_ebm054.py", "e_")
    cells.extend(ebm)
    md(
        """## Koreksi Kumo dari 6.000 baris di luar konteks

Outer-training dibagi menjadi 50.000 konteks Kumo dan 6.000 baris training koreksi.
Keduanya tidak beririsan dengan 14.000 baris outer-validation. Ini menghindari
pelatihan koreksi dari OOF seluruh fold yang dapat membawa label validation.

Regresi logistik memakai logit Kumo sebagai offset, basis spline risiko, dan
penalti L2. Koreksi nol mengembalikan probabilitas Kumo. Dua penalti, dua loss,
dan dua kekuatan koreksi menghasilkan delapan resep yang ditetapkan sebelumnya.
Hasil akhir adalah 75% Kumo terkoreksi + 25% spline baseline.
"""
    )
    correction, _ = module_cells(
        "research_context_correction054.py",
        "c_",
        extra={"audit": "a_audit", "report_markdown": "a_report_markdown"},
    )
    cells.extend(correction)
    md(
        """## Efek kalender dari training fold

Diagnostik residual dipakai untuk menyusun hipotesis, sehingga pemilihan ini tetap
mengandung optimisme seleksi. Koefisien bulan/tahun dipelajari hanya pada 56.000
baris outer-training, memakai prediksi spline sebagai offset. Spline itu sendiri
tidak pernah melihat label outer-validation. Koreksi log odds dipindahkan ke
baseline Kumo–spline untuk validation dan test. Delapan resep ditetapkan sebelumnya.
"""
    )
    calendar_features, _ = module_cells(
        "diagnose_calendar054.py", "d_", selected={"calendar_features"}
    )
    cells.extend(calendar_features)
    calendar, _ = module_cells(
        "research_calendar054.py",
        "k_",
        extra={
            "calendar_features": "d_calendar_features",
            "objective": "c_objective",
            "audit": "a_audit",
            "report_markdown": "a_report_markdown",
        },
    )
    cells.extend(calendar)
    md(
        """## Kumo Relational: riwayat sebagai tabel terkait

Setiap aplikasi terhubung ke profil pelanggan dan riwayat rekening, biro kredit,
belanja, serta pembayaran yang tersedia sebelum keputusan. ID hanya menjadi
penghubung graf. Pelanggan konteks dan query tidak beririsan; target hanya masuk
melalui y_context. Sampling riwayat memakai waktu dan posisi baris, tanpa label.

Kode pembentukan tabel berikut dapat dijalankan pada clone tanpa cache. Pemeriksaan
awal data asli tidak menghitung KS dan hanya menguji API, memori, serta inferensi.
Evaluasi utama tetap lima fold, delapan anggota, 50.000 konteks, satu hop.
Referensi: [NVIDIA Kumo Relational](https://huggingface.co/nvidia/Kumo-Relational).
"""
    )
    code("b_OUT = ROOT/'data/processed/relational_goal054'")
    builder, _ = module_cells(
        "build_relational054.py",
        "b_",
        selected={"main", "spans_and_recent"},
        extra={"OUT": "b_OUT"},
    )
    cells.extend(builder)
    table_helpers, _ = module_cells("preflight_relational054.py", "t_")
    cells.extend(table_helpers)
    md(
        """Agregasi graf satu hop dihemat dengan menghitung statistik tujuan hanya
untuk tabel pelanggan. Semua pesan masuk, bobot dan fitur tetap dipakai. Perbandingan
FP32/FP16 dengan implementasi SDK mencakup cache, indeks berulang, offset tabel dan
jalur fallback. Hasilnya ada pada `relational_readout_parity.json`."""
    )
    readout, _ = module_cells("relational_readout054.py", "p_")
    cells.extend(readout)
    relational, _ = module_cells(
        "research_relational054.py",
        "f_",
        extra={
            "REVISION": "t_REVISION",
            "related": "t_related",
            "table": "t_table",
            "install": "p_install",
        },
    )
    cells.extend(relational)
    probability_parity, _ = module_cells(
        "verify_relational_predictions054.py",
        "v_",
        selected={"verify"},
        extra={
            "load_inputs": "f_load_inputs",
            "fit_context": "f_fit_context",
            "predictions": "f_predictions",
            "install": "p_install",
        },
    )
    cells.extend(probability_parity)
    md(
        """## Gabungan relasional dan koreksi konteks

Bobot relasional 25%/50%/75% ditetapkan sebelum skor fold relasional pertama
tersedia. Sisa bobot berasal dari kandidat koreksi konteks. Tidak ada meta-model
yang dilatih dari label OOF. Kode berikut baru dapat dijalankan setelah seluruh
lima fold kedua sumber selesai; audit menghitung ulang formula dan hash sumber.
"""
    )
    blending, _ = module_cells("blend_relational054.py", "m_")
    cells.extend(blending)
    md(
        """## Audit independen dan ekspor

Audit memeriksa ID/konteks tiap fold, fingerprint fitur dan checksum prediksi.
Untuk model koreksi, audit memeriksa pemisahan tiga kelompok, gradien loss pada
koefisien tersimpan, dan merekonstruksi seluruh prediksi OOF/test. KS dihitung
memakai SciPy dan sklearn. Hanya eksperimen lima fold lengkap yang diberi skor pooled.
"""
    )
    auditing, _ = module_cells(
        "report_goal054.py",
        "a_",
        extra={
            "audit_correction": "c_audit_correction",
            "add_calendar": "k_add_calendar",
            "audit_calendar": "k_audit_calendar",
            "audit_relational": "f_audit_relational",
            "audit_blend": "m_audit_blend",
        },
    )
    cells.extend(auditing)
    md(
        """## Jalankan atau review eksperimen

Flag default membaca hasil yang tersedia agar notebook dapat dipakai mereview riset
yang masih berjalan. Pada clone tanpa prediksi model baru, aktifkan flag keluarga
yang ingin direproduksi. Hentikan proses training terpisah sebelum mengulang folder
eksperimen yang sama. Kode training seluruh keluarga tersedia pada sel sebelumnya.
"""
    )
    code(
        """RUN_TFM = False
RUN_EBM = False
RUN_CONTEXT_CORRECTION = False
RUN_CALENDAR = False
BUILD_RELATED_TABLES = False
RUN_RELATIONAL_PREFLIGHT = False
RUN_RELATIONAL_PARITY = False
RUN_RELATIONAL = False
RUN_FIXED_RELATIONAL_BLENDS = False
if RUN_TFM:
    g_run(argparse.Namespace(candidates=[s['name'] for s in g_specifications()],folds=None,batch_size=8192,test=True))
if RUN_EBM:
    e_run(['unweighted','class_balanced'])
if RUN_CONTEXT_CORRECTION:
    c_run()
if RUN_CALENDAR:
    k_run()
if BUILD_RELATED_TABLES:
    b_main()
if RUN_RELATIONAL_PREFLIGHT:
    f_run(preflight=True)
if RUN_RELATIONAL_PARITY:
    v_verify()
if RUN_RELATIONAL:
    f_run()
if RUN_FIXED_RELATIONAL_BLENDS:
    m_run()
research_report = a_audit()
a_report_markdown(research_report)
display(pd.DataFrame(research_report['comparisons']).drop(columns=['fold_ks'],errors='ignore'))
display(pd.DataFrame([{'candidate':name,'completed_folds':count} for name,count in research_report['completed_fold_counts'].items()]))
print('Target KS OOF >= 0,54 tercapai:',research_report['target_reached'])
print(research_report['caveat'])
"""
    )
    code(
        """completed = [row for row in research_report['comparisons'] if row.get('candidate')]
best_new = max(completed,key=lambda row:row['ks']) if completed else None
base_result = research_report['comparisons'][0]
fold_review = pd.DataFrame({'fold':range(1,6),'baseline_ks':base_result['fold_ks']})
if best_new:
    fold_review['candidate_ks'] = best_new['fold_ks']
    fold_review['delta'] = fold_review.candidate_ks-fold_review.baseline_ks
    print('Kandidat baru terbaik saat eksekusi:',best_new['name'])
display(fold_review)
"""
    )
    md(
        """## Submission kandidat goal

Ekspor goal berjalan jika kandidat baru mencapai KS OOF 0,54. Flag interim dapat
membuat kandidat riset yang pooled KS-nya meningkat dan membaik pada kelima fold;
file interim tersebut belum memenuhi goal dan skor public-nya belum diketahui.
Probabilitas test
dirata-ratakan dari model lima fold, dicampur sesuai bobot validasi, lalu diurutkan
melalui application_id pada sample submission. CSV baru tidak mewarisi skor public
0,56393. Checkpoint public dan submission sebelumnya dipertahankan.
"""
    )
    code(
        """EXPORT_IF_TARGET = True
EXPORT_INTERIM_CANDIDATE = True
exported_submission = None
eligible_target = bool(best_new and best_new['ks'] >= 0.54)
eligible_interim = bool(best_new and best_new['ks'] > base_result['ks'] and best_new['fold_wins'] == 5)
if (EXPORT_IF_TARGET and eligible_target) or (EXPORT_INTERIM_CANDIDATE and eligible_interim):
    exported_submission = a_export(research_report,best_new['candidate'],best_new['new_model_weight'])
    source = ROOT/exported_submission['file']
    local_submission = OUT/'submission.csv'
    local_submission.write_bytes(source.read_bytes())
    assert hashlib.sha256(local_submission.read_bytes()).hexdigest()==exported_submission['sha256']
    print('Submission:',source)
    print('Salinan notebook:',local_submission)
    print('Public score: belum diketahui')
    print('Target KS OOF 0,54 tercapai:',eligible_target)
    display(pd.read_csv(local_submission).head())
else:
    print('Belum ada kandidat yang memenuhi syarat ekspor aktif.')
assert hashlib.sha256((ROOT/g_PUBLIC_CHECKPOINT).read_bytes()).hexdigest()==g_PUBLIC_SHA256
"""
    )
    for c in cells:
        if c.cell_type == "code":
            ast.parse(c.source)
            c.outputs = []
            c.execution_count = None
    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3.11 (LOGIKA research)",
                "language": "python",
                "name": "logika-research054",
            },
            "language_info": {"name": "python", "version": "3.11.13"},
        },
    )
    nbf.validate(nb)
    path = ROOT / "notebook/12_goal054_research.ipynb"
    nbf.write(nb, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    path = build()
    if args.execute:
        from nbclient import NotebookClient

        protected = [ROOT / "analisis_logika.ipynb", *ROOT.glob("submission*.csv")]
        before = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in protected
        }
        nb = nbf.read(path, as_version=4)
        NotebookClient(
            nb,
            timeout=3600,
            kernel_name="logika-research054",
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
        result = dict(
            notebook=str(path.relative_to(ROOT)),
            notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            executed_code_cells=len(code_cells),
            errors=0,
            existing_files_preserved=after,
            mode="Review cached research; training flags remain explicit in notebook",
        )
        (DEST / "notebook_verification.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(path, flush=True)


if __name__ == "__main__":
    main()
