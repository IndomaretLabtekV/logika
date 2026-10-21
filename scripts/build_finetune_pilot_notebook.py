"""Readable EDA and nested-training audit for the single-fold fine-tuning pilot."""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import pprint

import nbformat as nbf
from build_kumo_notebook import function

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/optimization/kumo_finetune"


def build():
    previous = nbf.read(ROOT / "notebook/08_kumo_tabular.ipynb", as_version=4)
    cells = copy.deepcopy(previous.cells[:67])
    cells[
        0
    ].source = """# 11 — Pilot fine-tuning Kumo: EDA dan validasi bertingkat

Pilot ini melatih dua layer ICL terakhir, norm dan kepala klasifikasi Kumo Large.
Checkpoint 0/32/64/128 dipilih memakai ROC-AUC inner-validation, seluruhnya di dalam
outer-training. Outer-validation tidak dipakai untuk pembaruan bobot/pemilihan langkah.

Hanya fold pertama dari tiga fold seed 8247 yang dijalankan pada pilot ini.
**Checkpoint langkah 0 terpilih**: ROC-AUC internal ketiga checkpoint hasil training
sedikit lebih rendah. Prediksi outer fold tersebut tetap memakai bobot pretrained.
Hasil satu fold tidak menunjukkan performa fine-tuning pada seluruh dataset.
Pilot tidak mengekspor submission baru; checkpoint public 0,56262 tetap disimpan.

Kode mengikuti contoh resmi [NVIDIA SDM fine-tuning](https://github.com/NVIDIA/structured-data-models/blob/main/examples/tabular/finetune.py).
Gunakan Python 3.11, environments/teman/requirements-kumo.txt dan GPU CUDA. Sel berikut memuat EDA,
agregasi V5, training dan prediksi pilot secara langsung untuk direview.
"""
    cells[2].source = (
        cells[2]
        .source.replace('RUN_NAME = "08_kumo"', 'RUN_NAME = "11_finetune_pilot"')
        .replace("PUBLIC_SCORE = 0.56262", "PUBLIC_SCORE = None")
        .replace(
            'print("Output notebook:", OUT / "submission.csv")',
            'print("Output audit:", OUT)',
        )
    )

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))

    def code(text):
        ast.parse(text)
        cells.append(nbf.v4.new_code_cell(text))

    columns = json.loads(
        (
            ROOT / "artifacts/optimization/v5/confirmation_recipe_v5_tabm_weight1.json"
        ).read_text()
    )["spec"]["columns"]
    md(
        """## Data dan bobot yang boleh diperbarui

Fitur tetap 90 kolom yang sama dengan checkpoint Kumo. Dari outer-training,
15% menjadi inner-validation. Episode memakai 1.024 context + 512 query dari
inner-fitting. Transformasi numerik dipilih acak dari empat pilihan default.
AdamW memakai learning rate 1e-5, weight decay 1e-3, dan gradient clipping 1.
Training memakai BF16; evaluasi konteks memakai FP16 sebagaimana baseline.
"""
    )
    code(
        "DEST = ROOT / 'artifacts/optimization/kumo_finetune'\nNAME = 'v5_last2_ft128_e8'\nCHECKPOINT_STEPS = [0,32,64,128]\n_MODEL_POOL = None\nDERIVED_COLUMNS = []\nBASELINE_SPEC = "
        + pprint.pformat(
            dict(name="baseline_e8", columns=columns), width=90, sort_dicts=False
        )
        + "\ndef definitions():\n    return [BASELINE_SPEC]\n"
        + function("improve_kumo.py", "fingerprint")
        + "\n\n"
        + function("improve_kumo.py", "load_frame")
    )
    for name in ["ks", "metrics", "make_tables", "context_indices", "predict_batches"]:
        code(function("kumo_tabular.py", name))
    for name in [
        "id_hash",
        "parameter_snapshot",
        "restore_parameters",
        "model_bundle",
        "training_recipe",
        "inner_metrics",
    ]:
        code(function("finetune_kumo.py", name))
    md(
        """## Inner training dan pemilihan checkpoint

ROC-AUC dipakai untuk pilihan checkpoint karena lebih stabil daripada maksimum
jarak CDF pada satu subset kecil. Ini tidak mengubah metrik kompetisi: penilaian
model tetap memerlukan KS OOF pada seluruh outer fold. State pretrained dipulihkan
sebelum setiap fold; training tidak dimulai dari bobot hasil fold sebelumnya.
"""
    )
    code(function("finetune_kumo.py", "fit_inner"))
    md(
        """## Evaluasi pilot outer fold

Flag FORCE_RETRAIN_PILOT=False membaca cache yang resep dan inputnya cocok.
True mengulangi training 128 langkah. Jika checkpoint 0 terpilih, prediksi frozen
yang identik dapat dipakai kembali. Probabilitas test per fold adalah artefak audit,
belum merupakan submission ensemble seluruh fold.
"""
    )
    code(function("finetune_kumo.py", "frozen_sources"))
    code(function("finetune_kumo.py", "evaluate"))
    code(
        """FORCE_RETRAIN_PILOT = False
args = argparse.Namespace(splits=3,fold_seed=8247,folds=[0],test=True,force=FORCE_RETRAIN_PILOT)
evaluate(args)
directory = DEST / 'cv3_seed8247' / NAME
pilot = json.loads((directory / 'fold0_metadata.json').read_text())
display(pd.DataFrame(pilot['inner']['history']))
print('Checkpoint terpilih:',pilot['inner']['selected_step'])
print('KS/AUC outer fold pertama:',pilot['metrics'])
print('Memakai kembali prediksi frozen:',pilot['reused_frozen_predictions'])
print('Puncak memori GPU:',pilot['peak_gpu_gb'],'GB')
assert pilot['inner']['selected_step'] == max(pilot['inner']['history'],key=lambda row:row['auc'])['step']
"""
    )
    md(
        """## Batas kesimpulan

Pilot hanya mencakup satu outer fold. Belum ada seleksi gabungan atau konfirmasi
lima fold fine-tuning, sehingga tidak ada CSV kompetisi baru dari eksperimen ini.
Pipeline lanjutan tersedia pada scripts/run_kumo_finetune.py untuk melatih semua
fold dan memeriksa bobot tetap 25%/50% bila ingin meneruskan eksperimen.
Kandidat gabungan Kumo–spline direview terpisah pada notebook 10.
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
                "name": "logika-kumo",
                "language": "python",
            },
            "language_info": {"name": "python", "version": "3.11.13"},
        },
    )
    nbf.validate(nb)
    path = ROOT / "notebook/11_kumo_finetune_pilot.ipynb"
    nbf.write(nb, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    path = build()
    if args.execute:
        from nbclient import NotebookClient

        checkpoint = ROOT / "results/teman/submission_kumo.csv"
        before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        nb = nbf.read(path, as_version=4)
        NotebookClient(
            nb,
            timeout=3600,
            kernel_name="logika-kumo",
            resources={"metadata": {"path": str(ROOT)}},
            on_cell_start=lambda cell, cell_index: print(
                "Pilot notebook cell", cell_index, flush=True
            ),
        ).execute()
        nbf.write(nb, path)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert all(c.execution_count is not None for c in code_cells)
        assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
        (DEST / "notebook_verification.json").write_text(
            json.dumps(
                dict(
                    notebook=str(path.relative_to(ROOT)),
                    executed_code_cells=len(code_cells),
                    errors=0,
                    notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    checkpoint_056262_unchanged=True,
                    pilot_fold_count=1,
                    submission_exported=False,
                ),
                indent=2,
            )
            + "\n"
        )
    print(path, flush=True)


if __name__ == "__main__":
    main()
