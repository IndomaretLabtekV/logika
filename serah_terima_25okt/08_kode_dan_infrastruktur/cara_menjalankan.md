# Cara menjalankan (resep yang terbukti aman di laptop ini)

## Environment
- `cd /home/veinsan/Projects/logika/eda && ../.venv/bin/python <script>` (Python 3.13, polars 2.0.0, sdm 0.2.1).
  `python3` sistem **tidak** punya polars (kesalahan awal sesi).
- Shell tool = zsh: tidak ada word-splitting `$var` di `for` — gunakan `${a%%|*}` atau panggil satu per satu.

## Pola job berat
```bash
# guard dulu (pola ter-anchor; trik [x] supaya tidak menangkap shell sendiri)
(nohup ../.venv/bin/python thermal_guard.py out/NNN_telemetry.csv '^[^ ]*python3? 22[4]_strict' > out/NNN_guard.log 2>&1 &)
# job (satu member/fold per proses; PYTORCH_CUDA_ALLOC_CONF hanya dibaca saat CUDA init)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 taskset -c 0-11 nice -n 10 \
  ../.venv/bin/python 224_strict_tfm_variant.py --protocol <json> --out ../results/vNN/strict_variant \
  --pool-root ../results/v60/strict_tfm --stage outer --fold 0 --member <view>_s<seed>
# hentikan guard dengan kill PID (SIGTERM → otomatis SIGCONT semua target); JANGAN pkill -f dengan pola literal
G=$(ps -eo pid,args | awk '$3=="thermal_guard.py"{print $1}'); kill $G
```
- Batas memori Kumo large (fp16, 8 GB): buffer ≈ R × (kolom+4) × 512 B + ~1,9 GB overhead + cache KV per estimator.
  Terbukti muat: cap500@16k (6,10 GiB, fraction 0,85), cap300@28k (5,7 GiB), main_small 56k e8 (fraction 0,7),
  cap128@48k hanya sebagai 8×e1 berurutan (4,43 GiB).
- Waktu (dengan pause guard): cap500@8k ≈10 mnt/fold (44k query), cap500@16k ≈12,5, cap300@28k ≈11–14, main_small e8
  ≈7–12, LGB v22 5-fold ≈3 mnt, screening inner arm ≈4–6 mnt.

## Evaluasi kandidat
```bash
../.venv/bin/python 221_evaluate_recipe_candidates.py results/v64/strict_variant::cap300_ctx28k_s2026 \
  results/v62/strict_variant::main_small_s2026 --tag <nama>        # → results/v60/candidates/K__<nama>/
../.venv/bin/python 238_series_oof_eval.py <tabel1,tabel2> 0,1,2,3,4  # fitur dari results/v66/series_oof/<tabel>_f<k>.parquet
```

## Submisi
```bash
../.venv/bin/python 225_prepare_submission.py results/v60/candidates/<dir> <nama> <blend|kumo>   # validasi + preflight
kaggle competitions submit -c data-science-competition-logika-ui-2026 -f results/v63/<nama>/submission.csv -m "<isi + pertanyaan riset + perkiraan>"
kaggle competitions submissions -c data-science-competition-logika-ui-2026 --csv    # status (deskripsi mengandung koma → parse csv)
```
- Deskripsi ±300–400 karakter; cek Spearman vs submisi lama sebelum memperkirakan public.

## Replay notebook
```bash
cd notebook && ../.venv/bin/python build_final.py            # tulis final.ipynb (14 sel)
```
`run_nb.py` (simpan di scratchpad; dipakai 24 Okt):
```python
import os, sys, time, nbformat
from nbclient import NotebookClient
nb = nbformat.read(sys.argv[1], as_version=4)
os.environ["LOGIKA_OUTPUT_DIR"] = sys.argv[2]
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": sys.argv[3]}}).execute()
nbformat.write(nb, sys.argv[4])
```
Pemanggilan: `taskset -c 0-11 nice -n 10 .venv/bin/python run_nb.py notebook/final.ipynb <out_dir> notebook <executed.ipynb>`
lalu bandingkan SHA `<out_dir>/submission.csv` dengan `artifacts/history_utama/results/v59/v41_lowvar_today/reproduction.json`
(`submission_sha256` = `d7e17b07…`).

## Monitoring (Claude Code)
- Monitor log hasil: `tail -n0 -f <log> | grep --line-buffered -E "^(member)…|FAILED|Error"`; monitor kedaluwarsa tiap
  30 menit → re-arm. Alarm suhu: hanya bila panas **dan** tidak di-pause 2 sampel.
