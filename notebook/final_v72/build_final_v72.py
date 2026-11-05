"""Build the submitted v72 notebook in the findit-v17 layout.

Every code cell of v72 is copied verbatim (source, outputs, metadata) from
v72_source.ipynb. Markdown is rewritten, and a few new chart cells are added.
The chart cells only read variables that already exist in the notebook; their
outputs are produced here from results-final/, the artefacts of the same run.

Run from the repo root:  python notebook/final_v72/build_final_v72.py
Needs polars, numpy, pandas, scipy, scikit-learn, matplotlib, lightgbm.
"""
import base64
import contextlib
import copy
import io
import json
import sys
from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).parent / "v72_source.ipynb"
OUT = ROOT / "logika-v72-final-t4.ipynb"
RESULTS = ROOT / "results-final" / "v72_final"
HR = '<hr style="border: 2px solid #8E7B6B; margin-top: 10px;">'

src = json.loads(SRC.read_text(encoding="utf-8"))


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid4().hex[:8], "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "id": uuid4().hex[:8],
        "metadata": {"new_chart": True}, "outputs": [], "source": source,
    }


def orig(i: int) -> dict:
    cell = src["cells"][i]
    assert cell["cell_type"] == "code", i
    return copy.deepcopy(cell)


def section(title: str, anchor: str) -> dict:
    return md(f"{HR}\n\n# {title} <a name=\"{anchor}\"></a>\n\n{HR}")


def sub(title: str) -> dict:
    return md(f"---\n\n## {title}")


def insight(*paras: str) -> dict:
    return md("> ### Insights\n" + "\n>\n".join(f"> {p}" for p in paras))


cells = []

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
cells.append(md(
    f"{HR}\n"
    "<br>\n"
    '<h1 style="font-family:verdana; font-size:26px"><center>~ DSC LOGIKA UI 2026: Credit Risk dengan GBDT dan Rantai Markov ~</center></h1>\n'
    '<h3 style="font-family:verdana; font-size:14px; color:gray"><center>Prediksi gagal bayar DPD 90+ dalam 12 bulan pada pinjaman konsumen Indonesia</center></h3>\n'
    f"{HR}\n\n"
    "# **IndomaretLabtekV**\n\n"
    "### DSC0052"
))

cells.append(md(
    f"{HR}\n\n"
    "# Daftar Isi\n\n"
    "1. [**Pendahuluan**](#1)\n"
    "2. [**Inisialisasi**](#2)\n"
    "3. [**Pemuatan Data dan Audit Integritas**](#3)\n"
    "4. [**Rekayasa Fitur**](#4)\n"
    "5. [**Analisis Data Eksploratif**](#5)\n"
    "6. [**Strategi Validasi**](#6)\n"
    "7. [**Pemodelan**](#7)\n"
    "8. [**Analisis Kesalahan dan Ensemble**](#8)\n"
    "9. [**Riwayat Eksperimen dan Pemilihan Model**](#9)\n"
    "10. [**Inferensi dan Submisi**](#10)\n"
    "11. [**Pasca-Analisis**](#11)\n"
    "12. [**Daftar Pustaka**](#12)"
))

# ---------------------------------------------------------------------------
# 1. Pendahuluan
# ---------------------------------------------------------------------------
cells.append(section("Pendahuluan", "1"))

cells.append(md(
    "## Kompetisi\n\n"
    "Lembaga pembiayaan konsumen perlu mengenali pengajuan pinjaman yang berpotensi bermasalah sebelum dana dicairkan, "
    "hanya dengan informasi yang tersedia saat keputusan diambil. Pada kompetisi **DSC LOGIKA UI 2026**, peserta diminta "
    "membangun model yang memberi skor risiko untuk setiap aplikasi pinjaman. Sebuah pinjaman dianggap mengalami kesulitan "
    "bayar serius bila *Days Past Due* (DPD) akhir harinya mencapai 90 atau lebih dalam 12 bulan kalender sesudah pencairan.\n\n"
    "- **Tugas**: klasifikasi biner dengan keluaran skor peluang `default_probability`\n"
    "- **Target**: `target = 1` bila pinjaman mencapai DPD 90+ dalam 12 bulan, `0` bila tidak\n"
    "- **Metrik**: statistik Kolmogorov-Smirnov (KS)\n"
    "- **Penilaian babak penyisihan**: 45% *public score*, 45% *private score*, 10% reprodusibilitas *notebook*\n"
    "- **Batasan**: hanya data panitia, tanpa data eksternal, dan hanya informasi sebelum `decision_datetime`\n"
    "- **Platform**: Kaggle Notebook, GPU T4 x2"
))

cells.append(md(
    "---\n\n"
    "## Dataset\n\n"
    "*Dataset* berbentuk relasional: satu tabel aplikasi terpilih per customer, ditambah enam tabel riwayat yang terhubung "
    "lewat `customer_id`. Semua nilai uang dalam Rupiah dan seluruh *timestamp* memakai waktu Asia/Jakarta. Kolom "
    "`decision_datetime` adalah batas informasi: fitur hanya boleh memakai kejadian sebelum titik ini.\n\n"
    "| Tabel | Baris | Isi |\n"
    "|---|---:|---|\n"
    "| `train_applications` | 70.000 | Aplikasi terpilih beserta `target` |\n"
    "| `test_applications` | 30.000 | Aplikasi yang harus diprediksi |\n"
    "| `customer_profiles` | 100.000 | Demografi, pekerjaan, pendapatan, tanggungan, tempat tinggal |\n"
    "| `previous_applications` | 201.984 | Riwayat pengajuan pinjaman beserta keputusannya |\n"
    "| `repayment_history` | 1.361.341 | Cicilan per pinjaman lama: jatuh tempo, pembayaran, DPD |\n"
    "| `account_behavior` | 1.084.418 | Ringkasan rekening bulanan: saldo, arus masuk dan keluar |\n"
    "| `bureau_observations` | 797.563 | Laporan biro kredit: skor, utilisasi, *inquiry*, tunggakan |\n"
    "| `ecommerce_transactions` | 4.359.817 | Item belanja *online*: kategori, harga, metode bayar, status |\n\n"
    "Tingkat gagal bayar di *train* adalah 13,58%. Sebanyak 30% aplikasi berasal dari customer tanpa riwayat pinjaman, "
    "sehingga customer ini hanya punya data aplikasi, biro, dan *e-commerce*."
))

cells.append(md(
    "---\n\n"
    "## Metrik\n\n"
    "KS mengukur jarak vertikal terbesar antara distribusi kumulatif skor kelas gagal bayar ($F_1$) dan kelas lancar ($F_0$):\n\n"
    "$$\\text{KS} = \\max_s \\left| F_1(s) - F_0(s) \\right|$$\n\n"
    "Metrik ini hanya bergantung pada urutan skor, bukan nilai absolutnya. Akibatnya kalibrasi global tidak mengubah skor, "
    "dan rata-rata *rank* antarmodel aman dipakai sebagai cara *blending*. KS juga ditentukan oleh satu titik potong, "
    "jadi kesalahan yang mahal adalah salah urut di sekitar titik tersebut."
))

cells.append(md(
    "---\n\n"
    "## Pendekatan: Enam GBDT Global + Rantai Markov *Repayment*\n\n"
    "Pendekatan final (v72) memakai tiga keluarga *gradient boosted decision tree* (GBDT): LightGBM (Ke dkk., 2017), "
    "XGBoost (Chen & Guestrin, 2016), dan CatBoost (Prokhorenkova dkk., 2018). Masing-masing dipakai dalam dua "
    "konfigurasi: standar dan dangkal (sufiks `_s`). Keenam model membaca 242 fitur yang dibangun dari tujuh tabel, termasuk "
    "10 fitur rantai Markov yang memodelkan transisi status telat bayar per customer.\n\n"
    "```\n"
    "Delapan tabel mentah (cutoff decision_datetime)\n"
    "    |\n"
    "    +-- Pipeline fitur per tabel            [232 kolom]\n"
    "    +-- Rantai Markov repayment (5 status)   [10 kolom]\n"
    "         |\n"
    "    Matriks fitur 70.000 train + 30.000 test  [242 kolom]\n"
    "         |\n"
    "    LightGBM | LightGBM_s | XGBoost | XGBoost_s | CatBoost | CatBoost_s\n"
    "         |   StratifiedKFold 5 fold x 3 split seed = 15 booster per model\n"
    "         |   90 booster total\n"
    "    Rata-rata probabilitas per model\n"
    "         |\n"
    "    Rata-rata rank enam model (bobot sama) -> default_probability\n"
    "```\n\n"
    "Tidak ada bobot *blend* yang dituning, tidak ada *stacking*, dan tidak ada *postprocessing*. Resep enam model sudah "
    "dikunci sejak v22, sebelum blok Markov ditambahkan, sehingga pemilihannya tidak bergantung pada OOF akhir."
))

cells.append(md(
    "---\n\n"
    "## Ringkasan Temuan\n\n"
    "| Temuan | Konsekuensi |\n"
    "|---|---|\n"
    "| Telat bayar berbentuk rantai: *hazard* 0,14 sesudah bayar tepat waktu, 0,71 sesudah satu kali telat | Blok Markov per customer, *public* +0,005 |\n"
    "| Level harga belanja adalah sinyal risiko lintas kanal, juga untuk customer tanpa riwayat | Fitur `pp_*`, *public* +0,016 |\n"
    "| Perubahan struktur model menaikkan OOF tetapi menurunkan *public* | Resep tetap enam pohon global |\n"
    "| Pada enam file era Markov, korelasi OOF dengan *public* adalah -0,89 | OOF tertinggi tidak dipakai sebagai penentu tunggal |\n"
    "| Tabel sudah dipotong panitia di `decision_datetime` | Audit *look-ahead* bersih, *cutoff* tetap diterapkan ulang |\n"
    "| *Test* adalah split acak, AUC adversarial 0,501 | `StratifiedKFold` mewakili *test* |\n\n"
    "| Kandidat | OOF KS | *Public* KS |\n"
    "|---|---:|---:|\n"
    "| **v72 (final)** | **0,52580** | **0,56822** |\n"
    "| Enam GBDT tanpa Markov (v39a) | 0,52448 | 0,56337 |\n"
    "| OOF tertinggi, pohon per segmen + MLP (v75 recipe C) | 0,52861 | 0,56374 |"
))

cells.append(md(
    "---\n\n"
    "## Referensi Utama\n\n"
    "- [docs/context.md](docs/context.md) -- deskripsi kompetisi, aturan, dan format submisi\n"
    "- [handoff/2026-10-30/README.md](handoff/2026-10-30/README.md) -- serah terima jalur Markov, model per segmen, dan MLP\n"
    "- [research/autoresearch/findings.md](research/autoresearch/findings.md) -- temuan riset dan protokol yang dikunci sebelum hasil"
))

# ---------------------------------------------------------------------------
# 2. Inisialisasi
# ---------------------------------------------------------------------------
cells.append(section("Inisialisasi", "2"))

cells.append(md("## Environment"))
cells.append(md(
    "*Notebook* ini dijalankan di Kaggle dengan dua GPU Tesla T4. XGBoost dan CatBoost memakai GPU saat pelatihan, "
    "sedangkan LightGBM dan seluruh inferensi berjalan di CPU. Satu kali *Run All* dalam mode `end_to_end` tercatat "
    "selama 1.714 detik (sekitar 28,5 menit), dengan 1.575 detik di antaranya untuk melatih 90 *booster*."
))
cells.append(orig(12))
cells.append(md(
    "> Catatan untuk juri dan panitia: *notebook* ini dijalankan dalam mode `Settings.RUN_MODE = 'end_to_end'`, yaitu "
    "membangun fitur dari CSV mentah, melatih 90 *booster*, lalu menulis `submission.csv`. Cukup pasang dataset kompetisi "
    "dan jalankan *Run All* dengan akselerator GPU T4. Mode `'inference'` melewati pelatihan dan memuat *booster* dari "
    "`v72_bundle` yang terverifikasi SHA256; mode ini membutuhkan *bundle* terpasang sebagai *input* atau `ARTIFACT_URL` terisi."
))

cells.append(sub("Instalasi"))
cells.append(md(
    "Baris berikut memasang versi pustaka yang sama dengan saat artefak dibuat. Peringatan *dependency resolver* tentang "
    "`cudf-polars-cu12` tidak berpengaruh karena *notebook* ini tidak memakai `cudf`."
))
cells.append(orig(14))

cells.append(sub("Pustaka"))
cells.append(md(
    "Seluruh impor dikelompokkan dalam satu sel. Polars menangani agregasi tabel besar (sampai 4,4 juta baris "
    "*e-commerce*) dengan cepat, pandas dipakai sebagai format masukan model, sedangkan LightGBM, XGBoost, dan CatBoost "
    "menjadi tiga keluarga model. SciPy menyediakan `ks_2samp` untuk metrik dan `rankdata` untuk *blending*."
))
cells.append(orig(16))

cells.append(sub("Settings"))
cells.append(md(
    "Kelas `Settings` memusatkan konfigurasi: tiga *split seed* (42, 7, 2026), daftar enam model, mode jalan, perangkat "
    "untuk XGBoost dan CatBoost, serta jalur data dan keluaran. Fungsi `find_dir` mencari `train_applications.csv` di "
    "`/kaggle/input` lalu di folder lokal, sehingga *notebook* berjalan tanpa mengubah jalur. Ubah `RUN_MODE` menjadi "
    "`'inference'` untuk melewati pelatihan, atau `XGB_DEVICE = 'cpu'` dan `CAT_PARAMS = {'task_type': 'CPU'}` bila GPU "
    "tidak tersedia."
))
cells.append(orig(18))

cells.append(sub("Seed"))
cells.append(md(
    "Pengaturan *seed* dilakukan pada semua sumber keacakan Python dan NumPy. Setiap *booster* juga menerima *seed*-nya "
    "sendiri lewat parameter model. Pelatihan GPU tetap dapat menghasilkan selisih numerik kecil antareksekusi karena "
    "urutan reduksi paralel di *hardware*."
))
cells.append(orig(20))

cells.append(sub("Muat Artefak"))
cells.append(md(
    "Pada mode `inference`, sel ini mencari folder artefak yang sudah terbuka, lalu `v72_bundle.zip` yang terpasang "
    "sebagai *dataset*, lalu unduhan dari `ARTIFACT_URL`. SHA256 setiap berkas diverifikasi terhadap `manifest.json`, dan "
    "berkas yang berubah menghentikan *notebook*. Pada mode `end_to_end`, sel ini hanya menyiapkan folder `run/` untuk "
    "prediksi OOF, prediksi *test*, log, dan 90 *booster* hasil pelatihan."
))
cells.append(orig(22))

# ---------------------------------------------------------------------------
# 3. Pemuatan Data dan Audit Integritas
# ---------------------------------------------------------------------------
cells.append(section("Pemuatan Data dan Audit Integritas", "3"))
cells.append(md(
    "Sebelum membangun fitur, delapan tabel dibaca dan diaudit. Fokus audit adalah dua risiko terbesar pada data "
    "kredit: duplikasi unit observasi dan kebocoran informasi dari masa sesudah keputusan."
))

cells.append(md("## Ukuran Tabel"))
cells.append(md(
    "Delapan tabel dibaca dari CSV dengan Polars. Urutan ID pada `sample_submission` disimpan sebagai acuan urutan "
    "baris berkas submisi."
))
cells.append(orig(25))
cells.append(insight(
    "Tabel riwayat jauh lebih besar dari tabel aplikasi: rata-rata sekitar 44 baris *e-commerce*, 14 cicilan, 11 bulan "
    "rekening, dan 8 laporan biro per customer. Seluruh informasi ini harus diringkas menjadi satu baris per aplikasi, "
    "sehingga kualitas agregasi lebih menentukan skor daripada pilihan model."
))

cells.append(sub("Integritas dan Audit Kebocoran"))
cells.append(md(
    "Pemeriksaan berikut menghitung ID ganda, customer yang muncul di *train* dan *test*, serta peristiwa sesudah "
    "keputusan. Audit *repayment* membandingkan DPD terbesar setiap cicilan dengan jarak hari dari jatuh tempo ke "
    "keputusan: DPD yang lebih besar dari jarak itu berarti tabel memuat informasi masa depan."
))
cells.append(orig(27))
cells.append(insight(
    "Tidak ada ID ganda, tidak ada customer yang muncul di *train* dan *test*, dan tidak ada DPD yang melampaui jarak "
    "waktu ke keputusan. Kolom seperti `max_days_past_due` berpotensi membocorkan masa depan bila tabel diambil sesudah "
    "keputusan, tetapi audit menunjukkan panitia sudah memotongnya.",
    "Setiap customer hanya punya satu aplikasi terpilih, jadi unit independen adalah customer dan tidak ada grup yang "
    "bisa terbelah antar-*fold*. *Cutoff* `decision_datetime` tetap diterapkan ulang di setiap pembentuk fitur sebagai "
    "pengaman."
))

cells.append(sub("Fungsi Metrik"))
cells.append(md(
    "Implementasi KS yang dipakai di seluruh *notebook* sama dengan kode evaluasi panitia: `ks_2samp` antara skor kelas "
    "positif dan kelas negatif."
))
cells.append(orig(32))

# ---------------------------------------------------------------------------
# 4. Rekayasa Fitur
# ---------------------------------------------------------------------------
cells.append(section("Rekayasa Fitur", "4"))
cells.append(md(
    "Bagian ini membangun 242 fitur per aplikasi dari tujuh tabel. Analisis eksploratif di bagian 5 dilakukan sesudah "
    "fitur terbentuk karena observasinya memakai fitur yang sama dengan model."
))

cells.append(md("## Prinsip Anti-Kebocoran"))
cells.append(md(
    "Semua pembentuk fitur mengikuti aturan yang sama agar model hanya melihat informasi yang tersedia saat keputusan:\n\n"
    "- **Cutoff per tabel**: cicilan dipakai bila `due_date <= decision_datetime`, laporan biro bila "
    "`available_datetime <= decision_datetime`, dan transaksi bila `transaction_datetime <= decision_datetime`\n"
    "- **Bulan rekening yang sudah selesai**: baris `account_behavior` hanya dipakai bila bulan kalendernya berakhir "
    "sebelum keputusan\n"
    "- **Deduplikasi order**: `order_total` berulang di setiap item, jadi order diringkas ke satu baris sebelum agregasi\n"
    "- **Tanpa label**: tidak ada *target encoding*, dan referensi harga per kategori hanya memakai bulan sebelum bulan keputusan\n"
    "- **Urutan deterministik**: seri diurutkan dengan pemutus seri (`loan_id`, `bureau_observation_id`) agar hasil "
    "sama di setiap eksekusi"
))

cells.append(sub("Pipeline Fitur per Tabel"))
cells.append(md(
    "Setiap fungsi menghasilkan satu baris per customer, lalu `build_all` menggabungkannya dan menambah rasio lintas "
    "tabel. Prefiks kolom menunjukkan sumbernya:\n\n"
    "| Famili | Prefiks | Jumlah | Contoh sinyal |\n"
    "|---|---|---:|---|\n"
    "| Aplikasi dan rasio lintas tabel | tanpa prefiks | 26 | `dti_total`, `income_verif_ratio`, `has_history` |\n"
    "| Biro kredit | `bc_`, `bh_`, `bp_`, `bz_` | 50 | skor per *provider*, tren skor, *inquiry* eksternal |\n"
    "| *E-commerce* | `ec_`, `ec2_` | 40 | akselerasi order 30 hari, pembatalan, belanja malam |\n"
    "| Rekening | `ac_`, `cf_` | 38 | arus kas bebas, porsi arus masuk non-gaji, tren saldo |\n"
    "| *Repayment* | `rp_`, `rq_`, `sv_` | 36 | DPD maksimum per jendela, *streak* telat, cicilan pertama |\n"
    "| Aplikasi lama | `pv_`, `ob_` | 25 | tingkat penolakan, sisa kewajiban terjadwal |\n"
    "| Level harga belanja | `pp_` | 17 | z-score log harga satuan per kategori |\n"
    "| Rantai Markov | `mk_` | 10 | peluang mencapai 90+ dalam 12 cicilan |\n\n"
    "Fitur `pp_*` membutuhkan penjelasan khusus. Untuk setiap kategori produk, log harga satuan distandarkan terhadap "
    "rata-rata dan simpangan baku seluruh transaksi pada bulan-bulan sebelum bulan keputusan. Hasilnya mengukur apakah "
    "customer cenderung membeli barang di atas atau di bawah harga wajar kategorinya."
))
cells.append(orig(35))

cells.append(sub("Rantai Markov *Repayment*"))
cells.append(md(
    "Riwayat cicilan setiap pinjaman diubah menjadi urutan lima status berdasarkan `max_days_past_due`: lancar (0), "
    "1-29, 30-59, 60-89, dan 90+ hari. Untuk setiap customer, transisi antarstatus dihitung menjadi matriks $N$ berukuran "
    "5 x 5. Customer dengan sedikit cicilan punya estimasi yang sangat bising, jadi matriks transisi customer disusutkan "
    "(*shrinkage*) ke matriks populasi $\\bar{P}$ yang dihitung tanpa label:\n\n"
    "$$\\hat{P}_{ij} = \\frac{\\alpha \\bar{P}_{ij} + N_{ij}}{\\alpha + \\sum_k N_{ik}}, \\quad \\alpha = 5$$\n\n"
    "Status 90+ dijadikan status penyerap, lalu peluang mencapai status itu dihitung dengan mengalikan vektor status "
    "sebanyak 3, 6, dan 12 langkah dari status lancar. Selain peluang tersebut, blok ini mengeluarkan peluang masuk telat "
    "(`mk_entry`), peluang pulih (`mk_cure`), peluang memburuk (`mk_escalate`), peluang bertahan di 90+ (`mk_stay90`), "
    "dan jumlah transisi. Customer tanpa transisi mendapat nilai kosong, bukan nilai populasi, agar model dapat "
    "membedakannya.\n\n"
    "Fungsi ini membaca `repayment_history` tanpa filter *cutoff* tambahan karena audit di bagian 3 sudah memastikan "
    "tidak ada cicilan yang jatuh tempo sesudah keputusan."
))
cells.append(orig(37))

cells.append(sub("Membangun Matriks Fitur"))
cells.append(md(
    "Fitur dibangun untuk *train* dan *test* sekaligus dengan kode yang sama, sehingga transformasi keduanya identik. "
    "*Target* *test* tetap kosong dan tidak dipakai di pembentuk fitur mana pun."
))
cells.append(orig(39))
cells.append(insight(
    "Matriks akhir berisi 100.000 baris dan 247 kolom: 242 fitur ditambah lima kolom identitas (`application_id`, "
    "`customer_id`, `decision_datetime`, `target`, `is_train`). Kelima kolom identitas dikeluarkan sebelum pelatihan."
))

# ---------------------------------------------------------------------------
# 5. EDA
# ---------------------------------------------------------------------------
cells.append(section("Analisis Data Eksploratif", "5"))
cells.append(md(
    "Setiap observasi di bagian ini ditulis bersama konsekuensinya untuk model. Pola yang tidak mengubah keputusan "
    "pemodelan tidak ditampilkan."
))

cells.append(md("## Distribusi Target"))
cells.append(md(
    "Proporsi kelas menentukan apakah perlu pembobotan kelas dan bagaimana *fold* dibagi. Visualisasi berikut "
    "menunjukkan jumlah aplikasi lancar dan gagal bayar pada *train*."
))
cells.append(code(
    "_counts = np.bincount(y)\n"
    "_labels = ['Lancar (0)', 'Gagal bayar (1)']\n"
    "_colors = ['#2a78d6', '#eb6834']\n"
    "\n"
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))\n"
    "\n"
    "_bars = ax1.bar(_labels, _counts, color=_colors, width=0.55)\n"
    "ax1.bar_label(_bars, labels=[f'{c:,}' for c in _counts], padding=3)\n"
    "ax1.set_ylabel('Jumlah aplikasi')\n"
    "ax1.set_title('Distribusi Target (Train)')\n"
    "ax1.spines[['top', 'right']].set_visible(False)\n"
    "\n"
    "ax2.pie(_counts, labels=_labels, colors=_colors, autopct='%1.2f%%', startangle=90,\n"
    "        wedgeprops=dict(edgecolor='white', linewidth=2))\n"
    "ax2.set_title('Proporsi Target')\n"
    "\n"
    "fig.suptitle('Target Distribution', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "print(f'Total train : {len(y)}')\n"
    "print(f'Bad rate    : {y.mean():.4f}')\n"
    "print(f'Rasio 0:1   : {_counts[0] / _counts[1]:.2f}x')"
))
cells.append(insight(
    "Sekitar 13,6% aplikasi gagal bayar, dengan rasio kelas sekitar 6,4 banding 1. KS hanya membandingkan urutan skor "
    "kedua kelas, jadi ketidakseimbangan ini tidak perlu dikoreksi dengan *class weight* atau *oversampling*. Yang perlu "
    "dijaga adalah proporsi kelas yang sama di setiap *fold*, sehingga validasi memakai `StratifiedKFold`."
))

cells.append(sub("Telat Bayar Berbentuk Rantai"))
cells.append(md(
    "*Hazard* telat dihitung per panjang telat beruntun sebelumnya ($k$) pada cicilan customer *train*, tanpa melihat "
    "*target*. Bila telat bayar terjadi acak dengan tarif tetap, *hazard* akan sama di setiap $k$."
))
cells.append(orig(41))
cells.append(insight(
    "Sesudah cicilan yang dibayar tepat waktu, peluang telat pada cicilan berikutnya sekitar 0,14. Sesudah satu kali "
    "telat, peluang itu melompat ke sekitar 0,71, lalu terus naik sampai 0,85 sesudah tiga kali telat beruntun. Riwayat "
    "pembayaran lebih mirip rantai keadaan daripada tarif tetap.",
    "Fitur tarif telat biasa (`rp_frac_dpd1`) merata-ratakan seluruh cicilan dan kehilangan struktur ini. Hipotesisnya, "
    "peluang mencapai 90+ yang dihitung dari matriks transisi per customer menambah informasi di atas tarif telat. "
    "Hipotesis ini diuji sebagai Eksperimen E1 di bagian 9: blok Markov menaikkan OOF dan *public* sekaligus."
))

cells.append(sub("Level Harga Belanja"))
cells.append(md(
    "Tabel berikut menunjukkan *bad rate* per desil `pp_mean`, yaitu rata-rata z-score log harga satuan per kategori. "
    "Tabel dihitung untuk seluruh baris dan untuk customer tanpa riwayat pinjaman."
))
cells.append(orig(44))
cells.append(insight(
    "*Bad rate* naik monoton dari sekitar 2% di desil terbawah ke sekitar 40% di desil teratas, dengan KS satu fitur "
    "0,369. Pola yang sama muncul pada customer tanpa riwayat (1,4% ke 36,2%), kelompok yang tidak punya data "
    "*repayment* maupun rekening.",
    "Arah sinyal ini berlawanan dengan intuisi bahwa belanja mahal menandakan kemampuan bayar. Interpretasi yang "
    "konsisten dengan data adalah satu risiko laten per customer yang tercermin di beberapa kanal sekaligus. Penambahan "
    "fitur `pp_*` memberi kenaikan *public* terbesar dalam riwayat tim, yaitu +0,016."
))

cells.append(sub("Pergeseran Train dan Test"))
cells.append(md(
    "Jarak KS dua sampel antara *train* dan *test* dihitung untuk setiap kolom numerik. Lima kolom dengan pergeseran "
    "terbesar ditampilkan."
))
cells.append(orig(47))
cells.append(insight(
    "Pergeseran terbesar hanya 0,0185, jauh di bawah ambang yang bermakna. Pemeriksaan klasifikasi adversarial "
    "sebelumnya memberi AUC 0,501, setara tebakan acak. *Test* adalah split acak dari populasi yang sama, sehingga tidak "
    "perlu *reweighting* atau adaptasi domain, dan validasi silang acak sudah mewakili *test*."
))

# ---------------------------------------------------------------------------
# 6. Strategi Validasi
# ---------------------------------------------------------------------------
cells.append(section("Strategi Validasi", "6"))
cells.append(md(
    "Validasi harus mensimulasikan cara *test* dibentuk. Karena *test* adalah split acak customer, dan setiap customer "
    "hanya punya satu aplikasi, validasi cukup mensimulasikan pemilihan baris acak dengan proporsi kelas terjaga."
))

cells.append(md("## Skema Split"))
cells.append(md(
    "`StratifiedKFold` lima *fold* diulang dengan tiga *split seed* (42, 7, 2026). Pada riset, sebagian *fold* latih "
    "(1/7) disisihkan sebagai *inner validation* untuk *early stopping*, lalu model dilatih ulang di seluruh *fold* latih "
    "dengan jumlah pohon x1,15. Jumlah pohon hasil prosedur itu dicatat di `ROUNDS`, sehingga *notebook* ini langsung "
    "melatih dengan jumlah pohon tetap. *Fold* validasi hanya dipakai untuk menilai.\n\n"
    "```\n"
    "70.000 baris train\n"
    "    |\n"
    "    +-- untuk setiap split seed (42, 7, 2026) dan fold k = 0..4\n"
    "         |\n"
    "         +-- fold latih 4/5  -> booster dengan ROUNDS[model][seed_k] pohon\n"
    "         +-- fold valid 1/5  -> prediksi OOF (hanya dinilai)\n"
    "    |\n"
    "    OOF = rata-rata 3 seed per baris, test = rata-rata 15 booster\n"
    "```"
))

cells.append(sub("Checklist Kebocoran"))
cells.append(md(
    "- [x] *Fold* validasi tidak dipakai untuk *early stopping*\n"
    "- [x] Tidak ada *target encoding*\n"
    "- [x] Tidak ada seleksi fitur berbasis label\n"
    "- [x] Tidak ada *threshold tuning*, KS hanya memakai urutan\n"
    "- [x] Bobot *blend* tidak dituning, rata-rata *rank* bobot sama\n"
    "- [x] Tidak ada grup yang terbelah, satu customer satu baris\n"
    "- [x] *Cutoff* `decision_datetime` diterapkan di setiap tabel"
))

cells.append(sub("Penyusunan Frame Model"))
cells.append(md(
    "Fungsi `frames` mengubah matriks Polars menjadi pandas: kolom diurutkan alfabetis, empat kolom kategorikal "
    "dijadikan `Categorical` dengan kategori dari *train*, dan nilai tak hingga diganti `NaN`. Fungsi `outer` membuat "
    "pembagian *fold* yang sama untuk keenam model."
))
cells.append(orig(52))

# ---------------------------------------------------------------------------
# 7. Pemodelan
# ---------------------------------------------------------------------------
cells.append(section("Pemodelan", "7"))
cells.append(md(
    "Tiga keluarga GBDT dipakai karena masing-masing membangun pohon dengan cara berbeda: LightGBM tumbuh per daun, "
    "XGBoost per kedalaman, dan CatBoost memakai pohon simetris dengan *ordered boosting*. Versi dangkal (`_s`) "
    "menambah regularisasi dan membatasi interaksi, sehingga kesalahannya sedikit berbeda dari versi standar."
))

cells.append(md("## Arsitektur Ensemble"))
cells.append(md(
    "| Model | Kedalaman / daun | *Learning rate* | Regularisasi | Subsampel kolom | Rata-rata pohon |\n"
    "|---|---|---:|---|---:|---:|\n"
    "| `lgb` | 31 daun | 0,02 | `min_child_samples=100`, L2 5 | 0,5 | 604 |\n"
    "| `lgb_s` | 7 daun, kedalaman 3 | 0,02 | `min_child_samples=200`, L2 10 | 0,4 | 1.639 |\n"
    "| `xgb` | kedalaman 5 | 0,02 | `min_child_weight=5`, L2 5 | 0,5 | 838 |\n"
    "| `xgb_s` | kedalaman 3 | 0,02 | `min_child_weight=20`, L2 10 | 0,4 | 1.512 |\n"
    "| `cat` | kedalaman 6 | 0,03 | L2 5 | - | 1.395 |\n"
    "| `cat_s` | kedalaman 4 | 0,03 | L2 10 | - | 1.957 |\n\n"
    "Kolom kategorikal ditangani secara bawaan oleh setiap pustaka: `Categorical` pandas untuk LightGBM dan XGBoost, "
    "serta `cat_features` berupa *string* untuk CatBoost."
))

cells.append(sub("Konfigurasi dan `fit_one`"))
cells.append(md(
    "Parameter sama persis dengan kernel v72. `ROUNDS` menyimpan jumlah pohon untuk setiap kombinasi model, *seed*, dan "
    "*fold* dari log kernel, jadi tahap *early stopping* tidak diulang. `fit_one` melatih satu *booster* pada indeks latih "
    "dan memprediksi daftar *frame* yang diminta (*fold* validasi dan *test*)."
))
cells.append(orig(54))

cells.append(sub("Pelatihan 90 *Booster*"))
cells.append(md(
    "`train_member` melatih satu model untuk 3 *seed* x 5 *fold*. Setiap *booster* disimpan ke `CFG.MODEL_DIR`, KS *fold* "
    "validasinya dicatat, prediksi OOF dirata-ratakan antar-*seed*, dan prediksi *test* dirata-ratakan dari 15 *booster*. "
    "Peringatan CatBoost tentang AUC di GPU hanya memengaruhi frekuensi pencatatan metrik, bukan pelatihan."
))
cells.append(orig(56))
cells.append(insight(
    "Seluruh 90 *booster* selesai dalam sekitar 1.575 detik. CatBoost paling lama (sekitar 790 detik untuk dua "
    "konfigurasi), disusul LightGBM di CPU (sekitar 577 detik), sedangkan XGBoost di GPU hanya sekitar 208 detik. "
    "Matriks latih akhir berukuran 70.000 x 242 dan *test* 30.000 x 242."
))

cells.append(sub("Diagnostik per *Fold*"))
cells.append(md(
    "Skor per *fold* dibaca dari log pelatihan. Rata-rata, simpangan baku, *fold* terburuk, dan *fold* terbaik "
    "menunjukkan seberapa besar *noise* evaluasi dibanding selisih antarmodel."
))
cells.append(orig(58))
cells.append(code(
    "_ldf = logs.to_pandas()\n"
    "_data = [_ldf.loc[_ldf['model'] == m, 'ks'].to_numpy() for m in CFG.SIX]\n"
    "_rounds = [_ldf.loc[_ldf['model'] == m, 'budget'].mean() for m in CFG.SIX]\n"
    "\n"
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={'width_ratios': [3, 2]})\n"
    "\n"
    "ax1.boxplot(_data, widths=0.5, showfliers=False, medianprops=dict(color='#eb6834', linewidth=2))\n"
    "for i, d in enumerate(_data):\n"
    "    _jit = np.random.default_rng(i).uniform(-0.12, 0.12, len(d))\n"
    "    ax1.scatter(np.full(len(d), i + 1) + _jit, d, s=18, color='#2a78d6', alpha=0.7, zorder=3)\n"
    "ax1.set_xticks(range(1, len(CFG.SIX) + 1), CFG.SIX)\n"
    "ax1.set_ylabel('KS fold validasi')\n"
    "ax1.set_title('KS per Fold (5 fold x 3 seed)')\n"
    "ax1.spines[['top', 'right']].set_visible(False)\n"
    "\n"
    "_bars = ax2.bar(CFG.SIX, _rounds, color='#2a78d6', width=0.55)\n"
    "ax2.bar_label(_bars, fmt='%.0f', padding=3)\n"
    "ax2.set_ylabel('Rata-rata jumlah pohon')\n"
    "ax2.set_title('Budget Pohon per Model')\n"
    "ax2.spines[['top', 'right']].set_visible(False)\n"
    "\n"
    "fig.suptitle('Fold Diagnostics', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(insight(
    "Rata-rata KS per *fold* keenam model berada di rentang sempit 0,5246 sampai 0,5260, sedangkan rentang antar-*fold* "
    "dalam satu model mencapai sekitar 0,03 (0,512 sampai 0,547). Selisih antarmodel jauh lebih kecil dari *noise* "
    "*fold*, sehingga kandidat tidak boleh dibandingkan dari satu *fold*. Kandidat dinilai dengan kemenangan per *fold* "
    "dan *paired bootstrap*.",
    "Model dangkal membutuhkan dua sampai tiga kali lebih banyak pohon untuk mencapai skor yang setara. Kapasitas per "
    "pohon yang lebih kecil diimbangi jumlah pohon, dengan hasil akhir yang hampir sama."
))

# ---------------------------------------------------------------------------
# 8. Analisis Kesalahan dan Ensemble
# ---------------------------------------------------------------------------
cells.append(section("Analisis Kesalahan dan Ensemble", "8"))
cells.append(md(
    "Bagian ini memuat prediksi OOF keenam model, mengukur di kelompok mana model lemah, lalu menggabungkan keenamnya "
    "dengan rata-rata *rank*."
))

cells.append(md("## KS per Kelompok Riwayat"))
cells.append(md(
    "Prediksi OOF dan *test* dimuat dari `CFG.PRED_DIR` setelah urutan ID dan *target* diverifikasi. Setiap model lalu "
    "dinilai secara *pooled* dan terpisah untuk customer tanpa riwayat dan customer berriwayat. *Pooled* KS dapat "
    "menyembunyikan penurunan di salah satu kelompok, jadi evaluasi per kelompok diperlukan."
))
cells.append(orig(74))
cells.append(insight(
    "Semua model lebih lemah pada customer tanpa riwayat (KS 0,489 sampai 0,500) dibanding customer berriwayat (0,533 "
    "sampai 0,536). Kelompok tanpa riwayat hanya punya data aplikasi, biro, dan *e-commerce*, sehingga sinyalnya lebih "
    "sedikit. Selisih ini memotivasi eksperimen model per segmen di bagian 9.",
    "Di antara keenam model, CatBoost paling lemah pada kelompok tanpa riwayat, sedangkan LightGBM dan XGBoost standar "
    "paling kuat. Selisihnya kecil, dan rata-rata *rank* menutupi kelemahan setiap model."
))

cells.append(sub("Korelasi Antarmodel"))
cells.append(md(
    "Korelasi *rank* antarmodel menunjukkan seberapa banyak *noise* yang dapat dihilangkan oleh rata-rata. Korelasi "
    "mendekati 1 berarti model membuat kesalahan yang hampir sama."
))
cells.append(orig(76))
cells.append(insight(
    "Korelasi *rank* terendah adalah 0,9804, jadi keenam model sangat mirip. Manfaat *blend* karena itu kecil tetapi "
    "stabil, dan berasal dari perataan *noise* *seed* serta perbedaan cara membangun pohon. Pasangan dengan keluarga "
    "sama (misalnya `lgb` dan `lgb_s`) lebih berkorelasi daripada pasangan lintas keluarga."
))

cells.append(sub("Rank Blend"))
cells.append(md(
    "**Pemilihan kandidat.** Enam model dipilih sejak v22, sebelum blok Markov ada, sehingga bukan hasil seleksi pada "
    "OOF akhir.\n\n"
    "**Blend.** Rata-rata *rank* bobot sama. Pencarian bobot dan *blend* logit tidak mengalahkan rata-rata *rank* pada "
    "eksperimen 275, jadi tidak ada parameter yang dituning.\n\n"
    "**Stacking.** Tidak dipakai. *Stacking* tahap dua di atas OOF memberi sekitar +0,0007 per *fold* bahkan untuk blok "
    "kontrol yang tidak berguna, sehingga gain-nya merupakan artefak."
))
cells.append(orig(79))
cells.append(md(
    "Kurva KS di bawah menampilkan distribusi kumulatif skor *blend* untuk kedua kelas. Jarak vertikal terbesar antara "
    "kedua kurva adalah nilai KS, dan posisinya menunjukkan titik potong yang menentukan skor."
))
cells.append(code(
    "_grid = np.unique(oof_rank)\n"
    "_F1 = np.searchsorted(np.sort(oof_rank[y == 1]), _grid, side='right') / (y == 1).sum()\n"
    "_F0 = np.searchsorted(np.sort(oof_rank[y == 0]), _grid, side='right') / (y == 0).sum()\n"
    "_i = int(np.argmax(_F0 - _F1))\n"
    "_cut = _grid[_i]\n"
    "\n"
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))\n"
    "\n"
    "ax1.plot(_grid, _F0, color='#2a78d6', lw=2, label='Lancar (0)')\n"
    "ax1.plot(_grid, _F1, color='#eb6834', lw=2, label='Gagal bayar (1)')\n"
    "ax1.vlines(_cut, _F1[_i], _F0[_i], color='#52514e', ls='--', lw=1.5)\n"
    "ax1.annotate(f'KS = {_F0[_i] - _F1[_i]:.4f}\\nskor = {_cut:.3f}', (_cut, (_F0[_i] + _F1[_i]) / 2),\n"
    "             xytext=(10, 0), textcoords='offset points', va='center')\n"
    "ax1.set_xlabel('Skor rank blend OOF')\n"
    "ax1.set_ylabel('CDF empiris')\n"
    "ax1.set_title('Kurva KS (OOF Blend)')\n"
    "ax1.legend(loc='lower right')\n"
    "ax1.spines[['top', 'right']].set_visible(False)\n"
    "\n"
    "ax2.hist(oof_rank[y == 0], bins=50, density=True, alpha=0.6, color='#2a78d6', label='Lancar (0)')\n"
    "ax2.hist(oof_rank[y == 1], bins=50, density=True, alpha=0.6, color='#eb6834', label='Gagal bayar (1)')\n"
    "ax2.axvline(_cut, color='#52514e', ls='--', lw=1.5, label='Titik potong KS')\n"
    "ax2.set_xlabel('Skor rank blend OOF')\n"
    "ax2.set_ylabel('Densitas')\n"
    "ax2.set_title('Sebaran Skor per Kelas')\n"
    "ax2.legend()\n"
    "ax2.spines[['top', 'right']].set_visible(False)\n"
    "\n"
    "fig.suptitle('KS Curve', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "print(f'KS pada titik potong     : {_F0[_i] - _F1[_i]:.5f} (ks_2samp {ks(y, oof_rank):.5f})')\n"
    "print(f'Porsi di atas titik potong: {(oof_rank > _cut).mean():.4f}')\n"
    "print(f'Recall gagal bayar        : {1 - _F1[_i]:.4f}')\n"
    "print(f'Porsi lancar tertolak     : {1 - _F0[_i]:.4f}')"
))
cells.append("__INSIGHT_KS__")

# ---------------------------------------------------------------------------
# 9. Riwayat Eksperimen dan Pemilihan Model
# ---------------------------------------------------------------------------
cells.append(section("Riwayat Eksperimen dan Pemilihan Model", "9"))
cells.append(md(
    "Bagian ini merangkum 39 submisi dan eksperimen lokal yang membentuk resep final. Fakta, hipotesis, dan keputusan "
    "dipisahkan, dan hasil negatif tetap dicatat agar alasan setiap keputusan dapat ditelusuri. Angka OOF hanya "
    "dicantumkan bila dihitung dengan skema validasi yang sebanding."
))

cells.append(md("## Baseline"))
cells.append(md(
    "Skor konstan memberi KS 0. Tabel berikut menunjukkan perjalanan dari model awal ke basis enam GBDT.\n\n"
    "| Baseline | *Public* |\n"
    "|---|---:|\n"
    "| LightGBM awal | 0,50227 |\n"
    "| XGBoost awal | 0,50053 |\n"
    "| CatBoost awal | 0,50662 |\n"
    "| Regresi logistik *rank-gauss* (v39c) | 0,55835 |\n"
    "| Enam GBDT bersih (v39a) | 0,56337 |\n\n"
    "Regresi logistik dengan fitur yang sama sudah mencapai 0,55835. Artinya sebagian besar sinyal berasal dari fitur, "
    "bukan dari kapasitas model. Lompatan dari 0,50 ke 0,56 datang dari fitur perilaku dan level harga."
))

cells.append(sub("Dekomposisi Masalah"))
cells.append(md(
    "**Motivasi.** 30.000 aplikasi tidak punya riwayat pinjaman, sehingga tidak punya blok *repayment*, rekening, dan "
    "aplikasi lama. KS kelompok ini juga lebih rendah (bagian 8).\n\n"
    "**Hipotesis.** Model terpisah per `has_history` lebih tepat daripada satu model global.\n\n"
    "**Cara validasi.** *Fold* identik, lalu bandingkan OOF dan *public*.\n\n"
    "**Bukti.** OOF naik (`eight_seg` 0,52706, `xgb_seg16` 0,52783), tetapi *public* turun (0,56533 dan 0,56225, "
    "dibanding v72 0,56822).\n\n"
    "**Keputusan.** **REJECT.** Model global dipertahankan."
))

cells.append(sub("Famili Eksperimen"))
cells.append(md(
    "| Famili | Tujuan | Kandidat | *Public* terbaik | Keputusan |\n"
    "|---|---|---|---:|---|\n"
    "| A *Global structured* | Sinyal tabular utama | LightGBM, XGBoost, CatBoost | 0,56337 | KEEP |\n"
    "| B *Repayment dynamics* | Dinamika telat | Rantai Markov di atas famili A | 0,56822 | KEEP |\n"
    "| C *Segment* | Pisahkan customer tanpa riwayat | Pohon per `has_history` | 0,56533 | REJECT |\n"
    "| D *Smooth* | Bias induktif berbeda | MLP, RealMLP, FM, *varying coefficient* | 0,56374 | REJECT |\n"
    "| E *Foundation model* | Representasi *pretrained* | Kumo, Causilo, TabPFN | 0,56470 | REJECT |"
))

cells.append(sub("Eksperimen E1: Apakah Rantai Markov Menambah Informasi?"))
cells.append(md(
    "**Pertanyaan.** Apakah peluang mencapai 90+ per customer menaikkan skor di atas enam GBDT?\n\n"
    "**Hipotesis.** *Hazard* yang melompat sesudah telat pertama (bagian 5) tidak tertangkap penuh oleh tarif telat.\n\n"
    "```\n"
    "Baseline : 6 GBDT pada 232 kolom\n"
    "Perubahan: + 10 kolom Markov (f251)\n"
    "Validasi : StratifiedKFold 5 fold, inner 1/7, refit x1,15, seed 42, 7, 2026\n"
    "```\n\n"
    "| Metrik | Baseline | Eksperimen | Delta |\n"
    "|---|---:|---:|---:|\n"
    "| OOF KS | 0,52448 | 0,52580 | +0,00132 |\n"
    "| *Public* | 0,56337 | 0,56822 | +0,00485 |\n\n"
    "**Keputusan.** **KEEP.** Ini satu-satunya perubahan terakhir yang naik di OOF dan *public* sekaligus."
))

cells.append(sub("Eksperimen E2: Apakah Pohon per Segmen dan Model Halus Lebih Baik?"))
cells.append(md(
    "**Pertanyaan.** Apakah bias induktif yang berbeda menaikkan skor?\n\n"
    "```\n"
    "Baseline : v72\n"
    "Perubahan: + pohon per has_history\n"
    "           + varying coefficient, FM, MLP, RealMLP (N7)\n"
    "```\n\n"
    "| Metrik | v72 | N7 | Delta |\n"
    "|---|---:|---:|---:|\n"
    "| OOF KS | 0,52580 | 0,53047 | +0,00467 |\n"
    "| *Public* | 0,56822 | 0,55882 | -0,00940 |\n\n"
    "**Keputusan.** **REJECT.** Selisih *public* sekitar 3,4 simpangan baku dari simulasi 9.000 baris berpasangan. "
    "Gain OOF tidak berpindah ke *public*."
))

cells.append(sub("Registri Eksperimen"))
cells.append(md(
    "| ID | Famili | Ide utama | OOF | *Public* | Status |\n"
    "|---|---|---|---:|---:|---|\n"
    "| v39a | *global* | Enam GBDT bersih | 0,52448 | 0,56337 | KEEP sebagai basis |\n"
    "| v31 | *foundation* | GBDT + Causilo + Kumo | 0,52613 | 0,56405 | REJECT |\n"
    "| v70 | *foundation* | Kumo + *spline* + *offset* bulan | beda skema | 0,56470 | REJECT |\n"
    "| v71 | *dynamics* | Enam GBDT + Markov, 2 *seed* | tidak dicatat | 0,56649 | digantikan v72 |\n"
    "| v71c | *postprocessing* | v71 + *offset* logit per bulan | tidak dicatat | 0,56442 | REJECT |\n"
    "| **v72** | ***dynamics*** | **Enam GBDT + Markov, 3 *seed*** | **0,52580** | **0,56822** | **FINAL** |\n"
    "| v73 | *dynamics* | v72 dengan 8 *seed* | 0,52577 | 0,56719 | REJECT |\n"
    "| v74 `eight_seg` | *segment* | Delapan pohon + segmen | 0,52706 | 0,56533 | REJECT |\n"
    "| v74 `xgb_seg16` | *segment* | XGBoost per segmen, 16 *seed* | 0,52783 | 0,56225 | REJECT |\n"
    "| v75 C | *smooth* | Pohon segmen + MLP | 0,52861 | 0,56374 | REJECT |\n"
    "| v76 N7 | *smooth* | Pohon segmen + VC + FM + NN | 0,53047 | 0,55882 | REJECT |"
))

cells.append(sub("Postprocessing"))
cells.append(md(
    "Tidak ada *postprocessing* pada berkas final. Satu-satunya percobaan, *offset* logit per bulan keputusan, "
    "menurunkan *public*:\n\n"
    "```\n"
    "Sebelum : v71 public 0,56649\n"
    "Sesudah : v71c, offset logit per bulan, public 0,56442\n"
    "Delta   : -0,00207\n"
    "Putusan : REJECT\n"
    "```\n\n"
    "KS hanya memakai urutan, jadi kalibrasi global tidak mengubah skor. Koreksi per kelompok hanya berguna bila "
    "kelompok itu salah diurutkan relatif terhadap kelompok lain, dan data tidak menunjukkan pola tersebut."
))

cells.append(sub("Hasil Negatif"))
cells.append(md(
    "| Eksperimen | Harapan | Hasil | Alasan ditolak |\n"
    "|---|---|---|---|\n"
    "| Pohon per segmen | Customer tanpa riwayat lebih tepat | OOF +0,002, *public* -0,003 sampai -0,006 | Tidak berpindah ke *public* |\n"
    "| Pohon segmen + VC + FM + NN (N7) | Bias induktif berbeda | OOF +0,0047, *public* -0,0094 | Tidak berpindah ke *public* |\n"
    "| *Foundation model* tunggal | Representasi lebih kaya | *Public* 0,5533 sampai 0,5647 | Di bawah pohon |\n"
    "| *Stacking* tahap dua | Koreksi residual | Gain sama pada blok kontrol | Artefak *stacking* |\n"
    "| *Down-weighting* label mencurigakan | Label lebih bersih | -0,002 sampai -0,0035 | Membuang sinyal |\n"
    "| *Pseudo-label* | Data tambahan | Sekitar 0 pada 70 ribu label | Tidak diperlukan |\n"
    "| *Tuning* LightGBM dan XGBoost | Konfigurasi lebih baik | Tidak ada yang melampaui v22 | Sudah jenuh |\n"
    "| *Offset* logit per bulan | Koreksi waktu | *Public* -0,002 | *Noise* |\n"
    "| 150 ide dari literatur | Sinyal baru | Satu famili lolos uji ganda, gain sekitar +0,001 | Gagal kontrol acak |\n"
    "| 14 famili lintas tabel, 300 kolom | Sinyal residual | Tidak ada yang menang 4 dari 5 *fold* | *Null* |\n"
    "| Filter *state-space*, peluruhan waktu | Estimasi lebih bersih | Gain sama dengan kontrol acak | Gagal kontrol |"
))

cells.append(sub("Audit Hasil Mencurigakan"))
cells.append(md(
    "*Public* v72 (0,56822) berada sekitar 0,042 di atas OOF (0,52580). Selisih sebesar ini harus dicurigai sebelum "
    "dipercaya.\n\n"
    "- [x] *Preprocessing* tidak melihat *fold* validasi dan tidak ada transformasi berbasis label\n"
    "- [x] *Early stopping* tidak memakai *fold* validasi\n"
    "- [x] Tidak ada seleksi fitur berbasis label\n"
    "- [x] Tidak ada *target encoding*\n"
    "- [x] Tidak ada duplikat atau grup yang terbelah\n"
    "- [x] ID tidak dipakai sebagai fitur\n"
    "- [x] Tidak ada *threshold* yang dipilih\n"
    "- [x] Bobot *ensemble* tidak dioptimasi\n"
    "- [x] Model tidak melihat label *test*"
))
cells.append(insight(
    "**Mengapa skor ini mungkin nyata.** Selisih *public* terhadap OOF muncul pada semua famili model, jadi bukan ciri "
    "khusus v72. *Pipeline* tidak memakai label *test* di tahap mana pun.",
    "**Mengapa skor ini mungkin optimis.** *Public* hanya 9.000 baris dengan simpangan baku sekitar 0,011. Bias maksimum "
    "KS pada sampel kecil menyumbang sekitar +0,003, dan sisanya belum terjelaskan. *Private* bisa lebih dekat ke OOF.",
    "**Eksperimen verifikasi.** *Paired bootstrap* 9.000 baris dari OOF menunjukkan file yang sangat dekat seperti v72 "
    "dan v73 bergeser sekitar 0,001, sehingga urutan keduanya di *public* sebagian adalah undian."
))

cells.append(sub("Pemilihan Model Final"))
cells.append(md(
    "Rubrik memberi bobot 45% ke *public*, dan *public* berkas terpilih sudah diketahui. Karena itu setiap kandidat "
    "dinilai dengan 0,45 x selisih *public* + 0,45 x taksiran selisih *private*. Taksiran *private* menggabungkan OOF yang "
    "disusutkan 0,7 dengan selisih *public*, dibobot menurut *noise* masing-masing.\n\n"
    "| Kandidat | *Public* | OOF | Skor rubrik vs v72 | Stabilitas | Risiko kebocoran | Kompleksitas | Final |\n"
    "|---|---:|---:|---:|---|---|---|---|\n"
    "| **v72** | **0,56822** | 0,52580 | 0 | tinggi | rendah | rendah | **YA** |\n"
    "| v70 goal054 | 0,56470 | beda skema | -0,0007 | sedang | rendah | tinggi | tidak |\n"
    "| v73 | 0,56719 | 0,52577 | -0,0009 | tinggi | rendah | rendah | tidak |\n"
    "| v74 `eight_seg` | 0,56533 | 0,52706 | -0,0013 | sedang | rendah | sedang | tidak |\n"
    "| v75 C | 0,56374 | 0,52861 | -0,0016 | sedang | rendah | sedang | tidak |\n"
    "| v76 N7 | 0,55882 | 0,53047 | -0,0035 | rendah | rendah | tinggi | tidak |\n\n"
    "v72 tetap teratas pada sembilan kombinasi faktor penyusutan dan ketidakpastian *prior*. **Keputusan: v72.**"
))

# ---------------------------------------------------------------------------
# 10. Inferensi dan Submisi
# ---------------------------------------------------------------------------
cells.append(section("Inferensi dan Submisi", "10"))
cells.append(md(
    "Sel-sel berikut memuat ulang 90 *booster* dari disk, memprediksi *test*, memverifikasi hasilnya, lalu menulis "
    "berkas submisi. Bagian ini berjalan sama pada mode `end_to_end` maupun `inference`."
))

cells.append(md("## Kontrak Artefak"))
cells.append(md(
    "Artefak final mengikuti kontrak seragam, sehingga mode `inference` tidak perlu melatih ulang. Pada mode "
    "`end_to_end`, struktur yang sama ditulis ke `v72_final/run/`.\n\n"
    "```\n"
    "v72_bundle/\n"
    "|-- manifest.json          SHA256 semua berkas, versi, dan metadata\n"
    "|-- models/                90 booster: <model>_s<seed>_f<fold>.{txt,ubj,cbm}\n"
    "|-- kernel/                oof_, test_, log_ per model dari kernel Kaggle v72\n"
    "|-- refit/                 oof_, test_, log_ per model dari run end_to_end\n"
    "|-- submission_final.csv   output run end_to_end\n"
    "`-- submission_v72.csv     output kernel v72, sebagai pembanding\n"
    "```"
))

cells.append(sub("Muat *Booster* dan Inferensi"))
cells.append(md(
    "Setiap *booster* dimuat dari `CFG.MODEL_DIR` dan memprediksi baris *test* di CPU. XGBoost dipaksa ke `device='cpu'` "
    "agar inferensi tidak bergantung pada GPU. Prediksi per model adalah rata-rata probabilitas 15 *booster*, sama "
    "seperti saat pelatihan."
))
cells.append(orig(90))

cells.append(sub("Verifikasi *Booster*"))
cells.append(md(
    "Pemeriksaan pertama memastikan *booster* termuat dengan benar: prediksi hasil muat ulang harus sama dengan prediksi "
    "yang disimpan saat pelatihan. Pemeriksaan kedua membandingkan *blend* hasil inferensi dengan *blend* dari prediksi "
    "yang disimpan. Pada mode `end_to_end`, `PRED_DIR` dan `REFIT_DIR` menunjuk ke folder `run/` yang sama."
))
cells.append(orig(92))
cells.append(insight(
    "`selisih_maks_saat_simpan` sekitar 1e-16 untuk LightGBM dan CatBoost, artinya *booster* termuat identik. XGBoost "
    "berselisih sekitar 5e-7 karena dilatih dan memprediksi di GPU, sedangkan inferensi ulang berjalan di CPU. Selisih "
    "ini tidak mengubah urutan secara berarti: korelasi Spearman setiap model dan *blend* sama dengan 1,0.",
    "OOF *blend* dari kedua sumber sama persis (KS 0,52558), sehingga *output* inferensi *booster* dipakai sebagai "
    "`submission.csv`."
))

cells.append(sub("Laporan Validasi Final"))
cells.append(md(
    "Laporan ini dihitung dari OOF *blend* pada 70.000 baris: KS *pooled*, KS per *fold* untuk 15 *fold*, KS per kuartal "
    "bulan keputusan, dan KS per kelompok riwayat."
))
cells.append(orig(96))
cells.append(code(
    "fig, ax = plt.subplots(figsize=(14, 4.5))\n"
    "_seed_colors = ['#2a78d6', '#eb6834', '#1baf7a']\n"
    "for j, sd in enumerate(CFG.SEEDS):\n"
    "    _x = np.arange(j * CFG.N_FOLDS, (j + 1) * CFG.N_FOLDS)\n"
    "    _bars = ax.bar(_x, per_fold[j * CFG.N_FOLDS:(j + 1) * CFG.N_FOLDS], color=_seed_colors[j], width=0.7,\n"
    "                   label=f'split seed {sd}')\n"
    "    ax.bar_label(_bars, fmt='%.3f', padding=3, fontsize=8)\n"
    "ax.axhline(np.mean(per_fold), color='#52514e', ls='--', lw=1.5, label=f'rata-rata fold {np.mean(per_fold):.4f}')\n"
    "ax.axhline(ks(y, oof_rank), color='#0b0b0b', ls=':', lw=1.5, label=f'pooled OOF {ks(y, oof_rank):.4f}')\n"
    "ax.set_xticks(range(len(per_fold)), [f'f{k}' for _ in CFG.SEEDS for k in range(CFG.N_FOLDS)])\n"
    "ax.set_ylim(0.49, 0.56)\n"
    "ax.set_ylabel('KS blend')\n"
    "ax.set_title('KS Blend per Fold')\n"
    "ax.legend(loc='upper left', ncol=5, fontsize=9)\n"
    "ax.spines[['top', 'right']].set_visible(False)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(insight(
    "KS *pooled* *blend* adalah 0,52558 dengan AUC 0,84611. Per *fold*, rata-ratanya 0,52800 dengan simpangan baku "
    "0,00827, dari 0,51642 sampai 0,54648. Rata-rata per *fold* sedikit di atas KS *pooled* karena KS adalah statistik "
    "maksimum: pada sampel lebih kecil, nilainya cenderung bias ke atas.",
    "Kuartal pertama (Q1) adalah kuartal terlemah dengan KS 0,514, sedangkan Q2 tertinggi dengan 0,542. Selisih "
    "kelompok riwayat tetap sekitar 0,04 (0,497 tanpa riwayat lawan 0,537 berriwayat)."
))
cells.append(md(
    "### Keterbatasan yang Diketahui\n\n"
    "- XGBoost dan CatBoost dilatih di GPU, sehingga pelatihan ulang di perangkat lain dapat memberi prediksi yang "
    "berbeda tipis.\n"
    "- OOF mengandung efek seleksi dari riwayat eksperimen, walaupun resep enam model sudah dikunci lebih dulu.\n"
    "- Selisih *public* di bawah 0,002 tidak dapat dibedakan pada 9.000 baris."
))

cells.append(sub("Pembuatan Berkas Submisi"))
cells.append(md(
    "`submission.csv` adalah *output* langsung dari *booster* yang dimuat ulang, dengan rumus kernel v72: *rank* tiap "
    "model dibagi 30.000, lalu dirata-ratakan. `submission_kernel_v72.csv` memakai rumus yang sama pada prediksi *test* "
    "yang disimpan saat pelatihan. Berkas dibaca ulang dan divalidasi sebelum dianggap selesai."
))
cells.append(orig(100))
cells.append(md(
    "Pemeriksaan:\n\n"
    "- [x] Jumlah baris sama dengan *test* (30.000)\n"
    "- [x] Urutan ID sama dengan `sample_submission.csv`\n"
    "- [x] Tidak ada NaN\n"
    "- [x] Semua nilai di rentang [0, 1]\n"
    "- [x] Berkas terbaca ulang\n"
    "- [x] Dibandingkan dengan *blend* dari prediksi yang disimpan"
))
cells.append(code(
    "_idx = np.random.default_rng(42).choice(len(final_test_pred), 5000, replace=False)\n"
    "\n"
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))\n"
    "\n"
    "ax1.hist(final_test_pred, bins=50, color='#2a78d6', edgecolor='white')\n"
    "ax1.set_xlabel('default_probability')\n"
    "ax1.set_ylabel('Jumlah aplikasi')\n"
    "ax1.set_title('Distribusi Skor submission.csv')\n"
    "ax1.spines[['top', 'right']].set_visible(False)\n"
    "\n"
    "ax2.scatter(kernel_rank[_idx], final_test_pred[_idx], s=4, alpha=0.4, color='#2a78d6')\n"
    "ax2.plot([0, 1], [0, 1], ls='--', color='#52514e', lw=1)\n"
    "ax2.set_xlabel('Blend dari prediksi tersimpan')\n"
    "ax2.set_ylabel('Blend dari booster dimuat ulang')\n"
    "ax2.set_title('Konsistensi Inferensi (5.000 sampel)')\n"
    "ax2.spines[['top', 'right']].set_visible(False)\n"
    "\n"
    "fig.suptitle('Submission Check', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "print(f'Rentang skor     : {final_test_pred.min():.5f} - {final_test_pred.max():.5f}')\n"
    "print(f'Selisih maksimum : {np.abs(final_test_pred - kernel_rank).max():.2e}')"
))
cells.append(insight(
    "Skor submisi tersebar hampir seragam di [0, 1] karena merupakan rata-rata *rank*. Bentuk ini wajar untuk KS, yang "
    "hanya membaca urutan. Titik-titik pada diagram sebar berada tepat di garis diagonal: selisih terbesar sekitar 8e-6 "
    "berasal dari perbedaan GPU dan CPU pada XGBoost dan tidak mengubah urutan secara berarti."
))

# ---------------------------------------------------------------------------
# 11. Pasca-Analisis
# ---------------------------------------------------------------------------
cells.append(section("Pasca-Analisis", "11"))

cells.append(md("## Kontribusi Fitur"))
cells.append(md(
    "*Gain importance* LightGBM dirata-ratakan dari 15 *booster* `lgb`, lalu dikelompokkan per famili fitur. Analisis "
    "ini memperlihatkan blok informasi mana yang paling banyak dipakai model untuk memisahkan kelas. *Gain* mengukur "
    "kontribusi dalam pelatihan, bukan dampak kausal, sehingga fitur yang berkorelasi dapat saling berbagi *gain*."
))
cells.append(code(
    "_FAMILY = [('mk_', 'Markov repayment'), ('pp_', 'Level harga'), ('rp_', 'Repayment'), ('rq_', 'Repayment'),\n"
    "           ('sv_', 'Repayment'), ('ac_', 'Rekening'), ('cf_', 'Rekening'), ('ec_', 'E-commerce'),\n"
    "           ('ec2_', 'E-commerce'), ('bc_', 'Biro'), ('bh_', 'Biro'), ('bp_', 'Biro'), ('bz_', 'Biro'),\n"
    "           ('pv_', 'Aplikasi lama'), ('ob_', 'Aplikasi lama')]\n"
    "_FAM_COLORS = {'Markov repayment': '#2a78d6', 'Level harga': '#eb6834', 'Repayment': '#1baf7a',\n"
    "               'Rekening': '#eda100', 'E-commerce': '#e87ba4', 'Biro': '#008300',\n"
    "               'Aplikasi lama': '#6250d6', 'Aplikasi dan rasio': '#e34948'}\n"
    "\n"
    "_imp = {}\n"
    "for sd in CFG.SEEDS:\n"
    "    for k in range(CFG.N_FOLDS):\n"
    "        _b = lgb.Booster(model_file=str(CFG.MODEL_DIR / f'lgb_s{sd}_f{k}.txt'))\n"
    "        for _f, _g in zip(_b.feature_name(), _b.feature_importance('gain')):\n"
    "            _imp[_f] = _imp.get(_f, 0.0) + _g / (len(CFG.SEEDS) * CFG.N_FOLDS)\n"
    "_imp = pd.Series(_imp).sort_values(ascending=False)\n"
    "_share = _imp / _imp.sum()\n"
    "_fam = pd.Series({c: next((n for p, n in _FAMILY if c.startswith(p)), 'Aplikasi dan rasio') for c in _imp.index})\n"
    "_fam_share = _share.groupby(_fam).sum().sort_values()\n"
    "\n"
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [3, 2]})\n"
    "\n"
    "_top = _share.head(20)[::-1]\n"
    "_pos = np.arange(len(_top))\n"
    "for n, c in _FAM_COLORS.items():\n"
    "    _m = (_fam[_top.index] == n).to_numpy()\n"
    "    if _m.any():\n"
    "        ax1.barh(_pos[_m], _top.values[_m], color=c, label=n)\n"
    "ax1.set_yticks(_pos, _top.index)\n"
    "ax1.set_xlabel('Porsi gain')\n"
    "ax1.set_title('20 Fitur Teratas (LightGBM, 15 booster)')\n"
    "ax1.spines[['top', 'right']].set_visible(False)\n"
    "ax1.legend(loc='lower right', fontsize=9)\n"
    "\n"
    "_bars = ax2.barh(_fam_share.index, _fam_share.values, color=[_FAM_COLORS[n] for n in _fam_share.index])\n"
    "ax2.bar_label(_bars, fmt='%.3f', padding=3)\n"
    "ax2.set_xlabel('Porsi gain')\n"
    "ax2.set_title('Porsi Gain per Famili')\n"
    "ax2.spines[['top', 'right']].set_visible(False)\n"
    "\n"
    "fig.suptitle('Feature Contribution', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "print(pd.DataFrame({'famili': _fam[_imp.index[:10]], 'porsi_gain': _share.head(10).round(4)}))"
))
cells.append("__INSIGHT_IMP__")

cells.append(md(
    "### Kelebihan Pendekatan\n\n"
    "Resep final sederhana dan dapat diaudit: enam GBDT global, satu matriks fitur, dan rata-rata *rank* tanpa parameter "
    "yang dituning. Setiap blok fitur punya alasan yang dapat diuji pada data, dan blok Markov memodelkan dinamika "
    "telat bayar secara eksplisit tanpa melihat label. Karena tidak ada bobot *blend*, *stacking*, maupun "
    "*postprocessing*, risiko *overfitting* ke OOF atau ke *public* kecil. Seluruh *pipeline* juga selesai di bawah 30 "
    "menit di Kaggle dan hasilnya dapat diverifikasi ulang dari 90 *booster* yang disimpan.\n\n"
    "---\n\n"
    "### Keterbatasan yang Diketahui\n\n"
    "Customer tanpa riwayat tetap menjadi kelompok terlemah (KS sekitar 0,50), dan model per segmen yang menargetkan "
    "kelompok ini tidak berhasil di *public*. Selisih *public* 0,04 di atas OOF belum terjelaskan sepenuhnya, sehingga "
    "*private* bisa jauh lebih rendah dari *public*, dan hal ini berlaku untuk semua kandidat. Keenam model juga sangat "
    "berkorelasi (minimum 0,98), jadi *ensemble* hampir tidak menambah keberagaman.\n\n"
    "---\n\n"
    "### Arah Pengembangan Berikutnya\n\n"
    "Langkah yang paling informatif adalah *lockbox* 20%: sisihkan sebagian *train*, ulangi pemilihan antara enam model "
    "dan pohon segmen + MLP di 80% sisanya, lalu ukur di *lockbox*. Uji ini menjawab apakah gain OOF struktur model "
    "nyata atau ilusi seleksi. Selain itu, blok informasi baru sebaiknya hanya masuk *ensemble* bila lolos kontrol acak, "
    "misalnya sinyal lonjakan belanja menjelang keputusan dan ambruknya arus kas rekening di empat bulan terakhir."
))

# ---------------------------------------------------------------------------
# 12. Daftar Pustaka
# ---------------------------------------------------------------------------
cells.append(section("Daftar Pustaka", "12"))
cells.append(md(
    "Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD "
    "International Conference on Knowledge Discovery and Data Mining, 785-794. arXiv:1603.02754.\n\n"
    "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). *LightGBM: A Highly "
    "Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems 30 (NIPS 2017).\n\n"
    "Massey, F. J. (1951). *The Kolmogorov-Smirnov Test for Goodness of Fit*. Journal of the American Statistical "
    "Association, 46(253), 68-78.\n\n"
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., "
    "Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). "
    "*Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.\n\n"
    "Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). *CatBoost: Unbiased Boosting with "
    "Categorical Features*. Advances in Neural Information Processing Systems 31 (NeurIPS 2018). arXiv:1706.09516.\n\n"
    "Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). *Credit Scoring and Its Applications*. Society for Industrial "
    "and Applied Mathematics (SIAM)."
))

cells.append(md(f"{HR}\n\n# Terima Kasih! <a name=\"\"></a>\n\n{HR}"))
cells.append(md(
    "**Terima kasih khusus kepada:**\n\n"
    "Panitia **DSC LOGIKA UI 2026** yang telah menyediakan *dataset* dan platform kompetisi yang menantang."
))

# ---------------------------------------------------------------------------
# Execute the new chart cells against the run artefacts
# ---------------------------------------------------------------------------
SHIM = f"""
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ks_2samp, rankdata
from sklearn.model_selection import StratifiedKFold

RUN = Path({str(RESULTS / 'run')!r})


class CFG:
    SEEDS = [42, 7, 2026]
    N_FOLDS = 5
    SIX = ['lgb', 'lgb_s', 'xgb', 'xgb_s', 'cat', 'cat_s']
    MODEL_DIR = RUN / 'models'


def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def outer(y, seed):
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


oof_m = {{m: pl.read_parquet(RUN / f'oof_{{m}}.parquet') for m in CFG.SIX}}
y = oof_m['lgb']['target'].to_numpy().astype(int)
oof_m = {{m: a[m].to_numpy() for m, a in oof_m.items()}}
oof_rank = np.mean([rankdata(oof_m[m]) for m in CFG.SIX], 0) / len(y)
logs = pl.concat([pl.read_csv(RUN / f'log_{{m}}.csv') for m in CFG.SIX])
per_fold = [ks(y[va], oof_rank[va]) for sd in CFG.SEEDS for _, va in outer(y, sd)]
final_test_pred = pl.read_csv(RUN.parent / 'submission.csv')['default_probability'].to_numpy()
kernel_rank = pl.read_csv(RUN.parent / 'submission_kernel_v72.csv')['default_probability'].to_numpy()
assert round(ks(y, oof_rank), 5) == 0.52558 and round(float(np.mean(per_fold)), 5) == 0.528
"""


def run_new_cells():
    ns = {}
    exec(SHIM, ns)
    figs = []
    ns["plt"].show = lambda *a, **k: figs.append(plt.gcf())
    for cell in cells:
        if not isinstance(cell, dict) or not cell["metadata"].pop("new_chart", False):
            continue
        buf = io.StringIO()
        figs.clear()
        with contextlib.redirect_stdout(buf):
            exec(cell["source"], ns)
        outputs = []
        for fig in figs:
            png = io.BytesIO()
            fig.savefig(png, format="png", bbox_inches="tight", dpi=72)
            w, h = fig.get_size_inches() * fig.dpi
            outputs.append({
                "data": {"image/png": base64.b64encode(png.getvalue()).decode(),
                         "text/plain": [f"<Figure size {w:g}x{h:g} with {len(fig.axes)} Axes>"]},
                "metadata": {}, "output_type": "display_data",
            })
            plt.close(fig)
        if buf.getvalue():
            outputs.append({"name": "stdout", "output_type": "stream", "text": buf.getvalue().splitlines(True)})
        cell["outputs"] = outputs
    return ns


ns = run_new_cells()

# Insights that quote numbers produced by the new chart cells
np = ns["np"]
_y, _r = ns["y"], ns["oof_rank"]
_g = np.unique(_r)
_F1 = np.searchsorted(np.sort(_r[_y == 1]), _g, side="right") / (_y == 1).sum()
_F0 = np.searchsorted(np.sort(_r[_y == 0]), _g, side="right") / (_y == 0).sum()
_i = int(np.argmax(_F0 - _F1))


def idn(x, d):
    return f"{x:.{d}f}".replace(".", ",")


KS_TEXT = insight(
    f"Rata-rata *rank* memberi KS *pooled* {idn(ns['ks'](_y, _r), 5)} dan AUC 0,84611, sedikit di atas model tunggal "
    "terbaik (`lgb`, KS 0,52496 dan AUC 0,84532). Kenaikannya kecil karena keenam model sangat berkorelasi, tetapi "
    "*blend* tidak lebih buruk dari anggota mana pun dan menghilangkan risiko memilih satu model yang kebetulan unggul.",
    f"Kurva KS mencapai jarak terbesar di skor {idn(_g[_i], 3)}: {idn((1 - _F1[_i]) * 100, 1)}% aplikasi gagal bayar "
    f"berada di atas titik potong, sedangkan hanya {idn((1 - _F0[_i]) * 100, 1)}% aplikasi lancar ikut tertolak. Sekitar "
    f"{idn((_r > _g[_i]).mean() * 100, 0)}% aplikasi berada di atas titik ini. Kesalahan yang mahal adalah salah urut "
    "di sekitar titik potong tersebut, bukan di ekor distribusi."
)

_share = ns["_share"]
_fam_share = ns["_fam_share"].sort_values(ascending=False)
_top3 = "`{}`, `{}`, dan `{}`".format(*_share.index[:3])
IMP_TEXT = insight(
    f"Tiga fitur dengan *gain* terbesar adalah {_top3}. Per famili, {_fam_share.index[0]} menyumbang porsi *gain* "
    f"terbesar ({idn(_fam_share.iloc[0] * 100, 1)}%), disusul {_fam_share.index[1]} "
    f"({idn(_fam_share.iloc[1] * 100, 1)}%) dan {_fam_share.index[2]} ({idn(_fam_share.iloc[2] * 100, 1)}%).",
    f"Bila dihitung per kolom, rata-rata *gain* adalah {idn(100 / len(_share), 2)}% per kolom. Blok Markov (10 kolom) "
    f"mendapat {idn(ns['_fam_share']['Markov repayment'] * 10, 2)}% per kolom dan level harga (17 kolom) "
    f"{idn(ns['_fam_share']['Level harga'] * 100 / 17, 2)}% per kolom, jauh di atas rata-rata. Kedua blok yang memberi "
    "kenaikan *public* terbesar dalam riwayat tim memang termasuk blok yang paling padat informasi, sejalan dengan "
    "temuan EDA di bagian 5.",
    "Fitur Markov tidak menggantikan fitur *repayment* biasa: `rp_frac_dpd1` tetap fitur kedua terpenting. Blok "
    "Markov menambah informasi tentang dinamika telat, bukan sekadar meringkas ulang tarif telat."
)

cells = [KS_TEXT if c == "__INSIGHT_KS__" else IMP_TEXT if c == "__INSIGHT_IMP__" else c for c in cells]

# Every original code cell must appear exactly once, in the original order, untouched
orig_idx = [i for i, c in enumerate(src["cells"]) if c["cell_type"] == "code"]
kept = [c for c in cells if c["cell_type"] == "code" and "papermill" in c["metadata"]]
assert kept == [src["cells"][i] for i in orig_idx], "original code cells changed or reordered"

# ---------------------------------------------------------------------------
# Notebook writer
# ---------------------------------------------------------------------------
nb = {"nbformat": src["nbformat"], "nbformat_minor": src["nbformat_minor"], "metadata": src["metadata"], "cells": cells}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {len(cells)} cells -> {OUT}")
