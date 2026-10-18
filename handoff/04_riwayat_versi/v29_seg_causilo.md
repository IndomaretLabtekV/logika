# v29 (18 Okt dini hari): Causilo per segmen has_history — null

## Hipotesis (bukan tuning)
`fm_select` global memberi model ICL **44 dari 104 kolom yang 100% NaN untuk thin-file** (fold 0). Model ICL yang
melihat context campuran mungkin terganggu kolom kosong itu. Kalau model ICL difit terpisah per segmen (context dan
seleksi fitur sendiri), thin-file seharusnya membaik.

## Setup
- `valid2.seg_fit(base)` + prefix `seg_` di `run_valid2.py` (`seg_causilo`): untuk tiap segmen has_history ∈ {0, 1},
  fit Causilo hanya di baris train segmen itu dan prediksi baris segmen itu.
- Kernel `riantamaputra/logika-v29-segcausilo` (T4×2, seed 42, 5 fold, ~205 s/fold), push 01:10 WIB → `results/v29m/`.

## Hasil
| | OOF | thin-file | history |
|---|---|---|---|
| Causilo global (v22) | 0,52583 | 0,4963 | 0,5349 |
| seg_causilo | 0,52409 | 0,4953 | 0,5359 |

Di blend trees/causilo/kumo: mengganti Causilo +0,00154 (P 0,83, 3/5), menambah keduanya +0,00162 (P 0,85) → sama dengan
noise (`eda/out/109_seg_causilo.txt`).

## Kesimpulan
Kolom all-NaN tidak merugikan ICL (praktis diabaikan). Memisah context menghapus informasi lintas segmen dan kalibrasi
antar segmen sehingga KS pooled turun. Bersama `eda/108` (bobot blend per segmen −0,0005): **spesialisasi per segmen tidak
berguna** di data ini.
