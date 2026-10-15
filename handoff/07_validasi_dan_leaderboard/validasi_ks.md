# Validasi, noise KS, dan leaderboard

## Skema validasi
- **StratifiedKFold 5 fold, per seed** (fold dibuat ulang untuk setiap seed 42/7/2026). Alasannya: split random dengan periode sama (PSI < 0,002, KS train−test maksimum 0,010), dan satu customer hanya punya satu aplikasi, jadi tidak perlu group.
- Angka utama adalah **pooled OOF KS**. Mean fold KS sekitar +0,003 lebih tinggi karena maksimisasi per subset.
- Laporan per model: OOF KS, AUC, fold KS ± sd, simulasi |pub−priv|.
- Stabilitas blend: fold KS, thin-file vs history, KS per bulan (0,50–0,57, ini irisan random, bukan forward-time), dan plot CDF.
- Bandingkan versi dengan **bootstrap berpasangan** pada prediksi tetap, Spearman submission, dan held-out halves. Jangan pakai selisih skor mentah.

## Seberapa besar noise
| Besaran | Nilai |
|---|---|
| sd KS public (simulasi, share 30% / 50%) | 0,012 / 0,009 |
| \|pub − priv\| median / p95 | sekitar 0,009–0,011 / sekitar 0,026–0,029 |
| sd selisih public dua submission yang mirip (Spearman 0,998) | sekitar 0,0026 |
| Resolusi deteksi fitur dari screening OOF | sekitar 0,002 KS |
| Selisih greedy in-sample vs held-out | sekitar 0,001–0,002 |

Aturan praktis: selisih public < ±0,005 antara dua submission yang berkorelasi > 0,998 adalah noise.

## Offset public − OOF
- v1 +0,026, v2 +0,028, v3 +0,031, v5 +0,034, v6 +0,033. **Stabil sekitar +0,03.**
- Analisis v9: bias sub-sampel KS (subset lebih kecil cenderung KS lebih tinggi) cuma menjelaskan sekitar +0,002. Distribusi prediksi test identik dengan OOF, jadi bukan pergeseran data. Kesimpulannya: **subset public tetap yang kebetulan "mudah"**. Itu tidak bisa dimanfaatkan dan bisa saja tidak berlaku untuk private.
- Koreksi Astra: offset satu model **bukan konverter universal**. Jangan mengonversi top-1 public ke "target OOF" secara pasti. Dulu sempat ditulis "top-1 0,5718 ≈ OOF 0,541"; itu tidak defensible.

## Analisis gap ke top-1 (`eda/69`, dengan catatan Astra)
- Simulasi LB sintetis: 20 tim berkualitas setara, masing-masing mengambil best of 10 submission. Tim teratas di public rata-rata **+0,012 sampai +0,028** di atas tim setara, hanya karena keberuntungan dan seleksi.
- Gap kita ke top-1 sekitar 0,013 (0,57493 − 0,56145). Ini **bisa** dijelaskan oleh keberuntungan atau seleksi, **atau** metode yang memang lebih baik. Tanpa prediksi mereka, tidak bisa dibedakan.
- Untuk mencapai OOF sekitar 0,541 kita butuh spread log-odds sekitar 5% lebih lebar, artinya informasi baru. Di bawah asumsi model kita, logit scale 1,05 → KS 0,545 dan 1,10 → 0,562.

## Aturan keputusan submit yang dipakai sepanjang chat
- Submit versi baru hanya kalau held-out-nya melampaui versi terbaik dengan margin berpasangan yang berarti. Kalau tidak, submit hanya sebagai cek kepatuhan atau konsistensi (contoh: v6).
- Jangan pernah memilih submission hanya berdasarkan public. Bobot public dan private sama-sama 45%, jadi yang paling aman adalah submission dengan held-out terbaik **dan** dihasilkan oleh notebook reproducible.
- LB probing dilarang (rule 11).
