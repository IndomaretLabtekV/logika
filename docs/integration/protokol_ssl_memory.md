# Amendemen eksekusi SSL setelah laptop hang

21 Oktober, setelah user mengonfirmasi laptop hang dan di-force shutdown pada
run pertama. Tidak ada prediksi teacher penuh, skor search, atau skor outer dari
run itu. Amendemen dibuat sebelum hasil kualitas SSL tersedia.

- Pertahankan source156, data, split, fitur, teacher, e4, semua real context,
  threshold0,02/0,05 dan0,7/0,9, caps2k/10k, group0, serta gerbanggain0,003.
- Turunkan query chunk500→128 dan batas allocatorGPU0,9→0,7. Tujuannya memberi
  ruang VRAM untuk desktop dan memperkecil pekerjaan query per panggilan.
  Ini upaya mengurangi tekanan, bukan diagnosis pasti penyebab hang.
- Jangan mengurangi real context atau jumlah estimator. Jangan mengubah driver,
  service daya, clocks, atau konfigurasi sistem secara diam-diam.
- Gunakan direktori baru, karena config hash berbeda. Run lama disimpan sebagai
  bukti interupsi, bukan null kualitas: results/v49/ssl_asymmetric_f0.
- Smoke baru: results/v49/ssl_asymmetric_smoke_safe,512/128/512,cap64.
  Bandingkan prediksi dengan smokechunk500 pada fit/query yang sama: atol0,002,
  rtol0,005 sesuai toleransi parity154. Verifikasi ID/label/cache/memori lagi.
- Jika smoke sehat, fullrun: results/v49/ssl_asymmetric_safe_f0,
  --chunk128 --gpu-memory-fraction0.7 dengan parameter riset lainnya identik.
  Baseline matched dan SSL memakai chunk128 yang sama. Referensi lama chunk500
  tetap historis; jangan menafsirkan perubahan kecil numerik sebagai gainSSL.
- Catat telemetri GPU/RAM selama run. Bila OOM, jangan memperkecil context asli
  sebagai fallback. Simpan bukti, lalu putuskan konfigurasi baru secara eksplisit.

Hasil VIP yang telah selesai tidak berubah: Δouter−0,006096,P0,147;tidak diteruskan.
Tidak ada submission atau perubahan notebook produksi dari amendemen ini.
