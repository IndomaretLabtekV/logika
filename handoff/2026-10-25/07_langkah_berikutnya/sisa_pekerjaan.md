# Pekerjaan yang tersisa (urut prioritas)

## Wajib sebelum 30 Okt 23:59 WIB
0. **Tim Kaggle & nama resmi**: konfirmasi nomor peserta (kemungkinan DSC0152) dan nama tim resmi; bila sumber DSC0152
   masih tercatat sebagai tim Kaggle terpisah, gabungkan di Kaggle (aturan: satu peserta satu tim) dan ganti nama tim
   sesuai pendaftaran.
1. **Notebook submisi**: isi nomor peserta (DSCxxxx) di banner `notebook/build_final.py`; nama file
   `ParticipantNumber_TeamName_Notebook`; putuskan resep yang direpresentasikan (v59 replay sekarang / goal054 /
   hasil merge); uji replay ulang (`run_nb.py` nbclient — lihat `08_kode_dan_infrastruktur/cara_menjalankan.md`);
   bila memungkinkan uji mode fresh. Bukti verifikasi terakhir: `artifacts/optimization/continuation_24okt/notebook_verification_24okt.json`.
2. **Pilihan submisi untuk private** (Kaggle memilih submisi tim): pastikan submisi terbaik (goal054 0,56470 atau hasil
   merge) yang dipilih, dan notebook mereproduksinya.

## Riset (prioritas)
3. **Merge pipeline teman + user** (`rencana_merge.md`) — peluang terbesar.
4. Koreksi offset laten di atas Kumo user.
5. (Opsional, prior rendah) self-distillation cross-fitted dan R-CPL (protokol SSL 23 Okt), Kumo Relational (backbone
   TabICLv2, butuh unduh bobot 120 MB & memori besar).

## Kuota
- Window sampai 25 Okt 07:00: 1 slot tersisa (tidak dipakai saat serah terima). Mulai 25 Okt 07:00: 3 slot/hari s.d.
  30 Okt.

## Kebersihan repo (keputusan user)
6. Kandidat hapus besar opsional (lihat `08_kode_dan_infrastruktur/pembersihan_repo.md`).
7. Commit perubahan sesi ini bila user mau (belum di-commit; jangan push).

## Dokumentasi
8. `docs/integration/indeks_eksperimen.md` dan `handoff/04_riwayat_versi/jurnal_24okt_resep_lokal.md` sudah memuat
   semua eksperimen sesi ini; perbarui saat eksperimen baru selesai.
