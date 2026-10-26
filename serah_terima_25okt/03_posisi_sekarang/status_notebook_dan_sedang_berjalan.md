# Status notebook dan apa yang sedang berjalan

## Sedang berjalan
- **Tidak ada job yang berjalan** saat serah terima (GPU idle, thermal guard dimatikan, semua monitor dihentikan).
- Eksperimen terakhir yang terputus: `eda/249_nested_teammate_select.py` (fold 0–3 selesai, rata-rata Δ ≈ +0,0002 →
  null praktis; fold 4 tidak selesai karena hang #4; OOF parquet rusak sudah dihapus, log fold ada di
  `results/v66/teammate_select/log_lgb_tsel30.csv`).

## Notebook submisi
- Builder: `notebook/build_final.py` → `notebook/final.ipynb` (14 sel). Resep: **replay v59** (public 0,56296):
  memverifikasi hash 12 prediksi penyusun (audit 201) dan merekonstruksi CSV; mode `fresh`
  (`LOGIKA_PIPELINE_MODE=fresh`) memuat pipeline training lengkap (fitur → 6 GBDT → Causilo 5 seed → Kumo-70k 3 draw)
  dengan source modul di-embed + SHA256.
- Sesi ini: banner ditambahkan — "**DSCxxxx — lagi cape lomba asli**", anggota (akun Kaggle) athillazaidan,
  nicholaswisee, riantamaputra. Replay nbclient: 5/5 sel kode lulus, CSV **byte-identik** dengan submisi v59
  (SHA `d7e17b07…`). Bukti: `artifacts/optimization/continuation_24okt/notebook_verification_24okt.json`.
- `eda/features.py` (modified, belum commit): parameter `reference_customers=None` di `v5_features`; default
  mempertahankan resep historis → tidak mengubah replay/fresh.
- **Sisa:** (1) nomor peserta DSCxxxx; (2) nama file `ParticipantNumber_TeamName_Notebook`; (3) **keputusan resep**:
  notebook sekarang = v59 (0,56296), sedangkan public terbaik = goal054 teman (0,56470) yang pipeline-nya ada di
  `scripts/research_goal054.py` / `hybrid_kumo.py` (env Python 3.11, fitur V5) — kalau submisi goal054 atau hasil
  merge yang dijadikan pegangan, notebook harus diganti/diperluas dan diuji replay; (4) fresh training tidak diulang
  sejak 19 Okt (run lokal penuh 2,8 jam lulus saat itu).

## Keputusan yang menunggu user
1. Nomor peserta (banner & nama file notebook).
2. Izin menjalankan rencana merge (GPU ±1–2 jam, laptop perlu dipantau karena riwayat hang).
3. Resep mana yang direpresentasikan notebook (v59 / goal054 / hasil merge).
4. Kandidat hapus besar opsional (lihat `08_kode_dan_infrastruktur/pembersihan_repo.md`).
