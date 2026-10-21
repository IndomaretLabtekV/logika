# Satu fit pemulihan dengan inhibitor sesi sementara

21Okt21:15WIB. Boot21:09 setelah user mengonfirmasi hang lagi. Resume
sebelumnya terputus di replaycache;GPU342MiB/53°C/~20W, bukan contextfit.
Journal21:06:01 mencatat lockscreen dan errorJPAC powerd. Ini korelasi;
penyebab hang belum terbukti. Tidak mengubah driver, service, power limit,
clock, atau konfigurasi desktop permanen.

Jalankan tepat satu fit belum lengkap seed2026fold1 melalui entry pacing
yang sudah bit-exact. Sumber/recipe/context/e4/chunk/allocator dan selector
tidak diubah. `systemd-inhibit --what=idle:sleep --mode=block` hanya hidup
selama child; tujuannya mengurangi transisi idle/sleep sebagai satu mitigasi
operasional baru, bukan klaim kausal. Sesudah terminal0 tinjau telemetry
dan integritas output sebelum melanjutkan fit lain. Bila hang lagi, jangan
otomatis retry. AuditCPU201 yang sudah aktif ditunggu selesai dahulu.

Gunakan scope16Ghigh/20Gmax/1Gswap,CPUQuota300%,affinity0–11,NIaktual10,
threads4. Log dan telemetry baru terpisah;hasil11fit tidak ditimpa.
