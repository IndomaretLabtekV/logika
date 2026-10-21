# Pemulihan validasi setelah hang kedua21Oktober

Boot11:18 berhenti mencatat18:35:40, bootbaru18:40:11. User mengonfirmasi
hang lalu force-shutdown. Wrapper/child lama tidak ada;scope inactive. Sebelas
fit lengkap,seed2026fold1 terputus sebelum primary selesai. Log/telemetri lama
dipertahankan;statusrunning padaJSON adalah catatan tertinggal, bukan proses hidup.

Sampel terakhir:GPU5495MiB,77°C,RAMavailable24593MiB,childRSS4293MiB.
Journalmemuatnvidia-powerdSysPwrLimitGetInfo/JPACerrors;belum ada bukti kausal
atau kernelOOM/Xid. AC sekarangonline,bateraiFull,driver615.78.08. Tidak
menyimpulkan penyebab dari satu korelasi. Model yang sama telah menyelesaikan
11fit dengan konfigurasi ini;parameter matematika/model tidak diubah.

Mitigasi operasional dan instrumentasi:

- Runtime hook pada instance Kumo:sinkronisasi lalu jeda0,25detik sesudah
  setiap outer cell/column/row/ICLblock selama fit context;0,1detik sesudah
  setiap cached querychunk. Tidak menyentuh tensor,bobot,RNG,batchsize,
  context,query,e4,chunk128 atauallocator0,7.
- Logstart/end setiapblock context keJSONLflush/fsync;faulthandler45detik.
  Ini mempermudah pelacakan bila terputus lagi, bukan klaim driver telah diperbaiki.
- Scope affinity0–11,NIaktual10,OMP4,MemoryHigh16G/Max20G/Swap1G;
  CPUQuota300% untuk mengurangi beban berkelanjutan. Tidak mengubah settingOS.
- Entry runtime fail-closed: kegagalan import/wrapper menghentikan child,
  bukan meneruskan inference tanpa pacing. Sumber166/189/191/155/183 dan
  helper/modelpretrained tetap dihash sesuai manifest asli.

Sebelum resume,dua CPUchecks199 harus lolos:output/state/RNG tepat sama pada
tinyKumo denganGQA dan penolakan wrapperganda/arsitektur tak dikenal. Kemudian
probe200:context56k/e4/gen4200,128query pertama dari completedseed42fold0.
Bandingkan dengan cached baseline mentah;maxabs≤0,0005,meanabs≤0,0001,
hookfit/querybenar berjalan. Tidak menghitungKS atau memilih model dari probe.
Hanya jika lulus,resume191 pada direktori asli;cache11fit diverifikasi kembali,
kerjakan4fit yang belum lengkap,lalu report155 yang sama. Rekam wrapper/entry/
probe/checksources pada `results/v56/recovery_21okt1840/recovery_manifest.json`.

Hang baru harus ditinjau dari trace/log aktual sebelum menjalankan ulang.
Pacing mengurangi beban rata-rata;tidak membuktikan penyebab hang sudah hilang.
