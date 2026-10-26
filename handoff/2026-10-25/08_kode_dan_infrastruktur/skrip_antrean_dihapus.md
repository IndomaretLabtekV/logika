# Skrip antrean yang dihapus saat pembersihan 25 Okt (arsip isi)

Skrip sekali pakai di eda/ untuk menjalankan member/screen secara berurutan. Dihapus karena hanya orkestrasi;
isinya disimpan di sini agar urutan run bisa direproduksi. Pola umum: export OMP_NUM_THREADS=4; taskset -c 0-11 nice -n 10;
satu member/fold per proses; sleep 20 antar fit; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True untuk fraction 0,85.

### eda/run_220_recipe.sh
```bash
#!/usr/bin/env bash
# Member-major run of the locked strict recipe (amendment_01); resumable: 212 skips cached members.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 POLARS_MAX_THREADS=4
run() {  # member fraction folds...
  local m=$1 frac=$2; shift 2
  for f in "$@"; do
    echo "=== $(date +%T) $m f$f" 
    taskset -c 0-11 nice -n 10 ../.venv/bin/python 212_strict_tfm_views.py --stage outer --fold $f --member $m --memory-fraction $frac || echo "FAILED $m f$f"
    sleep 20  # cool-down between fits
  done
}
for m in pool_cap500_s2026 nested_small_s2026; do run $m 0.7 0 1 2 3 4; done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True run pool_uncapped_s2026 0.85 0 1 2 3 4
for m in pool_cap500_s2027 nested_small_s2027; do run $m 0.7 0 1 2 3 4; done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True run pool_uncapped_s2027 0.85 0 1 2 3 4
run baseline 0.7 0 1 2 3 4
echo "=== DONE $(date +%T)"
```

### eda/run_220b_queue.sh
```bash
#!/usr/bin/env bash
# Queue v2 (01:35): finish nested_small_s2026, then C2 screen, then remaining recipe members.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 POLARS_MAX_THREADS=4
while kill -0 9240 2>/dev/null; do sleep 10; done
run() { local m=$1 frac=$2; shift 2; for f in "$@"; do echo "=== $(date +%T) $m f$f"; taskset -c 0-11 nice -n 10 ../.venv/bin/python 212_strict_tfm_views.py --stage outer --fold $f --member $m --memory-fraction $frac || echo "FAILED $m f$f"; sleep 20; done; }
screen() { for a in "$@"; do echo "=== $(date +%T) screen $a"; taskset -c 0-11 nice -n 10 ../.venv/bin/python 222_context_ssl_screen.py $a || echo "FAILED screen $a"; sleep 20; done; }
run nested_small_s2026 0.7 1 2 3 4
screen C2_ctx8192 C2_ctx24576
screen S3_base_teacher S3_pseudo_d0 S3_pseudo_d1 S3_permuted_d0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True run pool_uncapped_s2026 0.85 0 1 2 3 4
for m in pool_cap500_s2027 nested_small_s2027; do run $m 0.7 0 1 2 3 4; done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True run pool_uncapped_s2027 0.85 0 1 2 3 4
echo "=== DONE $(date +%T)"
```

### eda/run_220c_queue.sh
```bash
#!/usr/bin/env bash
# Queue v3 (01:46): nested_small_s2026 f2-4, protocol baseline f0 (diagnosis), C2, S3, rest.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 POLARS_MAX_THREADS=4
while kill -0 10265 2>/dev/null; do sleep 10; done
run() { local m=$1 frac=$2; shift 2; for f in "$@"; do echo "=== $(date +%T) $m f$f"; taskset -c 0-11 nice -n 10 ../.venv/bin/python 212_strict_tfm_views.py --stage outer --fold $f --member $m --memory-fraction $frac || echo "FAILED $m f$f"; sleep 20; done; }
screen() { for a in "$@"; do echo "=== $(date +%T) screen $a"; taskset -c 0-11 nice -n 10 ../.venv/bin/python 222_context_ssl_screen.py $a || echo "FAILED screen $a"; sleep 20; done; }
run nested_small_s2026 0.7 2 3 4
run baseline 0.7 0
screen C2_ctx8192 C2_ctx24576
screen S3_base_teacher S3_pseudo_d0 S3_pseudo_d1 S3_permuted_d0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True run pool_uncapped_s2026 0.85 0 1 2 3 4
for m in pool_cap500_s2027 nested_small_s2027; do run $m 0.7 0 1 2 3 4; done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True run pool_uncapped_s2027 0.85 0 1 2 3 4
echo "=== DONE $(date +%T)"
```

### eda/run_220d_queue.sh
```bash
#!/usr/bin/env bash
# Queue v4 (02:12): nested_small_s2026 f4, protocol baseline f0, screens (C2/M/S3), rest of recipe.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 POLARS_MAX_THREADS=4
while kill -0 12084 2>/dev/null; do sleep 10; done
run() { local m=$1 frac=$2; shift 2; for f in "$@"; do echo "=== $(date +%T) $m f$f"; taskset -c 0-11 nice -n 10 ../.venv/bin/python 212_strict_tfm_views.py --stage outer --fold $f --member $m --memory-fraction $frac || echo "FAILED $m f$f"; sleep 20; done; }
screen() { for a in "$@"; do echo "=== $(date +%T) screen $a"; taskset -c 0-11 nice -n 10 ../.venv/bin/python 222_context_ssl_screen.py $a || echo "FAILED screen $a"; sleep 20; done; }
run nested_small_s2026 0.7 4
run baseline 0.7 0
screen C2_ctx8192 M_ctx8192 S3_base_teacher M_small C2_ctx16384 M_ctx16384
screen S3_pseudo_d0 S3_pseudo_d1 S3_permuted_d0
taskset -c 0-11 nice -n 10 ../.venv/bin/python 222_context_ssl_screen.py C2_ctx8192 M_ctx8192 S3_base_teacher M_small C2_ctx16384 M_ctx16384 S3_pseudo_d0 S3_pseudo_d1 S3_permuted_d0
echo "=== SCREEN_DONE $(date +%T)"
```

### eda/run_220e_queue.sh
```bash
#!/usr/bin/env bash
# Queue v5 (03:05): v62 variant 5-fold (cap500_ctx16k, main_small; s2026), then remaining S3 screen arms.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 POLARS_MAX_THREADS=4
P=../artifacts/optimization/continuation_24okt/protocol_v62_ctx16k.json
var() { local m=$1; shift; for f in "$@"; do echo "=== $(date +%T) v62 $m f$f"; taskset -c 0-11 nice -n 10 ../.venv/bin/python 224_strict_tfm_variant.py --protocol $P --out ../results/v62/strict_variant --pool-root ../results/v60/strict_tfm --stage outer --fold $f --member $m || echo "FAILED v62 $m f$f"; sleep 20; done; }
screen() { for a in "$@"; do echo "=== $(date +%T) screen $a"; taskset -c 0-11 nice -n 10 ../.venv/bin/python 222_context_ssl_screen.py $a || echo "FAILED screen $a"; sleep 20; done; }
var cap500_ctx16k_s2026 0 1 2 3 4
var main_small_s2026 0 1 2 3 4
screen S3_pseudo_d0 S3_pseudo_d1 S3_permuted_d0 M_ctx8192
taskset -c 0-11 nice -n 10 ../.venv/bin/python 222_context_ssl_screen.py C2_ctx8192 M_ctx8192 S3_base_teacher M_small C2_ctx16384 M_ctx16384 S3_pseudo_d0 S3_pseudo_d1 S3_permuted_d0
echo "=== QUEUE5_DONE $(date +%T)"
```

### eda/run_220f_queue.sh
```bash
#!/usr/bin/env bash
# Queue v6 (05:25): after queue v5 (S3 screen) -> width/context screen (226) -> v62 seed-2027 members.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 POLARS_MAX_THREADS=4
while kill -0 17378 2>/dev/null; do sleep 10; done
P=../artifacts/optimization/continuation_24okt/protocol_v62_ctx16k.json
var() { local m=$1; shift; for f in "$@"; do echo "=== $(date +%T) v62 $m f$f"; taskset -c 0-11 nice -n 10 ../.venv/bin/python 224_strict_tfm_variant.py --protocol $P --out ../results/v62/strict_variant --pool-root ../results/v60/strict_tfm --stage outer --fold $f --member $m || echo "FAILED v62 $m f$f"; sleep 20; done; }
for a in W_cap300_ctx28672 W_cap200_ctx40960; do echo "=== $(date +%T) screen226 $a"; taskset -c 0-11 nice -n 10 ../.venv/bin/python 226_width_context_tradeoff.py $a || echo "FAILED screen226 $a"; sleep 20; done
var main_small_s2027 0 1 2 3 4
var cap500_ctx16k_s2027 0 1 2 3 4
echo "=== QUEUE6_DONE $(date +%T)"
```

### eda/run_220g_queue.sh
```bash
#!/usr/bin/env bash
# Queue v7 (06:10): v64 cap300_ctx28k s2026 5-fold, then v62 seed-2027 members.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 POLARS_MAX_THREADS=4
var() { local p=$1 o=$2 m=$3; shift 3; for f in "$@"; do echo "=== $(date +%T) $o $m f$f"; taskset -c 0-11 nice -n 10 ../.venv/bin/python 224_strict_tfm_variant.py --protocol $p --out ../results/$o/strict_variant --pool-root ../results/v60/strict_tfm --stage outer --fold $f --member $m || echo "FAILED $o $m f$f"; sleep 20; done; }
P64=../artifacts/optimization/continuation_24okt/protocol_v64_cap300_ctx28k.json
P62=../artifacts/optimization/continuation_24okt/protocol_v62_ctx16k.json
var $P64 v64 cap300_ctx28k_s2026 0 1 2 3 4
var $P62 v62 main_small_s2027 0 1 2 3 4
var $P64 v64 cap300_ctx28k_s2027 0 1 2 3 4
echo "=== QUEUE7_DONE $(date +%T)"
```

### eda/run_229.sh
```bash
#!/usr/bin/env bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 POLARS_MAX_THREADS=4
for a in X_main100_full X_main150_full X_wide_cap128_full; do echo "=== $(date +%T) $a"; taskset -c 0-11 nice -n 10 ../.venv/bin/python 229_full_context_budget.py $a || echo "FAILED $a"; sleep 20; done
taskset -c 0-11 nice -n 10 ../.venv/bin/python 229_full_context_budget.py X_main100_full X_main150_full X_wide_cap128_full W_cap300_ctx28672
echo "=== DONE229 $(date +%T)"
```

### eda/run_230_main150.sh
```bash
#!/usr/bin/env bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 POLARS_MAX_THREADS=4
P=../artifacts/optimization/continuation_24okt/protocol_v65_main150.json
for m in main150_s2026 main150_s2027; do for f in 0 1 2 3 4; do echo "=== $(date +%T) v65 $m f$f"; taskset -c 0-11 nice -n 10 ../.venv/bin/python 224_strict_tfm_variant.py --protocol $P --out ../results/v65/strict_variant --pool-root ../results/v60/strict_tfm --stage outer --fold $f --member $m || echo "FAILED v65 $m f$f"; sleep 20; done; done
echo "=== DONE230 $(date +%T)"
```

### eda/run_231.sh
```bash
#!/usr/bin/env bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 POLARS_MAX_THREADS=4
for a in B_cap300_ctx28k_8x1 B_cap128_full_8x1; do echo "=== $(date +%T) $a"; taskset -c 0-11 nice -n 10 ../.venv/bin/python 231_sequential_estimators.py $a || echo "FAILED $a"; sleep 20; done
echo "=== DONE231 $(date +%T)"
```

### eda/run_237.sh
```bash
#!/usr/bin/env bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
P="taskset -c 0-11 nice -n 10 ../.venv/bin/python"
$P 237_series_oof.py ecom 0 || echo FAILED ecom
$P 238_series_oof_eval.py ecom 0 || echo FAILED eval ecom
$P 237_series_oof.py repay 0 || echo FAILED repay
$P 238_series_oof_eval.py repay 0 || echo FAILED eval repay
$P 239_knn_target_mean.py 0 || echo FAILED knn
$P 238_series_oof_eval.py knn 0 || echo FAILED eval knn
echo DONE237
```

### eda/run_237b.sh
```bash
#!/usr/bin/env bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
P="taskset -c 0-11 nice -n 10 ../.venv/bin/python"
$P 237_series_oof.py repay 1,2,3,4 || echo FAILED repay
$P 238_series_oof_eval.py repay 0,1,2,3,4 || echo FAILED eval
echo DONE237b
```

### eda/run_240b.sh
```bash
#!/usr/bin/env bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
P="taskset -c 0-11 nice -n 10 ../.venv/bin/python"
$P 238_series_oof_eval.py coles_ecom 0,1,2,3,4 || echo FAILED eval coles_ecom
$P 240_coles_embeddings.py repay 12 || echo FAILED coles repay
$P 238_series_oof_eval.py coles_repay 0,1,2,3,4 || echo FAILED eval coles_repay
$P 238_series_oof_eval.py coles_ecom,coles_repay 0,1,2,3,4 || echo FAILED eval both
echo DONE240b
```

### eda/run_244_seed2028.sh
```bash
#!/usr/bin/env bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 POLARS_MAX_THREADS=4
P=../artifacts/optimization/continuation_24okt/protocol_v69_seed2028.json
for m in main_small_s2028 cap300_ctx28k_s2028; do for f in 0 1 2 3 4; do echo "=== $(date +%T) v69 $m f$f"; taskset -c 0-11 nice -n 10 ../.venv/bin/python 224_strict_tfm_variant.py --protocol $P --out ../results/v69/strict_variant --pool-root ../results/v60/strict_tfm --stage outer --fold $f --member $m || echo "FAILED v69 $m f$f"; sleep 20; done; done
echo "=== DONE244 $(date +%T)"
```

