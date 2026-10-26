#!/usr/bin/env bash
# DSC0152 v66 on valid2 folds: five folds, seed 42, inner stopping and refit.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON="${LOGIKA_PYTHON:-$REPOSITORY/.venv/bin/python}"
export OMP_NUM_THREADS=4

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment missing: $PYTHON. See environments/README.md." >&2
    exit 1
fi

cd "$REPOSITORY/scripts/dsc0152"

run_cpu() {
    taskset -c "${LOGIKA_CPU_AFFINITY:-0-11}" nice -n 10 "$PYTHON" "$@"
}

if [[ ! -f agas/shared/cache/features_cache.pkl ]]; then
    run_cpu -c '
import os
import sys
sys.path.insert(0, "agas/shared")
import paths
import features
paths.CACHE.mkdir(parents=True, exist_ok=True)
frame = features.build()
temporary = str(paths.FEATURES_CACHE) + ".tmp"
frame.to_pickle(temporary)
os.replace(temporary, paths.FEATURES_CACHE)
print("features", frame.shape, flush=True)
'
fi

if [[ ! -f agas/shared/cache/extra_cache.pkl ]]; then
    run_cpu agas/shared/build_extra.py bureau,repayment,previous,account,ecommerce,interactions,gradenorm,rxtra,bxtra,axtra,ecommerce2,consistency,dataquality,stress
fi

run_cpu agas/v66/training/train_submission_v66.py \
    --folds 5 --fold-seeds 42 --lgb-seeds 0,1,2,3,4 \
    --valid2-folds --inner-stop --n-jobs 4 --out-prefix v66s42
echo DSC_DONE
