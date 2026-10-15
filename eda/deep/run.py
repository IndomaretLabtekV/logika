"""Run a numbered EDA stage, capture stdout and timing. No models or notebooks are executed."""
import os
os.environ.setdefault("POLARS_MAX_THREADS","4")
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
from pathlib import Path
import sys
import subprocess
import time
import resource

HERE=Path(__file__).resolve().parent
OUT=HERE/"out"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR",str(OUT/"mpl_cache"))
STAGES=["00_integrity.py","01_validation.py","02_temporal_features.py","03_residual_screen.py",
        "04_preprocessing.py","05_relational_export.py","06_feature_budget.py","07_observation_followup.py","08_numeric_semantics.py"]
if __name__=="__main__":
    stages=STAGES if len(sys.argv)==1 else [p for p in STAGES if p.startswith(sys.argv[1]+"_")]
    if not stages:raise SystemExit("Use no argument for all stages, or 00, 01, 02, 03, 04, 05, 06, 07, 08.")
    for name in stages:
        start=time.monotonic()
        with (OUT/(name[:-3]+".log")).open("w") as log:
            result=subprocess.run([sys.executable,"-B","-u",str(HERE/name)],stdout=log,stderr=subprocess.STDOUT)
        print(name,"returncode",result.returncode,"seconds",round(time.monotonic()-start,1),
              "cumulative_child_max_RSS_MiB",round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,1),flush=True)
        print((OUT/(name[:-3]+".log")).read_text()[-6500:],flush=True)
        if result.returncode:raise SystemExit(result.returncode)
