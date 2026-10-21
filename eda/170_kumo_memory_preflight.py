"""Full inner-context memory check after laptop hang; never computes a KS score.

Same Kumo/e4/48k fitting context, small128 query, lower allocator and query chunk.
Durable telemetry runs in a lightweight thread. No outer predictions/selection.
"""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import sdm
import valid2 as v
from common import ROOT,OUT,PQ,features


def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    stem=OUT/"170_kumo_memory_preflight"
    assert not stem.with_suffix(".json").exists(),"Keep the old report; use a new tag/source for another experiment"
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.7)
    spec=importlib.util.spec_from_file_location("preflight_vip",Path(__file__).with_name("152_vipcop_kumo.py"))
    vip=importlib.util.module_from_spec(spec);spec.loader.exec_module(vip)
    X,_,y,ids,_=v.frames(features())
    tr,va=v.outer(y,42)[0];fi,st=v.inner(y[tr],42,0);fi,st=tr[fi],tr[st]
    qs,qc=train_test_split(st,test_size=.5,stratify=y[st],random_state=42)
    qs=np.sort(qs);query=vip.strat(qs,y,128,np.random.default_rng(55))
    assert len(fi)==48000 and not np.intersect1d(fi,query).size and not np.intersect1d(fi,va).size
    cols=v.fm_select(X,y,fi,100)
    F=pd.concat([X.iloc[fi][cols],X.iloc[query][cols]],ignore_index=True)
    F=vip.prepare(F,np.arange(len(fi)),"raw")
    labels=np.r_[y[fi],np.zeros(len(query),dtype=int)]
    report={"engineering_only":True,"quality_score_computed":False,"outer_used":False,
            "real_context_rows":len(fi),"query_rows":len(query),"features":cols,
            "estimators":4,"generator_seed":4200,"chunk":128,"allocator_fraction":.7,
            "real_ids_json_sha256":hashlib.sha256(json.dumps(ids[fi].to_list(),separators=(",",":")).encode()).hexdigest(),
            "features_sha256":sha(PQ/"features.parquet"),"helper_sha256":sha(Path(__file__).with_name("152_vipcop_kumo.py")),
            "source_sha256":sha(Path(__file__).resolve()),"status":"starting"}
    start=time.monotonic();stop=threading.Event();stage={"name":"model_init"}

    def save_report():stem.with_suffix(".json").write_text(json.dumps(report,indent=2,allow_nan=False))

    def telemetry():
        with stem.with_suffix(".csv").open("w") as f:
            writer=csv.writer(f);writer.writerow(["elapsed_s","stage","gpu_util_pct","gpu_memory_mib","gpu_temp_c","gpu_power_w","rss_mib","mem_available_mib","error"])
            while not stop.is_set():
                error="";values=[""]*4
                try:
                    p=subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=3,check=True)
                    values=next(csv.reader(p.stdout.splitlines()));assert len(values)==4
                    values=[float(x.strip()) for x in values]
                except Exception as exc:error=type(exc).__name__
                mem={line.split(':')[0]:line.split(':')[1].strip().split()[0] for line in Path('/proc/meminfo').read_text().splitlines()}
                status={line.split(':')[0]:line.split(':')[1].strip().split()[0] for line in Path('/proc/self/status').read_text().splitlines() if ':' in line and line.split(':')[1].strip()}
                writer.writerow([round(time.monotonic()-start,3),stage["name"],*values,float(status["VmRSS"])/1024,float(mem["MemAvailable"])/1024,error])
                f.flush();os.fsync(f.fileno());stop.wait(2)

    save_report();thread=threading.Thread(target=telemetry,daemon=True);thread.start()
    print("ENGINEERING ONLY: context48000 query128 e4 chunk128 allocator0.7; no KS/outer",flush=True)
    try:
        model=sdm.models.KumoTabular(task="classification",size="large",device="cuda");model.eval()
        stage["name"]="table_tensor";print("model loaded; construct table",flush=True)
        predictor=vip.Predictor(F,labels,model)
        stage["name"]="cached_fit_predict";print("start full-context fit/predict",flush=True)
        torch.cuda.reset_peak_memory_stats()
        p=predictor.predict_cached(np.arange(len(fi)),np.arange(len(fi),len(F)),cols,4,4200,128)
        assert len(p)==128 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        report.update(status="complete",prediction_min=float(p.min()),prediction_max=float(p.max()),
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        print("full-context inference completed",report["peak_allocated_gib"],"GiB allocated",flush=True)
    except Exception as exc:
        report.update(status="error",error=type(exc).__name__,message=str(exc));raise
    finally:
        stop.set();thread.join(timeout=5);report["elapsed_seconds"]=time.monotonic()-start;save_report()
        d=pd.read_csv(stem.with_suffix(".csv"))
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].plot(d.elapsed_s,d.gpu_memory_mib,label="driver GPU MiB");axes[0].plot(d.elapsed_s,d.rss_mib,label="RSS MiB");axes[0].legend();axes[0].set(xlabel="seconds",ylabel="MiB",title="Full-context engineering check")
        axes[1].plot(d.elapsed_s,d.gpu_temp_c);axes[1].set(xlabel="seconds",ylabel="GPU temperature C",title="Telemetry; not a diagnosis of earlier hang")
        fig.tight_layout();fig.savefig(stem.with_suffix(".png"),dpi=130);plt.close(fig)


if __name__=="__main__":main()
