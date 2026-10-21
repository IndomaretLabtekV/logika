"""Run fixed-union validation sequentially: references, models, paired report.

Operational runner, no recipe selection or submission. Launch inside the user's
resource-limited scope; children inherit its affinity, NI and thread settings.
Every child has its own durable telemetry and preserved attempt log.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def save(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w") as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--seeds",default="42,7,2026")
    ap.add_argument("--folds",default="0,1,2,3,4")
    ap.add_argument("--bootstrap",type=int,default=1000)
    ap.add_argument("--tag",default="v56_fixed_union_rank_15fits")
    a=ap.parse_args();seeds=list(map(int,a.seeds.split(",")));folds=list(map(int,a.folds.split(",")))
    if not seeds or not folds or len(set(seeds))!=len(seeds) or len(set(folds))!=len(folds) or not set(folds)<=set(range(5)):
        raise ValueError("Invalid or repeated seed/fold")
    if a.bootstrap<2:raise ValueError("Invalid bootstrap count")
    work=a.run_root.resolve()
    if work==ROOT or work.is_relative_to(ROOT/"data") or work.is_relative_to(ROOT/"data_pq"):
        raise ValueError("Use a separate experiment root")
    work.mkdir(parents=True,exist_ok=True);logs=work/"validation_logs";logs.mkdir(exist_ok=True)
    scripts=(Path(__file__).name,"166_teammate_nested_features.py","189_fixed_kumo_union_rank.py","155_kumo_nested_eval.py","gpu_watch.py")
    sources={name:sha(ROOT/"eda"/name) for name in scripts}
    request={"run_root":str(work),"seeds":seeds,"folds":folds,"bootstrap":a.bootstrap,"tag":a.tag,
             "sources_sha256":sources,"submission_performed":False,"heavy_children_concurrent":1}
    mp=work/"validation_plan.json";progress_path=work/"validation_progress.json"
    if mp.exists():assert json.loads(mp.read_text())==request,"Changed orchestration/configuration; inspect existing run"
    else:save(mp,request)
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {"status":"running","attempts":[],"models_expected":len(seeds)*len(folds)}
    progress["status"]="running";save(progress_path,progress)
    figure=ROOT/"eda/out"/f"191_{work.name}_validation_progress.png"
    def plot():
        complete=[row for row in progress["attempts"] if row.get("exit_code")==0]
        counts={kind:sum(row["kind"]==kind for row in complete) for kind in ("features","model","report")}
        fig,ax=plt.subplots(figsize=(7,3));ax.bar(list(counts),list(counts.values()))
        ax.set(ylabel="Successful child executions (cache replays may repeat)",title=f"Fixed-policy validation: {progress['status']}")
        fig.tight_layout();fig.savefig(figure,dpi=130);plt.close(fig)
    def run(kind,key,script,args):
        for name,expected in sources.items():assert sha(ROOT/"eda"/name)==expected,"Source changed while validation was active"
        attempt=1+sum(row["key"]==key for row in progress["attempts"])
        stem=f"{key}_attempt{attempt:02d}";log=logs/f"{stem}.log";telemetry=logs/f"{stem}_telemetry.csv"
        if log.exists() or telemetry.exists():raise ValueError("Untracked previous attempt; preserve and inspect it")
        command=[sys.executable,str(ROOT/"eda"/script),*map(str,args)]
        row={"key":key,"kind":kind,"attempt":attempt,"command":command,"log":str(log),"telemetry":str(telemetry),"status":"running"}
        progress["attempts"].append(row);progress["current"]=key;save(progress_path,progress)
        print("START",key,"attempt",attempt,"log",log,flush=True);start=time.monotonic()
        with log.open("x") as output:
            child=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
            row["pid"]=child.pid;save(progress_path,progress)
            watcher=subprocess.Popen([sys.executable,str(ROOT/"eda/gpu_watch.py"),str(child.pid),"--out",str(telemetry)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
            code=child.wait();watch_code=watcher.wait()
        row.update(exit_code=code,watcher_exit_code=watch_code,elapsed_seconds=time.monotonic()-start,status="complete" if code==0 else "failed")
        save(progress_path,progress);plot();print("END",key,"exit",code,"seconds",round(row["elapsed_seconds"],1),flush=True)
        if code!=0:raise RuntimeError(f"Child {key} failed; no automatic retry. Inspect {log}")
    try:
        for seed in seeds:
            for fold in folds:
                cache=work/"teammate_features"/f"outer_s{seed}_f{fold}"
                run("features",f"features_s{seed}_f{fold}","166_teammate_nested_features.py",
                    ["--out",cache,"--partition","outer","--seed",seed,"--fold",fold])
                run("model",f"model_s{seed}_f{fold}","189_fixed_kumo_union_rank.py",
                    ["--feature-cache",cache,"--out",work/"fixed_union_rank","--seed",seed,"--fold",fold])
        if len(seeds)==1 or set(folds)==set(range(5)):
            run("report","paired_report","155_kumo_nested_eval.py",
                [work/"fixed_union_rank","--baseline-dir",work/"fixed_union_rank","--column","candidate","--baseline-column","baseline",
                 "--seeds",a.seeds,"--folds",a.folds,"--bootstrap",a.bootstrap,"--tag",a.tag])
        else:progress["report_note"]="Partial folds across seeds do not have common row coverage; finish five folds before aggregation."
        progress.update(status="complete",current=None);save(progress_path,progress);plot()
        print("Completed requested local fits; inspect155 report for the full gate. No submission.",flush=True)
    except BaseException as exc:
        progress.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        save(progress_path,progress);plot();raise


if __name__=="__main__":main()
