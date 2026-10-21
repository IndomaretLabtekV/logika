"""Durable read-only GPU/RAM telemetry for one existing process; no training.

Stops when the PID exits or is reused. Each row is fsynced, so an unclean laptop
shutdown leaves useful observations. Never controls clocks or kills the job.
"""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import time


def identity(pid):
    try:
        fields=Path(f"/proc/{pid}/stat").read_text().rsplit(") ",1)[1].split()
        return None if fields[0]=="Z" else fields[19]
    except FileNotFoundError:return None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pid",type=int)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--interval",type=float,default=2.)
    ap.add_argument("--max-samples",type=int,default=0)
    a=ap.parse_args();assert a.pid>0 and a.interval>=1 and a.max_samples>=0
    token=identity(a.pid);assert token is not None,"Target process already terminal"
    assert not a.out.exists(),"Do not overwrite previous telemetry"
    a.out.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();samples=0
    print(f"watch pid={a.pid}; durable CSV={a.out}; no device mutations",flush=True)
    with a.out.open("x") as f:
        w=csv.writer(f);w.writerow(["elapsed_s","gpu_util_pct","gpu_memory_mib","gpu_temp_c","gpu_power_w","rss_mib","mem_available_mib","error"])
        f.flush();os.fsync(f.fileno())
        while identity(a.pid)==token:
            values=[""]*4;error=""
            try:
                p=subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=3,check=True)
                fields=next(csv.reader(p.stdout.splitlines()));assert len(fields)==4
                values=[float(x.strip()) for x in fields]
            except Exception as exc:error=type(exc).__name__
            try:
                status=Path(f"/proc/{a.pid}/status").read_text()
                rss=float(next(line.split()[1] for line in status.splitlines() if line.startswith("VmRSS:")))/1024
            except (FileNotFoundError,StopIteration):break
            mem=Path("/proc/meminfo").read_text()
            available=float(next(line.split()[1] for line in mem.splitlines() if line.startswith("MemAvailable:")))/1024
            w.writerow([round(time.monotonic()-start,3),*values,rss,available,error]);f.flush();os.fsync(f.fileno())
            samples+=1
            if a.max_samples and samples>=a.max_samples:break
            time.sleep(a.interval)
    print(f"telemetry ended: {samples} samples",flush=True)


if __name__=="__main__":main()
