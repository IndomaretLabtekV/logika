"""Pause (SIGSTOP) heavy model jobs when hot, resume (SIGCONT) after cooling; durable telemetry CSV.

Usage: thermal_guard.py <telemetry.csv> <process-pattern> ...
Pause when GPU > 85 C, or the hottest CPU thermal zone is > 95 C on any sample (after 3 laptop hangs;
earlier 3-sample rule let CPU sit >95 C 73% of the time); resume below 78/85 C.
"""
import glob
import os
import signal
import subprocess
import sys
import time

GPU_HOT, CPU_HOT, GPU_OK, CPU_OK = 85, 95, 78, 85


def gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,power.draw,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    return [float(x) for x in out.strip().split(",")]


def cpu():
    temps = []
    for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            temps.append(int(open(path).read()) / 1000)
        except (OSError, ValueError):
            pass
    return max(temps)


def pids(patterns):
    out = subprocess.run(["pgrep", "-f", "|".join(patterns)], capture_output=True, text=True).stdout
    return [int(p) for p in out.split() if int(p) != os.getpid()]


def main():
    path, patterns = sys.argv[1], sys.argv[2:]
    paused = False
    hot_cpu = 0

    def release(*_):  # never leave a job SIGSTOPped when the guard is stopped/restarted
        for p in pids(patterns):
            os.kill(p, signal.SIGCONT)
        sys.exit(0)

    signal.signal(signal.SIGTERM, release)
    new = not os.path.exists(path)
    with open(path, "a") as log:
        if new:
            log.write("time,gpu_c,vram_mib,power_w,util,cpu_c,paused,pids\n")
        while True:
            g, c, targets = gpu(), cpu(), pids(patterns)
            hot_cpu = hot_cpu + 1 if c > CPU_HOT else 0
            if not paused and (g[0] > GPU_HOT or hot_cpu >= 1):
                for p in targets:
                    os.kill(p, signal.SIGSTOP)
                paused = True
            elif paused and g[0] < GPU_OK and c < CPU_OK:
                for p in targets:
                    os.kill(p, signal.SIGCONT)
                paused = False
            log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{g[0]},{g[1]},{g[2]},{g[3]},{c},{int(paused)},{len(targets)}\n")
            log.flush()
            os.fsync(log.fileno())
            time.sleep(5)


if __name__ == "__main__":
    main()
