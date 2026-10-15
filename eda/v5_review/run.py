"""Run only this CPU analysis suite, sequentially; save full logs and process resource usage."""
from pathlib import Path
import sys, subprocess, time, json, resource
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
allowed = sorted(p.stem for p in HERE.glob("[0-9][0-9]_*.py"))
chosen = sys.argv[1:] or allowed
assert chosen and all(n in allowed for n in chosen), "Unknown analysis"
stats = {}
p = OUT / "run_stats.json"
if p.exists():
    stats = json.loads(p.read_text())
for name in chosen:
    start = time.monotonic()
    with (OUT / (name + ".log")).open("w") as log:
        result = subprocess.run([sys.executable, "-B", "-u", str(HERE / (name + ".py"))],
            stdout=log, stderr=subprocess.STDOUT)
    if p.exists():
        stats.update(json.loads(p.read_text()))
    stats[name] = {"seconds": time.monotonic()-start, "exit_code": result.returncode,
        "cumulative_child_max_rss_MiB": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024}
    p.write_text(json.dumps(stats, indent=2)+"\n")
    print(name, stats[name], flush=True)
    print("\n".join((OUT / (name + ".log")).read_text().splitlines()[-25:]), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
