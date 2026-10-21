"""Read-only inventory before integrating a research snapshot; hashes are content evidence.

Does not inspect environments, Git objects, secrets or external paths. Writes only
its audit outputs, never changes the inventoried sources. Run once before migration.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/integration"
OUT = ROOT / "eda/out"
SKIP = {".git", ".venv", ".venv-zoo", ".venv-rel", "__pycache__", ".ipynb_checkpoints"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    assert not (AUDIT / "inventaris_awal.csv").exists(), "Do not overwrite pre-migration evidence"
    initial = subprocess.check_output(["git", "status", "--short", "--untracked-files=all"], cwd=ROOT, text=True)
    (AUDIT / "git_status_awal.txt").write_text(initial)
    (AUDIT / "git_diff_awal.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT))
    rows = []
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if any(c in SKIP for c in rel.parts) or p.is_symlink() or not p.is_file():
            continue
        if rel.parts[:2] == ("docs", "integration") or rel.name.startswith("158_"):
            continue
        origin = "teman" if rel.parts[0] == "Logika-main" else "utama"
        # Raw/derived data inventoried too: no content deduplication is assumed.
        rows.append(dict(path=str(rel), origin=origin, bytes=p.stat().st_size, sha256=sha(p),
                         suffix=p.suffix, group=rel.parts[1] if origin == "teman" and len(rel.parts)>1 else rel.parts[0]))
    with (AUDIT / "inventaris_awal.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["sha256"]].append(row)
    duplicates = [{"sha256": h, "bytes_each": rs[0]["bytes"], "paths": [r["path"] for r in rs],
                   "cross_source": len({r["origin"] for r in rs}) > 1}
                  for h, rs in groups.items() if len(rs)>1]
    (AUDIT / "duplikat_checksum.json").write_text(json.dumps(duplicates, indent=2))
    counts = Counter(r["origin"] for r in rows)
    summary = {"counts": counts, "bytes": {s: sum(r["bytes"] for r in rows if r["origin"]==s) for s in counts},
               "duplicate_groups": len(duplicates), "cross_source_groups": sum(d["cross_source"] for d in duplicates),
               "friend_git_directory": (ROOT / "Logika-main/.git").exists(),
               "friend_prediction_parquets": sum(r["suffix"]==".parquet" and r["origin"]=="teman" for r in rows),
               "friend_raw_csvs": [r["path"] for r in rows if r["origin"]=="teman" and "/data/raw/" in r["path"]]}
    (AUDIT / "ringkasan_inventaris.json").write_text(json.dumps(summary, indent=2))
    fig, ax = plt.subplots(figsize=(8, 4))
    for source, color in [("utama", "C0"), ("teman", "C1")]:
        suffixes = [".py", ".ipynb", ".json", ".md", ".parquet", ".csv"]
        vals = [sum(r["origin"]==source and r["suffix"]==s for r in rows) for s in suffixes]
        offset = -.2 if source == "utama" else .2
        ax.bar([i+offset for i in range(len(vals))], vals, width=.4, label=source, color=color)
    ax.set_xticks(range(len(suffixes)), suffixes); ax.set_ylabel("Files inventoried (SHA256)")
    ax.set_title("Research inventory before migration; no score comparison"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "158_repository_inventory.png", dpi=130); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
