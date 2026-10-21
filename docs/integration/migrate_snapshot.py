"""One-shot, checksum-guarded migration of the inventoried teammate snapshot.

Run from the project root with --apply. Model operations are never executed here.
Original bytes are verified before moves; subsequent text changes have reversible
patches. Historical JSON, notebook outputs, CSVs and third-party files stay intact.
"""
import argparse
import ast
import csv
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit, unquote

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "docs/integration"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def destination(relative):
    p = Path(relative)
    if p.parts[0] in {"scripts", "artifacts", "third_party"}:
        return str(p)
    if p.parts[0] == "notebooks":
        return str(Path("notebook") / ("README_teman.md" if p.name == "README.md" else p.name))
    if p.name.startswith("requirements"):
        return str(Path("environments/teman") / p.name)
    if p.name.startswith("submission") and p.suffix == ".csv":
        return str(Path("results/teman") / p.name)
    return {"README.md": "docs/integration/riset_teman.md",
            ".gitignore": "docs/integration/gitignore_teman.txt",
            "analisis_logika.ipynb": "notebook/analisis_logika.ipynb"}[str(p)]


def adapt(text):
    # Canonical data directory is ROOT/data for both implementations. No raw copy.
    text = text.replace("data/raw", "data").replace("notebooks/", "notebook/")
    text = re.sub(r"(['\"])notebooks\1", lambda m: m[1]+"notebook"+m[1], text)
    # Keep submissions from this source separate from the main version history.
    text = re.sub(r"\b(ROOT|root|r\.ROOT)\s*/\s*(['\"])(submission[^'\"\n]*\.csv)\2",
                  lambda m: f'{m[1]} / {m[2]}results/teman/{m[3]}{m[2]}', text)
    text = re.sub(r"\bROOT\s*/\s*CANONICAL_FILE\b", 'ROOT / "results/teman" / CANONICAL_FILE', text)
    text = re.sub(r"(?<![\w/-])(requirements(?:-[\w]+)?\.txt)", r"environments/teman/\1", text)
    return text


def links(text, old, new, mapping):
    def replace(m):
        url = m[2]
        u = urlsplit(url)
        if u.scheme or url.startswith(("#", "/")):
            return m[0]
        target = Path(os.path.normpath(str(Path(old).parent / unquote(u.path))))
        dest = mapping.get(str(target))
        if dest is None:
            return m[0]
        rel = os.path.relpath(dest, str(Path(new).parent))
        if u.fragment:
            rel += "#"+u.fragment
        return f"{m[1]}{rel}{m[3]}"
    return re.sub(r"(!?\[[^\]]*\]\()([^\s)]+)(\))", replace, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    rows = [r for r in csv.DictReader((AUDIT / "inventaris_awal.csv").open()) if r["origin"]=="teman"]
    mapping = {r["path"]: destination(str(Path(r["path"]).relative_to("Logika-main"))) for r in rows}
    assert len(set(mapping.values())) == len(rows), "Destination collision"
    for row in rows:
        src, dst = ROOT / row["path"], ROOT / mapping[row["path"]]
        assert src.is_file() and digest(src)==row["sha256"], f"Changed source: {src}"
        assert not dst.exists(), f"Existing destination: {dst}"
    print(f"Verified {len(rows)} original files, unique destinations; apply={args.apply}")
    if not args.apply:
        return
    moved = []
    for row in rows:
        old, new = row["path"], mapping[row["path"]]
        src, dst = ROOT / old, ROOT / new
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        assert digest(dst)==row["sha256"]
        moved.append({"source": old, "destination": new, "original_sha256": row["sha256"],
                      "bytes": row["bytes"], "new_sha256": row["sha256"]})
    patches = []
    for row in moved:
        p = ROOT / row["destination"]
        if p.suffix not in {".py", ".md", ".ipynb"} or row["destination"].startswith("third_party/"):
            continue
        before = p.read_text()
        if p.suffix == ".ipynb":
            obj = json.loads(before)
            for cell in obj["cells"]:
                value = cell["source"]
                source = "".join(value) if isinstance(value, list) else value
                source = adapt(source)
                cell["source"] = source.splitlines(keepends=True) if isinstance(value, list) else source
            after = json.dumps(obj, ensure_ascii=False, indent=1)+"\n"
        else:
            after = links(before, row["source"], row["destination"], mapping) if p.suffix==".md" else before
            # Local hyperlinks are already remapped. Don't rewrite them a second time.
            if p.suffix == ".md":
                after = after.replace("data/raw", "data")
                after = re.sub(r"(?<![\w/-])(requirements(?:-[\w]+)?\.txt)", r"environments/teman/\1", after)
            else:
                after = adapt(after)
                ast.parse(after, filename=str(p))
        if before != after:
            patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                               fromfile=row["source"], tofile=row["destination"]))
            p.write_text(after)
            row["new_sha256"] = digest(p)
    (AUDIT / "adaptasi_path.patch").write_text("".join(patches))
    with (AUDIT / "pemetaan_file.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(moved[0])); writer.writeheader(); writer.writerows(moved)
    # Keep the empty skeleton until the independent verification has passed.
    print(f"Moved and hash-verified {len(moved)} files; empty snapshot retained pending checks.")


if __name__ == "__main__":
    main()
