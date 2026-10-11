import hashlib
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAPERS = ROOT / "papers"
PAPERS.mkdir(parents=True, exist_ok=True)
MANIFEST = json.loads((ROOT / "papers.json").read_text(encoding="utf-8"))


def valid_pdf(path):
    if not path.exists() or path.stat().st_size <= 10_000:
        return False
    with path.open("rb") as file:
        return file.read(5) == b"%PDF-"


def fetch(paper):
    destination = PAPERS / paper["file"]
    errors = []
    if not valid_pdf(destination):
        for url in paper["urls"]:
            descriptor, temporary_name = tempfile.mkstemp(suffix=".pdf", dir=PAPERS)
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                result = subprocess.run(
                    ["curl", "--fail", "--location", "--silent", "--show-error",
                     "--connect-timeout", "10", "--max-time", "45", "--retry", "1",
                     "--user-agent", "Mozilla/5.0 (research paper archival)",
                     "--output", str(temporary), url],
                    capture_output=True, text=True, timeout=100,
                )
                if result.returncode == 0 and valid_pdf(temporary):
                    temporary.replace(destination)
                    break
                errors.append(f"{url}: {result.stderr.strip() or 'response was not a PDF'}")
            except subprocess.TimeoutExpired:
                errors.append(f"{url}: timed out")
            finally:
                temporary.unlink(missing_ok=True)
    if valid_pdf(destination):
        destination.chmod(0o644)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        return {"id": paper["id"], "status": "downloaded", "file": paper["file"],
                "bytes": destination.stat().st_size, "sha256": digest}
    return {"id": paper["id"], "status": "unavailable", "source": paper["source"],
            "errors": errors}


results = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(fetch, paper): paper["id"] for paper in MANIFEST}
    for future in as_completed(futures):
        result = future.result()
        results.append(result)
        print(f"{result['status']}: {result['id']}", flush=True)
results.sort(key=lambda item: item["id"])
(ROOT / "download_status.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print(f"Downloaded {sum(item['status'] == 'downloaded' for item in results)}/{len(results)} PDFs")
