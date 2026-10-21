"""Download the exact LOGIKA UI 2026 competition; never print credentials."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import zipfile

COMPETITION = "data-science-competition-logika-ui-2026"
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path)
    args = parser.parse_args()
    candidates = [args.credentials] if args.credentials else (
        sorted(ROOT.glob("*kaggle*.json"))
        + [Path.home() / ".kaggle/kaggle.json", Path.home() / ".config/kaggle/kaggle.json"]
    )
    credentials = next((p for p in candidates if p and p.is_file()), None)
    if credentials:
        values = json.loads(credentials.read_text())
        os.environ["KAGGLE_USERNAME"] = values["username"]
        os.environ["KAGGLE_KEY"] = values["key"]

    # Import after setting the environment because kaggle authenticates on import.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    raw = ROOT / "data"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = api.competition_list_files(COMPETITION).to_dict()
    (raw / "download_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    for entry in manifest["files"]:
        name = entry["name"]
        destination = raw / name
        if destination.exists() and destination.stat().st_size == entry["totalBytes"]:
            print(f"Already present: {name}", flush=True)
            continue
        print(f"Downloading {name} ({entry['totalBytes'] / 1e6:.1f} MB)", flush=True)
        api.competition_download_file(COMPETITION, name, path=str(raw), quiet=True)
        archive = raw / f"{name}.zip"
        if archive.exists():
            with zipfile.ZipFile(archive) as zf:
                for member in zf.infolist():
                    resolved = (raw / member.filename).resolve()
                    if not resolved.is_relative_to(raw.resolve()):
                        raise RuntimeError("Unsafe path in downloaded archive")
                zf.extractall(raw)
            archive.unlink()
        if not destination.exists() or destination.stat().st_size != entry["totalBytes"]:
            raise RuntimeError(f"Unexpected size for {name}")
    print("All competition files downloaded and sizes verified.", flush=True)


if __name__ == "__main__":
    main()
