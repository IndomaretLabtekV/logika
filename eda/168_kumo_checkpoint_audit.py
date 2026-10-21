"""CPU-only provenance of the local Kumo-large cache and teammate declarations.

No model deserialization, torch/GPU allocation, network, or checkpoint mutation.
This checks current bytes; absent teammate weight files prevent cross-machine
byte equality claims. The historical metadata is evidence of a declaration.
"""
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub.constants import HF_HUB_CACHE

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"eda/out"


def sha(path):
    with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    package=Path(importlib.util.find_spec("sdm").origin).parent
    source=package/"models/kumo/tabular/model.py"
    downloader=package/"models/_huggingface.py"
    tree=ast.parse(source.read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and
           isinstance(n.func,ast.Name) and n.func.id=="download_checkpoint"]
    assert len(calls)==1
    literals={item.arg:ast.literal_eval(item.value) for item in calls[0].keywords
              if isinstance(item.value,ast.Constant)}
    assert literals["repo_id"]=="nvidia/Kumo-Tabular" and literals["revision"]=="v1.0.1"
    cache=Path(HF_HUB_CACHE)/"models--nvidia--Kumo-Tabular"
    commit=(cache/"refs"/literals["revision"]).read_text().strip()
    assert re.fullmatch("[0-9a-f]{40}",commit)
    snapshot=cache/"snapshots"/commit
    checkpoint=snapshot/"large/classifier.pt"
    assert checkpoint.is_symlink() and checkpoint.is_file()
    # Hub2 may deduplicate the repository blob into another physical store.
    # The snapshot's logical blob name carries the expected SHA256; resolving
    # every symlink first would discard that name and inspect a different key.
    logical_blob=Path(os.path.normpath(checkpoint.parent/checkpoint.readlink()))
    assert logical_blob.parent==cache/"blobs" and re.fullmatch("[0-9a-f]{64}",logical_blob.name)
    blob=logical_blob.resolve(strict=True)
    checkpoint_hash=sha(blob)
    assert checkpoint_hash==logical_blob.name,"Local checkpoint does not match its snapshot blob SHA256"
    config_path=snapshot/"config.json"
    config=json.loads(config_path.read_text())
    assert config["checkpoint_files"]["large"]["classification"]=="large/classifier.pt"
    declarations=[]
    for path in sorted((ROOT/"artifacts/optimization/kumo/large_v5_ctx50000_e8").glob("fold*_oof.json")):
        recipe=json.loads(path.read_text())["recipe"]
        assert recipe["size"]=="large" and recipe["weights"]==literals["repo_id"]
        assert recipe["weights_revision"]==literals["revision"] and recipe["weights_commit"]==commit
        declarations.append({"path":str(path.relative_to(ROOT)),"sha256":sha(path),
                             "declared_commit":recipe["weights_commit"],"declared_sdm":recipe["packages"]["structured-data-models"]})
    assert len(declarations)==3
    version=importlib.metadata.version("structured-data-models")
    assert all(row["declared_sdm"]==version for row in declarations)
    report={"repository":literals["repo_id"],"revision":literals["revision"],"snapshot_commit":commit,
            "size":"large","checkpoint_bytes":blob.stat().st_size,"checkpoint_sha256":checkpoint_hash,
            "local_content_address_verified":True,"config_sha256":sha(config_path),
            "sdm_version":version,"model_loader_sha256":sha(source),"downloader_sha256":sha(downloader),
            "teammate_declarations":declarations,"source_sha256":sha(Path(__file__).resolve()),
            "model_loaded":False,"gpu_used":False,"network_used":False,
            "limits":["Current loader/cache is pinned to the same revision/commit declared by the teammate.",
                      "Teammate weight files are absent: historical byte equality cannot be proven.",
                      "Equal package version does not prove equal runtime kernels or predictions.",
                      "This audit cannot identify the cause of a score difference or demonstrate a gain."]}
    path=OUT/"168_kumo_checkpoint_audit.json"
    path.write_text(json.dumps(report,indent=2))
    fig,ax=plt.subplots(figsize=(8,3))
    labels=["local blob SHA256","loader revision","3 teammate commits","SDM version"]
    ax.barh(labels,[1,1,1,1]);ax.set(xlim=(0,1.5),xticks=[],title="Kumo-large provenance: declarations agree; historical bytes unavailable")
    for i in range(4):ax.text(1.03,i,"verified",va="center")
    fig.tight_layout();fig.savefig(OUT/"168_kumo_checkpoint_audit.png",dpi=130);plt.close(fig)
    print(f"Kumo large: {blob.stat().st_size:,} bytes SHA256={checkpoint_hash}")
    print(f"loader revision={literals['revision']} snapshot={commit} SDM={version}")
    print(f"{len(declarations)} teammate fold declarations agree; historical checkpoint bytes unavailable")
    print("No model loaded, GPU or network used; saved",path)


if __name__=="__main__":main()
