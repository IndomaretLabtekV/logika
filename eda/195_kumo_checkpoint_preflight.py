"""Runtime activation recomputation; optional exact two-step187 engineering probe.

Does not edit SDM, old experiments or model weights on disk. No quality metric.
GPU mode is explicitly --run-preflight and must be scheduled as the sole heavy job.
"""
import argparse
from functools import wraps
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import runpy
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.checkpoint import checkpoint

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def enable_checkpointing(model):
    if hasattr(model,"_logika_checkpointing"):raise ValueError("Checkpointing already installed")
    pattern=re.compile(r"(?:^|\.)(?:row_embedding\.(?:cell_embedding|col_blocks\.\d+|row_blocks\.\d+)|icl_block\.layers\.\d+)$")
    selected=[(name,block) for name,block in model.named_modules() if pattern.search(name)]
    if not selected:raise ValueError("No expected Kumo blocks found")
    state={"modules":[name for name,_ in selected],"compute_calls":{name:0 for name,_ in selected},"bypass_calls":0,
           "use_reentrant":False,"preserve_rng_state":True}
    def wrap(name,block):
        original=block.forward
        @wraps(original)
        def compute(*args,**kwargs):
            state["compute_calls"][name]+=1
            return original(*args,**kwargs)
        @wraps(original)
        def forward(*args,**kwargs):
            if torch.is_grad_enabled() and block.training:
                return checkpoint(compute,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
            state["bypass_calls"]+=1
            return original(*args,**kwargs)
        return forward
    for name,block in selected:block.forward=wrap(name,block)
    model._logika_checkpointing=state
    return state


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--fold",type=int,choices=range(5),default=0)
    ap.add_argument("--run-preflight",action="store_true")
    a=ap.parse_args()
    if not a.run_preflight:raise SystemExit("CPU parity checks: run196. GPU requires --run-preflight after the active heavy job ends.")
    progress=ROOT/"results/v56/validation_progress.json"
    if progress.exists() and json.loads(progress.read_text())["status"]=="running":
        raise RuntimeError("Fixed-union validation still active; refusing a second heavy job")
    work=a.out.resolve()
    if not work.is_relative_to(ROOT/"results") or work==ROOT/"results":raise ValueError("Use an owned experiment results subdirectory")
    if work.exists() and any(work.iterdir()):raise ValueError("Use fresh engineering output; inspect old results before retry")
    checks=ROOT/"eda/out/196_checkpoint_checks.json";checked=json.loads(checks.read_text())
    assert checked["failures"]==checked["errors"]==0 and checked["checks"]>=4
    assert checked["wrapper_sha256"]==sha(__file__)
    import sdm
    package=Path(importlib.util.find_spec("sdm").origin).parent
    sources=[Path(__file__),ROOT/"eda/187_kumo_sgd_preflight.py",ROOT/"eda/196_checkpoint_checks.py",
             Path(importlib.util.find_spec("torch.utils.checkpoint").origin)]
    sources += [package/"models/kumo/tabular"/name for name in ("model.py","row_embedding.py","cell_embedding.py","icl.py","block.py")]
    manifest={"sources_sha256":{str(p):sha(p) for p in sources},"protocol_sha256":sha(ROOT/"docs/integration/protokol_kumo_checkpoint_preflight.md"),
              "cpu_checks_sha256":sha(checks),"base_preflight":"187 --small-episode","seed":a.seed,"fold":a.fold,
              "activation_checkpointing":{"use_reentrant":False,"preserve_rng_state":True},"model_quality_evaluated":False,"weights_saved":False}
    work.mkdir(parents=True,exist_ok=True);(work/"wrapper_manifest.json").write_text(json.dumps(manifest,indent=2))
    original=sdm.models.KumoTabular;state={}
    def factory(*args,**kwargs):
        model=original(*args,**kwargs);state.update(enable_checkpointing(model))
        from sdm.models.kumo.tabular.model import MODEL_KWARGS
        cfg=MODEL_KWARGS["large"]
        expected=1+2*cfg["num_embedding_layers"]+cfg["num_icl_layers"]
        assert len(state["modules"])==expected,"Unexpected large Kumo architecture; inspect before training"
        return model
    previous_argv=sys.argv
    try:
        sdm.models.KumoTabular=factory
        sys.argv=[str(ROOT/"eda/187_kumo_sgd_preflight.py"),"--out",str(work/"episode512"),"--seed",str(a.seed),"--fold",str(a.fold),"--small-episode"]
        runpy.run_path(sys.argv[0],run_name="__main__")
    finally:
        sdm.models.KumoTabular=original;sys.argv=previous_argv
        (work/"checkpoint_activity.json").write_text(json.dumps(state,indent=2))
    report=json.loads((work/"episode512/report.json").read_text())
    report.update(wrapper_manifest_sha256=sha(work/"wrapper_manifest.json"),checkpoint_activity_sha256=sha(work/"checkpoint_activity.json"))
    # For the original seed/fold compare the exact episode, excluding run hash.
    old=ROOT/"results/v55/sgd_all_preflight_small/episode.json"
    if a.seed==42 and a.fold==0 and old.exists():
        before=json.loads(old.read_text());after=json.loads((work/"episode512/episode.json").read_text())
        before.pop("run_sha256");after.pop("run_sha256");assert before==after
        report["original_episode_byte_values_equal"]=True
    (work/"report.json").write_text(json.dumps(report,indent=2))
    fig,ax=plt.subplots(figsize=(8,3));ax.bar(["old forward OOM","checkpointed peak"],[5.148706912994385,report["peak_allocated_gib"]])
    ax.set(ylabel="Peak allocated GiB",title=f"Recomputation engineering: {report['status']}; no KS")
    fig.tight_layout();fig.savefig(ROOT/"eda/out"/f"195_{work.name}.png",dpi=130);plt.close(fig)
    print("Checkpoint engineering",report["status"],"steps",len(report["steps"]),"peak GiB",report["peak_allocated_gib"],"no quality evaluation",flush=True)


if __name__=="__main__":main()
