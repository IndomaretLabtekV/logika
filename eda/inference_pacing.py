"""Pace existing cached Kumo inference without changing tensors or RNG draws.

Operational mitigation and durable stage tracing, not a claimed driver fix.
The caller must schedule it as the sole heavy job.
"""
from functools import wraps
import json
import os
from pathlib import Path
import re
import time

import torch


def pace_model(model,block_pause=.25,query_pause=.1,synchronize=None,sleep=None,trace=None):
    if hasattr(model,"_logika_pacing"):raise ValueError("Inference pacing already installed")
    if block_pause<0 or query_pause<0:raise ValueError("Pacing delays must be nonnegative")
    sync=torch.cuda.synchronize if synchronize is None else synchronize
    sleep=time.sleep if sleep is None else sleep
    pattern=re.compile(r"(?:^|\.)(?:row_embedding\.(?:cell_embedding|col_blocks\.\d+|row_blocks\.\d+)|icl_block\.layers\.\d+)$")
    selected=[(name,block) for name,block in model.named_modules() if pattern.search(name)]
    if not selected:raise ValueError("No expected Kumo inference blocks found")
    state={"modules":[name for name,_ in selected],"phase":"idle","fit_blocks":0,"query_chunks":0,
           "block_pause":block_pause,"query_pause":query_pause}
    def emit(event,**kwargs):
        if trace is not None:trace({"event":event,"monotonic_s":time.monotonic(),**kwargs})
    def before(name):
        def hook(module,args):
            if state["phase"]=="fit":emit("block_start",module=name)
        return hook
    def after(name):
        def hook(module,args,output):
            if state["phase"]=="fit":
                sync();state["fit_blocks"]+=1;emit("block_complete",module=name,count=state["fit_blocks"]);sleep(block_pause)
            # Returning None preserves the original forward output object.
        return hook
    for name,block in selected:
        block.register_forward_pre_hook(before(name));block.register_forward_hook(after(name))
    original_fit,original_predict=model.fit,model.predict
    @wraps(original_fit)
    def fit(*args,**kwargs):
        if state["phase"]!="idle":raise RuntimeError("Unexpected nested Kumo fit")
        state["phase"]="fit";emit("fit_start")
        try:
            result=original_fit(*args,**kwargs);sync();emit("fit_complete",blocks=state["fit_blocks"]);return result
        finally:state["phase"]="idle"
    @wraps(original_predict)
    def predict(*args,**kwargs):
        result=original_predict(*args,**kwargs);sync();state["query_chunks"]+=1
        emit("query_complete",count=state["query_chunks"]);sleep(query_pause);return result
    model.fit,model.predict=fit,predict;model._logika_pacing=state
    return state


def install_factory(trace_dir,block_pause=.25,query_pause=.1):
    import sdm
    directory=Path(trace_dir).resolve();directory.mkdir(parents=True,exist_ok=True)
    path=directory/f"pid{os.getpid()}_stages.jsonl"
    def trace(record):
        with path.open("a") as stream:
            stream.write(json.dumps(record)+"\n");stream.flush();os.fsync(stream.fileno())
    original=sdm.models.KumoTabular
    def factory(*args,**kwargs):
        model=original(*args,**kwargs)
        state=pace_model(model,block_pause,query_pause,trace=trace)
        from sdm.models.kumo.tabular.model import MODEL_KWARGS
        cfg=MODEL_KWARGS["large"];assert len(state["modules"])==1+2*cfg["num_embedding_layers"]+cfg["num_icl_layers"]
        trace({"event":"pacing_installed","pid":os.getpid(),"modules":state["modules"],"block_pause":block_pause,"query_pause":query_pause})
        print("Paced Kumo inference installed; durable stage trace",path,flush=True)
        return model
    sdm.models.KumoTabular=factory
    return path
