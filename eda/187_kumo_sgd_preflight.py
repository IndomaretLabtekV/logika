"""Two full-gradient SGD steps on inner-fit only; engineering, no validation KS."""
import argparse
import gc
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import resource
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sdm
import torch

import valid2 as v
from common import ROOT,PQ,OUT,features
from nested_input_pools import prepare_stage,select_columns

spec=importlib.util.spec_from_file_location("sgd_base",Path(__file__).with_name("173_nested_kumo_inputs.py"))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
sha,digest,save=base.file_hash,base.digest,base.save_json
vip_spec=importlib.util.spec_from_file_location("sgd_vip",Path(__file__).with_name("152_vipcop_kumo.py"))
vip=importlib.util.module_from_spec(vip_spec);vip_spec.loader.exec_module(vip)


def positive_probability(output):
    columns=output.columns[sdm.Stype.numerical]
    if "1" not in columns:raise ValueError("Positive class column missing")
    return output.numerical[...,columns.index("1")].reshape(-1)


def group_name(name):
    return "row_embedding" if "row_embedding" in name else "icl_block" if "icl_block" in name else "other"


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--fold",type=int,choices=range(5),default=0)
    ap.add_argument("--small-episode",action="store_true",help="Separate engineering amendment: context512/query64")
    a=ap.parse_args();work=a.out.resolve();work.mkdir(parents=True,exist_ok=True)
    mp=work/"manifest.json";rp=work/"report.json"
    assert (PQ/"features.parquet").exists()
    n_context,n_query=(512,64) if a.small_episode else (1024,128)
    config={"seed":a.seed,"fold":a.fold,"steps":2,"episode_context":n_context,"episode_query":n_query,
            "optimizer":"SGD","lr":.01,"momentum":0,"weight_decay":0,"foreach":False,
            "amp":"float16","grad_scaler_initial_scale":128.,"clip_norm":1.,"estimators":1,"gpu_memory_fraction":.7}
    manifest={"config":config,"sources_sha256":{name:sha(ROOT/"eda"/name) for name in
              (Path(__file__).name,"173_nested_kumo_inputs.py","152_vipcop_kumo.py","nested_input_pools.py","valid2.py","common.py")},
              "features_sha256":sha(PQ/"features.parquet"),"checkpoint":base.checkpoint_identity(),
              "packages":{p:importlib.metadata.version(p) for p in ("numpy","pandas","polars","torch","structured-data-models","scikit-learn")},
              "protocol_sha256":sha(ROOT/"docs/integration/protokol_kumo_sgd_preflight.md"),
              "small_episode_protocol_sha256":sha(ROOT/"docs/integration/protokol_kumo_sgd_small_episode.md") if a.small_episode else None,
              "model_quality_evaluated":False,"heldout_labels_used_for_training":False}
    if mp.exists():
        assert json.loads(mp.read_text())==manifest,"Changed source/configuration; use a fresh engineering directory"
        if rp.exists() and json.loads(rp.read_text())["status"] in {"passed","oom","nonfinite_gradient"}:
            print("Existing terminal preflight",rp,flush=True);return
        raise ValueError("Interrupted engineering preflight; inspect its process/log, then use a fresh directory")
    if any(work.iterdir()):raise ValueError("Use fresh owned output")
    save(mp,manifest);run_hash=digest(manifest)
    X,_,y,train_ids,_=v.frames(features());ids=np.asarray(train_ids.to_list())
    tr,va=v.outer(y,a.seed)[a.fold];fi,stop=v.inner(y[tr],a.seed,a.fold);fi,stop=tr[fi],tr[stop]
    cols=select_columns(X,y,fi,extend_shortlist=True);assert len(cols)==104 and len(fi)==48000
    rng=np.random.default_rng(100*a.seed+a.fold+911)
    context=vip.strat(fi,y,n_context,rng)
    query=np.sort(rng.choice(np.setdiff1d(fi,context),n_query,replace=False))
    assert set(context)<=set(fi) and set(query)<=set(fi) and set(context).isdisjoint(query)
    assert set(np.r_[context,query]).isdisjoint(np.r_[stop,va])
    # Vocabulary sees inner-fit only; GPU tensors contain only the training episode.
    F=vip.prepare(X.iloc[fi][cols].reset_index(drop=True),np.arange(len(fi)),"raw")
    local={row:i for i,row in enumerate(fi)};positions=np.array([local[row] for row in np.r_[context,query]])
    F=F.iloc[positions].reset_index(drop=True)
    save(work/"episode.json",{"run_sha256":run_hash,"feature_fitting_ids":ids[fi].tolist(),"columns":cols,
                             "training_context_ids":ids[context].tolist(),"training_context_labels":y[context].tolist(),
                             "training_query_ids":ids[query].tolist(),"training_query_labels":y[query].tolist(),
                             "heldout_search_confirmation_outer_ids_in_episode":False})
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.7)
    dev=torch.device("cuda");start=time.monotonic();phase="tensor-construction";model=None;optimizer=None
    report={"run_sha256":run_hash,"status":"running","phase":phase,"steps":[],"model_quality_evaluated":False,
            "heldout_labels_used":False,"episode_query_labels_are_training_labels":True,"weights_saved":False}
    save(rp,report)
    try:
        tx=sdm.TableTensor.from_pandas(F,stypes={c:"numerical" for c in cols},device=dev)
        labels=np.r_[y[context],y[query]]
        ty=sdm.TableTensor.from_pandas(pd.DataFrame({"target":labels.astype(str)}),stypes={"target":"categorical"},device=dev)
        ci=torch.arange(len(context),device=dev);qi=torch.arange(len(context),len(F),device=dev)
        phase="model-load";report["phase"]=phase;save(rp,report)
        model=sdm.models.KumoTabular(task="classification",size="large",device=dev);model.train()
        for p in model.parameters():p.requires_grad_(True)
        named=list(model.named_parameters());params=[p for _,p in named]
        counts={group:sum(p.numel() for name,p in named if group_name(name)==group) for group in ("row_embedding","icl_block","other")}
        assert counts["row_embedding"]>0 and counts["icl_block"]>0 and all(p.requires_grad for p in params)
        snapshot_name,snapshot_param=next((name,p) for name,p in named if group_name(name)=="row_embedding")
        before=snapshot_param.detach().float().cpu().clone()
        optimizer=torch.optim.SGD(params,lr=.01,momentum=0,weight_decay=0,foreach=False)
        scaler=torch.amp.GradScaler("cuda",init_scale=128.)
        report.update(parameters=sum(counts.values()),parameter_groups=counts,snapshot_parameter=snapshot_name,
                      training_context_rows=len(context),training_query_rows=len(query),all_parameters_trainable=True)
        torch.cuda.reset_peak_memory_stats()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            phase="forward";report.update(phase=phase,current_step=step);save(rp,report)
            gen=torch.Generator(device=dev).manual_seed(100*a.seed+a.fold)
            with torch.amp.autocast("cuda",dtype=torch.float16):
                output=model(x_context=tx[ci],y_context=ty[ci],x_query=tx[qi],num_estimators=1,generator=gen)
            probability=positive_probability(output).float().clamp(1e-6,1-1e-6)
            loss=torch.nn.functional.binary_cross_entropy(probability,torch.as_tensor(y[query],device=dev,dtype=torch.float32))
            assert torch.isfinite(loss),"Nonfinite training loss"
            phase="backward";report["phase"]=phase;save(rp,report)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            phase="gradient-check"
            if not all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None):
                report.update(status="nonfinite_gradient",phase=phase);break
            grad_norms={};grad_counts={}
            for group in counts:
                tensors=[p.grad for name,p in named if group_name(name)==group and p.grad is not None]
                grad_counts[group]=sum(t.numel() for t in tensors)
                grad_norms[group]=float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(t.float()) for t in tensors])).item()) if tensors else 0.
            assert grad_norms["row_embedding"]>0 and grad_norms["icl_block"]>0,"Missing gradient in an intended training module"
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            phase="optimizer-step";report["phase"]=phase;save(rp,report)
            scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
            row={"step":step+1,"training_loss":float(loss.item()),"gradient_norms_before_clip":grad_norms,
                 "parameters_with_gradient":grad_counts,"global_norm_before_clip":float(norm.item()),
                 "grad_scale":float(scaler.get_scale()),"optimizer_state_entries":len(optimizer.state),
                 "allocated_gib":torch.cuda.memory_allocated()/2**30,"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
                 "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30}
            report["steps"].append(row);save(rp,report);print("engineering training step",json.dumps(row),flush=True)
        if report["status"]=="running":
            delta=float((snapshot_param.detach().float().cpu()-before).abs().max().item())
            assert delta>0,"Row-embedding snapshot did not update"
            assert len(report["steps"])==2 and len(optimizer.state)==0
            report.update(status="passed",phase="complete",row_embedding_snapshot_max_update=delta,optimizer_state_entries=0)
    except torch.cuda.OutOfMemoryError as exc:
        report.update(status="oom",phase=phase,error=str(exc))
    except Exception as exc:
        report.update(status="error",phase=phase,error=f"{type(exc).__name__}: {exc}")
        save(rp,report);raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-start,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        save(rp,report)
        if optimizer is not None:optimizer.zero_grad(set_to_none=True)
        if model is not None:model.clear()
        gc.collect();torch.cuda.empty_cache()
    fig,axes=plt.subplots(1,2,figsize=(10,4));rows=report["steps"]
    if rows:
        axes[0].plot([row["step"] for row in rows],[row["training_loss"] for row in rows],"o-")
        for group in ("row_embedding","icl_block"):axes[1].plot([row["step"] for row in rows],[row["gradient_norms_before_clip"][group] for row in rows],"o-",label=group)
        axes[1].legend()
    axes[0].set(xlabel="SGD step",ylabel="Training episode BCE",title=f"Engineering {report['status']}; no validation metric")
    axes[1].set(xlabel="SGD step",ylabel="Gradient norm before clipping",title="Full parameter gradient flow")
    fig.tight_layout();fig.savefig(OUT/f"187_{work.name}_{run_hash[:8]}.png",dpi=130);plt.close(fig)
    print("Engineering report",json.dumps(report),flush=True)


if __name__=="__main__":main()
