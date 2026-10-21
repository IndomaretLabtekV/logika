"""Compare paced full-context inference to an already completed baseline.

Uses128 saved outer-query IDs as numerical parity only. Does not compute KS,
select a recipe, change context/batching or write competition submissions.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import valid2 as v
from common import ROOT,PQ,OUT,features

spec=importlib.util.spec_from_file_location("paced_fixed_union",Path(__file__).with_name("189_fixed_kumo_union_rank.py"))
fixed=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixed)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();folder=a.out.resolve()
    if folder.exists() and any(folder.iterdir()):raise ValueError("Use fresh parity output")
    folder.mkdir(parents=True,exist_ok=True)
    old=ROOT/"results/v56/fixed_union_rank/s42_f0";manifest=json.loads((old/"manifest.json").read_text());audit=json.loads((old/"input_audit.json").read_text())
    for name,expected in manifest["sources_sha256"].items():assert fixed.sha(ROOT/"eda"/name)==expected
    assert fixed.sha(PQ/"features.parquet")==manifest["features_sha256"]
    assert fixed.base.checkpoint_identity()==manifest["checkpoint"]
    X,_,y,ids,_=v.frames(features());ids=np.asarray(ids.to_list());tr,va=v.outer(y,42)[0];queries=va[:128]
    previous=pl.read_parquet(ROOT/"results/v56/fixed_union_rank/s42_f0_outer.parquet")
    expected=pl.DataFrame({"application_id":ids[queries]}).join(previous,on="application_id",how="left",maintain_order="left")["baseline_raw"].to_numpy()
    request={"probe_source_sha256":fixed.sha(__file__),"pacing_source_sha256":fixed.sha(ROOT/"eda/inference_pacing.py"),
             "parent_manifest_sha256":fixed.sha(old/"manifest.json"),"context_rows":len(tr),"query_ids":ids[queries].tolist(),
             "estimators":4,"generator_seed":4200,"chunk":128,"gpu_memory_fraction":.7,
             "max_abs_tolerance":.0005,"mean_abs_tolerance":.0001,"quality_evaluated":False}
    fixed.save(folder/"manifest.json",request)
    runtime=fixed.PredictionCache(X,y,ids,folder,fixed.digest(request),42,0)
    # The operational entrypoint must install pacing before model construction.
    observed=runtime.predict("paced_primary",tr,{"parity":queries},audit["primary_columns"],{"parent":request["parent_manifest_sha256"]})["parity"]
    assert hasattr(runtime.model,"_logika_pacing"),"Missing pacing: stop before resuming validation"
    state=runtime.model._logika_pacing;difference=np.abs(observed.astype(float)-expected.astype(float))
    result={**request,"max_abs_error":float(difference.max()),"mean_abs_error":float(difference.mean()),
            "bit_exact":bool(np.array_equal(observed,expected)),"pacing_fit_blocks":state["fit_blocks"],"pacing_query_chunks":state["query_chunks"]}
    result["passed"]=bool(result["max_abs_error"]<=request["max_abs_tolerance"] and result["mean_abs_error"]<=request["mean_abs_tolerance"] and state["fit_blocks"]>0 and state["query_chunks"]>0)
    fixed.save(folder/"parity.json",result)
    fig,ax=plt.subplots(figsize=(7,3));ax.plot(difference);ax.axhline(request["max_abs_tolerance"],color="red",ls="--")
    ax.set(xlabel="Saved query position",ylabel="Absolute probability error",title="Paced full56k/e4 parity; no quality metric")
    fig.tight_layout();fig.savefig(OUT/"200_inference_pacing_probe.png",dpi=130);plt.close(fig)
    print("Paced full-context numerical parity",json.dumps(result),flush=True)
    if not result["passed"]:raise SystemExit("Parity failed; do not resume the validation recipe")


if __name__=="__main__":main()
