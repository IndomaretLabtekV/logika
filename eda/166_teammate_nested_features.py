"""Rebuild the full teammate feature pool with partition-specific references.

Feature-only CPU job: schedule after the current heavy job has terminated.
Raw CSVs are linked to the canonical data directory, not copied. Historical
artifacts and scripts remain untouched. Outputs are not OOF/model evidence.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

import valid2 as v
from teammate_feature_adapter import ROOT,RAW_TABLES,REFERENCE_SOURCES,prepare_workspace,source_namespace,run_reference_stages,feature_schema


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def save_json(path,value):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def fitting_ids(seed,fold,partition):
    source=ROOT/"data_pq/features.parquet"
    assert source.exists(),"Existing canonical feature/ID cache is required; do not implicitly rebuild it"
    meta=(pl.read_parquet(source,columns=["application_id","is_train","target"])
          .filter(pl.col("is_train")==1).sort("application_id"))
    assert meta.height==70000 and meta["application_id"].n_unique()==70000
    y=meta["target"].to_numpy().astype(int)
    tr,_=v.outer(y,seed)[fold]
    if partition=="inner":
        fi,_=v.inner(y[tr],seed,fold);rows=tr[fi]
    elif partition=="outer":rows=tr
    else:rows=np.arange(len(y))
    return meta["application_id"][rows].to_list()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--partition",choices=["inner","outer","all"],default="inner")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--fold",type=int,choices=range(5),default=0)
    a=ap.parse_args()
    work=a.out.resolve();mp=work/"manifest.json"
    if work.exists() and any(work.iterdir()) and not mp.exists():
        raise ValueError("Use a fresh experiment directory; refusing to adopt existing artifacts")
    fit=fitting_ids(a.seed,a.fold,a.partition)
    source_files=([ROOT/"eda/teammate_feature_adapter.py",Path(__file__).resolve(),ROOT/"eda/valid2.py",
                   ROOT/"artifacts/optimization/v5/confirmation_recipe_v5_tabm_weight1.json"]+
                  [ROOT/"scripts"/name for name in ("analyze.py","extend_features.py",*REFERENCE_SOURCES)])
    inputs={"partition":a.partition,"seed":a.seed,"fold":a.fold,"fitting_application_ids":fit,
            "source_sha256":{str(p.relative_to(ROOT)):sha(p) for p in source_files},
            "raw_sha256":{name:sha(ROOT/"data"/f"{name}.csv") for name in RAW_TABLES},
            "id_split_cache_sha256":sha(ROOT/"data_pq/features.parquet"),
            "packages":{p:importlib.metadata.version(p) for p in ("numpy","pandas","polars","pyarrow","scikit-learn")}}
    fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if mp.exists():
        manifest=json.loads(mp.read_text())
        assert manifest["inputs_sha256"]==fingerprint,"Inputs/source/packages changed; use a new experiment directory"
        if manifest["status"]=="complete":
            assert sha(work/manifest["output"] )==manifest["output_sha256"]
            print("Verified completed feature cache",work,flush=True);return
    else:
        manifest={"inputs":inputs,"inputs_sha256":fingerprint,"status":"running","stages":{},
                  "model_executed":False,"reference_population":"fitting partition customer histories only",
                  "label_policy":"erase every non-fitting target before V3/V4/V5 reference stages",
                  "historical_feature_formulas":"unchanged; original function bodies loaded from scripts/",
                  "limits":["Global historical90 column list is not treated as nested feature selection.",
                            "This cache supports new nested selection from the full feature pool.",
                            "No model-quality or public-score claim follows from a successful feature build."]}
    prepare_workspace(work,ROOT/"data")
    save_json(mp,manifest)
    t0=time.time()
    processed=work/"data/processed"
    base_path=processed/"application_features.parquet"
    if "base" in manifest["stages"]:
        assert sha(base_path)==manifest["stages"]["base"]["sha256"]
        assert sha(processed/"bureau_latest.parquet")==manifest["stages"]["base"]["bureau_latest_sha256"]
    else:
        start=time.time();ns=source_namespace("analyze.py",work)
        base=ns["build_features"]();del base,ns;gc.collect()
        manifest["stages"]["base"]={"seconds":time.time()-start,"sha256":sha(base_path),
                                   "bureau_latest_sha256":sha(processed/"bureau_latest.parquet")}
        save_json(mp,manifest)
    prefix_path=processed/"prefix_features_extended.parquet"
    if "extended" in manifest["stages"]:
        assert sha(prefix_path)==manifest["stages"]["extended"]["sha256"]
    else:
        start=time.time();ns=source_namespace("extend_features.py",work);ns["main"]();del ns;gc.collect()
        (processed/"application_features_extended.parquet").replace(prefix_path)
        manifest["stages"]["extended"]={"seconds":time.time()-start,"sha256":sha(prefix_path)}
        save_json(mp,manifest)
    extended=pd.read_parquet(prefix_path)
    start=time.time();frame=run_reference_stages(work,extended,fit);del extended;gc.collect()
    assert len(frame)==100000 and frame.application_id.is_unique and frame.customer_id.is_unique
    assert set(frame.loc[frame.split.eq("train"),"application_id"])==set(fit)
    assert frame.loc[frame.split.eq("test"),"target"].isna().all()
    official=pl.read_parquet(ROOT/"data_pq/features.parquet",columns=["application_id"])["application_id"].to_list()
    assert set(frame.application_id)==set(official)
    recipe=ROOT/"artifacts/optimization/v5/confirmation_recipe_v5_tabm_weight1.json"
    historic90=json.loads(recipe.read_text())["spec"]["columns"]
    assert len(historic90)==90 and set(historic90)<=set(frame.columns)
    schema=feature_schema(frame)
    columns=schema["numeric"]+schema["categorical"]
    numeric=frame[schema["numeric"]]
    assert set(v.CAT_COLS)<=set(schema["categorical"]),"Missing application category predictors"
    save_json(work/"feature_schema.json",schema)
    assert not np.isinf(numeric.to_numpy()).any()
    numeric.isna().mean().rename("missing_fraction").rename_axis("feature").to_csv(work/"feature_missingness.csv")
    frame.loc[frame.split.eq("train")].groupby("target",dropna=False).head(3).to_csv(work/"fitting_examples.csv",index=False)
    # Preserve the historical90 recipe as metadata; the next model may select from
    # the full pool only inside its fitting split, rather than reuse global labels.
    save_json(work/"historical90_columns.json",{"columns":historic90,"recipe_sha256":sha(recipe),
               "role":"reproduction reference; not a newly validated selector"})
    output=processed/"application_features_v5.parquet"
    manifest.update(status="complete",output=str(output.relative_to(work)),output_sha256=sha(output),
                    rows=len(frame),feature_count=len(columns),numeric_features=numeric.shape[1],
                    categorical_features=schema["categorical"],
                    fitting_rows=len(fit),unlabeled_query_rows=int(frame.split.eq("test").sum()),
                    historical90_columns_present=True,reference_stage_seconds=time.time()-start,
                    elapsed_seconds=time.time()-t0,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(["fitting","query"],[len(fit),len(frame)-len(fit)]);axes[0].set(ylabel="Applications",title="Reference fitting membership (query labels erased)")
    axes[1].hist(numeric.isna().mean(),bins=20);axes[1].set(xlabel="Missing fraction",ylabel="Features",title="Full teammate numeric feature pool")
    fig.tight_layout();fig.savefig(ROOT/"eda/out"/f"166_{work.name}_{fingerprint[:8]}.png",dpi=130);plt.close(fig)
    save_json(mp,manifest)
    print(json.dumps({k:manifest[k] for k in ("rows","feature_count","numeric_features","fitting_rows","unlabeled_query_rows","elapsed_seconds","peak_rss_gib")},indent=2),flush=True)


if __name__=="__main__":main()
