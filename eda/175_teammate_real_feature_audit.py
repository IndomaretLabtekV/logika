"""Independent real-data checks of rebuilt references, IDs, cutoffs and examples.

CPU-only, schedule as the sole heavy job. Uses no validation scores. Polars
recomputes V5 category reference and a few customer summaries from raw events;
the original builders use pandas. This is a feature audit, not quality evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from common import ROOT,OUT,PQ


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--tag",default="inner_s42_f0")
    a=ap.parse_args();work=a.cache.resolve()
    manifest=json.loads((work/"manifest.json").read_text())
    assert manifest["status"]=="complete"
    output=work/manifest["output"];assert sha(output)==manifest["output_sha256"]
    f=pl.read_parquet(output);fit=manifest["inputs"]["fitting_application_ids"]
    assert f.height==f["application_id"].n_unique()==f["customer_id"].n_unique()==100000
    canonical=pl.read_parquet(PQ/"features.parquet",columns=["application_id","target","is_train"])
    assert set(f["application_id"])==set(canonical["application_id"])
    assert set(f.filter(pl.col("split")=="train")["application_id"])==set(fit)
    assert f.filter(pl.col("split")!="train")["target"].null_count()==100000-len(fit)
    joined=f.filter(pl.col("split")=="train").join(canonical,on="application_id",validate="1:1",suffix="_canonical")
    assert np.array_equal(joined["target"].to_numpy(),joined["target_canonical"].to_numpy())
    schema=json.loads((work/"feature_schema.json").read_text())
    numeric=f.select(schema["numeric"]).to_numpy()
    assert not np.isinf(numeric).any()
    missing=f.select([pl.col(c).is_null().mean().alias(c) for c in schema["numeric"]]).row(0,named=True)
    examples=[]
    for label in (0,1):
        examples.extend(f.filter((pl.col("split")=="train")&(pl.col("target")==label)).sort("application_id").head(2)["application_id"].to_list())
    examples.extend(f.filter(pl.col("split")=="test").sort("application_id").head(2)["application_id"].to_list())
    small=f.filter(pl.col("application_id").is_in(examples))
    e=(pl.read_csv(ROOT/"data/ecommerce_transactions.csv",columns=["customer_id","product_category","quantity","item_total","transaction_datetime"],try_parse_dates=True)
       .join(f.select("customer_id","split","decision_datetime"),on="customer_id",validate="m:1"))
    future=int(e.select((pl.col("transaction_datetime")>=pl.col("decision_datetime")).sum()).item())
    assert future==0
    e=e.filter(pl.col("quantity")>0).with_columns((pl.col("item_total")/pl.col("quantity")).log1p().alias("log_price"))
    ref=e.filter(pl.col("split")=="train").group_by("product_category").agg(pl.col("log_price").median().alias("reference"))
    original=json.loads((work/"artifacts/optimization/v5/feature_audit.json").read_text())["category_log_price_reference"]
    ref_values=dict(zip(ref["product_category"],ref["reference"]))
    assert set(ref_values)==set(original)
    max_reference_error=max(abs(ref_values[key]-original[key]) for key in original)
    assert max_reference_error<1e-10
    e=e.join(ref,on="product_category",validate="m:1").with_columns((pl.col("log_price")-pl.col("reference")).alias("relative"))
    summaries=(e.filter(pl.col("customer_id").is_in(small["customer_id"].to_list())).group_by("customer_id")
               .agg(pl.col("relative").mean().alias("mean"),pl.col("relative").median().alias("median"),
                    pl.col("relative").std(ddof=1).alias("std"),pl.col("relative").count().alias("count")))
    checked=small.join(summaries,on="customer_id",how="left",validate="1:1")
    rows=[]
    for row in checked.iter_rows(named=True):
        for name in ("mean","median","std","count"):
            observed=row[f"v5_price_pooled_all_{name}"];expected=row[name]
            if expected is None or observed is None:
                assert expected is None and observed is None
                error=None
            else:
                error=abs(float(observed)-float(expected));assert error<1e-10
            rows.append({"application_id":row["application_id"],"split":row["split"],"target":row["target"],
                         "feature":f"v5_price_pooled_all_{name}","rebuilt":observed,"independent":expected,"abs_error":error})
    prefix=OUT/f"175_{a.tag}"
    pd.DataFrame(rows).to_csv(prefix.with_suffix(".csv"),index=False)
    quality=json.loads((work/"artifacts/quality_audit.json").read_text())
    cutoff_checks={f"{table}.{key}":value for table,items in quality.items() for key,value in items.items()
                   if key.endswith("at_or_after_cutoff") or key=="incomplete_month_at_cutoff"}
    assert len(cutoff_checks)>=8 and all(value==0 for value in cutoff_checks.values())
    report={"cache_manifest_sha256":sha(work/"manifest.json"),"feature_output_sha256":sha(output),"source_sha256":sha(Path(__file__).resolve()),
            "rows":f.height,"unique_customers":f["customer_id"].n_unique(),"numeric_features":len(schema["numeric"]),
            "fitting_rows":len(fit),"erased_query_targets":100000-len(fit),"categorical_features":schema["categorical"],
            "dtype_counts":pd.Series([str(dtype) for dtype in f.schema.values()]).value_counts().to_dict(),
            "missing_numeric_cells":int(np.isnan(numeric).sum()),"max_numeric_missing_fraction":max(missing.values()),
            "zero_missing_numeric_features":sum(value==0 for value in missing.values()),
            "cutoff_checks":cutoff_checks,"independent_future_ecommerce_rows":future,
            "reference_categories":len(ref_values),"max_reference_abs_error":max_reference_error,
            "example_applications":len(examples),"independent_summary_checks":len(rows),
            "max_example_abs_error":max(row["abs_error"] or 0 for row in rows),
            "model_executed":False,"quality_scores_used":False,
            "limits":["Examples verify V5 price summaries, not every numerical feature formula.",
                      "Cutoff counters for other tables come from the preserved source builders.",
                      "Customer-history references are fit-only, not expanding-time cross-customer references."]}
    prefix.with_suffix(".json").write_text(json.dumps(report,indent=2,allow_nan=False))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].bar(list(ref_values),list(ref_values.values()));axes[0].tick_params(axis="x",rotation=45)
    axes[0].set(ylabel="Median log1p(unit price)",title="Fit-only V5 references; independent recomputation")
    sample=[r for r in rows if r["feature"].endswith("_mean") and r["rebuilt"] is not None]
    axes[1].scatter([r["independent"] for r in sample],[r["rebuilt"] for r in sample])
    if sample:
        lo=min(r["independent"] for r in sample);hi=max(r["independent"] for r in sample)
        axes[1].plot([lo,hi],[lo,hi],"k:")
    axes[1].set(xlabel="Independent raw-event mean",ylabel="Rebuilt feature",title="Six context/query examples; no quality scores")
    fig.tight_layout();fig.savefig(prefix.with_suffix(".png"),dpi=130);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
