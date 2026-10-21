"""Count train/test membership of historical price references; do not fit a model.

Audits the population used by existing normalization code. No means, standard
deviations or test-fitted features are estimated here. Counts alone establish
whether test customers contribute to the legacy reference population.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"eda/out"
PQ=ROOT/"data_pq"


def main():
    parts=[]
    for name in ("train","test"):
        frame=pl.read_parquet(PQ/f"{name}_applications.parquet",columns=["customer_id","decision_datetime"])
        parts.append(frame.with_columns(source=pl.lit(name)))
    apps=pl.concat(parts)
    assert apps["customer_id"].n_unique()==100000
    months=apps.select(decision_month=pl.col("decision_datetime").dt.truncate("1mo")).unique().sort("decision_month")
    # Aggregation is counts only; label target is never loaded.
    counts=(pl.scan_parquet(PQ/"ecommerce_transactions.parquet")
            .select("customer_id","transaction_datetime","product_category")
            .join(apps.lazy().select("customer_id","source"),on="customer_id",how="left")
            .group_by("source","product_category",transaction_month=pl.col("transaction_datetime").dt.truncate("1mo"))
            .agg(rows=pl.len()).collect())
    assert counts.filter(pl.col("source").is_null()).is_empty()
    references=(months.join(counts,how="cross")
                .filter(pl.col("transaction_month")<pl.col("decision_month"))
                .group_by("decision_month","product_category","source").agg(rows=pl.col("rows").sum())
                .sort("decision_month","product_category","source"))
    references.write_csv(OUT/"162_price_reference_counts.csv")
    monthly=(references.group_by("decision_month","source").agg(rows=pl.col("rows").sum())
             .pivot(on="source",index="decision_month",values="rows")
             .with_columns(test_fraction=pl.col("test")/(pl.col("train")+pl.col("test")))
             .sort("decision_month"))
    monthly.write_csv(OUT/"162_price_reference_monthly.csv")
    result={"train_customers":70000,"test_customers":30000,"source_event_rows":int(counts["rows"].sum()),
            "test_reference_fraction_min":monthly["test_fraction"].min(),
            "test_reference_fraction_max":monthly["test_fraction"].max(),
            "legacy_main_reference_includes_test_customers":True,
            "legacy_teammate_reference_population":"all training customers; not refit per outer fold",
            "target_loaded":False,"model_executed":False,"means_or_std_fit":False,
            "source_sha256":hashlib.sha256((ROOT/"eda/features.py").read_bytes()).hexdigest(),
            "evidence":"eda/features.py v5_features builds raw monthly reference before joining cutoff; no customer train filter",
            "decision":"Preserve legacy baseline; new candidate must fit reference only from its fitting partition and respect time cutoff.",
            "limitation":"Counting reference population does not quantify model-score optimism or prove a quality gain."}
    (OUT/"162_reference_population_audit.json").write_text(json.dumps(result,indent=2))
    print(monthly);print(json.dumps(result,indent=2))
    fig,ax=plt.subplots(figsize=(9,3))
    ax.plot([str(x)[:7] for x in monthly["decision_month"]],monthly["test_fraction"]*100,marker="o")
    ax.set(xlabel="Application decision month",ylabel="Test-customer events (%)",
           title="Legacy price-reference population (counts only, no feature fitting)")
    ax.tick_params(axis="x",rotation=45);fig.tight_layout()
    fig.savefig(OUT/"162_reference_population_audit.png",dpi=130);plt.close(fig)


if __name__=="__main__":main()
