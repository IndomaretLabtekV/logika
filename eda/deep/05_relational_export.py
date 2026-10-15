"""Export cutoff-safe table grains and schema for later remote relational experiments; no model imports."""
from deep_common import *
import gc

def main():
    dest=OUT/"relational";dest.mkdir(exist_ok=True)
    A=apps();cut=A.select("customer_id",cutoff="decision_datetime")
    schema={"status":"Audited intermediate schema, NOT an executable TabPFN-Rel task config.",
            "prediction_entity":"application_id","cutoff":"one decision_datetime per selected application",
            "warning":"Profile and repayment states are selected-cutoff snapshots, not historical row states. Do not assign due_date as snapshot availability.",
            "tables":{}}
    def export(name,df,pk,time=None,foreign=None):
        assert df.select(pk).unique().height==df.height
        assert "target" not in df.columns
        df.write_parquet(dest/(name+".parquet"))
        schema["tables"][name]={"rows":df.height,"primary_key":pk,"availability_column":time,
                               "foreign_keys":foreign or {},"columns":df.columns}
        print(name,df.shape)
    export("applications",A.drop("target"),["application_id"],"decision_datetime",{"customer_id":"customers.customer_id"})
    export("customers",table("customer_profiles").join(cut,on="customer_id",validate="1:1").rename({"cutoff":"snapshot_at"}),
           ["customer_id"],"snapshot_at")
    prev=table("previous_applications").join(cut,on="customer_id",validate="m:1").filter(pl.col("decision_datetime")<=pl.col("cutoff")).drop("cutoff")
    export("previous_applications",prev,["application_id"],"decision_datetime",{"customer_id":"customers.customer_id"})
    loans=prev.filter(pl.col("loan_id").is_not_null()).select("loan_id","customer_id","application_id","decision_datetime","term_months","scheduled_monthly_installment")
    export("loans",loans,["loan_id"],"decision_datetime",{"application_id":"previous_applications.application_id","customer_id":"customers.customer_id"})
    r=table("repayment_history").join(cut,on="customer_id",validate="m:1").filter(pl.col("due_date")<=pl.col("cutoff")).rename({"cutoff":"snapshot_at"})
    export("repayment_snapshots",r,["installment_id"],"snapshot_at",{"loan_id":"loans.loan_id","customer_id":"customers.customer_id"})
    b=table("bureau_observations").join(cut,on="customer_id",validate="m:1").filter(pl.col("available_datetime")<=pl.col("cutoff")).drop("cutoff")
    # Historical and selected application IDs share one real namespace; preserve it in an explicit key table.
    parent=pl.concat([A.select("application_id","customer_id","decision_datetime"),prev.select("application_id","customer_id","decision_datetime")])
    export("all_application_keys",parent,["application_id"],"decision_datetime",{"customer_id":"customers.customer_id"})
    export("bureau",b,["bureau_observation_id"],"available_datetime",{"application_id":"all_application_keys.application_id","customer_id":"customers.customer_id"})
    ac=table("account_behavior").join(cut,on="customer_id",validate="m:1").with_columns(
        available_at=pl.col("month").dt.offset_by("1mo").cast(pl.Datetime))
    ac=ac.filter(pl.col("available_at")<=pl.col("cutoff")).drop("cutoff").with_columns(
        account_month_id=pl.col("customer_id")+"_"+pl.col("month").cast(pl.String))
    export("account_months",ac,["account_month_id"],"available_at",{"customer_id":"customers.customer_id"})
    e=table("ecommerce_transactions").join(cut,on="customer_id",validate="m:1").filter(pl.col("transaction_datetime")<=pl.col("cutoff")).drop("cutoff")
    fields=["customer_id","transaction_datetime","order_total","payment_method","order_status"]
    orders=e.group_by("order_id").agg(pl.col(fields).first())
    export("orders",orders,["order_id"],"transaction_datetime",{"customer_id":"customers.customer_id"})
    items=e.select("order_item_id","order_id","transaction_datetime","product_category","quantity","item_total")
    export("order_items",items,["order_item_id"],"transaction_datetime",{"order_id":"orders.order_id"})
    assert items["item_total"].sum()==orders["order_total"].sum()
    # Check every exported relationship, not just the largest tables.
    for name,spec in schema["tables"].items():
        for fk,parent in spec["foreign_keys"].items():
            pn,pk=parent.split(".")
            child=pl.read_parquet(dest/(name+".parquet"),columns=[fk]).drop_nulls().unique()
            par=pl.read_parquet(dest/(pn+".parquet"),columns=[pk]).unique()
            assert child.join(par,left_on=fk,right_on=pk,how="anti").height==0,(name,fk)
    (dest/"schema.json").write_text(json.dumps(schema,indent=2)+"\n")
    table("train_applications").select("application_id","target").write_parquet(dest/"labels_train_only.parquet")
    table("test_applications").select("application_id","decision_datetime").write_parquet(dest/"test_queries.parquet")
    save_json({"entities":A.height,"tables":len(schema["tables"]),"foreign_keys_checked":sum(len(s["foreign_keys"]) for s in schema["tables"].values()),
               "target_columns_in_predictor_tables":0,"order_item_money_conserved":True,
               "not_tested":"TabPFN-Rel integration, model inference and per-query temporal masking; require remote validation.",
               "sources":["https://github.com/PriorLabs/tabpfn-rel","https://github.com/PriorLabs/relarena/blob/main/docs/predictive-task.md"]},"05_relational_checks")
    fig,ax=plt.subplots(figsize=(13,6));ax.axis("off")
    nodes={"customers":(.1,.6),"applications":(.35,.88),"previous_applications":(.35,.63),"loans":(.60,.63),
           "repayment_snapshots":(.86,.63),"bureau":(.60,.88),"account_months":(.35,.36),
           "orders":(.60,.14),"order_items":(.86,.14)}
    edges=[("customers","applications"),("customers","previous_applications"),("previous_applications","loans"),
           ("loans","repayment_snapshots"),("applications","bureau"),("previous_applications","bureau"),
           ("customers","account_months"),("customers","orders"),("orders","order_items")]
    for u,v in edges:ax.annotate("",xy=nodes[v],xytext=nodes[u],arrowprops=dict(arrowstyle="->",color="grey"))
    for n,(x,y) in nodes.items():ax.text(x,y,n,ha="center",va="center",bbox=dict(boxstyle="round",facecolor="#d8eaf0"),fontsize=9)
    ax.set_title("Logical relationships: preserve table grain and per-application cutoff")
    figsave(fig,"05_relational_schema")

if __name__=="__main__":
    self_check()
    main()
