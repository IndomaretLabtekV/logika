"""Historical-loan snapshots: extra labelled training rows from loans the customers took BEFORE the selected one.

A previous approved loan whose 12-month window ended before the customer's selected decision has an observed outcome:
any installment reaching 90 DPD within 12 months of origination (same definition as the competition target).
Its features are rebuilt with build_all() AS OF that loan's decision time T:
  - every dated table is cut at T by the builders themselves (bureau availability, complete account months, orders,
    previous decisions);
  - repayment_history is a snapshot taken at the SELECTED cutoff, so installments due before T but paid after T are
    reset to their state at T (unpaid, DPD = floor(T - due), payment fields blank);
  - the snapshot loan itself is removed from previous_applications;
  - age and employment tenure (recorded at the selected decision) are moved back to T.
No competition label is used. Test customers' history is labelled the same way, which is allowed: it is provided data
dated before their own cutoff."""
import polars as pl

DAY_US = 86_400_000_000


def hist_labels(tables: dict) -> pl.DataFrame:
    """One row per previous approved loan with a complete 12-month window before the customer's selected cutoff."""
    apps = pl.concat([tables["train_applications"].drop("target"), tables["test_applications"]], how="diagonal_relaxed")
    cut = apps.select("customer_id", sel_cutoff=pl.col("decision_datetime"))
    loans = (tables["previous_applications"].filter(pl.col("loan_id").is_not_null()).join(cut, on="customer_id")
             .filter(pl.col("decision_datetime").dt.offset_by("12mo") <= pl.col("sel_cutoff")))
    r = tables["repayment_history"].join(loans.select("loan_id", orig="decision_datetime"), on="loan_id")
    # an installment reaches 90 DPD on due + 90 days; it counts if that day is inside the 12-month window
    r = r.with_columns(hit=(pl.col("max_days_past_due") >= 90)
                       & (pl.col("due_date") + pl.duration(days=90) <= pl.col("orig").dt.offset_by("12mo")))
    lab = r.group_by("loan_id").agg(hist_target=pl.col("hit").any().cast(pl.Int64), n_inst=pl.len())
    return loans.join(lab, on="loan_id", how="inner").sort("customer_id", "decision_datetime", "application_id")


def snapshot_tables(tables: dict, snap: pl.DataFrame) -> dict:
    """Tables as they were at each customer's snapshot time T. snap: one row per customer (customer_id, T,
    snap_app, sel_cutoff). Customers outside snap are dropped."""
    T = snap.select("customer_id", "T")
    r = tables["repayment_history"].join(T, on="customer_id").filter(pl.col("due_date") <= pl.col("T"))
    settled = (pl.col("amount_paid") >= pl.col("amount_due")) & (pl.col("last_payment_datetime") <= pl.col("T"))
    after = pl.col("last_payment_datetime") > pl.col("T")
    dpd = ((pl.col("T") - pl.col("due_date")).dt.total_microseconds() // DAY_US).clip(lower_bound=0).cast(pl.Int64)
    r = r.with_columns(
        amount_paid=pl.when(after).then(0).otherwise(pl.col("amount_paid")),
        last_payment_datetime=pl.when(after).then(None).otherwise(pl.col("last_payment_datetime")),
        days_past_due=pl.when(settled).then(pl.col("days_past_due")).otherwise(dpd),
        max_days_past_due=pl.when(settled).then(pl.col("max_days_past_due")).otherwise(dpd),
    ).drop("T")
    years = ((pl.col("sel_cutoff") - pl.col("T")).dt.total_days() / 365.25).floor().cast(pl.Int64)
    months = ((pl.col("sel_cutoff") - pl.col("T")).dt.total_days() / 30.44).floor().cast(pl.Int64)
    prof = tables["customer_profiles"].join(snap.select("customer_id", "T", "sel_cutoff"), on="customer_id").with_columns(
        age=pl.col("age") - years,
        employment_tenure_months=(pl.col("employment_tenure_months") - months).clip(lower_bound=0),
    ).drop("T", "sel_cutoff")
    keep = snap.select("customer_id")
    out = {k: v.join(keep, on="customer_id", how="semi") for k, v in tables.items()
           if k not in ("train_applications", "test_applications")}
    out["repayment_history"] = r
    out["customer_profiles"] = prof
    out["previous_applications"] = out["previous_applications"].join(
        snap.select(application_id="snap_app"), on="application_id", how="anti")
    return out


def snapshot_apps(lab: pl.DataFrame, schema: dict) -> pl.DataFrame:
    """Historical loans in the train_applications layout (approved amount and schedule; purpose unknown)."""
    return lab.select(
        "application_id", "customer_id", "application_datetime", "decision_datetime",
        loan_amount="approved_amount", term_months="term_months", annual_interest_rate="annual_interest_rate",
        proposed_monthly_installment="scheduled_monthly_installment",
        loan_purpose=pl.lit(None, pl.String), target="hist_target",
    ).cast({k: v for k, v in schema.items() if k != "loan_purpose"})


def build_hist(tables: dict, build_all) -> pl.DataFrame:
    """Feature rows (same columns as build_all) for every labelled historical loan, plus hist_term and snap_order."""
    lab = hist_labels(tables).with_columns(k=pl.col("decision_datetime").rank("ordinal").over("customer_id"))
    schema = tables["train_applications"].schema
    parts = []
    for k in range(1, int(lab["k"].max()) + 1):
        lk = lab.filter(pl.col("k") == k)
        snap = lk.select("customer_id", "sel_cutoff", T="decision_datetime", snap_app="application_id")
        tk = snapshot_tables(tables, snap)
        tk["train_applications"] = snapshot_apps(lk, schema)
        tk["test_applications"] = tables["test_applications"].clear()
        parts.append(build_all(tk).join(lk.select("application_id", hist_term="term_months", snap_order="k"),
                                        on="application_id", how="left"))
    return pl.concat(parts, how="diagonal_relaxed")
