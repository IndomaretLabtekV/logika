"""Deep Feature Synthesis over the LOGIKA relational graph: the feature half of the TabPFN-Rel recipe.

TabPFN-Rel itself cannot be installed on Kaggle (it requires Python < 3.13). Its model is DFS + TabPFN-3.5, so the
DFS part is rebuilt here natively. Rules follow fastdfs: every child row must be dated strictly before the
application's own cutoff, and aggregation primitives are generic (no hand-chosen windows or ratios).

Depth 2: customer <- child table, primitives count / mean / std / min / max / sum / skew / last per numeric column,
num_unique and value shares per categorical, days since first and last row.
Depth 3: child-of-child is aggregated to its parent first (installment -> loan, item -> order,
bureau report -> bureau pull), then the parent-level statistics are aggregated to the customer."""
import polars as pl

DAY = 86400.0
NUM_AGGS = ("mean", "std", "min", "max", "sum", "skew", "last")


def _num(cols, by_time, tie):
    out = []
    for c in cols:
        x = pl.col(c).cast(pl.Float64)
        out += [x.mean().alias(f"{c}__mean"), x.std().alias(f"{c}__std"), x.min().alias(f"{c}__min"),
                x.max().alias(f"{c}__max"), x.sum().alias(f"{c}__sum"), x.skew().alias(f"{c}__skew"),
                x.sort_by(by_time, tie).last().alias(f"{c}__last")]
    return out


def _cat(col, values):
    return [pl.col(col).n_unique().alias(f"{col}__nunique")] + [
        (pl.col(col) == v).mean().alias(f"{col}__share_{v}") for v in values]


def _agg(df, key, prefix, nums, cats, time_col, tie):
    """tie: a unique row id that breaks equal timestamps, so `last` is deterministic."""
    exprs = [pl.len().alias("count")] + _num(nums, time_col, tie)
    for c, vals in cats.items():
        exprs += _cat(c, vals)
    exprs += [(pl.col("cutoff") - pl.col(time_col)).dt.total_seconds().min().truediv(DAY).alias("days_since_last"),
              (pl.col("cutoff") - pl.col(time_col)).dt.total_seconds().max().truediv(DAY).alias("days_since_first")]
    g = df.group_by(key).agg(exprs)
    return g.rename({c: f"dfs_{prefix}__{c}" for c in g.columns if c != key})


def _second(df, key, prefix, cols):
    """Aggregate parent-level statistics (already one row per loan / order / pull) to the customer."""
    exprs = [pl.len().alias("n")]
    for c in cols:
        x = pl.col(c).cast(pl.Float64)
        exprs += [x.mean().alias(f"{c}__mean"), x.max().alias(f"{c}__max"), x.min().alias(f"{c}__min"), x.std().alias(f"{c}__std")]
    g = df.group_by(key).agg(exprs)
    return g.rename({c: f"dfs_{prefix}__{c}" for c in g.columns if c != key})


def build_dfs(tables: dict) -> pl.DataFrame:
    apps = pl.concat([tables["train_applications"], tables["test_applications"]], how="diagonal_relaxed")
    cut = apps.select("customer_id", cutoff=pl.col("decision_datetime"))
    before = lambda name, t: tables[name].join(cut, on="customer_id").filter(pl.col(t) < pl.col("cutoff"))  # noqa: E731

    prev = before("previous_applications", "decision_datetime")
    rep = before("repayment_history", "due_date")
    bur = before("bureau_observations", "available_datetime")
    acc = (tables["account_behavior"].join(cut, on="customer_id")
           .with_columns(avail=pl.col("month").dt.offset_by("1mo").cast(pl.Datetime("us")))
           .filter(pl.col("avail") < pl.col("cutoff")))
    items = before("ecommerce_transactions", "transaction_datetime")
    orders = items.group_by("order_id").agg(
        pl.col("customer_id", "cutoff", "transaction_datetime", "order_total", "payment_method", "order_status").first(),
        n_items=pl.len(), qty=pl.col("quantity").sum())

    parts = [
        _agg(prev, "customer_id", "prev",
             ["requested_amount", "term_months", "approved_amount", "annual_interest_rate", "scheduled_monthly_installment"],
             {"decision": ["approved", "rejected", "cancelled"]}, "decision_datetime", "application_id"),
        _agg(rep, "customer_id", "inst", ["amount_due", "amount_paid", "days_past_due", "max_days_past_due"], {}, "due_date", "installment_id"),
        _agg(acc, "customer_id", "acct",
             ["opening_balance", "closing_balance", "inflows", "outflows", "income_like_credits", "average_balance",
              "minimum_balance", "low_balance_days"], {}, "avail", "month"),
        _agg(bur, "customer_id", "bureau",
             ["score", "active_accounts", "outstanding_balance", "revolving_credit_limit", "revolving_outstanding_balance",
              "utilization", "inquiries_last_6m", "delinquent_accounts_last_12m"],
             {"bureau_provider": ["Bureau_A", "Bureau_B", "Bureau_C"]}, "available_datetime", "bureau_observation_id"),
        _agg(orders, "customer_id", "order", ["order_total", "n_items", "qty"],
             {"payment_method": ["COD", "e_wallet", "bank_transfer", "debit_card", "credit_card"],
              "order_status": ["completed", "cancelled"]}, "transaction_datetime", "order_id"),
        _agg(items, "customer_id", "item", ["quantity", "item_total"],
             {"product_category": ["groceries", "fashion", "electronics", "home", "health", "education", "personal_care"]},
             "transaction_datetime", "order_item_id"),
    ]
    # depth 3: installment -> loan -> customer, report -> bureau pull -> customer
    loan = rep.group_by("loan_id").agg(
        pl.col("customer_id").first(), n_inst=pl.len(), dpd_mean=pl.col("max_days_past_due").mean(),
        dpd_max=pl.col("max_days_past_due").max(), paid_mean=pl.col("amount_paid").mean(),
        due_mean=pl.col("amount_due").mean())
    parts.append(_second(loan, "customer_id", "loan", ["n_inst", "dpd_mean", "dpd_max", "paid_mean", "due_mean"]))
    pull = bur.group_by("customer_id", "application_id").agg(
        score_mean=pl.col("score").mean(), outstanding_mean=pl.col("outstanding_balance").mean(),
        inquiries_mean=pl.col("inquiries_last_6m").mean(), n_reports=pl.len())
    parts.append(_second(pull, "customer_id", "pull", ["score_mean", "outstanding_mean", "inquiries_mean", "n_reports"]))

    out = apps.select("application_id", "customer_id")
    for p in parts:
        out = out.join(p, on="customer_id", how="left")
    # DFS also keeps the anchor row itself plus calendar transforms of the cutoff
    anchor = apps.join(tables["customer_profiles"], on="customer_id", how="left").select(
        "customer_id", "loan_amount", "annual_interest_rate", "proposed_monthly_installment", "age",
        "employment_tenure_months", "declared_monthly_income", "dependents",
        "loan_purpose", "region", "employment_type", "housing_status",
        dfs_cut_month=pl.col("decision_datetime").dt.month(), dfs_cut_dow=pl.col("decision_datetime").dt.weekday(),
        dfs_cut_hour=pl.col("decision_datetime").dt.hour())
    return out.join(anchor, on="customer_id", how="left")
