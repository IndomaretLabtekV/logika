"""Arithmetic identities and anomalies a data generator leaves behind: annuity math, balance identities,
partial/over payments, missing installments, approved<requested, round amounts, selected app inside previous apps.
Every anomaly is reported with its train bad rate (base 13.6%)."""
import numpy as np
import polars as pl
from common import load

tr = load("train_applications")
te = load("test_applications")
apps = pl.concat([tr.drop("target"), te])
y = tr.select("customer_id", "target")
prev, rep, acc, bur = (load(t) for t in ["previous_applications", "repayment_history", "account_behavior", "bureau_observations"])


def rate(df, flag, name):
    d = df.select("customer_id", f=flag).group_by("customer_id").agg(pl.col("f").max()).join(y, on="customer_id")
    g = d.group_by("f").agg(n=pl.len(), bad=pl.col("target").mean()).sort("f")
    print(f"{name:<55}", "  ".join(f"[{r['f']}] n={r['n']:,} bad={r['bad']:.3f}" for r in g.iter_rows(named=True)))


def annuity(p, rate_yr, n):
    r = rate_yr / 12
    return p * r / (1 - (1 + r) ** (-n))


print("=== 1) annuity math")
a = tr.with_columns(ann=annuity(pl.col("loan_amount"), pl.col("annual_interest_rate"), pl.col("term_months")))
a = a.with_columns(d=pl.col("proposed_monthly_installment") - pl.col("ann"))
print("selected apps: installment - annuity (min, median, max)", a.select(pl.col("d").min().alias("a"), pl.col("d").median().alias("b"), pl.col("d").max().alias("c")).row(0))
rate(a, (pl.col("d").abs() > 2).cast(pl.Int8), "selected installment != annuity (|d|>2 IDR)")
p2 = prev.filter(pl.col("decision") == "approved").with_columns(
    ann=annuity(pl.col("approved_amount"), pl.col("annual_interest_rate"), pl.col("term_months")))
p2 = p2.with_columns(d=pl.col("scheduled_monthly_installment") - pl.col("ann"))
print("previous approved: installment - annuity", p2.select(pl.col("d").min().alias("a"), pl.col("d").median().alias("b"), pl.col("d").max().alias("c")).row(0))
print("loan_amount granularity: mod 100k == 0:", (tr["loan_amount"] % 100000 == 0).mean())

print("\n=== 2) previous applications")
rate(prev, (pl.col("decision") == "approved") & (pl.col("approved_amount") < pl.col("requested_amount")), "any approved < requested")
print("selected application_id present in previous_applications:", apps["application_id"].is_in(prev["application_id"].implode()).sum())
print("decision mix:", prev["decision"].value_counts().sort("decision").rows())
print("prev term_months:", prev["term_months"].value_counts().sort("term_months").rows())
pa = prev.join(apps.select("customer_id", cut="decision_datetime"), on="customer_id")
print("prev apps decided after selected cutoff:", (pa["decision_datetime"] > pa["cut"]).sum())

print("\n=== 3) repayment")
print("paid > due:", (rep["amount_paid"] > rep["amount_due"]).sum(), " partial (0<paid<due):",
      ((rep["amount_paid"] > 0) & (rep["amount_paid"] < rep["amount_due"])).sum(), " zero paid:", (rep["amount_paid"] == 0).sum())
rate(rep, ((pl.col("amount_paid") > 0) & (pl.col("amount_paid") < pl.col("amount_due"))).cast(pl.Int8), "any partial payment")
rate(rep, (pl.col("last_payment_datetime") < pl.col("due_date")).cast(pl.Int8), "any payment before due date")
early = rep.with_columns(e=(pl.col("due_date") - pl.col("last_payment_datetime")).dt.total_days())
print("days paid early (paid rows) quantiles:", early.filter(pl.col("amount_paid") >= pl.col("amount_due"))
      .select([pl.col("e").quantile(q).alias(str(q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)]).row(0))
print("dpd>max_dpd rows:", (rep["days_past_due"] > rep["max_days_past_due"]).sum())
loans_rep = rep.group_by("loan_id").agg(n=pl.len(), first_due=pl.col("due_date").min(), cust=pl.col("customer_id").first())
lp = loans_rep.join(prev.select("loan_id", "term_months", "decision_datetime"), on="loan_id", how="left")
print("repayment loans not in previous_applications:", lp["term_months"].is_null().sum())
appr = prev.filter(pl.col("decision") == "approved").join(apps.select("customer_id", cut="decision_datetime"), on="customer_id")
appr = appr.with_columns(exp_n=((pl.col("cut") - pl.col("decision_datetime")).dt.total_days() / 30.44).floor().clip(upper_bound=pl.col("term_months")))
chk = appr.join(loans_rep.select("loan_id", "n"), on="loan_id", how="left").with_columns(pl.col("n").fill_null(0))
chk = chk.with_columns(miss=pl.col("exp_n") - pl.col("n"))
print("approved loans: expected-observed installment count:", chk["miss"].value_counts().sort("miss").head(8).rows())
rate(chk, (pl.col("miss") >= 2).cast(pl.Int8), "approved loan missing >=2 expected installments")
rate(chk, (pl.col("n") == 0).cast(pl.Int8), "approved loan with no repayment rows")

print("\n=== 4) account identities")
acc2 = acc.sort("customer_id", "month").with_columns(
    id_err=pl.col("opening_balance") + pl.col("inflows") - pl.col("outflows") - pl.col("closing_balance"),
    chain_err=pl.col("opening_balance") - pl.col("closing_balance").shift(1).over("customer_id"),
    gap=(pl.col("month").diff().over("customer_id").dt.total_days()))
print("closing != opening+in-out:", (acc2["id_err"] != 0).sum(), " opening != prev closing:", (acc2["chain_err"].fill_null(0) != 0).sum(),
      " month gaps>31d:", (acc2["gap"].fill_null(0) > 31).sum())
print("negative balances: min_balance<0:", (acc["minimum_balance"] < 0).sum(), " closing<0:", (acc["closing_balance"] < 0).sum())
print("income_like > inflows:", (acc["income_like_credits"] > acc["inflows"]).sum(), " min>avg:", (acc["minimum_balance"] > acc["average_balance"]).sum())
rate(acc2, (pl.col("id_err") != 0).cast(pl.Int8), "any balance-identity break")
rate(acc, (pl.col("income_like_credits") == 0).cast(pl.Int8), "any month with zero income credits")
nm = acc.group_by("customer_id").agg(n=pl.len())
print("months per customer:", nm["n"].value_counts().sort("n").rows())

print("\n=== 5) bureau")
bb = bur.join(apps.select("customer_id", sel="application_id"), on="customer_id")
print("delay (available-report) days quantiles:", bb.select(((pl.col("available_datetime") - pl.col("report_datetime")).dt.total_days())
      .quantile(q).alias(str(q)) for q in (0, .1, .5, .9, 1)).row(0))
print("utilization null & limit>0:", ((bur["utilization"].is_null()) & (bur["revolving_credit_limit"] > 0)).sum(),
      " util != bal/limit:", bur.filter(pl.col("revolving_credit_limit") > 0).select(
          ((pl.col("revolving_outstanding_balance") / pl.col("revolving_credit_limit") - pl.col("utilization")).abs() > 1e-3).sum()).item())
print("revolving balance > outstanding:", (bur["revolving_outstanding_balance"] > bur["outstanding_balance"]).sum())
print("providers in selected pull:", bb.filter(pl.col("application_id") == pl.col("sel")).group_by("customer_id").len()["len"].value_counts().sort("len").rows())

print("\n=== 6) application timing")
t = tr.with_columns(lag_h=(pl.col("decision_datetime") - pl.col("application_datetime")).dt.total_seconds() / 3600,
                    dmin=pl.col("decision_datetime").dt.minute(), dsec=pl.col("decision_datetime").dt.second(),
                    ahour=pl.col("application_datetime").dt.hour(), dhour=pl.col("decision_datetime").dt.hour())
print("decision seconds all zero:", (t["dsec"] == 0).all(), " lag_h quantiles:", t.select(pl.col("lag_h").quantile(q).alias(str(q)) for q in (0, .1, .5, .9, 1)).row(0))
print(t.group_by(lag_b=pl.col("lag_h").cut([6, 24, 48, 96])).agg(n=pl.len(), bad=pl.col("target").mean()).sort("lag_b"))
print(t.group_by(dh=pl.col("dhour") // 4 * 4).agg(n=pl.len(), bad=pl.col("target").mean()).sort("dh"))
