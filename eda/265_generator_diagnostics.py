"""Step 3: how did the generator build repayment_history? Label-free structure checks, then a few label checks.

A  late rate (max_dpd > 0) by installment index, by calendar month of due_date, by loan order within the customer
B  persistence: correlation of loan-level late rates between two loans of the same customer, by gap between starts;
   split-half (installments 1-6 vs 7-12) inside one loan as the reference. Decay with gap means the latent drifts.
C  is P(late | prev on time) explained by the customer's other loans only (a fixed latent), or does it also depend on
   index / amount / month? Logistic deviance gain of each covariate on top of spline(LOO customer rate).
D  label: KS of the late rate from the customer's last loan vs first loan vs all loans, and by calendar recency.
"""
import numpy as np
import polars as pl
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

import common as c


def table(r, by):
    t = r.group_by(by).agg(n=pl.len(), late=pl.col("late").mean()).sort(by)
    print(t.to_pandas().to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def dev(X, y):
    m = LogisticRegression(C=1e4, max_iter=2000).fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return -2 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    r = (c.load("repayment_history").sort("customer_id", "loan_id", "due_date")
         .with_columns(late=(pl.col("max_days_past_due") > 0).cast(pl.Int8),
                       idx=pl.col("due_date").rank("ordinal").over("loan_id").cast(pl.Int32),
                       nidx=pl.len().over("loan_id"),
                       start=pl.col("due_date").min().over("loan_id"))
         .with_columns(order=pl.col("start").rank("dense").over("customer_id").cast(pl.Int32),
                       prev_late=pl.col("late").shift(1).over("loan_id"),
                       month=pl.col("due_date").dt.strftime("%Y-%m")))
    # the last installment of a loan can be censored (in progress), so it is left out of every rate below
    rc = r.filter(pl.col("idx") < pl.col("nidx"))

    print("== A: installments per loan"); table(r.with_columns(k=pl.col("nidx")), "k")
    print("== A: late rate by installment index"); table(rc, "idx")
    print("== A: late rate by loan order within customer"); table(rc.with_columns(o=pl.col("order").clip(1, 6)), "o")
    print("== A: late rate by due month"); table(rc, "month")

    loans = rc.group_by("customer_id", "loan_id").agg(
        rate=pl.col("late").mean(), start=pl.col("start").first(), order=pl.col("order").first(),
        h1=pl.col("late").filter(pl.col("idx") <= 6).mean(), h2=pl.col("late").filter(pl.col("idx") > 6).mean())
    print("== B: split-half corr inside a loan:",
          round(loans.drop_nulls(["h1", "h2"]).select(pl.corr("h1", "h2")).item(), 4))
    pairs = (loans.join(loans, on="customer_id", suffix="_b").filter(pl.col("order") < pl.col("order_b"))
             .with_columns(gap=(pl.col("start_b") - pl.col("start")).dt.total_days()))
    pairs = pairs.with_columns(gb=pl.col("gap").cut([90, 180, 365, 540, 730]))
    print("== B: corr of loan late rates (two loans of one customer) by gap in days")
    print(pairs.group_by("gb").agg(n=pl.len(), corr=pl.corr("rate", "rate_b")).sort("gb").to_pandas().to_string(index=False))

    # C: rows after an on-time installment; LOO customer rate from the customer's other loans
    cust = loans.group_by("customer_id").agg(s=pl.col("rate").sum(), k=pl.len())
    d = (rc.filter(pl.col("prev_late") == 0).join(loans.select("loan_id", "rate"), on="loan_id")
         .join(cust, on="customer_id").filter(pl.col("k") > 1)
         .with_columns(loo=(pl.col("s") - pl.col("rate")) / (pl.col("k") - 1)))
    d = d.sample(min(400_000, d.height), seed=0)
    y = d["late"].to_numpy().astype(float)
    base = SplineTransformer(n_knots=8, degree=3).fit_transform(d["loo"].to_numpy()[:, None])
    d0 = dev(base, y)
    print(f"== C: P(late | prev on time), {len(y)} rows, base = spline(LOO customer rate), base dev {d0:.0f}")
    mon = d["due_date"].dt.month().to_numpy()
    cands = {
        "idx (spline)": SplineTransformer(n_knots=5, degree=3).fit_transform(d["idx"].to_numpy()[:, None].astype(float)),
        "log amount_due (spline)": SplineTransformer(n_knots=5, degree=3).fit_transform(np.log1p(d["amount_due"].to_numpy())[:, None]),
        "calendar month dummies": np.eye(12)[mon - 1][:, 1:],
        "loan order (spline)": SplineTransformer(n_knots=4, degree=2).fit_transform(d["order"].to_numpy()[:, None].astype(float)),
        "due year-month (spline)": SplineTransformer(n_knots=8, degree=3).fit_transform(d["due_date"].dt.epoch("d").to_numpy()[:, None].astype(float)),
    }
    for k, X in cands.items():
        g = d0 - dev(np.hstack([base, X]), y)
        print(f"   {k:28s} chi2 {g:9.1f}  df {X.shape[1]:2d}  p {chi2.sf(g, X.shape[1]):.2e}")

    # D: label checks on train applications with history
    f = c.features().filter(pl.col("is_train") == 1).select("customer_id", "target", "decision_datetime")
    agg = loans.sort("start").group_by("customer_id").agg(
        all=pl.col("rate").mean(), last=pl.col("rate").last(), first=pl.col("rate").first(), n=pl.len())
    last_inst = r.group_by("customer_id").agg(last_due=pl.col("due_date").max())
    j = f.join(agg, on="customer_id").join(last_inst, on="customer_id").with_columns(
        since=(pl.col("decision_datetime") - pl.col("last_due")).dt.total_days())
    y = j["target"].to_numpy()
    print(f"== D: train apps with history {j.height}; bad rate {y.mean():.4f}")
    for k in ["all", "last", "first"]:
        print(f"   KS late rate {k:5s} {c.ks(y, j[k].to_numpy()):.4f}")
    m = j["n"].to_numpy() > 1
    print(f"   customers with >1 loan ({m.sum()}): KS last {c.ks(y[m], j['last'].to_numpy()[m]):.4f} "
          f"first {c.ks(y[m], j['first'].to_numpy()[m]):.4f} all {c.ks(y[m], j['all'].to_numpy()[m]):.4f}")
    print("   days from last due date to decision:", j["since"].describe().to_pandas().to_string(index=False))
    jb = j.with_columns(sb=pl.col("since").cut([30, 90, 180, 365]))
    for g, part in jb.group_by("sb"):
        yy = part["target"].to_numpy()
        if 0 < yy.mean() < 1 and len(yy) > 300:
            print(f"   since {str(g[0]):14s} n {len(yy):6d} bad {yy.mean():.4f} KS(all) {c.ks(yy, part['all'].to_numpy()):.4f}")


if __name__ == "__main__":
    main()
