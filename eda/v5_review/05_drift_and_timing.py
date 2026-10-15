"""Covariate/prediction drift and availability audit; no adversarial model or test-label inference."""
from shared import *
from common import psi
from scipy.stats import ks_2samp

def main():
    f, tr, te, y, raw, R, score = predictions()
    rows = []
    for c in [c for c in tr.columns if c not in IDS and c not in CATS]:
        a, b = tr[c].to_numpy().astype(float), te[c].to_numpy().astype(float)
        finite = a[np.isfinite(a)]; fill = np.median(finite) if len(finite) else 0.
        aa = np.nan_to_num(a, nan=fill, posinf=fill, neginf=fill)
        bb = np.nan_to_num(b, nan=fill, posinf=fill, neginf=fill)
        test = ks_2samp(aa, bb)
        rows.append(dict(feature=c, train_test_KS=float(test.statistic), p=float(test.pvalue), PSI=psi(a,b),
            missing_train=float((~np.isfinite(a)).mean()), missing_test=float((~np.isfinite(b)).mean())))
    drift = pd.DataFrame(rows); drift["q"] = bh(drift.p); drift = drift.sort_values("train_test_KS", ascending=False)
    csv(drift, "05_feature_drift")
    cats = []
    for c in CATS:
        a = tr[c].to_pandas().value_counts(normalize=True)
        b = te[c].to_pandas().value_counts(normalize=True)
        both = pd.concat([a,b], axis=1).fillna(0)
        cats.append(dict(feature=c, total_variation=float(abs(both.iloc[:,0]-both.iloc[:,1]).sum()/2),
            unseen_test_levels=sorted(set(te[c].to_list())-set(tr[c].to_list()))))
    csv(cats, "05_categorical_drift")
    score_rows = []
    for name, a in raw.items():
        b = np.load(ROOT / "results/v5" / ("test_" + name + ".npy"))
        score_rows.append(dict(model=name, raw_prediction_train_test_KS=ks_2samp(a,b).statistic,
            OOF_mean=a.mean(), test_mean=b.mean(), OOF_std=a.std(), test_std=b.std()))
    csv(score_rows, "05_raw_prediction_drift")
    cut = f.select("customer_id", cutoff="decision_datetime")
    audit = []
    specs = {
        "previous_applications": ["decision_datetime"],
        "repayment_history": ["due_date", "last_payment_datetime"],
        "account_behavior": ["month"],
        "bureau_observations": ["available_datetime", "report_datetime"],
        "ecommerce_transactions": ["transaction_datetime"]}
    for name, cols in specs.items():
        t = table(name).select("customer_id", *cols).join(cut, on="customer_id", validate="m:1")
        expr = pl.col(cols[0])
        if name == "account_behavior":
            expr = expr.dt.offset_by("1mo").cast(pl.Datetime)
        equal = t.filter(expr == pl.col("cutoff")).height
        future = t.filter(expr > pl.col("cutoff")).height
        detail = dict(table=name, rows=t.height, equal_cutoff=equal, after_cutoff=future)
        if name == "repayment_history":
            detail["last_payment_at_or_after_cutoff"] = t.filter(pl.col("last_payment_datetime") >= pl.col("cutoff")).height
            assert detail["last_payment_at_or_after_cutoff"] == 0
        assert future == 0
        audit.append(detail)
        del t
    csv(audit, "05_information_timing")
    prev = table("previous_applications")
    assert prev.select("application_id").join(f.select("application_id"), on="application_id").height == 0
    tm = tr["decision_datetime"].dt.month().to_numpy(); em = te["decision_datetime"].dt.month().to_numpy()
    month = csv([dict(month=m, train=int((tm==m).sum()), test=int((em==m).sum()),
        train_bad_rate=float(y[tm==m].mean()), saved_OOF_KS=ks(y[tm==m],score[tm==m])) for m in range(1,13)], "05_month_mix")
    js(dict(max_feature_train_test_KS=float(drift.train_test_KS.max()), max_PSI=float(drift.PSI.max()),
        features_drift_FDR005=drift.loc[drift.q<.05,"feature"].tolist(),
        thin_train=float((tr["has_history"].to_numpy()==0).mean()), thin_test=float((te["has_history"].to_numpy()==0).mean()),
        train_date_min=tr["decision_datetime"].min(),train_date_max=tr["decision_datetime"].max(),
        test_date_min=te["decision_datetime"].min(),test_date_max=te["decision_datetime"].max(),
        caveats=["Aggregate covariate similarity cannot guarantee public/private equality.",
                 "OOF is held-out prediction; test is a fold ensemble, so raw prediction variance is not directly comparable.",
                 "Repayment DPD values are cutoff snapshots, not values available on their old due dates.",
                 "Month-sliced random OOF is not a forward-time holdout."]), "05_drift_decision")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    top = drift.head(10).iloc[::-1]
    ax[0].barh(top.feature, top.train_test_KS); ax[0].set_title("Largest univariate train-test drift"); ax[0].set_xlabel("KS; median-imputed missing")
    ax[1].plot(month.month,month.train/month.train.sum(),"o-",label="train")
    ax[1].plot(month.month,month.test/month.test.sum(),"s--",label="test")
    ax[1].set_title("Decision-month mixture"); ax[1].legend()
    for name in ["lgb", "tabm", "tabpfn_rel"]:
        a=raw[name]; b=np.load(ROOT/"results/v5"/("test_"+name+".npy"))
        ax[2].plot(np.linspace(0,1,101),np.quantile(b,np.linspace(0,1,101))-np.quantile(a,np.linspace(0,1,101)),label=name)
    ax[2].set(xlabel="Prediction quantile",ylabel="Test minus OOF raw score",title="Before rank transforms can hide drift")
    ax[2].legend(); figsave(fig,"05_drift_timing")

if __name__ == "__main__":
    main()
