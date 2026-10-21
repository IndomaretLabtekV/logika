"""EDA tables, figures, fair baseline comparisons, and an Indonesian report."""
from __future__ import annotations

from datetime import datetime, timezone
import base64
import importlib.metadata
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from analyze import OUT, PROCESSED, ROOT, TABLES

FIGURES = OUT / "figures"
EXCLUDED = {"application_id", "customer_id", "application_datetime", "decision_datetime", "target", "split"}


def table(frame, name):
    frame.to_csv(TABLES / f"{name}.csv", index=False)
    return frame


def md_table(frame, decimals=3):
    def fmt(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:,.{decimals}f}"
        return str(value).replace("|", "/").replace("\n", " ")
    headers = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(fmt(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def group_risk(train, column, groups=None):
    grouping = train[column] if groups is None else groups
    result = train.groupby(grouping, observed=True).target.agg(n="size", defaults="sum", default_rate="mean").reset_index()
    result.columns = ["group", "n", "defaults", "default_rate"]
    z = 1.96
    n, p = result.n, result.default_rate
    mid = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    result["ci95_low"] = mid - half
    result["ci95_high"] = mid + half
    result["variable"] = column
    return result


def save_figure(name):
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_benchmarks(base):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.calibration import calibration_curve
    from xgboost import XGBClassifier

    train = base[base.split.eq("train")].reset_index(drop=True)
    all_cols = [c for c in train if c not in EXCLUDED and train[c].nunique(dropna=True) > 1]
    application_cols = ["loan_amount", "annual_interest_rate", "proposed_monthly_installment", "loan_purpose", "decision_delay_hours"]
    profile_cols = application_cols + ["age", "region", "employment_type", "employment_tenure_months",
                                       "declared_monthly_income", "dependents", "housing_status",
                                       "installment_income_ratio", "loan_income_ratio", "income_per_dependent"]
    train_idx, val_idx = train_test_split(np.arange(len(train)), test_size=.25, random_state=2026, stratify=train.target)
    temporal_train = np.flatnonzero(train.decision_datetime.lt(pd.Timestamp("2025-10-01")))
    temporal_val = np.flatnonzero(train.decision_datetime.ge(pd.Timestamp("2025-10-01")))
    specs = [("Prior konstan", "Stratified 25%", [], train_idx, val_idx),
             ("Aplikasi", "Stratified 25%", application_cols, train_idx, val_idx),
             ("Aplikasi + profil", "Stratified 25%", profile_cols, train_idx, val_idx),
             ("Semua riwayat", "Stratified 25%", all_cols, train_idx, val_idx),
             ("Prior konstan", "Temporal Okt–Des", [], temporal_train, temporal_val),
             ("Semua riwayat", "Temporal Okt–Des", all_cols, temporal_train, temporal_val)]
    rows, predictions, importances = [], [], []
    fig, ax = plt.subplots(figsize=(7, 5))
    calibration = []
    for label, strategy, columns, ti, vi in specs:
        print(f"Benchmark: {label}, {strategy} ({len(columns)} features)", flush=True)
        yt, yv = train.target.iloc[ti].astype(int), train.target.iloc[vi].astype(int)
        if not columns:
            probability = np.full(len(vi), yt.mean())
        else:
            categorical = [c for c in columns if train[c].dtype == object]
            numeric = [c for c in columns if c not in categorical]
            encoder = ColumnTransformer([
                ("num", "passthrough", numeric),
                ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)
            ], verbose_feature_names_out=False)
            model = Pipeline([("encode", encoder), ("model", XGBClassifier(
                n_estimators=300, max_depth=4, min_child_weight=40, learning_rate=.05,
                subsample=.85, colsample_bytree=.85, reg_lambda=20,
                objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                n_jobs=8, random_state=2026))])
            model.fit(train.iloc[ti][columns], yt)
            probability = model.predict_proba(train.iloc[vi][columns])[:, 1]
            if label == "Semua riwayat" and strategy == "Stratified 25%":
                names = encoder.get_feature_names_out()
                gain = model.named_steps["model"].feature_importances_
                importance = pd.DataFrame({"feature": names, "gain_importance": gain}).sort_values("gain_importance", ascending=False)
                table(importance, "feature_importance_gain")
                importances = importance.head(15).to_dict("records")
        ks_fpr, ks_tpr, _ = roc_curve(yv, probability, drop_intermediate=False)
        metrics = {"model": label, "validation": strategy, "train_n": len(ti), "validation_n": len(vi),
                   "feature_n": len(columns), "validation_positive_rate": float(yv.mean()),
                   "ks_statistic": float(np.max(np.abs(ks_tpr - ks_fpr))),
                   "roc_auc": roc_auc_score(yv, probability), "average_precision": average_precision_score(yv, probability),
                   "log_loss": log_loss(yv, probability), "brier_score": brier_score_loss(yv, probability)}
        rows.append(metrics)
        prediction = train.iloc[vi][["application_id", "target", "decision_datetime"]].copy()
        prediction["probability"] = probability
        prediction["model"] = label
        prediction["validation"] = strategy
        predictions.append(prediction)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        if strategy == "Stratified 25%":
            fpr, tpr, _ = roc_curve(yv, probability)
            ax.plot(fpr, tpr, label=f"{label} (AUC {metrics['roc_auc']:.3f})")
        if label == "Semua riwayat":
            actual, estimated = calibration_curve(yv, probability, n_bins=10, strategy="quantile")
            calibration.append((strategy, estimated, actual))
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC pada validation stratified")
    ax.legend(fontsize=9)
    save_figure("model_roc")
    plt.figure(figsize=(7, 5))
    for label, estimated, actual in calibration:
        plt.plot(estimated, actual, "o-", label=label)
    plt.plot([0, .8], [0, .8], "--", color="grey")
    plt.xlabel("Rata-rata probabilitas prediksi")
    plt.ylabel("Proporsi gagal bayar aktual")
    plt.title("Kalibrasi baseline dengan semua riwayat")
    plt.legend()
    save_figure("calibration")
    pd.concat(predictions, ignore_index=True).to_parquet(OUT / "validation_predictions.parquet", index=False)
    table(pd.DataFrame(rows), "model_benchmarks")
    (OUT / "model_config.json").write_text(json.dumps({
        "seed": 2026, "stratified_validation_fraction": .25, "temporal_cutoff": "2025-10-01",
        "excluded_columns": sorted(EXCLUDED), "all_feature_columns": all_cols,
        "hyperparameters": {"n_estimators": 300, "max_depth": 4, "min_child_weight": 40,
                            "learning_rate": .05, "subsample": .85, "colsample_bytree": .85,
                            "reg_lambda": 20, "n_jobs": 8, "tree_method": "hist"},
        "selection": "Fixed parameters; no tuning or early stopping on validation.",
        "competition_metric": "Kolmogorov-Smirnov statistic (KS), supplied by user from Evaluation."}, indent=2))
    return pd.DataFrame(rows), importances


def make_report(base, run_models=True, benchmark_results=None):
    OUT.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    train = base[base.split.eq("train")].copy()
    test = base[base.split.eq("test")].copy()
    inventory = pd.read_csv(TABLES / "table_inventory.csv")
    quality = pd.read_csv(TABLES / "column_quality.csv")
    coverage = pd.read_csv(TABLES / "history_coverage.csv")
    audit = json.loads((OUT / "quality_audit.json").read_text())
    table(train.select_dtypes(include="number").describe(percentiles=[.01, .05, .5, .95, .99]).T.reset_index().rename(columns={"index": "feature"}), "numeric_summary")
    feature_missing = pd.DataFrame({"feature": [c for c in base if c not in EXCLUDED]})
    for split, subset in [("train", train), ("test", test)]:
        feature_missing[f"{split}_missing_pct"] = [100 * subset[c].isna().mean() for c in feature_missing.feature]
    table(feature_missing, "feature_missingness")

    risks = []
    for col in ["annual_interest_rate", "employment_type", "housing_status", "loan_purpose", "region", "has_previous", "has_account", "has_repayments"]:
        risks.append(group_risk(train, col))
    for col in ["installment_income_ratio", "bureau_score_mean", "bureau_delinquent_accounts_max", "repayment_historical_dpd_max",
                "observed_declared_income_ratio", "account_low_balance_days_mean", "ecom_cancel_rate"]:
        observed = train[col].notna()
        groups = pd.Series("Tidak tersedia", index=train.index, name=col)
        if observed.any():
            cut = pd.qcut(train.loc[observed, col], 5, duplicates="drop")
            groups.loc[observed] = cut.astype(str)
        risks.append(group_risk(train, col, groups))
    risk = table(pd.concat(risks, ignore_index=True), "default_by_group")
    numeric = [c for c in train.select_dtypes(include="number") if c not in EXCLUDED and train[c].nunique() > 1]
    correlations = train[numeric + ["target"]].corr()["target"].drop("target").rename("pearson_target_correlation")
    correlations = correlations.to_frame().assign(absolute_correlation=lambda x: x.pearson_target_correlation.abs()).sort_values("absolute_correlation", ascending=False).reset_index().rename(columns={"index": "feature"})
    table(correlations, "target_correlations")

    drift_rows = []
    for col in numeric:
        a, b = train[col].dropna(), test[col].dropna()
        if len(a) and len(b):
            pooled_sd = np.sqrt((a.var() + b.var()) / 2)
            drift_rows.append({"feature": col, "train_mean": a.mean(), "test_mean": b.mean(),
                               "standardized_mean_difference": (b.mean() - a.mean()) / pooled_sd if pooled_sd else 0,
                               "ks_statistic": ks_2samp(a, b).statistic,
                               "train_missing_pct": 100 * train[col].isna().mean(),
                               "test_missing_pct": 100 * test[col].isna().mean()})
    drift = pd.DataFrame(drift_rows).sort_values("ks_statistic", ascending=False)
    table(drift, "train_test_numeric_drift")
    categorical_drift = []
    for col in ["loan_purpose", "region", "employment_type", "housing_status"]:
        a, b = train[col].value_counts(normalize=True), test[col].value_counts(normalize=True)
        categories = a.index.union(b.index)
        categorical_drift.append({"feature": col, "total_variation_distance": .5 * (a.reindex(categories, fill_value=0) - b.reindex(categories, fill_value=0)).abs().sum()})
    table(pd.DataFrame(categorical_drift), "train_test_categorical_drift")
    monthly = train.groupby(train.decision_datetime.dt.strftime("%Y-%m")).target.agg(n="size", defaults="sum", default_rate="mean").reset_index().rename(columns={"decision_datetime": "month"})
    monthly["test_n"] = test.groupby(test.decision_datetime.dt.strftime("%Y-%m")).size().reindex(monthly.month).to_numpy()
    table(monthly, "monthly_target")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = train.target.value_counts().sort_index()
    axes[0].bar(["0: tidak mencapai DPD90", "1: mencapai DPD90"], counts, color=["#2563eb", "#dc2626"])
    for i, value in enumerate(counts):
        axes[0].text(i, value + 800, f"{value:,} ({value/len(train):.2%})", ha="center", fontsize=9)
    axes[0].set_ylim(0, counts.max() * 1.15)
    axes[0].set_title("Komposisi target pada train")
    axes[1].plot(monthly.month, monthly.default_rate * 100, "o-", color="#dc2626")
    axes[1].tick_params(axis="x", rotation=60)
    axes[1].set(title="Proporsi target per bulan keputusan", ylabel="Target = 1 (%)")
    save_figure("target_and_time")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    chart_features = ["annual_interest_rate", "bureau_score_mean", "repayment_historical_dpd_max"]
    for ax, feature in zip(axes, chart_features):
        data = risk[risk.variable.eq(feature)].copy()
        if feature == "annual_interest_rate":
            labels = [f"{float(x):.0%}" for x in data.group]
        else:
            labels = data.group.astype(str).tolist()
        values = data.default_rate * 100
        errors = np.vstack([values - data.ci95_low * 100, data.ci95_high * 100 - values])
        ax.bar(np.arange(len(data)), values, color="#2563eb", yerr=errors, capsize=3)
        ax.set_xticks(np.arange(len(data)), labels, rotation=45, ha="right", fontsize=8)
        ax.set(title=feature, ylabel="Target = 1 (%)")
    save_figure("risk_signals")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cov = coverage[coverage.table.isin(["previous_applications", "repayment_history", "account_behavior", "bureau_observations", "ecommerce_transactions"])].pivot(index="table", columns="split", values="coverage_pct")
    cov.plot.barh(ax=axes[0], color=["#f97316", "#2563eb"])
    axes[0].set(xlim=(0, 105), xlabel="Pelanggan tercakup (%)", ylabel="", title="Ketersediaan riwayat")
    top = correlations.head(12).iloc[::-1]
    axes[1].barh(top.feature, top.pearson_target_correlation, color="#2563eb")
    axes[1].set(xlabel="Korelasi Pearson dengan target", title="Sinyal numerik (asosiasi)")
    save_figure("coverage_and_correlation")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["loan_amount", "declared_monthly_income", "installment_income_ratio"]):
        lim = np.nanquantile(base[col], .99)
        bins = np.linspace(0, lim, 45)
        ax.hist(train[col], bins=bins, density=True, alpha=.5, label="train", color="#2563eb")
        ax.hist(test[col], bins=bins, density=True, alpha=.5, label="test", color="#f97316")
        ax.set(title=col, xlabel="IDR" if "ratio" not in col else "Rasio")
        ax.legend()
    save_figure("train_test_distributions")

    benchmarks, importances = benchmark_results if benchmark_results is not None else (run_benchmarks(base) if run_models else (pd.DataFrame(), []))
    income = train.declared_monthly_income.median()
    ecom_audit = audit["ecommerce_transactions"]
    double_factor = ecom_audit["naive_completed_spend_idr"] / ecom_audit["correct_completed_spend_idr"]
    month_min = monthly.loc[monthly.default_rate.idxmin()]
    month_max = monthly.loc[monthly.default_rate.idxmax()]
    score_edges = pd.qcut(train.bureau_score_mean.dropna(), 5, duplicates="drop", retbins=True)[1]
    score_low = train[train.bureau_score_mean.le(score_edges[1])]
    score_high = train[train.bureau_score_mean.gt(score_edges[-2])]
    dpd_low = train[train.repayment_historical_dpd_max.le(30)]
    dpd_high = train[train.repayment_historical_dpd_max.ge(90)]
    burden_groups = pd.qcut(train.installment_income_ratio, 5, duplicates="drop")
    burden_low = train[burden_groups.eq(burden_groups.cat.categories[0])]
    burden_high = train[burden_groups.eq(burden_groups.cat.categories[-1])]
    absent_history = train[train.has_previous.eq(0)]
    present_history = train[train.has_previous.eq(1)]
    interest = risk[risk.variable.eq("annual_interest_rate")][["group", "n", "default_rate", "ci95_low", "ci95_high"]].copy()
    interest["group"] = interest.group.map(lambda v: f"{float(v):.0%}")
    interest.columns = ["Bunga tahunan", "Jumlah", "Proporsi default", "CI95 bawah", "CI95 atas"]
    nullable = quality[quality.missing.gt(0)][["table", "column", "missing", "missing_pct"]]
    results = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "train_rows": len(train), "test_rows": len(test), "defaults": int(train.target.sum()),
               "default_rate": float(train.target.mean()), "majority_accuracy": float(1 - train.target.mean()),
               "total_raw_rows": int(inventory.rows.sum()), "raw_size_mb": float(inventory.size_mb.sum()),
               "feature_columns": len([c for c in base if c not in EXCLUDED]),
               "naive_ecommerce_spend_inflation_factor": double_factor,
               "max_train_test_ks": float(drift.ks_statistic.max()),
               "max_absolute_smd": float(drift.standardized_mean_difference.abs().max()),
               "benchmarks": benchmarks.to_dict("records"), "top_model_gain_features": importances,
               "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "matplotlib", "scipy", "scikit-learn", "xgboost", "kaggle"]}}
    (OUT / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    source = "https://www.kaggle.com/competitions/data-science-competition-logika-ui-2026/data"
    text = f"""# Analisis data LOGIKA UI 2026

Sumber: [data kompetisi Kaggle]({source}), diunduh melalui API Kaggle dengan kredensial lokal. Analisis mengikuti `DATA_DICTIONARY.md` versi 2. Tanggal berkas di manifest: 10 Oktober 2026. Waktu dalam data memakai Asia/Jakarta dan nominal uang dalam rupiah bulat. Region dan penyedia biro adalah fiktif menurut kamus data.

## Temuan utama

- Train berisi **{len(train):,} aplikasi** dan test **{len(test):,} aplikasi**. Ada tepat satu aplikasi terpilih per pelanggan; pelanggan train dan test tidak tumpang tindih.
- Target 1 berarti DPD mencapai **90 hari dalam 12 bulan kalender**, sebanyak **{int(train.target.sum()):,} atau {train.target.mean():.2%}**. Memprediksi semua sebagai 0 saja menghasilkan accuracy **{1-train.target.mean():.2%}**, sehingga accuracy tidak cukup untuk menilai model.
- Semua waktu observasi riwayat yang diperiksa mendahului keputusan terpilih; bulan rekening juga sudah lengkap. Aturan waktu tetap diterapkan saat membentuk fitur.
- Data e-commerce berisi **{ecom_audit['items']:,} barang dari {ecom_audit['orders']:,} pesanan**. Menjumlahkan `order_total` per barang pada pesanan completed membesar-besarkan belanja menjadi **{double_factor:.3f} kali** nilai yang benar.
- `sample_submission.csv` dan test mempunyai ID yang sama, **urutan berbeda**. Penyusunan prediksi harus menggunakan join pada `application_id`.

## Struktur dan skala data

{md_table(inventory.rename(columns={'table':'Tabel','rows':'Baris','columns':'Kolom','customers':'Pelanggan','size_mb':'Ukuran MB'}), 2)}

Tabel inti adalah train/test aplikasi. Profil dihubungkan melalui `customer_id` dengan validasi one-to-one. Lima tabel riwayat diagregasi menjadi satu baris per pelanggan sebelum join; menggabungkan baris mentah sekaligus akan menggandakan baris dan nominal. `previous_applications` juga menghubungkan pinjaman historis melalui `loan_id` ke riwayat cicilan. Biro mempunyai observasi untuk aplikasi terpilih dan historis.

## Kualitas data dan arti nilai kosong

{md_table(nullable, 2)}

Kolom inti train/test dan profil tidak mempunyai nilai kosong. Audit menemukan {sum(v.get('duplicate_rows',0) for v in audit.values()):,} duplikasi baris persis dan {sum(v.get('duplicate_primary_keys',0) for v in audit.values()):,} duplikasi kunci utama/komposit. Semua pelanggan pada tabel pendukung terdapat pada daftar aplikasi terpilih. Audit rinci ada di `quality_audit.json`, termasuk konsistensi saldo, total pesanan, pasangan loan/customer, rentang skor, dan rumus cicilan.

- `loan_id`, nominal disetujui, bunga dan cicilan historis kosong pada aplikasi rejected/cancelled; ini sesuai struktur keputusan, bukan otomatis kesalahan data.
- Skor biro dapat kosong untuk laporan tanpa skor. `utilization` tidak terdefinisi jika limit tidak positif; jangan mengubah semuanya menjadi nol.
- `last_payment_datetime` kosong ketika belum dibayar. Riwayat yang sama sekali tidak tersedia dipertahankan sebagai NaN dengan indikator ketersediaan, agar berbeda dari nilai nol yang benar-benar teramati.
- Tidak ada penghapusan outlier otomatis. Nilai tinggi dapat merupakan kasus kredit yang sah; statistik persentil tersedia di `numeric_summary.csv`.

{md_table(cov.reset_index(), 2)}

![Cakupan dan korelasi](figures/coverage_and_correlation.png)

## Distribusi dan hubungan dengan target

Median pinjaman pada train **Rp{train.loan_amount.median():,.0f}**; median pendapatan bulanan **Rp{income:,.0f}**. Median cicilan/pendapatan deklarasi adalah **{train.installment_income_ratio.median():.2%}**, dengan persentil 95 sebesar **{train.installment_income_ratio.quantile(.95):.2%}**. Semua tenor aplikasi terpilih adalah 12 bulan, sehingga `term_months` tidak memberikan variasi prediktif.

Proporsi target per bulan keputusan berkisar **{month_min.default_rate:.2%} ({month_min.month})** sampai **{month_max.default_rate:.2%} ({month_max.month})**. Train dan test mencakup Januari–Desember 2025 pada waktu keputusan.

![Target dan waktu](figures/target_and_time.png)

{md_table(interest, 4)}

Bunga lebih tinggi berasosiasi dengan proporsi default lebih tinggi. Hal ini dapat mencerminkan penetapan harga berdasarkan risiko yang sudah diketahui; tabel ini tidak membuktikan bunga menyebabkan default. Kelompok bunga 32% kecil, sehingga estimasinya lebih tidak pasti. Error bar menunjukkan interval Wilson 95%, dan seluruh analisis kelompok memakai label train saja.

Temuan yang lebih langsung dari riwayat:

- Rata-rata skor biro pada kelompok ≤{score_edges[1]:.1f} mempunyai proporsi default **{score_low.target.mean():.2%}** ({len(score_low):,} aplikasi), sedangkan kelompok >{score_edges[-2]:.1f} **{score_high.target.mean():.2%}** ({len(score_high):,} aplikasi).
- Di antara pelanggan dengan riwayat cicilan, keterlambatan historis maksimum ≤30 hari berasosiasi dengan default **{dpd_low.target.mean():.2%}** ({len(dpd_low):,} aplikasi), sementara maksimum ≥90 hari **{dpd_high.target.mean():.2%}** ({len(dpd_high):,} aplikasi).
- Pada kelompok cicilan/pendapatan deklarasi terendah, proporsi default **{burden_low.target.mean():.2%}**; pada kelompok tertinggi **{burden_high.target.mean():.2%}**. Batas kelompok kuantil tersedia pada tabel lengkap.
- Pelanggan tanpa riwayat aplikasi sebelumnya mempunyai proporsi default **{absent_history.target.mean():.2%}**, dibandingkan **{present_history.target.mean():.2%}** pada yang memiliki riwayat. Perbedaan populasi ini menunjukkan bahwa ketiadaan riwayat perlu dipertahankan sebagai indikator.

Angka di atas adalah asosiasi pada dataset kompetisi ini. Kelompok kuantil dibentuk pada train dan hasil yang ditampilkan dibulatkan.

![Sinyal risiko](figures/risk_signals.png)

{md_table(correlations.head(12)[['feature','pearson_target_correlation']], 4)}

Korelasi ini bersifat univariat dan menggunakan observasi yang tersedia. Fitur dengan tingkat kehilangan berbeda mempunyai populasi pembanding berbeda; korelasi, asosiasi kelompok, dan importance model tidak menyatakan sebab akibat. Tabel `default_by_group.csv` memuat jumlah, proporsi target, serta interval kepercayaan untuk kategori dan kelompok kuantil.

## Perbandingan train dan test

![Distribusi train dan test](figures/train_test_distributions.png)

Gambar distribusi dibatasi hingga persentil 99 agar terbaca; tabel statistik tetap menggunakan seluruh data. Statistik KS terbesar adalah **{drift.ks_statistic.max():.4f}**, dan perbedaan rata-rata terstandar absolut terbesar **{drift.standardized_mean_difference.abs().max():.4f}**. Ini menunjukkan {"perbedaan marginal yang kecil pada fitur yang diperiksa" if drift.ks_statistic.max() < .05 and drift.standardized_mean_difference.abs().max() < .1 else "beberapa perbedaan distribusi yang perlu diperiksa"}; tidak menjamin distribusi gabungan atau hubungan fitur-target sama. Target test tidak tersedia, sehingga proporsi default test tidak dapat diverifikasi.

{md_table(drift.head(8)[['feature','ks_statistic','standardized_mean_difference','train_missing_pct','test_missing_pct']], 4)}

## Pembentukan fitur tanpa kebocoran waktu

Cutoff adalah `decision_datetime` aplikasi terpilih, mengikuti definisi profil pada keputusan dan snapshot riwayat sebelum cutoff. Untuk prediksi pada saat pengajuan, seluruh cutoff dan fitur yang baru tersedia saat keputusan perlu ditinjau ulang.

1. Aplikasi historis hanya digunakan jika keputusan historis mendahului cutoff.
2. Cicilan jatuh tempo sebelum cutoff digunakan; pembayaran dan DPD adalah snapshot sebelum cutoff menurut kamus data. Waktu pembayaran yang tersedia diverifikasi mendahului cutoff.
3. Rekening memakai bulan kalender yang sudah selesai; fitur terbaru dan rata-rata tiga bulan lengkap dibentuk terpisah.
4. Biro difilter pada **report_datetime dan available_datetime sebelum cutoff**. Diambil laporan bertanggal paling baru per pelanggan/provider dari seluruh aplikasi yang tersedia, dengan waktu ketersediaan sebagai tie-break. Saldo antarprovider dirangkum lewat max/mean; tidak diasumsikan mewakili utang unik untuk dijumlahkan.
5. E-commerce difilter sebelum cutoff, dideduplicasi per order, dan hanya completed dihitung sebagai belanja. Cancelled menjadi fitur tingkat pembatalan; belanja kategori berasal dari `item_total`.

Fitur mencakup rasio cicilan terhadap pendapatan, pendapatan rekening versus deklarasi, saldo versus cicilan, variasi pendapatan, riwayat keterlambatan, skor/utilization biro, jumlah dan recency aplikasi historis, belanja 90/180 hari, kategori dan metode pembayaran. Hasilnya **{len([c for c in base if c not in EXCLUDED])} kolom fitur**, termasuk tenor konstan yang dikeluarkan saat baseline. ID tidak pernah dipakai sebagai angka atau kategori model; tanggal mentah dan split juga dikeluarkan.
"""
    if len(benchmarks):
        app_model = benchmarks[benchmarks.model.eq("Aplikasi")].iloc[0]
        profile_model = benchmarks[benchmarks.model.eq("Aplikasi + profil")].iloc[0]
        rich_model = benchmarks[benchmarks.model.eq("Semua riwayat") & benchmarks.validation.eq("Stratified 25%")].iloc[0]
        timed_model = benchmarks[benchmarks.model.eq("Semua riwayat") & benchmarks.validation.eq("Temporal Okt–Des")].iloc[0]
        prior_model = benchmarks[benchmarks.model.eq("Prior konstan") & benchmarks.validation.eq("Stratified 25%")].iloc[0]
        text += f"""
## Baseline untuk mengukur informasi prediktif

Metrik resmi leaderboard adalah **Kolmogorov-Smirnov (KS)**, sesuai kutipan Evaluation yang diberikan pengguna. KS adalah perbedaan maksimum antara distribusi skor dua kelas. KS, ROC-AUC, average precision (AP), log loss, dan Brier di bawah dihitung lokal; **bukan skor Kaggle**. KS/ROC-AUC/AP lebih tinggi lebih baik; log loss/Brier lebih rendah lebih baik. Hasil optimasi khusus KS tersedia pada `artifacts/optimization/`, terpisah dari baseline analisis ini.

{md_table(benchmarks[['model','validation','train_n','validation_n','ks_statistic','roc_auc','average_precision','log_loss','brier_score']], 4)}

Model XGBoost memakai parameter tetap, seed 2026, 300 pohon kedalaman 4. Perbandingan tiga kelompok fitur memakai holdout stratified 25% yang sama. Tidak ada pencarian parameter atau early stopping pada validation. Encoder kategorikal hanya fit pada bagian train; XGBoost menangani NaN numerik. ID dan tanggal mentah dikeluarkan. Pemeriksaan temporal melatih Januari–September dan menilai Oktober–Desember; label harus sudah matang jika skema ini diterapkan pada data operasional.

Pada eksperimen ini, menambahkan profil saja memberi kenaikan ROC-AUC kecil, dari **{app_model.roc_auc:.4f} menjadi {profile_model.roc_auc:.4f}**. Menambahkan seluruh riwayat menaikkannya menjadi **{rich_model.roc_auc:.4f}**, dengan AP **{rich_model.average_precision:.4f}** dibanding prior AP **{prior_model.average_precision:.4f}**. ROC-AUC temporal **{timed_model.roc_auc:.4f}** menunjukkan hasil tetap serupa pada periode akhir tahun yang diperiksa. Riwayat memberi tambahan informasi yang jauh lebih besar daripada profil pada baseline ini; ini belum mengisolasi kontribusi masing-masing tabel riwayat.

![ROC baseline](figures/model_roc.png)

![Kalibrasi](figures/calibration.png)

{md_table(pd.DataFrame(importances).head(10), 4)}

Importance di atas adalah gain pohon pada baseline stratified. Fitur yang berkorelasi dapat saling menggantikan dan membagi gain; importance bukan bukti kausal atau jaminan stabilitas. Hasil holdout merupakan estimasi awal, dan pengambilan keputusan model akhir memerlukan cross-validation yang sesuai metrik resmi. Validasi temporal hanya satu periode.
"""
    text += """
## Prioritas berikutnya

1. Gunakan KS sesuai Evaluation; verifikasi ketentuan penggunaan fitur dan waktu prediksi pada rules kompetisi yang tepat. Jangan memakai metrik dari kompetisi simulasi dengan slug yang mirip.
2. Gunakan baseline semua riwayat sebagai titik awal. Bandingkan beberapa fold stratified karena train/test mempunyai pelanggan berbeda dan periode yang sama, lalu lakukan pemeriksaan temporal tambahan.
3. Uji ablation masing-masing sumber riwayat dan jendela waktu lebih pendek; tinjau kalibrasi jika metrik membutuhkan probabilitas yang akurat.
4. Simpan prediksi dengan ID, lalu join ke urutan `sample_submission.csv`; validasi 30.000 ID unik, tanpa missing, dan probabilitas dalam [0,1].

## Menjalankan ulang

```bash
python -m pip install -r environments/teman/requirements.txt
python scripts/download_data.py
python scripts/analyze.py
# Untuk memakai agregasi yang sudah tersedia:
python scripts/analyze.py --reuse-features
# Memperbarui laporan dari hasil fitur dan baseline yang sudah ada:
python scripts/analyze.py --report-only
```

`analisis_logika.ipynb` adalah notebook pembaca hasil dan contoh reproduksi. `data/` berisi data asli; `data/processed/application_features.parquet` berisi agregasi; `artifacts/tables/` berisi tabel lengkap; `artifacts/figures/` berisi PNG yang dapat diekspor. Kredensial hanya digunakan oleh skrip unduh, tidak disalin ke notebook atau laporan, dan dikecualikan melalui `.gitignore`. Tidak ada submission yang dikirim ke Kaggle.
"""
    (OUT / "laporan_analisis.md").write_text(text)
    import markdown
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    # Embed charts so the HTML report can be opened/shared as one standalone file.
    def inline_chart(match):
        png = OUT / match.group(1)
        encoded = base64.b64encode(png.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{encoded}"'
    body = re.sub(r'src="(figures/[^"/]+\.png)"', inline_chart, body)
    document = f"""<!doctype html><html lang="id"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Analisis LOGIKA UI 2026</title><style>body{{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#172033;background:#fafbfc}}h1,h2{{line-height:1.25;color:#17346c}}h2{{margin-top:42px}}table{{display:block;overflow:auto;border-collapse:collapse;font-size:13px;background:white}}td,th{{padding:8px 12px;border:1px solid #dbe2ee;text-align:left}}th{{background:#edf3ff}}img{{max-width:100%;height:auto}}pre{{padding:16px;overflow:auto;background:#eaf0f8;border-radius:8px}}a{{color:#2458b8}}code{{font-size:.9em}}</style><main>{body}</main></html>"""
    (OUT / "laporan_analisis.html").write_text(document)
    create_notebook()
    print("Reports saved to artifacts/laporan_analisis.md and .html", flush=True)
    return results


def create_notebook():
    def cell(kind, content):
        value = {"cell_type": kind, "metadata": {}, "source": content.splitlines(keepends=True)}
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        return value
    cells = [
        cell("markdown", "# Analisis LOGIKA UI 2026\n\nNotebook ini membaca hasil analisis yang telah dijalankan. Gunakan `python scripts/analyze.py` untuk membangun ulang seluruh tabel, fitur, dan baseline. Definisi data tersedia pada `data/DATA_DICTIONARY.md`. Kredensial tidak diperlukan untuk membaca hasil lokal."),
        cell("code", "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\n\nROOT = Path.cwd()\nif not (ROOT / 'artifacts').exists():\n    ROOT = ROOT.parent\nART = ROOT / 'artifacts'\nsummary = json.loads((ART / 'summary.json').read_text())\nfeatures = pd.read_parquet(ROOT / 'data/processed/application_features.parquet')\ntrain = features[features['split'].eq('train')].copy()\ntest = features[features['split'].eq('test')].copy()\ndisplay(pd.Series({k: v for k, v in summary.items() if k not in ['benchmarks', 'top_model_gain_features', 'versions']}))"),
        cell("markdown", "## Struktur dan kualitas\n\nNilai kosong struktural, misalnya pinjaman ditolak atau utilization tanpa limit, mempunyai arti berbeda dari nol."),
        cell("code", "display(pd.read_csv(ART / 'tables/table_inventory.csv'))\ndisplay(pd.read_csv(ART / 'tables/history_coverage.csv'))\nquality = pd.read_csv(ART / 'tables/column_quality.csv')\ndisplay(quality[quality['missing'].gt(0)])\naudit = json.loads((ART / 'quality_audit.json').read_text())\ndisplay(audit)"),
        cell("markdown", "## Target, waktu, dan asosiasi\n\nTarget 1 berarti mencapai DPD 90 dalam 12 bulan kalender. Asosiasi dan importance tidak membuktikan sebab akibat."),
        cell("code", "display(Image(filename=str(ART / 'figures/target_and_time.png')))\ndisplay(Image(filename=str(ART / 'figures/risk_signals.png')))\nrisk = pd.read_csv(ART / 'tables/default_by_group.csv')\ndisplay(risk[risk['variable'].eq('annual_interest_rate')])\ndisplay(pd.read_csv(ART / 'tables/target_correlations.csv').head(15))"),
        cell("markdown", "## Distribusi train dan test\n\nPerbedaan marginal yang kecil tidak menjamin hubungan fitur-target identik. Label test tidak tersedia."),
        cell("code", "display(Image(filename=str(ART / 'figures/train_test_distributions.png')))\ndisplay(pd.read_csv(ART / 'tables/train_test_numeric_drift.csv').head(15))\ndisplay(pd.read_csv(ART / 'tables/feature_missingness.csv').sort_values('train_missing_pct', ascending=False).head(20))"),
        cell("markdown", "## Baseline dan kalibrasi\n\nParameter tetap; holdout stratified dan temporal. Metrik resmi adalah Kolmogorov-Smirnov (KS), sesuai Evaluation yang diberikan pengguna. Semua angka tabel ini adalah hasil lokal, bukan skor leaderboard. Hasil optimasi terbaru ada pada `artifacts/optimization/`."),
        cell("code", "benchmark_path = ART / 'tables/model_benchmarks.csv'\nif benchmark_path.exists() and summary['benchmarks']:\n    display(pd.read_csv(benchmark_path))\n    display(Image(filename=str(ART / 'figures/model_roc.png')))\n    display(Image(filename=str(ART / 'figures/calibration.png')))\n    display(pd.read_csv(ART / 'tables/feature_importance_gain.csv').head(20))\ncv_path = ART / 'optimization/cross_validation_summary.json'\nif cv_path.exists():\n    cv = json.loads(cv_path.read_text())\n    display(Markdown('### Optimasi KS terbaru — validasi silang 3 fold'))\n    display(pd.DataFrame({'Baseline': cv['baseline'], 'Baru': cv['selected']}).T[['ks', 'mean_fold_ks', 'roc_auc', 'average_precision']])\n    display(cv['selected']['recipe'])"),
        cell("markdown", "## Pemeriksaan yang dapat dijalankan ulang\n\nAssertion ini memeriksa integritas hasil, batas waktu, serta risiko salah urutan submission."),
        cell("code", "assert len(train) == 70000 and len(test) == 30000\nassert features['application_id'].is_unique and features['customer_id'].is_unique\nassert set(train['customer_id']).isdisjoint(test['customer_id'])\nassert train['target'].isin([0, 1]).all() and test['target'].isna().all()\nassert audit['submission']['same_id_set_as_test']\nfor checks in audit.values():\n    for name, value in checks.items():\n        if name.endswith('_at_or_after_cutoff') or name == 'incomplete_month_at_cutoff':\n            assert value == 0, (name, value)\nprint('Integritas dasar dan batas waktu: OK')"),
        cell("markdown", "## Template penyusunan submission\n\nIsi `predictions` dengan probabilitas model yang Anda latih. Join dengan validasi one-to-one; kode template tidak mengirim submission."),
        cell("code", "def align_submission(predictions):\n    sample = pd.read_csv(ROOT / 'data/sample_submission.csv')\n    assert predictions['application_id'].is_unique\n    assert set(predictions['application_id']) == set(test['application_id'])\n    result = sample[['application_id']].merge(\n        predictions[['application_id', 'default_probability']],\n        on='application_id', how='left', validate='one_to_one', sort=False\n    )\n    assert result['application_id'].equals(sample['application_id'])\n    assert result['default_probability'].notna().all()\n    assert result['default_probability'].between(0, 1).all()\n    return result"),
        cell("markdown", "## Laporan lengkap\n\nBuka `artifacts/laporan_analisis.html` untuk laporan dengan seluruh grafik; CSV lengkap tersedia di `artifacts/tables/`.\n\nReproduksi dari terminal: `python scripts/analyze.py --reuse-features`. Untuk membangun agregasi ulang, hilangkan `--reuse-features`."),
    ]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, item in enumerate(cells):
        item["id"] = f"logika-{i:02d}"
    (ROOT / "notebook/analisis_logika.ipynb").write_text(json.dumps(notebook, indent=2, ensure_ascii=False))
