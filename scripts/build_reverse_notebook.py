"""Create a self-contained review of RNG probes, jitter inference, and generator approximations."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

from build_review_notebooks import ROOT, DEST, source, md, code, section


def build():
    cells = []
    md(
        cells,
        """# 07 — Reverse engineering pola sintetis, RNG, dan jitter

Tujuan notebook ini adalah merekonstruksi aturan observasi dan mengestimasi noise,
bukan menganggap seed generator sudah ditemukan. **Seed RNG asli belum diketahui.**

Kode probe seed, inferensi efek customer, uncertainty, simulasi pendekatan, dan
validasi model terlihat di sel. Data mentah diunduh melalui skrip proyek. Paket:
`environments/teman/requirements-optimization.txt` dan `environments/teman/requirements-notebook.txt`.

Seed untuk simulasi dan pembagian audit dipilih oleh kita; tidak sama dengan seed asli
dataset. Notebook ini tidak mengirim atau mengganti submission.
""",
    )
    section(
        cells,
        "1. Persiapan dan kontrol eksekusi",
        "Audit dapat dijalankan dari data mentah tanpa model versi sebelumnya. "
        "Validasi prediksi baru memerlukan fitur dan OOF versi 5; bila belum tersedia, "
        "sel menampilkan hasil historis dan menjelaskan prasyaratnya.",
        """
from pathlib import Path
import hashlib, itertools, json, os, random, sys, time
import numpy as np
import pandas as pd
from scipy import optimize, special, stats
import matplotlib.pyplot as plt
from IPython.display import display
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, log_loss
from threadpoolctl import threadpool_limits
if Path('/tmp/logika-boosting-libs').exists():
    sys.path.insert(0,'/tmp/logika-boosting-libs')
from catboost import CatBoostClassifier

RERUN_RNG_PROBE = True
RECOMPUTE_LATENT_AUDIT = True
RECOMPUTE_MODEL_VALIDATION = True
FORCE_RETRAIN_REVERSE_MODELS = False
RNG_SEED_LIMIT = 10000

ROOT = Path(os.environ.get('LOGIKA_PROJECT_DIR',Path.cwd())).resolve()
if not (ROOT/'scripts/reverse_synthetic_patterns.py').exists():
    ROOT = next(p for p in ROOT.parents if (p/'scripts/reverse_synthetic_patterns.py').exists())
RAW = ROOT/'data'
OUT = ROOT/'artifacts/notebook_runs/07_reverse'
OUT.mkdir(parents=True,exist_ok=True)
EXCLUDED = {'application_id','customer_id','application_datetime','decision_datetime','target','split'}
print('NumPy:',np.__version__,'; output:',OUT)
""",
    )
    section(
        cells,
        "2. EDA: populasi, waktu, dan nominal",
        "Ada satu aplikasi terpilih per customer. ID digunakan untuk join dan diagnosis seed, "
        "bukan sebagai fitur numerik. Semua sejarah harus tersedia sebelum keputusan.",
        """
profiles = pd.read_csv(RAW/'customer_profiles.csv')
train_raw = pd.read_csv(RAW/'train_applications.csv',parse_dates=['application_datetime','decision_datetime'])
test_raw = pd.read_csv(RAW/'test_applications.csv',parse_dates=['application_datetime','decision_datetime'])
train_raw['split']='train';test_raw['split']='test'
applications = pd.concat([train_raw,test_raw],ignore_index=True)
base = applications.merge(profiles,on='customer_id',how='left',validate='one_to_one',sort=False).set_index('customer_id')
assert base.index.is_unique and set(train_raw.customer_id).isdisjoint(test_raw.customer_id)
assert applications.decision_datetime.gt(applications.application_datetime).all()
display(profiles.describe(include='all').T)
fig,axes=plt.subplots(1,2,figsize=(11,3.5))
profiles.age.value_counts().sort_index().plot.bar(ax=axes[0])
axes[0].set(title='Usia: support 23–60',xlabel='Usia',ylabel='Jumlah')
lag_hours=(applications.decision_datetime-applications.application_datetime).dt.total_seconds()/3600
axes[1].hist(lag_hours,bins=np.linspace(0,120,121),color='#0284c7')
axes[1].set(title='Delay keputusan: tiga rentang dengan celah',xlabel='Jam',ylabel='Jumlah')
plt.tight_layout();plt.show()
""",
    )
    code(
        cells,
        source("reverse_synthetic_patterns.py", "distribution_summary")
        + "\n\n"
        + source("reverse_synthetic_patterns.py", "aggregate_ratio_grid")
        + "\n\n"
        + source("reverse_synthetic_patterns.py", "reconstruct_basic_rules"),
    )
    section(
        cells,
        "3. Aturan yang dapat direkonstruksi",
        "Rounding berbeda dari noise acak. Bandingkan ceil, floor, dan nearest integer "
        "untuk formula anuitas. Probabilitas campuran workflow tetap hipotesis, meski support dan celahnya teramati.",
        """
audit={'basic_rules':reconstruct_basic_rules(profiles,applications)}
display(pd.Series(audit['basic_rules']['installment']))
display(pd.DataFrame(audit['basic_rules']['workflow_lag_mixture']['components']))
print('Celah 3–6 jam:',audit['basic_rules']['workflow_lag_mixture']['gap_3_6_hours_rows'])
print('Celah 30–36 jam:',audit['basic_rules']['workflow_lag_mixture']['gap_30_36_hours_rows'])
""",
    )
    md(
        cells,
        """## 4. Probe RNG yang terbatas dan dapat dibantah

Hipotesis: usia berasal dari stream integer uniform 23..60. Uji pertama memeriksa
PCG64 dan MT19937 pada 10.006 seed, prefix 12 usia, offset dalam 512 draw, dan stride
1–16. Kecocokan prefix wajib diverifikasi pada seluruh 100.000 usia.

Hipotesis tambahan: ID merupakan hash template umum, atau RNG per customer diinisialisasi
dari potongan ID/indeks profil. Memilih offset terbaik menaikkan match rate secara kebetulan;
match beberapa persen bukan penemuan seed. Hasil negatif tidak menyingkirkan urutan
call yang diselingi, row shuffle, library lain, atau seed di luar rentang.

[Dokumentasi NumPy tentang syarat reproduksi stream RNG](https://numpy.org/doc/stable/reference/random/compatibility.html).
""",
    )
    for name in [
        "probe_age_sequence",
        "probe_id_hashes",
        "probe_per_customer_rng",
        "probe_raw_identifier_stream",
        "untemper_mt_word",
        "predict_mt_after_observed_words",
        "probe_mt_state_clone",
    ]:
        code(cells, source("reverse_rng_probe.py", name))
    code(
        cells,
        """
if RERUN_RNG_PROBE:
    rng_report={'numpy_version':np.__version__,
                'age_stream':probe_age_sequence(profiles.age.to_numpy(),RNG_SEED_LIMIT),
                'id_templates':probe_id_hashes(profiles.customer_id.to_numpy()),
                'per_customer_seed_probe':probe_per_customer_rng(profiles.customer_id.to_numpy(),profiles.age.to_numpy())}
    rng_report['raw_identifier_stream_probe']=probe_raw_identifier_stream(profiles.customer_id.to_numpy(),RNG_SEED_LIMIT)
    rng_report['mt_state_clone_probe']=probe_mt_state_clone(profiles.customer_id.to_numpy())
    rng_report['original_generator_seed_recovered']=any(h['exact_full_sequence'] for h in rng_report['age_stream']['hits'])
else:
    rng_report=json.loads((ROOT/'artifacts/reverse_engineering/rng_probe.json').read_text())
(OUT/'rng_probe.json').write_text(json.dumps(rng_report,indent=2))
print('Seed asli ditemukan:',rng_report['original_generator_seed_recovered'])
display(pd.DataFrame(rng_report['per_customer_seed_probe']['results']))
print('Kecocokan stream umur:',rng_report['age_stream']['hits'])
print('Kecocokan hash ID:',rng_report['id_templates']['confirmed_three_row_matches'])
print('Kecocokan ID sebagai raw RNG:',rng_report['raw_identifier_stream_probe']['first_id_exact_matches'])
display(pd.DataFrame(rng_report['mt_state_clone_probe']['results']))
print('Positive control cloning MT berhasil:',rng_report['mt_state_clone_probe']['python_positive_control_passed'],rng_report['mt_state_clone_probe']['numpy_positive_control_passed'])
""",
    )
    md(
        cells,
        """## 5. Memisahkan preferensi harga dan noise item

Model pendekatan: `log(unit_price) = efek kategori + efek customer + efek order + noise item`.
Selisih dua item dalam order yang sama membatalkan customer/order. Varians efek order
dipisahkan dari noise item dengan koreksi ukuran basket. Precision weighting dan empirical
Bayes mengestimasi preferensi customer beserta standard error.

Semua basket boleh mengungkapkan pilihan harga, termasuk cancelled; nilai cancelled
tidak dihitung sebagai spending. Parameter dipelajari dari sejarah customer train tanpa target.
Intercept kategori dan efek customer memiliki origin bersama yang tidak teridentifikasi;
mean efek customer nol hanya konvensi. Estimasi bukan nilai laten asli yang diketahui.
""",
    )
    code(cells, source("reverse_synthetic_patterns.py", "fit_price_latent"))
    md(
        cells,
        """## 6. Jitter biro, income, dan pembayaran

Biro: pasangan provider pada aplikasi yang sama mengurangi variasi kemampuan customer.
Asumsi independensi memberi `Var(A-B)=Var(error_A)+Var(error_B)`. Income memakai
selisih log antarbulan; tail/shock/tren dapat mencampur noise efektif. Pembayaran memakai
prior beta-binomial dari frekuensi historis, dengan maturity filter untuk ambang DPD.
Dependensi cicilan bisa membuat uncertainty terlalu optimistis. Label default tidak dipakai.
""",
    )
    for name in [
        "fit_bureau_latent",
        "fit_account_latent",
        "beta_binomial_prior",
        "fit_repayment_latent",
    ]:
        code(cells, source("reverse_synthetic_patterns.py", name))
    section(
        cells,
        "7. Jalankan inferensi pada data mentah",
        "Hasil per customer disimpan di folder runtime notebook. Bila fitur eksperimen tersedia, "
        "semua 21 kolom diperiksa identik dengan hasilnya. Cache audit hanya dipakai jika flag dimatikan.",
        """
if RECOMPUTE_LATENT_AUDIT:
    price,audit['ecommerce'],price_samples=fit_price_latent(base)
    bureau,audit['bureau'],bureau_difference=fit_bureau_latent(base)
    account,audit['account'],income_difference=fit_account_latent(base)
    repayment,audit['repayment']=fit_repayment_latent(base)
    latent_features=pd.concat([price,bureau,account,repayment],axis=1)
    assert latent_features.index.equals(base.index) and latent_features.columns.is_unique
    assert not any('target' in c for c in latent_features.columns)
    latent_features.reset_index().to_parquet(OUT/'latent_features.parquet',index=False)
    audit['target_used_in_latent_feature_estimation']=False
    audit['added_features']=latent_features.columns.tolist()
    (OUT/'pattern_audit.json').write_text(json.dumps(audit,indent=2))
    stored_features=ROOT/'data/processed/application_features_reverse.parquet'
    if stored_features.exists():
        stored=pd.read_parquet(stored_features,columns=['customer_id']+latent_features.columns.tolist()).set_index('customer_id')
        pd.testing.assert_frame_equal(latent_features,stored,check_exact=True)
        print('Seluruh fitur laten identik dengan eksperimen.')
    display(latent_features.head())
else:
    audit=json.loads((ROOT/'artifacts/reverse_engineering/pattern_audit.json').read_text())
display(pd.Series({k:audit['ecommerce'][k] for k in ['estimated_item_log_noise_sd','estimated_shared_order_log_noise_sd','estimated_customer_effect_prior_sd','order_split_customer_effect_correlation']}))
display(pd.Series(audit['bureau']['estimated_provider_noise_sd']))
display(pd.Series(audit['account']['estimated_iid_log_income_noise_sd_by_employment']))
display(pd.DataFrame(audit['repayment']['beta_binomial_priors']).T)
""",
    )
    section(
        cells,
        "8. Plot distribusi noise",
        "Kurva normal hanya pembanding bentuk distribusi. Parameter terfit tidak mengidentifikasi "
        "RNG, dan fitted KS distance tidak digunakan sebagai p-value normalitas.",
        """
if RECOMPUTE_LATENT_AUDIT:
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    panels=[(price_samples['item_pair_difference'],'Selisih log harga antaritem dalam order'),
            (bureau_difference,'Selisih skor A dan B pada aplikasi sama'),
            (income_difference,'Perubahan log income antarbulan'),
            (price_samples['latent_price'],'Estimasi efek preferensi customer')]
    for ax,(values,title) in zip(axes.flat,panels):
        values=np.asarray(values);values=values[np.isfinite(values)]
        ax.hist(values,bins=80,density=True,color='#0284c7',alpha=.65)
        mean,sd=stats.norm.fit(values);grid=np.linspace(np.quantile(values,.001),np.quantile(values,.999),250)
        ax.plot(grid,stats.norm.pdf(grid,mean,sd),color='#ea580c')
        ax.set(title=title,xlabel='Residual / nilai laten',ylabel='Density')
    plt.tight_layout();plt.show()
""",
    )
    md(
        cells,
        """## 9. Generator pendekatan dan demo denoising

Kode ini menggambarkan beberapa pola yang ditemukan. **Seed demo adalah pilihan kita**.
Probabilitas kategori dan ukuran basket dalam simulasi harga sengaja sederhana; prototype
tidak mereproduksi seluruh populasi, urutan RNG, atau label test. Known latent tersedia
hanya dalam simulasi sehingga RMSE denoising bisa diukur; hasil itu bukan KS kompetisi.
""",
    )
    for name in ["simulate_workflow", "simulate_price_measurements"]:
        code(cells, source("synthetic_generator_prototype.py", name))
    code(
        cells,
        """
demo_workflow=simulate_workflow(audit,n=10000,demo_seed=91403)
demo_price=simulate_price_measurements(audit,n_customers=2000,orders_per_customer=12,demo_seed=73103)
assert demo_price.item_total.mod(demo_price.quantity*1000).eq(0).all()
assert demo_price.unit_price.ge(5000).all()
demo_price['adjusted_log_price']=np.log(demo_price.unit_price)-demo_price.product_category.map(audit['ecommerce']['category_log_intercepts'])
observed=demo_price.groupby('synthetic_customer_index').adjusted_log_price.mean()
truth=demo_price.groupby('synthetic_customer_index').known_demo_latent.first()
counts=demo_price.groupby('synthetic_customer_index').size()
prior_var=audit['ecommerce']['estimated_customer_effect_prior_sd']**2
noise_var=audit['ecommerce']['estimated_item_log_noise_sd']**2
posterior=observed*prior_var/(prior_var+noise_var/counts)
display(pd.Series({'RMSE_rata_rata_item_demo':np.sqrt(np.mean((observed-truth)**2)),
                   'RMSE_posterior_sederhana_demo':np.sqrt(np.mean((posterior-truth)**2))}))
display(demo_workflow.groupby('route').observed_delay_seconds.agg(['size','min','max']))
display(demo_price.head())
""",
    )
    md(
        cells,
        """## 10. Uji apakah pola laten membantu KS

CatBoost 500 pohon dan spline dilatih pada tiga fold yang sama dengan versi 5.
Imputasi/encoder model hanya dilatih pada ti. Parameter laten memakai sejarah tanpa label
target. Bobot 25%/50%/100% diperiksa; hasil screening memakai label yang juga dilaporkan,
sehingga mengandung optimisme seleksi. Angka tiga fold tidak dibandingkan langsung dengan
KS sepuluh fold. Tidak ada file submission yang diganti.

Pada clone baru, jalankan notebook 06 dan skrip pembentuk fitur reverse sebelum mengaktifkan
validasi model ini. Audit/simulasi di bagian sebelumnya dapat berjalan tanpa prasyarat model.
""",
    )
    code(
        cells,
        source("optimize_ks.py", "ks")
        + "\n\n"
        + source("optimize_ks.py", "metrics")
        + "\n\n"
        + source("spline_model.py", "prepare")
        + "\n\n"
        + source("model_v5.py", "INTERACTION_FEATURES")
        + "\n\n"
        + source("model_v5.py", "fit_spline_plus"),
    )
    code(
        cells,
        """
def fit_reverse_model(spec,data,ti):
    if spec['family']=='spline_plus':
        return fit_spline_plus(spec,data,ti)
    columns=spec['columns'];x=data[columns].copy()
    categorical=x.select_dtypes(include='object').columns.tolist()
    for c in categorical:x[c]=x[c].fillna('MISSING').astype(str)
    model=CatBoostClassifier(iterations=500,learning_rate=.04,loss_function='Logloss',
                             eval_metric='AUC',random_seed=2026,thread_count=8,verbose=False,
                             allow_writing_files=False,**spec['params'])
    model.fit(x.iloc[ti],data.target.iloc[ti].astype(int),cat_features=categorical)
    return {'family':'cat','model':model,'columns':columns,'categorical':categorical,'weight':6}

def predict_reverse_model(bundle,data):
    if bundle['family']=='cat':
        x=data[bundle['columns']].copy()
        for c in bundle['categorical']:x[c]=x[c].fillna('MISSING').astype(str)
        p=bundle['model'].predict_proba(x)[:,1]
    else:
        x=prepare(data,bundle['columns'],bundle['categorical'])
        with threadpool_limits(limits=4):
            p=bundle['model'].predict_proba(bundle['encoder'].transform(x))[:,1]
    weight=bundle.get('weight',1)
    return p/(weight*(1-p)+p)
""",
    )
    code(
        cells,
        """
feature_file=ROOT/'data/processed/application_features_reverse.parquet'
source_oof=ROOT/'artifacts/optimization/v5/selection_oof_tabm.parquet'
validation_report=json.loads((ROOT/'artifacts/reverse_engineering/validation_comparison.json').read_text())
if RECOMPUTE_MODEL_VALIDATION and feature_file.exists() and source_oof.exists():
    frame=pd.read_parquet(feature_file)
    data=frame[frame.split.eq('train')].reset_index(drop=True)
    previous=pd.read_parquet(source_oof)
    assert previous.application_id.equals(data.application_id)
    assert previous.target.astype(int).equals(data.target.astype(int))
    oof=previous[['application_id','target']].copy()
    oof['version5']=.75*previous.version4+.25*previous.v5_tabm_weight1
    specs=json.loads((ROOT/'artifacts/reverse_engineering/validation_definitions.json').read_text())
    folds=list(StratifiedKFold(3,shuffle=True,random_state=8247).split(data,data.target))
    for spec in specs:
        assert not set(spec['columns']) & EXCLUDED
        oof[spec['name']]=np.nan
        for fold,(ti,vi) in enumerate(folds,1):
            filename=f'fold{fold}_{spec["name"]}.joblib'
            local=OUT/'models'/filename;local.parent.mkdir(exist_ok=True)
            historical=ROOT/'artifacts/reverse_engineering/models'/filename
            cache=local if local.exists() else historical
            if cache.exists() and not FORCE_RETRAIN_REVERSE_MODELS:
                bundle=joblib.load(cache)
                assert bundle['columns']==spec['columns']
                actual=bundle['model'].get_params()
                if spec['family']=='cat':
                    assert all(actual.get(k)==v for k,v in spec['params'].items())
                    assert actual['iterations']==500 and actual['random_seed']==2026
                else:
                    assert actual['C']==spec['params']['C']
            else:
                bundle=fit_reverse_model(spec,data,ti)
                joblib.dump(bundle,local,compress=3)
            oof.loc[vi,spec['name']]=predict_reverse_model(bundle,data.iloc[vi])
            print(spec['name'],'fold',fold,'selesai')
    historical=ROOT/'artifacts/reverse_engineering/validation_oof.parquet'
    if historical.exists() and not FORCE_RETRAIN_REVERSE_MODELS:
        expected=pd.read_parquet(historical)
        pd.testing.assert_frame_equal(oof,expected,check_exact=True)
        print('Seluruh prediksi OOF identik dengan eksperimen.')
    comparison=[]
    for name in oof.columns[2:]:
        for w in ([0] if name=='version5' else [.25,.5,1]):
            p=oof.version5 if name=='version5' else (1-w)*oof.version5+w*oof[name]
            values=[ks(oof.target.iloc[vi],p.iloc[vi]) for _,vi in folds]
            comparison.append({'name':name,'new_model_weight':w,**metrics(oof.target,p),
                               'fold_ks':values,'mean_fold_ks':float(np.mean(values))})
    validation_report['comparison']=comparison
    (OUT/'validation_comparison.json').write_text(json.dumps(validation_report,indent=2))
else:
    print('Menampilkan validasi historis. Untuk menghitung ulang, siapkan fitur/OOF versi 5 dan fitur reverse.')
display(pd.DataFrame(validation_report['comparison'])[['name','new_model_weight','ks','mean_fold_ks','roc_auc']])
""",
    )
    md(
        cells,
        """## 11. Kesimpulan dan batas yang harus diperiksa

Beberapa aturan observasi berhasil direkonstruksi: ceil anuitas, grid harga sebelum quantity,
carry saldo, dan support campuran uniform delay. Skala jitter harga/biro dapat diestimasi dari
pengukuran berulang. **Seed dan bitstream generator asli belum berhasil dipulihkan.**

Fitur laten belum meningkatkan KS secara konsisten pada percobaan ini. Kenaikan KS gabungan
spline sangat kecil dan rata-rata fold turun; submission sebelumnya tetap digunakan.
Mean customer nol, Gaussian error, provider independence dan iid income adalah asumsi,
bukan fakta tentang kode generator. Model mendekati distribusi data tidak sama dengan
menemukan program RNG asli atau mengetahui label test.
""",
    )
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
            "logika": {
                "purpose": "rng_and_jitter_reverse_engineering",
                "self_contained_code": True,
                "original_rng_seed_known": False,
                "changes_submission": False,
            },
        },
    )
    nbf.validate(notebook)
    path = DEST / "07_reverse_engineering_rng.ipynb"
    nbf.write(notebook, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    path = build()
    print("Created:", path, flush=True)
    if args.execute:
        from nbclient import NotebookClient

        notebook = nbf.read(path, as_version=4)
        try:
            NotebookClient(
                notebook,
                timeout=1800,
                kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()
        finally:
            nbf.write(notebook, path)
        print("Passed:", path, flush=True)


if __name__ == "__main__":
    main()
