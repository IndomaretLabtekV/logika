"""Create readable, executable notebooks for all five historical submissions."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import pprint
import textwrap

import nbformat as nbf
try:
    import black
except ImportError:
    black = None

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'notebook'


def source(filename, name):
    text = (ROOT / 'scripts' / filename).read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.Assign)) and
                (getattr(n, 'name', None) == name or
                 isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)))
    return ast.get_source_segment(text, node)


def body(filename, name):
    text = (ROOT / 'scripts' / filename).read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent('\n'.join(text.splitlines()[node.body[0].lineno-1:node.end_lineno]))


def md(cells, text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(cells, text):
    text = textwrap.dedent(text).strip()
    ast.parse(text)
    if black is not None:
        text = black.format_str(text, mode=black.Mode(line_length=95)).strip()
    cells.append(nbf.v4.new_code_cell(text))


def section(cells, title, explanation, content):
    md(cells, f'## {title}\n\n{explanation}')
    code(cells, content)


def literal(value):
    return pprint.pformat(value, width=95, sort_dicts=False)


SETUP = '''
from pathlib import Path
import gc, hashlib, itertools, json, os, time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier

# Ubah empat pilihan ini jika ingin membangun dan melatih semuanya ulang.
FORCE_REBUILD_FEATURES = False
FORCE_RETRAIN_MODELS = False
FORCE_RECOMPUTE_VALIDATION = False
RUN_OPTIONAL_SEARCH = False

ROOT = Path(os.environ.get('LOGIKA_PROJECT_DIR', Path.cwd())).resolve()
if not (ROOT / 'scripts/analyze.py').exists():
    ROOT = next(p for p in ROOT.parents if (p / 'scripts/analyze.py').exists())
RAW = ROOT / 'data'
PROCESSED = Path(os.environ.get('LOGIKA_FEATURE_DIR', ROOT / 'data/processed'))
OUT = ROOT / 'artifacts/notebook_runs' / RUN_NAME
TABLES = OUT / 'tables'
for folder in [PROCESSED, OUT, TABLES, OUT / 'models']:
    folder.mkdir(parents=True, exist_ok=True)
KEY, TARGET = 'customer_id', 'target'
EXCLUDED = {'application_id', 'customer_id', 'application_datetime',
            'decision_datetime', 'target', 'split'}
pd.set_option('display.max_columns', 12)
pd.set_option('display.max_rows', 20)
print('Proyek:', ROOT)
print('Output notebook:', OUT / 'submission.csv')
'''

EDA = '''
train_raw = pd.read_csv(RAW / 'train_applications.csv',
                        parse_dates=['application_datetime', 'decision_datetime'])
test_raw = pd.read_csv(RAW / 'test_applications.csv',
                       parse_dates=['application_datetime', 'decision_datetime'])
sample = pd.read_csv(RAW / 'sample_submission.csv')
profiles_eda = pd.read_csv(RAW / 'customer_profiles.csv')
assert train_raw.application_id.is_unique and test_raw.application_id.is_unique
assert set(train_raw.customer_id).isdisjoint(test_raw.customer_id)
assert set(test_raw.application_id) == set(sample.application_id)
assert train_raw.target.isin([0, 1]).all()
display(pd.DataFrame([
    {'dataset': 'train', 'baris': len(train_raw), 'kolom': len(train_raw.columns),
     'target_positif': int(train_raw.target.sum()), 'rasio_positif': train_raw.target.mean()},
    {'dataset': 'test', 'baris': len(test_raw), 'kolom': len(test_raw.columns),
     'target_positif': np.nan, 'rasio_positif': np.nan},
]))
display(pd.DataFrame({'tipe': train_raw.dtypes.astype(str),
                      'missing': train_raw.isna().sum()}))
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
train_raw.target.value_counts().sort_index().plot.bar(ax=axes[0], rot=0)
axes[0].set(title='Jumlah aplikasi menurut target', xlabel='Target', ylabel='Jumlah')
monthly = train_raw.groupby(train_raw.decision_datetime.dt.to_period('M')).target.agg(['mean', 'size'])
monthly['mean'].plot(marker='o', ax=axes[1])
axes[1].set(title='Proporsi target 1 per bulan', xlabel='Bulan keputusan', ylabel='Proporsi')
plt.tight_layout(); plt.show()
profile = train_raw[['customer_id', 'target']].merge(profiles_eda, on='customer_id', validate='one_to_one')
display(profile.groupby('employment_type').target.agg(aplikasi='size', rasio_target_1='mean'))
print('Urutan sample sama dengan test:', sample.application_id.equals(test_raw.application_id))
del profile, profiles_eda
'''

CV_LOOP = '''
def cv_predict(specs, frames, fold_count, fold_seed, seed_per_fold, source_dir,
               source_template, source_oof=None):
    """Latih pada ti, validasi pada vi, lalu rata-ratakan probabilitas test antarfold."""
    reference = next(iter(frames.values()))
    train = reference[reference.split.eq('train')].reset_index(drop=True)
    test = reference[reference.split.eq('test')].reset_index(drop=True)
    y = train.target.astype(int)
    folds = list(StratifiedKFold(n_splits=fold_count, shuffle=True,
                                random_state=fold_seed).split(train, y))
    oof = pd.DataFrame({'application_id': train.application_id, 'target': y})
    predicted = pd.DataFrame({'application_id': test.application_id})
    cached_oof = None
    if source_oof and (ROOT / source_oof).exists() and not (
            FORCE_RETRAIN_MODELS or FORCE_RECOMPUTE_VALIDATION):
        cached_oof = pd.read_parquet(ROOT / source_oof)
        assert cached_oof.application_id.equals(train.application_id)
        assert cached_oof.target.astype(int).equals(y)
    for spec in specs:
        name = spec['name']
        frame = frames[spec['feature_set']]
        data = frame[frame.split.eq('train')].reset_index(drop=True)
        testing = frame[frame.split.eq('test')].reset_index(drop=True)
        assert data.application_id.equals(train.application_id)
        assert testing.application_id.equals(test.application_id)
        assert data.target.astype(int).equals(y)
        assert not set(spec['columns']) & EXCLUDED
        oof[name] = np.nan
        predicted[name] = 0.0
        for fold, (ti, vi) in enumerate(folds, 1):
            seed = 2026 + fold if seed_per_fold else 2026
            filename = source_template.format(fold=fold, name=name.replace('/', '_'))
            historical = ROOT / source_dir / filename
            local = OUT / 'models' / f'{fold_count}fold_{fold_seed}_{filename}'
            cache = local if local.exists() else historical
            if cache.exists() and not FORCE_RETRAIN_MODELS:
                bundle = joblib.load(cache)
                assert bundle['columns'] == spec['columns'], 'Daftar fitur cache berbeda.'
                actual = bundle['model'].get_params()
                if spec['family'] == 'spline':
                    assert actual['C'] == spec['params']['C'], 'Ubah FORCE_RETRAIN_MODELS jika mengubah parameter.'
                else:
                    assert all(actual.get(k) == v for k, v in spec['params'].items()), 'Parameter cache berbeda; aktifkan FORCE_RETRAIN_MODELS.'
                    seed_key = 'random_seed' if spec['family'] == 'cat' else 'random_state'
                    assert actual[seed_key] == seed
                    tree_key = 'iterations' if spec['family'] == 'cat' else 'n_estimators'
                    assert actual[tree_key] == spec['iterations'], 'Jumlah pohon cache berbeda; aktifkan FORCE_RETRAIN_MODELS.'
            else:
                if spec['family'] == 'spline':
                    bundle = fit(data, ti)
                else:
                    bundle, _, _ = fit_candidate(spec, data, ti, None, spec['columns'],
                                                  iterations=spec['iterations'], seed=seed)
                joblib.dump(bundle, local, compress=3)
            # Cache OOF berasal dari validasi historis, bukan prediksi model full train.
            if cached_oof is not None and name in cached_oof:
                oof.loc[vi, name] = cached_oof[name].iloc[vi].to_numpy()
            else:
                predictor = predict if spec['family'] == 'spline' else predict_bundle
                oof.loc[vi, name] = predictor(bundle, data.iloc[vi])
            predictor = predict if spec['family'] == 'spline' else predict_bundle
            predicted[name] += predictor(bundle, testing) / fold_count
        print(name, 'KS OOF:', round(ks(y, oof[name]), 5))
    assert oof.notna().all().all()
    assert np.isfinite(predicted.drop(columns='application_id')).all().all()
    return oof, predicted
'''

STAGE_SEARCH = '''
def review_stage_search(specs, frame, saved_oof, source_dirs):
    """Periksa jumlah pohon tanpa melatih ulang jika prediksi OOF historis tersedia."""
    data = frame[frame.split.eq('train')].reset_index(drop=True)
    y = data.target.astype(int)
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=8247).split(data, y))
    if (ROOT / saved_oof).exists() and not RUN_OPTIONAL_SEARCH:
        oof = pd.read_parquet(ROOT / saved_oof)
        assert oof.application_id.equals(data.application_id)
        assert oof.target.astype(int).equals(y)
        print('Review prediksi validasi dari eksperimen historis.')
    elif RUN_OPTIONAL_SEARCH:
        oof = pd.DataFrame({'application_id': data.application_id, 'target': y})
        for spec in specs:
            for trees in spec['stages']:
                oof[f'{spec["name"]}_{trees}'] = np.nan
            for fold, (ti, vi) in enumerate(folds, 1):
                cache = ROOT / source_dirs[spec['name']] / f'fold{fold}_{spec["name"]}.joblib'
                if cache.exists() and not FORCE_RETRAIN_MODELS:
                    bundle = joblib.load(cache)
                else:
                    bundle, _, _ = fit_candidate(spec, data, ti, None, spec['columns'],
                                                  iterations=max(spec['stages']), seed=2026)
                for trees in spec['stages']:
                    oof.loc[vi, f'{spec["name"]}_{trees}'] = stage_probability(bundle, data.iloc[vi], trees)
    else:
        print('Prediksi pencarian belum tersedia; aktifkan RUN_OPTIONAL_SEARCH untuk menghitungnya.')
        return None
    rows = []
    for name in oof.columns[2:]:
        scores = metrics(y, oof[name])
        scores['mean_fold_ks'] = np.mean([ks(y.iloc[vi], oof[name].iloc[vi]) for _, vi in folds])
        rows.append({'kandidat': name, **scores})
    display(pd.DataFrame(rows).sort_values('mean_fold_ks', ascending=False))
    return oof
'''

BLEND_REVIEW = '''
def review_blend_weights(oof, anchor, stage):
    """Terapkan grid bobot dan kriteria pemilihan historis pada OOF tiga fold."""
    y = oof.target.astype(int)
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=8247).split(oof, y))
    baseline_ks = ks(y, oof[anchor])
    baseline_folds = [ks(y.iloc[vi], oof[anchor].iloc[vi]) for _, vi in folds]
    names = [c for c in oof if c not in ['application_id', 'target', anchor]]
    if stage == 3:
        # Pilih satu jumlah pohon terbaik per konfigurasi, lalu tiga anggota baru.
        winners = {}
        for name in names:
            prefix = name.rsplit('_', 1)[0]
            score = np.mean([ks(y.iloc[vi], oof[name].iloc[vi]) for _, vi in folds])
            if prefix not in winners or score > winners[prefix][1]:
                winners[prefix] = (name, score)
        pool = [n for n, _ in sorted(winners.values(), key=lambda item: item[1], reverse=True)[:3]] + [anchor]
        recipes = [{n: 1.0} for n in pool]
        for a, b in itertools.combinations(pool, 2):
            recipes.extend([{a: w, b: 1-w} for w in [.25, .5, .75]])
        recipes.extend([{n: 1/3 for n in members} for members in itertools.combinations(pool, 3)])
    else:
        recipes = [{anchor: 1.0}]
        for name in names:
            recipes.extend([{anchor: 1-w, name: w} for w in [.25, .5, .75]])
        for name in [n for n in names if n != 'spline']:
            for weights in [[.5, .25, .25], [1/3, 1/3, 1/3], [.25, .5, .25], [.25, .25, .5]]:
                recipes.append(dict(zip([anchor, name, 'spline'], weights)))
    results = []
    for recipe in recipes:
        probability = sum(w * oof[n] for n, w in recipe.items())
        per_fold = [ks(y.iloc[vi], probability.iloc[vi]) for _, vi in folds]
        gains = np.array(per_fold) - baseline_folds
        pooled = ks(y, probability)
        mean = float(np.mean(per_fold))
        if stage == 3:
            eligible = pooled > baseline_ks and mean > np.mean(baseline_folds)
        else:
            eligible = (pooled >= baseline_ks+.002 and mean >= np.mean(baseline_folds)+.001
                        and (gains > 0).sum() >= 2 and gains.min() >= -.0025)
        results.append({'recipe': recipe, 'ks': pooled, 'mean_fold_ks': mean,
                        'fold_gains': gains.tolist(), 'eligible': bool(eligible)})
    ranking = pd.DataFrame(results).sort_values(['mean_fold_ks', 'ks'], ascending=False)
    display(ranking.head(8))
    if ranking.eligible.any():
        print('Resep yang lolos dan memiliki mean fold KS tertinggi:', ranking[ranking.eligible].iloc[0].recipe)
    return ranking
'''

EXPORT = '''
# test_probability berada dalam urutan frame fitur, sehingga harus di-join ke sample.
predictions = pd.DataFrame({'application_id': test_data.application_id.to_numpy(),
                            'default_probability': np.asarray(test_probability)})
assert predictions.application_id.is_unique
assert set(predictions.application_id) == set(sample.application_id)
submission = sample[['application_id']].merge(predictions, on='application_id',
                                              how='left', validate='one_to_one', sort=False)
assert submission.shape == (len(sample), 2)
assert submission.application_id.equals(sample.application_id)
assert submission.columns.tolist() == ['application_id', 'default_probability']
assert np.isfinite(submission.default_probability).all()
assert submission.default_probability.between(0, 1).all()
destination = OUT / 'submission.csv'
submission.to_csv(destination, index=False, float_format='%.12f')
reloaded = pd.read_csv(destination)
assert reloaded.application_id.equals(sample.application_id)
assert reloaded.columns.tolist() == sample.columns.tolist()
assert reloaded.default_probability.notna().all()
assert reloaded.default_probability.between(0, 1).all()
result = {'file': str(destination.relative_to(ROOT)), 'rows': len(reloaded),
          'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
          'historical_file': CANONICAL_FILE, 'historical_public_score': PUBLIC_SCORE}
historical = ROOT / "results/teman" / CANONICAL_FILE
if historical.exists():
    checkpoint = pd.read_csv(historical)
    assert checkpoint.application_id.equals(reloaded.application_id)
    result['maximum_probability_difference'] = float(np.max(np.abs(
        checkpoint.default_probability.to_numpy() - reloaded.default_probability.to_numpy())))
    result['matches_historical_predictions_1e9'] = result['maximum_probability_difference'] < 1e-9
    result['exact_sha256_match'] = result['sha256'] == hashlib.sha256(historical.read_bytes()).hexdigest()
(OUT / 'verification.json').write_text(json.dumps(result, indent=2))
display(pd.Series(result, name='Pemeriksaan submission'))
display(reloaded.default_probability.describe().to_frame('probabilitas'))
print('CSV siap:', destination)
'''


def feature_cells(cells, level):
    section(cells, '4. Fitur dasar: alat bantu dan pengaturan cache',
            'Sel fitur berikut berisi kode agregasi yang digunakan pada eksperimen asli. '
            'Jika parquet tersedia, blok `if NEED_...` dilewati. Aktifkan '
            '`FORCE_REBUILD_FEATURES` untuk membaca ulang seluruh riwayat. '
            'Kode tetap terlihat dan dapat diedit langsung di notebook.',
            source('analyze.py', 'safe_ratio') + '\n\n' + source('analyze.py', 'read') + '\n\n' +
            source('extend_features.py', 'ratio') + '\n\n' +
            "NEED_BASE = FORCE_REBUILD_FEATURES or not (PROCESSED / 'application_features.parquet').exists()\n" +
            ("NEED_BASE = NEED_BASE or not (PROCESSED / 'bureau_latest.parquet').exists()\n" if level >= 3 else '') +
            "print('Bangun fitur dasar:', NEED_BASE)")
    chunks = [
        ('Persiapan audit', 'Setiap aplikasi memiliki satu customer. Audit mencatat duplikasi, relasi tabel, missing value, dan ketersediaan riwayat.', None),
        ('Profil dan rasio kemampuan bayar', 'Gabungkan profil satu-ke-satu; rasio cicilan/pendapatan lebih mudah dipelajari model daripada dua jumlah rupiah terpisah.', 'inspect(train.drop'),
        ('Riwayat aplikasi', 'Ambil keputusan sebelum cutoff. Hitung persetujuan, penolakan, dan jarak waktu aplikasi terakhir.', 'name = "previous_applications"'),
        ('Riwayat pembayaran', 'DPD adalah days past due. Snapshot pembayaran harus diketahui sebelum keputusan, termasuk waktu pembayaran terakhir.', 'name = "repayment_history"'),
        ('Perilaku rekening', 'Hanya bulan kalender yang sudah selesai yang digunakan; rangkum pemasukan, pengeluaran, saldo, dan hari saldo rendah.', 'name = "account_behavior"'),
        ('Biro kredit', 'Laporan harus sudah dibuat dan tersedia sebelum cutoff. Gunakan laporan terakhir per provider; saldo provider tidak dijumlahkan karena cakupan dapat tumpang tindih.', 'name = "bureau_observations"'),
        ('Transaksi e-commerce', '`order_total` berulang di setiap barang. Deduplicasi pesanan sebelum menjumlahkan belanja; transaksi cancelled bukan pengeluaran.', 'name = "ecommerce_transactions"'),
        ('Simpan fitur dasar', 'Nilai tak terhingga menjadi NaN. Nilai kosong struktural tetap tersedia sebagai sinyal untuk model pohon.', 'base.replace([np.inf'),
    ]
    content = body('analyze.py', 'build_features').replace('nonlocal base', 'global base').replace('return base', 'features_base = base.copy()')
    split_feature_code(cells, content, chunks, 'NEED_BASE')
    code(cells, "features_base = pd.read_parquet(PROCESSED / 'application_features.parquet')\nprint('Fitur dasar:', features_base.shape)")
    if level < 2:
        return
    section(cells, '5. Fitur extended: perilaku terbaru dan beban utang',
            'Jendela 30/60/90/180/365 hari membedakan pola terbaru dari seluruh sejarah. '
            'Perkiraan sisa pokok menggunakan amortisasi kontraktual dan jumlah cicilan yang diamati; nilainya bukan saldo aktual.',
            "NEED_EXTENDED = FORCE_REBUILD_FEATURES or not (PROCESSED / 'application_features_extended.parquet').exists()")
    content = body('extend_features.py', 'main').replace("(ROOT/'artifacts/extended_feature_audit.json')", "(OUT/'extended_feature_audit.json')")
    chunks = [
        ('Persiapan agregasi extended', 'Semua blok fitur dipertahankan pada indeks customer yang sama.', None),
        ('Keterlambatan dan hasil per pinjaman', 'Rangkum DPD, pembayaran parsial, keterlambatan terbaru, serta hasil per pinjaman.', "rep=history('repayment_history'"),
        ('Pinjaman aktif dan cicilan kontraktual', 'Estimasi jumlah cicilan tersisa dan total cicilan aktif membantu mengukur beban utang.', "prev=history('previous_applications'"),
        ('Riwayat dan perubahan biro', 'Bandingkan snapshot aplikasi sekarang, riwayat provider, dan perubahan skor.', "bureau=history('bureau_observations'"),
        ('Tren rekening', 'Bandingkan tiga bulan terakhir terhadap tiga bulan sebelumnya, setelah memeriksa bulan selesai.', "account=history('account_behavior'"),
        ('Jendela belanja', 'Deduplicasi pesanan; hitung jumlah, belanja, pembayaran, dan kategori pada beberapa jendela.', "ecom=history('ecommerce_transactions'"),
        ('Rasio lintas tabel', 'Gabungkan cicilan, pendapatan, utang, saldo, dan belanja; ID tetap dikeluarkan dari model.', 'full=pd.concat([base]+parts'),
    ]
    split_feature_code(cells, content, chunks, 'NEED_EXTENDED')
    code(cells, "features_extended = pd.read_parquet(PROCESSED / 'application_features_extended.parquet')\nprint('Fitur extended:', features_extended.shape)")
    if level < 3:
        return
    section(cells, '6. Fitur versi 3: harga dan shrinkage',
            'Shrinkage mengurangi ketidakstabilan persentase keterlambatan dari customer dengan sedikit cicilan. '
            'Rumusnya `(jumlah kejadian + strength × prior) / (jumlah observasi + strength)`. '
            'Prior berasal dari riwayat pembayaran, bukan label target.',
            "NEED_V3 = FORCE_REBUILD_FEATURES or not (PROCESSED / 'application_features_v3.parquet').exists()")
    content = body('build_features_v3.py', 'main')
    content = content.replace("ROOT/'data/processed/application_features_extended.parquet'", "PROCESSED/'application_features_extended.parquet'")
    content = content.replace("ROOT/'data/processed/bureau_latest.parquet'", "PROCESSED/'bureau_latest.parquet'")
    content = content.replace("ROOT/'data/processed/application_features_v3.parquet'", "PROCESSED/'application_features_v3.parquet'")
    content = content.replace("ROOT/'artifacts/optimization/v3_feature_audit.json'", "OUT/'v3_feature_audit.json'")
    chunks = [
        ('Prior keterlambatan historis', 'Gunakan jumlah kejadian DPD 1/7/30/60/90 dan beberapa kekuatan prior.', None),
        ('Skor biro dengan bobot recency', 'Laporan yang lebih lama memperoleh bobot eksponensial yang lebih kecil.', 'bureau=pd.read_parquet'),
        ('Harga per kategori dan variasi belanja', 'Harga unit adalah item_total/quantity. Kelompokkan kategori, harga, jumlah barang, dan belanja bulanan.', 'ecom=pd.read_csv'),
        ('Cakupan cicilan oleh arus kas', 'Surplus sebelum dan sesudah cicilan membantu merangkum kemampuan bayar berdasarkan transaksi rekening.', 'extra=pd.DataFrame'),
    ]
    split_feature_code(cells, content, chunks, 'NEED_V3')
    code(cells, "features_v3 = pd.read_parquet(PROCESSED / 'application_features_v3.parquet')\nprint('Fitur versi 3:', features_v3.shape)")
    if level < 4:
        return
    section(cells, '7. Fitur versi 4: normalisasi harga dan cicilan yang cukup umur',
            'Harga dinormalisasi terhadap kategori agar kategori yang berbeda dapat diringkas bersama. '
            'Cicilan yang baru jatuh tempo tidak dapat mencapai DPD 90; denominator kejadian tersebut perlu cukup umur. '
            'Normalisasi dan prior menggunakan riwayat customer training tanpa label target.',
            "NEED_V4 = FORCE_REBUILD_FEATURES or not (PROCESSED / 'application_features_v4.parquet').exists()")
    content = body('build_features_v4.py', 'main')
    content = content.replace("ROOT / 'data/processed/application_features_v3.parquet'", "PROCESSED / 'application_features_v3.parquet'")
    content = content.replace("ROOT / 'data/processed/application_features_v4.parquet'", "PROCESSED / 'application_features_v4.parquet'")
    content = content.replace("ROOT / 'artifacts/optimization/v4'", "OUT")
    chunks = [
        ('Harga yang dinormalisasi menurut kategori', 'Kurangi log harga dengan median kategori dari riwayat customer training. Bandingkan bobot per barang, pesanan, dan kategori.', None),
        ('Keterlambatan pada cicilan matang', 'Untuk ambang t hari, gunakan hanya cicilan berumur setidaknya t hari; hitung pula proporsi dengan bobot setara per pinjaman.', 'repayments = pd.read_csv'),
        ('Skor dengan bobot setara antarprovider', 'Pusatkan skor menurut provider dan agregasikan riwayatnya; jangan menjumlahkan saldo provider.', 'bureau = pd.read_csv'),
        ('Simpan fitur versi 4', 'Periksa bahwa ID dan label tidak berubah setelah agregasi.', 'result = pd.concat'),
    ]
    split_feature_code(cells, content, chunks, 'NEED_V4')
    code(cells, "features_v4 = pd.read_parquet(PROCESSED / 'application_features_v4.parquet')\nprint('Fitur versi 4:', features_v4.shape)")


def split_feature_code(cells, content, chunks, guard):
    positions = [0] + [content.index(marker) for _, _, marker in chunks[1:]] + [len(content)]
    for i, (title, explanation, _) in enumerate(chunks):
        md(cells, f'### {title}\n\n{explanation}')
        code(cells, f'if {guard}:\n' + textwrap.indent(content[positions[i]:positions[i+1]].rstrip(), '    '))


def common_cells(cells, version, level):
    names = {1: 'Baseline XGBoost', 2: 'Tuning KS dan submission versi 2',
             3: 'Ensemble KS tiga fold', 4: 'Submission versi 3', 5: 'Submission versi 4'}
    historical = {1: .52623, 2: None, 3: .54710, 4: .55845, 5: None}[version]
    filenames = {1: 'artifacts/optimization/submission_baseline_0_52623.csv', 2: 'submission_v2.csv',
                 3: 'submission_ks.csv', 4: 'submission_v3.csv', 5: 'submission_v4.csv'}
    md(cells, f'''# {names[version]} — EDA sampai submission.csv

    Notebook ini mereview pipeline yang menghasilkan **{filenames[version]}**.
    Skor public historis: **{str(historical) if historical is not None else 'belum tercatat'}**.
    Skor tersebut milik file historis, bukan jaminan untuk eksperimen yang Anda ubah.

    **Urutan membaca:** persiapan → EDA → pembentukan fitur → preprocessing →
    validasi dan pemilihan model → prediksi test → pemeriksaan dan ekspor CSV.

    Jalankan **Kernel → Restart & Run All**. Model/fitur cache dipakai jika tersedia;
    `.fit()` otomatis berjalan jika model belum tersedia. Untuk melatih ulang seluruhnya,
    ubah flag pada sel persiapan. Pencarian parameter yang mahal bersifat opsional.

    Output setiap notebook disimpan pada direktori versinya sendiri dengan nama
    **submission.csv**, agar hasil beberapa versi bisa dibandingkan.
    Notebook tidak mengirim submission ke Kaggle.

    Instal dependensi dari direktori proyek: `python -m pip install -r environments/teman/requirements-optimization.txt -r environments/teman/requirements-notebook.txt`.
    Data mentah harus tersedia di `data/`; unduh sekali dengan `python scripts/download_data.py`.
    Kredensial tidak perlu ditulis dalam notebook.
    ''')
    section(cells, '1. Persiapan lingkungan', 'Semua jalur relatif terhadap proyek. Cache mempercepat review; kode training tetap ada di sel notebook.',
            f"RUN_NAME = {('0'+str(version)+'_'+['baseline','v2','ks','v3','v4'][version-1])!r}\nCANONICAL_FILE = {filenames[version]!r}\nPUBLIC_SCORE = {historical!r}\n" + SETUP)
    section(cells, '2. Memahami target dan data',
            '`target=1` berarti pinjaman mencapai DPD ≥90 dalam 12 bulan setelah origination. '
            'Test tidak memiliki target. Ada satu aplikasi terpilih per customer, sehingga agregasi seluruh tabel riwayat memakai customer_id. '
            'Perbedaan rasio target menurut kelompok bersifat deskriptif dan bukan kesimpulan sebab-akibat.', EDA)
    section(cells, '3. Metrik KS dan risiko kebocoran',
            'KS adalah maksimum selisih dua distribusi kumulatif skor. Dalam ROC, nilainya sama dengan maksimum |TPR−FPR|. '
            'KS mengukur pemisahan; submission tetap berupa probabilitas kontinu. '
            'Model validasi tidak boleh dilatih pada baris validasi. '
            'Informasi riwayat harus tersedia sebelum decision_datetime, dan opaque ID tidak menjadi fitur.',
            '\n\n'.join(source('optimize_ks.py', n) for n in ['ks', 'lgb_ks', 'metrics']))
    feature_cells(cells, level)
    feature_name = {1: 'features_base', 2: 'features_extended', 3: 'features_v3', 4: 'features_v4'}[level]
    section(cells, '8. Periksa frame model dan missing value',
            'Jumlah baris dan kesesuaian label diperiksa sebelum modeling. Missing karena tidak ada riwayat tidak disamakan dengan kejadian pembayaran bernilai nol.',
            f'''model_frame = {feature_name}
train_data = model_frame[model_frame.split.eq('train')].reset_index(drop=True)
test_data = model_frame[model_frame.split.eq('test')].reset_index(drop=True)
assert len(train_data) == len(train_raw) and len(test_data) == len(test_raw)
aligned_labels = train_raw.set_index('application_id').target.reindex(train_data.application_id)
assert np.array_equal(train_data.target.to_numpy(), aligned_labels.to_numpy())
assert set(train_data.customer_id).isdisjoint(test_data.customer_id)
assert test_data.target.isna().all()
base_columns = [c for c in features_base if c not in EXCLUDED and features_base[c].nunique(dropna=True) > 1]
numeric_columns = [c for c in model_frame.select_dtypes(include='number') if c not in EXCLUDED]
assert not np.isinf(model_frame[numeric_columns].to_numpy()).any()
missing = train_data.drop(columns=list(EXCLUDED & set(train_data.columns))).isna().mean().sort_values(ascending=False)
display(missing.head(12).mul(100).to_frame('missing_%'))
for flag in ['has_previous', 'has_repayments', 'has_account']:
    display(train_data.groupby(flag).target.agg(aplikasi='size', rasio_target_1='mean'))
''')
    section(cells, '9. Preprocessing dan probabilitas model',
            'XGBoost/LightGBM menerima missing numerik; kategori diimputasi lalu di-one-hot encode. '
            'CatBoost menerima kategori langsung setelah missing kategori diberi label MISSING. '
            'Encoder selalu di-fit pada bagian training. Jika kelas positif dibobotkan w, '
            'gunakan koreksi p/(w×(1−p)+p) sebelum menggabungkan probabilitas model. '
            'Koreksi ini merupakan penyesuaian prior secara pendekatan, bukan jaminan kalibrasi sempurna.',
            source('optimize_ks.py', 'make_encoder') + ('' if version == 1 else '\n\n' + source('optimize_ks.py', 'weighted_probability') + '\n\n' + source('optimize_ks.py', 'predict_bundle')))
    if version != 1:
        section(cells, '10. Fungsi training yang digunakan',
            'Berikut kode training lengkap. Parameter model ditetapkan di sel berikutnya. '
            'Saat tuning, vi digunakan untuk early stopping; saat CV/final, vi=None dan jumlah pohon sudah tetap. '
            'CatBoost memakai AUC untuk penghentian awal, kemudian memilih jumlah pohon berdasarkan KS pada interval tetap.',
                source('optimize_ks.py', 'fit_candidate'))
    if level >= 2:
        compact = json.loads((ROOT / 'artifacts/optimization/compact_manifest.json').read_text())['columns']
        section(cells, '11. Daftar fitur yang dikunci',
                'Extended memakai seluruh fitur yang bervariasi. Compact mempertahankan 200 kolom dari tahap development historis: '
                'fitur dasar plus fitur dengan gain tinggi. Daftar ini dibuat eksplisit agar reproduksi tidak berubah. '
                'Seleksi supervised sebelumnya dapat memberi optimisme pada validasi silang berikutnya yang memakai label training yang sama.',
                "EXTENDED_COLS = [c for c in features_extended if c not in EXCLUDED and features_extended[c].nunique(dropna=True) > 1]\n" +
                'COMPACT_COLS = ' + literal(compact) + '\nassert not set(COMPACT_COLS) & EXCLUDED\nprint("Jumlah fitur extended/compact:", len(EXTENDED_COLS), len(COMPACT_COLS))')
    if level >= 3:
        code(cells, "V3_COLS = list(dict.fromkeys(COMPACT_COLS + [c for c in features_v3 if c.startswith('v3_') and features_v3[c].nunique(dropna=True) > 1]))\nprint('Jumlah fitur model v3:', len(V3_COLS))")
    if level >= 4:
        code(cells, "V4_COLS = list(dict.fromkeys(V3_COLS + [c for c in features_v4 if c.startswith('v4_') and features_v4[c].nunique(dropna=True) > 1]))\nprint('Jumlah fitur model v4:', len(V4_COLS))")
        section(cells, '12. Model spline: hubungan numerik yang halus',
                'Regresi logistik menerima basis spline untuk variabel numerik yang bervariasi. '
                'Ini memberi bentuk nonlinier yang halus sebagai pelengkap pohon. '
                'Nilai numerik ditransformasi signed-log; median, knot, scaling, dan encoder kategori dipelajari hanya pada baris training. '
                'C=0,05 mengendalikan regularisasi; n_knots=5 dan degree=3 mengendalikan bentuk spline.',
                source('spline_model.py', 'FEATURES') + '\n\n' + source('spline_model.py', 'prepare'))
        code(cells, source('spline_model.py', 'fit'))
        code(cells, source('spline_model.py', 'predict'))
    if version >= 3:
        section(cells, '13. Training dan prediksi silang antarfold',
                'OOF berarti out-of-fold: setiap baris train diprediksi oleh model yang tidak berlatih pada baris tersebut. '
                'Test diprediksi oleh semua model fold lalu probabilitasnya dirata-ratakan. '
                'Jika OOF historis dipakai, tabelnya diperiksa melalui ID dan target; prediksi test tetap dihitung ulang dari model. '
                'Untuk menghitung OOF ulang dari model, aktifkan FORCE_RECOMPUTE_VALIDATION. '
                'Cache model disimpan per versi/fold, sedangkan model historis hanya dibaca.', CV_LOOP)


def legacy_specs(include_xgb=True):
    specs = [
        {'name': 'extended/xgb_depth3', 'family': 'xgb', 'feature_set': 'extended', 'iterations': 829,
         'params': {'max_depth': 3, 'min_child_weight': 30, 'reg_lambda': 20, 'scale_pos_weight': 1}},
        {'name': 'extended/lgb_leaves31', 'family': 'lgb', 'feature_set': 'extended', 'iterations': 486,
         'params': {'num_leaves': 31, 'min_child_samples': 180, 'reg_lambda': 20, 'scale_pos_weight': 1}},
        {'name': 'compact/cat_depth5', 'family': 'cat', 'feature_set': 'extended', 'iterations': 400,
         'params': {'depth': 5, 'l2_leaf_reg': 12}},
    ]
    return specs if include_xgb else specs[1:]


def legacy_cells(cells, include_xgb=False):
    section(cells, 'Model tiga fold pembentuk versi KS',
            'Resep final KS historis adalah 25% LightGBM + 75% CatBoost. '
            'Fold memakai seed 8247; semua model memakai seed 2026 dan jumlah pohon tetap.',
            'KS_SPECS = ' + literal(legacy_specs(include_xgb)) + '''
for spec in KS_SPECS:
    spec['columns'] = COMPACT_COLS if spec['family'] == 'cat' else EXTENDED_COLS
ks_oof, ks_test = cv_predict(
    KS_SPECS, {'extended': features_extended}, 3, 8247, False,
    'artifacts/optimization', 'cv_fold{fold}_{name}.joblib',
    'artifacts/optimization/cross_validation_oof.parquet')
KS_RECIPE = {'extended/lgb_leaves31': .25, 'compact/cat_depth5': .75}
ks_oof['ensemble_ks'] = sum(w * ks_oof[n] for n, w in KS_RECIPE.items())
ks_test['ensemble_ks'] = sum(w * ks_test[n] for n, w in KS_RECIPE.items())
display(pd.Series(metrics(ks_oof.target, ks_oof.ensemble_ks), name='Versi KS'))
''')


def version3_cells(cells):
    legacy_cells(cells)
    specs = [
        {'name': 'price_lgb31_balanced_250', 'family': 'lgb', 'feature_set': 'v3', 'iterations': 250,
         'params': {'num_leaves': 31, 'min_child_samples': 200, 'reg_lambda': 40, 'scale_pos_weight': 6}},
        {'name': 'price_cat5_balanced_700', 'family': 'cat', 'feature_set': 'v3', 'iterations': 700,
         'params': {'depth': 5, 'l2_leaf_reg': 25, 'class_weights': [1, 6]}},
    ]
    section(cells, 'Model sepuluh fold pembentuk versi 3',
            'Dua model balanced menggunakan bobot positif 6 dan 388 fitur. '
            'Sepuluh fold memakai seed 9173; seed model adalah 2026+nomor fold. '
            'Ensemble versi KS yang lama tetap menjadi satu komponen. '
            'Ukuran training anggota lama sekitar 67%, anggota baru 90%.',
            'V3_SPECS = ' + literal(specs) + '''
for spec in V3_SPECS:
    spec['columns'] = V3_COLS
v3_oof, v3_test = cv_predict(
    V3_SPECS, {'v3': features_v3}, 10, 9173, True,
    'artifacts/optimization/tenfold', 'fold{fold}_{name}.joblib',
    'artifacts/optimization/tenfold/tenfold_oof.parquet')
assert v3_oof.application_id.equals(ks_oof.application_id)
assert v3_test.application_id.equals(ks_test.application_id)
v3_oof['ensemble_ks'] = ks_oof.ensemble_ks
v3_test['ensemble_ks'] = ks_test.ensemble_ks
V3_RECIPE = {'price_lgb31_balanced_250': 1/3,
             'price_cat5_balanced_700': 1/3, 'ensemble_ks': 1/3}
v3_oof['version3'] = sum(w * v3_oof[n] for n, w in V3_RECIPE.items())
v3_test['version3'] = sum(w * v3_test[n] for n, w in V3_RECIPE.items())
display(pd.Series(metrics(v3_oof.target, v3_oof.version3), name='Versi 3'))
''')


def validation_plot(cells, name, frame, column):
    section(cells, 'Membaca hasil validasi',
            'KS OOF keseluruhan memakai satu ambang pada seluruh baris. Rata-rata KS fold memakai maksimum per fold. '
            'Jangan membandingkan langsung rata-rata fold 3 versus 10 karena ukuran fold memengaruhi statistik maksimum. '
            'Skor ini berbeda dari public leaderboard. Pemilihan fitur, jumlah pohon, dan bobot menggunakan label training yang sama dapat memberi optimisme seleksi.',
            f'''validation_frame = {frame}
validation_probability = validation_frame[{column!r}].to_numpy()
y_valid = validation_frame.target.astype(int).to_numpy()
scores = metrics(y_valid, validation_probability)
assert abs(scores['ks'] - ks_2samp(validation_probability[y_valid == 1],
                                 validation_probability[y_valid == 0]).statistic) < 1e-12
display(pd.Series(scores, name={name!r}))
fpr, tpr, thresholds = roc_curve(y_valid, validation_probability, drop_intermediate=False)
best = np.argmax(tpr-fpr)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(fpr, tpr); axes[0].plot([0, 1], [0, 1], '--', color='gray')
axes[0].set(title='ROC validasi', xlabel='FPR', ylabel='TPR')
fraction = (y_valid.mean()*tpr + (1-y_valid.mean())*fpr)
axes[1].plot(fraction, tpr-fpr)
axes[1].scatter([fraction[best]], [tpr[best]-fpr[best]])
axes[1].set(title=f"KS OOF = {{scores['ks']:.5f}}", xlabel='Proporsi skor risiko tertinggi', ylabel='TPR - FPR')
plt.tight_layout(); plt.show()
print('Ambang untuk memahami validasi:', round(float(thresholds[best]), 4))
print('Submission tetap probabilitas kontinu; ambang ini tidak diterapkan ke CSV.')
''')


def search_cells(cells, version):
    md(cells, '## Review seluruh kandidat sebelum model final\n\n'
       'Bagian ini memperlihatkan kandidat yang pernah dibandingkan. Secara default ia membaca prediksi/hasil '
       'eksperimen historis. Aktifkan `RUN_OPTIONAL_SEARCH=True` untuk menjalankan pencarian lagi. '
       'Parameter dan bobot final historis pada sel berikutnya tetap eksplisit; ubah sel itu jika Anda ingin eksperimen baru.')
    if version == 2:
        code(cells, source('optimize_ks.py', 'candidates'))
        code(cells, '''
development, confirmation_indices = train_test_split(
    np.arange(len(train_data)), test_size=.25, random_state=2026,
    stratify=train_data.target)
core_indices, tuning_indices = train_test_split(
    development, test_size=.2, random_state=4917, stratify=train_data.target.iloc[development])
assert set(confirmation_indices).isdisjoint(core_indices)
assert set(confirmation_indices).isdisjoint(tuning_indices)
print('Core / tuning / confirmation:', len(core_indices), len(tuning_indices), len(confirmation_indices))
tuning_rows = []
for feature_set, frame, columns in [('base', features_base, base_columns),
                                   ('extended', features_extended, EXTENDED_COLS),
                                   ('compact', features_extended, COMPACT_COLS)]:
    data = frame[frame.split.eq('train')].reset_index(drop=True)
    specs = candidates()
    if feature_set == 'compact':
        specs = [s for s in specs if s['name'] in ['xgb_depth3', 'lgb_leaves15', 'lgb_leaves31', 'cat_depth5']]
    if RUN_OPTIONAL_SEARCH:
        for spec in specs:
            bundle, probability, trees = fit_candidate(spec, data, core_indices, tuning_indices, columns)
            tuning_rows.append({'feature_set': feature_set, 'name': spec['name'], 'iterations': trees,
                                **metrics(data.target.iloc[tuning_indices], probability)})
            if feature_set == 'extended' and spec['name'] == 'lgb_leaves31':
                # Kode seleksi gain yang mendasari daftar compact historis.
                names = bundle['encoder'].get_feature_names_out()
                gains = bundle['model'].booster_.feature_importance(importance_type='gain')
                gain_ranking = pd.Series(gains, index=names).sort_values(ascending=False)
                chosen = list(base_columns)
                cats = [c for c in columns if data[c].dtype == object]
                for name in gain_ranking.index:
                    raw = name if name in columns else next((c for c in sorted(cats, key=len, reverse=True)
                                                            if name.startswith(c+'_')), None)
                    if raw and raw not in chosen:
                        chosen.append(raw)
                    if len(chosen) >= 200:
                        break
                print('Compact hasil pencarian ini sama dengan historis:', chosen == COMPACT_COLS)
    else:
        manifest_file = ROOT / f'artifacts/optimization/{feature_set}_manifest.json'
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            tuning_rows.extend({'feature_set': feature_set, 'name': s['name'], 'iterations': s['iterations'],
                                **s['metrics']} for s in manifest['candidates'])
display(pd.DataFrame(tuning_rows).sort_values('ks', ascending=False))
''')
        return
    code(cells, source('refine_ks.py', 'stage_probability'))
    code(cells, STAGE_SEARCH)
    if version == 4:
        dirs = ['refinement', 'refinement_balanced']
        specs = []
        sources = {}
        for directory in dirs:
            definitions = json.loads((ROOT / f'artifacts/optimization/{directory}/definitions.json').read_text())
            for spec in definitions:
                copied = {k: v for k, v in spec.items() if k != 'columns'}
                copied['column_group'] = 'V3_COLS' if spec.get('add_v3') else ('EXTENDED_COLS' if spec['name'].startswith('extended') else 'COMPACT_COLS')
                specs.append(copied)
                sources[spec['name']] = 'artifacts/optimization/' + directory
        path = 'artifacts/optimization/tenfold/selection_oof.parquet'
    else:
        definitions = json.loads((ROOT / 'artifacts/optimization/v4/definitions.json').read_text())
        specs = [{**{k: v for k, v in s.items() if k != 'columns'},
                  'column_group': 'V4_COLS' if s['name'].startswith('v4_') else 'V3_COLS'} for s in definitions]
        sources = {s['name']: 'artifacts/optimization/v4' for s in definitions}
        path = 'artifacts/optimization/v4/selection_oof.parquet'
    code(cells, 'SEARCH_SPECS = ' + literal(specs) + '\nSEARCH_SOURCE_DIRS = ' + literal(sources) + f'''
for spec in SEARCH_SPECS:
    spec['columns'] = globals()[spec['column_group']]
search_oof = review_stage_search(SEARCH_SPECS, {'features_v3' if version == 4 else 'features_v4'},
                                {path!r}, SEARCH_SOURCE_DIRS)
''')
    md(cells, '### Mengapa bobot ensemble dipilih?\n\n'
       'Kode berikut mereview grid bobot pada OOF tiga fold. Versi 3 mensyaratkan kenaikan KS gabungan dan rata-rata fold. '
       'Versi 4 juga mensyaratkan kenaikan minimum, minimal dua fold membaik, dan batas penurunan fold terburuk. '
       'Pemilihan memakai label training yang sama, sehingga hasil tetap mengandung optimisme seleksi.')
    code(cells, BLEND_REVIEW)
    if version == 4:
        code(cells, "if search_oof is not None:\n    blend_ranking = review_blend_weights(search_oof, 'public_054710', 3)")
    else:
        code(cells, '''
spline_path = ROOT / 'artifacts/optimization/v4/spline_oof.parquet'
if search_oof is not None and spline_path.exists():
    spline_oof = pd.read_parquet(spline_path)
    assert spline_oof.application_id.equals(search_oof.application_id)
    blend_input = search_oof.copy()
    blend_input['spline'] = spline_oof.spline
    blend_ranking = review_blend_weights(blend_input, 'public_055845', 4)
else:
    print('OOF spline untuk review bobot belum tersedia. Resep final historis ditulis eksplisit di sel modeling.')
''')
    if version == 5:
        code(cells, '''
summary = ROOT / 'artifacts/optimization/v4/spline_summary.json'
if summary.exists():
    result = json.loads(summary.read_text())
    display(pd.DataFrame([{'candidate': 'spline', **result['metrics']}]
                        + [{'candidate': 'baseline+spline, weight='+w, **s} for w, s in result['blends'].items()]))
''')


def build(version):
    level = {1: 1, 2: 2, 3: 2, 4: 3, 5: 4}[version]
    cells = []
    common_cells(cells, version, level)
    if version == 1:
        params = json.loads((ROOT / 'artifacts/model_config.json').read_text())['hyperparameters']
        section(cells, '11. Baseline dan validasi holdout',
                'Parameter XGBoost ditetapkan sebelum validasi. Holdout 25% memakai seed 2026. '
                'Aplikasi saja, aplikasi+profil, dan semua riwayat pernah dibandingkan; tabel historis ditampilkan bila tersedia. '
                'Prediksi holdout berasal dari model yang berlatih hanya pada 75%, bukan model final pada seluruh train.',
                'BASELINE_PARAMS = ' + literal(dict(params, objective='binary:logistic', eval_metric='logloss', random_state=2026)) + '''
ti, vi = train_test_split(np.arange(len(train_data)), test_size=.25,
                         random_state=2026, stratify=train_data.target)
stored = ROOT / 'artifacts/validation_predictions.parquet'
if stored.exists() and not (FORCE_RETRAIN_MODELS or FORCE_RECOMPUTE_VALIDATION):
    p = pd.read_parquet(stored)
    p = p[p.model.eq('Semua riwayat') & p.validation.eq('Stratified 25%')]
    validation = train_data.iloc[vi][['application_id', 'target']].merge(
        p[['application_id', 'probability']], on='application_id', validate='one_to_one', how='left')
    assert validation.probability.notna().all()
else:
    encoder, _ = make_encoder(train_data[base_columns])
    validation_model = Pipeline([('encode', encoder), ('model', XGBClassifier(**BASELINE_PARAMS))])
    validation_model.fit(train_data.iloc[ti][base_columns], train_data.target.iloc[ti].astype(int))
    validation = train_data.iloc[vi][['application_id', 'target']].copy()
    validation['probability'] = validation_model.predict_proba(train_data.iloc[vi][base_columns])[:, 1]
table = ROOT / 'artifacts/tables/model_benchmarks.csv'
if table.exists():
    display(pd.read_csv(table))
''')
        section(cells, 'Bandingkan sumber informasi pada baseline',
                'Kode berikut mengulang perbandingan prior konstan, aplikasi saja, aplikasi+profil, dan semua riwayat. '
                'Ada pemeriksaan tambahan dengan training sebelum Oktober dan validasi Oktober–Desember 2025. '
                'Secara default tabel hasil historis dibaca; aktifkan RUN_OPTIONAL_SEARCH untuk melatih seluruh benchmark lagi.', '''
application_cols = ['loan_amount', 'annual_interest_rate', 'proposed_monthly_installment',
                    'loan_purpose', 'decision_delay_hours']
profile_cols = application_cols + ['age', 'region', 'employment_type', 'employment_tenure_months',
                                   'declared_monthly_income', 'dependents', 'housing_status',
                                   'installment_income_ratio', 'loan_income_ratio', 'income_per_dependent']
before_october = np.flatnonzero(train_data.decision_datetime.lt(pd.Timestamp('2025-10-01')))
october_onward = np.flatnonzero(train_data.decision_datetime.ge(pd.Timestamp('2025-10-01')))
benchmarks = [('Prior konstan', 'Stratified 25%', [], ti, vi),
              ('Aplikasi', 'Stratified 25%', application_cols, ti, vi),
              ('Aplikasi + profil', 'Stratified 25%', profile_cols, ti, vi),
              ('Semua riwayat', 'Stratified 25%', base_columns, ti, vi),
              ('Prior konstan', 'Temporal Okt–Des', [], before_october, october_onward),
              ('Semua riwayat', 'Temporal Okt–Des', base_columns, before_october, october_onward)]
if RUN_OPTIONAL_SEARCH:
    benchmark_rows = []
    for label, strategy, cols, training_rows, valid_rows in benchmarks:
        y_training = train_data.target.iloc[training_rows].astype(int)
        if not cols:
            probability = np.full(len(valid_rows), y_training.mean())
        else:
            encoder, _ = make_encoder(train_data[cols])
            model = Pipeline([('encode', encoder), ('model', XGBClassifier(**BASELINE_PARAMS))])
            model.fit(train_data.iloc[training_rows][cols], y_training)
            probability = model.predict_proba(train_data.iloc[valid_rows][cols])[:, 1]
        benchmark_rows.append({'model': label, 'validation': strategy, 'features': len(cols),
                               **metrics(train_data.target.iloc[valid_rows], probability)})
    display(pd.DataFrame(benchmark_rows))
else:
    print('Daftar sumber fitur sudah terlihat; hasil historis terdapat pada tabel validasi di atas.')
''')
        validation_plot(cells, 'Baseline holdout 25%', 'validation', 'probability')
        section(cells, '12. Latih ulang baseline pada seluruh label dan prediksi test',
                'Setelah mengevaluasi holdout, model final memakai seluruh 70.000 label. '
                'Cache yang dimuat adalah model final asli; bila tidak tersedia, Pipeline.fit dijalankan.', '''
source_model = ROOT / 'artifacts/submission_model.joblib'
local_model = OUT / 'models/baseline_full.joblib'
cache = local_model if local_model.exists() else source_model
if cache.exists() and not FORCE_RETRAIN_MODELS:
    final_model = joblib.load(cache)
    actual = final_model.named_steps['model'].get_params()
    assert all(actual.get(k) == v for k, v in BASELINE_PARAMS.items()), 'Parameter cache berbeda; aktifkan FORCE_RETRAIN_MODELS.'
else:
    encoder, _ = make_encoder(train_data[base_columns])
    final_model = Pipeline([('encode', encoder), ('model', XGBClassifier(**BASELINE_PARAMS))])
    final_model.fit(train_data[base_columns], train_data.target.astype(int))
    joblib.dump(final_model, local_model, compress=3)
test_probability = final_model.predict_proba(test_data[base_columns])[:, 1]
''')
    elif version == 2:
        search_cells(cells, version)
        spec = legacy_specs()[1]
        section(cells, '12. Konfirmasi kandidat terpilih pada holdout terpisah',
                'Kandidat yang dikunci adalah extended/LightGBM, 486 pohon. '
                '17.500 baris confirmation dipisahkan dari core dan tuning. '
                'Kandidat dengan KS confirmation tertinggi tidak otomatis mengganti pilihan yang sudah dikunci pada tuning.',
                'V2_SPEC = ' + literal(spec) + '''
V2_SPEC['columns'] = EXTENDED_COLS
stored = ROOT / 'artifacts/optimization/confirmation_predictions.parquet'
if stored.exists() and not (FORCE_RETRAIN_MODELS or FORCE_RECOMPUTE_VALIDATION):
    validation = pd.read_parquet(stored)
    assert set(validation.application_id) == set(train_data.application_id.iloc[confirmation_indices])
    validation['probability'] = validation['extended/lgb_leaves31']
else:
    bundle, _, _ = fit_candidate(V2_SPEC, train_data, development, None,
                                 EXTENDED_COLS, iterations=V2_SPEC['iterations'], seed=2026)
    validation = train_data.iloc[confirmation_indices][['application_id', 'target']].copy()
    validation['probability'] = predict_bundle(bundle, train_data.iloc[confirmation_indices])
display(pd.Series(metrics(validation.target, validation.probability), name='Confirmation V2'))
temporal = ROOT / 'artifacts/optimization/temporal_report.json'
if temporal.exists():
    record = json.loads(temporal.read_text())
    display(pd.DataFrame({k: record[k] for k in ['old_baseline', 'new_recipe']}).loc[['ks', 'roc_auc']])
''')
        validation_plot(cells, 'V2 confirmation', 'validation', 'probability')
        section(cells, '13. Full-data fit dan prediksi test', 'Model final berlatih pada seluruh label dengan parameter dan jumlah pohon yang sama.', '''
cache = ROOT / 'artifacts/optimization/final_extended_lgb_leaves31.joblib'
local = OUT / 'models/v2_full.joblib'
cache = local if local.exists() else cache
if cache.exists() and not FORCE_RETRAIN_MODELS:
    bundle = joblib.load(cache)
    assert bundle['columns'] == EXTENDED_COLS
    assert all(bundle['model'].get_params().get(k) == v for k, v in V2_SPEC['params'].items()), 'Parameter cache berbeda; aktifkan FORCE_RETRAIN_MODELS.'
    assert bundle['model'].get_params()['n_estimators'] == V2_SPEC['iterations'], 'Jumlah pohon cache berbeda; aktifkan FORCE_RETRAIN_MODELS.'
else:
    bundle, _, _ = fit_candidate(V2_SPEC, train_data, np.arange(len(train_data)), None,
                                 EXTENDED_COLS, iterations=V2_SPEC['iterations'], seed=2026)
    joblib.dump(bundle, local, compress=3)
test_probability = predict_bundle(bundle, test_data)
''')
    elif version == 3:
        legacy_cells(cells, include_xgb=True)
        section(cells, '14. Pemilihan bobot ensemble KS',
                'Bandingkan bobot kasar 25%, 50%, dan 75% pada tiga kandidat. '
                'Kriteria pemilihan adalah rata-rata KS fold, bukan hanya KS gabungan. '
                'Pemilihan bobot pada OOF menyebabkan optimisme seleksi.', '''
names = [s['name'] for s in KS_SPECS]
recipes = [{n: 1.0} for n in names]
for a, b in itertools.combinations(names, 2):
    recipes.extend([{a: w, b: 1-w} for w in [.25, .5, .75]])
recipes.append({n: 1/3 for n in names})
recipes.extend([{n: .5 if n == dominant else .25 for n in names} for dominant in names])
folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=8247).split(ks_oof, ks_oof.target))
rows = []
for recipe in recipes:
    p = sum(w * ks_oof[n] for n, w in recipe.items())
    rows.append({'recipe': recipe, **metrics(ks_oof.target, p),
                 'mean_fold_ks': np.mean([ks(ks_oof.target.iloc[vi], p.iloc[vi]) for _, vi in folds])})
display(pd.DataFrame(rows).sort_values('mean_fold_ks', ascending=False).head(8))
print('Resep historis yang direproduksi:', KS_RECIPE)
test_probability = ks_test.ensemble_ks.to_numpy()
''')
        validation_plot(cells, 'Ensemble KS tiga fold', 'ks_oof', 'ensemble_ks')
    elif version == 4:
        search_cells(cells, version)
        version3_cells(cells)
        validation_plot(cells, 'Versi 3 OOF', 'v3_oof', 'version3')
        code(cells, 'test_probability = v3_test.version3.to_numpy()')
    else:
        search_cells(cells, version)
        version3_cells(cells)
        specs = [
            {'name': 'v4_cat5_balanced_500', 'family': 'cat', 'feature_set': 'v4', 'iterations': 500,
             'params': {'depth': 5, 'l2_leaf_reg': 25, 'class_weights': [1, 6]}},
            {'name': 'spline', 'family': 'spline', 'feature_set': 'v4', 'params': {'C': .05, 'n_knots': 5, 'degree': 3}},
        ]
        section(cells, 'Model sepuluh fold dan gabungan akhir versi 4',
                'Versi 4 menggabungkan versi 3, CatBoost dengan fitur baru, dan spline, masing-masing sepertiga. '
                'Model baru memakai pembagian sepuluh fold yang sama dengan versi 3.',
                'V4_SPECS = ' + literal(specs) + '''
for spec in V4_SPECS:
    spec['columns'] = [c for c in FEATURES if c in features_v4] if spec['family'] == 'spline' else V4_COLS
v4_oof, v4_test = cv_predict(
    V4_SPECS, {'v4': features_v4}, 10, 9173, True,
    'artifacts/optimization/v4/final_models', 'fold{fold}_{name}.joblib',
    'artifacts/optimization/v4/combined_final_oof.parquet')
assert v4_oof.application_id.equals(v3_oof.application_id)
assert v4_test.application_id.equals(v3_test.application_id)
v4_oof['version3'] = v3_oof.version3
v4_test['version3'] = v3_test.version3
V4_RECIPE = {'version3': 1/3, 'v4_cat5_balanced_500': 1/3, 'spline': 1/3}
v4_oof['version4'] = sum(w * v4_oof[n] for n, w in V4_RECIPE.items())
v4_test['version4'] = sum(w * v4_test[n] for n, w in V4_RECIPE.items())
display(pd.DataFrame([{'versi': 'v3', **metrics(v4_oof.target, v4_oof.version3)},
                      {'versi': 'v4', **metrics(v4_oof.target, v4_oof.version4)}]))
folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=9173).split(v4_oof, v4_oof.target))
comparison = pd.DataFrame([
    {'fold': i, 'KS_v3': ks(v4_oof.target.iloc[vi], v4_oof.version3.iloc[vi]),
     'KS_v4': ks(v4_oof.target.iloc[vi], v4_oof.version4.iloc[vi])}
    for i, (_, vi) in enumerate(folds, 1)])
comparison['selisih'] = comparison.KS_v4 - comparison.KS_v3
display(comparison)
print('Fold yang meningkat:', int(comparison.selisih.gt(0).sum()), '/ 10')
test_probability = v4_test.version4.to_numpy()
''')
        validation_plot(cells, 'Versi 4 OOF', 'v4_oof', 'version4')
    section(cells, 'Terakhir: hasilkan submission.csv',
            'Kolom yang diminta adalah application_id dan default_probability. '
            'Merge satu-ke-satu memastikan urutan sample, lalu pemeriksaan membaca ulang CSV yang sudah ditulis. '
            'File historis/root tidak ditimpa. Bandingkan probabilitas dan SHA dengan checkpoint untuk memeriksa reproduksi. '
            'Selisih pembulatan kecil dapat muncul jika versi paket atau urutan penjumlahan berbeda.', EXPORT)
    md(cells, '''## Pertanyaan untuk review kode

    1. Apakah semua riwayat diketahui sebelum keputusan aplikasi?
    2. Apakah ada ID, target, atau informasi setelah keputusan yang masuk ke fitur?
    3. Apakah encoder dan model dilatih hanya pada baris training setiap fold?
    4. Apakah pilihan jumlah pohon/bobot memakai validasi yang juga dilaporkan?
    5. Apakah probabilitas test dirata-ratakan dengan bobot yang benar?
    6. Apakah CSV mengikuti header dan urutan sample, tanpa index tambahan?

    Public score bukan KS validasi lokal. Saat Anda mengubah parameter, skor historis
    tidak otomatis menjadi skor model baru. Gunakan CSV dari sel terakhir untuk mencoba hasilnya.
    ''')
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python', 'version': '3.10'},
                         'logika': {'version': version, 'self_contained_code': True, 'canonical_file': cells[0].source.split('**')[1]}}
    nbf.validate(notebook)
    filenames = ['01_baseline_xgboost.ipynb', '02_tuning_ks_v2.ipynb', '03_ensemble_ks.ipynb',
                 '04_submission_v3.ipynb', '05_submission_v4.ipynb']
    path = DEST / filenames[version-1]
    nbf.write(notebook, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    DEST.mkdir(exist_ok=True)
    paths = [build(v) for v in range(1, 6)]
    for path in paths:
        print('Created:', path.name, flush=True)
    if args.execute:
        from nbclient import NotebookClient
        for path in paths:
            notebook = nbf.read(path, as_version=4)
            print('Executing:', path.name, flush=True)
            try:
                NotebookClient(notebook, timeout=1200, kernel_name='python3',
                               resources={'metadata': {'path': str(ROOT)}}).execute()
            finally:
                nbf.write(notebook, path)
            print('Passed:', path.name, flush=True)


if __name__ == '__main__':
    main()
