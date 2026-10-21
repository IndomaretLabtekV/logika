"""Validate a fixed final blend, preserve the public checkpoint, and export version 4."""
from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

from optimize_ks import ROOT, OUT, ks, metrics

DESTINATION = OUT / 'v4'


def write_figure(y, baseline, probability):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    positive_rate = y.mean()
    for name, score, color in [('Versi 3', baseline, '#64748b'), ('Versi 4', probability, '#0284c7')]:
        fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)
        gap = tpr-fpr
        fraction = positive_rate*tpr+(1-positive_rate)*fpr
        best = int(np.argmax(gap))
        axes[0].plot(fraction, gap, color=color, label=f'{name}: KS {gap[best]:.5f}')
        axes[0].scatter([fraction[best]], [gap[best]], color=color, s=25)
        axes[1].plot(fpr, tpr, color=color, label=f'{name}: AUC {metrics(y, score)["roc_auc"]:.5f}')
    axes[0].set(xlabel='Proporsi aplikasi dengan skor tertinggi', ylabel='TPR - FPR', title='Statistik KS sepanjang urutan risiko')
    axes[1].plot([0, 1], [0, 1], color='#cbd5e1', linestyle='--')
    axes[1].set(xlabel='False positive rate', ylabel='True positive rate', title='Kurva ROC')
    for ax in axes:
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=.15)
    fig.suptitle('Validasi lokal: 70.000 prediksi OOF pada pembagian fold yang sama', fontsize=11)
    fig.tight_layout()
    (ROOT / 'artifacts/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / 'artifacts/figures/ks_v3_v4.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    config = json.loads((DESTINATION / 'locked_recipe.json').read_text())
    original = ROOT / config['baseline_file']
    assert hashlib.sha256(original.read_bytes()).hexdigest() == config['baseline_sha256']
    old_config = json.loads((OUT / 'tenfold/locked_recipe.json').read_text())
    old = pd.read_parquet(OUT / 'tenfold/tenfold_oof.parquet')
    old_test = pd.read_parquet(OUT / 'tenfold/tenfold_test_predictions.parquet')
    assert old.application_id.is_unique and old_test.application_id.is_unique
    oof = pd.DataFrame({'application_id': old.application_id, 'target': old.target,
                        'public_055845': sum(w * old[n] for n, w in old_config['recipe'].items())})
    predicted = pd.DataFrame({'application_id': old_test.application_id,
                              'public_055845': sum(w * old_test[n] for n, w in old_config['recipe'].items())})
    model_paths = []
    for name in config['recipe']:
        if name == 'public_055845':
            continue
        part = pd.read_parquet(DESTINATION / f'final_oof_{name}.parquet')
        test_part = pd.read_parquet(DESTINATION / f'final_test_{name}.parquet')
        assert part.application_id.equals(oof.application_id) and part.target.equals(oof.target)
        assert test_part.application_id.equals(predicted.application_id)
        assert np.isfinite(part[name]).all() and np.isfinite(test_part[name]).all()
        assert part[name].between(0, 1).all() and test_part[name].between(0, 1).all()
        oof[name] = part[name]
        predicted[name] = test_part[name]
        model_paths.extend(str((DESTINATION / 'final_models' / f'fold{fold}_{name}.joblib').relative_to(ROOT)) for fold in range(1, 11))
    y = oof.target.astype(int).to_numpy()
    baseline = oof.public_055845.to_numpy()
    probability = sum(w * oof[n] for n, w in config['recipe'].items()).to_numpy()
    test_probability = sum(w * predicted[n] for n, w in config['recipe'].items()).to_numpy()
    old_scores = metrics(y, baseline)
    new_scores = metrics(y, probability)
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=9173).split(oof, y))
    old_folds = [ks(y[vi], baseline[vi]) for _, vi in folds]
    new_folds = [ks(y[vi], probability[vi]) for _, vi in folds]
    old_scores.update(mean_fold_ks=float(np.mean(old_folds)), fold_ks=old_folds)
    new_scores.update(mean_fold_ks=float(np.mean(new_folds)), fold_ks=new_folds)
    assert new_scores['ks'] > old_scores['ks'] + .001, 'Final pooled improvement is too small.'
    assert new_scores['mean_fold_ks'] > old_scores['mean_fold_ks'], 'Final mean fold KS did not improve.'
    independent_ks = float(ks_2samp(probability[y == 1], probability[y == 0]).statistic)
    assert abs(independent_ks-new_scores['ks']) < 1e-12
    rng = np.random.default_rng(20261013)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    bootstrap_gains = []
    for _ in range(200):
        selected = np.r_[rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
        bootstrap_gains.append(ks(y[selected], probability[selected]) - ks(y[selected], baseline[selected]))
    bootstrap = {'replicates': 200, 'mean_gain': float(np.mean(bootstrap_gains)),
                 'p05': float(np.quantile(bootstrap_gains, .05)), 'p95': float(np.quantile(bootstrap_gains, .95)),
                 'fraction_positive': float(np.mean(np.array(bootstrap_gains) > 0)),
                 'caveat': 'Paired row bootstrap of fixed OOF predictions; it does not correct selection optimism or certify a leaderboard gain.'}
    sample = pd.read_csv(ROOT / 'data/sample_submission.csv')
    raw_prediction = pd.DataFrame({'application_id': predicted.application_id, 'default_probability': test_probability})
    assert set(raw_prediction.application_id) == set(sample.application_id)
    submission = sample[['application_id']].merge(raw_prediction, on='application_id', how='left', sort=False, validate='one_to_one')
    assert submission.shape == (30000, 2) and submission.application_id.is_unique
    assert submission.application_id.equals(sample.application_id)
    assert np.isfinite(submission.default_probability).all() and submission.default_probability.between(0, 1).all()
    old_submission = pd.read_csv(original)
    baseline_export = sample[['application_id']].merge(predicted[['application_id', 'public_055845']], on='application_id', how='left', validate='one_to_one', sort=False)
    assert np.max(np.abs(baseline_export.public_055845 - old_submission.default_probability)) < 1e-11
    checkpoint = OUT / 'submission_public_0_55845.csv'
    if not checkpoint.exists():
        shutil.copy2(original, checkpoint)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == config['baseline_sha256']
    path = ROOT / 'results/teman/submission_v4.csv'
    submission.to_csv(path, index=False, float_format='%.12f')
    reread = pd.read_csv(path)
    assert reread.columns.tolist() == sample.columns.tolist()
    assert reread.application_id.equals(sample.application_id)
    assert reread.default_probability.notna().all() and reread.default_probability.between(0, 1).all()
    assert not np.allclose(reread.default_probability, old_submission.default_probability)
    old_metadata = DESTINATION / 'submission_public_0_55845_metadata.json'
    if not old_metadata.exists():
        shutil.copy2(ROOT / 'artifacts/submission_metadata.json', old_metadata)
    shutil.copy2(path, ROOT / 'results/teman/submission.csv')
    report = {'file': 'submission_v4.csv', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'recipe': config['recipe'], 'baseline_public_score': .55845, 'new_public_score': None,
              'selection_3fold': {'baseline': config['baseline'], 'new': config['selection']},
              'confirmation_10fold': {'baseline': old_scores, 'new': new_scores,
                                      'fold_wins': int((np.array(new_folds) > old_folds).sum())},
              'paired_bootstrap': bootstrap, 'training_rows': 70000, 'test_rows': 30000,
              'probability_summary': reread.default_probability.describe().to_dict(),
              'new_model_paths': model_paths, 'baseline_file': 'submission_v3.csv',
              'promoted_to_submission_csv': True,
              'checks': {'correct_header': True, 'sample_id_order': True, 'unique_ids': True,
                         'finite_probability': True, 'probability_bounds': True,
                         'baseline_prediction_reproduction': True, 'independent_ks_check': True},
              'caveat': config['caveat']}
    for relative in model_paths:
        assert (ROOT / relative).exists()
    (DESTINATION / 'submission_v4_metadata.json').write_text(json.dumps(report, indent=2))
    (ROOT / 'artifacts/submission_metadata.json').write_text(json.dumps(dict(report, file='submission.csv'), indent=2))
    oof['version4'] = probability
    oof.to_parquet(DESTINATION / 'combined_final_oof.parquet', index=False)
    predicted['version4'] = test_probability
    predicted.to_parquet(DESTINATION / 'combined_final_test.parquet', index=False)
    cohort = pd.read_parquet(ROOT / 'data/processed/application_features_v4.parquet')
    cohort = cohort[cohort.split.eq('train')].reset_index(drop=True)
    assert cohort.application_id.equals(oof.application_id)
    cohort_metrics = []
    for column in ['has_repayments', 'employment_type']:
        for value, rows in cohort.groupby(column).groups.items():
            indices = np.asarray(rows)
            if len(indices) < 500 or len(np.unique(y[indices])) < 2:
                continue
            cohort_metrics.append({'group': column, 'value': str(value), 'rows': len(indices),
                                   'baseline_ks': ks(y[indices], baseline[indices]), 'new_ks': ks(y[indices], probability[indices])})
    (DESTINATION / 'cohort_metrics.json').write_text(json.dumps(cohort_metrics, indent=2))
    write_figure(y, baseline, probability)
    text = f'''# Submission versi 4 untuk KS

Checkpoint versi 3 memperoleh public KS **0,55845**, terkonfirmasi melalui API Kaggle. Skor public versi 4 belum tersedia. Skrip ini membuat file untuk diunggah sendiri dan tidak mengirim submission otomatis.

Fitur baru merangkum harga yang dinormalisasi menurut kategori, frekuensi keterlambatan pada cicilan yang sudah cukup umur untuk mencapai ambang DPD, serta skor biro yang dirata-ratakan dengan bobot setara antarprovider. Referensi harga dan prior keterlambatan memakai riwayat customer training tanpa label target. Audit memeriksa seluruh waktu observasi sebelum keputusan.

Gabungan akhir memakai resep berikut:

```json
{json.dumps(config['recipe'], indent=2)}
```

| Perbandingan pada baris dan pembagian fold yang sama | Versi 3 | Versi 4 |
|---|---:|---:|
| KS OOF keseluruhan, seleksi tiga fold | {config['baseline']['ks']:.5f} | {config['selection']['ks']:.5f} |
| Rata-rata KS, seleksi tiga fold | {config['baseline']['mean_fold_ks']:.5f} | {config['selection']['mean_fold_ks']:.5f} |
| KS OOF keseluruhan, tahap sepuluh fold | {old_scores['ks']:.5f} | {new_scores['ks']:.5f} |
| Rata-rata KS, tahap sepuluh fold | {old_scores['mean_fold_ks']:.5f} | {new_scores['mean_fold_ks']:.5f} |
| ROC-AUC OOF, tahap sepuluh fold | {old_scores['roc_auc']:.5f} | {new_scores['roc_auc']:.5f} |

![Kurva KS dan ROC prediksi OOF versi 3 dan 4](../../figures/ks_v3_v4.png)

Pada tahap sepuluh fold, KS meningkat pada **{report['confirmation_10fold']['fold_wins']} dari 10 fold**. Tidak ada perbandingan langsung rata-rata statistik maksimum KS antara ukuran fold yang berbeda. Estimasi bootstrap berpasangan tersedia pada metadata; estimasi itu memakai prediksi OOF tetap dan tidak mengoreksi optimisme seleksi.

Model baru memakai 90% label per fold. Komponen versi 3 dipertahankan dari prediksi yang sudah tersimpan, termasuk anggota historisnya yang dilatih pada tiga fold. Fitur, jumlah pohon, dan bobot dipilih menggunakan label training yang sama, sehingga hasil lokal adalah estimasi dengan optimisme seleksi dan tidak menjamin peningkatan leaderboard. Tidak ada label test yang dipakai.

File `submission_v4.csv` berisi 30.000 probabilitas kontinu sesuai header dan urutan sample submission. SHA-256: `{report['sha256']}`. `submission.csv` memuat isi yang sama. Versi 3 serta checkpoint public persisnya dipertahankan.

Reproduksi setelah membangun fitur versi 3:

```bash
python scripts/build_features_v4.py
python scripts/evaluate_v4.py
python scripts/evaluate_spline_v4.py
python scripts/select_v4.py
'''
    for name in config['recipe']:
        if name != 'public_055845':
            text += f'python scripts/train_v4_candidate.py --candidate {name}\n'
    text += 'python scripts/export_v4_submission.py\n```\n'
    (DESTINATION / 'laporan_submission_v4.md').write_text(text)
    print(json.dumps({'file': str(path), 'recipe': config['recipe'], 'baseline': old_scores,
                      'new': new_scores, 'bootstrap': bootstrap, 'sha256': report['sha256']}), flush=True)


if __name__ == '__main__':
    main()
