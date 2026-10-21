"""Confirm a recipe selected on tuning data, then refit and export submission_v2.csv."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

from optimize_ks import OUT, ROOT, fit_candidate, ks, metrics, predict_bundle


def main():
    manifests=[]
    predictions=None
    for feature_set in ['base','extended','compact']:
        manifest=json.loads((OUT/f'{feature_set}_manifest.json').read_text())
        manifests.append(manifest)
        pred=pd.read_parquet(OUT/f'{feature_set}_tuning_predictions.parquet')
        names={c:f'{feature_set}/{c}' for c in pred if c not in ['application_id','target']}
        pred=pred.rename(columns=names)
        if predictions is None:
            predictions=pred
        else:
            assert predictions.application_id.equals(pred.application_id)
            assert predictions.target.equals(pred.target)
            predictions=pd.concat([predictions,pred.drop(columns=['application_id','target'])],axis=1)
    definitions={}
    for manifest in manifests:
        for candidate in manifest['candidates']:
            name=f"{manifest['feature_set']}/{candidate['name']}"
            definitions[name]=dict(candidate,columns=manifest['columns'],feature_file=manifest['feature_file'])
    # At most one member from each model family to limit redundant ensembles.
    members=[]
    for family in ['xgb','lgb','cat']:
        group=[n for n,s in definitions.items() if s['family']==family]
        members.append(max(group,key=lambda n:definitions[n]['metrics']['ks']))
    recipes=[]
    for n in members:
        recipes.append({n:1.0})
    for a,b in itertools.combinations(members,2):
        for w in [.25,.5,.75]:
            recipes.append({a:w,b:1-w})
    recipes.append({n:1/len(members) for n in members})
    for chosen in members:
        recipes.append({n:.5 if n==chosen else .25 for n in members})
    rows=[]
    for recipe in recipes:
        p=sum(w*predictions[n].to_numpy() for n,w in recipe.items())
        rows.append({'recipe':recipe,**metrics(predictions.target,p)})
    rows.sort(key=lambda r:r['ks'],reverse=True)
    locked=rows[0]
    chosen=locked['recipe']
    (OUT/'recipe_search.json').write_text(json.dumps(rows,indent=2))
    (OUT/'locked_recipe.json').write_text(json.dumps(locked,indent=2))
    print('Locked recipe selected only on tuning:',json.dumps(locked),flush=True)

    frames={}
    for manifest in manifests:
        file=manifest['feature_file']
        if file not in frames:
            frames[file]=pd.read_parquet(ROOT/'data/processed'/file)
    training={file:df[df.split.eq('train')].reset_index(drop=True) for file,df in frames.items()}
    testing={file:df[df.split.eq('test')].reset_index(drop=True) for file,df in frames.items()}
    base=training['application_features.parquet']
    for df in training.values():
        assert df.application_id.equals(base.application_id)
        assert df.target.equals(base.target)
    splits=np.load(OUT/'base_splits.npz')
    development,confirmation=splits['development'],splits['confirmation']
    assert set(development).isdisjoint(confirmation)
    validation=pd.DataFrame({'application_id':base.application_id.iloc[confirmation].to_numpy(),
                             'target':base.target.iloc[confirmation].to_numpy()})
    confirmed={}
    for name in members:
        spec=definitions[name]
        df=training[spec['feature_file']]
        print(f'Confirming {name}, {spec["iterations"]} trees on original holdout',flush=True)
        bundle,_,_=fit_candidate(spec,df,development,None,spec['columns'],iterations=spec['iterations'])
        p=predict_bundle(bundle,df.iloc[confirmation])
        validation[name]=p
        confirmed[name]=metrics(validation.target,p)
        joblib.dump(bundle,OUT/f'confirmation_{name.replace("/","_")}.joblib',compress=3)
        print(json.dumps({'name':name,**confirmed[name]}),flush=True)
    validation['new_recipe']=sum(w*validation[n] for n,w in chosen.items())
    stored=pd.read_parquet(ROOT/'artifacts/validation_predictions.parquet')
    stored=stored[stored.model.eq('Semua riwayat')&stored.validation.eq('Stratified 25%')]
    validation=validation.merge(stored[['application_id','probability']].rename(columns={'probability':'old_baseline'}),
                                on='application_id',how='left',validate='one_to_one',sort=False)
    assert validation.old_baseline.notna().all()
    old_metrics=metrics(validation.target,validation.old_baseline)
    new_metrics=metrics(validation.target,validation.new_recipe)
    confirmed['old_baseline']=old_metrics
    confirmed['locked_recipe']=new_metrics
    validation.to_parquet(OUT/'confirmation_predictions.parquet',index=False)
    print('Confirmation comparison:',json.dumps({'old':old_metrics,'new':new_metrics}),flush=True)

    # Paired bootstrap assesses uncertainty in the improvement on the same customers.
    rng=np.random.default_rng(2026)
    y=validation.target.to_numpy()
    old=validation.old_baseline.to_numpy()
    new=validation.new_recipe.to_numpy()
    positives=np.flatnonzero(y==1)
    negatives=np.flatnonzero(y==0)
    differences=[]
    for _ in range(200):
        sampled=np.concatenate([rng.choice(positives,len(positives),replace=True),rng.choice(negatives,len(negatives),replace=True)])
        differences.append(ks(y[sampled],new[sampled])-ks(y[sampled],old[sampled]))
    interval=np.quantile(differences,[.025,.975]).tolist()
    confirmation_report={'tune':locked,'members':confirmed,'ks_difference':new_metrics['ks']-old_metrics['ks'],
                         'paired_bootstrap_95_interval':interval,'validation_rows':len(validation),
                         'protocol':'Parameters and ensemble weights selected on separate 10,500-row tuning set. Confirmation has 17,500 rows and was not used for tuning.'}
    (OUT/'confirmation_report.json').write_text(json.dumps(confirmation_report,indent=2))
    if new_metrics['ks']<=old_metrics['ks']:
        print('The locked recipe did not improve confirmation KS; export requires further investigation.',flush=True)
        return

    # A second check uses the same temporal split as the original baseline.
    temporal_train=np.flatnonzero(base.decision_datetime.lt(pd.Timestamp('2025-10-01')))
    temporal_val=np.flatnonzero(base.decision_datetime.ge(pd.Timestamp('2025-10-01')))
    temporal_probability=np.zeros(len(temporal_val))
    for name,weight in chosen.items():
        spec=definitions[name]
        df=training[spec['feature_file']]
        print(f'Temporal check: {name}',flush=True)
        bundle,_,_=fit_candidate(spec,df,temporal_train,None,spec['columns'],iterations=spec['iterations'])
        temporal_probability+=weight*predict_bundle(bundle,df.iloc[temporal_val])
    temporal_stored=stored=pd.read_parquet(ROOT/'artifacts/validation_predictions.parquet')
    temporal_stored=temporal_stored[temporal_stored.model.eq('Semua riwayat')&temporal_stored.validation.eq('Temporal Okt–Des')]
    temporal_ids=base.application_id.iloc[temporal_val]
    old_temporal=temporal_stored.set_index('application_id').probability.reindex(temporal_ids).to_numpy()
    assert np.isfinite(old_temporal).all()
    temporal={'old_baseline':metrics(base.target.iloc[temporal_val],old_temporal),
              'new_recipe':metrics(base.target.iloc[temporal_val],temporal_probability),
              'validation_rows':len(temporal_val)}
    (OUT/'temporal_report.json').write_text(json.dumps(temporal,indent=2))
    print('Temporal comparison:',json.dumps(temporal),flush=True)

    # Refit on all 70,000 labels, using fixed iterations and the locked recipe.
    sample=pd.read_csv(ROOT/'data/sample_submission.csv')
    test_reference=testing['application_features.parquet']
    test_probability=np.zeros(len(test_reference))
    for name,weight in chosen.items():
        spec=definitions[name]
        df=training[spec['feature_file']]
        test=testing[spec['feature_file']]
        assert test.application_id.equals(test_reference.application_id)
        print(f'Full-data fit: {name}, weight {weight}',flush=True)
        bundle,_,_=fit_candidate(spec,df,np.arange(len(df)),None,spec['columns'],iterations=spec['iterations'])
        test_probability+=weight*predict_bundle(bundle,test)
        joblib.dump(bundle,OUT/f'final_{name.replace("/","_")}.joblib',compress=3)
    pred=pd.DataFrame({'application_id':test_reference.application_id,'default_probability':test_probability})
    assert set(pred.application_id)==set(sample.application_id)
    assert pred.application_id.is_unique
    submission=sample[['application_id']].merge(pred,on='application_id',how='left',validate='one_to_one',sort=False)
    assert submission.application_id.equals(sample.application_id)
    assert submission.columns.tolist()==sample.columns.tolist()
    assert len(submission)==30000 and submission.application_id.is_unique
    assert np.isfinite(submission.default_probability).all()
    assert submission.default_probability.between(0,1).all()
    destination=ROOT / 'results/teman/submission_v2.csv'
    submission.to_csv(destination,index=False,float_format='%.12f')
    reread=pd.read_csv(destination)
    assert reread.application_id.equals(sample.application_id)
    assert reread.shape==(30000,2) and reread.default_probability.between(0,1).all()
    original=ROOT / 'results/teman/submission.csv'
    if original.exists():
        backup=OUT/'submission_baseline_0_52623.csv'
        if not backup.exists():shutil.copy2(original,backup)
    shutil.copy2(destination,original)
    metadata={'file':'submission_v2.csv','sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
              'metric':'Kolmogorov-Smirnov statistic (KS), supplied by user from Evaluation',
              'recipe':chosen,'confirmation':confirmation_report,'temporal':temporal,
              'train_rows':70000,'test_rows':30000,'old_public_score':.52623,
              'new_public_score':None,'probability_summary':reread.default_probability.describe().to_dict(),
              'checks':{'sample_id_order':True,'no_missing':True,'unique_ids':True,'valid_probability_range':True}}
    (OUT/'submission_v2_metadata.json').write_text(json.dumps(metadata,indent=2))
    report=f'''# Peningkatan submission berdasarkan KS

Metrik resmi adalah Kolmogorov-Smirnov (KS), sesuai kutipan Evaluation yang diberikan pengguna. Skor publik baseline terkonfirmasi melalui API Kaggle: 0,52623. Skor publik versi baru belum tersedia; tidak ada pengunggahan otomatis.

Pemilihan parameter dan bobot gabungan menggunakan set tuning 10.500 baris yang dipisahkan dari holdout asli 17.500 baris. Semua fitur dibentuk sebelum keputusan; ID dan target tidak dipakai sebagai fitur. Kandidat diperiksa pada KS, ROC-AUC, AP, dan log loss. Hasil tuning lengkap dan seluruh resep tersimpan di direktori ini.

| Validasi | KS baseline | KS baru |
|---|---:|---:|
| Holdout stratified yang sama | {old_metrics['ks']:.5f} | {new_metrics['ks']:.5f} |
| Temporal Oktober–Desember | {temporal['old_baseline']['ks']:.5f} | {temporal['new_recipe']['ks']:.5f} |

Selisih KS holdout: {confirmation_report['ks_difference']:.5f}. Interval bootstrap berpasangan 95% untuk selisih: [{interval[0]:.5f}, {interval[1]:.5f}]. Holdout asli sebelumnya dipakai untuk melaporkan baseline; pada tahap optimasi ini ia tidak dipakai memilih parameter atau bobot. Hasil lokal tidak menjamin skor publik/private yang sama atau mencapai 0,56.

Fitur tambahan mencakup beban pinjaman aktif, perkiraan sisa pokok berdasarkan amortisasi terjadwal, keterlambatan terbaru dan per pinjaman, riwayat/tren biro, perubahan rekening, belanja 30/60/90/180/365 hari, kategori dan metode pembayaran. Perkiraan sisa pokok adalah fitur estimasi, bukan saldo aktual. Subset compact dipilih dari gain model yang dilatih pada bagian development; confirmation tidak digunakan untuk seleksi fitur.

Resep yang dikunci sebelum confirmation:

```json
{json.dumps(chosen,indent=2)}
```

Model akhir dilatih ulang pada seluruh 70.000 label dengan jumlah pohon dan bobot tetap. `submission_v2.csv` mempunyai 30.000 probabilitas kontinu, dua kolom yang benar, ID unik dan urutan yang sama dengan sample submission. `submission.csv` diperbarui dengan isi yang sama; versi baseline dipertahankan sebagai `submission_baseline_0_52623.csv`.

Probabilitas tetap kontinu; mengubah threshold atau melakukan transformasi monoton tidak menambah kemampuan pemisahan yang diukur KS. Tidak ada threshold yang diterapkan pada submission.
'''
    (OUT/'laporan_perbaikan_ks.md').write_text(report)
    print('Created submission_v2.csv and updated submission.csv; all export checks passed.',flush=True)


if __name__=='__main__':
    main()
