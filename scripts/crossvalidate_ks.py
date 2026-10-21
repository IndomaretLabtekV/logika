"""Three-fold validation of fixed candidates; export a blend of fold models."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from optimize_ks import OUT, ROOT, fit_candidate, metrics, predict_bundle


def baseline_bundle(data,indices):
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    from optimize_ks import make_encoder
    config=json.loads((ROOT/'artifacts/model_config.json').read_text())
    cols=config['all_feature_columns']
    encoder,_=make_encoder(data[cols])
    x=encoder.fit_transform(data.iloc[indices][cols])
    params=dict(config['hyperparameters'],objective='binary:logistic',eval_metric='logloss',random_state=2026)
    model=XGBClassifier(**params)
    model.fit(x,data.target.iloc[indices].astype(int))
    return {'family':'xgb','encoder':encoder,'columns':cols,'model':model,'weight':1}


def main():
    OUT.mkdir(exist_ok=True)
    manifest={s:json.loads((OUT/f'{s}_manifest.json').read_text()) for s in ['base','extended','compact']}
    definitions={f'{s}/{c["name"]}':dict(c,columns=m['columns'],feature_file=m['feature_file'])
                 for s,m in manifest.items() for c in m['candidates']}
    names=[]
    for family in ['xgb','lgb','cat']:
        matching=[n for n,c in definitions.items() if c['family']==family]
        names.append(max(matching,key=lambda n:definitions[n]['metrics']['ks']))
    file_names=set(c['feature_file'] for c in definitions.values())
    frames={file:pd.read_parquet(ROOT/'data/processed'/file) for file in file_names}
    train={file:df[df.split.eq('train')].reset_index(drop=True) for file,df in frames.items()}
    test={file:df[df.split.eq('test')].reset_index(drop=True) for file,df in frames.items()}
    reference=train['application_features.parquet']
    test_reference=test['application_features.parquet']
    y=reference.target.astype(int)
    for df in train.values():
        assert df.application_id.equals(reference.application_id)
        assert df.target.equals(reference.target)
    oof=pd.DataFrame({'application_id':reference.application_id,'target':y})
    test_pred=pd.DataFrame({'application_id':test_reference.application_id})
    for name in ['baseline']+names:
        oof[name]=np.nan
        test_pred[name]=0.0
    folds=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=8247).split(reference,y))
    results=[]
    for fold,(ti,vi) in enumerate(folds,1):
        print(f'CV fold {fold}/3: original baseline',flush=True)
        bundle=baseline_bundle(reference,ti)
        p=predict_bundle(bundle,reference.iloc[vi])
        oof.loc[vi,'baseline']=p
        test_pred['baseline']+=predict_bundle(bundle,test_reference)/3
        result={'fold':fold,'name':'baseline',**metrics(y.iloc[vi],p)}
        results.append(result)
        print(json.dumps(result),flush=True)
        for name in names:
            spec=definitions[name]
            df=train[spec['feature_file']]
            print(f'CV fold {fold}/3: {name}, fixed {spec["iterations"]} trees',flush=True)
            bundle,_,_=fit_candidate(spec,df,ti,None,spec['columns'],iterations=spec['iterations'])
            p=predict_bundle(bundle,df.iloc[vi])
            oof.loc[vi,name]=p
            test_pred[name]+=predict_bundle(bundle,test[spec['feature_file']])/3
            result={'fold':fold,'name':name,**metrics(y.iloc[vi],p)}
            results.append(result)
            joblib.dump(bundle,OUT/f'cv_fold{fold}_{name.replace("/","_")}.joblib',compress=3)
            print(json.dumps(result),flush=True)
        oof.to_parquet(OUT/'cross_validation_oof.parquet',index=False)
        test_pred.to_parquet(OUT/'cross_validation_test_predictions.parquet',index=False)
        pd.DataFrame(results).to_csv(OUT/'cross_validation_fold_metrics.csv',index=False)
    assert oof[['baseline']+names].notna().all().all()
    recipes=[{n:1.0} for n in names]
    for a,b in itertools.combinations(names,2):
        recipes.extend([{a:w,b:1-w} for w in [.25,.5,.75]])
    recipes.append({n:1/3 for n in names})
    recipes.extend([{n:.5 if n==dominant else .25 for n in names} for dominant in names])
    summaries=[]
    for recipe in recipes:
        p=sum(weight*oof[name] for name,weight in recipe.items())
        scores=metrics(y,p)
        fold_ks=[metrics(y.iloc[vi],p.iloc[vi])['ks'] for _,vi in folds]
        summaries.append({'recipe':recipe,**scores,'mean_fold_ks':float(np.mean(fold_ks)),
                          'fold_ks':fold_ks})
    # Use the average per-fold KS, rather than exploiting one aggregate threshold.
    summaries.sort(key=lambda row:row['mean_fold_ks'],reverse=True)
    best=summaries[0]
    recipe=best['recipe']
    baseline=metrics(y,oof.baseline)
    baseline_fold_ks=[metrics(y.iloc[vi],oof.baseline.iloc[vi])['ks'] for _,vi in folds]
    baseline['mean_fold_ks']=float(np.mean(baseline_fold_ks))
    baseline['fold_ks']=baseline_fold_ks
    (OUT/'cross_validation_recipes.json').write_text(json.dumps(summaries,indent=2))
    print('Cross-validation selected recipe:',json.dumps(best),flush=True)
    print('Cross-validation baseline:',json.dumps(baseline),flush=True)

    p=sum(weight*test_pred[name] for name,weight in recipe.items())
    pred=pd.DataFrame({'application_id':test_reference.application_id,'default_probability':p})
    sample=pd.read_csv(ROOT/'data/sample_submission.csv')
    assert set(sample.application_id)==set(pred.application_id)
    submission=sample[['application_id']].merge(pred,on='application_id',how='left',validate='one_to_one',sort=False)
    assert submission.application_id.equals(sample.application_id)
    assert submission.application_id.is_unique and len(submission)==30000
    assert np.isfinite(submission.default_probability).all()
    assert submission.default_probability.between(0,1).all()
    destination=ROOT / 'results/teman/submission_ks.csv'
    submission.to_csv(destination,index=False,float_format='%.12f')
    reread=pd.read_csv(destination)
    assert reread.shape==(30000,2) and reread.columns.tolist()==sample.columns.tolist()
    assert reread.application_id.equals(sample.application_id)
    assert reread.default_probability.between(0,1).all() and reread.default_probability.notna().all()
    # Keep the full-data tuned alternative; promote CV only when its fold KS improves.
    promoted=best['mean_fold_ks']>baseline['mean_fold_ks'] and best['ks']>baseline['ks']
    if promoted:
        shutil.copy2(destination,ROOT / 'results/teman/submission.csv')
    report={'metric':'KS','seed':8247,'folds':3,'candidate_count':len(definitions),
            'fixed_model_names':names,'baseline':baseline,'selected':best,
            'recipe_selection':'highest mean per-fold KS among fixed coarse blend weights',
            'caveat':'Parameters were fixed before this CV. CV predictions were used to select blend weights, so these metrics are validation estimates, not an unbiased final test score.',
            'test_prediction_method':'Mean of three held-out-fold training models per selected family, then the selected family blend.',
            'train_rows':70000,'test_rows':30000,'file':'submission_ks.csv','promoted_to_submission_csv':promoted,
            'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'public_score':None,
            'probability_summary':reread.default_probability.describe().to_dict()}
    (OUT/'cross_validation_summary.json').write_text(json.dumps(report,indent=2))
    if promoted:
        metadata=ROOT/'artifacts/submission_metadata.json'
        backup=OUT/'submission_baseline_metadata.json'
        if metadata.exists() and not backup.exists():
            shutil.copy2(metadata,backup)
        latest=dict(report,file='submission.csv',source_file='submission_ks.csv',
                    model='Weighted ensemble of three-fold LightGBM and CatBoost candidates',
                    model_artifacts=[str((OUT/f'cv_fold{fold}_{name.replace("/","_")}.joblib').relative_to(ROOT))
                                     for name in recipe for fold in range(1,4)])
        metadata.write_text(json.dumps(latest,indent=2))
    text=f'''# Validasi silang dan submission KS

Metrik resmi kompetisi adalah KS berdasarkan Evaluation yang diberikan pengguna. Nilai publik baseline terdahulu adalah 0,52623. Skor publik submission baru belum diketahui.

Tiga kandidat berikut dipilih pada set tuning terpisah, kemudian parameternya dibekukan sebelum validasi silang 3 fold dengan seed 8247: {', '.join(names)}. Jumlah pohon juga dibekukan; tidak ada early stopping pada fold validasi ini. Setiap prediksi OOF berasal dari model yang tidak dilatih pada baris tersebut.

| Ukuran | Baseline asli | Kandidat terpilih |
|---|---:|---:|
| Rata-rata KS tiga fold | {baseline['mean_fold_ks']:.5f} | {best['mean_fold_ks']:.5f} |
| KS seluruh prediksi OOF | {baseline['ks']:.5f} | {best['ks']:.5f} |
| ROC-AUC OOF | {baseline['roc_auc']:.5f} | {best['roc_auc']:.5f} |
| Average precision OOF | {baseline['average_precision']:.5f} | {best['average_precision']:.5f} |

Bobot gabungan dipilih dari grid kecil menggunakan rata-rata KS per fold. Oleh karena OOF digunakan untuk memilih bobot, angka ini merupakan hasil validasi dan dapat mengandung optimisme seleksi. Skor ini bukan hasil leaderboard dan tidak menjamin mencapai 0,56.

```json
{json.dumps(recipe,indent=2)}
```

Prediksi test merupakan rata-rata tiga model fold untuk setiap anggota resep, kemudian digabungkan dengan bobot di atas. Fitur tidak memakai ID atau target, dan audit waktu observasi dilakukan sebelum pelatihan. `submission_ks.csv` mempunyai 30.000 baris sesuai urutan sample submission, dengan probabilitas kontinu valid. Versi model satu kali fit pada seluruh data tetap tersedia di `submission_v2.csv`. `submission.csv` {'diperbarui menjadi versi validasi silang' if promoted else 'tetap menggunakan versi sebelumnya'}.

Reproduksi:

```bash
python -m pip install -r environments/teman/requirements-optimization.txt
python scripts/extend_features.py
python scripts/optimize_ks.py --features base
python scripts/optimize_ks.py --features extended
python scripts/optimize_ks.py --features compact
python scripts/crossvalidate_ks.py
```
'''
    (OUT/'laporan_validasi_silang_ks.md').write_text(text)
    print('Created submission_ks.csv; promoted to submission.csv:',promoted,flush=True)


if __name__=='__main__':
    main()
