"""Fit a locked refinement recipe with 10-fold bagging and export a new file."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil

from optimize_ks import ROOT,OUT,fit_candidate,metrics,predict_bundle
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

FINAL=OUT/'tenfold'


def main():
    config=json.loads((FINAL/'locked_recipe.json').read_text())
    full=pd.read_parquet(ROOT/'data/processed'/config['final_feature_file'])
    data=full[full.split.eq('train')].reset_index(drop=True)
    test=full[full.split.eq('test')].reset_index(drop=True)
    y=data.target.astype(int)
    recipe=config['recipe']
    count=config['final_folds']
    folds=list(StratifiedKFold(n_splits=count,shuffle=True,random_state=config['final_fold_seed']).split(data,y))
    original=pd.read_parquet(FINAL/'selection_oof.parquet')
    assert original.application_id.equals(data.application_id)
    old_test=pd.read_parquet(FINAL/'selection_test_predictions.parquet')
    assert old_test.application_id.equals(test.application_id)
    oof=pd.DataFrame({'application_id':data.application_id,'target':y})
    predicted=pd.DataFrame({'application_id':test.application_id})
    for name in recipe:
        oof[name]=original.public_054710 if name=='public_054710' else np.nan
        predicted[name]=old_test.public_054710 if name=='public_054710' else 0.0
    rows=[]

    def train_one(fold,name,ti,vi):
        spec=config['definitions'][name]
        assert not set(spec['columns'])&{'target','split','customer_id','application_id','application_datetime','decision_datetime'}
        print(f'Training fold {fold}/{count}: {name} ({len(ti):,} rows)',flush=True)
        bundle,_,_=fit_candidate(spec,data,ti,None,spec['columns'],iterations=spec['iterations'],seed=2026+fold)
        pv=predict_bundle(bundle,data.iloc[vi])
        pt=predict_bundle(bundle,test)
        assert np.isfinite(pv).all() and np.isfinite(pt).all()
        model_path=FINAL/f'fold{fold}_{name}.joblib'
        joblib.dump(bundle,model_path,compress=3)
        return fold,name,vi,pv,pt,metrics(y.iloc[vi],pv)

    tasks=[(fold,name,ti,vi) for fold,(ti,vi) in enumerate(folds,1)
           for name in recipe if name!='public_054710']
    # Each independent model uses eight threads; run two models concurrently.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(train_one,*task) for task in tasks]
        for future in as_completed(futures):
            fold,name,vi,pv,pt,score=future.result()
            oof.loc[vi,name]=pv
            predicted[name]+=pt/count
            row={'fold':fold,'name':name,**score}
            rows.append(row)
            oof.to_parquet(FINAL/'tenfold_oof.parquet',index=False)
            predicted.to_parquet(FINAL/'tenfold_test_predictions.parquet',index=False)
            pd.DataFrame(rows).to_csv(FINAL/'fold_metrics.csv',index=False)
            print('Completed:',json.dumps(row),flush=True)
    assert oof.notna().all().all()
    p=sum(weight*oof[name] for name,weight in recipe.items())
    scores=metrics(y,p)
    old_score=metrics(y,original.public_054710)
    old_tenfold_ks=[metrics(y.iloc[vi],original.public_054710.iloc[vi])['ks'] for _,vi in folds]
    new_tenfold_ks=[metrics(y.iloc[vi],p.iloc[vi])['ks'] for _,vi in folds]
    scores['mean_fold_ks']=float(np.mean(new_tenfold_ks))
    scores['fold_ks']=new_tenfold_ks
    old_score['mean_on_same_tenfold_groups']=float(np.mean(old_tenfold_ks))
    report={'recipe':recipe,'selection_comparison_3fold':{'champion':config['champion'],'new':config['selection']},
            'tenfold_aggregate_oof':{'champion':old_score,'new':scores},
            'final_folds':count,'fold_seed':config['final_fold_seed'],'model_seeds':'2026 + fold',
            'champion_public_score':.54710,'new_public_score':None,
            'caveat':'Feature selection and parameters were previously selected on these training labels. Scores are validation estimates, not an unbiased final test. New OOF models use 90% training rows per fold; champion OOF used 67%.'}
    (FINAL/'validation_report.json').write_text(json.dumps(report,indent=2))
    print('Validation report:',json.dumps(report),flush=True)
    export_submission(data,test,predicted,config,report)


def export_submission(data,test,predicted,config,report):
    sample=pd.read_csv(ROOT/'data/sample_submission.csv')
    probability=sum(weight*predicted[name] for name,weight in config['recipe'].items())
    raw_prediction=pd.DataFrame({'application_id':test.application_id,'default_probability':probability})
    assert set(raw_prediction.application_id)==set(sample.application_id)
    assert raw_prediction.application_id.is_unique
    submission=sample[['application_id']].merge(raw_prediction,on='application_id',how='left',validate='one_to_one',sort=False)
    assert submission.application_id.equals(sample.application_id)
    assert submission.shape==(30000,2) and submission.application_id.is_unique
    assert np.isfinite(submission.default_probability).all() and submission.default_probability.between(0,1).all()
    path=ROOT / 'results/teman/submission_v3.csv'
    submission.to_csv(path,index=False,float_format='%.12f')
    reread=pd.read_csv(path)
    assert reread.columns.tolist()==sample.columns.tolist()
    assert reread.application_id.equals(sample.application_id)
    assert reread.default_probability.notna().all() and reread.default_probability.between(0,1).all()
    checkpoint=OUT/'submission_public_0_54710.csv'
    assert not np.allclose(pd.read_csv(checkpoint).default_probability,reread.default_probability)
    new=report['tenfold_aggregate_oof']['new']
    old=report['tenfold_aggregate_oof']['champion']
    promote=new['ks']>old['ks']
    if promote:shutil.copy2(path,ROOT / 'results/teman/submission.csv')
    metadata=dict(report,file='submission_v3.csv',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  training_rows=70000,test_rows=30000,source_file='submission_v3.csv',
                  probability_summary=reread.default_probability.describe().to_dict(),promoted_to_submission_csv=promote,
                  model_artifacts=[str((FINAL/f'fold{fold}_{name}.joblib').relative_to(ROOT))
                                   for name in config['recipe'] if name!='public_054710' for fold in range(1,config['final_folds']+1)],
                  champion_file=str(checkpoint.relative_to(ROOT)),
                  checks={'correct_columns':True,'sample_id_order':True,'unique_ids':True,'no_missing':True,'valid_probability_range':True})
    (FINAL/'submission_v3_metadata.json').write_text(json.dumps(metadata,indent=2))
    if promote:
        current=ROOT/'artifacts/submission_metadata.json'
        old_metadata=FINAL/'submission_public_0_54710_metadata.json'
        if not old_metadata.exists():shutil.copy2(current,old_metadata)
        current.write_text(json.dumps(dict(metadata,file='submission.csv'),indent=2))
    old3=config['champion'];new3=config['selection']
    text=f'''# Submission versi 3 untuk KS

Versi sebelumnya (`submission_ks.csv`) memperoleh public KS **0,54710**, terkonfirmasi lewat API Kaggle. Saat pengecekan 12 Oktober 2026, peringkat 10 bernilai 0,56333. Public KS versi baru belum tersedia; tidak ada upload otomatis.

Penambahan fitur mencakup harga per unit dan komposisi belanja per kategori, variasi belanja bulanan, shrinkage frekuensi keterlambatan historis, skor biro berbobot usia laporan, dan cakupan cicilan oleh arus kas. Semua observasi diaudit sebelum keputusan. Prior keterlambatan berasal dari kejadian historis yang diketahui, bukan label target.

| Validasi yang sama, tiga fold | Versi 0,54710 | Resep baru |
|---|---:|---:|
| Rata-rata KS fold | {old3['mean_fold_ks']:.5f} | {new3['mean_fold_ks']:.5f} |
| KS seluruh OOF | {old3['ks']:.5f} | {new3['ks']:.5f} |
| ROC-AUC OOF | {old3['roc_auc']:.5f} | {new3['roc_auc']:.5f} |

Setelah memilih jumlah pohon dan bobot, resep dikunci sebelum pelatihan akhir 10 fold. Setiap model baru berlatih pada 90% label dan seed bervariasi menurut fold. Parameter dan bobot tidak diubah berdasarkan hasil 10 fold. Prediksi tiap anggota dirata-ratakan antarfold; jika resep memuat `public_054710`, bagian itu memakai prediksi versi lama yang sudah tersimpan.

```json
{json.dumps(config['recipe'],indent=2)}
```

KS OOF gabungan pada tahap akhir: **{new['ks']:.5f}**, dibanding **{old['ks']:.5f}** untuk prediksi OOF versi lama pada 70.000 baris yang sama. OOF baru memakai model dengan proporsi training 90%; OOF versi lama 67%. Skor rata-rata fold 3 dan 10 tidak dibandingkan langsung karena ukuran fold memengaruhi statistik maksimum KS.

Seleksi fitur, jumlah pohon, dan bobot pernah memakai label training yang sama; hasil lokal dapat mengandung optimisme seleksi dan tidak menjamin skor public/private tertentu. Tidak ada label test yang digunakan.

File `submission_v3.csv` berisi 30.000 probabilitas kontinu sesuai header dan urutan sample submission; seluruh probabilitas finite dan di [0,1]. Checkpoint 0,54710 dipertahankan terpisah. `submission.csv` {'diperbarui ke versi 3' if promote else 'tetap memakai versi sebelumnya'}.

Reproduksi:

```bash
python scripts/build_features_v3.py
python scripts/refine_ks.py
python scripts/refine_ks.py --balanced
python scripts/select_refinement.py
python scripts/train_tenfold_submission.py
```
'''
    (FINAL/'laporan_submission_v3.md').write_text(text)
    print(f'Created submission_v3.csv; format checks passed; promoted={promote}',flush=True)


if __name__=='__main__':main()
