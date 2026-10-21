"""Evaluate richer purchase features and fixed tree counts on the same CV folds."""
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path
import sys
import time

from optimize_ks import ROOT, OUT, fit_candidate, make_encoder, metrics, predict_bundle
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

STAGE_OUT=OUT/'refinement'


def stage_probability(bundle, frame, trees):
    x=frame[bundle['columns']].copy()
    if bundle['family']=='cat':
        for c in bundle['categorical']:x[c]=x[c].fillna('MISSING').astype(str)
        p=bundle['model'].predict_proba(x,ntree_end=trees)[:,1]
        weight=bundle.get('weight',1)
        return p/(weight*(1-p)+p)
    transformed=bundle['encoder'].transform(x)
    p=bundle['model'].predict_proba(transformed,num_iteration=trees)[:,1]
    weight=bundle['weight']
    return p/(weight*(1-p)+p)


def definitions():
    compact=json.loads((OUT/'compact_manifest.json').read_text())['columns']
    extended=json.loads((OUT/'extended_manifest.json').read_text())['columns']
    return [
        {'name':'compact_cat5','family':'cat','params':{'depth':5,'l2_leaf_reg':12},'columns':compact,'stages':[400,600,900,1200]},
        {'name':'price_cat5','family':'cat','params':{'depth':5,'l2_leaf_reg':15},'columns':compact,'add_v3':True,'stages':[400,700,1000]},
        {'name':'price_lgb31','family':'lgb','params':{'num_leaves':31,'min_child_samples':180,'reg_lambda':20,'scale_pos_weight':1},'columns':compact,'add_v3':True,'stages':[400,700,1000]},
        {'name':'extended_lgb31','family':'lgb','params':{'num_leaves':31,'min_child_samples':180,'reg_lambda':20,'scale_pos_weight':1},'columns':extended,'stages':[486,700,1000]},
    ]


def main():
    global STAGE_OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--balanced',action='store_true')
    args=parser.parse_args()
    if args.balanced:STAGE_OUT=OUT/'refinement_balanced'
    STAGE_OUT.mkdir(parents=True,exist_ok=True)
    full=pd.read_parquet(ROOT/'data/processed/application_features_v3.parquet')
    data=full[full.split.eq('train')].reset_index(drop=True)
    test=full[full.split.eq('test')].reset_index(drop=True)
    specs=definitions()
    if args.balanced:
        compact=json.loads((OUT/'compact_manifest.json').read_text())['columns']
        specs=[
            {'name':'price_cat5_balanced','family':'cat','params':{'depth':5,'l2_leaf_reg':25,'class_weights':[1,6]},'columns':compact,'add_v3':True,'stages':[400,700,1000]},
            {'name':'price_lgb31_balanced','family':'lgb','params':{'num_leaves':31,'min_child_samples':200,'reg_lambda':40,'scale_pos_weight':6},'columns':compact,'add_v3':True,'stages':[250,400,600]},
        ]
    added=[c for c in data if c.startswith('v3_') and data[c].nunique(dropna=True)>1]
    for spec in specs:
        if spec.get('add_v3'):spec['columns']=list(dict.fromkeys(spec['columns']+added))
        assert not set(spec['columns'])&{'target','split','customer_id','application_id','decision_datetime','application_datetime'}
    (STAGE_OUT/'definitions.json').write_text(json.dumps(specs,indent=2))
    y=data.target.astype(int)
    folds=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=8247).split(data,y))
    old=pd.read_parquet(OUT/'cross_validation_oof.parquet')
    assert old.application_id.equals(data.application_id)
    reference=json.loads((OUT/'cross_validation_summary.json').read_text())
    old_p=sum(w*old[n] for n,w in reference['selected']['recipe'].items())
    oof=pd.DataFrame({'application_id':data.application_id,'target':y,'public_054710':old_p})
    old_test=pd.read_parquet(OUT/'cross_validation_test_predictions.parquet')
    assert old_test.application_id.equals(test.application_id)
    test_pred=pd.DataFrame({'application_id':test.application_id,
                            'public_054710':sum(w*old_test[n] for n,w in reference['selected']['recipe'].items())})
    for spec in specs:
        for stage in spec['stages']:
            name=f'{spec["name"]}_{stage}'
            oof[name]=np.nan
            test_pred[name]=0.0
    rows=[]
    for fold,(ti,vi) in enumerate(folds,1):
        for spec in specs:
            started=time.monotonic()
            print(f'Fold {fold}/3: {spec["name"]}, {len(spec["columns"])} features, {max(spec["stages"])} trees',flush=True)
            bundle,_,_=fit_candidate(spec,data,ti,None,spec['columns'],iterations=max(spec['stages']))
            for stage in spec['stages']:
                name=f'{spec["name"]}_{stage}'
                p=stage_probability(bundle,data.iloc[vi],stage)
                oof.loc[vi,name]=p
                test_pred[name]+=stage_probability(bundle,test,stage)/3
                row={'fold':fold,'name':name,'trees':stage,'features':len(spec['columns']),**metrics(y.iloc[vi],p)}
                rows.append(row)
                print(json.dumps(row),flush=True)
            joblib.dump(bundle,STAGE_OUT/f'fold{fold}_{spec["name"]}.joblib',compress=3)
            oof.to_parquet(STAGE_OUT/'oof.parquet',index=False)
            test_pred.to_parquet(STAGE_OUT/'test_predictions.parquet',index=False)
            pd.DataFrame(rows).to_csv(STAGE_OUT/'fold_metrics.csv',index=False)
            print(f'Completed {spec["name"]} in {time.monotonic()-started:.1f}s',flush=True)
    assert oof.notna().all().all()
    summaries=[]
    for name in oof.columns[2:]:
        scores=metrics(y,oof[name])
        per_fold=[metrics(y.iloc[vi],oof[name].iloc[vi])['ks'] for _,vi in folds]
        summaries.append({'name':name,**scores,'mean_fold_ks':float(np.mean(per_fold)),'fold_ks':per_fold})
    summaries.sort(key=lambda r:r['mean_fold_ks'],reverse=True)
    (STAGE_OUT/'summary.json').write_text(json.dumps(summaries,indent=2))
    pd.DataFrame(summaries).drop(columns='fold_ks').to_csv(STAGE_OUT/'summary.csv',index=False)
    print('Final three-fold comparisons:',json.dumps(summaries[:8]),flush=True)


if __name__=='__main__':
    main()
