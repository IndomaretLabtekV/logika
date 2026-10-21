"""Select stronger credit-risk models using a separate tuning set and KS."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

# Optional local install used in this workspace; normal installed packages also work.
if Path('/tmp/logika-boosting-libs').exists():
    sys.path.insert(0, '/tmp/logika-boosting-libs')

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/optimization'
EXCLUDED = {'application_id','customer_id','application_datetime','decision_datetime','target','split'}


def ks(y, probability):
    fpr, tpr, _ = roc_curve(y, probability, drop_intermediate=False)
    return float(np.max(np.abs(tpr - fpr)))


def lgb_ks(y, probability):
    return 'ks', ks(y, probability), True


def metrics(y, probability):
    return {'ks': ks(y, probability), 'roc_auc': roc_auc_score(y, probability),
            'average_precision': average_precision_score(y, probability),
            'log_loss': log_loss(y, probability)}


def make_encoder(frame):
    categorical = frame.select_dtypes(include='object').columns.tolist()
    numeric = [c for c in frame if c not in categorical]
    encoder = ColumnTransformer([
        ('num', 'passthrough', numeric),
        ('cat', Pipeline([('fill', SimpleImputer(strategy='most_frequent')),
                          ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical)
    ], verbose_feature_names_out=False)
    return encoder, categorical


def candidates():
    return [
        {'name':'xgb_depth3', 'family':'xgb', 'params':{'max_depth':3,'min_child_weight':30,'reg_lambda':20,'scale_pos_weight':1}},
        {'name':'xgb_depth4_weight3', 'family':'xgb', 'params':{'max_depth':4,'min_child_weight':70,'reg_lambda':25,'scale_pos_weight':3}},
        {'name':'lgb_leaves15', 'family':'lgb', 'params':{'num_leaves':15,'min_child_samples':100,'reg_lambda':10,'scale_pos_weight':1}},
        {'name':'lgb_leaves31', 'family':'lgb', 'params':{'num_leaves':31,'min_child_samples':180,'reg_lambda':20,'scale_pos_weight':1}},
        {'name':'lgb_leaves31_weight3', 'family':'lgb', 'params':{'num_leaves':31,'min_child_samples':200,'reg_lambda':20,'scale_pos_weight':3}},
        {'name':'cat_depth5', 'family':'cat', 'params':{'depth':5,'l2_leaf_reg':12}},
        {'name':'cat_depth6', 'family':'cat', 'params':{'depth':6,'l2_leaf_reg':15}},
    ]


def weighted_probability(model, matrix, weight=1):
    p = model.predict_proba(matrix)[:,1]
    # Undo the training class weighting before combining probability predictions.
    return p / (weight * (1-p) + p)


def fit_candidate(spec, data, ti, vi, columns, iterations=None, seed=2026):
    x = data[columns]
    y = data.target.astype(int)
    params = dict(spec['params'])
    family = spec['family']
    encoder, categorical = make_encoder(x)
    if family == 'cat':
        xc = x.copy()
        for col in categorical:
            xc[col] = xc[col].fillna('MISSING').astype(str)
        model = CatBoostClassifier(
            iterations=iterations or 2200, learning_rate=.04, loss_function='Logloss',
            eval_metric='AUC', random_seed=seed, thread_count=8, verbose=False,
            allow_writing_files=False, **params)
        if vi is not None:
            model.fit(xc.iloc[ti], y.iloc[ti], cat_features=categorical,
                      eval_set=(xc.iloc[vi],y.iloc[vi]), early_stopping_rounds=200, use_best_model=False)
            # Compare KS at fixed intervals, rather than selecting only on AUC.
            tree_values = list(range(50, model.tree_count_+1, 25))
            if model.tree_count_ not in tree_values:
                tree_values.append(model.tree_count_)
            best = max(tree_values, key=lambda n: ks(y.iloc[vi],model.predict_proba(xc.iloc[vi],ntree_end=n)[:,1]))
            model.shrink(best)
            probability = model.predict_proba(xc.iloc[vi])[:,1]
        else:
            model.fit(xc.iloc[ti], y.iloc[ti], cat_features=categorical)
            probability = None
        class_weights=params.get('class_weights',[1,1])
        weight=class_weights[1]/class_weights[0]
        if probability is not None:
            probability=probability/(weight*(1-probability)+probability)
        bundle = {'family':family,'model':model,'columns':columns,'categorical':categorical,'weight':weight}
        best = model.tree_count_
    else:
        xt = encoder.fit_transform(x.iloc[ti])
        xv = encoder.transform(x.iloc[vi]) if vi is not None else None
        weight = params.get('scale_pos_weight',1)
        if family == 'xgb':
            callbacks = [EarlyStopping(rounds=150,metric_name='ks',maximize=True,save_best=True)] if vi is not None else None
            model = XGBClassifier(
                n_estimators=iterations or 2400, learning_rate=.035, subsample=.85,
                colsample_bytree=.85, objective='binary:logistic', tree_method='hist',
                n_jobs=8, random_state=seed, eval_metric=ks if vi is not None else 'logloss',
                callbacks=callbacks, **params)
            model.fit(xt,y.iloc[ti],eval_set=[(xv,y.iloc[vi])] if vi is not None else None,verbose=False)
            best = model.best_iteration+1 if vi is not None else iterations
        else:
            model = LGBMClassifier(
                n_estimators=iterations or 2400, learning_rate=.03,
                objective='binary', metric='None', subsample=.85, subsample_freq=1,
                colsample_bytree=.85, reg_alpha=.1, n_jobs=8, random_state=seed,
                verbosity=-1, **params)
            kwargs = {'eval_X':xv, 'eval_y':y.iloc[vi], 'eval_metric':lgb_ks,
                      'callbacks':[early_stopping(150,first_metric_only=True,verbose=False),log_evaluation(0)]} if vi is not None else {}
            model.fit(xt,y.iloc[ti],**kwargs)
            best = model.best_iteration_ if vi is not None else iterations
        probability = weighted_probability(model,xv,weight) if vi is not None else None
        bundle = {'family':family,'model':model,'encoder':encoder,'columns':columns,'weight':weight}
    return bundle, probability, int(best)


def predict_bundle(bundle, frame):
    x = frame[bundle['columns']].copy()
    if bundle['family']=='cat':
        for col in bundle['categorical']:
            x[col] = x[col].fillna('MISSING').astype(str)
        p=bundle['model'].predict_proba(x)[:,1]
        weight=bundle.get('weight',1)
        return p/(weight*(1-p)+p)
    return weighted_probability(bundle['model'],bundle['encoder'].transform(x),bundle['weight'])


def tune(feature_set):
    OUT.mkdir(parents=True,exist_ok=True)
    file_name = 'application_features.parquet' if feature_set=='base' else 'application_features_extended.parquet'
    data = pd.read_parquet(ROOT/'data/processed'/file_name)
    data = data[data.split.eq('train')].reset_index(drop=True)
    columns = [c for c in data if c not in EXCLUDED and data[c].nunique(dropna=True)>1]
    if feature_set=='compact':
        # Gain is learned on the development training rows, never on confirmation.
        reference=joblib.load(OUT/'tune_extended_lgb_leaves31.joblib')
        names=reference['encoder'].get_feature_names_out()
        gains=reference['model'].booster_.feature_importance(importance_type='gain')
        ranking=pd.Series(gains,index=names).sort_values(ascending=False)
        baseline=json.loads((ROOT/'artifacts/model_config.json').read_text())['all_feature_columns']
        cats=[c for c in columns if data[c].dtype==object]
        chosen=list(baseline)
        for name in ranking.index:
            raw=name if name in columns else next((c for c in sorted(cats,key=len,reverse=True) if name.startswith(c+'_')),None)
            if raw and raw not in chosen:
                chosen.append(raw)
            if len(chosen)>=200:
                break
        columns=chosen
    # Preserve the original 17,500-row benchmark holdout for confirmation.
    development, confirmation = train_test_split(np.arange(len(data)),test_size=.25,random_state=2026,stratify=data.target)
    ti, vi = train_test_split(development,test_size=.2,random_state=4917,stratify=data.target.iloc[development])
    assert set(confirmation).isdisjoint(ti) and set(confirmation).isdisjoint(vi)
    rows = []
    predictions = pd.DataFrame({'application_id':data.application_id.iloc[vi].to_numpy(),'target':data.target.iloc[vi].to_numpy()})
    manifest = {'feature_set':feature_set,'feature_file':file_name,'columns':columns,
                'train_n':len(ti),'tune_n':len(vi),'confirmation_n':len(confirmation),'candidates':[],
                'selection_metric':'KS statistic','tuning_seed':4917,'confirmation_seed':2026}
    np.savez(OUT/f'{feature_set}_splits.npz',train=ti,tune=vi,confirmation=confirmation,development=development)
    specs=candidates()
    if feature_set=='compact':
        specs=[s for s in specs if s['name'] in ['xgb_depth3','lgb_leaves15','lgb_leaves31','cat_depth5']]
    for spec in specs:
        started = time.monotonic()
        print(f"Tuning {feature_set}/{spec['name']}: {len(columns)} features",flush=True)
        bundle, probability, best = fit_candidate(spec,data,ti,vi,columns)
        score = metrics(data.target.iloc[vi],probability)
        result = dict(spec,iterations=best,feature_set=feature_set,**score,seconds=time.monotonic()-started)
        rows.append(result)
        manifest['candidates'].append(dict(spec,iterations=best,feature_set=feature_set,metrics=score))
        predictions[spec['name']] = probability
        joblib.dump(bundle,OUT/f'tune_{feature_set}_{spec["name"]}.joblib',compress=3)
        predictions.to_parquet(OUT/f'{feature_set}_tuning_predictions.parquet',index=False)
        pd.DataFrame(rows).drop(columns='params').to_csv(OUT/f'{feature_set}_tuning_results.csv',index=False)
        (OUT/f'{feature_set}_manifest.json').write_text(json.dumps(manifest,indent=2))
        print(json.dumps(result),flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',choices=['base','extended','compact'],default='base')
    args=parser.parse_args()
    tune(args.features)


if __name__=='__main__':
    main()
