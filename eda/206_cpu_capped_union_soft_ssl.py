"""Nested CPU LightGBM capped-union features and prior-preserving soft pseudo-labels.

One documented policy, no outer weight/parameter search. Run --screen on42/0
before --evaluate on five folds and three seeds. Each fit also predicts test.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

import valid2 as v
from common import ROOT,PQ,OUT,features
from teammate_feature_adapter import feature_schema
from nested_input_pools import teammate_inputs
from conditional_input_pools import gain_columns

spec=importlib.util.spec_from_file_location('cpu_union_base',Path(__file__).with_name('173_nested_kumo_inputs.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def params(seed,soft=False):
    p={**v.LGB,'num_threads':4,'deterministic':True,'force_col_wise':True,'seed':seed,
       'feature_fraction_seed':seed,'bagging_seed':seed,'data_random_seed':seed}
    p.pop('n_jobs',None)
    if soft:p.update(objective='cross_entropy',metric='cross_entropy')
    return p


def vocab(F,fit):
    out=F.copy(deep=False)
    for c in v.CAT_COLS:
        cats=sorted(F.iloc[fit][c].dropna().astype(str).unique())
        out[c]=pd.Categorical(F[c].astype('string'),categories=cats)
    return out


def pseudo(prob,prior):
    p=np.asarray(prob,dtype=float);assert p.shape==(30000,) and np.isfinite(p).all()
    n=2000;np_=int(round(n*prior));nn=n-np_;order=np.argsort(p,kind='stable')
    neg,pos=order[:nn],order[-np_:]
    lo,hi=float(p[neg].mean()),float(p[pos].mean())
    if not lo<prior<hi:raise ValueError('Pseudo confidence groups do not bracket the real prior')
    alpha=(prior-lo)/(hi-lo);weights=np.r_[np.full(nn,500*(1-alpha)/nn),np.full(np_,500*alpha/np_)]
    rows=np.r_[neg,pos];assert len(set(rows))==n and np.all(weights>0)
    assert abs(weights.sum()-500)<1e-8 and abs(np.average(p[rows],weights=weights)-prior)<1e-12
    return rows,p[rows],weights,{'rows':n,'positive_tail_rows':np_,'negative_tail_rows':nn,'prior':prior,
       'low_tail_mean':lo,'high_tail_mean':hi,'weighted_soft_prior':float(np.average(p[rows],weights=weights)),
       'total_weight':float(weights.sum())}


def fit_student(F,y,real,test,teacher,rounds,seed):
    p=teacher.predict(F.iloc[test],num_threads=4);ix,soft,w,info=pseudo(p,float(y[real].mean()))
    selected=test[ix];assert set(real).isdisjoint(selected)
    training=pd.concat([F.iloc[real],F.iloc[selected]],ignore_index=True)
    labels=np.r_[y[real],soft];weights=np.r_[np.ones(len(real)),w]
    model=lgb.train(params(seed,True),lgb.Dataset(training,label=labels,weight=weights),num_boost_round=rounds)
    info.update(test_row_positions=ix.tolist(),soft_targets=soft.tolist(),pseudo_weights=w.tolist(),real_rows=len(real))
    return model,info,p


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results/v58/cpu_capped_union_ssl');ap.add_argument('--seed',type=int,default=42);ap.add_argument('--fold',type=int,choices=range(5),default=0)
    a_group=ap.add_mutually_exclusive_group(required=True);a_group.add_argument('--screen',action='store_true');a_group.add_argument('--evaluate',action='store_true');a=ap.parse_args()
    work=a.out.resolve();folder=work/f's{a.seed}_f{a.fold}';folder.mkdir(parents=True,exist_ok=True)
    source_names=(Path(__file__).name,'valid2.py','common.py','173_nested_kumo_inputs.py','nested_input_pools.py','teammate_feature_adapter.py','conditional_input_pools.py','204_cpu_union_soft_ssl.py')
    request={'seed':a.seed,'fold':a.fold,'sources_sha256':{n:sha(ROOT/'eda'/n) for n in source_names},
             'protocol_sha256':sha(ROOT/'docs/integration/protokol_cpu_capped_union_21okt.md'),'parent_protocol_sha256':sha(ROOT/'docs/integration/protokol_cpu_union_ssl_21okt.md'),'features_sha256':sha(PQ/'features.parquet'),
             'packages':{n:importlib.metadata.version(n) for n in ('lightgbm','pandas','polars','numpy','scikit-learn')},
             'outer_recipe':'0.75*v31+0.25*rank(union_policy); no outer selection','gpu_used':False}
    manifest=folder/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==request,'Changed recipe/source; use fresh output'
    else:save(manifest,request)
    if (folder/'complete.json').exists():
        record=json.loads((folder/'complete.json').read_text())
        for rel,expected in record['outputs_sha256'].items():assert sha(work/rel)==expected
        print('Verified completed CPU fit',a.seed,a.fold,flush=True);return
    X,Xt,y,trids,teids=v.frames(features());ids=np.r_[trids.to_numpy(),teids.to_numpy()];labels=np.r_[y,np.zeros(30000)]
    assert len(ids)==len(set(ids))==100000 and len(y)==70000
    tr,va=v.outer(y,a.seed)[a.fold];fi,st=v.inner(y[tr],a.seed,a.fold);fi,st=tr[fi],tr[st];test=np.arange(70000,100000)
    cache=ROOT/'results/v56/teammate_features'/f'outer_s{a.seed}_f{a.fold}'
    frame,reference=base.read_pool(cache,'outer',a.seed,a.fold,trids[tr].to_list())
    friend=teammate_inputs(frame,trids.to_list(),y,trids[tr].to_list());numeric=feature_schema(friend)['numeric']
    numeric_test=frame.set_index('application_id').loc[teids.to_list(),numeric].reset_index(drop=True)
    main_all=pd.concat([X,Xt],ignore_index=True);extra=pd.concat([friend[numeric],numeric_test],ignore_index=True).rename(columns=lambda c:'teman__'+c)
    union=pd.concat([main_all,extra],axis=1);assert union.shape==(100000,main_all.shape[1]+len(numeric))
    assert all(c not in v.ID_COLS for c in union.columns)
    del X,Xt,friend,frame,numeric_test,extra;gc.collect()
    save(folder/'splits.json',{'reference':reference,'outer_train_ids':ids[tr].tolist(),'outer_validation_ids':ids[va].tolist(),
         'inner_fit_ids':ids[fi].tolist(),'inner_stop_ids':ids[st].tolist(),'official_test_ids':ids[test].tolist()})
    def capped(F,rows,stage):
        friend_cols=[c for c in F if c.startswith('teman__')]
        selection,gain,metadata=gain_columns(F[friend_cols+v.CAT_COLS],labels,rows,seed=a.seed)
        selection=[c for c in selection if c.startswith('teman__')]
        assert len(selection)==100
        cols=main_all.columns.tolist()+selection
        save(folder/f'{stage}_selector.json',{'fit_ids':ids[rows].tolist(),'columns':cols,'selector':metadata})
        gain.to_csv(folder/f'{stage}_gain.csv')
        return F[cols]
    def teacher(F,name):
        if name=='union':F=capped(F,fi,'inner')
        early=folder/f'{name}_inner_stop.txt';refit=folder/f'{name}_inner_refit.txt';budget=folder/f'{name}_budget.json'
        G=vocab(F,fi);start=time.monotonic()
        if budget.exists():
            cfg=json.loads(budget.read_text());assert sha(early)==cfg['early_sha256'] and sha(refit)==cfg['refit_sha256']
            model=lgb.Booster(model_file=str(refit));n=cfg['rounds']
        else:
            d=lgb.Dataset(G.iloc[fi],label=y[fi]);m=lgb.train(params(a.seed),d,num_boost_round=3000,valid_sets=[lgb.Dataset(G.iloc[st],label=y[st],reference=d)],callbacks=[lgb.early_stopping(200,verbose=False)])
            n=math.ceil(m.best_iteration*1.15);m.save_model(str(early));del m,d;gc.collect()
            model=lgb.train(params(a.seed),lgb.Dataset(G.iloc[fi],label=y[fi]),num_boost_round=n);model.save_model(str(refit))
            save(budget,{'rounds':n,'early_sha256':sha(early),'refit_sha256':sha(refit)})
        p=model.predict(G.iloc[st],num_threads=4);print(name,'inner KS',v.ks(y[st],p),'rounds',n,'seconds',time.monotonic()-start,flush=True)
        return G,model,n,p
    G,main_model,nmain,pm=teacher(main_all,'main');del G,main_model;gc.collect()
    U,union_teacher,nunion,pu=teacher(union,'union')
    inner_model=folder/'union_inner_ssl.txt';inner_meta=folder/'inner_pseudo.json'
    if inner_model.exists() and inner_meta.exists():
        student=lgb.Booster(model_file=str(inner_model));info=json.loads(inner_meta.read_text());assert info['model_sha256']==sha(inner_model)
    else:
        student,info,teacher_test=fit_student(U,labels,fi,test,union_teacher,nunion,a.seed);student.save_model(str(inner_model));info['model_sha256']=sha(inner_model);save(inner_meta,info)
        pl.DataFrame({'application_id':ids[test],'teacher_prediction':teacher_test}).write_parquet(folder/'inner_teacher_test.parquet')
    ps=student.predict(U.iloc[st],num_threads=4);use_ssl=v.ks(y[st],ps)>v.ks(y[st],pu)
    decision={'main_ks':v.ks(y[st],pm),'union_ks':v.ks(y[st],pu),'ssl_ks':v.ks(y[st],ps),'use_ssl':use_ssl,
              'promising_inner':max(v.ks(y[st],pu),v.ks(y[st],ps))>v.ks(y[st],pm),'outer_scores_used':False}
    select=folder/'frozen_selection.json'
    if select.exists():assert json.loads(select.read_text())==decision
    else:save(select,decision)
    print('FROZEN INNER DECISION',json.dumps(decision),flush=True)
    pl.DataFrame({'application_id':ids[st],'target':y[st],'main':pm,'union':pu,'ssl':ps}).write_parquet(folder/'inner_predictions.parquet')
    del U,union_teacher,student;gc.collect()
    fig,ax=plt.subplots(figsize=(7,3));ax.bar(['main','union','softSSL'],[decision[k] for k in ('main_ks','union_ks','ssl_ks')]);ax.set(title='Inner only; outer selection prohibited',ylabel='KS');fig.tight_layout();fig.savefig(OUT/f'206_s{a.seed}_f{a.fold}_inner.png',dpi=130);plt.close(fig)
    if a.screen:return
    predictions={};models={}
    for name,F,n in (('main',main_all,nmain),('union',union,nunion)):
        if name=='union':F=capped(F,tr,'outer')
        G=vocab(F,tr);path=folder/f'{name}_outer.txt';m=lgb.train(params(a.seed),lgb.Dataset(G.iloc[tr],label=y[tr]),num_boost_round=n);m.save_model(str(path));models[name]=sha(path)
        predictions[name]=m.predict(G.iloc[np.r_[va,test]],num_threads=4)
        if name=='union':
            if use_ssl:
                s,info,p=fit_student(G,labels,tr,test,m,n,a.seed);sp=folder/'ssl_outer.txt';s.save_model(str(sp));models['ssl']=sha(sp);info['model_sha256']=sha(sp);save(folder/'outer_pseudo.json',info)
                pl.DataFrame({'application_id':ids[test],'teacher_prediction':p}).write_parquet(folder/'outer_teacher_test.parquet')
                predictions['candidate']=s.predict(G.iloc[np.r_[va,test]],num_threads=4);del s
            else:predictions['candidate']=predictions['union']
        del G,m;gc.collect()
    outputs={}
    for role,rows,sl in (('outer',va,slice(0,len(va))),('test',test,slice(len(va),None))):
        data={'application_id':ids[rows],**{name:p[sl] for name,p in predictions.items()}}
        if role=='outer':data['target']=y[va]
        path=work/f's{a.seed}_f{a.fold}_{role}.parquet';pl.DataFrame(data).write_parquet(path);outputs[str(path.relative_to(work))]=sha(path)
    score={name:v.ks(y[va],p[:len(va)]) for name,p in predictions.items()};save(folder/'complete.json',{'outputs_sha256':outputs,'models_sha256':models,'selection_sha256':sha(select),'manifest_sha256':sha(manifest),'outer_ks':score})
    os.sync();print('COMPLETE',a.seed,a.fold,json.dumps(score),flush=True)


if __name__=='__main__':main()
