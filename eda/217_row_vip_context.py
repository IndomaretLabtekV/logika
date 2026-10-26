"""Local frozen-Kumo block-row VIP-COP adaptation; no backbone training.

One fixed compact feature view. Optimize real-row block membership on inner
search, lock one candidate, then compare on inner confirmation. Historical
exposure means neither inner subset is a pristine holdout. Never reads outer
labels for scoring, never predicts test, never exports a submission.
"""
import argparse
import faulthandler
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import valid2 as v
from common import ROOT
from strict_kumo_engine import predict, checkpoint_identity, KumoInferenceError

OUT=ROOT/'results/eda_23okt/row_vip'
CONFIG={'outer_seed':42,'fold':0,'model_seed':20261023,'sampling_seed':217,
        'groups':16,'selected_groups':8,'rounds':8,'batch':2,'temperatures':[.005,.02],
        'uniform_exploration':.1,'n_numeric':100,'estimators':4,'chunk':128,'memory_fraction':.7,
        'utility':'KS','selection':'largest sum of positive block coefficients across temperatures; top8 blocks',
        'search_confirmation_seed':217042,'paired_bootstrap':1000,
        'scope':'Frozen-model inner experiment only; no weight training, outer score or submission',
        'novelty':'Global real-row blocks, not historical feature VIP/retrieval/pseudo augmentation; strict preprocessing; identical search/confirmation fidelity',
        'limits':['Grouped adaptation, not individual-row Shapley values or exact paper reproduction.',
                  'Fixed two temperatures, fixed budget, no Hyperband; KS replaces paper AUROC.',
                  'Inner data were historically exposed; confirmation is held out from this search, not pristine research holdout.']}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))


def groups_for(rows,y,count,seed):
    rng=np.random.default_rng(seed)
    if len(rows)%count:
        raise ValueError('Equal-sized blocks require divisible fitting row count')
    positive=np.array_split(rng.permutation(rows[y[rows]==1]),count)
    negative=rng.permutation(rows[y[rows]==0])
    groups=[]; offset=0
    for part in positive:
        take=len(rows)//count-len(part)
        groups.append(np.sort(np.concatenate([part,negative[offset:offset+take]])))
        offset+=take
    if offset!=len(negative):
        raise ValueError('Rows must have binary labels')
    return groups


def membership(groups,selected):
    return np.sort(np.concatenate([groups[int(j)] for j in selected]))


def update(phi,coalitions,scores,step):
    # Unweighted least squares for fixed-size coalitions, as paper footnote2.
    gradient=np.zeros_like(phi)
    for selected,score in zip(coalitions,scores):
        gradient[selected]+=2*(phi[selected].sum()-score)/len(scores)
    return phi-step*gradient


def paired(y,a,b,seed):
    rng=np.random.default_rng(seed); classes=[np.flatnonzero(y==j) for j in [0,1]]
    deltas=[]
    for _ in range(CONFIG['paired_bootstrap']):
        rows=np.concatenate([rng.choice(c,len(c),replace=True) for c in classes])
        deltas.append(v.ks(y[rows],a[rows])-v.ks(y[rows],b[rows]))
    return {'delta_ks':v.ks(y,a)-v.ks(y,b),'percentile95':np.quantile(deltas,[.025,.975]).tolist(),
            'bootstrap_fraction_positive':float(np.mean(np.array(deltas)>0)),
            'caveat':'Conditional on this fixed split/recipe; not correction for historical research selection or model-seed variance.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare-only',action='store_true')
    a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    root=ROOT/'results/eda_23okt/affine_view'
    frame=pl.read_parquet(root/'features.parquet')
    recipe=json.loads((root/'preprocessing.json').read_text())
    train=pl.read_csv(ROOT/'data/train_applications.csv').sort('application_id')
    ids=train['application_id'].to_numpy(); yt=train['target'].to_numpy().astype(int)
    tr,outer=v.outer(yt,42)[0]; fi,st=v.inner(yt[tr],42,0)
    s,c=train_test_split(tr[st],test_size=.5,stratify=yt[tr[st]],random_state=CONFIG['search_confirmation_seed'])
    lookup={x:i for i,x in enumerate(frame['application_id'])}
    fit=np.array([lookup[x] for x in ids[tr[fi]]]); search=np.array([lookup[x] for x in ids[s]]); confirm=np.array([lookup[x] for x in ids[c]])
    assert not set(fit)&set(search) and not set(fit)&set(confirm) and not set(search)&set(confirm)
    assert recipe['fitting_ids_sha256']==hashlib.sha256('\n'.join(sorted(ids[tr[fi]])).encode()).hexdigest()
    # Selector and frozen model receive fitting labels only; query targets remain NaN.
    labels=frame['target'].to_numpy().astype(float)
    assert np.isfinite(labels[fit]).all() and np.isnan(labels[search]).all() and np.isnan(labels[confirm]).all()
    X=frame.select(recipe['kept_columns']).to_pandas()
    cols=v.fm_select(X,labels,fit,n_num=CONFIG['n_numeric'])
    group=groups_for(fit,labels,CONFIG['groups'],CONFIG['sampling_seed'])
    assert len(set(np.concatenate(group)))==len(fit)
    identity=checkpoint_identity('large')
    manifest={'config':CONFIG,'source_sha256':sha(__file__),'engine_sha256':sha(ROOT/'eda/strict_kumo_engine.py'),
              'selector_sha256':sha(ROOT/'eda/valid2.py'),'input_sha256':sha(root/'features.parquet'),
              'recipe_sha256':sha(root/'preprocessing.json'),'train_sha256':sha(ROOT/'data/train_applications.csv'),
              'checkpoint':identity,'columns':cols,'fitting_ids':frame['application_id'].gather(fit).to_list(),
              'search_ids':frame['application_id'].gather(search).to_list(),'confirmation_ids':frame['application_id'].gather(confirm).to_list(),
              'group_rows':[g.tolist() for g in group],'group_counts':[len(g) for g in group],
              'group_positive_counts':[int(labels[g].sum()) for g in group],
              'outer_rows_not_scored':len(outer)}
    mp=OUT/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:
        raise ValueError('Manifest differs; never silently reuse changed experiment')
    save(mp,manifest); run_hash=sha(mp)
    print(json.dumps({'prepared':True,'columns':len(cols),'groups':manifest['group_counts'],'positive_counts':manifest['group_positive_counts'],'manifest_sha256':run_hash}),flush=True)
    if a.prepare_only:
        return
    cache=OUT/'predictions'; cache.mkdir(exist_ok=True)
    ids_all=frame['application_id'].to_numpy()

    def infer(context,query):
        key=hashlib.sha256((run_hash+context.astype('<i8').tobytes().hex()+query.astype('<i8').tobytes().hex()).encode()).hexdigest()
        ppath=cache/(key+'.parquet'); meta=cache/(key+'.json')
        if ppath.exists() and meta.exists():
            info=json.loads(meta.read_text()); saved=pl.read_parquet(ppath)
            if info['run_hash']!=run_hash or info['prediction_sha256']!=sha(ppath) or saved['application_id'].to_list()!=ids_all[query].tolist():
                raise ValueError('Prediction cache mismatch')
            p=saved['p'].to_numpy()
            if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
                raise ValueError('Invalid cached probabilities')
            return p
        start=time.monotonic()
        # A hung SDK call terminates with traceback; no retry/reduction.
        faulthandler.dump_traceback_later(900,exit=True)
        try:
            p,info=predict(X,context,query,labels,cols,CONFIG['model_seed'],
                           n_estimators=CONFIG['estimators'],chunk=CONFIG['chunk'],memory_fraction=CONFIG['memory_fraction'])
        except KumoInferenceError as e:
            save(OUT/'inference_failure.json',e.metadata); raise
        finally:
            faulthandler.cancel_dump_traceback_later()
        pl.DataFrame({'application_id':ids_all[query],'p':p}).write_parquet(ppath)
        info.update(run_hash=run_hash,prediction_sha256=sha(ppath),elapsed_seconds=time.monotonic()-start)
        save(meta,info)
        print(json.dumps({'inferred':key,'context':len(context),'query':len(query),'seconds':info['elapsed_seconds'],'peak_gib':info['max_allocated_gib']}),flush=True)
        return p

    rng=np.random.default_rng(CONFIG['sampling_seed'])
    k=CONFIG['selected_groups']; n=CONFIG['groups']
    reference=np.sort(rng.choice(n,k,replace=False))
    baseline=infer(membership(group,reference),search)
    baseline_ks=v.ks(yt[s],baseline)
    log=[]; chains=[]
    for chain,tau in enumerate(CONFIG['temperatures']):
        phi=np.full(n,1/n)
        crng=np.random.default_rng(CONFIG['sampling_seed']+100+chain)
        for round_no in range(CONFIG['rounds']):
            probability=(1-CONFIG['uniform_exploration'])*softmax(phi/tau)+CONFIG['uniform_exploration']/n
            coalitions=[]; scores=[]
            for batch in range(CONFIG['batch']):
                chosen=np.sort(crng.choice(n,k,replace=False,p=probability))
                score=v.ks(yt[s],infer(membership(group,chosen),search))
                coalitions.append(chosen); scores.append(score)
                log.append({'arm':'vip','chain':chain,'temperature':tau,'round':round_no,'batch':batch,'groups':chosen.tolist(),'search_ks':score})
            phi=update(phi,coalitions,scores,.5/k)
            save(OUT/'search_trace.json',log)
            print(json.dumps({'chain':chain,'round':round_no,'scores':scores,'phi_spread':float(np.ptp(phi))}),flush=True)
        chosen=np.sort(np.argsort(-phi,kind='stable')[:k])
        chains.append({'temperature':tau,'phi':phi.tolist(),'groups':chosen.tolist(),'estimated_utility':float(np.maximum(phi[chosen],0).sum())})
    selected=max(chains,key=lambda r:r['estimated_utility'])
    # Equal evaluation count controls for ordinary random-search winner selection.
    uniform=[]; urng=np.random.default_rng(CONFIG['sampling_seed']+200)
    for j in range(len(CONFIG['temperatures'])*CONFIG['rounds']*CONFIG['batch']):
        chosen=np.sort(urng.choice(n,k,replace=False))
        score=v.ks(yt[s],infer(membership(group,chosen),search))
        row={'arm':'uniform','draw':j,'groups':chosen.tolist(),'search_ks':score}
        uniform.append(row); log.append(row); save(OUT/'search_trace.json',log)
    random_best=max(uniform,key=lambda r:r['search_ks'])
    lock={'run_hash':run_hash,'chains':chains,'selected_vip':selected,'uniform_search_best':random_best,
          'fixed_random_groups':reference.tolist(),'fixed_random_search_ks':baseline_ks,
          'confirmation_used_for_selection':False,'outer_used':False}
    lp=OUT/'locked_selection.json'
    if lp.exists() and json.loads(lp.read_text())!=lock:
        raise ValueError('Previously locked selection differs')
    save(lp,lock)
    predictions={}; metrics={}
    for name,context in [('fixed_random',membership(group,reference)),('uniform_best',membership(group,random_best['groups'])),
                         ('vip',membership(group,selected['groups'])),('all_real',fit)]:
        p=infer(context,confirm); predictions[name]=p
        metrics[name]={'confirmation_ks':v.ks(yt[c],p),'confirmation_auc':roc_auc_score(yt[c],p),'context_rows':len(context)}
    comparisons={name:paired(yt[c],predictions['vip'],predictions[name],217) for name in ['fixed_random','uniform_best','all_real']}
    result={'run_hash':run_hash,'selection_sha256':sha(lp),'metrics':metrics,'vip_comparisons':comparisons,
            'confirmation_rows':len(confirm),'outer_rows_scored':0,'full_oof':False,
            'decision':'Development evidence only. No automatic promotion or KS0.53 claim; no outer/test predictions.'}
    save(OUT/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
