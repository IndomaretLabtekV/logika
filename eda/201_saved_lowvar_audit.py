"""Audit the pre-existing October18 low-variance recipe, without model fitting.

This is a deployment analogue: Kumo OOF uses56k contexts, test uses70k.
No weights are selected, no public labels are available, and no submission API
is called. Reconstruct both historical and candidate files from their sources.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata

import valid2 as v
from common import ROOT,OUT,PQ
from screen2 import boot

TREES=('lgb','lgb_s','xgb','xgb_s','cat','cat_s')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def rk(p):return rankdata(p,method='average')/len(p)


def main():
    r=ROOT/'results';fingerprints={}
    canonical=pl.read_parquet(PQ/'features.parquet',columns=['application_id','is_train','target'])
    train=canonical.filter(pl.col('is_train')==1).sort('application_id');y=train['target'].to_numpy().astype(int)
    test=pl.read_csv(ROOT/'data/sample_submission.csv');assert train.height==70000 and test.height==30000
    def load(relative,ids,target=None):
        path=r/relative;fingerprints[str(path.relative_to(ROOT))]=sha(path);d=pl.read_parquet(path)
        assert d.height==ids.height and d['application_id'].n_unique()==d.height
        assert set(d['application_id'])==set(ids['application_id'])
        d=ids.select('application_id').join(d,on='application_id',how='left',maintain_order='left',validate='1:1')
        if target is not None and 'target' in d.columns:assert np.array_equal(d['target'].to_numpy(),target)
        cols=[c for c in d.columns if c not in ('application_id','target')];assert len(cols)==1
        p=d[cols[0]].to_numpy();assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        return p
    def oof(relative):return load(relative,train,y)
    def te(relative):return load(relative,test)
    tree=np.mean([rk(oof(f'v22/v22/gbdt/oof_{m}.parquet')) for m in TREES],axis=0)
    c1=oof('v22/v22/fm/oof_causilo.parquet');k1=oof('v28m/oof_kumo_large_c56000.parquet')
    cs=[c1]+[oof(f'screen/seeds/s{s}/oof_causilo.parquet') for s in (1,2,7,2026)]
    k3=(k1+2*oof('v38/oof_kumo_large_c56000.parquet'))/3
    # Exactly the already written129 policy, not a weight search.
    candidate=(rk(tree)+rk(np.mean(cs,axis=0))+rk(k3))/3
    reference=(tree+rk(c1)+rk(k1))/3
    trees_test=np.mean([rk(te(f'v22/v22/gbdt/test_{m}.parquet')) for m in TREES],axis=0)
    ct=[(1,te('v22/v22/fm/test_causilo.parquet')),(2,te('v37/test_causilo.parquet')),(2,te('v41/causilo_s12/test_causilo.parquet'))]
    kt=[te(p) for p in ('v34/test_kumo_full70k.parquet','v41/test_kumo_full70k_g1.parquet','v41/test_kumo_full70k_g2.parquet')]
    prediction=(rk(trees_test)+rk(sum(w*p for w,p in ct)/5)+rk(np.mean(kt,axis=0)))/3
    legacy=pl.read_csv(r/'v41/v31_lowvar.csv');legacy=test.select('application_id').join(legacy,on='application_id',how='left',maintain_order='left')
    assert np.allclose(prediction,legacy['default_probability'].to_numpy(),atol=1e-15,rtol=0)
    historical=(trees_test+rk(ct[0][1])+rk(te('v28m/test_kumo_large_c56000.parquet')))/3
    submitted=pl.read_csv(r/'v31/submission.csv');submitted=test.select('application_id').join(submitted,on='application_id',how='left',maintain_order='left')
    historical_error=float(abs(historical-submitted['default_probability'].to_numpy()).max())
    assert historical_error<1e-7,'Wrong historical recipe constituent'
    mean,sd,p=boot(y,candidate,reference,n=1000)
    folds=[v.ks(y[va],candidate[va])-v.ks(y[va],reference[va]) for _,va in v.outer(y,42)]
    # Public/private have9k/21k rows. This conditional finite-population proxy
    # measures perturbation uncertainty, not an exact forecast distribution.
    rng=np.random.default_rng(201);pub=[];private=[]
    for _ in range(1000):
        order=rng.permutation(len(y));i=order[:9000];j=order[9000:30000]
        pub.append(v.ks(y[i],candidate[i])-v.ks(y[i],reference[i]))
        private.append(v.ks(y[j],candidate[j])-v.ks(y[j],reference[j]))
    pl.DataFrame({'application_id':train['application_id'],'target':y,'candidate':candidate,'v31_reference':reference}).write_parquet(OUT/'201_saved_lowvar_oof.parquet')
    result={'candidate_policy':'October18 eda129: rank trees1/3, Causilo5seeds1/3, Kumo3draws1/3',
      'candidate_ks':v.ks(y,candidate),'v31_reference_ks':v.ks(y,reference),'delta':v.ks(y,candidate)-v.ks(y,reference),
      'bootstrap_delta_mean':float(mean),'bootstrap_sd':float(sd),'p_gt_zero':float(p),'fold_deltas':folds,'fold_wins':sum(d>0 for d in folds),
      'public_subset_delta_mean':float(np.mean(pub)),'public_subset_delta_sd':float(np.std(pub)),
      'private_subset_delta_mean':float(np.mean(private)),'private_subset_delta_sd':float(np.std(private)),
      'public_forecast_mean':.56405+float(np.mean(pub)),'public_forecast_sd':float(np.hypot(np.std(pub),.0013)),
      'historical_test_reconstruction_max_error':historical_error,'candidate_test_reproduced':True,
      'source_sha256':sha(__file__),'inputs_sha256':fingerprints,'candidate_csv_sha256':sha(r/'v41/v31_lowvar.csv'),
      'breakthrough':False,'submission_performed':False,
      'limits':['Historical aggregated Kumo OOF proves seed average, not individual per-seed provenance.',
                'Kumo OOF56k and deployment70k differ; not a matched deployment experiment.',
                'v41 test draws use estimator offsets, OOF average uses outer seeds; analogue only.',
                'Legacy feature reference has limitations documented in validation_v2/integration audit.',
                '1,000 paired bootstrap samples customers after seed averaging; exploratory research selection is not corrected.',
                'Public forecast anchored to historical v31; no guarantee of improvement or0.570.']}
    (OUT/'201_saved_lowvar_audit.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(figsize=(8,4));ax.hist(pub,bins=45,alpha=.6,label='9k public proxy');ax.hist(private,bins=45,alpha=.6,label='21k private proxy');ax.axvline(0,color='black');ax.legend();ax.set(xlabel='Paired KS delta versus v31',title='Existing low-variance candidate; no new model fit');fig.tight_layout();fig.savefig(OUT/'201_saved_lowvar_audit.png',dpi=130);plt.close(fig)
    print(json.dumps({k:val for k,val in result.items() if k!='inputs_sha256'},indent=2),flush=True)


if __name__=='__main__':main()
