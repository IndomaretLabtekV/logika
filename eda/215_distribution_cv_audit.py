"""CPU-only distribution/CV and conditional-signal diagnostics; no model fitting.

Quantiles and feature shortlists use inner fitting rows only. Outer labels are
used only for split prevalence audit, never feature/context selection.
"""
import json
from pathlib import Path
import hashlib

import numpy as np
import polars as pl
from scipy.stats import ks_2samp, rankdata
import valid2 as v
from common import ROOT

OUT = ROOT / 'results/eda_23okt/deeper_audit'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def bins(x, edges):
    return np.where(np.isfinite(x), np.searchsorted(edges, x, side='right'), len(edges)+1)


def frequencies(x, edges):
    counts = np.bincount(bins(x, edges), minlength=len(edges)+2).astype(float)
    return (counts + .5) / (len(x) + .5*len(counts))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = ROOT/'results/eda_23okt/preprocessing'
    frame = pl.read_parquet(src/'features.parquet')
    recipe = json.loads((src/'preprocessing.json').read_text())
    train = pl.read_csv(ROOT/'data/train_applications.csv').sort('application_id')
    test = pl.read_csv(ROOT/'data/test_applications.csv').sort('application_id')
    ids = train['application_id'].to_numpy()
    y = train['target'].to_numpy()
    tr, outer = v.outer(y, 42)[0]
    fi, st = v.inner(y[tr], 42, 0)
    lookup = {x:i for i,x in enumerate(frame['application_id'])}
    fitting = np.array([lookup[x] for x in ids[tr[fi]]])
    testing = np.flatnonzero(frame['is_train'].to_numpy() == 0)
    assert hashlib.sha256('\n'.join(sorted(ids[tr[fi]])).encode()).hexdigest() == recipe['fitting_ids_sha256']
    monthly = []
    for name, data in [('train',train),('test',test)]:
        for month, g in data.with_columns(pl.col('decision_datetime').str.slice(0,7).alias('month')).group_by('month'):
            row = {'split':name, 'month':month[0], 'rows':g.height,
                   'fraction':g.height/data.height, 'positive_rate':None,
                   'wilson_low':None, 'wilson_high':None}
            if name == 'train':
                n = g.height; p = g['target'].mean(); z = 1.959963984540054
                center = (p+z*z/(2*n))/(1+z*z/n)
                radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
                row.update(positive_rate=p,wilson_low=center-radius,wilson_high=center+radius)
            monthly.append(row)
    pl.DataFrame(monthly).sort('month','split').write_csv(OUT/'monthly_population.csv')
    fold_rows = []
    for fold,(fit,val) in enumerate(v.outer(y,42)):
        fit_dates = train['decision_datetime'].to_numpy()[fit]
        val_dates = train['decision_datetime'].to_numpy()[val]
        fold_rows.append({'fold':fold,'fit_rows':len(fit),'val_rows':len(val),
                          'fit_positive_rate':float(y[fit].mean()),'val_positive_rate':float(y[val].mean()),
                          'fit_min':str(fit_dates.min()),'fit_max':str(fit_dates.max()),
                          'val_min':str(val_dates.min()),'val_max':str(val_dates.max()),
                          'customer_overlap':len(set(train['customer_id'].to_numpy()[fit]) & set(train['customer_id'].to_numpy()[val]))})
    pl.DataFrame(fold_rows).write_csv(OUT/'outer_partition_audit.csv')
    drift=[]; category=[]; arrays={}
    for c in recipe['kept_columns']:
        if c in recipe['categories']:
            a=frame[c].gather(fitting).fill_null('__MISSING__').to_list()
            b=frame[c].gather(testing).fill_null('__MISSING__').to_list()
            from collections import Counter
            ca,cb=Counter(a),Counter(b)
            category.append({'feature':c,'total_variation':sum(abs(ca[k]/len(a)-cb[k]/len(b)) for k in ca.keys()|cb.keys())/2,
                             'test_unknown_fraction':sum(n for k,n in cb.items() if k not in ca)/len(b)})
            continue
        x=frame[c].to_numpy().astype(float); a=x[fitting]; b=x[testing]
        aa=a[np.isfinite(a)]; bb=b[np.isfinite(b)]
        edges=np.unique(np.quantile(aa,np.arange(.1,1,.1)))
        pa,pb=frequencies(a,edges),frequencies(b,edges)
        q001,q999=np.quantile(aa,[.001,.999])
        drift.append({'feature':c,'psi_fit_bins_missing_separate':float(np.sum((pb-pa)*np.log(pb/pa))),
                      'observed_ks_fit_test':float(ks_2samp(aa,bb).statistic) if len(bb) else None,
                      'missing_fit':float((~np.isfinite(a)).mean()),'missing_test':float((~np.isfinite(b)).mean()),
                      'test_beyond_fit_range':float(((bb<aa.min())|(bb>aa.max())).mean()) if len(bb) else None,
                      'fit_q001':float(q001),'fit_q999':float(q999),
                      'test_outside_fit_q001_q999':float(((bb<q001)|(bb>q999)).mean()) if len(bb) else None})
        arrays[c]=a
    dt=pl.DataFrame(drift).sort('psi_fit_bins_missing_separate',descending=True)
    dt.write_csv(OUT/'numeric_distribution_shift.csv')
    pl.DataFrame(category).write_csv(OUT/'categorical_distribution_shift.csv')
    diagnostics=pl.read_csv(src/'feature_diagnostics.csv')
    # Exploratory shortlist fixed from fitting-only marginal evidence, not confirmation.
    shortlist=diagnostics.filter((pl.col('family')=='prefix') & (pl.col('max_abs_spearman_to_main_sample')<.95) & pl.col('feature').is_in(recipe['kept_columns'])).head(20)
    yf=y[tr[fi]].astype(float)
    conditional=[]; subgroup=[]
    for r in shortlist.iter_rows(named=True):
        c,base=r['feature'],r['closest_main']
        x,z=arrays[c],arrays[base]
        ze=np.unique(np.quantile(z[np.isfinite(z)],np.arange(.1,1,.1)))
        group=bins(z,ze)
        # Missingness receives its own stratum: do not confuse absent history with zero DPD.
        rx=rankdata(np.where(np.isfinite(x),x,r['median'])).astype(float)
        ry=yf.copy()
        for g in np.unique(group):
            m=group==g; rx[m]-=rx[m].mean(); ry[m]-=ry[m].mean()
        denominator=np.linalg.norm(rx)*np.linalg.norm(ry)
        conditional.append({'feature':c,'condition_on':base,'conditioning':'fitting deciles plus separate missing stratum',
                            'within_stratum_rank_label_correlation':float(rx@ry/denominator) if denominator else 0.,
                            'fit_marginal_ks':r['fit_ks_median_imputed'],
                            'warning':'Exploratory fitting association; not conditional independence, model gain, or unbiased validation.'})
        for state in [False,True]:
            m=np.isfinite(x)==state
            if m.any():
                subgroup.append({'feature':c,'observed':state,'rows':int(m.sum()),'positive_rate':float(yf[m].mean())})
    pl.DataFrame(conditional).write_csv(OUT/'conditional_signal_exploratory.csv')
    pl.DataFrame(subgroup).write_csv(OUT/'missingness_risk_fit.csv')
    summary={'mode':'CPU-only EDA; no model training or inference','fitting_rows':len(fitting),
             'numeric_features':len(drift),'categorical_features':len(category),
             'max_numeric_psi':dt['psi_fit_bins_missing_separate'].max(),
             'numeric_psi_gt_01':dt.filter(pl.col('psi_fit_bins_missing_separate')>.1).height,
             'max_observed_distribution_ks':dt['observed_ks_fit_test'].max(),
             'train_time_range':[train['decision_datetime'].min(),train['decision_datetime'].max()],
             'test_time_range':[test['decision_datetime'].min(),test['decision_datetime'].max()],
             'train_customer_duplicates':train.height-train['customer_id'].n_unique(),
             'test_customer_duplicates':test.height-test['customer_id'].n_unique(),
             'train_test_customer_overlap':len(set(train['customer_id'])&set(test['customer_id'])),
             'source_sha256':digest(__file__),'prepared_features_sha256':digest(src/'features.parquet'),
             'limits':['No multivariate adversarial classifier fitted.','Small marginal shifts do not prove exchangeability or label stability.',
                       'Monthly prevalence is descriptive, not a selector objective.','Label maturation timestamps absent from application schema; cannot certify production-time label availability.',
                       'Conditional associations assessed only on fitting rows; no new confirmation/outer feature selection.','No KS0.53 claim.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2),flush=True)
    print(dt.head(5),flush=True)
    print(pl.DataFrame(conditional).select('feature','within_stratum_rank_label_correlation'),flush=True)


if __name__=='__main__':
    main()
