"""Fixed aggregation and error diagnostics, without fitting weights or models."""
from shared7 import *

def cutoff(y,s):
    order=np.argsort(s,kind='stable');a=y[order];v=s[order]
    ends=np.r_[np.flatnonzero(np.diff(v)),len(v)-1]
    d=np.cumsum(1-a)/(1-a).sum()-np.cumsum(a)/a.sum()
    return v[ends[np.argmax(d[ends])]]

def main():
    _,tr,_,p,_=saved7();y=p['target'].to_numpy();names=p.columns[3:]
    raw={n:p[n].to_numpy() for n in names};r={n:rankdata(v)/len(y) for n,v in raw.items()}
    candidates={'selected_v7':p['blend'].to_numpy()}
    for mode in ['probability','rank','logit']:
        values=raw if mode=='probability' else r if mode=='rank' else {n:np.log(np.clip(a,1e-6,1-1e-6)/(1-np.clip(a,1e-6,1-1e-6))) for n,a in raw.items()}
        candidates[mode+'_80trees_20tabm']=.8*np.mean([values[n] for n in TREES],axis=0)+.2*values['tabm']
    z=csv([dict(candidate=n,KS=ks(y,s),AUC=auc(y,s)) for n,s in candidates.items()],'01_fixed_aggregation')
    rows=[];preds={};h=folds(y,seed=71015,k=2)
    for n,s in candidates.items():
        pred=np.zeros(len(y),bool)
        for k in [0,1]:
            fit=h!=k;val=~fit;t=cutoff(y[fit],s[fit]);pred[val]=s[val]>t
            rows.append(dict(candidate=n,half=k,threshold=float(t),sensitivity=float(pred[val & (y==1)].mean()),specificity=float((~pred[val & (y==0)]).mean()),held_KS=ks(y[val],s[val])))
        preds[n]=pred
    csv(rows,'01_cross_half_thresholds')
    # Counterexample: AUC and KS orderings need not agree, using the actual saved models.
    pairs=[]
    for i,n in enumerate(names):
        for m in names[i+1:]:
            da=auc(y,raw[n])-auc(y,raw[m]);dk=ks(y,raw[n])-ks(y,raw[m])
            if da*dk<0:pairs.append(dict(first=n,second=m,AUC_difference=da,KS_difference=dk))
    csv(pairs,'01_auc_ks_disagreements')
    s=candidates['selected_v7'];err=preds['selected_v7']!=y
    b=qbins(s,s[h==0],10);risk=[]
    for j in np.unique(b):
        m=(b==j)&(h==1)
        risk.append(dict(score_bin=int(j),n=int(m.sum()),bad_rate=float(y[m].mean()),error_rate=float(err[m].mean())))
    risk=csv(risk,'01_held_risk_bins')
    sample=tr.select('application_id','target','has_history','pp_mean','bc_score_mean','annual_interest_rate').with_columns(score=pl.Series(s),error=pl.Series(err),prediction=pl.Series(preds['selected_v7'].astype(int)))
    csv(sample.filter(pl.col('error') & pl.Series(h==1)).group_by('target').head(20).to_pandas(),'01_error_examples')
    fig,ax=plt.subplots(1,3,figsize=(15,4));ax[0].bar(z.candidate,z.KS);ax[0].set_ylim(.50,.54);ax[0].tick_params(axis='x',rotation=90)
    for n in ['lgb','tabm','tabm_p','causilo','tabpfn_rel']:
        o=np.argsort(raw[n]);a=y[o];fpr=np.cumsum(1-a)/(1-a).sum();tpr=np.cumsum(a)/a.sum();ax[1].plot(np.arange(len(y))/len(y),fpr-tpr,label=n)
    ax[1].set(xlabel='Score quantile',ylabel='CDF good - CDF bad');ax[1].legend(fontsize=7)
    ax[2].plot(risk.score_bin,risk.bad_rate,'o-',label='bad rate');ax[2].plot(risk.score_bin,risk.error_rate,'o-',label='classification error');ax[2].legend();ax[2].set_title('Other-half threshold, selected OOF');figsave(fig,'01_metric_diagnostics')
    assert cutoff(np.array([0,0,1,1]),np.array([0.,0.,1.,1.]))==0
    js(dict(no_weight_optimization=True,warning='Threshold is cross-half; model/feature/blend selection already used these labels. These are diagnostics, not new untouched validation. Ranking scores are not calibrated probabilities.'),'01_manifest')

if __name__=='__main__':main()
