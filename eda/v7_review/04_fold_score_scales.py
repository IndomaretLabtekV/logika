"""Distinguish fold score scales from optimism caused by maximizing KS per fold."""
from shared7 import *

def notebook_folds(y):
    # Binary StratifiedKFold allocation, RandomState shuffle as in the notebook's sklearn splitter.
    _,first,inv=np.unique(y,return_index=True,return_inverse=True)
    _,encoding=np.unique(first,return_inverse=True);encoded=encoding[inv]
    ordered=np.sort(encoded);allocation=np.array([np.bincount(ordered[i::5],minlength=2) for i in range(5)])
    rng=np.random.RandomState(42);out=np.empty(len(y),int)
    for k in range(2):
        f=np.arange(5).repeat(allocation[:,k]);rng.shuffle(f);out[encoded==k]=f
    return out

def main():
    _,tr,_,p,_=saved7();y=p['target'].to_numpy();fd=notebook_folds(y)
    check=[ks(y[fd==i],p['tabpfn'].to_numpy()[fd==i]) for i in range(5)]
    assert np.allclose(check,[.5195,.5307,.5345,.5207,.5102],rtol=0,atol=.00005),check
    rows=[];summ=[];rng=np.random.default_rng(741015)
    for name in ['tabpfn','tabpfn_rel','causilo','blend']:
        s=p[name].to_numpy();normalized=np.zeros(len(y));foldks=[]
        for k in range(5):
            m=fd==k;foldks.append(ks(y[m],s[m]));normalized[m]=rankdata(s[m])/m.sum()
            rows.append(dict(model=name,fold=k,KS=foldks[-1],mean_score=s[m].mean(),p10=np.quantile(s[m],.1),p90=np.quantile(s[m],.9)))
        random_gaps=[]
        for repeat in range(80):
            q=rng.permutation(len(y))%5
            random_gaps.append(np.mean([ks(y[q==k],s[q==k]) for k in range(5)])-ks(y,s))
        summ.append(dict(model=name,pooled_KS=ks(y,s),mean_fold_KS=np.mean(foldks),actual_fold_gap=np.mean(foldks)-ks(y,s),random_partition_gap_mean=np.mean(random_gaps),random_partition_gap_p95=np.quantile(random_gaps,.95),fold_rank_normalized_KS=ks(y,normalized)))
    detail=csv(rows,'04_fold_scores');summary=csv(summ,'04_fold_summary')
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for name,g in detail.groupby('model'):ax[0].plot(g.fold,g.mean_score,'o-',label=name)
    ax[0].legend();ax[0].set(xlabel='Seed-42 fold',ylabel='Mean saved score')
    ax[1].bar(np.arange(len(summary))-.2,summary.actual_fold_gap,width=.4,label='Actual fold gap');ax[1].bar(np.arange(len(summary))+.2,summary.random_partition_gap_mean,width=.4,label='Random partition mean gap');ax[1].set_xticks(range(len(summary)),summary.model);ax[1].legend();ax[1].set_ylabel('Mean subset KS minus pooled KS');figsave(fig,'04_fold_scales')
    js(dict(seed42_fold_reconstruction_matches_notebook=True,warning='Fold-normalized scores are a diagnostic only. Saved test predictions are already averaged across folds; cannot reproduce matching test transformation. Blend averages multiple seeds, so seed-42 groups are not pure model folds for the blend.'),'04_manifest')

if __name__=='__main__':main()
