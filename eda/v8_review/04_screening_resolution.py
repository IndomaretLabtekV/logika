"""Recheck prior candidates after preserving sparse discrete levels and structural zeros."""
from shared8 import *

def main():
    _,tr,_,_,_,_,_,s,_=saved8();y=tr['target'].to_numpy();half=folds(y,seed=81015,k=2)==0
    paths=[ROOT/'eda/v6_review/out/01_candidates.parquet',ROOT/'eda/v7_review/out/02_price_candidates.parquet',OUT/'03_selected_history_features.parquet']
    X=tr.select('application_id')
    for path in paths:
        frame=pl.read_parquet(path);cols=[c for c in frame.columns if c not in IDS]
        assert not set(cols)&(set(X.columns)-{'application_id'})
        X=X.join(frame.select('application_id',*cols),on='application_id',validate='1:1',maintain_order='left')
    result=screen(X,'04_corrected_joint_screen')
    strata=qbins(s,s[half],20)*24+tr['has_history'].to_numpy()*12+tr['decision_datetime'].dt.month().to_numpy()-1
    old=[]
    for c in X.columns[1:]:
        a=X[c].cast(pl.Float64).to_numpy();bins=qbins(a,a[half],5)
        old.append(dict(feature=c,old_p_confirmation=conditional_test(y[~half],bins[~half],strata[~half])[2]))
    z=result.merge(pd.DataFrame(old),on='feature',validate='one_to_one');z['old_q_confirmation']=bh(z.old_p_confirmation)
    csv(z,'04_before_after_screening')
    fixture=np.r_[np.zeros(95),np.ones(5)];oldbins=qbins(fixture,fixture,5);newbins=feature_bins(fixture,fixture)
    assert len(np.unique(oldbins))==1 and len(np.unique(newbins))==2
    csv([dict(value=v,rows=int((fixture==v).sum()),old_bin=int(oldbins[fixture==v][0]),new_bin=int(newbins[fixture==v][0])) for v in [0,1]],'04_bin_fixture')
    fig,ax=plt.subplots(1,2,figsize=(14,8));ix=np.arange(len(z));ax[0].barh(ix-.2,z.old_bin_count,height=.4,label='Quantile-only');ax[0].barh(ix+.2,z.preserved_bin_count,height=.4,label='Preserve discrete/zero');ax[0].set_yticks(ix,z.feature,fontsize=7);ax[0].legend()
    ax[1].scatter(-np.log10(z.old_p_confirmation.clip(lower=1e-15)),-np.log10(z.p_confirmation.clip(lower=1e-15)));ax[1].set(xlabel='Old -log10 confirmation p',ylabel='Corrected -log10 confirmation p',title='Same v8 predictions and split, different bin resolution');figsave(fig,'04_screening_resolution')
    js(dict(candidates=len(z),replicating_after_joint_BH=int(z.replicates.sum()),features_with_more_bins=int((z.preserved_bin_count>z.old_bin_count).sum()),warning='A test unable to distinguish predictor levels is uninformative. Corrected non-significance is not proof of no signal; conditioning is coarse and labels/OOF already used in selection.'),'04_summary')

if __name__=='__main__':main()
