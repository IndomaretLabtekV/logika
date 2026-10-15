"""CPU-only diagnostics; no model packages or notebook execution."""
from pathlib import Path
import sys,os
HERE=Path(__file__).resolve().parent;OUT=HERE/'out';OUT.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl_cache'))
sys.path.insert(0,str(HERE.parent/'v7_review'))
import shared7 as prior
prior.old.previous.OUT=OUT
from shared7 import np,pd,pl,plt,csv,js,figsave,data,table,ks,folds,qbins,bh,conditional_test,CATS,IDS,TREES,rankdata,ROOT,ks_plan,planned_ks,digest,auc,json

def saved8():
    f,_,_=data();folder=ROOT/'results/v8'
    tr=pl.DataFrame({'application_id':np.load(folder/'train_ids.npy')}).join(f,on='application_id',validate='1:1',maintain_order='left')
    te=pl.DataFrame({'application_id':np.load(folder/'test_ids.npy')}).join(f,on='application_id',validate='1:1',maintain_order='left')
    assert tr.height==70000 and te.height==30000 and tr['target'].null_count()==0
    names=sorted(p.stem[4:] for p in folder.glob('oof_*.npy'))
    raw={n:np.load(folder/f'oof_{n}.npy') for n in names};test={n:np.load(folder/f'test_{n}.npy') for n in names}
    assert all(x.shape==(tr.height,) and np.isfinite(x).all() for x in raw.values())
    R=np.column_stack([rankdata(raw[n])/tr.height for n in names]);Rt=np.column_stack([rankdata(test[n])/te.height for n in names])
    sub=te.select('application_id').join(pl.read_csv(folder/'submission.csv'),on='application_id',validate='1:1',maintain_order='left')['default_probability'].to_numpy()
    # Algebraic provenance check against the saved CSV, not a predictor fit or target-driven weight search.
    A=np.column_stack([np.ones(te.height),Rt]);coef,_,rank,_=np.linalg.lstsq(A,sub,rcond=None)
    residual=float(abs(A@coef-sub).max());assert residual<1e-10 and rank==A.shape[1]
    w=coef[1:]/coef[1:].sum();assert w.min()>-1e-10 and abs(w.sum()-1)<1e-12
    score=R@w
    return f,tr,te,names,raw,test,w,score,residual

def hist():return pl.read_parquet(ROOT/'data_pq/hist_labels.parquet')

def feature_bins(x,reference):
    values=np.unique(reference[np.isfinite(reference)])
    if len(values)<=12:
        return np.where(np.isfinite(x),np.searchsorted(values,x),len(values)+1)
    # A dominant zero must not share the only quantile bin with the rare positive tail.
    return np.where(x==0,0,1+qbins(x,reference[np.isfinite(reference)&(reference!=0)],5))

def screen(X,tag):
    _,tr,_,_,_,_,_,s,_=saved8();y=tr['target'].to_numpy();h=folds(y,seed=81015,k=2)==0
    frame=tr.select('application_id').join(X,on='application_id',validate='1:1',maintain_order='left')
    strata=qbins(s,s[h],20)*24+tr['has_history'].to_numpy()*12+tr['decision_datetime'].dt.month().to_numpy()-1;rows=[]
    for c in frame.columns[1:]:
        a=frame[c].cast(pl.Float64).to_numpy();b=feature_bins(a,a[h]);m=np.isfinite(a)
        row=dict(feature=c,coverage=float(m.mean()),observed_KS=ks(y[m],a[m]),old_bin_count=int(len(np.unique(qbins(a,a[h],5)))),preserved_bin_count=int(len(np.unique(b))))
        for name,idx in [('discovery',h),('confirmation',~h)]:row['p_'+name]=conditional_test(y[idx],b[idx],strata[idx])[2]
        rows.append(row)
    z=pd.DataFrame(rows)
    for n in ['discovery','confirmation']:z['q_'+n]=bh(z['p_'+n])
    z['replicates']=(z.q_discovery<.05)&(z.q_confirmation<.05)
    check=feature_bins(np.array([0.,1.,np.nan]),np.r_[np.zeros(100),1.])
    assert len(np.unique(check))==3
    return csv(z,tag)
