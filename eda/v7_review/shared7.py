"""CPU diagnostics only; reuse the existing exact KS and data helpers."""
from pathlib import Path
import sys,os
HERE=Path(__file__).resolve().parent
OUT=HERE/'out'; OUT.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl_cache'))
sys.path.insert(0,str(HERE.parent/'v6_review'))
import review_common as old
old.previous.OUT=OUT
from review_common import np,pd,pl,plt,csv,js,figsave,data,table,ks,folds,qbins,bh,conditional_test,CATS,IDS,TREES,rankdata,ROOT,ks_plan,planned_ks,digest
import json

def saved7():
    f,tr,te=data()
    p=pl.read_parquet(ROOT/'results/v7/oof_all.parquet')
    assert p.height==tr.height and p['application_id'].n_unique()==tr.height
    p=tr.select('application_id').join(p,on='application_id',validate='1:1',maintain_order='left')
    assert p['target'].equals(tr['target'])
    t=pl.read_parquet(ROOT/'results/v7/test_all.parquet')
    t=te.select('application_id').join(t,on='application_id',validate='1:1',maintain_order='left')
    sub=pl.read_csv(ROOT/'results/v7/submission.csv')
    s=t['blend'].to_numpy(); expected=(s-s.min())/np.ptp(s)
    c=te.select('application_id').with_columns(expected=pl.Series(expected)).join(sub,on='application_id',validate='1:1')
    assert c.height==te.height and np.allclose(c['expected'],c['default_probability'],rtol=0,atol=1e-12)
    return f,tr,te,p,t

def auc(y,s):
    n=y.sum();return float((rankdata(s)[y==1].sum()-n*(n+1)/2)/(n*(len(y)-n)))

def screen(X,tag):
    _,tr,_,p,_=saved7();y=p['target'].to_numpy();s=p['blend'].to_numpy()
    X=tr.select('application_id').join(X,on='application_id',validate='1:1',maintain_order='left')
    h=folds(y,seed=71015,k=2)==0
    strata=qbins(s,s[h],20)*24+tr['has_history'].to_numpy()*12+tr['decision_datetime'].dt.month().to_numpy()-1
    rows=[]
    for name in X.columns[1:]:
        a=X[name].cast(pl.Float64).to_numpy();b=qbins(a,a[h],5);m=np.isfinite(a)
        rows.append(dict(feature=name,coverage=m.mean(),observed_KS=ks(y[m],a[m]),p_discovery=conditional_test(y[h],b[h],strata[h])[2],p_confirmation=conditional_test(y[~h],b[~h],strata[~h])[2]))
    z=pd.DataFrame(rows)
    for name in ['discovery','confirmation']:z['q_'+name]=bh(z['p_'+name])
    z['replicates']=(z.q_discovery<.05)&(z.q_confirmation<.05)
    return csv(z,tag)
