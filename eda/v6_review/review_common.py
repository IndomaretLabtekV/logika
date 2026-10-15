"""CPU-only audit helpers, reusing existing data/statistics/plotting utilities."""
from pathlib import Path
import sys, os
HERE=Path(__file__).resolve().parent
OUT=HERE/"out"; OUT.mkdir(parents=True,exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR",str(OUT/"mpl_cache"))
sys.path.insert(0,str(HERE.parent/"v5_review"))
import shared as previous
previous.OUT=OUT
from shared import np,pd,pl,plt,csv,js,figsave,digest,data,table,ks,signed_ks,folds,qbins,bh,conditional_test,CATS,IDS,TREES,rankdata,ROOT
from common import psi

def bundle(version):
    f,tr,te=data(); y=tr["target"].to_numpy()
    names=TREES+(["tabpfn_rel","tabm"] if version==5 else ["tabpfn","tabm","tabm_p"])
    w=np.array([4,0,0,1,9,3,0,13] if version==5 else [3,1,0,0,2,1,6,10,7])/30
    folder=ROOT/f"results/v{version}"
    raw={n:np.load(folder/f"oof_{n}.npy") for n in names}
    test={n:np.load(folder/f"test_{n}.npy") for n in names}
    assert all(a.shape==(tr.height,) and np.isfinite(a).all() for a in raw.values())
    assert all(a.shape==(te.height,) and np.isfinite(a).all() for a in test.values())
    R=np.column_stack([rankdata(raw[n])/tr.height for n in names])
    Rt=np.column_stack([rankdata(test[n])/te.height for n in names])
    score=R@w; pred=Rt@w; pred=(pred-pred.min())/np.ptp(pred)
    sub=pl.read_csv(folder/"submission.csv")
    ck=te.select("application_id").with_columns(expected=pl.Series(pred)).join(sub,on="application_id",validate="1:1")
    assert ck.height==te.height and np.allclose(ck["expected"],ck["default_probability"],atol=1e-12)
    expected=.5274 if version==5 else .5283
    assert abs(ks(y,score)-expected)<.00005, "Cached OOF order disagrees with notebook"
    return dict(f=f,tr=tr,te=te,y=y,names=names,w=w,raw=raw,test=test,R=R,Rt=Rt,score=score)

def ks_plan(y,s):
    order=np.argsort(s,kind="stable")
    return order,np.asarray(y)[order],np.r_[np.flatnonzero(np.diff(np.asarray(s)[order])!=0),len(s)-1]

def planned_ks(plan,weights):
    order,y,ends=plan; w=weights[order]
    bad=w*y; good=w*(1-y)
    if bad.sum()==0 or good.sum()==0: return 0.
    return float(np.abs(np.cumsum(bad)[ends]/bad.sum()-np.cumsum(good)[ends]/good.sum()).max())

def checks():
    y=np.array([0,1,0,1,1]);s=np.array([0.,0.,1.,2.,2.]);w=np.array([2,1,0,1,3])
    assert np.isclose(planned_ks(ks_plan(y,s),w),ks(np.repeat(y,w),np.repeat(s,w)))
    assert np.isclose(planned_ks(ks_plan(y,s),np.ones(len(y))),ks(y,s))

def screen(frame,tag):
    b=bundle(6);tr,y,s=b["tr"],b["y"],b["score"]
    X=tr.select("application_id").join(frame,on="application_id",validate="1:1",maintain_order="left")
    cols=[c for c in X.columns if c not in IDS]
    half=folds(y,seed=61014,k=2)==0
    history=tr["has_history"].to_numpy();month=tr["decision_datetime"].dt.month().to_numpy()
    risk=qbins(s,s[half],30)*24+history*12+month-1
    rows=[]
    for c in cols:
        x=X[c].cast(pl.Float64).to_numpy();bins=qbins(x,x[half],5)
        row=dict(feature=c,coverage=float(np.isfinite(x).mean()),observed_univariate_KS=ks(y[np.isfinite(x)],x[np.isfinite(x)]) if np.isfinite(x).sum()>1 else 0.)
        for part,mask in [("discovery",half),("confirmation",~half)]:
            row["p_"+part]=conditional_test(y[mask],bins[mask],risk[mask])[2]
        rows.append(row)
    result=pd.DataFrame(rows)
    for p in ["discovery","confirmation"]:result["q_"+p]=bh(result["p_"+p])
    result["replicates"]=(result.q_discovery<.05)&(result.q_confirmation<.05)
    return csv(result.sort_values(["q_confirmation","p_confirmation"]),tag)
