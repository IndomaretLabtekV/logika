"""Same-class, same-category synthetic context from the declared fitting rows."""
import numpy as np
import pandas as pd

import valid2 as v

CATEGORIES=tuple(v.CAT_COLS)


def augment(X,y,ids,fit_rows,columns,cap,seed,prefix,clone=False):
    y,ids,fit=np.asarray(y),np.asarray(ids),np.asarray(fit_rows,dtype=int)
    if len(X)!=len(y) or len(X)!=len(ids) or len(set(ids))!=len(ids):raise ValueError("Unaligned application IDs/targets")
    if not len(fit) or len(set(fit))!=len(fit) or np.any(fit<0) or np.any(fit>=len(X)) or not 1<=cap<=len(fit):
        raise ValueError("Invalid fitting membership or synthetic cap")
    if set(y[fit])!={0,1} or not set(CATEGORIES)<=set(columns) or not set(columns)<=set(X):
        raise ValueError("Expected binary context and complete model columns")
    if "rp_frac_dpd1" not in X:raise ValueError("Missing declared repayment-history indicator")
    numeric=[c for c in columns if c not in CATEGORIES]
    values=X.iloc[fit][numeric].to_numpy(dtype=float)
    if np.isinf(values).any():raise ValueError("Infinite fitting values")
    binary=[c for i,c in enumerate(numeric) if set(np.unique(values[np.isfinite(values[:,i]),i]))<={0,1}]
    rng=np.random.default_rng(seed);yf=y[fit];n1=int(round(cap*yf.mean()))
    anchors=np.r_[rng.choice(np.flatnonzero(yf==0),cap-n1,replace=False),rng.choice(np.flatnonzero(yf==1),n1,replace=False)]
    rng.shuffle(anchors)
    cats=X.iloc[fit][list(CATEGORIES)].astype("string").fillna("__missing__").to_numpy()
    history=X.iloc[fit]["rp_frac_dpd1"].notna().to_numpy()
    keys=[(int(yf[i]),bool(history[i]),*row.tolist()) for i,row in enumerate(cats)]
    groups={}
    for i,key in enumerate(keys):groups.setdefault(key,[]).append(i)
    donors=anchors.copy()
    if not clone:
        for i,a in enumerate(anchors):
            choices=groups[keys[a]]
            if len(choices)>1:
                b=a
                while b==a:b=choices[int(rng.integers(len(choices)))]
                donors[i]=b
    lambdas=np.ones(cap) if clone else rng.beta(.2,.2,cap)
    result=X.iloc[fit[anchors]][columns].reset_index(drop=True).copy()
    if not clone:
        a,b=values[anchors],values[donors]
        mixed=np.where(np.isfinite(a)&np.isfinite(b),lambdas[:,None]*a+(1-lambdas[:,None])*b,a)
        for c in binary:mixed[:,numeric.index(c)]=a[:,numeric.index(c)]
        result[numeric]=mixed
    synthetic=[f"{prefix}_{i:05d}" for i in range(cap)]
    if set(synthetic)&set(ids):raise ValueError("Synthetic IDs collide with official applications")
    parents=pd.DataFrame({"synthetic_id":synthetic,"parent_a":ids[fit[anchors]],"parent_b":ids[fit[donors]],
                          "label":yf[anchors].astype(int),"lambda":lambdas})
    assert np.array_equal(yf[anchors],yf[donors]) and np.array_equal(history[anchors],history[donors])
    assert np.array_equal(result.isna().to_numpy(),X.iloc[fit[anchors]][columns].isna().to_numpy())
    metadata={"fitting_rows":len(fit),"synthetic_rows":cap,"positive_rows":n1,"clone_control":bool(clone),
              "same_parent_rows":int((anchors==donors).sum()),"binary_columns":binary,"numeric_columns":numeric,
              "beta":[.2,.2],"seed":seed,"lambda_min":float(lambdas.min()),"lambda_max":float(lambdas.max()),
              "lambda_mean":float(lambdas.mean()),"query_labels_used":False,"parent_policy":"same class/category/history; fit-only"}
    return result,parents,metadata
