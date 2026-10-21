"""Fit-only LightGBM gain ranking and lossless input-family union."""
import hashlib

import numpy as np
import pandas as pd

import valid2 as v
from teammate_feature_adapter import feature_schema


def union_inputs(main,friend):
    if len(main)!=len(friend) or not main.index.equals(friend.index):raise ValueError("Unaligned input pools")
    for c in v.CAT_COLS:
        if not main[c].astype("string").equals(friend[c].astype("string")):raise ValueError(f"Category/ID alignment mismatch: {c}")
    schema=feature_schema(friend)
    if schema["excluded"] or set(schema["categorical"])!=set(v.CAT_COLS):raise ValueError("Unexpected union predictors")
    extra=friend[schema["numeric"]].rename(columns=lambda name:"teman__"+name)
    if set(extra)&set(main):raise ValueError("Union prefix collision")
    return pd.concat([main,extra],axis=1)


def gain_columns(X,y,fit_rows,seed,n_numeric=100):
    import lightgbm as lgb
    y=np.asarray(y);rows=np.asarray(fit_rows,dtype=int)
    if len(X)!=len(y) or not len(rows) or len(set(rows))!=len(rows) or np.any(rows<0) or np.any(rows>=len(y)):
        raise ValueError("Invalid or unaligned fitting rows")
    if set(y[rows])!={0,1}:raise ValueError("Context needs two binary classes")
    schema=feature_schema(X);nums=schema["numeric"]
    if schema["excluded"] or set(schema["categorical"])!=set(v.CAT_COLS):raise ValueError("Unexpected selector predictors")
    if len(nums)<n_numeric or n_numeric<1:raise ValueError("Insufficient numeric pool")
    params={**v.LGB,"num_threads":4,"deterministic":True,"force_col_wise":True,
            "seed":seed,"feature_fraction_seed":seed,"bagging_seed":seed,"data_random_seed":seed}
    params.pop("n_jobs",None)
    sub=X.iloc[rows][nums]
    model=lgb.train(params,lgb.Dataset(sub,label=y[rows]),num_boost_round=400)
    gain=pd.Series(model.feature_importance(importance_type="gain"),index=nums,name="gain")
    ranked=sorted(nums,key=lambda c:-gain[c])
    shortlist=min(2*n_numeric,len(nums))
    while True:
        candidate=ranked[:shortlist];corr=sub[candidate].rank().corr().abs().fillna(0.)
        keep=[]
        for c in candidate:
            if all(corr.loc[c,k]<.95 for k in keep):keep.append(c)
            if len(keep)==n_numeric:break
        if len(keep)==n_numeric:break
        if shortlist==len(nums):raise ValueError("Full conditional ranking cannot fill numeric budget")
        shortlist=min(2*shortlist,len(nums))
    metadata={"params":params,"rounds":400,"fitting_rows":len(rows),"numeric_pool":len(nums),"shortlist":shortlist,
              "selected_numeric":len(keep),"ranking":"LightGBM gain; stable ties in source column order",
              "booster_sha256":hashlib.sha256(model.model_to_string().encode()).hexdigest(),"query_labels_used":False}
    return keep+v.CAT_COLS,gain,metadata
