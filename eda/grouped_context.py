"""Fit-only covariate geometry for shared-neighborhood context retrieval."""
import hashlib
import json

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder,QuantileTransformer

import valid2 as v


def neighborhoods(X,fit_rows,query_rows,columns,seed,n_clusters=8,n_context=12000):
    fit=np.asarray(fit_rows,dtype=int);query=np.asarray(query_rows,dtype=int)
    joined=np.r_[fit,query]
    if not len(fit) or not len(query) or len(set(joined))!=len(joined) or np.any(joined<0) or np.any(joined>=len(X)):
        raise ValueError("Context/query must be nonempty, unique, disjoint and in bounds")
    if not 1<=n_clusters<=len(fit) or not 1<=n_context<=len(fit):raise ValueError("Invalid context budget")
    numeric=[c for c in columns if c not in v.CAT_COLS][:20]
    if not numeric or not set(v.CAT_COLS)<=set(columns):raise ValueError("Expected numeric and four application categories")
    qt=QuantileTransformer(n_quantiles=min(256,len(fit)),output_distribution="uniform",subsample=None,random_state=seed)
    f=X.iloc[fit][numeric].to_numpy(dtype=float);q=X.iloc[query][numeric].to_numpy(dtype=float)
    if np.isinf(f).any() or np.isinf(q).any():raise ValueError("Infinite retrieval coordinates")
    numeric_fit=qt.fit_transform(f);numeric_query=qt.transform(q)
    encode=OneHotEncoder(handle_unknown="ignore",sparse_output=False,dtype=np.float32)
    cats_fit=X.iloc[fit][v.CAT_COLS].astype("string").fillna("__missing__")
    cats_query=X.iloc[query][v.CAT_COLS].astype("string").fillna("__missing__")
    cf=encode.fit_transform(cats_fit);cq=encode.transform(cats_query)
    Zf=np.c_[np.nan_to_num(numeric_fit,nan=.5),.5*np.isnan(f),.5*cf].astype(np.float32)
    Zq=np.c_[np.nan_to_num(numeric_query,nan=.5),.5*np.isnan(q),.5*cq].astype(np.float32)
    cluster=MiniBatchKMeans(n_clusters=n_clusters,n_init=3,batch_size=2048,max_iter=100,random_state=seed)
    cluster.fit(Zf);assignment=cluster.predict(Zq)
    contexts={}
    for key,center in enumerate(cluster.cluster_centers_):
        distance=np.square(Zf-center).sum(axis=1)
        nearest=np.lexsort((fit,distance))[:n_context]
        contexts[key]=np.sort(fit[nearest])
    state={"numeric_columns":numeric,"category_vocabulary":[values.tolist() for values in encode.categories_],
           "quantiles":qt.quantiles_.tolist(),"references":qt.references_.tolist(),"centers":cluster.cluster_centers_.tolist(),
           "seed":seed,"n_clusters":n_clusters,"n_context":n_context,"missing_scale":.5,"onehot_scale":.5}
    fingerprint=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    unknown={c:int((~cats_query[c].isin(encode.categories_[i])).sum()) for i,c in enumerate(v.CAT_COLS)}
    return {"contexts":contexts,"assignment":assignment,"centers":cluster.cluster_centers_,
            "geometry_sha256":fingerprint,"geometry":state,"unknown_query_categories":unknown,
            "fit_cluster_counts":np.bincount(cluster.predict(Zf),minlength=n_clusters).tolist(),
            "query_cluster_counts":np.bincount(assignment,minlength=n_clusters).tolist()}
