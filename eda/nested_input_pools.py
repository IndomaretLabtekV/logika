"""Prepare paired Kumo input families without using query labels or vocabularies.

Reference statistics are built separately by166. This adapter verifies their
fitting membership, aligns canonical application IDs, and retains only numeric
features plus the same four application categories for the planned comparison.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import valid2 as v
from teammate_feature_adapter import METADATA_COLUMNS,feature_schema

APPLICATION_CATEGORIES=tuple(v.CAT_COLS)
_vip=None


def teammate_inputs(frame,canonical_ids,y,fit_ids):
    ids=list(canonical_ids);fitting=list(fit_ids);y=np.asarray(y)
    if len(ids)!=len(y) or len(ids)!=len(set(ids)) or len(fitting)!=len(set(fitting)) or not fitting:
        raise ValueError("Canonical/reference IDs must be aligned, unique and nonempty")
    if not set(fitting)<=set(ids):raise ValueError("Reference IDs must belong to canonical training applications")
    if not frame.application_id.is_unique or not frame.customer_id.is_unique:
        raise ValueError("Expected one application per customer")
    if not set(ids)<=set(frame.application_id):raise ValueError("Feature pool lacks canonical query IDs")
    if not set(frame.split.unique())<={"train","test"}:raise ValueError("Unknown fitting membership")
    observed=set(frame.loc[frame.split.eq("train"),"application_id"])
    if observed!=set(fitting):raise ValueError("Feature references were fitted on different IDs")
    if frame.loc[~frame.application_id.isin(fitting),"target"].notna().any():
        raise ValueError("Non-fitting targets must be erased in teammate cache")
    target=dict(zip(ids,y))
    real=frame[frame.application_id.isin(fitting)]
    if not np.array_equal(real.target.to_numpy(),np.array([target[key] for key in real.application_id])):
        raise ValueError("Reference target/ID mismatch")
    schema=feature_schema(frame)
    if not set(APPLICATION_CATEGORIES)<=set(schema["categorical"]):raise ValueError("Missing application categories")
    columns=schema["numeric"]+list(APPLICATION_CATEGORIES)
    return frame.set_index("application_id").loc[ids,columns].reset_index(drop=True).copy()


def select_columns(X,y,fit_rows,n_numeric=100,extend_shortlist=False):
    schema=feature_schema(X)
    if set(schema["categorical"])!=set(APPLICATION_CATEGORIES) or schema["excluded"]:
        raise ValueError("Selector requires numeric pool plus four application categories only")
    if n_numeric<=0 or len(schema["numeric"])<n_numeric:raise ValueError("Insufficient numeric feature pool")
    # Default remains the unchanged baseline. The pre-quality structural amendment
    # doubles its shortlist only if redundant columns prevent filling the budget.
    # Each pass calls the original KS ranking and Spearman filter, then keeps its
    # first n_numeric columns. A baseline that already fills its budget is exact.
    trial=n_numeric
    while True:
        selected=v.fm_select(X,np.asarray(y),np.asarray(fit_rows),n_num=trial)
        numeric=[c for c in selected if c not in APPLICATION_CATEGORIES]
        if len(numeric)>=n_numeric:return numeric[:n_numeric]+list(APPLICATION_CATEGORIES)
        if not extend_shortlist:return selected
        if 2*trial>=len(schema["numeric"]):raise ValueError("Full ranked pool cannot fill the independent numeric budget")
        trial=min(2*trial,len(schema["numeric"]))


def prepare_stage(X,y,ids,real_rows,queries,columns,representation):
    global _vip
    if representation not in {"raw","native"}:raise ValueError("Unknown Kumo representation")
    real=np.asarray(real_rows,dtype=int);ids=np.asarray(ids);y=np.asarray(y)
    if not len(real) or len(X)!=len(ids) or len(X)!=len(y) or len(set(ids))!=len(ids):
        raise ValueError("Unaligned/empty stage")
    if not queries or any(not len(rows) for rows in queries.values()):raise ValueError("Nonempty queries required")
    parts=[real,*[np.asarray(rows,dtype=int) for rows in queries.values()]]
    all_rows=np.concatenate(parts)
    if len(np.unique(all_rows))!=len(all_rows) or np.any(all_rows<0) or np.any(all_rows>=len(X)):
        raise ValueError("Stage context/query overlap, duplication or invalid row")
    if len(set(y[real]))!=2:raise ValueError("Context must contain both classes")
    if len(columns)!=len(set(columns)) or set(columns)&METADATA_COLUMNS or not set(columns)<=set(X.columns):
        raise ValueError("Invalid model columns")
    F=pd.concat([X.iloc[rows][columns] for rows in parts],ignore_index=True)
    if _vip is None:
        spec=importlib.util.spec_from_file_location("nested_vip_prepare",Path(__file__).with_name("152_vipcop_kumo.py"))
        _vip=importlib.util.module_from_spec(spec);spec.loader.exec_module(_vip)
    F=_vip.prepare(F,np.arange(len(real)),"raw" if representation=="raw" else "categorical")
    labels=np.zeros(len(F),dtype=int);labels[:len(real)]=y[real]
    positions={};start=len(real)
    for name,rows in queries.items():
        positions[name]=np.arange(start,start+len(rows));start+=len(rows)
    return {"frame":F,"labels":labels,"real":np.arange(len(real)),"queries":positions,"ids":ids[all_rows]}
