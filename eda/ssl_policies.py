"""Deterministic pseudo-context selection; no targets or model fitting.

Asymmetric confidence is an experimental class-dependent threshold policy. It is
not FlexMatch/FreeMatch: those methods also adapt thresholds during training.
"""
import numpy as np


def policy_name(policy):
    kind=policy["kind"]
    if kind=="none":return "noSSL"
    if kind=="confidence":stem=f"confidence_{policy['threshold']:g}"
    elif kind=="asymmetric":stem=f"asym_n{policy['negative_threshold']:g}_p{policy['positive_threshold']:g}"
    elif kind=="groups":stem="groups_"+"-".join(map(str,policy["groups"]))
    else:raise ValueError(f"Unknown pseudo policy: {kind}")
    return f"{stem}_cap{policy['cap']}"


def group_ids(p):
    p=np.asarray(p)
    return np.select([p<=.02,(p>.02)&(p<=.05),(p>=.95)&(p<.98),p>=.98],
                     [0,1,2,3],default=-1).astype(int)


def pseudo_rows(p,policy,seed):
    p=np.asarray(p,dtype=float)
    if p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("Teacher probabilities must be a finite vector in [0,1]")
    kind=policy["kind"]
    if kind=="none":return np.empty(0,dtype=int)
    cap=policy["cap"]
    if not isinstance(cap,int) or cap<=0:raise ValueError("Pseudo cap must be a positive integer")
    if kind=="confidence":
        threshold=policy["threshold"]
        if not 0<=threshold<.5:raise ValueError("Symmetric confidence threshold must be below 0.5")
        mask=(p<=threshold)|(p>=1-threshold)
    elif kind=="asymmetric":
        negative,positive=policy["negative_threshold"],policy["positive_threshold"]
        if not 0<=negative<.5<=positive<=1:raise ValueError("Class thresholds must be separated at 0.5")
        mask=(p<=negative)|(p>=positive)
    elif kind=="groups":
        if not set(policy["groups"])<=set(range(4)):raise ValueError("Unknown confidence shell")
        mask=np.isin(group_ids(p),policy["groups"])
    else:raise ValueError(f"Unknown pseudo policy: {kind}")
    eligible=np.flatnonzero(mask)
    if len(eligible)>cap:eligible=np.random.default_rng(seed).choice(eligible,cap,replace=False)
    return np.sort(eligible)
