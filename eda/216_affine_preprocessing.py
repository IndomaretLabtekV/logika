"""Optional fit-only affine-redundancy view; preserves missingness and original scale.

This removes only columns reconstructible from an earlier kept feature to
floating-point tolerance, not merely high-correlated features. It is an
experimental feature view, not an assumed accuracy improvement.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import valid2 as v
from common import ROOT


def fit_affine_view(frame, columns, fitting_ids, *, tolerance=1e-10):
    if not 0 < tolerance <= 1e-8:
        raise ValueError('tolerance must be positive and at most 1e-8')
    ids=list(fitting_ids)
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(frame['application_id']):
        raise ValueError('Fitting IDs must be nonempty, unique and present')
    if len(set(columns)) != len(columns) or set(columns)&set(v.ID_COLS):
        raise ValueError('Predictors must be unique and exclude metadata/labels')
    fit=frame.filter(pl.col('application_id').is_in(ids))
    kept=[]; removed={}; mask_groups={}
    for c in columns:
        if not fit[c].dtype.is_numeric():
            kept.append(c); continue
        x=fit[c].to_numpy().astype(float)
        valid=np.isfinite(x)
        key=hashlib.sha256(valid.tobytes()).hexdigest()
        xx=x[valid]
        if len(xx)<3 or np.ptp(xx)==0:
            kept.append(c); continue
        lo,hi=int(np.argmin(xx)),int(np.argmax(xx))
        scale=max(float(np.max(np.abs(xx))),float(np.ptp(xx)),1.)
        for representative,z in mask_groups.get(key,[]):
            if z[hi]==z[lo]:
                continue
            slope=float((xx[hi]-xx[lo])/(z[hi]-z[lo]))
            intercept=float(xx[lo]-slope*z[lo])
            error=float(np.max(np.abs(xx-(slope*z+intercept)))/scale)
            if np.isfinite(slope) and slope != 0 and error <= tolerance:
                removed[c]={'representative':representative,'slope':slope,'intercept':intercept,
                            'fit_relative_max_error':error,'fit_scale':scale}
                break
        else:
            kept.append(c)
            mask_groups.setdefault(key,[]).append((c,xx))
    return {'kept_columns':kept,'removed_columns':removed,'relative_tolerance':tolerance,
            'fitting_ids_sha256':hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),
            'policy':'Optional affine view; fitting-only decisions, same missing-mask, earlier main representative preferred; no imputation, clipping, rank conversion or labels.'}


def transform(frame, recipe):
    meta=[c for c in v.ID_COLS if c in frame.columns]
    return frame.select(meta+recipe['kept_columns'])


def main():
    root=ROOT/'results/eda_23okt/preprocessing'
    out=ROOT/'results/eda_23okt/affine_view'; out.mkdir(parents=True,exist_ok=True)
    frame=pl.read_parquet(root/'features.parquet')
    old=json.loads((root/'preprocessing.json').read_text())
    fit_ids=frame.filter(pl.col('target').is_not_null())['application_id'].to_list()
    recipe=fit_affine_view(frame,old['kept_columns'],fit_ids)
    assert recipe['fitting_ids_sha256']==old['fitting_ids_sha256']
    prepared=transform(frame,recipe)
    assert prepared.height==frame.height and prepared['target'].null_count()==frame['target'].null_count()
    prepared.write_parquet(out/'features.parquet',compression='zstd')
    # Query covariates are inspected AFTER the fitted recipe is frozen, never used to revise it.
    audit=[]; query=frame.filter(pl.col('target').is_null())
    for c,r in recipe['removed_columns'].items():
        x=query[c].to_numpy().astype(float); z=query[r['representative']].to_numpy().astype(float)
        finite=np.isfinite(x)&np.isfinite(z)
        err=np.abs(x[finite]-(r['slope']*z[finite]+r['intercept']))/r['fit_scale']
        audit.append({'feature':c,**r,'query_missing_mask_disagreement':int((np.isfinite(x)!=np.isfinite(z)).sum()),
                      'query_rows_outside_tolerance':int((err>recipe['relative_tolerance']).sum()),
                      'query_max_relative_error':float(err.max()) if len(err) else None})
    (out/'preprocessing.json').write_text(json.dumps(recipe,indent=2,allow_nan=False))
    (out/'query_reconstruction_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    summary={'input_predictors':len(old['kept_columns']),'output_predictors':len(recipe['kept_columns']),
             'removed_affine_redundancies':len(audit),'query_reconstruction_failure_columns':sum(bool(r['query_missing_mask_disagreement'] or r['query_rows_outside_tolerance']) for r in audit),
             'fitting_rows':len(fit_ids),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'input_sha256':hashlib.sha256((root/'features.parquet').read_bytes()).hexdigest(),
             'output_sha256':hashlib.sha256((out/'features.parquet').read_bytes()).hexdigest(),
             'status':'Optional view; no accuracy gain claimed; refit within each future fitting partition.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2),flush=True)
    print(json.dumps(audit[:10],indent=2),flush=True)


if __name__=='__main__':
    main()
