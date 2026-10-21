"""Run the original teammate feature functions with isolated, nested references.

The numerical formulas remain in scripts/build_features_v3/v4/v5.py. Only their
artifact roots and the train/reference membership are supplied here. Query
targets are erased before reference-fitting stages are called.
"""
import ast
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REFERENCE_SOURCES=("build_features_v3.py","build_features_v4.py","build_features_v5.py")
RAW_TABLES=("train_applications","test_applications","customer_profiles","previous_applications",
            "repayment_history","account_behavior","bureau_observations","ecommerce_transactions","sample_submission")
METADATA_COLUMNS={"application_id","customer_id","target","split","decision_date","application_datetime","decision_datetime"}
CATEGORICAL_COLUMNS={"loan_purpose","region","employment_type","housing_status",
                     "v2_prev_last_decision","v2_ecom_last_payment_method"}


def feature_schema(frame):
    """Classify source predictors without converting strings or fitting vocabulary.

    The full pool contains two history categories beyond the baseline's four.
    Unexpected text/date predictors fail explicitly rather than become numeric
    codes or accidentally expose timestamp/identifier columns to selection.
    """
    if not frame.columns.is_unique:raise ValueError("Duplicate feature columns")
    schema={"numeric":[],"categorical":[],"excluded":[]}
    for name in frame.columns:
        if name in METADATA_COLUMNS:schema["excluded"].append(name)
        elif name in CATEGORICAL_COLUMNS:
            values=frame[name].dropna()
            if not values.map(lambda value:isinstance(value,str)).all():
                raise ValueError(f"Expected text category values: {name}")
            schema["categorical"].append(name)
        elif pd.api.types.is_numeric_dtype(frame[name]):schema["numeric"].append(name)
        else:raise ValueError(f"Unsupported predictor dtype {frame[name].dtype}: {name}")
    return schema


def prepare_workspace(work_root,raw_directory):
    work,raw=Path(work_root).resolve(),Path(raw_directory).resolve()
    if work==ROOT or raw.is_relative_to(work) or work.is_relative_to(raw):
        raise ValueError("Workspace and canonical raw data must be separate")
    # Preflight all paths before creating any links or replacing any file.
    for directory in (work/"data",work/"data/processed",work/"artifacts"):
        if not directory.resolve().is_relative_to(work):
            raise ValueError("Workspace output directory escapes its root")
    links=[]
    for name in RAW_TABLES:
        src,dst=raw/f"{name}.csv",work/"data"/f"{name}.csv"
        if not src.is_file():raise ValueError(f"Missing canonical raw table: {src}")
        if dst.exists() or dst.is_symlink():
            if not dst.is_symlink() or dst.resolve()!=src.resolve():
                raise ValueError(f"Refusing to replace an existing raw input: {dst}")
        else:links.append((src.resolve(),dst))
    for directory in (work/"data/processed",work/"artifacts/optimization",work/"artifacts/tables"):
        directory.mkdir(parents=True,exist_ok=True)
    for src,dst in links:dst.symlink_to(src)
    return work


def mask_reference_population(frame,fit_application_ids):
    ids=list(fit_application_ids)
    if not ids or len(ids)!=len(set(ids)):
        raise ValueError("Reference application IDs must be nonempty and unique")
    if not frame.application_id.is_unique or not frame.customer_id.is_unique:
        raise ValueError("Expected one application per customer")
    allowed=set(frame.loc[frame.split.eq("train"),"application_id"])
    if not set(ids)<=allowed:
        raise ValueError("Reference IDs must belong to the official training population")
    result=frame.copy()
    fit=result.application_id.isin(ids)
    result["split"]=pd.Series("test",index=result.index).mask(fit,"train")
    result.loc[~fit,"target"]=float("nan")
    assert result.loc[~fit,"target"].isna().all()
    assert int(result.split.eq("train").sum())==len(ids)
    return result


def source_namespace(filename,work_root):
    """Load original declarations with a redirected root; never call model code.

    V5's optimize_ks import only supplies ROOT/OUT. Supplying these constants here
    avoids importing unrelated model-training dependencies. All function bodies
    and numerical operations are compiled unchanged from the actual source.
    """
    if filename not in (*REFERENCE_SOURCES,"analyze.py","extend_features.py"):
        raise ValueError("Not an approved feature-only source")
    work=Path(work_root).resolve()
    if work==ROOT or work==ROOT/"data" or work==ROOT/"artifacts":
        raise ValueError("A separate experiment work directory is required")
    source=ROOT/"scripts"/filename
    tree=ast.parse(source.read_text(),filename=str(source))
    tree.body=[n for n in tree.body if not isinstance(n,ast.ImportFrom) or n.module!="optimize_ks"]
    namespace={"__name__":"_teammate_feature_source","__file__":str(source)}
    exec(compile(tree,str(source),"exec"),namespace)
    namespace.update(ROOT=work,RAW=work/"data",PROCESSED=work/"data/processed",
                     OUT=work/"artifacts" if filename=="analyze.py" else work/"artifacts/optimization",
                     TABLES=work/"artifacts/tables")
    return namespace


def run_reference_stages(work_root,extended_frame,fit_application_ids):
    work=Path(work_root).resolve()
    # Validate the root before writing anything. Baseline sources/artifacts stay intact.
    source_namespace("build_features_v3.py",work)
    for name in ("data/processed","artifacts/optimization"):
        p=work/name
        if not p.resolve().is_relative_to(work):
            raise ValueError("Output paths must remain inside the experiment work directory")
        p.mkdir(parents=True,exist_ok=True)
    masked=mask_reference_population(extended_frame,fit_application_ids)
    masked.to_parquet(work/"data/processed/application_features_extended.parquet",index=False)
    for filename in REFERENCE_SOURCES:
        namespace=source_namespace(filename,work)
        namespace["main"]()
    result=pd.read_parquet(work/"data/processed/application_features_v5.parquet")
    assert result.application_id.equals(masked.application_id)
    assert result.customer_id.equals(masked.customer_id)
    assert result.target.equals(masked.target)
    assert result.loc[result.split.eq("test"),"target"].isna().all()
    assert result.columns.is_unique
    return result
