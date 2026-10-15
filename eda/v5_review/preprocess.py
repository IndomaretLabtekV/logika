"""Reuse verified preprocessing; repair the class-ordered EXAONE probe split without importing any model."""
import importlib.util
from pathlib import Path
import numpy as np
_spec = importlib.util.spec_from_file_location("existing_preprocessing", Path(__file__).resolve().parents[1] / "deep/04_preprocessing.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
FoldNumeric = _module.NumericPreprocessor
category_fit_transform = _module.categorical_fit_transform

def stratified_probe(y, n_each=3000, seed=42):
    y = np.asarray(y); rng = np.random.default_rng(seed)
    groups = []
    for cls in [0, 1]:
        idx = rng.permutation(np.flatnonzero(y == cls))
        count = int(round(n_each * len(idx) / len(y))) if cls == 1 else n_each-int(round(n_each * y.mean()))
        assert len(idx) >= 2*count and count > 0
        groups.append((idx[:count], idx[count:2*count]))
    context = rng.permutation(np.concatenate([p[0] for p in groups]))
    query = rng.permutation(np.concatenate([p[1] for p in groups]))
    assert not np.intersect1d(context, query).size
    assert len(np.unique(y[context])) == len(np.unique(y[query])) == 2
    return context, query


def self_check():
    _module.checks()
    context, query = stratified_probe(np.tile([0, 0, 0, 1], 100), 50)
    assert len(context) == len(query) == 50
