"""Neural-network input preparation (TabM). Pure numpy/scipy so it runs and is verified locally (eda/32).

Per numeric column: mid-rank ECDF -> standard-normal map fitted on train rows only (rank-preserving, so univariate
KS is unchanged; a value shared by many rows, such as 0 days late, lands in the middle of its tie block, not at
an extreme), clipped to +-5, NaN -> 0 (the median). Columns with an identical missingness pattern share one
missing-indicator, because thin-file customers miss whole tables at once. Constant columns are dropped."""
import numpy as np
from scipy.special import ndtri


def nn_prep(train: np.ndarray, test: np.ndarray):
    """train/test: float arrays (rows x numeric features), NaN/inf allowed. Returns (Xtr, Xte, info), float32."""
    train = np.where(np.isfinite(train), train, np.nan)
    test = np.where(np.isfinite(test), test, np.nan)
    ref = [np.sort(train[~np.isnan(train[:, j]), j]) for j in range(train.shape[1])]
    keep = [j for j, s in enumerate(ref) if len(s) and s[0] < s[-1]]

    def transform(a):
        out = np.zeros((a.shape[0], len(keep)), np.float32)
        for i, j in enumerate(keep):
            s, x = ref[j], a[:, j]
            m = ~np.isnan(x)
            u = (np.searchsorted(s, x[m], side="left") + np.searchsorted(s, x[m], side="right")) / (2 * len(s))
            out[m, i] = np.clip(ndtri(np.clip(u, 0.5 / len(s), 1 - 0.5 / len(s))), -5, 5)
        return out

    miss_tr, miss_te = np.isnan(train[:, keep]), np.isnan(test[:, keep])
    seen, ind_cols = set(), []
    for i in range(miss_tr.shape[1]):
        key = np.packbits(miss_tr[:, i]).tobytes()
        if miss_tr[:, i].any() and key not in seen:
            seen.add(key)
            ind_cols.append(i)
    Xtr = np.hstack([transform(train), miss_tr[:, ind_cols].astype(np.float32)])
    Xte = np.hstack([transform(test), miss_te[:, ind_cols].astype(np.float32)])
    return Xtr, Xte, {"kept": keep, "indicator_of": ind_cols}
