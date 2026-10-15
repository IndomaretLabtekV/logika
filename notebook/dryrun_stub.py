"""Local dry run of a built notebook WITHOUT any model: torch, lightgbm, xgboost, catboost, sklearn, tabpfn, tabm and
exaone are replaced by stubs that return random predictions, so only data loading, feature building, leakage tests,
TabM/FM preprocessing, blending and submission code paths execute. KS values printed by a dry run are meaningless.

Usage: ../.venv/bin/python dryrun_stub.py v7.ipynb   (writes to ../output or ../output_v7, delete it afterwards)"""
import contextlib
import json
import sys
import time
import types

import matplotlib
matplotlib.use("Agg")
import numpy as np
import scipy.stats  # noqa: F401  (import before the torch stub)

nb = json.load(open(sys.argv[1]))
rng = np.random.default_rng(0)
torch = types.ModuleType("torch")
torch.__version__ = "stub"
torch.Tensor = type("Tensor", (), {})
torch.cuda = types.SimpleNamespace(device_count=lambda: 0, is_available=lambda: False, manual_seed_all=lambda s: None,
                                   OutOfMemoryError=MemoryError, empty_cache=lambda: None,
                                   device=lambda d: contextlib.nullcontext())
torch.manual_seed = lambda s: None
torch.device = lambda s: s
sys.modules["torch"] = torch
for name in ["tabm", "exaonetabular", "tabpfn"]:
    sys.modules[name] = None
for name in ["lightgbm", "xgboost", "catboost", "sklearn", "sklearn.model_selection", "sklearn.metrics"]:
    sys.modules[name] = types.ModuleType(name)


class Dataset:
    def __init__(self, X, y, weight=None, categorical_feature=None, reference=None):
        assert all(c in X.columns for c in categorical_feature)
        assert weight is None or len(weight) == len(X) == len(y)
        self.cols = list(X.columns)


class Booster:
    def __init__(self, cols):
        self.cols, self.best_iteration = cols, 10

    def predict(self, X, num_iteration=None):
        assert list(X.columns) == self.cols, "predict columns differ from train columns"
        return rng.random(len(X))

    def feature_name(self):
        return self.cols

    def feature_importance(self, kind):
        return rng.random(len(self.cols))


L = sys.modules["lightgbm"]
L.Dataset = Dataset
L.train = lambda params, dtr, n, valid_sets, callbacks: Booster(dtr.cols)
L.early_stopping = lambda n, verbose: None


class DMatrix:
    def __init__(self, X, y=None, weight=None, enable_categorical=False):
        assert weight is None or len(weight) == len(X)
        self.n = len(X)


class XB:
    best_iteration = 10

    def predict(self, d, iteration_range):
        return rng.random(d.n)


X = sys.modules["xgboost"]
X.DMatrix = DMatrix
X.train = lambda params, dtr, n, evals, early_stopping_rounds, verbose_eval: XB()


class CatBoostClassifier:
    def __init__(self, **kw):
        assert "depth" in kw

    def fit(self, pool, eval_set):
        pass

    def predict_proba(self, X):
        p = rng.random(len(X))
        return np.c_[1 - p, p]


C = sys.modules["catboost"]
C.CatBoostClassifier = CatBoostClassifier
C.Pool = lambda X, y, cat_features, weight=None: X


class SKF:
    def __init__(self, n, shuffle, random_state):
        self.n, self.rs = n, random_state

    def split(self, X, y):
        idx = np.random.default_rng(self.rs).permutation(len(y))
        parts = np.array_split(idx, self.n)
        for k in range(self.n):
            yield np.concatenate(parts[:k] + parts[k + 1:]), parts[k]


sys.modules["sklearn.model_selection"].StratifiedKFold = SKF
sys.modules["sklearn.metrics"].roc_auc_score = lambda y, p: float("nan")

g = {}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code" or c["source"].startswith("%%"):
        continue
    src = "\n".join(line for line in c["source"].split("\n") if not line.lstrip().startswith(("%", "!")))
    t = time.time()
    exec(compile(src.replace("plt.show()", "plt.close('all')"), f"cell{i}", "exec"), g)
    print(f"--- cell {i} ok ({time.time() - t:.1f}s)", flush=True)

# model-free code paths that the stubs skip
XN_TR, XN_TE, info = g["nn_prep"](g["X_tr"][g["NUM_COLS"]].to_numpy(dtype=float), g["X_te"][g["NUM_COLS"]].to_numpy(dtype=float))
print("TabM prep:", XN_TR.shape, XN_TE.shape, "finite", np.isfinite(XN_TR).all() and np.isfinite(XN_TE).all())
if "EXAONE_NUM" in g:
    cols = g["select_fm_features"](g["folds"](42)[0][0], g["EXAONE_NUM"])
    print(f"EXAONE columns selected on fold 0: {len(cols)} (limit 100)", g["fm_matrix"](g["X_tr"].iloc[:50], cols).shape)
if "VIEWS" in g:
    for view in g["VIEWS"]:
        cols = g["select_fm_features"](g["folds"](42)[0][0], None, view)
        M = g["fm_matrix"](g["VIEWS"][view][0].iloc[:100], cols)
        print(f"FM view {view}: {len(cols)} columns selected on fold 0, matrix {M.shape} {M.dtype}")


class FakeFM:
    whole_query = True

    def __init__(self, cat_idx, dev):
        self.dev = dev

    def fit(self, X, y):
        assert X.dtype == np.float32 and len(X) == len(y)
        return self

    def predict_proba(self, X):
        p = rng.random(len(X))
        return np.c_[1 - p, p]


g["run_fm"]("fake_fm", FakeFM, 20000, devices=["cuda:0", "cuda:1"])
assert "fake_fm" in g["RESULTS"], "parallel run_fm did not report"
assert g["RESULTS"]["fake_fm"]["test"].shape == (len(g["X_te"]),)
g["run_fm"]("fake_fm_late", FakeFM, 20000, devices=["cuda:0", "cuda:1"], fold_min=1e6)


class FakeTabICL:
    def __init__(self, **kw):
        assert "n_estimators" in kw

    def fit(self, X, y):
        self.width = X.shape[1]
        assert X.shape[1] > 104, "missingness-group flags were not appended"
        return self

    def predict_proba(self, X):
        assert X.shape[1] == self.width
        p = rng.random(len(X))
        return np.c_[1 - p, p]


class FakeCausilo:
    def __init__(self, **kw):
        pass

    def fit(self, X, y):
        cats = [c for c in X.columns if str(X[c].dtype) == "category"]
        assert len(cats) == 4, cats
        return self

    def predict_proba(self, X):
        assert sum(str(X[c].dtype) == "category" for c in X.columns) == 4
        p = rng.random(len(X))
        return np.c_[1 - p, p]


class FakeRealMLP:
    def __init__(self, **kw):
        pass

    def fit(self, X, y, Xv, yv, cat_col_names):
        for d in (X, Xv):
            num = d.drop(columns=cat_col_names)
            assert np.isfinite(num.to_numpy()).all(), "RealMLP input has missing numbers"
            assert all(str(d[c].dtype) == "category" for c in cat_col_names)
        assert list(X.columns) == list(Xv.columns)
        return self

    def predict_proba(self, X):
        p = rng.random(len(X))
        return np.c_[1 - p, p]


g["CausiloClassifier"], g["RealMLP_TD_Classifier"] = FakeCausilo, FakeRealMLP
if "TabICLModel" in g:
    g["TabICLClassifier"] = FakeTabICL
    g["run_fm"]("fake_tabicl", g["TabICLModel"], 20000, devices=["cuda:0", "cuda:1"])
g["run_fm"]("fake_causilo", g["CausiloModel"], 20000, devices=["cuda:0", "cuda:1"])
g["run_gbdt"]("fake_realmlp", g["make_realmlp_fit"](), seeds=[42])
for n in ("fake_causilo", "fake_realmlp") + (("fake_tabicl",) if "TabICLModel" in g else ()):
    assert n in g["RESULTS"], n
assert "fake_fm_late" not in g["RESULTS"], "deadline guard did not drop the FM"
g["CFG"].DEADLINE = time.time()
print("deadline guard returns:", g["run_gbdt"]("dummy", lambda tri, vai, seed: (rng.random(len(vai)), rng.random(len(g['X_te'])))))
