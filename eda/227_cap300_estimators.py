"""Inner fold-0 screen: e16 vs e8 for the cap300 @ 28672 wide view (random 300-of-896 columns per
estimator, so more estimators cover more columns). e8 arm is the cached 226 prediction.
Usage: cd eda && ../.venv/bin/python 227_cap300_estimators.py <arm>
"""
import importlib.util

from common import ROOT

spec = importlib.util.spec_from_file_location("screen222", ROOT / "eda/222_context_ssl_screen.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
s.OUT = ROOT / "results/v61/estimators_f0"
s.ARMS = {"W_cap300_ctx28672": ("wide", 28672, .85, "fit"), "W_cap300_ctx28672_e16": ("wide", 28672, .85, "fit")}
_predict = s.predict


def predict(*args, **kw):
    kw["column_cap"] = 300
    if kw["n_estimators"] == 8 and predict.arm.endswith("_e16"):
        kw["n_estimators"] = 16
    return _predict(*args, **kw)


s.predict = predict
s.__file__ = __file__
if __name__ == "__main__":
    import sys
    predict.arm = sys.argv[1]
    s.main()
