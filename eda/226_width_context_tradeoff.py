"""Inner fold-0 screen: trade columns for context at ~fixed VRAM for the wide 896 pool.
cap300 @ 28672 rows and cap200 @ 40960 rows (per-estimator Latin-shuffled column subsets) vs
C2_ctx16384 (cap500 @ 16384, cache copied from 222's folder). Reuses 222's pipeline, new folder.
Usage: cd eda && ../.venv/bin/python 226_width_context_tradeoff.py <arm>   (one uncached arm per process)
"""
import importlib.util

from common import ROOT

spec = importlib.util.spec_from_file_location("screen222", ROOT / "eda/222_context_ssl_screen.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
CAP = {28672: 300, 40960: 200, 16384: 500}  # rows*(cap+4)*256*2 B ~ 4.2-4.4 GB each
s.OUT = ROOT / "results/v61/width_context_f0"
s.ARMS = {"W_cap300_ctx28672": ("wide", 28672, .85, "fit"),
          "W_cap200_ctx40960": ("wide", 40960, .85, "fit"),
          "C2_ctx16384": ("wide", 16384, .85, "fit")}
_predict = s.predict


def predict(*args, **kw):
    kw["column_cap"] = CAP[kw["max_context"]]
    return _predict(*args, **kw)


s.predict = predict
s.__file__ = __file__  # manifest records this script's hash
if __name__ == "__main__":
    s.main()
