"""E-commerce: spending patterns, cancellations, payment methods."""
import polars as pl
from common import analyze, features

f = features()
cols = [c for c in f.columns if c.startswith("ec_")] + ["ecspend_to_income"]
analyze(f, cols, "07_ecom", top_plot=16)
