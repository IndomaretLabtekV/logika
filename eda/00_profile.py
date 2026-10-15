"""Schema, null rate, cardinality, row counts for every table."""
import polars as pl
from common import TABLES, load

for t in TABLES:
    df = load(t)
    print(f"\n===== {t}  shape={df.shape}")
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append((c, str(s.dtype), s.null_count() / len(s), s.n_unique(),
                     str(s.min()) if s.dtype != pl.String else str(s.drop_nulls().head(2).to_list()),
                     str(s.max()) if s.dtype != pl.String else ""))
    with pl.Config(tbl_rows=50, tbl_cols=10, fmt_str_lengths=60, tbl_width_chars=200):
        print(pl.DataFrame(rows, schema=["col", "dtype", "null_rate", "n_unique", "min", "max"], orient="row"))
