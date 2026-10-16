"""Read public scores of the probe submissions against their OOF expectations (arithmetic only).
Usage: python eda/s04_probe_analysis.py S1=0.55xx S2=0.55xx S3=0.56xx   (any subset)"""
import sys
from pathlib import Path

import polars as pl

V8_PUBLIC = 0.56145
exp = pl.read_csv(Path(__file__).resolve().parents[1] / "submissions/probe/expectations.csv")
pub = {k: float(v) for k, v in (a.split("=") for a in sys.argv[1:])}
rows = []
for r in exp.iter_rows(named=True):
    key = r["sub"].split("_")[0]
    if key in pub:
        d = pub[key] - V8_PUBLIC
        rows.append(dict(sub=r["sub"], public=pub[key], d_public=round(d, 5), d_oof=r["d_oof_vs_v8"],
                         slope=round(d / r["d_oof_vs_v8"], 2), z_vs_expected=round((pub[key] - r["exp_public"]) / r["sd_public_diff"], 2)))
print(pl.DataFrame(rows))
if {"S1", "S2"} <= pub.keys():
    # OOF says NN/FM beats GBDT by 0.0016; sd of a difference of two ~0.003 diffs is ~0.004
    gap = pub["S2"] - pub["S1"]
    print(f"public S2-S1 {gap:+.4f} (OOF +0.0016); |z| ~ {abs(gap - 0.0016) / 0.004:.1f}  -> family preference shift only if |z| > 2")
for r in rows:
    flag = "consistent with OOF" if abs(r["z_vs_expected"]) < 2 else "OFF expectation (|z|>=2): public subset behaves differently"
    print(f"{r['sub']}: {flag}")
