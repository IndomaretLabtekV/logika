"""CPU-only audit of Float64 -> SDM default Float32 input and monetary log.

Uses inner-fit rows only for feature statistics; does not fit any model, choose
features from held-out labels, or change the production feature cache. Prints
numbers and saves a diagnostic plot and per-column CSV in eda/out.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
import sdm

import valid2 as v
from common import OUT, PQ, features


def describe(name, x, y):
    finite = np.isfinite(x)
    raw = x[finite]
    fp32 = raw.astype(np.float32).astype(np.float64)
    u64, u32 = np.unique(raw), np.unique(fp32)
    cast = np.full_like(x, np.nan)
    cast[finite] = fp32
    adjacent = np.diff(u64)
    return {"feature": name, "finite": int(finite.sum()), "unique64": len(u64), "unique32": len(u32),
            "unique_loss_fraction": float(1 - len(u32) / max(len(u64), 1)),
            "max_abs_cast_error": float(np.max(np.abs(raw - fp32))) if len(raw) else 0.,
            "median_nonzero_gap64": float(np.median(adjacent)) if len(adjacent) else 0.,
            "ks64": v.ks_nan(y, x), "ks32": v.ks_nan(y, cast),
            "ks_delta_cast": v.ks_nan(y, cast) - v.ks_nan(y, x)}


def main():
    X, _, y, _, _ = v.frames(features())
    tr, _ = v.outer(y, 42)[0]
    fi, _ = v.inner(y[tr], 42, 0)
    fi = tr[fi]
    sub, yf = X.iloc[fi], y[fi]
    nums = [c for c in X if c not in v.CAT_COLS]
    selected = v.fm_select(X, y, fi, 100)
    money = json.loads((OUT / "153_kumo_input_audit_money.json").read_text())["columns"]
    rows = []
    for c in nums:
        x = sub[c].to_numpy(float)
        row = describe(c, x, yf)
        row["selected100"] = c in selected
        row["audited_money"] = c in money
        if c in money:
            z = np.sign(x) * np.log1p(np.abs(x))
            d = describe(c, z, yf)
            row.update({f"moneylog_{key}": val for key, val in d.items() if key != "feature"})
        rows.append(row)
    df = pd.DataFrame(rows)
    # Observe the actual public conversion method on CPU, not just NumPy simulation.
    sample = sub.iloc[:64][["loan_amount", "pp_mean", "loan_purpose"]].copy()
    cat = sorted(sub["loan_purpose"].dropna().unique())
    sample["loan_purpose"] = pd.Categorical(sample["loan_purpose"], categories=cat)
    table = sdm.TableTensor.from_pandas(sample, sdm.infer_stypes(sample, overrides={
        "loan_amount": "numerical", "pp_mean": "numerical", "loan_purpose": "categorical"}), device="cpu")
    numeric = sample[["loan_amount", "pp_mean"]].to_numpy(dtype=np.float32, copy=True)
    assert torch.equal(torch.nan_to_num(table.numerical), torch.nan_to_num(torch.as_tensor(numeric)))
    print("inner-fit", len(fi), "numeric columns", len(nums), "default torch dtype", torch.get_default_dtype(),
          "observed TableTensor dtype", table.numerical.dtype, flush=True)
    print("columns with ties added by Float32", int((df.unique32 < df.unique64).sum()),
          "selected100", int(((df.unique32 < df.unique64) & df.selected100).sum()), flush=True)
    print(df.sort_values("ks_delta_cast").loc[:, ["feature", "unique64", "unique32", "unique_loss_fraction",
                                                "ks64", "ks32", "ks_delta_cast", "selected100"]].head(15).to_string(index=False), flush=True)
    monetary = df[df.audited_money]
    print("money-only max absolute KS change Float32", float(monetary.ks_delta_cast.abs().max()),
          "after signed-log", float(monetary.moneylog_ks_delta_cast.abs().max()), flush=True)
    df.to_csv(OUT / "157_kumo_precision_audit.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    worst = df.sort_values("unique_loss_fraction").tail(18)
    axes[0].barh(worst.feature, worst.unique_loss_fraction)
    axes[0].set(xlabel="Fraction of unique Float64 values merged in Float32", title="Observed input conversion, inner-fit only")
    axes[1].scatter(monetary.ks_delta_cast, monetary.moneylog_ks_delta_cast, alpha=.7)
    axes[1].axhline(0, color="black", lw=.6)
    axes[1].axvline(0, color="black", lw=.6)
    axes[1].set(xlabel="Raw-money Float32 KS minus Float64 KS", ylabel="Money-log Float32 KS minus Float64 KS",
                title="Precision effects, not model-quality evidence")
    fig.tight_layout()
    fig.savefig(OUT / "157_kumo_precision_audit.png", dpi=130)
    plt.close(fig)
    result = {"rows": len(fi), "numeric_features": len(nums), "actual_table_dtype": str(table.numerical.dtype),
              "columns_with_added_ties": int((df.unique32 < df.unique64).sum()),
              "max_abs_ks_delta": float(df.ks_delta_cast.abs().max()),
              "monetary_max_abs_ks_delta": float(monetary.ks_delta_cast.abs().max()),
              "monetary_log_max_abs_ks_delta": float(monetary.moneylog_ks_delta_cast.abs().max()),
              "features_sha256": hashlib.sha256((PQ / "features.parquet").read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limits": ["No model inference and no improvement claim.", "No new feature or fitted transform was created.",
                         "Cutoff and duplication follow unchanged cache; this audit only casts row values."]}
    (OUT / "157_kumo_precision_audit.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
