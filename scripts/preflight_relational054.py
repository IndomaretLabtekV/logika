"""Check Kumo Relational's cached multi-table API before a real-data experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_goal054 as r

REVISION = "2bd603d3d8f25f67a7aaa8579908595e20567e22"


def table(frame, device="cuda"):
    import sdm

    frame = frame.copy()
    stypes = {}
    for column in frame:
        if column == "customer_id":
            # Parquet profiles may store IDs as Arrow dictionary categories.
            # SDM's ID block expects strings, not a dictionary-to-object array.
            frame[column] = frame[column].astype(str)
            stypes[column] = "id"
        elif pd.api.types.is_datetime64_any_dtype(frame[column]):
            stypes[column] = "datetime"
        elif pd.api.types.is_numeric_dtype(frame[column]):
            stypes[column] = "numerical"
        else:
            stypes[column] = "categorical"
    return sdm.TableTensor.from_pandas(frame, stypes=stypes, device=device)


def related(frames, ids, device="cuda"):
    import sdm

    ids = set(ids)
    selected = {
        name: frame[frame.customer_id.isin(ids)].reset_index(drop=True)
        for name, frame in frames.items()
    }
    assert set(selected["users"].customer_id) == ids
    assert all(set(frame.customer_id).issubset(ids) for frame in selected.values())
    tables = {name: table(frame, device=device) for name, frame in selected.items()}
    relationships = [
        dict(
            left_table=name,
            left_columns="customer_id",
            right_table="users",
            right_columns="customer_id",
        )
        for name in tables
        if name != "users"
    ]
    result = sdm.RelatedTables(
        tables=tables,
        relationships=relationships,
        task_links=[
            dict(task_columns="customer_id", table="users", table_columns="customer_id")
        ],
    )
    return result


def main():
    import torch
    import sdm
    from huggingface_hub import hf_hub_download

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.15)
    model_file = hf_hub_download(
        "nvidia/Kumo-Relational", "classifier.pt", revision=REVISION, token=False
    )
    ids = [f"preflight-customer-{i}" for i in range(24)]
    users = pd.DataFrame(dict(customer_id=ids, age=np.arange(24) + 25))
    cid = np.repeat(ids, 2)
    prior = pd.to_datetime(np.tile(["2024-11-01", "2024-12-01"], 24))
    accounts = pd.DataFrame(
        dict(
            customer_id=cid,
            month=prior,
            income=np.arange(48) * 100 + 3000000,
            outflow=np.arange(48) * 90 + 2000000,
        )
    )
    # One query customer lacks account history, as in the real data.
    accounts = accounts[accounts.customer_id.ne(ids[-1])].copy()
    bureau = pd.DataFrame(
        dict(
            customer_id=cid,
            report_date=prior,
            provider=np.tile(["A", "B"], 24),
            score=np.arange(48) + 600.0,
        )
    )
    bureau.loc[3, "score"] = np.nan
    ecommerce = pd.DataFrame(
        dict(
            customer_id=cid,
            event_date=prior,
            category=np.tile(["fashion", "groceries"], 24),
            unit_price=np.arange(48) * 1000 + 5000,
        )
    )
    repayment = pd.DataFrame(
        dict(
            customer_id=cid,
            due_date=prior,
            dpd=np.tile([0, 7], 24),
            amount_due=np.arange(48) * 1000 + 100000,
        )
    )
    frames = dict(
        users=users,
        accounts=accounts,
        bureau=bureau,
        ecommerce=ecommerce,
        repayment=repayment,
    )
    task = pd.DataFrame(
        dict(
            customer_id=ids,
            decision_datetime=pd.to_datetime(["2025-01-01"] * 24),
            loan_amount=np.arange(24) * 100000 + 2000000,
        )
    )
    context_ids, query_ids = ids[:16], ids[16:]
    assert set(context_ids).isdisjoint(query_ids)
    xc, xq = table(task.iloc[:16]), table(task.iloc[16:])
    yc = sdm.TableTensor.from_pandas(
        pd.DataFrame({"target": np.tile([0, 1], 8)}),
        stypes={"target": "categorical"},
        device="cuda",
    )
    rc, rq = related(frames, context_ids), related(frames, query_ids)
    print("RELATIONAL MODEL LOAD", flush=True)
    model = sdm.models.KumoRelational(task="classification", device="cuda")
    torch.manual_seed(2026)
    print("RELATIONAL CONTEXT FIT", flush=True)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        model.fit(
            xc,
            yc,
            rc,
            num_estimators=1,
            estimator_batch_size=1,
            num_hops=1,
            generator=torch.Generator(device="cuda").manual_seed(2026),
        )
        output = model.predict(xq, rq)
        probability = output[:, "1"].numerical.reshape(-1).float().cpu().numpy()
    assert (
        len(probability) == 8
        and np.isfinite(probability).all()
        and ((probability >= 0) & (probability <= 1)).all()
    )
    result = dict(
        passed=True,
        weights="nvidia/Kumo-Relational",
        weights_revision=REVISION,
        weights_sha256=r.checksum(Path(model_file)),
        context_rows=16,
        query_rows=8,
        estimators=1,
        num_hops=1,
        tables={name: len(frame) for name, frame in frames.items()},
        ids_used_only_as_links=True,
        context_and_query_customers_disjoint=True,
        missing_account_history_tested=True,
        numeric_missing_values_tested=True,
        note="Synthetic API test only; this is not a LOGIKA validation score.",
    )
    r.write_json(r.DEST / "relational_preflight.json", result)
    print("RELATIONAL PREFLIGHT PASS", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
