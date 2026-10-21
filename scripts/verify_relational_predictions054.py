"""Compare original and optimized cached probabilities on real, unlabeled queries."""

from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import StratifiedKFold

import research_goal054 as r
from research_relational054 import load_inputs, fit_context, predictions
from relational_readout054 import install


def verify():
    import torch
    import sdm

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.1)
    data, test, columns, frames, manifest = load_inputs()
    ti, vi = next(
        StratifiedKFold(5, shuffle=True, random_state=31027).split(data, data.target)
    )
    ci = r.context_indices(ti, data.target.astype(int), 512, 2026)
    vi = vi[:32]
    assert not np.intersect1d(ci, vi).size
    model = sdm.models.KumoRelational(task="classification", device="cuda")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    values = []
    for optimized in [False, True]:
        if optimized:
            install(model)
        torch.manual_seed(2026)
        fit_context(model, data, columns, frames, ci, 1, 2026)
        values.append(predictions(model, data.iloc[vi], columns, frames, 32))
        model.clear()
        torch.cuda.empty_cache()
    error = np.abs(values[0].astype(float) - values[1].astype(float))
    assert np.isfinite(error).all() and error.max() < 0.003
    result = dict(
        passed=True,
        context_rows=512,
        query_rows=32,
        estimators=1,
        maximum_probability_difference=float(error.max()),
        mean_probability_difference=float(error.mean()),
        absolute_tolerance=0.003,
        backend_sha256=r.checksum(r.ROOT / "scripts/relational_readout054.py"),
        context_ids_sha256=r.ids_hash(data, ci),
        query_ids_sha256=r.ids_hash(data, vi),
        validation_labels_used=False,
        validation_scores_computed=False,
        caveat="Real-data technical parity at 512 context rows; original full graph exceeds the memory budget, so full-scale bit identity is not claimed.",
    )
    r.write_json(r.DEST / "relational_probability_parity.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    verify()
