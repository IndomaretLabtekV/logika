"""Run the declared TabPFN queue, stopping after a full OOF target result.

Stopping the queue does not mark the goal complete: independent verification,
the review notebook, and candidate submission still have to be produced.
"""

from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np
from sklearn.model_selection import train_test_split

import research_goal054 as r


def preflight():
    import torch
    from pydantic import TypeAdapter

    spec = r.specifications()[0]
    data, _ = r.inputs([spec])
    ti, vi = train_test_split(
        np.arange(len(data)),
        train_size=1024,
        test_size=32,
        stratify=data.target,
        random_state=2026,
    )
    x = r.model_features(data, spec["columns"])
    path = r.checkpoint_path(spec)
    old = r.classifier(spec, path, 0, estimators=1)
    old.set_params(fit_mode="fit_preprocessors", kv_cache_precision=None)
    print("PREFLIGHT original execution", flush=True)
    old.fit(x.iloc[ti], data.target.iloc[ti].astype(int))
    previous = old.predict_proba(x.iloc[vi])[:, 1]
    del old
    gc.collect()
    torch.cuda.empty_cache()
    print("PREFLIGHT cached execution", flush=True)
    model = r.classifier(spec, path, 0, estimators=1)
    model.fit(x.iloc[ti], data.target.iloc[ti].astype(int))
    actual = model.predict_proba(x.iloc[vi])[:, 1]
    difference = float(np.max(np.abs(actual - previous)))
    assert len(actual) == 32 and np.isfinite(actual).all()
    assert difference < 0.01, "Unexpected difference between inference implementations."
    fitted = model.get_inference_config()
    config = TypeAdapter(type(fitted)).dump_python(fitted, mode="json")
    json.dumps(config)
    result = dict(
        passed=True,
        context_rows=1024,
        query_rows=32,
        estimators=1,
        weights_revision=r.WEIGHT_REVISION,
        weights_sha256=r.checksum(path),
        original_vs_cached_max_abs_difference=difference,
        cache_location="CPU",
        cache_precision="auto",
        config_serialization_passed=True,
    )
    r.write_json(r.DEST / "cache_preflight.json", result)
    print("PREFLIGHT PASS", json.dumps(result), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--candidates", nargs="+", default=[s["name"] for s in r.specifications()]
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not args.skip_preflight:
        preflight()
    for name in args.candidates:
        started = time.monotonic()
        r.run(
            argparse.Namespace(
                candidates=[name], folds=None, batch_size=8192, test=True
            )
        )
        result = r.compare()
        print(
            "CANDIDATE COMPLETE",
            name,
            "seconds",
            time.monotonic() - started,
            flush=True,
        )
        if result["target_reached"]:
            print(
                "TARGET REACHED: independent audit, notebook and submission still required.",
                flush=True,
            )
            break
    print("DECLARED QUEUE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
