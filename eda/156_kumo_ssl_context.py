#!/usr/bin/env python3
"""Nested Kumo-large SSL context augmentation using only the organizer's test rows.
This is a NEW transductive SSL hypothesis, not a reproduction of VIP-COP's
same-class labeled-training mixup augmentation. Only the group coalition search
is VIP-COP-inspired; no synthetic labeled-training mixup is implemented here.

Numeric features stay raw. --category-type numerical reproduces baseline integer
codes; categorical is an explicitly separate experiment, never an outer-tuned choice.
All real context rows remain present. Only pseudo context is capped (at most 10k).
Smoke budgets change the experiment and are not evidence for the full-data gate.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import sdm
import valid2 as v
from common import OUT, PQ, features
from ssl_policies import group_ids, policy_name, pseudo_rows


def load_helper(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vip = load_helper("ssl_vip", "152_vipcop_kumo.py")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def save_frame(path, frame):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(path)


def score(y, p):
    assert len(np.unique(y)) == 2
    return {"ks": v.ks(y, p), "auc": float(roc_auc_score(y, p))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--evaluate", action="store_true", help="Unlock outer only after selection.json is saved")
    ap.add_argument("--category-type", choices=["numerical", "categorical"], default="numerical")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--gpu-memory-fraction", type=float, default=.9)
    ap.add_argument("--caps", default="2000,10000", help="Predetermined pseudo-context caps; each must be <=10000")
    ap.add_argument("--positive-thresholds", default="", help="Optional class-specific positive thresholds, e.g. 0.7,0.9; negatives stay .02/.05")
    ap.add_argument("--group-coalitions", type=int, default=8, help="0 disables group coalition surrogate")
    ap.add_argument("--min-inner-gain", type=float, default=.003)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-fit", type=int, default=1024)
    ap.add_argument("--smoke-query", type=int, default=256)
    ap.add_argument("--smoke-test", type=int, default=1024)
    a = ap.parse_args()
    caps = sorted(set(map(int, a.caps.split(","))))
    positive_thresholds = sorted(set(map(float, a.positive_thresholds.split(",")))) if a.positive_thresholds else []
    assert all(.5 <= p <= 1 for p in positive_thresholds)
    assert caps and all(0 < cap <= 10000 for cap in caps)
    assert 0 <= a.fold < v.N_FOLDS and a.chunk > 0 and 0 <= a.group_coalitions <= 14
    assert a.min_inner_gain >= 0
    assert min(a.smoke_fit, a.smoke_query, a.smoke_test) >= 16
    assert 0 < a.gpu_memory_fraction < 1
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction)
    a.out.mkdir(parents=True, exist_ok=True)
    folder = a.out / f"s{a.seed}_f{a.fold}"
    folder.mkdir(exist_ok=True)
    # Refuse implicit feature rebuilding: this analysis never runs an EDA build.
    assert (PQ / "features.parquet").exists(), "Prepare the existing local feature artifact first"
    config = {k: value for k, value in vars(a).items() if k not in {"out", "evaluate"}}
    config.update(caps=caps, model="KumoTabular-large", estimators=4, n_numeric=100,
                  context="all-real-plus-capped-pseudo", pseudo_label="teacher_probability>=0.5")
    packages = {p: importlib.metadata.version(p) for p in
                ("numpy", "pandas", "polars", "torch", "structured-data-models", "scikit-learn", "matplotlib")}
    sources = {name: file_hash(Path(__file__).with_name(name)) for name in
               (Path(__file__).name, "ssl_policies.py", "152_vipcop_kumo.py", "154_kumo_recipe_audit.py", "valid2.py", "common.py")}
    manifest = {"config": config, "config_sha256": digest(config), "sources_sha256": sources,
                "packages": packages, "packages_sha256": digest(packages),
                "features_sha256": file_hash(PQ / "features.parquet"),
                "method": "Kumo teacher / hard-pseudo context SSL; nested search and disjoint confirmation",
                "research_status": "new transductive SSL hypothesis; NOT VIP-COP sample-augmentation reproduction",
                "smoke_only": a.smoke, "external_data": False,
                "assumptions": ["Organizer test features are permitted for transductive SSL; test labels unavailable",
                                "Inherited valid2 stratified application folds, not customer-group folds",
                                "Confidence shells are not calibrated correctness probabilities",
                                "No OOM recovery that removes real rows; lower pseudo cap in a NEW experiment"]}
    mp = folder / "manifest.json"
    if mp.exists():
        assert json.loads(mp.read_text()) == manifest, "Resume rejected: config/source/package/data hashes differ"
    else:
        save_json(mp, manifest)
    run_hash = digest(manifest)
    X, Xt, y, ids, test_ids = v.frames(features())
    train_ids, test_ids = np.asarray(ids.to_list()), np.asarray(test_ids.to_list())
    assert len(set(train_ids)) == len(y) and len(set(test_ids)) == len(Xt)
    assert not set(train_ids).intersection(test_ids), "Train/test application IDs overlap"
    tr, va = v.outer(y, a.seed)[a.fold]
    fi, st = v.inner(y[tr], a.seed, a.fold)
    fi, st = tr[fi], tr[st]
    qs, qc = train_test_split(st, test_size=.5, stratify=y[st], random_state=a.seed + a.fold)
    qs, qc = np.sort(qs), np.sort(qc)
    test_rows = np.arange(len(Xt))
    if a.smoke:
        fi = vip.strat(fi, y, a.smoke_fit, np.random.default_rng(a.seed + 11))
        tr = vip.strat(tr, y, a.smoke_fit, np.random.default_rng(a.seed + 12))
        qs = vip.strat(qs, y, a.smoke_query, np.random.default_rng(a.seed + 13))
        qc = vip.strat(qc, y, a.smoke_query, np.random.default_rng(a.seed + 14))
        va = vip.strat(va, y, a.smoke_query, np.random.default_rng(a.seed + 15))
        test_rows = np.sort(np.random.default_rng(a.seed + 16).choice(test_rows, min(a.smoke_test, len(test_rows)), replace=False))
    assert len(test_rows) > 0
    assert not np.intersect1d(fi, np.concatenate([qs, qc, va])).size
    assert not np.intersect1d(qs, qc).size and not np.intersect1d(tr, va).size
    split_ids = {name: train_ids[rows].tolist() for name, rows in
                 (("inner_fit", fi), ("inner_search", qs), ("inner_confirmation", qc),
                  ("outer_train", tr), ("outer_validation", va))}
    split_ids["organizer_test"] = test_ids[test_rows].tolist()
    save_json(folder / "splits.json", {"run_sha256": run_hash, "ids": split_ids,
                                      "teacher_context_disjointness": "asserted by application_id and row index"})
    print(f"{'SMOKE ONLY ' if a.smoke else ''}s{a.seed} f{a.fold}: " +
          " ".join(f"{name}={len(rows)}" for name, rows in split_ids.items()), flush=True)
    model = None

    def stage(real, queries):
        # No outer rows (features OR labels) enter the inner stage's Predictor.
        cols = v.fm_select(X, y, real, 100)
        frames = [X.iloc[real][cols], Xt.iloc[test_rows][cols]]
        row_ids = list(train_ids[real]) + list(test_ids[test_rows])
        roles = ["real"] * len(real) + ["test"] * len(test_rows)
        query_positions = {}
        for name, rows in queries.items():
            query_positions[name] = np.arange(len(row_ids), len(row_ids) + len(rows))
            frames.append(X.iloc[rows][cols])
            row_ids.extend(train_ids[rows])
            roles.extend([name] * len(rows))
        F = pd.concat(frames, ignore_index=True)
        F = vip.prepare(F, np.arange(len(real)), "raw" if a.category_type == "numerical" else "categorical")
        labels = np.zeros(len(F), dtype=int)
        labels[:len(real)] = y[real]
        return {"F": F, "labels": labels, "real": np.arange(len(real)),
                "test": np.arange(len(real), len(real) + len(test_rows)), "queries": query_positions,
                "cols": cols, "ids": np.asarray(row_ids), "roles": np.asarray(roles)}

    def prediction(stg, name, context, query, labels, pseudo_probability=None):
        nonlocal model
        assert len(np.unique(labels[context])) == 2
        assert not np.intersect1d(context, query).size
        assert len(set(stg["ids"][context])) == len(context)
        assert not set(stg["ids"][context]).intersection(stg["ids"][query])
        real_rows = context[stg["roles"][context] == "real"]
        assert np.array_equal(real_rows, stg["real"]), "Real context must never be shrunk for SSL"
        pseudo = context[stg["roles"][context] == "test"]
        assert len(context) == len(real_rows) + len(pseudo) and len(pseudo) <= 10000
        fit = {"run_sha256": run_hash, "name": name, "columns": stg["cols"],
               "estimators": 4, "generator_seed": a.seed * 100 + a.fold,
               "real_context_ids": stg["ids"][real_rows].tolist(),
               "real_labels": labels[real_rows].tolist(), "pseudo_context_ids": stg["ids"][pseudo].tolist(),
               "pseudo_labels": labels[pseudo].tolist(), "query_ids": stg["ids"][query].tolist(),
               "query_roles": stg["roles"][query].tolist(), "disjoint_context_query": True,
               "all_real_context_retained": True}
        if pseudo_probability is not None:
            fit["pseudo_teacher_probability"] = pseudo_probability.tolist()
        fp, pp, cp = (folder / f"{name}_{suffix}" for suffix in ("fit.json", "predictions.parquet", "cache.json"))
        if cp.exists():
            cached = json.loads(cp.read_text())
            assert fp.exists() and pp.exists() and json.loads(fp.read_text()) == fit
            assert cached == {"fit_sha256": digest(fit), "predictions_sha256": file_hash(pp)}
            saved = pl.read_parquet(pp)
            assert saved["application_id"].to_list() == fit["query_ids"]
            p = saved["prediction"].to_numpy()
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            print(f"cached {name}: context={len(context)} query={len(query)}", flush=True)
            return p
        save_json(fp, fit)
        if model is None:
            model = sdm.models.KumoTabular(task="classification", size="large", device="cuda")
            model.eval()
        torch.cuda.reset_peak_memory_stats()
        predictor = vip.Predictor(stg["F"], labels, model)
        try:
            p = predictor.predict_cached(context, query, stg["cols"], 4, a.seed * 100 + a.fold, a.chunk)
        finally:
            del predictor
            gc.collect()
            torch.cuda.empty_cache()
        memory = {"max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        save_json(folder / f"{name}_memory.json", memory)
        save_frame(pp, pl.DataFrame({"application_id": stg["ids"][query], "role": stg["roles"][query], "prediction": p}))
        save_json(cp, {"fit_sha256": digest(fit), "predictions_sha256": file_hash(pp)})
        print(f"computed {name}: real={len(real_rows)} pseudo={len(pseudo)} query={len(query)} memory={memory}", flush=True)
        return p

    def teacher(stg, name):
        query = np.concatenate([stg["test"], *stg["queries"].values()])
        p = prediction(stg, name, stg["real"], query, stg["labels"])
        output = {"test": p[:len(stg["test"])]}
        start = len(stg["test"])
        for role, positions in stg["queries"].items():
            output[role] = p[start:start + len(positions)]
            start += len(positions)
        stg["teacher_predictions"] = output
        stg["teacher_name"] = name
        save_frame(folder / f"{name}_pseudo_candidates.parquet", pl.DataFrame({
            "application_id": test_ids[test_rows], "teacher_probability": output["test"],
            "pseudo_label": (output["test"] >= .5).astype(int),
            "confidence_group": group_ids(output["test"]),
            "eligible_02": (output["test"] <= .02) | (output["test"] >= .98),
            "eligible_05": (output["test"] <= .05) | (output["test"] >= .95)}))
        return output

    def student(stg, teacher_p, policy, role, name):
        chosen = pseudo_rows(teacher_p, policy, a.seed + a.fold + 97)
        pos = stg["test"][chosen]
        labels = stg["labels"].copy()
        labels[pos] = (teacher_p[chosen] >= .5).astype(int)
        save_frame(folder / f"{name}_pseudo.parquet", pl.DataFrame({
            "application_id": test_ids[test_rows[chosen]], "teacher_probability": teacher_p[chosen],
            "pseudo_label": labels[pos], "confidence_group": group_ids(teacher_p)[chosen]}))
        if not len(chosen):
            print(f"{name}: no eligible pseudo context; reuse noSSL", flush=True)
            p = stg["teacher_predictions"][role]
            save_json(folder / f"{name}_fit.json", {
                "run_sha256": run_hash, "teacher_fit_reference": f"{stg['teacher_name']}_fit.json",
                "reuse_reason": "empty pseudo selection; identical noSSL real context",
                "policy": policy, "pseudo_context_ids": [],
                "query_ids": stg["ids"][stg["queries"][role]].tolist(),
                "all_real_context_retained": True, "disjoint_context_query": True})
            save_frame(folder / f"{name}_predictions.parquet", pl.DataFrame({
                "application_id": stg["ids"][stg["queries"][role]], "prediction": p}))
            return p
        return prediction(stg, name, np.concatenate([stg["real"], pos]), stg["queries"][role], labels,
                          teacher_p[chosen])

    selection_path = folder / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        assert selection["run_sha256"] == run_hash
        print("frozen inner selection", selection["winner"], flush=True)
    else:
        inner = stage(fi, {"search": qs, "confirmation": qc})
        tp = teacher(inner, "inner_teacher")
        baseline = {"kind": "none", "cap": 0}
        search_rows = [{"name": "noSSL", "policy": baseline, **score(y[qs], tp["search"]), "n_pseudo": 0}]
        policies = [{"kind": "confidence", "threshold": threshold, "cap": cap} for threshold in (.02, .05) for cap in caps]
        policies.extend({"kind": "asymmetric", "negative_threshold": neg, "positive_threshold": pos, "cap": cap}
                        for neg in (.02, .05) for pos in positive_thresholds for cap in caps)
        masks = []
        if a.group_coalitions:
            # VIP-COP-inspired group coalitions: Ridge predicts SEARCH utility of
            # including confidence shells; this is not the original VIP-COP algorithm.
            rng = np.random.default_rng(a.seed + a.fold + 201)
            masks = [np.array([(integer >> bit) & 1 for bit in range(4)])
                     for integer in rng.choice(np.arange(1, 15), a.group_coalitions, replace=False)]
            policies.extend({"kind": "groups", "groups": np.flatnonzero(mask).tolist(), "cap": max(caps)} for mask in masks)
        seen = set()
        for policy in policies:
            name = policy_name(policy)
            if name in seen:
                continue
            seen.add(name)
            n = len(pseudo_rows(tp["test"], policy, a.seed + a.fold + 97))
            p = tp["search"] if n == 0 else student(inner, tp["test"], policy, "search", f"inner_search_{name}")
            chosen = pseudo_rows(tp["test"], policy, a.seed + a.fold + 97)
            row = {"name": name, "policy": policy, **score(y[qs], p), "n_pseudo": n,
                   "pseudo_positive": int(np.sum(tp["test"][chosen] >= .5)),
                   "pseudo_negative": int(np.sum(tp["test"][chosen] < .5))}
            search_rows.append(row)
            print("search", row, flush=True)
            save_json(folder / "search.json", search_rows)
        if masks:
            responses = [next(row["ks"] for row in search_rows if row["name"] == policy_name(
                {"kind": "groups", "groups": np.flatnonzero(mask).tolist(), "cap": max(caps)})) for mask in masks]
            ridge = Ridge(alpha=1.).fit(np.asarray(masks), np.asarray(responses))
            groups = np.flatnonzero(ridge.coef_ > 0).tolist()
            save_json(folder / "group_surrogate.json", {"groups": ["p<=.02", ".02<p<=.05", ".95<=p<.98", "p>=.98"],
                "coalitions": [mask.tolist() for mask in masks], "search_ks": responses,
                "coefficients": ridge.coef_.tolist(), "selected_positive_groups": groups,
                "interpretation": "search-only association; not pseudo-label correctness or causal attribution"})
            if groups:
                policy = {"kind": "groups", "groups": groups, "cap": max(caps)}
                name = policy_name(policy)
                if name not in seen:
                    n = len(pseudo_rows(tp["test"], policy, a.seed + a.fold + 97))
                    p = tp["search"] if n == 0 else student(inner, tp["test"], policy, "search", f"inner_search_{name}")
                    search_rows.append({"name": name, "policy": policy, **score(y[qs], p), "n_pseudo": n})
        # Stable ties prefer baseline; confirm just one SEARCH-selected challenger.
        top = max(search_rows, key=lambda row: row["ks"])
        confirm_rows = [{"name": "noSSL", "policy": baseline, **score(y[qc], tp["confirmation"])}]
        winner, reason = baseline, "no inner-search challenger exceeds the predetermined KS gain gate"
        if top["name"] != "noSSL" and top["ks"] - search_rows[0]["ks"] > a.min_inner_gain:
            p = student(inner, tp["test"], top["policy"], "confirmation", "inner_confirmation_" + top["name"])
            confirm_rows.append({"name": top["name"], "policy": top["policy"], **score(y[qc], p)})
            gain = confirm_rows[-1]["ks"] - confirm_rows[0]["ks"]
            if gain > a.min_inner_gain:
                winner, reason = top["policy"], "search and disjoint confirmation exceed the predetermined KS gain gate"
            else:
                reason = "fallback noSSL: disjoint confirmation does not exceed the predetermined KS gain gate"
        selection = {"run_sha256": run_hash, "winner": policy_name(winner), "policy": winner,
                     "reason": reason, "min_inner_gain": a.min_inner_gain, "search": search_rows,
                     "confirmation": confirm_rows, "inner_columns": inner["cols"],
                     "outer_seen": False, "pseudo_cap_fixed_before_outer": winner["cap"],
                     "outer_feature_policy": "refit fm_select and categorical vocabulary on outer-train only"}
        save_json(folder / "search.json", search_rows)
        save_json(selection_path, selection)
        del inner
        print("selection frozen before outer", selection["winner"], reason, flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, key in zip(axes, ("search", "confirmation")):
        rows = selection[key]
        axis.bar(np.arange(len(rows)), [row["ks"] for row in rows])
        axis.set_xticks(np.arange(len(rows)), [row["name"] for row in rows], rotation=90, fontsize=7)
        axis.set(ylabel="KS", title=f"{'SMOKE ' if a.smoke else ''}Inner {key}; no outer feedback")
    fig.tight_layout()
    plot_path = OUT / f"156_ssl_{a.out.name}_s{a.seed}_f{a.fold}_{run_hash[:8]}.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    if a.evaluate:
        # The only entrance to outer feature materialization and outer teacher fitting.
        assert selection_path.exists() and json.loads(selection_path.read_text()) == selection
        outer = stage(tr, {"outer": va})
        tp = teacher(outer, "outer_teacher")
        winner = selection["policy"]
        p = tp["outer"] if winner["kind"] == "none" else student(outer, tp["test"], winner, "outer", "outer_selected")
        outer_path = a.out / f"s{a.seed}_f{a.fold}_outer.parquet"
        save_frame(outer_path, pl.DataFrame({"application_id": train_ids[va], "target": y[va],
                   "baseline": tp["outer"], "candidate": p}))
        result = {"run_sha256": run_hash, "winner": selection["winner"], "smoke_only": a.smoke,
                  "baseline": score(y[va], tp["outer"]), "selected": score(y[va], p),
                  "selection_sha256": file_hash(selection_path), "outer_rows": len(va),
                  "outer_predictions": str(outer_path), "outer_predictions_sha256": file_hash(outer_path),
                  "evaluation_role": "untouched outer-only; never used for pseudo selection"}
        result["delta_ks"] = result["selected"]["ks"] - result["baseline"]["ks"]
        save_json(folder / "outer_metrics.json", result)
        print("outer FROZEN evaluation", json.dumps(result), flush=True)
    else:
        print("Outer untouched; rerun identical command with --evaluate to unlock frozen evaluation", flush=True)
    print("Artifacts", folder, "plot", plot_path, flush=True)


if __name__ == "__main__":
    main()
