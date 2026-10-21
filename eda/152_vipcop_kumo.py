"""VIP-COp-inspired Kumo large feature search; validation_v2 outer folds remain evaluation-only.

Adaptation, not a paper reproduction: fixed-size coalition ridge surrogate, standardized
softmax temperatures, serial GPU calls. Search AUROC on half the inner-stop rows; select
among top-VIP contexts and the untouched baseline by KS on the other half with full
inner-fit context/e4. No test distribution or outer label enters selection.
Run from eda: ../.venv/bin/python 152_vipcop_kumo.py --out ../results/v46/vip_raw
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
from scipy.special import softmax
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import sdm
import valid2 as v
from common import OUT, PQ, features


def strat(pool, y, n, rng):
    n = min(n, len(pool))
    n1 = int(round(n * y[pool].mean()))
    return np.sort(np.concatenate([rng.choice(pool[y[pool] == c], k, replace=False)
                                  for c, k in ((0, n - n1), (1, n1))]))


def prepare(X, fit_rows, mode):
    """Row-wise transforms only; categorical vocabulary comes from fit rows, never queries."""
    F = X.copy()
    for c in v.CAT_COLS:
        cats = sorted(X.iloc[fit_rows][c].dropna().unique())
        F[c] = pd.Categorical(X[c], categories=cats)
        if mode != "categorical":
            F[c] = F[c].cat.codes.astype(float).replace(-1, np.nan)
    if mode == "log":
        for c in F.columns:
            if c not in v.CAT_COLS:
                x = F[c].to_numpy(dtype=float)
                F[c] = np.sign(x) * np.log1p(np.abs(x))
    return F.reset_index(drop=True)


class Predictor:
    def __init__(self, F, y, model):
        self.dev = torch.device("cuda")
        self.model = model
        stypes = {c: "categorical" if isinstance(F[c].dtype, pd.CategoricalDtype) else "numerical"
                  for c in F.columns}
        self.T = sdm.TableTensor.from_pandas(df=F, stypes=sdm.infer_stypes(F, overrides=stypes), device=self.dev)
        labels = pd.DataFrame({"target": y.astype(str)})
        self.Y = sdm.TableTensor.from_pandas(df=labels, stypes=sdm.infer_stypes(labels, overrides={"target": "categorical"}),
                                           device=self.dev)

    def predict(self, ctx, query, cols, n_est=1, seed=0, chunk=2000):
        ci = torch.as_tensor(ctx, device=self.dev)
        xc, yc = self.T[ci].select_columns(cols), self.Y[ci]
        g = torch.Generator(device=self.dev).manual_seed(seed)
        out = []
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            for start in range(0, len(query), chunk):
                qi = torch.as_tensor(query[start:start + chunk], device=self.dev)
                r = self.model(x_context=xc, y_context=yc, x_query=self.T[qi].select_columns(cols),
                               num_estimators=n_est, generator=g)
                p = r.to_pandas()["1"].to_numpy()
                assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                out.append(p)
        return np.concatenate(out)

    def predict_cached(self, ctx, query, cols, n_est=1, seed=0, chunk=500):
        """Reuse a fixed context/estimator draw across query chunks; always clear it."""
        ci = torch.as_tensor(ctx, device=self.dev)
        try:
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                self.model.fit(self.T[ci].select_columns(cols), self.Y[ci], num_estimators=n_est,
                               generator=torch.Generator(device=self.dev).manual_seed(seed))
                out = []
                for start in range(0, len(query), chunk):
                    qi = torch.as_tensor(query[start:start + chunk], device=self.dev)
                    p = self.model.predict(self.T[qi].select_columns(cols)).to_pandas()["1"].to_numpy()
                    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                    out.append(p)
                return np.concatenate(out)
        finally:
            self.model.clear()


def plot_search(history, phi, nums, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    h = pd.DataFrame(history)
    if len(h):
        for tau, g in h.groupby("tau"):
            axes[0].plot(g["round"], g["best_auc"], label=f"tau={tau:g}")
        axes[0].legend()
    axes[0].set(xlabel="Round", ylabel="Best search AUROC", title="Inner search only")
    ix = np.argsort(phi)[-15:]
    axes[1].barh(np.array(nums)[ix], phi[ix])
    axes[1].set(title="Selected chain surrogate coefficients", xlabel="Ridge coefficient (not causal importance)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_state(path, **state):
    """Atomic durable round checkpoint; interruption loses at most one round."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **state)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--taus", default="1,0.5,0.25")
    ap.add_argument("--d", type=int, default=100)
    ap.add_argument("--nctx", type=int, default=20000)
    ap.add_argument("--nq", type=int, default=4000)
    ap.add_argument("--confirm-nctx", type=int, default=56000)
    ap.add_argument("--confirm-nq", type=int, default=4000)
    ap.add_argument("--confirm-est", type=int, default=4)
    ap.add_argument("--search-est", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gpu-memory-fraction", type=float, default=.9)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--mode", choices=["raw", "log", "categorical"], default="raw")
    ap.add_argument("--evaluate", action="store_true", help="Score outer only after inner selection is saved")
    ap.add_argument("--chunk", type=int, default=500)
    a = ap.parse_args()
    assert a.rounds > 0 and a.batch > 0
    assert 1 <= a.threads <= 8 and 0 < a.gpu_memory_fraction < 1
    torch.set_num_threads(a.threads)
    torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction)
    a.out.mkdir(parents=True, exist_ok=True)
    f = features()
    X, _, y, ids, _ = v.frames(f)
    assert ids.n_unique() == len(y)
    nums = [c for c in X.columns if c not in v.CAT_COLS]
    assert 0 < a.d <= len(nums)
    config = {**vars(a), "out": str(a.out)}
    manifest = {"config": config, "packages": {p: importlib.metadata.version(p) for p in
                ("numpy", "pandas", "polars", "torch", "structured-data-models", "scikit-learn")},
                "features_sha256": hashlib.sha256((PQ / "features.parquet").read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "method": "VIP-COp-inspired ridge; fixed cached estimator draw; disjoint inner search/confirmation; per-round resume"}
    mp = a.out / "manifest.json"
    if mp.exists():
        previous = json.loads(mp.read_text())
        # --evaluate may be added after inspecting saved inner results, not after inspecting outer scores.
        assert {k: val for k, val in previous["config"].items() if k != "evaluate"} == {
            k: val for k, val in config.items() if k != "evaluate"}, "Use a new output directory for a new experiment"
        assert previous["features_sha256"] == manifest["features_sha256"]
        assert previous["source_sha256"] == manifest["source_sha256"], "Source changed; use a new run directory"
        assert previous["packages"] == manifest["packages"], "Package versions changed"
    else:
        mp.write_text(json.dumps(manifest, indent=2))
    model = sdm.models.KumoTabular(task="classification", size="large", device="cuda")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    for seed in map(int, a.seeds.split(",")):
        for fold in map(int, a.folds.split(",")):
            t0 = time.time()
            prefix = a.out / f"s{seed}_f{fold}"
            selection_path = prefix.with_name(prefix.name + "_selection.json")
            tr, va = v.outer(y, seed)[fold]
            fi, st = v.inner(y[tr], seed, fold)
            fi, st = tr[fi], tr[st]
            qs, qc = train_test_split(st, train_size=min(a.nq, len(st) // 2), stratify=y[st], random_state=seed + fold)
            qs, qc = np.sort(qs), np.sort(qc)
            assert not np.intersect1d(fi, st).size and not np.intersect1d(tr, va).size
            assert not np.intersect1d(qs, qc).size
            qc = strat(qc, y, a.confirm_nq, np.random.default_rng(seed + fold + 3))
            cc = strat(fi, y, a.confirm_nctx, np.random.default_rng(seed + fold + 4))
            ctx = strat(fi, y, a.nctx, np.random.default_rng(seed + fold + 1))
            pr = Predictor(prepare(X, fi, a.mode), y, model)
            if selection_path.exists():
                selection = json.loads(selection_path.read_text())
                print(f"resume saved inner choice s{seed} f{fold}: {selection['winner']}", flush=True)
            else:
                print(f"s{seed} f{fold} inner fit={len(fi)} search={len(qs)} confirm={len(qc)} outer={len(va)} "
                      f"context={len(ctx)} mode={a.mode}", flush=True)
                fm = v.fm_select(X, y, fi, a.d)
                def score(idx):
                    # Fixed source-column order avoids feature permutation masquerading as feature selection.
                    cols = [nums[i] for i in sorted(idx)] + v.CAT_COLS
                    return float(roc_auc_score(y[qs], pr.predict_cached(ctx, qs, cols, n_est=a.search_est, chunk=a.chunk)))
                fm_idx = [nums.index(c) for c in fm if c not in v.CAT_COLS]
                base_auc = score(fm_idx)
                print(f"inner-fit fm_select canonical search AUROC={base_auc:.6f}", flush=True)
                candidates = [("fm_legacy", fm), ("fm_canonical", sorted(fm[:-len(v.CAT_COLS)]) + v.CAT_COLS)]
                rng = np.random.default_rng(seed * 100 + fold)
                history, phis = [], {}
                coalitions, responses = [], []
                for tau in map(float, a.taus.split(",")):
                    assert tau > 0
                    state_path = prefix.with_name(prefix.name + f"_tau{tau:g}_state.npz")
                    if state_path.exists():
                        with np.load(state_path, allow_pickle=False) as state:
                            C, P = list(state["masks"].astype(float)), state["auc"].tolist()
                            phi = state["phi"]
                            first_round = int(state["round"])
                            rng.bit_generator.state = json.loads(state["rng_json"].item())
                        assert len(P) == len(C) == first_round * a.batch
                        assert np.asarray(C).shape == (len(P), len(nums))
                        print(f"resume tau={tau:g} after round={first_round}", flush=True)
                        for t in range(first_round):
                            end = (t + 1) * a.batch
                            history.append(dict(tau=tau, round=t + 1, mean_auc=float(np.mean(P[end-a.batch:end])),
                                                best_auc=float(max(P[:end]))))
                    else:
                        C, P, first_round = [], [], 0
                        phi = np.zeros(len(nums))
                    for t in range(first_round, a.rounds):
                        z = (phi - phi.mean()) / max(phi.std(), 1e-12)
                        prob = softmax(z / tau)
                        for _ in range(a.batch):
                            idx = rng.choice(len(nums), a.d, replace=False, p=prob)
                            mask = np.zeros(len(nums))
                            mask[idx] = 1
                            C.append(mask)
                            P.append(score(idx))
                        phi = Ridge(alpha=1.0).fit(np.asarray(C), np.asarray(P)).coef_
                        history.append(dict(tau=tau, round=t + 1, mean_auc=float(np.mean(P[-a.batch:])),
                                            best_auc=float(max(P))))
                        save_state(state_path, masks=np.asarray(C, dtype=np.uint8), auc=np.asarray(P), phi=phi,
                                   round=np.asarray(t + 1), rng_json=np.asarray(json.dumps(rng.bit_generator.state)))
                        pd.DataFrame(history).to_csv(prefix.with_name(prefix.name + "_search.csv"), index=False)
                        print(f"tau={tau:g} round={t+1} best={max(P):.6f} mean={np.mean(P[-a.batch:]):.6f} "
                              f"elapsed={time.time()-t0:.0f}s gpu_peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB "
                              f"rss_peak={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.2f}GiB", flush=True)
                    top = np.argsort(-phi)[:a.d]
                    name = f"vip_tau{tau:g}"
                    candidates.append((name, [nums[i] for i in sorted(top)] + v.CAT_COLS))
                    phis[name] = phi.copy()
                    coalitions.extend(C)
                    responses.extend(P)
                    pd.DataFrame(history).to_csv(prefix.with_name(prefix.name + "_search.csv"), index=False)
                    np.savez_compressed(prefix.with_name(prefix.name + "_coalitions.npz"),
                                        masks=np.asarray(coalitions, dtype=np.uint8), auc=np.asarray(responses))
                confirm = []
                for name, cols in candidates:
                    p = pr.predict_cached(cc, qc, cols, n_est=a.confirm_est, seed=seed * 100, chunk=a.chunk)
                    row = {"name": name, "ks": v.ks(y[qc], p), "auc": float(roc_auc_score(y[qc], p)), "cols": cols}
                    confirm.append(row)
                    print(f"confirm {name}: KS={row['ks']:.6f} AUC={row['auc']:.6f} elapsed={time.time()-t0:.0f}s", flush=True)
                # Stable tie break preserves the pre-existing baseline. Selection is entirely inside outer-train.
                winner = max(confirm, key=lambda row: row["ks"])
                selection = {"winner": winner["name"], "cols": winner["cols"], "confirmation": confirm,
                             "search_fm_auc": base_auc, "seed": seed, "fold": fold, "mode": a.mode,
                             "fit_ids_sha256": hashlib.sha256(ids[fi].to_numpy().tobytes()).hexdigest(),
                             "search_ids": ids[qs].to_list(), "confirmation_ids": ids[qc].to_list(),
                             "seconds": time.time() - t0}
                selection_path.write_text(json.dumps(selection, indent=2))
                phi = phis.get(winner["name"], next(iter(phis.values())))
                pd.DataFrame({"feature": nums, "phi": phi}).to_csv(prefix.with_name(prefix.name + "_phi.csv"), index=False)
                plot_search(history, phi, nums, OUT / f"152_{a.out.name}_s{seed}_f{fold}.png")
                print(f"saved inner winner={winner['name']} before any outer evaluation", flush=True)
            del pr
            torch.cuda.empty_cache()
            if a.evaluate:
                op = prefix.with_name(prefix.name + "_outer.parquet")
                if op.exists():
                    print(f"skip existing outer {op}", flush=True)
                    continue
                # Policy frozen inner; refit the baseline feature ranking only on outer-train.
                pr = Predictor(prepare(X, tr, a.mode), y, model)
                base_cols = v.fm_select(X, y, tr, a.d)
                baseline = pr.predict_cached(tr, va, base_cols, n_est=4, seed=seed * 100, chunk=a.chunk)
                cols = base_cols if selection["winner"] == "fm_legacy" else (
                    sorted(base_cols[:-len(v.CAT_COLS)]) + v.CAT_COLS if selection["winner"] == "fm_canonical" else selection["cols"])
                p = baseline if cols == base_cols else pr.predict_cached(
                    tr, va, cols, n_est=4, seed=seed * 100, chunk=a.chunk)
                pl.DataFrame({"application_id": ids[va], "target": y[va], "candidate": p, "baseline": baseline}).write_parquet(op)
                print(f"outer s{seed} f{fold} KS={v.ks(y[va], p):.6f} baseline={v.ks(y[va], baseline):.6f} "
                      f"winner={selection['winner']} peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
                del pr
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
