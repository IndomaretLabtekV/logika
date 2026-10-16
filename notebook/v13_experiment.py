"""Fixed additive-versus-pairwise EBM experiment; embedded in the v10 data harness."""
BASE = dict(max_bins=128, max_interaction_bins=16, outer_bags=4, inner_bags=0,
            learning_rate=.03, max_rounds=10000, early_stopping_rounds=100,
            validation_size=.15, min_samples_leaf=50, smoothing_rounds=100,
            interaction_smoothing_rounds=100, n_jobs=4, random_state=42)
CANDIDATES = {"control": dict(interactions=0), "tuned": dict(interactions=20)}


def run(X, y, test, library, out):
    assert Path("/kaggle/input").exists(), "Real fitting is restricted to Kaggle"
    X, test = X.copy(), test.copy()
    for c in X.select_dtypes("category").columns:
        X[c], test[c] = X[c].astype(object), test[c].astype(object)
    oof = {n: np.full(len(y), np.nan) for n in CANDIDATES}
    test_pred = {n: np.zeros(len(test)) for n in CANDIDATES}
    folds = np.full(len(y), -1)
    stats = []
    deadline = time.monotonic() + 11 * 3600
    for fold, (train, valid) in enumerate(splits(y, 5, 42)):
        assert not np.intersect1d(train, valid).size
        for name, changes in CANDIDATES.items():
            if time.monotonic() > deadline:
                raise TimeoutError("Partial fold outputs saved; no incomplete submission emitted")
            started = time.monotonic()
            model = ExplainableBoostingClassifier(**dict(BASE, **changes))
            model.fit(X.iloc[train], y[train])
            pv = model.predict_proba(X.iloc[valid])[:, 1]
            pt = model.predict_proba(test)[:, 1]
            assert np.array_equal(model.classes_, [0, 1])
            assert np.isfinite(pv).all() and np.isfinite(pt).all()
            assert ((pv >= 0) & (pv <= 1)).all() and ((pt >= 0) & (pt <= 1)).all()
            oof[name][valid] = pv
            test_pred[name] += pt / 5
            np.savez(out / f"{name}_fold{fold}.npz", rows=valid, prediction=pv, test=pt)
            row = dict(fold=fold, candidate=name, iterations=np.asarray(model.best_iteration_).tolist(),
                       terms=len(model.term_names_), seconds=time.monotonic()-started,
                       outer_ks=ks(y[valid], pv))
            stats.append(row)
            (out / "fit_stats.json").write_text(json.dumps(stats, indent=2))
            pd.DataFrame(dict(term=model.term_names_, importance=model.term_importances())).to_csv(
                out / f"importance_{name}_fold{fold}.csv", index=False)
            print(json.dumps(row), flush=True)
            del model
        folds[valid] = fold
    assert (folds >= 0).all() and all(np.isfinite(p).all() for p in oof.values())
    return oof, test_pred, folds
