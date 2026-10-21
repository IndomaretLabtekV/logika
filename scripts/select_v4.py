"""Lock a coarse version-4 blend using three-fold OOF, before final tree training."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from optimize_ks import ROOT, OUT, metrics

DESTINATION = OUT / 'v4'


def main():
    oof = pd.read_parquet(DESTINATION / 'selection_oof.parquet')
    spline = pd.read_parquet(DESTINATION / 'spline_oof.parquet')
    assert oof.application_id.equals(spline.application_id)
    assert oof.target.astype(int).equals(spline.target.astype(int))
    assert oof.notna().all().all()
    oof['spline'] = spline.spline
    y = oof.target.astype(int)
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=8247).split(oof, y))
    baseline = metrics(y, oof.public_055845)
    baseline_folds = [metrics(y.iloc[vi], oof.public_055845.iloc[vi])['ks'] for _, vi in folds]
    baseline['fold_ks'] = baseline_folds
    baseline['mean_fold_ks'] = float(np.mean(baseline_folds))
    tree_names = list(oof.columns[3:-1])
    recipes = [{'public_055845': 1.0}]
    for name in tree_names + ['spline']:
        for weight in [.25, .5, .75]:
            recipes.append({'public_055845': 1-weight, name: weight})
    for name in tree_names:
        for weights in [[.5, .25, .25], [1/3, 1/3, 1/3], [.25, .5, .25], [.25, .25, .5]]:
            recipes.append(dict(zip(['public_055845', name, 'spline'], weights)))
    results = []
    for recipe in recipes:
        p = sum(w * oof[n] for n, w in recipe.items())
        per_fold = [metrics(y.iloc[vi], p.iloc[vi])['ks'] for _, vi in folds]
        gains = np.array(per_fold) - baseline_folds
        row = {'recipe': recipe, **metrics(y, p), 'fold_ks': per_fold,
               'mean_fold_ks': float(np.mean(per_fold)), 'fold_gains': gains.tolist()}
        row['eligible'] = bool(row['ks'] >= baseline['ks'] + .002 and
                               row['mean_fold_ks'] >= baseline['mean_fold_ks'] + .001 and
                               (gains > 0).sum() >= 2 and gains.min() >= -.0025)
        results.append(row)
    results.sort(key=lambda r: (r['mean_fold_ks'], r['ks']), reverse=True)
    eligible = [r for r in results if r['eligible']]
    if not eligible:
        raise RuntimeError('No candidate clears the improvement gates; keep the public checkpoint.')
    selected = eligible[0]
    config = {'recipe': selected['recipe'], 'selection': selected, 'baseline': baseline,
              'selection_fold_seed': 8247, 'final_fold_seed': 9173, 'final_folds': 10,
              'baseline_public_score': .55845, 'new_public_score': None,
              'baseline_file': 'submission_v3.csv',
              'baseline_sha256': 'bfd1b620ddaa85cd45ea2e8ce2c4f2ce3bcab660f09657dfbaae74aaa59525db',
              'rule': 'Coarse weights; at least .002 pooled KS and .001 mean fold KS gain; at least two fold wins; worst fold loss at most .0025.',
              'caveat': 'Feature, tree-count and blend selection reuse training labels. Validation results contain selection optimism and do not guarantee leaderboard scores.'}
    (DESTINATION / 'locked_recipe.json').write_text(json.dumps(config, indent=2))
    (DESTINATION / 'recipe_search.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({'baseline': baseline, 'selected': selected, 'top5': results[:5]}), flush=True)


if __name__ == '__main__':
    main()
