"""Lock a refinement recipe before the larger-fold training run."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from optimize_ks import ROOT,OUT,metrics

FINAL=OUT/'tenfold'


def main():
    FINAL.mkdir(parents=True,exist_ok=True)
    directories=[OUT/'refinement',OUT/'refinement_balanced']
    all_summaries=[]
    definitions={}
    oof=None
    test=None
    for directory in directories:
        summary=json.loads((directory/'summary.json').read_text())
        specs=json.loads((directory/'definitions.json').read_text())
        for row in summary:
            if row['name']=='public_054710':continue
            prefix,tree_count=row['name'].rsplit('_',1)
            spec=next(s for s in specs if s['name']==prefix)
            definitions[row['name']]=dict(spec,iterations=int(tree_count),source=str(directory.relative_to(ROOT)))
            all_summaries.append(row)
        part=pd.read_parquet(directory/'oof.parquet')
        predicted=pd.read_parquet(directory/'test_predictions.parquet')
        if oof is None:oof=part;test=predicted
        else:
            assert part.application_id.equals(oof.application_id) and part.target.equals(oof.target)
            assert predicted.application_id.equals(test.application_id)
            oof=pd.concat([oof,part.drop(columns=['application_id','target','public_054710'])],axis=1)
            test=pd.concat([test,predicted.drop(columns=['application_id','public_054710'])],axis=1)
    # Keep the best fixed tree count per configuration, then at most three new members.
    winners={}
    for row in all_summaries:
        prefix=row['name'].rsplit('_',1)[0]
        if prefix not in winners or row['mean_fold_ks']>winners[prefix]['mean_fold_ks']:
            winners[prefix]=row
    pool=[r['name'] for r in sorted(winners.values(),key=lambda r:r['mean_fold_ks'],reverse=True)[:3]]
    pool.append('public_054710')
    recipes=[{n:1.0} for n in pool]
    for a,b in itertools.combinations(pool,2):
        recipes.extend([{a:w,b:1-w} for w in [.25,.5,.75]])
    for members in itertools.combinations(pool,3):
        recipes.append({n:1/3 for n in members})
    y=oof.target.astype(int)
    folds=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=8247).split(oof,y))
    results=[]
    for recipe in recipes:
        p=sum(weight*oof[name] for name,weight in recipe.items())
        per_fold=[metrics(y.iloc[vi],p.iloc[vi])['ks'] for _,vi in folds]
        results.append({'recipe':recipe,**metrics(y,p),'mean_fold_ks':float(np.mean(per_fold)),'fold_ks':per_fold})
    results.sort(key=lambda r:r['mean_fold_ks'],reverse=True)
    baseline=next(r for r in results if r['recipe']=={'public_054710':1.0})
    # Require both aggregate OOF KS and average fold KS to improve the champion.
    eligible=[r for r in results if r['ks']>baseline['ks'] and r['mean_fold_ks']>baseline['mean_fold_ks']]
    if not eligible:
        raise RuntimeError('No refinement improves both KS measures; preserve the champion.')
    locked=eligible[0]
    config={'recipe':locked['recipe'],'selection':locked,'champion':baseline,
            'definitions':{n:definitions[n] for n in locked['recipe'] if n!='public_054710'},
            'selection_fold_seed':8247,'final_folds':10,'final_fold_seed':9173,
            'final_feature_file':'application_features_v3.parquet',
            'new_public_score':None,'champion_public_score':.54710,
            'caveat':'Tree counts and weights selected using these validation folds; local estimates can contain selection optimism.'}
    (FINAL/'locked_recipe.json').write_text(json.dumps(config,indent=2))
    (FINAL/'recipe_search.json').write_text(json.dumps(results,indent=2))
    oof.to_parquet(FINAL/'selection_oof.parquet',index=False)
    test.to_parquet(FINAL/'selection_test_predictions.parquet',index=False)
    print('Locked next recipe:',json.dumps(locked),flush=True)
    print('Current champion on the same folds:',json.dumps(baseline),flush=True)


if __name__=='__main__':main()
