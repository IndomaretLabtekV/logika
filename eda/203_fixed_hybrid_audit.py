"""Evaluate one fixed v31 component replacement; never search blend weights.

The incremental policy is documented before this hybrid evaluation. This does
not change155's stronger breakthrough criterion. All15 fixed Kumo fits and
their complete audit are required before any paired calculation.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import valid2 as v
from common import ROOT,PQ,OUT

spec=importlib.util.spec_from_file_location('hybrid_export',Path(__file__).with_name('193_fixed_union_test.py'))
export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)
spec=importlib.util.spec_from_file_location('hybrid_eval',Path(__file__).with_name('155_kumo_nested_eval.py'))
evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
sha=export.sha
POLICY=ROOT/'docs/integration/submission_incremental_21okt.md'


def read(path):return json.loads(Path(path).read_text())


def inputs(root):
    plan=read(root/'validation_plan.json');report_path=OUT/f"155_{plan['tag']}.json";report=read(report_path)
    audit_path=OUT/f'192_{root.name}_provenance_complete.json';audit=read(audit_path)
    assert read(root/'validation_progress.json')['status']=='complete'
    assert report['complete_5fold_3seed'] and report['rows']==70000
    assert report['seeds']==[42,7,2026] and report['folds']==list(range(5))
    assert audit['complete'] and audit['completed_audits']==15 and all(r['passed'] for r in audit['fits'])
    assert audit['audit_source_sha256']==sha(ROOT/'eda/192_fixed_union_provenance.py')
    fits=export.validate_fit_records(root,report,audit)
    identity={'policy_sha256':sha(POLICY),'source_sha256':sha(__file__),'features_sha256':sha(PQ/'features.parquet'),
              'report_sha256':sha(report_path),'audit_sha256':sha(audit_path),'validation_plan_sha256':sha(root/'validation_plan.json'),
              'fits':fits,'historical_identity':export.historical_identity()}
    assert report['features_sha256']==identity['features_sha256']
    return identity


def vectors(root):
    canonical=pl.read_parquet(PQ/'features.parquet',columns=['application_id','target','is_train']).filter(pl.col('is_train')==1).sort('application_id')
    ids=canonical['application_id'];y=canonical['target'].to_numpy().astype(int)
    p,ps,_=evaluation.collect(root/'fixed_union_rank',ids,y,[42,7,2026],list(range(5)),'candidate')
    b,bs,_=evaluation.collect(root/'fixed_union_rank',ids,y,[42,7,2026],list(range(5)),'baseline')
    def load(relative):
        p=evaluation.read_prediction(ROOT/'results'/relative,ids,y);assert np.isfinite(p).all();return p
    # v31 assigns1/18 to each tree: no second tree-family rank transform.
    trees=np.mean([evaluation.rk(load(f'v22/v22/gbdt/oof_{m}.parquet')) for m in evaluation.TREES],axis=0)
    c=evaluation.rk(load('v22/v22/fm/oof_causilo.parquet'));k=evaluation.rk(load('v28m/oof_kumo_large_c56000.parquet'))
    return ids,y,(trees+c+evaluation.rk(p))/3,(trees+c+evaluation.rk(b))/3,(trees+c+k)/3,ps,bs,trees,c


def compute(root):
    identity=inputs(root);ids,y,p,b,h,ps,bs,t,c=vectors(root)
    masks=[np.isin(np.arange(len(y)),va) for _,va in v.outer(y,42)]
    comparisons={}
    for name,reference in (('v31_actual_recipe',h),('matched_primary_hybrid',b)):
        comparisons[name]=evaluation.compare(y,p,reference,1000,masks)
        print(name,json.dumps(comparisons[name]),flush=True)
    seeds=[]
    for seed,a,base in zip((42,7,2026),ps,bs):
        hp=(t+c+evaluation.rk(a))/3;hb=(t+c+evaluation.rk(base))/3
        seeds.append({'seed':seed,'candidate_ks':v.ks(y,hp),'matched_ks':v.ks(y,hb),'delta':v.ks(y,hp)-v.ks(y,hb)})
    eligible=all(r['delta']>0 and r['fold_wins']>=3 and r['p_gt_zero']>=.60 for r in comparisons.values())
    result={'identity':identity,'comparisons':comparisons,'seed_hybrid_diagnostics':seeds,'incremental_eligible':eligible,
            'breakthrough':False,'submission_performed':False,
            'recipe':'v31 fixed: six trees1/18 each, Causilo seed42 1/3, fixed-union Kumo3seed1/3',
            'bootstrap':'1000 paired customer-row resamples after averaging seeds; RNG0; no seed-row duplication',
            'limits':['Incremental criterion differs from the original breakthrough gate; no0.570 promise.',
                      'Input family was developed on inner seed42fold0; hybrid policy documented after partial Kumo outer evidence.',
                      'Legacy tree/Causilo components retain their documented feature-reference/provenance limitations.',
                      'Conditional bootstrap does not correct repeated research selection or infer private labels.']}
    return result,(ids,y,p,b,h)


def require_incremental(root):
    path=OUT/f'203_{root.name}_fixed_hybrid.json';reported=read(path)
    identity=inputs(root)
    if reported['identity']!=identity:raise ValueError('Hybrid evidence does not match current inputs')
    binding_path=root/'hybrid_release_binding.json'
    request={'report_sha256':sha(path),'identity':identity}
    binding=read(binding_path) if binding_path.exists() else None
    if binding is None or binding['request']!=request:
        actual,_=compute(root)
        if reported!=actual:raise ValueError('Hybrid statistics do not match the fixed recipe')
        binding={'request':request,'verified':True,'eligible':actual['incremental_eligible']}
        export.save(binding_path,binding)
    if binding['verified'] is not True or binding['eligible'] is not True or reported['incremental_eligible'] is not True:
        raise ValueError('Fixed hybrid does not meet the documented incremental criterion')
    return {'hybrid_report_sha256':sha(path),'hybrid_binding_sha256':sha(binding_path),'policy_sha256':identity['policy_sha256']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',type=Path,default=ROOT/'results/v56');a=ap.parse_args();root=a.run_root.resolve()
    result,(ids,y,p,b,h)=compute(root)
    pl.DataFrame({'application_id':ids,'target':y,'candidate':p,'matched_primary_hybrid':b,'v31_actual_recipe':h}).write_parquet(OUT/f'203_{root.name}_fixed_hybrid_oof.parquet')
    export.save(OUT/f'203_{root.name}_fixed_hybrid.json',result)
    names=list(result['comparisons']);fig,ax=plt.subplots(figsize=(8,3));ax.barh(names,[result['comparisons'][n]['delta'] for n in names],xerr=[result['comparisons'][n]['bootstrap_sd'] for n in names]);ax.axvline(0,color='black');ax.set(xlabel='Paired KS delta ± bootstrap SD',title='One prescribed component replacement; no weight search');fig.tight_layout();fig.savefig(OUT/f'203_{root.name}_fixed_hybrid.png',dpi=130);plt.close(fig)
    print('Incremental eligible',result['incremental_eligible'],'breakthrough not established; no submission',flush=True)


if __name__=='__main__':main()
