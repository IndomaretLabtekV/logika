"""Reproduce the fixed October18 low-variance ensemble; never upload.

Replay checks real cached constituent hashes and recreates the CSV exactly.
Fresh builds features and trains every constituent sequentially in a separate
project. Fresh GPU training is not claimed to have been repeated today.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata

TREES=('lgb','lgb_s','xgb','xgb_s','cat','cat_s')
MODULES=('features.py','nn_prep.py','valid2.py','run_valid2.py','kumo_fulltrain.py','common.py')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:json.dump(data,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def align(sample,df):
    expected=['application_id','default_probability']
    if df.columns!=expected or sample.columns!=expected or df.height!=30000 or sample.height!=30000:
        raise ValueError('Submission columns/row count mismatch')
    if df['application_id'].n_unique()!=30000 or sample['application_id'].n_unique()!=30000:
        raise ValueError('Duplicate application IDs')
    if set(df['application_id'])!=set(sample['application_id']):raise ValueError('Submission ID mismatch')
    result=sample.select('application_id').join(df,on='application_id',how='left',maintain_order='left',validate='1:1')
    p=result['default_probability'].to_numpy()
    if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():raise ValueError('Non-finite or invalid prediction')
    return result


def blend(repository,out,audit=None):
    sample=pl.read_csv(repository/'data/sample_submission.csv');fingerprints={}
    def load(relative):
        path=repository/'results'/relative;digest=sha(path);fingerprints[str(path.relative_to(repository))]=digest
        if audit is not None and audit['inputs_sha256'].get(str(path.relative_to(repository)))!=digest:
            raise ValueError('Cached prediction differs from the completed local audit')
        d=pl.read_parquet(path);cols=[c for c in d.columns if c not in ('application_id','target')]
        if len(cols)!=1:raise ValueError('Ambiguous constituent prediction')
        return align(sample,d.select('application_id',pl.col(cols[0]).alias('default_probability')))['default_probability'].to_numpy()
    rk=lambda p:rankdata(p,method='average')/len(p)
    tree=rk(np.mean([rk(load(f'v22/v22/gbdt/test_{m}.parquet')) for m in TREES],axis=0))
    causilo=[(1,load('v22/v22/fm/test_causilo.parquet')),(2,load('v37/test_causilo.parquet')),(2,load('v41/causilo_s12/test_causilo.parquet'))]
    c=rk(sum(w*p for w,p in causilo)/5)
    kumo=[load(p) for p in ('v34/test_kumo_full70k.parquet','v41/test_kumo_full70k_g1.parquet','v41/test_kumo_full70k_g2.parquet')]
    k=rk(np.mean(kumo,axis=0));score=(tree+c+k)/3
    output=align(sample,pl.DataFrame({'application_id':sample['application_id'],'default_probability':score}))
    if audit is not None:
        legacy_path=repository/'results/v41/v31_lowvar.csv'
        if sha(legacy_path)!=audit['candidate_csv_sha256']:raise ValueError('Historical candidate changed since audit')
        legacy=align(sample,pl.read_csv(legacy_path))
        if not np.array_equal(score,legacy['default_probability'].to_numpy()):raise ValueError('Cached recipe failed exact numerical reproduction')
    path=out/'submission.csv';temp=path.with_suffix('.csv.tmp');output.write_csv(temp)
    with temp.open('rb') as f:os.fsync(f.fileno())
    temp.replace(path);again=align(sample,pl.read_csv(path));np.testing.assert_array_equal(again['default_probability'].to_numpy(),score)
    components=pl.DataFrame({'application_id':sample['application_id'],'tree_rank':tree,'causilo_rank':c,'kumo_rank':k})
    components.write_parquet(out/'components.parquet')
    fig,ax=plt.subplots(figsize=(7,3));ax.hist(score,bins=40);ax.set(xlabel='Fixed ensemble score',ylabel='Test applications',title='Low-variance ensemble: 30,000 applications');fig.tight_layout();fig.savefig(out/'score_distribution.png',dpi=130);fig.savefig(Path(__file__).resolve().parent/'out/209_lowvar_distribution.png',dpi=130);plt.close(fig)
    return {'recipe':'rank(mean6tree-ranks)/3 + rank(mean5Causilo-seeds)/3 + rank(mean3Kumo70k-draws)/3',
            'inputs_sha256':fingerprints,'submission_sha256':sha(path),'sample_sha256':sha(repository/'data/sample_submission.csv'),
            'rows':30000,'columns':output.columns,'exact_replay':audit is not None,'submission_performed':False}


def fresh(repository,out,data):
    from huggingface_hub import hf_hub_download
    checkpoint=hf_hub_download('nvidia/Kumo-Tabular','large/classifier.pt',revision='v1.0.1')
    if sha(checkpoint)!='cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613':
        raise ValueError('Pretrained Kumo checkpoint differs from the audited public bytes')
    project=out/'fresh_project'
    if project.exists():raise ValueError('Fresh training needs a new output directory')
    (project/'eda/out').mkdir(parents=True);(project/'data_pq').mkdir();(project/'data').symlink_to(data.resolve(),target_is_directory=True)
    for name in MODULES:shutil.copy2(repository/'eda'/name,project/'eda'/name)
    env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','POLARS_MAX_THREADS':'4',
         'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
    commands=[]
    def run(name,script,args):
        bootstrap="import sys,runpy; import valid2 as v; v.THREADS=4; v.LGB['n_jobs']=4; v.LGB_S['n_jobs']=4; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
        cmd=[sys.executable,'-c',bootstrap,str(project/'eda'/script),*map(str,args)];commands.append(cmd)
        with (out/f'fresh_{name}.log').open('x') as f:subprocess.run(cmd,cwd=project/'eda',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    # common.features builds the identical canonical features from organizer CSVs.
    subprocess.run([sys.executable,'-c','from common import features; print(features().shape)'],cwd=project/'eda',env=env,check=True)
    rp=project/'results'
    run('trees','run_valid2.py',[rp/'v22/v22/gbdt',','.join(TREES),'--seeds','42,7,2026'])
    for name,path,seeds in (('causilo42','v22/v22/fm','42'),('causilo7_2026','v37','7,2026'),('causilo1_2','v41/causilo_s12','1,2')):
        run(name,'run_valid2.py',[rp/path,'causilo','--seeds',seeds])
    for off in range(3):run(f'kumo_g{off}','kumo_fulltrain.py',[rp/('v34' if off==0 else 'v41'),'--g_off',off,'--chunk','500'])
    return project,commands


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--mode',choices=['replay','fresh'],default='replay');ap.add_argument('--data',type=Path);a=ap.parse_args()
    repository=a.repository.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);(Path(__file__).resolve().parent/'out').mkdir(exist_ok=True)
    if out==repository or out.is_relative_to(repository/'data'):raise ValueError('Use a separate results directory')
    audit=None;commands=[];input_repository=repository
    if a.mode=='replay':
        audit_path=repository/'eda/out/201_saved_lowvar_audit.json';audit=json.loads(audit_path.read_text())
        assert audit['source_sha256']==sha(repository/'eda/201_saved_lowvar_audit.py') and audit['candidate_test_reproduced']
    else:input_repository,commands=fresh(repository,out,a.data or repository/'data')
    result=blend(input_repository,out,audit)
    result.update(mode=a.mode,reproducer_sha256=sha(__file__),source_sha256={n:sha(repository/'eda'/n) for n in MODULES},
                  fresh_training_commands=commands,training_repeated_today=a.mode=='fresh',
                  packages={p:importlib.metadata.version(p) for p in ('numpy','polars','pandas','scipy')})
    if audit is not None:
        result.update(audit_sha256=sha(audit_path),local_ks=audit['candidate_ks'],local_reference_ks=audit['v31_reference_ks'],
                      bootstrap_p_gt_zero=audit['p_gt_zero'],breakthrough=False,
                      public_forecast_mean=audit['public_forecast_mean'],public_forecast_sd=audit['public_forecast_sd'])
    save(out/'reproduction.json',result);os.sync()
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
