"""Build the current low-variance recipe notebook; the old v34 builder is archived.

Default mode replays audited local constituent predictions without GPU fitting.
Fresh mode contains the full sequential feature/model pipeline.
"""
import hashlib
import json
from pathlib import Path
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'notebook/final.ipynb'
MODULES=['features.py','nn_prep.py','valid2.py','run_valid2.py','kumo_fulltrain.py','common.py','209_reproduce_lowvar.py']
SOURCES={name:(ROOT/'eda'/name).read_text() for name in MODULES}
HASHES={name:hashlib.sha256(source.encode()).hexdigest() for name,source in SOURCES.items()}
cells=[]
def md(s):cells.append({'cell_type':'markdown','id':uuid4().hex[:8],'metadata':{},'source':s})
def code(s):
    compile(s,'notebook-cell','exec')
    cells.append({'cell_type':'code','id':uuid4().hex[:8],'metadata':{},'source':s,'execution_count':None,'outputs':[]})

md('# Credit risk: fixed low-variance ensemble\n\nDSC LOGIKA UI2026. Six GBDT, Causilo5outer seeds, and Kumo large3estimator draws. No upload or external training data.')
md('## 1. Aim and metric\n\nPredict30,000official test applications. KS is the maximum distance between class score distributions. The recipe was specified on18October; no weights are selected from the public leaderboard. Kaggle uses30%public/70%private rows; competition grading assigns45%to each leaderboard.')
md('## 2. Validation evidence\n\nThe October21 local re-audit gives OOF analogue KS0.526220 versus v31 0.526126. Paired bootstrap P(delta>0)=0.365,2/5fold wins: this is **not a demonstrated breakthrough**. The forecast before submission is0.563621 ±0.002229. Kumo OOF uses56k training contexts and outer-seed averaging; deployment uses70k contexts and estimator-draw averaging. These differ. Historical aggregated files do not prove every modern per-fit provenance invariant.\n\nReplay below reconstructs the CSV from hash-checked real predictions. It does not claim GPU models were retrained today. Fresh mode supplies all feature and model steps; its full training was not repeated during this hardware-unstable session.')
md('## 3. Settings and environment\n\nRun from the repository or notebook directory. Set LOGIKA_REPOSITORY for another location. Replay needs the local prediction artifacts. Set LOGIKA_PIPELINE_MODE=fresh to rebuild from organizer data and public pretrained models in an isolated output project; one model process runs at a time. No API submission is included.')
code("import os,sys,json,hashlib,subprocess\nfrom pathlib import Path\nimport importlib.metadata\nimport numpy as np\nimport polars as pl\nMODE=os.environ.get('LOGIKA_PIPELINE_MODE','replay')\nREPOSITORY=Path(os.environ.get('LOGIKA_REPOSITORY',str(Path.cwd()))).resolve()\nif not (REPOSITORY/'data').exists() and (REPOSITORY.parent/'data').exists(): REPOSITORY=REPOSITORY.parent\nOUTPUT_DIR=Path(os.environ.get('LOGIKA_OUTPUT_DIR',str(REPOSITORY/'results/v59/v41_lowvar_today/notebook_replay'))).resolve()\nOUTPUT_DIR.mkdir(parents=True,exist_ok=True)\nCODE_ROOT=OUTPUT_DIR/'embedded_project'\n(CODE_ROOT/'eda/out').mkdir(parents=True,exist_ok=True)\nfor name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS'): os.environ[name]='4'\nassert MODE in ('replay','fresh')\nprint('mode',MODE,'output',OUTPUT_DIR)")
code("PINNED={'numpy':'2.5.4','pandas':'3.0.6','polars':'2.0.0','scipy':'1.18.1'}\nFRESH_PINNED={'lightgbm':'4.7.0','xgboost':'3.4.1','catboost':'1.2.10','structured-data-models':'0.2.1','causilo':'1.0.3','torch':'2.14.1'}\nrequired={**PINNED,**(FRESH_PINNED if MODE=='fresh' else {})}\nfor package,expected in required.items():\n    actual=importlib.metadata.version(package)\n    assert actual==expected,(package,actual,expected)\nprint(required)")
md('## 4. Feature and model source\n\nEight organizer tables produce232application-level predictors. Builders filter related events at decision_datetime. No test target is used. validation_v2 uses five outer folds; stopping is chosen in an inner split and followed by refitting. All source below is embedded and SHA256 checked. NVIDIA Kumo-Tabular uses public v1.0.1 weights, checkpoint SHA256 cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613. Fresh execution checks those bytes before training.')
code('SOURCES='+repr(SOURCES)+'\nSOURCE_SHA256='+repr(HASHES)+"\nfor name,source in SOURCES.items():\n    assert hashlib.sha256(source.encode()).hexdigest()==SOURCE_SHA256[name]\n    compile(source,name,'exec')\n    (CODE_ROOT/'eda'/name).write_text(source)\nprint('Embedded modules',list(SOURCES))")
md('## 5. Reproduction\n\nTrees average5fold×3seed models per GBDT. Causilo averages25contexts from outer seeds42,7,2026,1,2. Kumo uses the full70k training set, e4, and generator offsets0,1,2. Fresh mode builds all these sequentially. Replay instead checks all12constituent file hashes against audit201 and reproduces the already specified recipe exactly.')
code("command=[sys.executable,str(CODE_ROOT/'eda/209_reproduce_lowvar.py'),'--repository',str(REPOSITORY if MODE=='replay' else CODE_ROOT),'--out',str(OUTPUT_DIR),'--mode',MODE]\nif MODE=='fresh': command.extend(['--data',str(REPOSITORY/'data')])\nsubprocess.run(command,check=True,env=os.environ.copy())")
md('## 6. Fixed blending and CSV validation\n\nScore = (rank(mean of six tree percentile ranks) + rank(mean Causilo probabilities over five seeds) + rank(mean Kumo probabilities over three draws))/3. Average ties are retained. This keeps one-third weight on each family. Scores are rankings for KS rather than calibrated default probabilities.')
code("submission=pl.read_csv(OUTPUT_DIR/'submission.csv')\nsample=pl.read_csv(REPOSITORY/'data/sample_submission.csv')\nassert submission.columns==['application_id','default_probability']\nassert submission.height==sample.height==30000\nassert submission['application_id'].n_unique()==30000\nassert submission['application_id'].to_list()==sample['application_id'].to_list()\np=submission['default_probability'].to_numpy()\nassert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()\nmanifest=json.loads((OUTPUT_DIR/'reproduction.json').read_text())\nassert hashlib.sha256((OUTPUT_DIR/'submission.csv').read_bytes()).hexdigest()==manifest['submission_sha256']\nprint('Validated CSV',manifest['submission_sha256'],'exact replay',manifest['exact_replay'])")
md('## 7. Interpretation\n\nThis is the pre-existing low-variance candidate released under the user’s latest instruction to submit today. Local evidence for a gain is weak. No claim of0.570public, superiority over v31, or a completed new Kumo breakthrough is made. The untouched v34 production recipe is archived in notebook/archive/production_v34_21okt/. Public/private results must be recorded separately after an actual submission receipt.')
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.13.16'},'logika_recipe':'v41_lowvar_released_v59','source_sha256':HASHES},'cells':cells}
OUT.write_text(json.dumps(nb,indent=1,ensure_ascii=False))
print('Wrote',len(cells),'cells to',OUT)
