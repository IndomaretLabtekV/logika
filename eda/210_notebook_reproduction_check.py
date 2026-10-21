"""Execute every production notebook code cell in local replay mode.

This checks the actual embedded pipeline, package pins, source extraction,
cached prediction provenance, CSV values and ordering. It does not run fresh
GPU model training. No submission API is invoked.
"""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    notebook=ROOT/'notebook/final.ipynb';nb=json.loads(notebook.read_text());out=ROOT/'results/v59/v41_lowvar_today'
    for name,expected in nb['metadata']['source_sha256'].items():assert sha(ROOT/'eda'/name)==expected
    variables={'LOGIKA_REPOSITORY':str(ROOT),'LOGIKA_PIPELINE_MODE':'replay','LOGIKA_OUTPUT_DIR':str(out/'notebook_replay')}
    previous={k:os.environ.get(k) for k in variables};os.environ.update(variables)
    executed=copy.deepcopy(nb);state={'__name__':'__notebook__'};rows=[]
    try:
        for index,cell in enumerate(executed['cells']):
            if cell['cell_type']!='code':continue
            source=cell['source'];source=''.join(source) if isinstance(source,list) else source
            captured=io.StringIO();start=time.monotonic()
            with contextlib.redirect_stdout(captured),contextlib.redirect_stderr(captured):exec(compile(source,f'notebook-cell-{index}','exec'),state)
            cell['execution_count']=len(rows)+1;cell['outputs']=[{'output_type':'stream','name':'stdout','text':captured.getvalue()}]
            rows.append({'cell_index':index,'status':'passed','seconds':time.monotonic()-start});print('CELL',index,'passed',flush=True)
    finally:
        for k,value in previous.items():
            if value is None:os.environ.pop(k,None)
            else:os.environ[k]=value
    actual=out/'notebook_replay/submission.csv';prepared=out/'submission.csv'
    assert sha(actual)==sha(prepared),'Notebook and prepared candidate differ'
    result={'notebook_sha256':sha(notebook),'builder_sha256':sha(ROOT/'notebook/build_final.py'),'code_cells':len(rows),'cells':rows,
            'all_code_cells_executed':True,'mode':'replay','fresh_training_executed':False,'csv_exact_sha256_match':True,
            'csv_sha256':sha(actual),'embedded_source_count':len(nb['metadata']['source_sha256']),'source_sha256':nb['metadata']['source_sha256'],
            'submission_performed':False,'verifier_sha256':sha(__file__)}
    (out/'executed_replay.ipynb').write_text(json.dumps(executed,indent=1,ensure_ascii=False))
    (out/'notebook_verification.json').write_text(json.dumps(result,indent=2));(ROOT/'eda/out/210_notebook_reproduction_check.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(figsize=(6,3));ax.bar(['executed cells','failed'],[len(rows),0]);ax.set(ylabel='Notebook code cells',title='Actual local replay; no fresh GPU fit');fig.tight_layout();fig.savefig(ROOT/'eda/out/210_notebook_reproduction_check.png',dpi=130);plt.close(fig)
    os.sync();print('All',len(rows),'code cells executed; CSV byte-identical',sha(actual),flush=True)


if __name__=='__main__':main()
