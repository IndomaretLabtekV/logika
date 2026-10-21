"""Resume exactly one interrupted fixed fit, or its unchanged full report.

Runtime only: no policy selection, no submission. Use the documented scope and
temporary idle/sleep inhibitor. Each successful child has durable provenance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=range(1,5));ap.add_argument('--report',action='store_true');a=ap.parse_args()
    assert (a.fold is not None)!=a.report
    r=ROOT/'results/v56';plan=json.loads((r/'validation_plan.json').read_text())
    for name,expected in plan['sources_sha256'].items():assert sha(ROOT/'eda'/name)==expected
    probe=json.loads((r/'recovery_21okt1840/probe/parity.json').read_text());assert probe['passed'] and probe['bit_exact']
    recovery=r/'recovery_21okt2109';logs=r/'validation_logs';progress=r/'validation_progress.json'
    d=json.loads(progress.read_text());d.update(status='running',recovery_runner_sha256=sha(__file__));save(progress,d)
    python=r/'recovery_21okt1840/paced_python'
    def run(kind,key,script,args):
        attempt=1+sum(x['key']==key for x in d['attempts']);stem=f'{key}_attempt{attempt:02d}'
        log=logs/(stem+'.log');telemetry=logs/(stem+'_telemetry.csv');assert not log.exists() and not telemetry.exists()
        cmd=[str(python),str(ROOT/'eda'/script),*map(str,args)]
        row={'key':key,'kind':kind,'attempt':attempt,'command':cmd,'log':str(log),'telemetry':str(telemetry),'status':'running'}
        d['attempts'].append(row);d['current']=key;save(progress,d);print('START',key,log,flush=True);start=time.monotonic()
        with log.open('x') as f:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);row['pid']=child.pid;save(progress,d)
            watcher=subprocess.Popen([sys.executable,str(ROOT/'eda/gpu_watch.py'),str(child.pid),'--out',str(telemetry)],stdout=f,stderr=subprocess.STDOUT)
            code=child.wait();watch_code=watcher.wait()
        row.update(exit_code=code,watcher_exit_code=watch_code,elapsed_seconds=time.monotonic()-start,status='complete' if code==0 else 'failed');save(progress,d)
        print('END',key,code,row['elapsed_seconds'],flush=True)
        assert code==0,'No automatic retry; inspect preserved log'
    try:
        if a.report:
            for seed in (42,7,2026):
                for fold in range(5):
                    path=r/'fixed_union_rank'/f's{seed}_f{fold}_outer.parquet'
                    metric=json.loads((path.parent/f's{seed}_f{fold}'/'outer_metrics.json').read_text())
                    assert sha(path)==metric['outer_predictions_sha256']
            run('report','paired_report','155_kumo_nested_eval.py',[r/'fixed_union_rank','--baseline-dir',r/'fixed_union_rank','--column','candidate','--baseline-column','baseline','--seeds','42,7,2026','--folds','0,1,2,3,4','--bootstrap','1000','--tag',plan['tag']])
            d.update(status='complete',current=None);save(progress,d)
        else:
            fold=a.fold;cache=r/'teammate_features'/f'outer_s2026_f{fold}'
            run('features',f'features_s2026_f{fold}','166_teammate_nested_features.py',['--out',cache,'--partition','outer','--seed','2026','--fold',fold])
            run('model',f'model_s2026_f{fold}','189_fixed_kumo_union_rank.py',['--feature-cache',cache,'--out',r/'fixed_union_rank','--seed','2026','--fold',fold])
            d.update(status='awaiting_next_fit',current=None);save(progress,d)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        count=len(list((r/'fixed_union_rank').glob('*_outer.parquet')));fig,ax=plt.subplots(figsize=(6,3));ax.bar(['complete','remaining'],[count,15-count]);ax.set(title='Unchanged fixed validation; one-fit recovery',ylabel='Fits');fig.tight_layout();fig.savefig(ROOT/'eda/out/202_recovery_progress.png',dpi=130);plt.close(fig)
        print('Completed contexts',count,'of15; no submission',flush=True)
    except BaseException:
        d.update(status='failed');save(progress,d);raise


if __name__=='__main__':main()
