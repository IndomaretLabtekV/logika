"""Sequential15-fit CPU execution of the frozen204 policy; no upload."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--capped',action='store_true');a=ap.parse_args()
    model_script='206_cpu_capped_union_soft_ssl.py' if a.capped else '204_cpu_union_soft_ssl.py'
    work=ROOT/('results/v58/cpu_capped_union_ssl' if a.capped else 'results/v57/cpu_union_ssl');work.mkdir(parents=True,exist_ok=True);logs=work/'logs';logs.mkdir(exist_ok=True)
    screen=json.loads((work/'s42_f0/frozen_selection.json').read_text());assert screen['promising_inner'],'Screen did not support continuation'
    sources={n:sha(ROOT/'eda'/n) for n in (model_script,'166_teammate_nested_features.py','valid2.py',Path(__file__).name)}
    protocol=ROOT/'docs/integration'/('protokol_cpu_capped_union_21okt.md' if a.capped else 'protokol_cpu_union_ssl_21okt.md');protocol_hash=sha(protocol)
    progress={'status':'running','source_sha256':sources,'protocol_sha256':protocol_hash,'children':[],
              'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'heavy_children_concurrent':1}
    path=work/'progress.json'
    def save():
        tmp=path.with_suffix('.json.tmp')
        with tmp.open('w') as f:json.dump(progress,f,indent=2);f.flush();os.fsync(f.fileno())
        tmp.replace(path)
    if path.exists():
        previous=json.loads(path.read_text());assert previous['source_sha256']==sources and previous['protocol_sha256']==protocol_hash
        archive=work/f'progress_before_resume_{time.time_ns()}.json';archive.write_bytes(path.read_bytes())
    save()
    def run(name,script,args):
        for n,s in sources.items():assert sha(ROOT/'eda'/n)==s,'Frozen source changed'
        assert sha(protocol)==protocol_hash
        log=logs/f'{name}_{time.time_ns()}.log';cmd=[sys.executable,str(ROOT/'eda'/script),*map(str,args)]
        row={'name':name,'command':cmd,'log':str(log),'status':'running'};progress['children'].append(row);save();start=time.monotonic();print('START',name,log,flush=True)
        with log.open('x') as f:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);row['pid']=child.pid;save();code=child.wait()
        row.update(exit_code=code,status='complete' if code==0 else 'failed',seconds=time.monotonic()-start);save();print('END',name,code,row['seconds'],flush=True)
        if code:raise RuntimeError('Child failed; no automatic retry')
    try:
        for seed in (42,7,2026):
            for fold in range(5):
                cache=ROOT/'results/v56/teammate_features'/f'outer_s{seed}_f{fold}'
                run(f'features_s{seed}_f{fold}','166_teammate_nested_features.py',['--out',cache,'--partition','outer','--seed',seed,'--fold',fold])
                run(f'model_s{seed}_f{fold}',model_script,['--out',work,'--seed',seed,'--fold',fold,'--evaluate'])
        progress['status']='complete';save()
    except BaseException:
        progress['status']='failed';save();raise
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    count=len(list(work.glob('*_outer.parquet')));fig,ax=plt.subplots(figsize=(6,3));ax.bar(['complete','remaining'],[count,15-count]);ax.set(title='CPU nested union/soft-SSL execution',ylabel='Fits');fig.tight_layout();fig.savefig(ROOT/'eda/out/205_cpu_union_progress.png',dpi=130);plt.close(fig)
    os.sync();print('Completed',count,'/15; evaluate paired statistics before export; no submission',flush=True)


if __name__=='__main__':main()
