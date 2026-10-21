"""Read-only publication checks; no data, model imports, training or network.

Default: check tracked files plus nonignored new files, as git add -A would.
--staged: check the actual index bytes, including newly staged modifications.
Reports paths/types only, never a matched secret value.
"""
import argparse
import ast
import collections
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SECRET_PATTERNS={
    'private_key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'github_token':r'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{40,})\b',
    'hf_token':r'\bhf_[A-Za-z0-9]{25,}\b',
    'openai_token':r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}\b',
    'aws_key':r'\bAKIA[0-9A-Z]{16}\b',
    'kaggle_key':r'"(?:key|api_key)"\s*:\s*"[a-f0-9]{32}"',
}
BLOCKED_SUFFIXES={'.parquet','.npy','.npz','.pt','.pkl','.cbm','.ubj','.ckpt','.joblib','.safetensors','.onnx','.pyc','.pem','.key','.zip','.tmp','.log'}


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--staged',action='store_true');a=ap.parse_args()
    names={p.decode() for p in git('ls-files','-z').split(b'\0') if p}
    if not a.staged:names|={p.decode() for p in git('ls-files','--others','--exclude-standard','-z').split(b'\0') if p}
    errors=[];counts=collections.Counter();sizes=collections.Counter();total=0;largest=[];skipped_magic_cells=0
    for name in sorted(names):
        path=ROOT/name
        if not a.staged and not path.is_file():continue
        parts=Path(name).parts;suffix=Path(name).suffix.lower()
        if parts[0] in {'data','data_pq','results','log'} or parts[0].startswith('.venv') or suffix in BLOCKED_SUFFIXES:
            errors.append({'path':name,'error':'Local artifact or credential file must not be published'})
        if Path(name).name in {'.env','secrets.env','credentials.json','kaggle.json'}:
            errors.append({'path':name,'error':'Credential filename'})
        raw=git('show',':'+name) if a.staged else path.read_bytes()
        n=len(raw);total+=n;counts[parts[0]]+=1;sizes[parts[0]]+=n;largest.append((n,name))
        if n>20*1024*1024:errors.append({'path':name,'error':'Unexpected publication file over20MiB','bytes':n})
        try:text=raw.decode('utf-8')
        except UnicodeDecodeError:continue
        for kind,pattern in SECRET_PATTERNS.items():
            for match in re.finditer(pattern,text):errors.append({'path':name,'error':'Potential '+kind,'line':text.count('\n',0,match.start())+1})
        if suffix=='.py':
            try:ast.parse(text,filename=name)
            except SyntaxError as exc:errors.append({'path':name,'error':'Python syntax','line':exc.lineno})
        elif suffix=='.ipynb':
            try:
                nb=json.loads(text)
                if nb.get('nbformat')!=4 or not isinstance(nb.get('cells'),list):raise ValueError('Notebook structure')
                for index,cell in enumerate(nb['cells']):
                    if cell.get('cell_type')!='code':continue
                    source=cell.get('source','');source=''.join(source) if isinstance(source,list) else source
                    if any(line.lstrip().startswith(('%','!')) for line in source.splitlines()):skipped_magic_cells+=1;continue
                    try:ast.parse(source,filename=f'{name}:cell{index}')
                    except SyntaxError as exc:errors.append({'path':name,'error':'Notebook Python syntax','cell':index,'line':exc.lineno})
            except (ValueError,TypeError,KeyError):errors.append({'path':name,'error':'Invalid notebook JSON/schema'})
    for required in ('README.md','.gitignore','requirements.txt','tools/check_repository.py','docs/github.md','eda/valid2.py','scripts/analyze.py','third_party/structured-data-models/LICENSE','third_party/structured-data-models/NOTICE'):
        if required not in names:errors.append({'path':required,'error':'Required source/documentation missing from publication'})
    summary={'mode':'index' if a.staged else 'prospective git add -A','files':sum(counts.values()),'bytes':total,
             'groups':{name:{'files':counts[name],'bytes':sizes[name]} for name in sorted(counts)},
             'largest_files':[{'bytes':n,'path':p} for n,p in sorted(largest,reverse=True)[:8]],
             'notebook_magic_cells_not_syntax_checked':skipped_magic_cells,'model_executed':False,'network_used':False,'errors':errors}
    report=ROOT/'.repo-checks';report.mkdir(exist_ok=True);(report/('staged.json' if a.staged else 'prospective.json')).write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2));return 1 if errors else 0


if __name__=='__main__':sys.exit(main())
