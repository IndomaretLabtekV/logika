"""Check publication files without importing models, reading datasets or using the network.

Default: inspect the working files that git add -A would publish.
--staged: inspect the actual index bytes, including symlinks and staged deletions.
Only paths and finding types are reported; matched secret values are never printed.
"""
import argparse
import ast
import collections
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = {
    'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'github_token': r'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{40,})\b',
    'hf_token': r'\bhf_[A-Za-z0-9]{25,}\b',
    'openai_token': r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}\b',
    'aws_key': r'\bAKIA[0-9A-Z]{16}\b',
    'kaggle_key': r'"(?:key|api_key)"\s*:\s*"[a-f0-9]{32}"',
}
BLOCKED_SUFFIXES = {
    '.parquet', '.npy', '.npz', '.arrow', '.feather', '.duckdb', '.wal',
    '.sqlite', '.sqlite3', '.pt', '.pth', '.pkl', '.pickle', '.cbm', '.ubj',
    '.ckpt', '.joblib', '.safetensors', '.onnx', '.h5', '.hdf5', '.model',
    '.pyc', '.pyo', '.pem', '.key', '.zip', '.tar', '.gz', '.tgz', '.7z',
    '.tmp', '.swp', '.swo', '.log',
}
BLOCKED_PARTS = {
    '.aws', '.agents', '.codex', '.kaggle', '__pycache__', 'mpl_cache',
    '.cache', '.ipynb_checkpoints', 'catboost_info', '.repo-checks',
    '.pytest_cache', '.mypy_cache', '.ruff_cache', '.idea', '.vscode',
    'node_modules', 'checkpoints', 'lightning_logs', 'wandb',
}
LOCAL_ROOTS = {'data', 'data_pq', 'results', 'log', 'exports'}
# These exact archival trees are hash-recorded evidence, including executed runs.
PRESERVED_NOTEBOOKS = ('research/snapshots/', 'notebook/archive/')
REQUIRED_FILES = (
    'README.md', '.gitignore', 'requirements.txt', 'tools/check_repository.py',
    'docs/github.md', 'eda/valid2.py', 'scripts/analyze.py',
    'third_party/structured-data-models/LICENSE',
    'third_party/structured-data-models/NOTICE',
)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def credential_name(name):
    base = PurePosixPath(name).name.lower()
    return (
        base == '.env' or base.startswith('.env.')
        or base in {'secrets.env', 'credentials.json'}
        or ('kaggle' in base and base.endswith('.json'))
        or (base.startswith('service-account') and base.endswith('.json'))
    )


def audit_entry(name, raw, *, symlink=False):
    """Return findings for one Git entry; never follow a symlink on disk."""
    errors = []
    skipped = 0
    path = PurePosixPath(name)
    if (path.parts[0] in LOCAL_ROOTS or path.suffix.lower() in BLOCKED_SUFFIXES
            or any(part in BLOCKED_PARTS or part.startswith('.venv') or part == 'venv'
                   for part in path.parts)
            or (path.parts[0] == 'submissions' and path.suffix.lower() == '.csv')):
        errors.append({'path': name, 'error': 'Local artifact must not be published'})
    if credential_name(name):
        errors.append({'path': name, 'error': 'Credential filename'})
    if len(raw) > 20 * 1024 * 1024:
        errors.append({'path': name, 'error': 'Unexpected publication file over 20 MiB'})
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        if symlink:
            errors.append({'path': name, 'error': 'Invalid symlink target encoding'})
        return errors, skipped
    for kind, pattern in SECRET_PATTERNS.items():
        for match in re.finditer(pattern, text):
            errors.append({'path': name, 'error': 'Potential ' + kind,
                           'line': text.count('\n', 0, match.start()) + 1})
    if symlink:
        target = posixpath.normpath(posixpath.join(str(path.parent), text))
        # Relative aliases into local data/results are intentional for DSC0152.
        if (posixpath.isabs(text) or target == '..' or target.startswith('../')
                or credential_name(target)
                or any(part in BLOCKED_PARTS for part in PurePosixPath(target).parts)):
            errors.append({'path': name, 'error': 'Unsafe or machine-specific symlink target'})
        return errors, skipped
    if path.suffix == '.py':
        try:
            ast.parse(text, filename=name)
        except SyntaxError as exc:
            errors.append({'path': name, 'error': 'Python syntax', 'line': exc.lineno})
    elif path.suffix == '.ipynb':
        try:
            nb = json.loads(text)
            if nb.get('nbformat') != 4 or not isinstance(nb.get('cells'), list):
                raise ValueError('Notebook structure')
            preserve = name.startswith(PRESERVED_NOTEBOOKS)
            if nb.get('metadata', {}).get('widgets') and not preserve:
                errors.append({'path': name, 'error': 'Remove notebook widget state'})
            for index, cell in enumerate(nb['cells']):
                if cell.get('cell_type') != 'code':
                    continue
                if not preserve and (cell.get('outputs') or cell.get('execution_count') is not None):
                    errors.append({'path': name, 'error': 'Clear notebook outputs and execution count',
                                   'cell': index})
                source = cell.get('source', '')
                source = ''.join(source) if isinstance(source, list) else source
                if any(line.lstrip().startswith(('%', '!')) for line in source.splitlines()):
                    skipped += 1
                    continue
                try:
                    ast.parse(source, filename=f'{name}:cell{index}')
                except SyntaxError as exc:
                    errors.append({'path': name, 'error': 'Notebook Python syntax',
                                   'cell': index, 'line': exc.lineno})
        except (ValueError, TypeError, KeyError, AttributeError):
            errors.append({'path': name, 'error': 'Invalid notebook JSON/schema'})
    return errors, skipped


def collect_entries(staged=False):
    """Read index modes so staged symlinks need no live target or working copy."""
    indexed = {}
    for record in git('ls-files', '--stage', '-z').split(b'\0'):
        if not record:
            continue
        header, name = record.split(b'\t', 1)
        mode, _, stage = header.decode().split()
        if stage != '0':
            raise ValueError('Resolve merge conflicts before publication')
        indexed[os.fsdecode(name)] = mode
    names = set(indexed)
    if not staged:
        names.update(os.fsdecode(p) for p in git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0') if p)
    for name in sorted(names):
        path = ROOT / name
        if staged:
            if indexed[name] == '160000':
                raise ValueError(f'Unexpected submodule: {name}')
            yield name, git('show', ':' + name), indexed[name] == '120000'
        elif path.is_symlink():
            yield name, os.fsencode(os.readlink(path)), True
        elif path.is_file():
            yield name, path.read_bytes(), False
        # A deleted working file is excluded, just as it would be by git add -A.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true')
    args = parser.parse_args()
    errors = []
    counts = collections.Counter()
    sizes = collections.Counter()
    largest = []
    skipped = symlinks = 0
    names = set()
    try:
        for name, raw, symlink in collect_entries(args.staged):
            names.add(name)
            group = PurePosixPath(name).parts[0]
            counts[group] += 1
            sizes[group] += len(raw)
            largest.append((len(raw), name))
            findings, magic = audit_entry(name, raw, symlink=symlink)
            errors.extend(findings)
            skipped += magic
            symlinks += symlink
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        errors.append({'error': 'Unable to inspect publication files', 'detail': str(exc)})
    for required in REQUIRED_FILES:
        if required not in names:
            errors.append({'path': required, 'error': 'Required source/documentation missing'})
    summary = {
        'mode': 'index' if args.staged else 'prospective git add -A',
        'files': sum(counts.values()), 'bytes': sum(sizes.values()),
        'groups': {name: {'files': counts[name], 'bytes': sizes[name]} for name in sorted(counts)},
        'largest_files': [{'bytes': n, 'path': p} for n, p in sorted(largest, reverse=True)[:8]],
        'symlinks_checked_without_following': symlinks,
        'notebook_magic_cells_not_syntax_checked': skipped,
        'model_executed': False, 'network_used': False, 'errors': errors,
    }
    report = ROOT / '.repo-checks'
    report.mkdir(exist_ok=True)
    destination = report / ('staged.json' if args.staged else 'prospective.json')
    destination.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
