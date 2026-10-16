"""Build the isolated nested tuning notebook without executing any model."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def md(source):
    return dict(cell_type='markdown', metadata={}, source=source)


def code(source):
    return dict(cell_type='code', metadata={}, source=source, outputs=[], execution_count=None)


def section(title, anchor):
    return md(f'---\n\n# {title} <a name="{anchor}"></a>\n\n---')


def source_without_imports_or_docstrings(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body.pop(0)
    return ast.unparse(tree)


feature_source = (ROOT.parent / 'eda/features.py').read_text()
experiment = (ROOT / 'v10_experiment.py').read_text()
imports = '\n'.join(ast.unparse(n) for n in ast.parse(experiment).body
                    if isinstance(n, (ast.Import, ast.ImportFrom)))
cells = [
    md('---\n\n# DSC LOGIKA UI 2026: Nested Tuning\n\n*An isolated test of LightGBM capacity and regularization.*'),
    md('---\n\n## TeamName (DSCxxxx)\n\n- Name 1\n- Name 2\n- Name 3'),
    md('---\n\n## Table of Contents\n\n1. [Introduction](#1)\n2. [Initialization](#2)\n3. [Features](#3)\n4. [Experiment](#4)'),
    section('Introduction', '1'),
    md('The experiment tests whether tuning improves the unchanged feature pipeline. '
       'Five outer folds estimate the full selection procedure. Each outer training partition '
       'uses three inner folds to choose from twelve fixed configurations and determine training length. '
       'Outer labels are never used for early stopping or parameter selection. '
       'Control and tuned models use the same outer partitions and inner stopping protocol. '
       'This is a new control, not an exact reproduction of v9, whose stopping used outer labels. '
       'Feature hypotheses were previously explored on this dataset, so this is not a pristine unseen holdout.'),
    section('Initialization', '2'),
    md('The notebook runs on Kaggle with competition data attached and Internet enabled. '
       'LightGBM uses the Kaggle CPU; this experiment does not need GPU or foundation-model secrets.'),
    code('%pip install -q lightgbm==4.6.0 polars==2.0.0 pandas pyarrow scipy matplotlib'),
    md('## Libraries\n\nThe required libraries and feature source fingerprint are loaded before execution.'),
    code(imports + '\nimport lightgbm as lgb\nFEATURE_SHA256 = ' + repr(hashlib.sha256(feature_source.encode()).hexdigest())),
    section('Features', '3'),
    md('The existing cutoff-aware feature builders are embedded from eda/features.py. '
       'Customer and application identifiers are excluded from predictors. '
       'The category price reference is historical and label-free, retaining the established transductive protocol.'),
    code(source_without_imports_or_docstrings(feature_source)),
    section('Experiment', '4'),
    md('The candidate grid changes leaf counts, leaf support, feature sampling and regularization. '
       'The settings follow [LightGBM tuning documentation](https://lightgbm.readthedocs.io/en/v4.5.0/Parameters-Tuning.html). '
       'Each completed fold saves its predictions. An interrupted run produces no complete submission. '
       'A candidate merits a larger repeat only if pooled KS rises by more than 0.002 and at least four folds improve. '
       'The paired bootstrap describes uncertainty conditional on these fitted models; it does not capture retraining variability.'),
    code(source_without_imports_or_docstrings(experiment)),
    code('main(build_all, lgb)'),
    md('#### Insights\n\n> Inspect v10/metrics.json and comparison.png after execution. '
       'The run writes ID-aligned OOF predictions, inner trial scores, chosen parameters and two submission files. '
       'No improvement is claimed until the run completes. A public score above 0.57768 remains unverified until Kaggle scores a submission.'),
]
for i, cell in enumerate(cells):
    cell['id'] = f'v10-{i:03d}'
    if cell['cell_type'] == 'code' and not cell['source'].startswith('%'):
        compile(cell['source'], f'cell-{i}', 'exec')
nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
          metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                        language_info=dict(name='python', version='3.13.0')))
out = ROOT / 'v10.ipynb'
out.write_text(json.dumps(nb, indent=1))
print(f'Wrote {len(cells)} cells -> {out}')
