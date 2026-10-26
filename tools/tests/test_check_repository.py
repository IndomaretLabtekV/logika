"""Publication checks using small, disposable Git repositories; no ML dependencies."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import check_repository as checker


class PublicationChecks(unittest.TestCase):
    def findings(self, name, raw=b'', **kwargs):
        return checker.audit_entry(name, raw, **kwargs)[0]

    def test_nested_credentials_and_artifacts_are_rejected(self):
        for name in ('scripts/.env.local', 'config/service-account-prod.json',
                     'config/my-kaggle-token.json', 'nested/.aws/config',
                     'nested/model.pth', 'data/train.csv', 'submissions/final.csv',
                     'eda/out/mpl_cache/fontlist.json'):
            with self.subTest(name=name):
                self.assertTrue(self.findings(name))

    def test_compact_reports_and_python_source_are_allowed(self):
        self.assertFalse(self.findings('artifacts/metrics.json', b'{"ks": 0.5}'))
        self.assertFalse(self.findings('eda/features.py', b'value = 1\n'))

    def test_secrets_are_reported_without_printing_the_value(self):
        secret = ('gh' + 'p_' + 'a' * 30).encode()
        findings = self.findings('config.txt', secret)
        self.assertTrue(findings)
        self.assertNotIn(secret.decode(), json.dumps(findings))

    def test_python_syntax_is_checked(self):
        self.assertEqual(self.findings('bad.py', b'def broken(:')[0]['error'], 'Python syntax')

    def test_notebook_outputs_require_cleanup_but_archival_evidence_is_preserved(self):
        nb = {'nbformat': 4, 'metadata': {}, 'cells': [
            {'cell_type': 'code', 'source': ['value = 1'], 'outputs': [], 'execution_count': 1},
        ]}
        raw = json.dumps(nb).encode()
        self.assertTrue(self.findings('notebook/final.ipynb', raw))
        self.assertFalse(self.findings('research/snapshots/run.ipynb', raw))
        nb['cells'][0]['execution_count'] = None
        self.assertFalse(self.findings('notebook/final.ipynb', json.dumps(nb).encode()))

    def test_malformed_notebooks_fail_cleanly(self):
        for raw in (b'[]', b'{"nbformat": 4, "cells": [null]}', b'{'):
            with self.subTest(raw=raw):
                self.assertEqual(self.findings('bad.ipynb', raw)[0]['error'], 'Invalid notebook JSON/schema')

    def test_symlinks_are_inspected_as_targets_not_source(self):
        self.assertFalse(self.findings('scripts/dsc0152/data', b'../../data', symlink=True))
        for target in (b'/tmp/model', b'../../outside', b'../.aws/credentials', b'../config/.env'):
            with self.subTest(target=target):
                self.assertTrue(self.findings('scripts/link', target, symlink=True))

    def test_working_tree_and_index_have_distinct_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                subprocess.run(['git', *args], cwd=root, check=True, capture_output=True)
            git('init', '-q')
            (root / '.gitignore').write_text('private/\n')
            (root / 'source.py').write_text('value = 1\n')
            (root / 'deleted.py').write_text('value = 0\n')
            (root / 'local.py').symlink_to('private/missing.py')
            git('add', '.')
            (root / 'source.py').write_text('value = 2\n')
            (root / 'deleted.py').unlink()
            (root / 'new.py').write_text('value = 3\n')
            (root / 'private').mkdir()
            (root / 'private/key.txt').write_text('not published')
            with patch.object(checker, 'ROOT', root):
                working = {name: (raw, link) for name, raw, link in checker.collect_entries()}
                staged = {name: (raw, link) for name, raw, link in checker.collect_entries(True)}
            self.assertEqual(working['source.py'][0], b'value = 2\n')
            self.assertEqual(staged['source.py'][0], b'value = 1\n')
            self.assertNotIn('deleted.py', working)
            self.assertIn('deleted.py', staged)
            self.assertIn('new.py', working)
            self.assertNotIn('new.py', staged)
            self.assertNotIn('private/key.txt', working)
            self.assertEqual(working['local.py'], (b'private/missing.py', True))
            self.assertEqual(staged['local.py'], (b'private/missing.py', True))


if __name__ == '__main__':
    unittest.main()
