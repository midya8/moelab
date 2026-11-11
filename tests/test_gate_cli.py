"""Strict parser, isolated verifier, complete controls and CLI round trips."""
import copy
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from moelab.gate_cli import load
from moelab.gate_regret import audit
from moelab.gate_verify import verify

ROOT = Path(__file__).resolve().parents[1]


class GateCLITests(unittest.TestCase):
    def test_controls_and_reproducible_generation(self):
        from scripts.generate_gate_fixtures import fixtures
        actual = fixtures()
        frozen = json.loads((ROOT/'moelab/route_data/gate-fixtures.json').read_text())
        self.assertEqual(len(actual), 10)
        self.assertEqual(actual, frozen)
        for f in frozen:
            self.assertTrue(verify(f['input'], f['report'])['verified'])

    def test_strict_json_and_cli_round_trip(self):
        f = json.loads((ROOT/'moelab/route_data/gate-fixtures.json').read_text())[0]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'input.json'
            r = Path(tmp)/'report.json'
            for text in ['{"x":0,"x":1}', '{"x":0,"\\u0078":1}', 'NaN', '{"x":1.0}',
                         '{"x":1e0}', '['*50+'0'+']'*50, '0'*161, '{} trailing']:
                p.write_text(text)
                with self.assertRaises(ValueError, msg=text):
                    load(p)
            p.write_bytes(b' '*8_000_001)
            with self.assertRaises(ValueError):
                load(p)
            p.write_text(json.dumps(f['input']))
            result = subprocess.run([sys.executable, '-m', 'moelab.gate_cli', 'audit', str(p)],
                                    text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(result.stdout), f['report'])
            r.write_text(result.stdout)
            result = subprocess.run([sys.executable, '-m', 'moelab.gate_cli', 'verify', str(p), str(r)],
                                    text=True, capture_output=True, check=True)
            self.assertTrue(json.loads(result.stdout)['verified'])
            wrong = copy.deepcopy(f['report']); wrong['worst']['count'] = '999'
            r.write_text(json.dumps(wrong))
            result = subprocess.run([sys.executable, '-m', 'moelab.gate_cli', 'verify', str(p), str(r)],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, '')
            self.assertIn('Cannot process gate-table evidence:', result.stderr)

    def test_verifier_has_no_producer_dependency(self):
        f = json.loads((ROOT/'moelab/route_data/gate-fixtures.json').read_text())[0]
        real_import = __import__
        def guarded(name, *args, **kwargs):
            if name.endswith(('gate_regret', 'capacity', 'gate_cli')):
                raise AssertionError('verifier imported producer or producer parser')
            return real_import(name, *args, **kwargs)
        with patch('builtins.__import__', guarded):
            module = importlib.reload(sys.modules['moelab.gate_verify'])
            self.assertTrue(module.verify(f['input'], f['report'])['verified'])

    def test_every_primitive_report_value_is_checked(self):
        f = json.loads((ROOT/'moelab/route_data/gate-fixtures.json').read_text())[0]
        report = f['report']
        def leaves(value, path=()):
            if type(value) is dict:
                for k, v in value.items():
                    yield from leaves(v, path+(k,))
            elif type(value) is list:
                for k, v in enumerate(value):
                    yield from leaves(v, path+(k,))
            else:
                yield path, value
        count = 0
        for path, value in leaves(report):
            altered = copy.deepcopy(report)
            parent = altered
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = not value if type(value) is bool else value+1 if type(value) is int else 'tampered'
            with self.assertRaises(ValueError, msg=str(path)):
                verify(f['input'], altered)
            count += 1
        self.assertGreater(count, 100)
