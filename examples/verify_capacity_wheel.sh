#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python -m pip wheel --no-deps "$root" --wheel-dir "$tmp/wheels" >/dev/null
python -m venv "$tmp/venv"
"$tmp/venv/bin/python" -m pip install --no-index "$tmp"/wheels/*.whl >/dev/null
cd "$tmp"
"$tmp/venv/bin/python" - <<'PY'
import importlib,json,pathlib
from moelab.capacity import audit
from moelab.capverify import verify
import moelab
for name in ['cli','data','eval','moe','envelope','routes','route_fixtures','capacity','capverify','capacity_cli']:importlib.import_module('moelab.'+name)
fixtures=json.loads((pathlib.Path(moelab.__file__).parent/'route_data/capacity-fixtures.json').read_text())
assert len(fixtures)==14
for item in fixtures:
    assert audit(item['input'])==item['report'];assert verify(item['input'],item['report'])['verified']
pathlib.Path('input.json').write_text(json.dumps(fixtures[0]['input']))
print('Installed 10 modules and all 14 complete exact controls independently replay outside checkout')
PY
"$tmp/venv/bin/python" -m moelab.capacity_cli audit input.json > report.json
"$tmp/venv/bin/python" -m moelab.capacity_cli verify input.json report.json > verified.json
"$tmp/venv/bin/python" -m moelab.capacity_cli demo > demo.json
"$tmp/venv/bin/python" - <<'PY'
import json,pathlib
assert json.load(open('verified.json'))['verified']
assert len(json.load(open('demo.json')))==14
pathlib.Path('bad.json').write_text('{"probabilities":[],"probabilities":[]}')
r=json.load(open('report.json'));r['optimum']['count']='wrong';pathlib.Path('tamper.json').write_text(json.dumps(r))
print('Installed audit, independent verify and demo CLI modes all pass')
PY
if "$tmp/venv/bin/python" -m moelab.capacity_cli audit bad.json >/dev/null 2>&1; then echo 'duplicate accepted';exit 1;fi
if "$tmp/venv/bin/python" -m moelab.capacity_cli verify input.json tamper.json >/dev/null 2>&1; then echo 'tampering accepted';exit 1;fi
