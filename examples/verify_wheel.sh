#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python -m pip wheel --no-deps "$root" --wheel-dir "$tmp/wheels" >/dev/null
python -m venv "$tmp/venv"
"$tmp/venv/bin/python" -m pip install --no-index "$tmp"/wheels/*.whl >/dev/null
cd "$tmp"
"$tmp/venv/bin/python" -m moelab route-demo regenerated > demo.json
"$tmp/venv/bin/python" -m moelab evaluate --format json > evaluate.json
"$tmp/venv/bin/python" -m moelab gates --max-mse .06 --min-purity .95 --max-expert-share .26 > gates.txt
"$tmp/venv/bin/python" - <<'PY'
import gzip,importlib,json,pathlib,math
for name in ['cli','data','eval','moe','envelope','routes','route_fixtures']:importlib.import_module('moelab.'+name)
import moelab
from moelab.routes import audit
root=pathlib.Path(moelab.__file__).parent/'route_data'
f=json.loads((root/'frozen.json').read_text());r=json.loads(pathlib.Path('regenerated/frozen.json').read_text());assert f==r
w=json.loads(gzip.decompress((root/'3-witnesses.json.gz').read_bytes()));v=json.loads(gzip.decompress(pathlib.Path('regenerated/3-witnesses.json.gz').read_bytes()));assert w==v;assert len(w)==32
assert json.load(open('evaluate.json'))==f['original_report']==json.load(open('demo.json'))['original_report']
assert 'All gates passed' in open('gates.txt').read()
pathlib.Path('input.json').write_text(json.dumps(f['controls']['jump']))
print('7 modules import; original aux/noaux/dense training, full models/data/report and all 32 witnesses regenerate outside checkout')
PY
"$tmp/venv/bin/python" -m moelab route-audit input.json --probe .5 > audit.json
"$tmp/venv/bin/python" - <<'PY'
import json
r=json.load(open('audit.json'));assert next(p for p in r['topology']['points']if p['route_changes'])['jump']==1
print('All four commands execute; installed native jump equals one')
PY
