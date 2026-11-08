import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from moelab.routes import audit

class TestRouteCLI(unittest.TestCase):
    def call(self,body,*args):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'input.json';p.write_text(body)
            return subprocess.run([sys.executable,'-m','moelab','route-audit',str(p),*args],capture_output=True,text=True)
    def test_native_command_matches_library(self):
        f=json.loads((Path(__file__).resolve().parents[1]/'moelab/route_data/frozen.json').read_text());p=f['controls']['triple_tie'];r=self.call(json.dumps(p),'--probe','.5');self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout),audit(p))
    def test_duplicate_nonfinite_and_oversize_rejected(self):
        for body in ['{"model":0,"model":1}','{"model":NaN}',' '*1000001]:
            r=self.call(body);self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'');self.assertIn('Cannot audit',r.stderr)
    def test_bounds_and_bad_probe_rejected(self):
        r=self.call('{}','--probe','nan');self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'')
    def test_help_exposes_both_new_commands(self):
        r=subprocess.run([sys.executable,'-m','moelab','--help'],capture_output=True,text=True);self.assertEqual(r.returncode,0);self.assertIn('route-audit',r.stdout);self.assertIn('route-demo',r.stdout)
