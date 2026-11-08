import json,math,sys
from pathlib import Path
import unittest
from moelab.route_fixtures import build

class TestRouteFixtures(unittest.TestCase):
    def test_original_training_and_entire_report_reproduce(self):
        frozen=json.loads((Path(__file__).resolve().parents[1]/'moelab/route_data/frozen.json').read_text())
        actual=build()
        def compare(a,b):
            if isinstance(a,dict):
                self.assertEqual(a.keys(),b.keys())
                for k in a:compare(a[k],b[k])
            elif isinstance(a,list):
                self.assertEqual(len(a),len(b))
                for x,y in zip(a,b):compare(x,y)
            elif type(a)in(int,float):self.assertTrue(math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11),(a,b))
            else:self.assertEqual(a,b)
        compare(actual,frozen)
        self.assertEqual(len(frozen['splits']['original_train']),240)
        self.assertEqual(len(frozen['splits']['original_test']),100)
        self.assertEqual(set(frozen['models']),{'aux','no_aux','initialization'})
