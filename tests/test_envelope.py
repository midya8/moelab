import unittest
from fractions import Fraction
from moelab.envelope import envelope

class TestEnvelope(unittest.TestCase):
    def test_half_crossing(self):
        r=envelope([0,1],[1,-1]); p=r['points'][1]
        self.assertEqual(p['at']['numerator'],'1'); self.assertEqual(p['at']['denominator'],'2')
        self.assertEqual([c['winner'] for c in r['cells']],[1,0]);self.assertTrue(p['route_changes']);self.assertEqual(p['winner'],0)
    def test_hidden_pair_crossing(self):
        r=envelope([0,1,5],[1,-1,0]);self.assertTrue(any(p['at']['value']==.5 for p in r['points']));self.assertFalse(any(p['route_changes']for p in r['points']));self.assertTrue(all(c['winner']==2 for c in r['cells']))
    def test_identical_lines(self):
        r=envelope([2,2,0],[0,0,0]);self.assertEqual(r['cells'][0]['co_maximal'],[0,1]);self.assertTrue(r['parallel_pairs'][0]['identical'])
    def test_triple_crossing(self):
        r=envelope([0,1,.5],[1,-1,0]);p=r['points'][1];self.assertEqual(p['co_maximal'],[0,1,2]);self.assertEqual(p['winner'],0)
    def test_endpoint_only_tie(self):
        r=envelope([0,0],[1,0]);self.assertEqual(r['points'][0]['co_maximal'],[0,1]);self.assertIsNone(r['points'][0]['left_winner']);self.assertFalse(r['points'][0]['route_changes'])
    def test_subgrid_cell(self):
        # The center expert owns an interval much narrower than a plot grid.
        d=2**-40;r=envelope([0,.5+d,1],[1,0,-1]);cells=[c for c in r['cells']if c['winner']==1];self.assertTrue(cells)
        self.assertAlmostEqual(sum(c['right']['value']-c['left']['value'] for c in cells),2*d)
    def test_binary_rational(self):
        r=envelope([0,.1],[1,0]);p=r['points'][1]['at'];self.assertEqual(Fraction(int(p['numerator']),int(p['denominator'])),Fraction(.1))
    def test_reject_bad(self):
        for a,b,lo,hi in [([True,0],[0,1],0,1),([float('nan'),0],[0,1],0,1),([0],[0],0,1),([0,1],[0,1],1,1),([0,1],[0],0,1),([0,1e7],[0,1],0,1)]:
            with self.assertRaises(ValueError):envelope(a,b,lo,hi)
    def test_sampled_independent_integer_scores(self):
        # Independent exact integer comparisons at 201 non-boundary probes.
        a=[4,-1,2,-8];b=[-7,9,0,15];r=envelope(a,b,-2,3)
        for k in range(201):
            t=Fraction(-2)+Fraction(5*k,200);hits=[c for c in r['cells']if Fraction(int(c['left']['numerator']),int(c['left']['denominator']))<t<Fraction(int(c['right']['numerator']),int(c['right']['denominator']))]
            if hits:self.assertEqual(hits[0]['winner'],max(range(4),key=lambda i:a[i]+b[i]*t))

    def test_outside_crossing_overflow_is_json_safe(self):
        import json
        r=envelope([0,1],[0,5e-324]);p=r['pair_crossings'][0]
        self.assertFalse(p['in_domain']);self.assertIsNone(p['at']['value']);json.dumps(r,allow_nan=False)
    def test_huge_integer_rejected_not_overflow(self):
        with self.assertRaises(ValueError):envelope([10**1000,0],[1,0])
