"""Independent vertex and Cartesian-product oracles, not solver self-agreement."""
import copy
import itertools
import random
import unittest
from fractions import Fraction as F

from moelab.gate_regret import audit, support
from moelab.gate_verify import verify


def vertices(lower, upper):
    # At a box/hyperplane vertex all but at most one coordinate are endpoints.
    points = set()
    for free in range(len(lower)):
        rest = [j for j in range(len(lower)) if j != free]
        for bits in itertools.product([0, 1], repeat=len(rest)):
            p = [None]*len(lower)
            for j, bit in zip(rest, bits):
                p[j] = upper[j] if bit else lower[j]
            p[free] = 1-sum(p[j] for j in rest)
            if lower[free] <= p[free] <= upper[free]:
                points.add(tuple(p))
    return sorted(points)


def row_value(lower, upper, coefficients):
    return max(sum((x*c for x, c in zip(p, coefficients)), F(0))
               for p in vertices(lower, upper))


def oracle(payload):
    q = [[F(x) for x in row] for row in payload['probabilities']]
    lower = [[F(x) for x in row] for row in payload['lower']]
    upper = [[F(x) for x in row] for row in payload['upper']]
    n, m = len(q), len(q[0])
    cap = payload['capacities']
    options = [sorted(row)+[-1] for row in payload['eligible']]
    routes = [a for a in itertools.product(*options)
              if all(a.count(j) <= cap[j] for j in range(m))]
    coverage = max(sum(j >= 0 for j in a) for a in routes)
    full = [a for a in routes if sum(j >= 0 for j in a) == coverage]
    def nominal(a):
        return sum((q[i][j] for i, j in enumerate(a) if j >= 0), F(0))
    best = max(map(nominal, full))
    nominal_opt = [a for a in full if nominal(a) == best]
    a = nominal_opt[0]
    local = [{b: row_value(lower[i], upper[i],
                          [F(int(k == b)-int(k == a[i])) for k in range(m)])
              for b in options[i]} for i in range(n)]
    def tails(i, used):
        values = []
        for b in itertools.product(*options[i:]):
            if any(used[j]+b.count(j) > cap[j] for j in range(m)):
                continue
            score = sum((local[i+k][j] for k, j in enumerate(b)), F(0))
            values.append((b, (sum(j >= 0 for j in b), score)))
        best = max(x[1] for x in values)
        tied = [b for b, score in values if score == best]
        return best, tied
    worst, tied = tails(0, [0]*m)
    return {'coverage': coverage, 'mass': best, 'reference': list(a),
            'nominal_count': len(nominal_opt), 'regret': worst[1],
            'challenger': list(tied[0]), 'worst_count': len(tied), 'tails': tails}


class GateRegretTests(unittest.TestCase):
    def payload(self):
        return {'probabilities': [['2/5', '3/10', '3/10']],
                'capacities': [1, 1, 0], 'eligible': [[0, 1]],
                'lower': [['3/10', '1/5', '3/10']],
                'upper': [['3/5', '1/2', '3/10']]}

    def compare(self, p, all_states=True):
        before = copy.deepcopy(p)
        r = audit(p)
        self.assertEqual(p, before)
        self.assertTrue(verify(p, r)["verified"])
        o = oracle(p)
        self.assertEqual(r['nominal']['dispatched'], o['coverage'])
        self.assertEqual(F(r['nominal']['score']), o['mass'])
        self.assertEqual(r['nominal']['canonical'], o['reference'])
        self.assertEqual(int(r['nominal']['count']), o['nominal_count'])
        self.assertEqual(F(r['regret']), o['regret'])
        self.assertEqual(r['worst']['canonical'], o['challenger'])
        self.assertEqual(int(r['worst']['count']), o['worst_count'])
        self.assertGreaterEqual(F(r['separate_extrema_bound']), F(r['regret']))
        # Check every emitted state/edge against enumerated suffix assignments.
        if all_states:
            for s in r['worst']['states']:
                expected, tied = o['tails'](s['token'], s['used'])
                self.assertEqual(s['objective'], [expected[0], str(expected[1])])
                self.assertEqual(int(s['count']), len(tied))
                for edge in s['edges']:
                    b = edge['expert']
                    self.assertEqual(edge['optimal'], any(t[0] == b for t in tied))
        table = [[F(x) for x in row] for row in r['adversarial_probabilities']]
        for i, row in enumerate(table):
            self.assertEqual(sum(row), 1)
            for j, x in enumerate(row):
                self.assertTrue(F(p['lower'][i][j]) <= x <= F(p['upper'][i][j]))
        def mass(route):
            return sum((table[i][j] for i, j in enumerate(route) if j >= 0), F(0))
        self.assertEqual(mass(o['challenger'])-mass(o['reference']), o['regret'])
        # At its attaining table B must itself maximize mass among all
        # maximum-coverage routes, not merely beat the fixed reference.
        choices = [sorted(row)+[-1] for row in p['eligible']]
        for route in itertools.product(*choices):
            if sum(j >= 0 for j in route) != o['coverage']:
                continue
            if any(route.count(j) > p['capacities'][j] for j in range(len(p['capacities']))):
                continue
            self.assertLessEqual(mass(route), mass(o['challenger']))
        return r

    def test_simplex_coupling_changes_the_box_answer(self):
        r = self.compare(self.payload())
        self.assertEqual(r['regret'], '1/10')
        self.assertEqual(r['adversarial_probabilities'], [['3/10', '2/5', '3/10']])
        self.assertEqual(r['separate_extrema_bound'], '1/5')

    def test_one_shared_table_not_two_independent_extrema(self):
        p = {'probabilities': [['1/2', '1/2']], 'capacities': [1, 1],
             'eligible': [[0]], 'lower': [['2/5', '2/5']], 'upper': [['3/5', '3/5']]}
        r = self.compare(p)
        self.assertEqual(r['regret'], '0')
        self.assertEqual(r['separate_extrema_bound'], '1/5')

    def test_zero_width_recovers_nominal_optima(self):
        p = self.payload()
        p['lower'] = p['upper'] = copy.deepcopy(p['probabilities'])
        self.assertEqual(self.compare(p)['regret'], '0')

    def test_tied_nominal_routes_and_regret_challengers(self):
        p = {'probabilities': [['1/2', '1/2']]*2, 'capacities': [1, 1],
             'eligible': [[0, 1]]*2, 'lower': [[0, 0]]*2, 'upper': [[1, 1]]*2}
        r = self.compare(p)
        self.assertEqual(r['regret'], '2')
        self.assertEqual(r['nominal']['count'], '2')

    def test_no_coverage_all_skips(self):
        p = self.payload()
        p['capacities'] = [0]*3
        r = self.compare(p)
        self.assertEqual(r['regret'], '0')
        self.assertEqual(r['worst']['canonical'], [-1])

    def test_missing_eligibility_and_zero_gates(self):
        p = {'probabilities': [[1, 0], [0, 1], [1, 0]], 'capacities': [1, 1],
             'eligible': [[], [0], [0, 1]], 'lower': [[0, 0]]*3, 'upper': [[1, 1]]*3}
        self.compare(p)

    def test_random_tables_and_every_state(self):
        rng = random.Random(20401)
        for _ in range(90):
            m = rng.randint(2, 4)
            n = rng.randint(1, 4)
            q, lower, upper = [], [], []
            for i in range(n):
                raw = [rng.randint(0, 5) for j in range(m)]
                if not sum(raw):
                    raw[0] = 1
                row = [F(x, sum(raw)) for x in raw]
                q.append(list(map(str, row)))
                lower.append([str(max(F(0), x-F(rng.randint(0, 3), 8))) for x in row])
                upper.append([str(min(F(1), x+F(rng.randint(0, 3), 8))) for x in row])
            p = {'probabilities': q, 'lower': lower, 'upper': upper,
                 'capacities': [rng.randint(0, n) for j in range(m)],
                 'eligible': [[j for j in range(m) if rng.randrange(3)] for i in range(n)]}
            self.compare(p)

    def test_support_vertices_and_dual_gap(self):
        rng = random.Random(9081)
        for _ in range(500):
            m = rng.randint(2, 4)
            raw = [rng.randint(1, 9) for _ in range(m)]
            p = [F(x, sum(raw)) for x in raw]
            lo = [max(F(0), x-F(rng.randrange(4), 5)) for x in p]
            hi = [min(F(1), x+F(rng.randrange(4), 5)) for x in p]
            c = [F(rng.randint(-4, 4), 3) for j in range(m)]
            r = support(lo, hi, c)
            self.assertEqual(F(r['value']), row_value(lo, hi, c))
            self.assertEqual(r['value'], r['dual_bound'])

    def test_input_validation(self):
        original = self.payload()
        bad = []
        p = copy.deepcopy(original);p['extra'] = 1;bad.append(p)
        p = copy.deepcopy(original);p['capacities'][0] = True;bad.append(p)
        p = copy.deepcopy(original);p['lower'][0][0] = '1/2';bad.append(p)
        p = copy.deepcopy(original);p['upper'][0][0] = '1/5';bad.append(p)
        p = copy.deepcopy(original);p['lower'][0][0] = False;bad.append(p)
        p = copy.deepcopy(original);p['eligible'][0] = [0, 0];bad.append(p)
        p = copy.deepcopy(original);p['upper'][0][0] = '2';bad.append(p)
        p = copy.deepcopy(original);p['lower'] = [];bad.append(p)
        p = copy.deepcopy(original);p['lower'][0][0] = '٣/١٠';bad.append(p)
        for p in bad:
            with self.assertRaises(ValueError):
                audit(p)

    def test_report_has_no_mutable_input_alias(self):
        p = self.payload()
        before = copy.deepcopy(p)
        r = audit(p)
        r['input']['capacities'][0] = 0
        r['input']['eligible'][0].clear()
        self.assertEqual(p, before)

    def test_independent_verifier_rejects_changed_evidence(self):
        p = self.payload()
        original = audit(p)
        mutations = [
            ('regret', '1/5'), ('reference_mass', '0'),
            ('adversarial_probabilities', [['3/10', '1/2', '3/10']]),
            ('schema', 'another-schema'),
        ]
        for key, value in mutations:
            r = copy.deepcopy(original)
            r[key] = value
            with self.assertRaises(ValueError):
                verify(p, r)
        for field in ['value', 'dual_bound', 'lambda']:
            r = copy.deepcopy(original)
            r['row_supports'][0][1][field] = '99'
            with self.assertRaises(ValueError):
                verify(p, r)
        r = copy.deepcopy(original)
        r['worst']['states'][0]['edges'][0]['optimal'] = 1
        with self.assertRaises(ValueError):
            verify(p, r)

    def test_full_bounds_and_seven_rows(self):
        p = {'probabilities': [['1/2', '1/2']]*7, 'capacities': [3, 3],
             'eligible': [[0, 1]]*7, 'lower': [[0, 0]]*7, 'upper': [[1, 1]]*7}
        self.compare(p, all_states=False)

    def test_wider_intervals_never_reduce_fixed_reference_regret(self):
        p = self.payload()
        values = []
        for denominator in [100, 40, 20, 10, 5, 2, 1]:
            p['lower'] = [[str(max(F(0), F(x)-F(1, denominator))) for x in row]
                          for row in p['probabilities']]
            p['upper'] = [[str(min(F(1), F(x)+F(1, denominator))) for x in row]
                          for row in p['probabilities']]
            values.append(F(self.compare(p)['regret']))
        self.assertEqual(values, sorted(values))

    def test_unique_reference_is_expert_permutation_equivariant(self):
        p = self.payload()
        r = self.compare(p)
        for perm in itertools.permutations(range(3)):
            old_to_new = {j: perm.index(j) for j in perm}
            other = copy.deepcopy(p)
            for name in ['probabilities', 'lower', 'upper']:
                other[name] = [[row[j] for j in perm] for row in p[name]]
            other['capacities'] = [p['capacities'][j] for j in perm]
            other['eligible'] = [[old_to_new[j] for j in row] for row in p['eligible']]
            actual = self.compare(other)
            self.assertEqual(actual['regret'], r['regret'])
            self.assertEqual(actual['nominal']['canonical'], [old_to_new[j] for j in r['nominal']['canonical']])

    def test_largest_shape_with_all_actions(self):
        p = {'probabilities': [['1/4']*4]*7, 'capacities': [2]*4,
             'eligible': [list(range(4))]*7, 'lower': [[0]*4]*7, 'upper': [[1]*4]*7}
        r = audit(p)
        self.assertEqual(r['regret'], '7')
        self.assertEqual(len(r['worst']['states']), 324)
        self.assertTrue(verify(p, r)['verified'])
