"""Independent vertex/assignment verifier. Does not import the audit producer.

Valid alternative row primal/dual witnesses are accepted. Assignment ties and
all state records follow the documented canonical ordering. This checks the
declared input and complete report, not its authorship or provenance.
"""
import itertools
import re
from fractions import Fraction as F


def _r(x, bits=256, canonical=False):
    if type(x) is int and not canonical:
        x = str(x)
    if type(x) is not str or len(x) > (160 if bits == 256 else bits*2) or re.fullmatch(r'-?[0-9]+(?:/[1-9][0-9]*)?', x) is None:
        raise ValueError('rational syntax')
    v = F(x)
    if max(v.numerator.bit_length(), v.denominator.bit_length()) > bits:
        raise ValueError('rational size')
    if canonical and str(v) != x:
        raise ValueError('noncanonical output fraction')
    return v


def _input(p):
    if type(p) is not dict or set(p) != {'probabilities', 'capacities', 'eligible', 'lower', 'upper'}:
        raise ValueError('input fields')
    probs = p['probabilities']
    if type(probs) is not list or not 1 <= len(probs) <= 7 or type(probs[0]) is not list or not 2 <= len(probs[0]) <= 4:
        raise ValueError('input bounds')
    n, m = len(probs), len(probs[0])
    matrices = []
    for name in ['probabilities', 'lower', 'upper']:
        matrix = p[name]
        if type(matrix) is not list or len(matrix) != n or any(type(row) is not list or len(row) != m for row in matrix):
            raise ValueError('matrix dimensions')
        matrices.append([[_r(x) for x in row] for row in matrix])
    q, lo, hi = matrices
    cap = p['capacities']
    eligible = p['eligible']
    if type(cap) is not list or len(cap) != m or any(type(c) is not int or not 0 <= c <= n for c in cap):
        raise ValueError('capacities')
    if type(eligible) is not list or len(eligible) != n:
        raise ValueError('eligibility dimensions')
    for i in range(n):
        if sum(q[i]) != 1 or any(not 0 <= lo[i][j] <= q[i][j] <= hi[i][j] <= 1 for j in range(m)):
            raise ValueError('probability bounds')
        row = eligible[i]
        if type(row) is not list or any(type(j) is not int or not 0 <= j < m for j in row) or len(row) != len(set(row)):
            raise ValueError('eligibility')
    return q, list(cap), [sorted(row) for row in eligible], lo, hi


def _vertices(lo, hi):
    result = set()
    for free in range(len(lo)):
        others = [j for j in range(len(lo)) if j != free]
        for ends in itertools.product([0, 1], repeat=len(others)):
            row = [F(0)]*len(lo)
            for j, side in zip(others, ends):
                row[j] = hi[j] if side else lo[j]
            row[free] = 1-sum(row[j] for j in others)
            if lo[free] <= row[free] <= hi[free]:
                result.add(tuple(row))
    if not result:
        raise ValueError('empty simplex')
    return result


def _same(a, b, path='report'):
    if type(a) is not type(b):
        raise ValueError('type mismatch at '+path)
    if type(a) is dict:
        if a.keys() != b.keys():
            raise ValueError('fields differ at '+path)
        for k in a:
            _same(a[k], b[k], path+'/'+k)
    elif type(a) is list:
        if len(a) != len(b):
            raise ValueError('length differs at '+path)
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, path+'/'+str(i))
    elif a != b:
        raise ValueError('value differs at '+path)


def verify(payload, report):
    q, cap, eligible, lo, hi = _input(payload)
    n, m = len(q), len(cap)
    if type(report) is not dict:
        raise ValueError('report object required')
    choices = [row+[-1] for row in eligible]
    # Enumerate complete feasible assignments, then group their actual prefixes.
    routes = [r for r in itertools.product(*choices)
              if all(r.count(j) <= cap[j] for j in range(m))]
    prefixes = set()
    for route in routes:
        used = [0]*m
        prefixes.add((0, tuple(used)))
        for i, j in enumerate(route):
            if j >= 0:
                used[j] += 1
            prefixes.add((i+1, tuple(used)))
    def key(i, used):
        return str(i)+':'+','.join(map(str, used))
    def enumerate_report(weights):
        def tails(i, used):
            pairs = []
            for r in itertools.product(*choices[i:]):
                if any(used[j]+r.count(j) > cap[j] for j in range(m)):
                    continue
                score = sum((weights[i+k][j] for k, j in enumerate(r)), F(0))
                pairs.append((r, (sum(j >= 0 for j in r), score)))
            return pairs
        pairs = tails(0, (0,)*m)
        best = max(s for _, s in pairs)
        opt = [r for r, s in pairs if s == best]
        states = []
        for i, used in sorted(prefixes):
            all_tails = tails(i, used)
            obj = max(s for _, s in all_tails)
            tied = [r for r, s in all_tails if s == obj]
            edges = []
            if i < n:
                for j in choices[i]:
                    following = [s for r, s in all_tails if r[0] == j]
                    if not following:
                        continue
                    successor = tuple(u+int(k == j) for k, u in enumerate(used))
                    val = max(following)
                    edges.append({'expert': j, 'next': key(i+1, successor),
                                  'objective': [val[0], str(val[1])], 'optimal': val == obj})
            states.append({'id': key(i, used), 'token': i, 'used': list(used),
                           'objective': [obj[0], str(obj[1])], 'count': str(len(tied)), 'edges': edges})
        return {'dispatched': best[0], 'score': str(best[1]), 'count': str(len(opt)),
                'canonical': list(opt[0]), 'states': states}
    nominal = enumerate_report([{j: q[i][j] if j >= 0 else F(0) for j in choices[i]} for i in range(n)])
    a = nominal['canonical']
    vertices = [_vertices(lo[i], hi[i]) for i in range(n)]
    def maximum(i, c):
        return max(sum((x*y for x, y in zip(row, c)), F(0)) for row in vertices[i])
    given = report.get('row_supports')
    if type(given) is not list or len(given) != n:
        raise ValueError('row supports dimensions')
    weights = []
    for i in range(n):
        if type(given[i]) is not list or len(given[i]) != len(choices[i]):
            raise ValueError('row action supports')
        local = {}
        for j, row in zip(choices[i], given[i]):
            if type(row) is not dict or set(row) != {'expert', 'coefficient', 'point', 'value', 'lambda', 'dual_terms', 'dual_bound'}:
                raise ValueError('support fields')
            c = [F(int(k == j)-int(k == a[i])) for k in range(m)]
            _same(row['expert'], j)
            _same(row['coefficient'], list(map(str, c)))
            if type(row['point']) is not list or len(row['point']) != m:
                raise ValueError('support point dimensions')
            point = [_r(x, 16384, canonical=True) for x in row['point']]
            if sum(point) != 1 or any(not lo[i][k] <= point[k] <= hi[i][k] for k in range(m)):
                raise ValueError('infeasible support witness')
            value = sum((x*y for x, y in zip(point, c)), F(0))
            if value != maximum(i, c):
                raise ValueError('support is not a vertex-certified optimum')
            lam = _r(row['lambda'], 16384, canonical=True)
            terms = [max((c[k]-lam)*lo[i][k], (c[k]-lam)*hi[i][k]) for k in range(m)]
            if lam+sum(terms) != value:
                raise ValueError('nonzero support dual gap')
            _same(row['value'], str(value))
            _same(row['dual_terms'], list(map(str, terms)))
            _same(row['dual_bound'], str(value))
            local[j] = value
        weights.append(local)
    worst = enumerate_report(weights)
    b = worst['canonical']
    table = [next(row['point'] for row in given[i] if row['expert'] == j) for i, j in enumerate(b)]
    def mass(route):
        return sum((F(table[i][j]) for i, j in enumerate(route) if j >= 0), F(0))
    ref_mass, other_mass = mass(a), mass(b)
    upper_weights = [{j: maximum(i, [F(k == j) for k in range(m)]) for j in choices[i]} for i in range(n)]
    independent_best = enumerate_report(upper_weights)
    ref_min = -sum((maximum(i, [-F(k == a[i]) for k in range(m)]) for i in range(n)), F(0))
    separate = F(independent_best['score'])-ref_min
    evidence = report.get('separate_extrema')
    if type(evidence) is not dict or set(evidence) != {'challenger', 'best_probabilities', 'reference_worst_probabilities', 'best_mass', 'reference_worst_mass'}:
        raise ValueError('separate extrema fields')
    def extreme_table(name, route, sign):
        table = evidence[name]
        if type(table) is not list or len(table) != n:
            raise ValueError('extrema table dimensions')
        for i, row in enumerate(table):
            if type(row) is not list or len(row) != m:
                raise ValueError('extrema row dimensions')
            point = [_r(x, 16384, canonical=True) for x in row]
            if sum(point) != 1 or any(not lo[i][k] <= point[k] <= hi[i][k] for k in range(m)):
                raise ValueError('infeasible extrema table')
            c = [sign*F(k == route[i]) for k in range(m)]
            if sum((x*y for x, y in zip(point, c)), F(0)) != maximum(i, c):
                raise ValueError('nonoptimal extrema table')
        return table
    best_table = extreme_table('best_probabilities', independent_best['canonical'], 1)
    min_table = extreme_table('reference_worst_probabilities', a, -1)
    expected = {'schema': 'moelab-gate-regret-v1',
                'input': {'probabilities': [list(map(str, row)) for row in q], 'capacities': cap,
                          'eligible': eligible, 'lower': [list(map(str, row)) for row in lo],
                          'upper': [list(map(str, row)) for row in hi]},
                'nominal': nominal, 'row_supports': given, 'worst': worst,
                'adversarial_probabilities': table, 'reference_mass': str(ref_mass),
                'challenger_mass': str(other_mass), 'regret': str(other_mass-ref_mass),
                'separate_extrema_bound': str(separate),
                'separate_extrema_gap': str(separate-(other_mass-ref_mass)),
                'separate_extrema': {'challenger': independent_best,
                                    'best_probabilities': best_table,
                                    'reference_worst_probabilities': min_table,
                                    'best_mass': independent_best['score'],
                                    'reference_worst_mass': str(ref_min)}}
    if F(worst['score']) != other_mass-ref_mass:
        raise ValueError('attainment mismatch')
    _same(expected, report)
    return {'verified': True, 'regret': expected['regret'],
            'maximum_coverage': nominal['dispatched'], 'states': len(worst['states'])}
