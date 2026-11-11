"""Exact retained-gate-mass regret on a supplied bounded simplex table.

The reference is a nominally optimal maximum-coverage assignment. This audits
its worst objective regret, not a minmax-regret choice of reference, model
uncertainty, input perturbations, task error, training or serving performance.
"""
from fractions import Fraction as F
from functools import lru_cache
import re

from .capacity import rat, validate


def parse(payload):
    keys = {'probabilities', 'capacities', 'eligible', 'lower', 'upper'}
    if type(payload) is not dict or set(payload) != keys:
        raise ValueError('expected probabilities, capacities, eligible, lower, upper')
    p = payload['probabilities']
    if type(p) is not list or not p or type(p[0]) is not list:
        raise ValueError('probability rows required')
    n, m = len(p), len(p[0])
    if not 1 <= n <= 7 or not 2 <= m <= 4:
        raise ValueError('1..7 tokens, 2..4 experts required')
    def gate_rat(value):
        if type(value) is str and not re.fullmatch(r'-?[0-9]+(?:/[1-9][0-9]*)?', value):
            raise ValueError('ASCII rational syntax required')
        return rat(value)
    for row in p:
        if type(row) is not list or len(row) != m:
            raise ValueError('probability dimensions')
        for value in row:
            gate_rat(value)
    q, cap, eligible, _, _ = validate({
        'probabilities': p, 'capacities': payload['capacities'],
        'eligible': payload['eligible'], 'expert_values': [[0]*m for _ in p],
        'targets': [0]*n,
    })
    bounds = []
    for name in ['lower', 'upper']:
        rows = payload[name]
        if type(rows) is not list or len(rows) != n:
            raise ValueError('bound dimensions')
        out = []
        for row in rows:
            if type(row) is not list or len(row) != m:
                raise ValueError('bound dimensions')
            out.append(list(map(gate_rat, row)))
        bounds.append(out)
    lower, upper = bounds
    for i in range(n):
        if any(not F(0) <= lower[i][j] <= q[i][j] <= upper[i][j] <= F(1)
               for j in range(m)):
            raise ValueError('bounds must contain the nominal probability row in [0,1]')
    return q, list(cap), [list(row) for row in eligible], lower, upper


def support(lower, upper, coefficient):
    """Linear support of one bounded probability simplex, with a dual bound."""
    if not (len(lower) == len(upper) == len(coefficient)) or not lower:
        raise ValueError('support dimensions')
    if any(not F(0) <= l <= u <= F(1) for l, u in zip(lower, upper)):
        raise ValueError('support bounds')
    remaining = F(1) - sum(lower)
    if remaining < 0 or sum(upper) < 1:
        raise ValueError('empty bounded simplex')
    point = list(lower)
    order = sorted(range(len(lower)), key=lambda j: (-coefficient[j], j))
    lam = max(coefficient)
    for j in order:
        amount = min(remaining, upper[j] - lower[j])
        point[j] += amount
        remaining -= amount
        if amount:
            lam = coefficient[j]
        if not remaining:
            break
    assert not remaining
    value = sum((c*x for c, x in zip(coefficient, point)), F(0))
    terms = [max((c-lam)*l, (c-lam)*u)
             for c, l, u in zip(coefficient, lower, upper)]
    dual = lam + sum(terms)
    assert value == dual
    return {'coefficient': list(map(str, coefficient)), 'point': list(map(str, point)),
            'value': str(value), 'lambda': str(lam), 'dual_terms': list(map(str, terms)),
            'dual_bound': str(dual)}


def solve(cap, eligible, weights):
    """Maximize coverage, then additive score; retain all feasible state edges."""
    n, m = len(eligible), len(cap)
    def actions(i, used):
        return [j for j in eligible[i] if used[j] < cap[j]] + [-1]
    def step(used, j):
        return tuple(x + (k == j) for k, x in enumerate(used))
    @lru_cache(None)
    def best(i, used):
        if i == n:
            return (0, F(0)), 1, ()
        entries = []
        for j in actions(i, used):
            obj, count, _ = best(i+1, step(used, j))
            entries.append((j, (obj[0]+(j >= 0), obj[1]+weights[i][j]), count))
        optimum = max(row[1] for row in entries)
        return (optimum, sum(row[2] for row in entries if row[1] == optimum),
                tuple(row[0] for row in entries if row[1] == optimum))
    def key(i, used):
        return str(i)+':'+','.join(map(str, used))
    zero = (0,)*m
    objective, count, _ = best(0, zero)
    assignment = []
    used = zero
    for i in range(n):
        j = best(i, used)[2][0]
        assignment.append(j)
        used = step(used, j)
    layer = {zero}
    states = []
    for i in range(n+1):
        following = set()
        for used in sorted(layer):
            obj, ways, optimal = best(i, used)
            edges = []
            if i < n:
                for j in actions(i, used):
                    nxt = step(used, j)
                    following.add(nxt)
                    tail = best(i+1, nxt)[0]
                    score = (tail[0]+(j >= 0), tail[1]+weights[i][j])
                    edges.append({'expert': j, 'next': key(i+1, nxt),
                                  'objective': [score[0], str(score[1])],
                                  'optimal': j in optimal})
            states.append({'id': key(i, used), 'token': i, 'used': list(used),
                           'objective': [obj[0], str(obj[1])], 'count': str(ways),
                           'edges': edges})
        layer = following
    return {'dispatched': objective[0], 'score': str(objective[1]),
            'count': str(count), 'canonical': assignment, 'states': states}


def audit(payload):
    q, cap, eligible, lower, upper = parse(payload)
    n, m = len(q), len(cap)
    choices = [row+[-1] for row in eligible]
    nominal = solve(cap, eligible, [{j: row[j] if j >= 0 else F(0)
                                    for j in choices[i]} for i, row in enumerate(q)])
    reference = nominal['canonical']
    rows = []
    weights = []
    for i in range(n):
        records = []
        scores = {}
        for j in choices[i]:
            coeff = [F(int(k == j)-int(k == reference[i])) for k in range(m)]
            record = support(lower[i], upper[i], coeff)
            records.append({'expert': j, **record})
            scores[j] = F(record['value'])
        rows.append(records)
        weights.append(scores)
    worst = solve(cap, eligible, weights)
    challenger = worst['canonical']
    table = [[F(x) for x in next(r for r in rows[i] if r['expert'] == j)['point']]
             for i, j in enumerate(challenger)]
    def mass(route, table):
        return sum((table[i][j] for i, j in enumerate(route) if j >= 0), F(0))
    reference_mass, challenger_mass = mass(reference, table), mass(challenger, table)
    assert worst['dispatched'] == nominal['dispatched']
    assert challenger_mass-reference_mass == F(worst['score']) >= 0
    # These separate extrema may use incompatible gate tables. They give an
    # upper bound, not an attainable regret witness on one shared table.
    upper_scores = []
    min_reference = F(0)
    for i in range(n):
        upper_scores.append({j: F(support(lower[i], upper[i], [F(k == j) for k in range(m)])['value'])
                             for j in choices[i]})
        min_reference -= F(support(lower[i], upper[i], [-F(k == reference[i]) for k in range(m)])['value'])
    separate = solve(cap, eligible, upper_scores)
    separate_bound = F(separate['score'])-min_reference
    assert separate_bound >= F(worst['score'])
    best_table = [support(lower[i], upper[i], [F(k == j) for k in range(m)])['point']
                  for i, j in enumerate(separate['canonical'])]
    worst_reference_table = [support(lower[i], upper[i], [-F(k == reference[i]) for k in range(m)])['point']
                             for i in range(n)]
    return {'schema': 'moelab-gate-regret-v1',
            'input': {'probabilities': [list(map(str, r)) for r in q], 'capacities': cap,
                      'eligible': eligible, 'lower': [list(map(str, r)) for r in lower],
                      'upper': [list(map(str, r)) for r in upper]},
            'nominal': nominal, 'row_supports': rows, 'worst': worst,
            'adversarial_probabilities': [list(map(str, r)) for r in table],
            'reference_mass': str(reference_mass), 'challenger_mass': str(challenger_mass),
            'regret': worst['score'], 'separate_extrema_bound': str(separate_bound),
            'separate_extrema_gap': str(separate_bound-F(worst['score'])),
            'separate_extrema': {'challenger': separate,
                                'best_probabilities': best_table,
                                'reference_worst_probabilities': worst_reference_table,
                                'best_mass': separate['score'],
                                'reference_worst_mass': str(min_reference)}}
