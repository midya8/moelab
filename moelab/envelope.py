"""Exact top-1 envelope of affine scores on a bounded scalar interval.

Fractions represent the exact rational values of supplied finite binary64
coefficients, not the rounded result of a device's dot-product evaluation.
All pair crossings are enumerated. Only exact rational comparisons determine
open-cell ownership and point ties. Lowest expert index breaks point ties.
This module does not approximate top-1 routing by a sampled grid.
"""
from fractions import Fraction
import math


def _number(value):
    if type(value) not in (int, float) or abs(value) > 1e6 or not math.isfinite(value):
        raise ValueError('expected a finite numeric coefficient within +/-1e6')
    return Fraction(value)


def _rational(value):
    try:
        approximate = float(value)
        if not math.isfinite(approximate):
            approximate = None
    except OverflowError:
        approximate = None
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator),
            'value': approximate}


def envelope(intercepts, slopes, lower=0, upper=1):
    """Return all pair intersections, open cells and exact point-tie decisions.

    A score is intercept[e]+lambda*slope[e]. Parallel identical lines remain
    co-maximal throughout their cells. A nonwinning pair crossing remains in
    the audit's candidate list but cannot be mistaken for a route transition.
    Endpoint events have only one in-domain neighboring cell.
    """
    if not isinstance(intercepts, list) or not isinstance(slopes, list):
        raise ValueError('coefficients must be lists')
    if not 2 <= len(intercepts) <= 16 or len(intercepts) != len(slopes):
        raise ValueError('expected 2..16 matching affine scores')
    a, b = list(map(_number, intercepts)), list(map(_number, slopes))
    lo, hi = _number(lower), _number(upper)
    if lo >= hi:
        raise ValueError('lower must be below upper')
    return rational_envelope(a,b,lo,hi)


def rational_envelope(a,b,lo,hi):
    # Internal, validated Fraction coefficients supplied by the model-path audit.
    knots = {lo, hi}
    pairs, parallel = [], []
    for i in range(len(a)):
        for j in range(i+1, len(a)):
            if b[i] == b[j]:
                parallel.append({'pair': [i,j], 'identical': a[i] == a[j]})
            else:
                t = (a[j]-a[i])/(b[i]-b[j])
                inside = lo <= t <= hi
                pairs.append({'pair': [i,j], 'at': _rational(t), 'in_domain': inside})
                if inside:
                    knots.add(t)
    knots = sorted(knots)
    def tied(t):
        scores = [ai+t*bi for ai,bi in zip(a,b)]
        best = max(scores)
        return [i for i,s in enumerate(scores) if s == best]
    cells = []
    for left,right in zip(knots, knots[1:]):
        winners = tied((left+right)/2)
        cells.append({'left': _rational(left), 'right': _rational(right),
                      'winner': winners[0], 'co_maximal': winners})
    points = []
    for i,t in enumerate(knots):
        winners = tied(t)
        left = cells[i-1]['winner'] if i else None
        right = cells[i]['winner'] if i < len(cells) else None
        points.append({'at': _rational(t), 'winner': winners[0], 'co_maximal': winners,
                       'left_winner': left, 'right_winner': right,
                       'route_changes': left is not None and right is not None and left != right})
    return {'schema': 'moelab-envelope-v1', 'arithmetic': 'exact-rational-binary64-coefficients',
            'tie_rule': 'lowest-index', 'pair_crossings': pairs, 'parallel_pairs': parallel,
            'cells': cells, 'points': points}
