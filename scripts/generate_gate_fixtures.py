"""Regenerate declared finite controls, then independently replay each report."""
import copy
import json
from pathlib import Path
from moelab.gate_regret import audit
from moelab.gate_verify import verify

ROOT = Path(__file__).resolve().parents[1]


def fixtures():
    half = {'probabilities': [['1/2', '1/2']]*2, 'capacities': [1, 1],
            'eligible': [[0, 1]]*2, 'lower': [[0, 0]]*2, 'upper': [[1, 1]]*2}
    cases = []
    def add(name, description, p):
        cases.append({'id': name, 'description': description, 'input': copy.deepcopy(p)})
    add('simplex_gap', 'Normalization prevents the independent box contrast: regret 1/10, separate extrema bound 1/5.',
        {'probabilities': [['2/5', '3/10', '3/10']], 'capacities': [1, 1, 0], 'eligible': [[0, 1]],
         'lower': [['3/10', '1/5', '3/10']], 'upper': [['3/5', '1/2', '3/10']]})
    add('shared_route', 'Only one dispatch is possible. Shared entries cancel: regret zero, separate extrema bound 1/5.',
        {'probabilities': [['1/2', '1/2']], 'capacities': [1, 1], 'eligible': [[0]],
         'lower': [['2/5', '2/5']], 'upper': [['3/5', '3/5']]})
    add('tied_reference', 'Two nominal dispatch ties. The canonical reference can lose two full units of gate mass.', half)
    p = copy.deepcopy(half); p['lower'] = p['upper'] = copy.deepcopy(p['probabilities'])
    add('fixed_table', 'Zero-width bounds: all nominal optima remain tied and regret is zero.', p)
    p = copy.deepcopy(half); p['capacities'] = [0, 0]
    add('no_capacity', 'All tokens skip. No gate mass is retained under any table.', p)
    add('eligibility', 'Missing eligibility and zero nominal gates remain legal. Skipping counts as a dispatch action, not token deletion.',
        {'probabilities': [[1, 0], [0, 1], [1, 0]], 'capacities': [1, 1], 'eligible': [[], [0], [0, 1]],
         'lower': [[0, 0]]*3, 'upper': [[1, 1]]*3})
    add('capacity_drop', 'Three tokens share two slots. The worst challenger may change which token skips.',
        {'probabilities': [['3/4', '1/4'], ['2/3', '1/3'], ['3/5', '2/5']], 'capacities': [1, 1],
         'eligible': [[0, 1]]*3, 'lower': [['1/4', '0']]*3, 'upper': [['1', '3/4']]*3})
    add('stable_reference', 'Wide enough for mass changes, but the same dispatch stays best throughout these bounds.',
        {'probabilities': [['4/5', '1/5'], ['1/5', '4/5']], 'capacities': [1, 1], 'eligible': [[0, 1]]*2,
         'lower': [['7/10', '1/10'], ['1/10', '7/10']], 'upper': [['9/10', '3/10'], ['3/10', '9/10']]})
    add('tiny_fraction', 'Exact non-decimal fractions retain a small nonzero regret without a floating tolerance.',
        {'probabilities': [['1/2', '1/2']], 'capacities': [1, 1], 'eligible': [[0, 1]],
         'lower': [['499999999999/1000000000000']*2], 'upper': [['500000000001/1000000000000']*2]})
    add('four_experts', 'Asymmetric bounds, expert eligibility and capacity interact across four experts.',
        {'probabilities': [['2/5', '3/10', '1/5', '1/10'], ['1/10', '1/5', '3/10', '2/5'], ['1/4']*4],
         'capacities': [1, 1, 1, 0], 'eligible': [[0, 1, 2], [1, 2, 3], [0, 2]],
         'lower': [['1/5', '1/10', '1/10', '0'], ['0', '1/10', '1/5', '1/5'], ['1/8']*4],
         'upper': [['3/5', '1/2', '2/5', '1/5'], ['1/5', '2/5', '1/2', '3/5'], ['1/2']*4]})
    for case in cases:
        case['report'] = audit(case['input'])
        verify(case['input'], case['report'])
    return cases


if __name__ == '__main__':
    data = fixtures()
    (ROOT/'moelab/route_data/gate-fixtures.json').write_text(json.dumps(data, separators=(',', ':'))+'\n')
    (ROOT/'docs/gate-data.js').write_text('globalThis.GATE_FIXTURES = '+json.dumps(data, separators=(',', ':'))+';\n')
    print(json.dumps([{ 'id': f['id'], 'regret': f['report']['regret'], 'bound': f['report']['separate_extrema_bound']} for f in data], indent=2))
