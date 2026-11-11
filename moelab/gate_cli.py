"""Bounded strict JSON entry points for gate-table regret and independent replay."""
import argparse
import json
import sys
from pathlib import Path


def load(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(8_000_001)
    if len(raw) > 8_000_000:
        raise ValueError('JSON exceeds eight million bytes')
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('floats and nonfinite JSON are not allowed')
    def integer(value):
        if len(value) > 160:
            raise ValueError('JSON integer too long')
        return int(value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad,
                       parse_float=bad, parse_int=integer)
    def depth(item, level=0):
        if level > 48:
            raise ValueError('JSON depth exceeds 48')
        if type(item) is dict:
            for child in item.values():
                depth(child, level+1)
        elif type(item) is list:
            for child in item:
                depth(child, level+1)
    depth(value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description='Exact supplied gate-table regret. Not model or task robustness.')
    sub = parser.add_subparsers(dest='command', required=True)
    audit_parser = sub.add_parser('audit')
    audit_parser.add_argument('input')
    verify_parser = sub.add_parser('verify')
    verify_parser.add_argument('input')
    verify_parser.add_argument('report')
    sub.add_parser('demo')
    args = parser.parse_args(argv)
    try:
        if args.command == 'audit':
            from .gate_regret import audit
            result = audit(load(args.input))
        elif args.command == 'verify':
            from .gate_verify import verify
            result = verify(load(args.input), load(args.report))
        else:
            from .gate_regret import audit
            fixtures = load(Path(__file__).parent/'route_data/gate-fixtures.json')
            result = [{'id': f['id'], 'report': audit(f['input'])} for f in fixtures]
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, RecursionError, OverflowError) as exc:
        print('Cannot process gate-table evidence: '+str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
