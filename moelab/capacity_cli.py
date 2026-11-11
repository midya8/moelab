"""Strict bounded JSON CLI for capacity dispatch and independent evidence replay."""
import argparse
import json
import sys
from pathlib import Path


def load(path):
    raw=Path(path).read_bytes()
    if len(raw)>8_000_000:raise ValueError('JSON exceeds eight million bytes')
    def pairs(rows):
        out={}
        for k,v in rows:
            if k in out:raise ValueError('duplicate JSON key')
            out[k]=v
        return out
    def bad(x):raise ValueError('nonfinite JSON')
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad,parse_float=bad)
    def depth(x,d=0):
        if d>48:raise ValueError('JSON depth exceeds 48')
        if isinstance(x,dict):
            for v in x.values():depth(v,d+1)
        elif isinstance(x,list):
            for v in x:depth(v,d+1)
    depth(value);return value


def main(argv=None):
    parser=argparse.ArgumentParser(description='Exact finite capacity dispatch. No training or throughput claim.')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('audit');p.add_argument('input')
    p=sub.add_parser('verify');p.add_argument('input');p.add_argument('report')
    sub.add_parser('demo')
    args=parser.parse_args(argv)
    try:
        if args.command=='audit':
            from .capacity import audit
            result=audit(load(args.input))
        elif args.command=='verify':
            from .capverify import verify
            result=verify(load(args.input),load(args.report))
        else:
            from .capacity import audit
            result=[{'id':r['id'],'report':audit(r['input'])}for r in load(Path(__file__).parent/'route_data/capacity-fixtures.json')]
        print(json.dumps(result,sort_keys=True,allow_nan=False));return 0
    except (ValueError,OSError,RecursionError,OverflowError)as exc:
        print('Cannot process capacity evidence: '+str(exc),file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())
