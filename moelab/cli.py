"""MoeLab CLI: evaluate mixture-of-experts routing, enforce gates both sides."""

from __future__ import annotations

import argparse
import json
import sys

from .eval import format_terminal, run_evaluation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="moelab",
        description="Mixture-of-experts from scratch with measured routing, specialization, and load balance.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--format", choices=("terminal", "json"), default="terminal")

    p_gate = sub.add_parser("gates")
    p_gate.add_argument("--max-mse", type=float, default=None,
                        help="maximum MoE top-1 test MSE")
    p_gate.add_argument("--min-purity", type=float, default=None,
                        help="minimum worst-regime routing purity")
    p_gate.add_argument("--max-expert-share", type=float, default=None,
                        help="maximum single-expert usage share with the aux loss on")

    p_route=sub.add_parser('route-audit',help='audit exact route cells and numeric predictions')
    p_route.add_argument('input')
    p_route.add_argument('--probe',type=float,default=.5)
    p_demo=sub.add_parser('route-demo',help='regenerate original training and routing witnesses')
    p_demo.add_argument('output')
    args = parser.parse_args(argv)
    if args.command=='route-audit':
        from pathlib import Path
        from .routes import audit
        try:
            data=Path(args.input).read_bytes()
            if len(data)>1000000:raise ValueError('JSON input exceeds one million bytes')
            def strict(pairs):
                out={}
                for key,value in pairs:
                    if key in out:raise ValueError('duplicate JSON key')
                    out[key]=value
                return out
            def invalid(value):raise ValueError('nonfinite JSON number')
            payload=json.loads(data,object_pairs_hook=strict,parse_constant=invalid)
            # Match the browser's bounded structural nesting contract.
            def depth(value,n=0):
                if n>64:raise ValueError('JSON nesting exceeds 64')
                if isinstance(value,dict):
                    for item in value.values():depth(item,n+1)
                elif isinstance(value,list):
                    for item in value:depth(item,n+1)
            depth(payload)
            print(json.dumps(audit(payload,args.probe),sort_keys=True,allow_nan=False))
            return 0
        except (OSError,ValueError,RecursionError)as exc:
            print('Cannot audit: '+str(exc),file=sys.stderr)
            return 2
    if args.command=='route-demo':
        from pathlib import Path
        from .route_fixtures import write
        print(json.dumps(write(Path(args.output)/'frozen.json'),sort_keys=True))
        return 0
    results = run_evaluation()

    if args.command == "evaluate":
        print(json.dumps(results, indent=2, sort_keys=True) if args.format == "json"
              else format_terminal(results))
        return 0

    failures = []
    if args.max_mse is not None and results["moe_mse_top1"] > args.max_mse:
        failures.append(f"MoE top-1 MSE {results['moe_mse_top1']:.4f} > allowed {args.max_mse}")
    if args.min_purity is not None and results["routing_purity_min"] < args.min_purity:
        failures.append(f"routing purity {results['routing_purity_min']:.4f} < required {args.min_purity}")
    if args.max_expert_share is not None and results["aux_max_share"] > args.max_expert_share:
        failures.append(f"max expert share {results['aux_max_share']:.4f} > allowed {args.max_expert_share}")

    print(format_terminal(results))
    if failures:
        print("\nGATE FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
