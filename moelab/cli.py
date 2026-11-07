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

    args = parser.parse_args(argv)
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
