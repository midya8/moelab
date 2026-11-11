"""Measured MoE evaluation: regime-fit, routing specialization, load balance.

Trains on the regime-switching fixture: a retained-probability top-1 mixture, the same mixture
without the load-balancing term, and a same-budget dense MLP baseline.
Reports test MSE (soft and top-1 routing), routing purity per regime, expert
usage shares, and param counts - all measured.
"""

from __future__ import annotations

import random
from typing import Dict, List

from .data import DIM, N_REGIMES, make_split
from .moe import MoE


def mse(pred, data) -> float:
    return sum((p - t) ** 2 for p, t in zip(pred, (d[1] for d in data))) / len(data)


def train_moe(model: MoE, data, epochs: int, lr: float, aux: float) -> None:
    for _ in range(epochs):
        for x, t, _ in data:
            model.train_step(x, t, lr, aux)


class DenseMLP:
    """One-hidden-layer tanh MLP, hand-derived backprop, as a parameter-budget-matched baseline."""

    def __init__(self, dim: int, hidden: int, rng: random.Random):
        self.W1 = [[rng.gauss(0, 0.2) for _ in range(dim)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.w2 = [rng.gauss(0, 0.2) for _ in range(hidden)]
        self.b2 = 0.0

    def predict(self, x: List[float]) -> float:
        import math
        h = [math.tanh(sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        return sum(w * hi for w, hi in zip(self.w2, h)) + self.b2

    def train_step(self, x: List[float], t: float, lr: float) -> float:
        import math
        h = [math.tanh(sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        y = sum(w * hi for w, hi in zip(self.w2, h)) + self.b2
        s = 2.0 * (y - t)
        for i in range(len(h)):
            dw2 = s * h[i]
            dh = s * self.w2[i] * (1.0 - h[i] ** 2)
            for j in range(len(x)):
                self.W1[i][j] -= lr * dh * x[j]
            self.b1[i] -= lr * dh
            self.w2[i] -= lr * dw2
        self.b2 -= lr * s
        return (y - t) ** 2

    def param_count(self) -> int:
        return len(self.W1) * len(self.W1[0]) + len(self.b1) + len(self.w2) + 1


def run_evaluation(n_per_regime: int = 60, test_per_regime: int = 25,
                   epochs: int = 80, lr: float = 0.1, mlp_lr: float = 0.02,
                   aux: float = 0.05, seed: int = 11) -> Dict:
    train = make_split(n_per_regime, seed)
    test = make_split(test_per_regime, seed + 1000)

    moe = MoE(DIM, N_REGIMES, random.Random(seed))
    train_moe(moe, train, epochs, lr, aux)
    moe_noaux = MoE(DIM, N_REGIMES, random.Random(seed))
    train_moe(moe_noaux, train, epochs, lr, 0.0)

    hidden = 8  # param count 8*16+8+8+1 = 145, close to MoE's 4*17+4*16+4 = 136
    mlp = DenseMLP(DIM, hidden, random.Random(seed))
    for _ in range(epochs):
        for x, t, _ in train:
            mlp.train_step(x, t, mlp_lr)

    def routing_stats(model: MoE):
        # per-regime top-expert purity and expert usage shares over the test set
        per_regime = {}
        shares = [0] * model.n_experts
        for regime in range(N_REGIMES):
            subset = [d for d in test if d[2] == regime]
            tops = [model.top_expert(x) for x, _, _ in subset]
            counts = [tops.count(e) for e in range(model.n_experts)]
            per_regime[regime] = max(counts) / len(subset)
            for e in tops:
                shares[e] += 1
        return per_regime, [s / len(test) for s in shares]

    purity, shares = routing_stats(moe)
    purity_na, shares_na = routing_stats(moe_noaux)

    return {
        "moe_mse_soft": mse([moe.predict_soft(x) for x, _, _ in test], test),
        "moe_mse_top1": mse([moe.predict_top1(x) for x, _, _ in test], test),
        "moe_noaux_mse_top1": mse([moe_noaux.predict_top1(x) for x, _, _ in test], test),
        "mlp_mse": mse([mlp.predict(x) for x, _, _ in test], test),
        "moe_params": moe.param_count(),
        "mlp_params": mlp.param_count(),
        "routing_purity_min": min(purity.values()),
        "expert_shares": shares,
        "noaux_max_share": max(shares_na),
        "aux_max_share": max(shares),
    }


def format_terminal(r: Dict) -> str:
    return "\n".join([
        "MoeLab evaluation - mixture-of-experts on a regime-switching fixture",
        "",
        f"MoE test MSE (soft routing):  {r['moe_mse_soft']:.4f}",
        f"MoE test MSE (top-1 routing): {r['moe_mse_top1']:.4f}   ({r['moe_params']} params)",
        f"Dense MLP baseline test MSE:  {r['mlp_mse']:.4f}   ({r['mlp_params']} params)",
        f"Routing purity (worst regime): {r['routing_purity_min']:.4f}",
        f"Expert usage shares (aux=on):  {[round(s, 3) for s in r['expert_shares']]}",
        f"Max expert share without aux loss: {r['noaux_max_share']:.4f} "
        f"(with aux: {r['aux_max_share']:.4f})",
    ])
