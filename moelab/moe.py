"""Retained-probability top-1 mixture with hand-derived gradients.

For gates g = softmax(R x + c) and expert outputs h_e = W_e . x + b_e,
the trained prediction is y = g_e h_e, where e is the lowest-index maximum
gate. predict_soft() separately evaluates sum_k g_k h_k; it is not the
training objective.

Hold the selected branch e fixed. For L = (y - t)^2 and s = 2(y - t):

  dL/dW_ej = s g_e x_j,   dL/db_e = s g_e
  dL/dz_k  = s h_e g_e (delta_ek - g_k),   z = R x + c
  dL/dR_kj = (dL/dz_k) x_j,   dL/dc_k = dL/dz_k

Other experts receive no task-loss update. These are ordinary derivatives
inside a routing cell, not differentiation through argmax or a soft backward
surrogate. Finite differences are checked away from routing boundaries.

Each training row adds aux_weight * A, with A = E sum_k g_k^2. If
S2 = sum_k g_k^2, then dA/dz_k = 2 E g_k (g_k - S2). Training uses
per-example updates; this is neither the squared batch-mean gate penalty nor
Switch's batch-frequency/probability statistic. Reported task MSE excludes A.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple


def softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(t - m) for t in logits]
    z = sum(exps)
    return [e / z for e in exps]


class MoE:
    def __init__(self, dim: int, n_experts: int, rng: random.Random):
        self.dim = dim
        self.n_experts = n_experts
        self.W = [[rng.gauss(0, 0.1) for _ in range(dim)] for _ in range(n_experts)]
        self.b = [0.0] * n_experts
        self.R = [[rng.gauss(0, 0.1) for _ in range(dim)] for _ in range(n_experts)]
        self.c = [0.0] * n_experts

    def expert_out(self, e: int, x: List[float]) -> float:
        return sum(w * xi for w, xi in zip(self.W[e], x)) + self.b[e]

    def gates(self, x: List[float]) -> List[float]:
        return softmax([sum(r * xi for r, xi in zip(row, x)) + ci
                        for row, ci in zip(self.R, self.c)])

    def predict_soft(self, x: List[float]) -> float:
        g = self.gates(x)
        return sum(ge * self.expert_out(e, x) for e, ge in enumerate(g))

    def predict_top1(self, x: List[float]) -> float:
        # the trained sparse prediction: gate probability times chosen expert
        g = self.gates(x)
        e = g.index(max(g))
        return g[e] * self.expert_out(e, x)

    def top_expert(self, x: List[float]) -> int:
        g = self.gates(x)
        return g.index(max(g))

    def train_step(self, x: List[float], t: float, lr: float, aux_weight: float) -> float:
        # Hold the chosen route fixed. Differentiate its retained gate
        # probability and expert output, not the discrete argmax choice.
        g = self.gates(x)
        e = g.index(max(g))
        ge = g[e]
        ye = self.expert_out(e, x)
        y = ge * ye
        s = 2.0 * (y - t)
        dW_e = [s * ge * xj for xj in x]
        db_e = s * ge
        # dL/dz_k = s * ye * g_e (delta_ek - g_k); z = R x + c
        dz = [s * ye * ge * ((1.0 if k == e else 0.0) - g[k]) for k in range(self.n_experts)]
        if aux_weight:
            s2 = sum(gk * gk for gk in g)
            for k in range(self.n_experts):
                dz[k] += aux_weight * 2.0 * self.n_experts * g[k] * (g[k] - s2)
        for j in range(self.dim):
            self.W[e][j] -= lr * dW_e[j]
            for k in range(self.n_experts):
                self.R[k][j] -= lr * dz[k] * x[j]
        self.b[e] -= lr * db_e
        for k in range(self.n_experts):
            self.c[k] -= lr * dz[k]
        return (y - t) ** 2

    def param_count(self) -> int:
        return self.n_experts * (self.dim + 1) + self.n_experts * self.dim + self.n_experts
