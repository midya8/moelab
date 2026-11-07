"""Soft-gated mixture-of-experts with hand-derived gradients.

y(x) = sum_e g_e(x) * (w_e . x + b_e),   g = softmax(R x + c)

Gradients (derived by hand, finite-difference verified in tests), for
L = (y - t)^2, s = 2(y - t):

  dL/dw_e   = s * g_e * x
  dL/db_e   = s * g_e
  dL/dR_ej  = s * y_e * g_e * (x_j - sum_k g_k ... ) via the softmax Jacobian:
              dy/dR_ej = y_e * g_e * x_j - g_e * x_j * sum_k g_k y_k
                       = g_e * x_j * (y_e - y)
  dL/dc_e   = g_e * (y_e - y) * s

so dL/dR_ej = s * g_e * (y_e - y) * x_j and dL/dc_e = s * g_e * (y_e - y).
The load-balancing auxiliary loss penalizes the squared L2 of mean gate
probabilities: aux = E * sum_e (mean g_e)^2, minimized when every expert gets
an equal share; its gradient follows the same chain rule and is included in
training but never in reported task loss.
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
        # Switch-style sparse objective: y = g_top * expert_top(x). The argmax
        # selection is treated as constant (straight-through on the choice);
        # gradients flow through the gate probability and the chosen expert.
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
