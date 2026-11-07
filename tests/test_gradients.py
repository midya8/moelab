import random
import unittest

from moelab.moe import MoE

DIM, E, LR, EPS = 16, 4, 1e-3, 1e-7


class TestSparseGradients(unittest.TestCase):
    """Finite-difference verification of the hand-derived sparse-MoE gradients.

    Task loss: (g_top * expert_top(x) - t)^2 with straight-through argmax.
    Aux loss:  E * sum_k g_k^2 (load balancing).
    """

    def setUp(self):
        rng = random.Random(0)
        self.m = MoE(DIM, E, rng)
        self.x = [rng.gauss(0, 1) for _ in range(DIM)]
        self.t = 0.7

    def _loss(self, aux):
        g = self.m.gates(self.x)
        e = g.index(max(g))
        task = (g[e] * self.m.expert_out(e, self.x) - self.t) ** 2
        return task + (aux * E * sum(gk * gk for gk in g) if aux else 0.0)

    def _analytic(self, aux):
        m = self.m
        snap = ([r[:] for r in m.W], [r[:] for r in m.R], m.b[:], m.c[:])
        m.train_step(self.x, self.t, LR, aux)
        ana = {
            "W": [[(snap[0][e][j] - m.W[e][j]) / LR for j in range(DIM)] for e in range(E)],
            "R": [[(snap[1][e][j] - m.R[e][j]) / LR for j in range(DIM)] for e in range(E)],
            "b": [(snap[2][e] - m.b[e]) / LR for e in range(E)],
            "c": [(snap[3][e] - m.c[e]) / LR for e in range(E)],
        }
        m.W, m.R, m.b, m.c = [r[:] for r in snap[0]], [r[:] for r in snap[1]], snap[2][:], snap[3][:]
        return ana

    def _check(self, aux, tol):
        ana = self._analytic(aux)
        for e in range(E):
            for j in range(DIM):
                for P, key in ((self.m.W, "W"), (self.m.R, "R")):
                    o = P[e][j]
                    P[e][j] = o + EPS
                    lp = self._loss(aux)
                    P[e][j] = o - EPS
                    lm = self._loss(aux)
                    P[e][j] = o
                    self.assertAlmostEqual((lp - lm) / (2 * EPS), ana[key][e][j], delta=tol)
            for P, key in ((self.m.b, "b"), (self.m.c, "c")):
                o = P[e]
                P[e] = o + EPS
                lp = self._loss(aux)
                P[e] = o - EPS
                lm = self._loss(aux)
                P[e] = o
                self.assertAlmostEqual((lp - lm) / (2 * EPS), ana[key][e], delta=tol)

    def test_task_gradients(self):
        self._check(0.0, 1e-5)

    def test_task_plus_aux_gradients(self):
        self._check(0.05, 1e-3)  # looser: FD rounding on the stiffer combined loss

    def test_top1_prediction_uses_gate_probability(self):
        g = self.m.gates(self.x)
        e = g.index(max(g))
        self.assertAlmostEqual(self.m.predict_top1(self.x), g[e] * self.m.expert_out(e, self.x))


if __name__ == "__main__":
    unittest.main()
