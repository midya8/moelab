import unittest

from moelab.eval import run_evaluation


class TestEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = run_evaluation()

    def test_moe_fit_and_param_parity(self):
        # deterministic regression checks on the mixture itself. The MLP
        # baseline comparison is reported by `evaluate`, not gated here: a
        # stronger baseline catching up is a result to report, not a defect.
        self.assertLess(self.r["moe_mse_top1"], 0.06)
        self.assertLessEqual(abs(self.r["moe_params"] - self.r["mlp_params"]), 9)

    def test_router_specializes(self):
        self.assertEqual(self.r["routing_purity_min"], 1.0)

    def test_usage_balanced(self):
        self.assertLessEqual(self.r["aux_max_share"], 0.26)

    def test_soft_routing_degrades_sparse_model(self):
        # measured inversion: the objective is sparse, so averaging specialized
        # experts costs accuracy - pinned so the README number stays honest.
        self.assertGreater(self.r["moe_mse_soft"], 5 * self.r["moe_mse_top1"])

    def test_aux_term_not_load_bearing_on_this_fixture(self):
        # honest fixture result: symmetric regimes self-balance, and the aux
        # term slightly costs fit. Pinned so the claim stays measured.
        self.assertLessEqual(self.r["noaux_max_share"], 0.26)
        self.assertLess(self.r["moe_noaux_mse_top1"], self.r["moe_mse_top1"])

    def test_determinism(self):
        self.assertEqual(run_evaluation(), self.r)


if __name__ == "__main__":
    unittest.main()
