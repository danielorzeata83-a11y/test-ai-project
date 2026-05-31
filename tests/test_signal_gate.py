"""Phase 4 tests: Registry, Validation Gate, Signal bot. Run via:
    python -m unittest discover -s tests

Uses a synthetic price fixture with persistent per-ticker drift, so momentum is
a genuine edge the Gate should accept, while random noise should be rejected.
"""

import math
import os
import random
import tempfile
import unittest

from alpha_pipeline.alphas import ALPHAS, momentum
from alpha_pipeline.analyst import AnalystBot
from alpha_pipeline.contracts import PortfolioState, SafeWeights
from alpha_pipeline.execution import ExecutionBot
from alpha_pipeline.gate import GateConfig, evaluate_alpha
from alpha_pipeline.journal import Journal
from alpha_pipeline.orchestrator import Orchestrator
from alpha_pipeline.registry import Registry
from alpha_pipeline.signal import SignalBot

UNIVERSE = tuple("T%02d" % i for i in range(40))
LAST = "0399"


def signal_prices(n_days=400, seed=3):
    """Persistent per-ticker drift -> momentum tracks the cross-section."""
    rng = random.Random(seed)
    drifts = {t: rng.uniform(-0.0015, 0.0015) for t in UNIVERSE}
    out = {}
    for t in UNIVERSE:
        price = 100.0
        rows = []
        for i in range(n_days):
            r = drifts[t] + rng.gauss(0, 0.008)
            price *= math.exp(r)
            rows.append((str(i).zfill(4), price, 1e6))
        out[t] = rows
    return out


class TestRegistry(unittest.TestCase):
    def test_persists_across_restart(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "reg.json")
        r1 = Registry(path)
        self.assertEqual(r1.active_names(), ())
        r1.promote("momentum", {"ic": 0.06})
        # New instance reads the same file.
        r2 = Registry(path)
        self.assertTrue(r2.is_promoted("momentum"))
        self.assertEqual(r2.active_names(), ("momentum",))
        r2.retire("momentum")
        self.assertEqual(Registry(path).active_names(), ())


class TestGate(unittest.TestCase):
    def setUp(self):
        self.prices = signal_prices()
        self.reg = Registry(os.path.join(tempfile.mkdtemp(), "reg.json"))
        self.cfg = GateConfig()

    def test_accepts_real_alpha(self):
        res = evaluate_alpha(self.prices, momentum, "momentum", self.reg, self.cfg)
        self.assertTrue(res.passed, res)
        self.assertTrue(res.metrics["t_stat"] > 3.0)
        self.assertTrue(res.metrics["net_sharpe"] > 0.5)

    def test_rejects_noise(self):
        rng = random.Random(0)
        noise = lambda feats: {t: rng.gauss(0, 1) for t in feats}
        res = evaluate_alpha(self.prices, noise, "noise", self.reg, self.cfg)
        self.assertFalse(res.passed)
        # Statistician and skeptic should both reject pure noise.
        self.assertFalse(res.checks["statistician"])
        self.assertFalse(res.checks["skeptic"])

    def test_rejects_correlated_clone(self):
        # Promote momentum first, then offer a near-clone of it.
        self.reg.promote("momentum", {})
        rng = random.Random(1)

        def clone(feats):
            base = momentum(feats)
            return {t: base[t] * 1.1 + rng.gauss(0, 1e-6) for t in feats}

        res = evaluate_alpha(self.prices, clone, "clone", self.reg, self.cfg)
        # Quality checks pass (it IS momentum), but the librarian catches it.
        self.assertGreater(res.metrics["max_correlation"], 0.6)
        self.assertFalse(res.checks["librarian"])
        self.assertFalse(res.passed)


class TestSignalBot(unittest.TestCase):
    def test_balanced_long_short(self):
        reg = Registry(os.path.join(tempfile.mkdtemp(), "reg.json"))
        reg.promote("momentum", {})
        bot = SignalBot(reg, top_frac=0.2)

        panel = AnalystBot(UNIVERSE, lambda: signal_prices()).run(LAST)
        self.assertTrue(panel.valid)
        tw = bot.run(panel)
        # 40 tickers * 0.2 = 8 long, 8 short.
        longs = [w for w in tw.weights.values() if w > 0]
        shorts = [w for w in tw.weights.values() if w < 0]
        self.assertEqual(len(longs), 8)
        self.assertEqual(len(shorts), 8)
        self.assertAlmostEqual(tw.gross, 1.0, places=6)
        self.assertAlmostEqual(tw.net, 0.0, places=6)

    def test_empty_registry_no_positions(self):
        reg = Registry(os.path.join(tempfile.mkdtemp(), "reg.json"))
        bot = SignalBot(reg)
        panel = AnalystBot(UNIVERSE, lambda: signal_prices()).run(LAST)
        tw = bot.run(panel)
        self.assertEqual(tw.weights, {})


class TestEndToEnd(unittest.TestCase):
    def test_full_chain_with_registry(self):
        tmp = tempfile.mkdtemp()
        reg = Registry(os.path.join(tmp, "reg.json"))
        reg.promote("momentum", {})

        class PassRisk:
            def run(self, target, state, panel):
                w = dict(target.weights)
                return SafeWeights(
                    target.date, w,
                    sum(abs(x) for x in w.values()), sum(w.values()),
                )

        prices = signal_prices()
        with Journal(os.path.join(tmp, "j.db")) as j:
            orch = Orchestrator(
                j, AnalystBot(UNIVERSE, lambda: prices),
                SignalBot(reg), PassRisk(), ExecutionBot(),
            )
            state = PortfolioState.initial("0398", 100_000.0)
            state, trades = orch.run_cycle(LAST, state)
            self.assertEqual(len(trades), 16)  # 8 long + 8 short
            self.assertEqual(j.cycle_status(LAST), "success")
            # Registry reloads correctly from JSON.
            self.assertEqual(Registry(os.path.join(tmp, "reg.json")).active_names(),
                             ("momentum",))


if __name__ == "__main__":
    unittest.main()
