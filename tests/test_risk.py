"""Phase 5 tests: Risk bot + regime HMM. Run via:
    python -m unittest discover -s tests

Each scenario triggers a different control in the chain. The guiding invariant:
every control can only reduce exposure, never increase it.
"""

import random
import unittest

from alpha_pipeline.contracts import FeaturePanel, PortfolioState, TargetWeights
from alpha_pipeline.regime import RegimeHMM
from alpha_pipeline.risk import RiskBot, RiskConfig

UNIVERSE = tuple("T%02d" % i for i in range(20))


def panel(date="d1", vol=0.01, valid=True, market_returns=None):
    data = {t: {"close": 100.0, "ret_1d": 0.0, "vol_21": vol} for t in UNIVERSE}
    meta = {}
    if market_returns is not None:
        meta["market_returns"] = market_returns
    return FeaturePanel(date, UNIVERSE, data, valid, meta)


def state(nav=100_000.0, peak=100_000.0):
    return PortfolioState("d0", nav, {}, nav, peak)


class TestRiskControls(unittest.TestCase):
    def setUp(self):
        # Low-vol panel so vol-targeting doesn't interfere with structural tests.
        self.panel = panel(vol=0.005)
        self.state = state()

    def test_per_position_cap(self):
        bot = RiskBot(RiskConfig(max_position=0.05, target_vol=10.0))
        tw = TargetWeights("d1", {"T00": 0.30, "T01": -0.30})
        safe = bot.run(tw, self.state, self.panel)
        self.assertLessEqual(max(abs(x) for x in safe.weights.values()), 0.05 + 1e-9)
        self.assertTrue(any("per-position" in a for a in safe.risk_actions))

    def test_dollar_neutrality(self):
        bot = RiskBot(RiskConfig(target_vol=10.0))
        tw = TargetWeights("d1", {"T00": 0.04, "T01": 0.04, "T02": -0.02})
        safe = bot.run(tw, self.state, self.panel)
        self.assertAlmostEqual(safe.net_exposure, 0.0, places=6)

    def test_neutrality_does_not_breach_cap(self):
        # Re-centering a skewed book must not push any name past max_position.
        bot = RiskBot(RiskConfig(max_position=0.05, target_vol=10.0))
        tw = TargetWeights("d1", {"T00": 0.05, "T01": 0.05, "T02": 0.05, "T03": -0.05})
        safe = bot.run(tw, self.state, self.panel)
        self.assertLessEqual(max(abs(x) for x in safe.weights.values()), 0.05 + 1e-9)

    def test_gross_cap(self):
        bot = RiskBot(RiskConfig(max_position=1.0, max_gross=1.0, target_vol=10.0))
        tw = TargetWeights("d1", {"T00": 0.9, "T01": -0.9})  # gross 1.8
        safe = bot.run(tw, self.state, self.panel)
        self.assertLessEqual(safe.gross_exposure, 1.0 + 1e-9)
        self.assertTrue(any("gross cap" in a for a in safe.risk_actions))

    def test_vol_target_scales_down(self):
        # High per-name vol -> portfolio vol exceeds target -> scaled down.
        bot = RiskBot(RiskConfig(max_position=1.0, max_gross=10.0, target_vol=0.10))
        hi = panel(vol=0.05)
        tw = TargetWeights("d1", {"T00": 1.0, "T01": -1.0})
        safe = bot.run(tw, self.state, hi)
        self.assertLess(safe.gross_exposure, 2.0)
        self.assertTrue(any("vol target" in a for a in safe.risk_actions))

    def test_drawdown_breaker_halts(self):
        bot = RiskBot(RiskConfig(max_drawdown=0.15, target_vol=10.0))
        deep = state(nav=80_000.0, peak=100_000.0)  # -20% drawdown
        tw = TargetWeights("d1", {"T00": 0.02, "T01": -0.02})
        safe = bot.run(tw, deep, self.panel)
        self.assertTrue(safe.halt_request)
        self.assertEqual(safe.weights, {})
        self.assertTrue(any("drawdown" in a for a in safe.risk_actions))

    def test_invalid_panel_halts(self):
        bot = RiskBot()
        tw = TargetWeights("d1", {"T00": 0.02})
        safe = bot.run(tw, self.state, panel(valid=False))
        self.assertTrue(safe.halt_request)
        self.assertEqual(safe.weights, {})

    def test_regime_overlay_cuts_exposure(self):
        # Build a market series ending in a clear high-vol burst.
        rng = random.Random(0)
        calm = [rng.gauss(0, 0.006) for _ in range(120)]
        crisis = [rng.gauss(0, 0.030) for _ in range(40)]
        series = calm + crisis
        bot = RiskBot(RiskConfig(max_position=1.0, max_gross=10.0,
                                 target_vol=10.0, regime_cut=0.3))
        p = panel(vol=0.005, market_returns=series)
        tw = TargetWeights("d1", {"T00": 1.0, "T01": -1.0})
        safe = bot.run(tw, self.state, p)
        self.assertTrue(any("regime overlay" in a for a in safe.risk_actions))
        self.assertLess(safe.gross_exposure, 2.0)

    def test_controls_only_reduce(self):
        # Whatever the config, output gross never exceeds input gross.
        bot = RiskBot(RiskConfig(max_position=1.0, max_gross=10.0, target_vol=10.0))
        tw = TargetWeights("d1", {"T00": 0.3, "T01": -0.3})
        safe = bot.run(tw, self.state, self.panel)
        self.assertLessEqual(safe.gross_exposure, tw.gross + 1e-9)


class TestRegimeHMM(unittest.TestCase):
    def test_recovers_two_regimes(self):
        rng = random.Random(0)
        rets, true = [], []
        st = 0
        for _ in range(800):
            if st == 0 and rng.random() < 0.01:
                st = 1
            elif st == 1 and rng.random() < 0.06:
                st = 0
            rets.append(rng.gauss(0, 0.008 if st == 0 else 0.025))
            true.append(st)
        hmm = RegimeHMM().fit(rets)
        # State 1 is the high-variance one (anti-label-switching).
        self.assertGreater(hmm.var[1], hmm.var[0])
        probs = hmm.crisis_probabilities(rets)
        pred = [1 if p > 0.5 else 0 for p in probs]
        acc = sum(1 for a, b in zip(pred, true) if a == b) / len(true)
        self.assertGreater(acc, 0.85)

    def test_durations_positive(self):
        rng = random.Random(1)
        rets = [rng.gauss(0, 0.01) for _ in range(200)]
        hmm = RegimeHMM().fit(rets)
        d0, d1 = hmm.expected_durations()
        self.assertGreater(d0, 0)
        self.assertGreater(d1, 0)


if __name__ == "__main__":
    unittest.main()
