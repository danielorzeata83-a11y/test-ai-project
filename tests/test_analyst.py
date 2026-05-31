"""Phase 3 tests: Analyst bot. Run via:
    python -m unittest discover -s tests
"""

import os
import tempfile
import unittest

from alpha_pipeline.analyst import AnalystBot
from alpha_pipeline.contracts import PortfolioState, SafeWeights, TargetWeights
from alpha_pipeline.data import load_synthetic
from alpha_pipeline.execution import ExecutionBot
from alpha_pipeline.journal import Journal
from alpha_pipeline.orchestrator import Orchestrator

UNIVERSE = ("AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM", "JNJ", "PG", "KO", "WMT")
# Last synthetic date for n_days=80 starting 2024-01-01.
LAST_DATE = "2024-03-20"


def clean_loader(seed=1):
    def _load():
        return load_synthetic(UNIVERSE, n_days=80, seed=seed)
    return _load


class TestAnalystFundamentals(unittest.TestCase):
    def test_fundamentals_merged_into_panel(self):
        funds = {t: {"earnings_yield": 0.05, "roe": 0.2, "profit_margin": 0.1}
                 for t in UNIVERSE}
        bot = AnalystBot(UNIVERSE, clean_loader(), fundamentals=funds)
        panel = bot.run(LAST_DATE)
        self.assertTrue(panel.valid)
        for t in panel.tickers:
            # Price features still present...
            self.assertIn("mom_21", panel.data[t])
            # ...and fundamentals merged in.
            self.assertEqual(panel.data[t]["roe"], 0.2)
        self.assertEqual(panel.metadata["fundamentals_loaded"], len(UNIVERSE))

    def test_fundamentals_loader_fail_soft(self):
        def bad_funds():
            raise ConnectionError("overview down")

        bot = AnalystBot(UNIVERSE, clean_loader(), fundamentals=bad_funds)
        panel = bot.run(LAST_DATE)
        # Price-only panel still valid; error recorded, no crash.
        self.assertTrue(panel.valid)
        self.assertIn("fundamentals_error", panel.metadata)
        self.assertNotIn("roe", panel.data[panel.tickers[0]])

    def test_partial_fundamentals_only_some_tickers(self):
        funds = {UNIVERSE[0]: {"earnings_yield": 0.1, "roe": 0.3, "profit_margin": 0.2}}
        bot = AnalystBot(UNIVERSE, clean_loader(), fundamentals=funds)
        panel = bot.run(LAST_DATE)
        self.assertIn("roe", panel.data[UNIVERSE[0]])
        self.assertNotIn("roe", panel.data[UNIVERSE[1]])  # no funds -> price only


class TestAnalyst(unittest.TestCase):
    def test_happy_path(self):
        bot = AnalystBot(UNIVERSE, clean_loader())
        panel = bot.run(LAST_DATE)
        self.assertTrue(panel.valid)
        self.assertEqual(len(panel.tickers), len(UNIVERSE))
        # Every included ticker has the full feature set.
        for t in panel.tickers:
            self.assertIn("close", panel.data[t])
            self.assertIn("mom_21", panel.data[t])

    def test_spike_excludes_single_ticker(self):
        def loader():
            data = load_synthetic(UNIVERSE, n_days=80, seed=1)
            # Inject an 80% jump on AAPL's last close.
            d, c, v = data["AAPL"][-1]
            data["AAPL"][-1] = (d, c * 1.8, v)
            return data

        bot = AnalystBot(UNIVERSE, loader)
        panel = bot.run(LAST_DATE)
        self.assertTrue(panel.valid)  # 9/10 = 90% > 80% threshold
        self.assertNotIn("AAPL", panel.tickers)
        self.assertIn("AAPL", panel.metadata["excluded"])

    def test_low_coverage_refuses_panel(self):
        def loader():
            full = load_synthetic(UNIVERSE, n_days=80, seed=1)
            # Only 3 of 10 tickers have data.
            return {t: full[t] for t in UNIVERSE[:3]}

        bot = AnalystBot(UNIVERSE, loader)
        panel = bot.run(LAST_DATE)
        self.assertFalse(panel.valid)
        self.assertIn("error", panel.metadata)
        self.assertTrue(set(panel.metadata["missing"]) >= set(UNIVERSE[3:]))

    def test_insufficient_history_excluded(self):
        def loader():
            full = load_synthetic(UNIVERSE, n_days=80, seed=1)
            full["AAPL"] = full["AAPL"][:10]  # too short
            return full

        bot = AnalystBot(UNIVERSE, loader)
        panel = bot.run(LAST_DATE)
        self.assertNotIn("AAPL", panel.tickers)
        self.assertIn("insufficient history", panel.metadata["excluded"]["AAPL"])

    def test_loader_exception_is_fail_soft(self):
        def loader():
            raise ConnectionError("network down")

        bot = AnalystBot(UNIVERSE, loader)
        panel = bot.run(LAST_DATE)
        # No exception propagated; just an invalid panel with the reason.
        self.assertFalse(panel.valid)
        self.assertIn("loader failed", panel.metadata["error"])

    def test_date_filter_respects_asof(self):
        # Asking as of an early date yields too little history -> all excluded.
        bot = AnalystBot(UNIVERSE, clean_loader())
        panel = bot.run("2024-01-10")
        self.assertFalse(panel.valid)

    def test_integration_through_orchestrator(self):
        """Real AnalystBot replacing the mock; rest of the chain untouched."""

        class PassSignal:
            def run(self, panel):
                # Equal-weight long the top half by momentum, short the rest.
                ranked = sorted(
                    panel.tickers, key=lambda t: panel.data[t]["mom_21"]
                )
                n = len(ranked) // 2
                w = {t: -0.5 / n for t in ranked[:n]}
                w.update({t: 0.5 / n for t in ranked[-n:]})
                return TargetWeights(panel.date, w)

        class PassRisk:
            def run(self, target, state, panel):
                w = dict(target.weights)
                return SafeWeights(
                    target.date, w,
                    sum(abs(x) for x in w.values()),
                    sum(w.values()),
                )

        tmp = tempfile.mkdtemp()
        with Journal(os.path.join(tmp, "j.db")) as j:
            orch = Orchestrator(
                j, AnalystBot(UNIVERSE, clean_loader()),
                PassSignal(), PassRisk(), ExecutionBot(),
            )
            state = PortfolioState.initial("2024-03-19", 100_000.0)
            state, trades = orch.run_cycle(LAST_DATE, state)
            self.assertTrue(len(trades) > 0)
            self.assertEqual(j.cycle_status(LAST_DATE), "success")
            self.assertEqual(j.recover_state(LAST_DATE), state)


if __name__ == "__main__":
    unittest.main()
