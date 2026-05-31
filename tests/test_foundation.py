"""Phase 1 foundation tests. Run: python -m unittest discover -s tests

Three scenarios, each verifying a different guarantee:
  1. PortfolioState NAV/drawdown math + immutability (basis for replay).
  2. Contracts reject bad data at construction, not later.
  3. Journal idempotency + exact state replay.
"""

import os
import tempfile
import unittest

from alpha_pipeline.contracts import (
    FeaturePanel,
    PortfolioState,
    SafeWeights,
    TargetWeights,
    Trade,
)
from alpha_pipeline.journal import CycleAlreadyLogged, Journal


class TestPortfolioMath(unittest.TestCase):
    def test_nav_peak_and_immutability(self):
        s0 = PortfolioState.initial("2024-01-01", 100_000.0)

        # Buy 10 shares @ $200 with $2 cost.
        buy = Trade("2024-01-02", "AAPL", 10.0, 200.0, 2.0)
        s1 = s0.apply_trades("2024-01-02", (buy,))
        self.assertAlmostEqual(s1.cash, 100_000.0 - 2000.0 - 2.0)
        self.assertAlmostEqual(s1.nav, 99_998.0)  # 97998 cash + 2000 position
        self.assertEqual(s1.positions, {"AAPL": 10.0})

        # Mark up to $210 -> NAV 100,098, new peak.
        s2 = s1.mark_to_market("2024-01-03", {"AAPL": 210.0})
        self.assertAlmostEqual(s2.nav, 100_098.0)
        self.assertAlmostEqual(s2.peak_nav, 100_098.0)
        self.assertAlmostEqual(s2.drawdown, 0.0)

        # Mark down to $190 -> NAV drops, peak stays, drawdown negative.
        s3 = s2.mark_to_market("2024-01-04", {"AAPL": 190.0})
        self.assertAlmostEqual(s3.nav, 99_898.0)
        self.assertAlmostEqual(s3.peak_nav, 100_098.0)
        self.assertAlmostEqual(s3.drawdown, (99_898.0 - 100_098.0) / 100_098.0)

        # Immutability: s0 never changed through all of the above.
        self.assertEqual(s0.cash, 100_000.0)
        self.assertEqual(s0.positions, {})
        self.assertEqual(s0.nav, 100_000.0)

    def test_halt_returns_new_state(self):
        s0 = PortfolioState.initial("2024-01-01", 100_000.0)
        s1 = s0.halt("drawdown breaker")
        self.assertTrue(s1.halted)
        self.assertEqual(s1.halt_reason, "drawdown breaker")
        self.assertFalse(s0.halted)  # original untouched

    def test_partial_rebalance_values_untraded_holdings(self):
        # Hold AAPL and MSFT, then trade only AAPL. MSFT must keep its real
        # value via marks, not be valued at zero.
        s0 = PortfolioState.initial("2024-01-01", 100_000.0)
        s1 = s0.apply_trades(
            "2024-01-02",
            (Trade("2024-01-02", "AAPL", 10.0, 200.0, 0.0),
             Trade("2024-01-02", "MSFT", 5.0, 100.0, 0.0)),
        )
        # Now trade only AAPL; provide marks for both held names.
        s2 = s1.apply_trades(
            "2024-01-03",
            (Trade("2024-01-03", "AAPL", 5.0, 210.0, 0.0),),
            marks={"AAPL": 210.0, "MSFT": 110.0},
        )
        # AAPL: 15 @ 210 = 3150; MSFT: 5 @ 110 = 550 (revalued, not zero).
        expected_cash = 100_000.0 - 2000.0 - 500.0 - 5 * 210.0
        self.assertAlmostEqual(s2.cash, expected_cash)
        self.assertAlmostEqual(s2.nav, expected_cash + 3150.0 + 550.0)

    def test_missing_mark_fails_closed(self):
        # A held name with no trade and no mark must raise, not value at zero.
        s0 = PortfolioState.initial("2024-01-01", 100_000.0)
        s1 = s0.apply_trades(
            "2024-01-02",
            (Trade("2024-01-02", "MSFT", 5.0, 100.0, 0.0),),
        )
        with self.assertRaises(ValueError):
            # Trade AAPL but give no mark for the still-held MSFT.
            s1.apply_trades(
                "2024-01-03",
                (Trade("2024-01-03", "AAPL", 1.0, 200.0, 0.0),),
                marks={"AAPL": 200.0},
            )


class TestContractValidation(unittest.TestCase):
    def test_bad_inputs_rejected_at_construction(self):
        # Empty date.
        with self.assertRaises(ValueError):
            TargetWeights("", {"AAPL": 0.5})
        # Non-finite weight.
        with self.assertRaises(ValueError):
            TargetWeights("2024-01-01", {"AAPL": float("nan")})
        # Empty ticker.
        with self.assertRaises(ValueError):
            Trade("2024-01-01", "  ", 1.0, 100.0, 0.0)
        # Negative price.
        with self.assertRaises(ValueError):
            Trade("2024-01-01", "AAPL", 1.0, -100.0, 0.0)
        # Negative cost.
        with self.assertRaises(ValueError):
            Trade("2024-01-01", "AAPL", 1.0, 100.0, -1.0)
        # Panel data keys not matching tickers.
        with self.assertRaises(ValueError):
            FeaturePanel("2024-01-01", ("AAPL",), {"MSFT": {"close": 1.0}}, True)

    def test_safe_weights_cannot_lie(self):
        w = {"AAPL": 0.5, "MSFT": -0.5}  # actual gross 1.0, net 0.0
        # Honest declaration is accepted.
        ok = SafeWeights("2024-01-01", w, gross_exposure=1.0, net_exposure=0.0)
        self.assertAlmostEqual(ok.gross_exposure, 1.0)
        # Lying about gross exposure is rejected.
        with self.assertRaises(ValueError):
            SafeWeights("2024-01-01", w, gross_exposure=99.0, net_exposure=0.0)
        # Lying about net exposure is rejected.
        with self.assertRaises(ValueError):
            SafeWeights("2024-01-01", w, gross_exposure=1.0, net_exposure=5.0)


class TestContractImmutability(unittest.TestCase):
    def test_mappings_are_read_only(self):
        import types

        panel = FeaturePanel(
            "d1", ("AAPL",), {"AAPL": {"close": 100.0}}, True, metadata={"k": 1}
        )
        tw = TargetWeights("d1", {"AAPL": 0.5})
        sw = SafeWeights("d1", {"AAPL": 0.5}, 0.5, 0.5)
        ps = PortfolioState("d1", 100.0, {"AAPL": 1.0}, 200.0, 200.0)

        for m in (panel.data, panel.data["AAPL"], panel.metadata,
                  tw.weights, sw.weights, ps.positions):
            self.assertIsInstance(m, types.MappingProxyType)
            with self.assertRaises(TypeError):
                m["x"] = 1  # cannot mutate

    def test_caller_dict_mutation_does_not_leak(self):
        # Mutating the source dict after construction must not change the
        # contract (it stored a copy, not the caller's reference).
        src = {"AAPL": 0.5}
        tw = TargetWeights("d1", src)
        src["AAPL"] = 999.0
        src["MSFT"] = 1.0
        self.assertEqual(dict(tw.weights), {"AAPL": 0.5})


class TestJournalReplay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "journal.db")

    def test_idempotency_and_exact_replay(self):
        panel = FeaturePanel(
            "2024-01-02", ("AAPL",), {"AAPL": {"close": 200.0}}, True,
            metadata={"coverage": 1.0},
        )
        target = TargetWeights("2024-01-02", {"AAPL": 1.0})
        safe = SafeWeights("2024-01-02", {"AAPL": 1.0}, 1.0, 1.0)
        trade = Trade("2024-01-02", "AAPL", 10.0, 200.0, 2.0)
        state = PortfolioState.initial("2024-01-02", 100_000.0).apply_trades(
            "2024-01-02", (trade,)
        )

        with Journal(self.db) as j:
            with j.cycle("2024-01-02"):
                j.log_panel(panel)
                j.log_target(target)
                j.log_safe(safe)
                j.log_trades((trade,))
                j.log_state(state)

            # Re-running the same date is rejected.
            with self.assertRaises(CycleAlreadyLogged):
                with j.cycle("2024-01-02"):
                    pass

            self.assertEqual(j.cycle_status("2024-01-02"), "success")

            # State recovered from disk equals the in-memory state exactly.
            recovered = j.recover_state("2024-01-02")
            self.assertEqual(recovered, state)

            # Trades read back.
            trades_back = j.read_trades("2024-01-02")
            self.assertEqual(trades_back, (trade,))

            # A different day runs normally.
            with j.cycle("2024-01-03"):
                j.log_event("INFO", "new day ok", date="2024-01-03")
            self.assertEqual(j.cycle_status("2024-01-03"), "success")

    def test_failed_cycle_marked_failed(self):
        with Journal(self.db) as j:
            with self.assertRaises(RuntimeError):
                with j.cycle("2024-01-02"):
                    raise RuntimeError("boom")
            self.assertEqual(j.cycle_status("2024-01-02"), "failed")


if __name__ == "__main__":
    unittest.main()
