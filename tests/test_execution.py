"""Phase 2 tests: Execution + Orchestrator. Run via:
    python -m unittest discover -s tests

Five scenarios, each verifying a different guarantee of the chain.
"""

import os
import tempfile
import unittest

from alpha_pipeline.contracts import (
    FeaturePanel,
    PortfolioState,
    SafeWeights,
    TargetWeights,
)
from alpha_pipeline.execution import ExecutionBot
from alpha_pipeline.journal import Journal
from alpha_pipeline.orchestrator import Orchestrator

UNIVERSE = ("AAPL", "MSFT", "GOOGL", "AMZN")
WEIGHTS = {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": -0.25, "AMZN": -0.25}


class MockAnalyst:
    """Returns a panel with flat $100 prices, or invalid when told to."""

    def __init__(self, valid: bool = True, price: float = 100.0):
        self.valid = valid
        self.price = price

    def run(self, date: str) -> FeaturePanel:
        data = {t: {"close": self.price} for t in UNIVERSE}
        return FeaturePanel(date, UNIVERSE, data, self.valid)


class MockSignal:
    def run(self, panel: FeaturePanel) -> TargetWeights:
        return TargetWeights(panel.date, dict(WEIGHTS))


class MockRisk:
    """Passes weights through; can request a halt."""

    def __init__(self, halt: bool = False):
        self.halt = halt

    def run(self, target, state, panel) -> SafeWeights:
        w = dict(target.weights)
        return SafeWeights(
            target.date,
            w,
            gross_exposure=sum(abs(x) for x in w.values()),
            net_exposure=sum(w.values()),
            halt_request=self.halt,
        )


class CrashSignal:
    def run(self, panel):
        raise RuntimeError("signal bug")


def make_orch(journal, analyst=None, signal=None, risk=None):
    return Orchestrator(
        journal,
        analyst or MockAnalyst(),
        signal or MockSignal(),
        risk or MockRisk(),
        ExecutionBot(),
    )


class TestExecutionOrchestrator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "j.db")
        self.s0 = PortfolioState.initial("2024-01-01", 100_000.0)

    def test_basic_cycle(self):
        with Journal(self.db) as j:
            orch = make_orch(j)
            state, trades = orch.run_cycle("2024-01-02", self.s0)
            # Four targets from flat -> four trades.
            self.assertEqual(len(trades), 4)
            # Gross 1.0 traded from flat = $100k notional * 6bps = $60 cost.
            self.assertAlmostEqual(state.nav, 99_940.0, places=2)
            self.assertEqual(j.cycle_status("2024-01-02"), "success")

    def test_ten_cycles_propagate_and_replay(self):
        with Journal(self.db) as j:
            orch = make_orch(j, analyst=MockAnalyst(price=100.0))
            state = self.s0
            for i in range(2, 12):
                date = f"2024-01-{i:02d}"
                # Vary price slightly each day to force small rebalances.
                orch.analyst.price = 100.0 + (i - 2) * 0.5
                state, _ = orch.run_cycle(date, state)
            self.assertTrue(state.nav > 0)
            self.assertFalse(state.halted)
            # State recovered from disk equals in-memory exactly.
            recovered = j.recover_state(state.date)
            self.assertEqual(recovered, state)
            self.assertEqual(j.latest_state(), state)

    def test_invalid_panel_stops_chain(self):
        with Journal(self.db) as j:
            orch = make_orch(j, analyst=MockAnalyst(valid=False))
            state, trades = orch.run_cycle("2024-01-02", self.s0)
            self.assertEqual(trades, ())
            self.assertEqual(state, self.s0)  # untouched
            # Ran and decided to do nothing -> success, but no target logged.
            self.assertEqual(j.cycle_status("2024-01-02"), "success")
            self.assertEqual(j.read_trades("2024-01-02"), ())

    def test_crash_halts_state(self):
        with Journal(self.db) as j:
            orch = make_orch(j, signal=CrashSignal())
            state, trades = orch.run_cycle("2024-01-02", self.s0)
            self.assertTrue(state.halted)
            self.assertIn("signal bug", state.halt_reason)
            self.assertEqual(trades, ())
            self.assertEqual(j.cycle_status("2024-01-02"), "failed")
            # Next cycle refuses to start because state is halted.
            state2, trades2 = orch.run_cycle("2024-01-03", state)
            self.assertEqual(trades2, ())
            self.assertTrue(state2.halted)

    def test_idempotent_cycle(self):
        with Journal(self.db) as j:
            orch = make_orch(j)
            state1, trades1 = orch.run_cycle("2024-01-02", self.s0)
            self.assertEqual(len(trades1), 4)
            # Second run of same date is a no-op.
            state2, trades2 = orch.run_cycle("2024-01-02", state1)
            self.assertEqual(trades2, ())
            self.assertEqual(state2, state1)

    def test_halt_request_flattens_and_halts(self):
        with Journal(self.db) as j:
            orch = make_orch(j, risk=MockRisk(halt=True))
            state, _ = orch.run_cycle("2024-01-02", self.s0)
            self.assertTrue(state.halted)
            self.assertEqual(j.cycle_status("2024-01-02"), "success")


if __name__ == "__main__":
    unittest.main()
