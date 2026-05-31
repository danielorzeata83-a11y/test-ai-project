"""Phase 7 tests: frequency-aware annualization, intraday path, real-data
loaders (CSV proves the yfinance/Alpaca contract offline). Run via:
    python -m unittest discover -s tests
"""

import os
import tempfile
import unittest

from alpha_pipeline import evaluate as ev
from alpha_pipeline.config import Config
from alpha_pipeline.data import load_csv, load_synthetic_intraday, SP100


class TestFrequencyAnnualization(unittest.TestCase):
    def test_sharpe_scales_with_periods(self):
        rets = [0.001, -0.0005, 0.0008, 0.0002, -0.0003] * 20
        daily = ev.sharpe(rets, periods=252)
        intraday = ev.sharpe(rets, periods=252 * 78)
        # Same stream annualized at higher frequency -> larger by exactly sqrt78.
        self.assertAlmostEqual(intraday / daily, 78 ** 0.5, places=6)

    def test_dsr_respects_periods(self):
        rets = [0.001, -0.0005, 0.0008, 0.0002, -0.0003] * 40
        s = ev.sharpe(rets, periods=252 * 78)
        dsr = ev.deflated_sharpe(rets, s, n_trials=20, periods=252 * 78)
        self.assertTrue(0.0 <= dsr <= 1.0)


class TestIntradayData(unittest.TestCase):
    def test_synthetic_intraday_shape(self):
        U = ("A", "B", "C")
        data = load_synthetic_intraday(U, n_days=3, bars_per_day=78, seed=1)
        self.assertEqual(len(data["A"]), 3 * 78)
        ts = [r[0] for r in data["A"]]
        self.assertEqual(ts, sorted(ts))  # timestamps sort ascending
        self.assertIn(":", ts[0])         # carry HH:MM

    def test_intraday_run_end_to_end(self):
        from run import run

        cfg = Config.load("config.yaml")
        cfg.raw["universe"]["source"] = "synthetic_intraday"
        cfg.raw["frequency"]["bars_per_day"] = 78
        report = run(cfg, days=12, seed=3, workdir=tempfile.mkdtemp())
        self.assertGreater(report.n_cycles, 500)
        self.assertFalse(report.halted)


class TestCsvRealDataContract(unittest.TestCase):
    """CSV proves the same neutral contract the yfinance/Alpaca loaders return,
    validated offline (no network)."""

    def test_csv_roundtrip(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "prices.csv")
        with open(path, "w") as f:
            f.write("date,ticker,close,volume\n")
            f.write("2024-01-02,AAPL,100.0,1000\n")
            f.write("2024-01-01,AAPL,99.0,900\n")  # out of order on purpose
            f.write("2024-01-01,MSFT,200.0,500\n")
        data = load_csv(path)
        self.assertEqual([r[0] for r in data["AAPL"]],
                         ["2024-01-01", "2024-01-02"])  # sorted ascending
        self.assertEqual(data["MSFT"][0], ("2024-01-01", 200.0, 500.0))

    def test_sp100_universe_available(self):
        # ~100 liquid large-caps; exact OEX membership drifts over time.
        self.assertGreaterEqual(len(SP100), 95)
        self.assertIn("AAPL", SP100)
        self.assertEqual(len(set(SP100)), len(SP100))  # no duplicates


if __name__ == "__main__":
    unittest.main()
