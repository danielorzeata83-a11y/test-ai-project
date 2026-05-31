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


class TestBuildHistoryAlignment(unittest.TestCase):
    """build_history must align cross-sections by calendar date, not by list
    position, so ragged per-ticker series don't pair mismatched days."""

    def _const_score(self, feats):
        # Score = a value the test injects per ticker, so we can detect whether
        # a date pairs the right ticker rows.
        return {t: f["close"] for t, f in feats.items()}

    def test_ragged_dates_pair_same_calendar_day(self):
        # B is missing one mid-series day; A is complete. With positional
        # indexing the tail would misalign by one day after the gap.
        def day(i):
            return f"2024-{1 + i // 28:02d}-{1 + i % 28:02d}"

        a = [(day(i), 100.0 + i, 1.0) for i in range(80)]
        b = [(day(i), 200.0 + i, 1.0) for i in range(80) if i != 40]
        prices = {"A": a, "B": b}

        hist = ev.build_history(prices, self._const_score, horizon=1)
        # Every cross-section's score for each ticker must equal that ticker's
        # close on the cross-section's own date (proves date-aligned lookup).
        a_by_date = {d: c for d, c, _ in a}
        b_by_date = {d: c for d, c, _ in b}
        for cross in hist:
            d = cross["date"]
            for t, score in cross["scores"].items():
                expected = (a_by_date if t == "A" else b_by_date)[d]
                self.assertEqual(score, expected)
        # The day B is missing must not appear as a B score.
        self.assertTrue(all(day(40) != c["date"] or "B" not in c["scores"]
                            for c in hist))


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


class TestAlphaVantageParsing(unittest.TestCase):
    """Validate the Alpha Vantage response parsing offline (no network), proving
    it produces the same neutral contract."""

    def test_parses_intraday_payload_ascending(self):
        from alpha_pipeline.data import _parse_alphavantage

        payload = {
            "Meta Data": {"2. Symbol": "IBM"},
            "Time Series (5min)": {  # API returns newest-first
                "2024-01-02 16:00:00": {"4. close": "102.0", "5. volume": "300"},
                "2024-01-02 15:55:00": {"4. close": "101.0", "5. volume": "200"},
                "2024-01-02 15:50:00": {"4. close": "100.0", "5. volume": "100"},
            },
        }
        rows = _parse_alphavantage("IBM", payload, "5min")
        self.assertEqual([r[0] for r in rows], [
            "2024-01-02 15:50:00", "2024-01-02 15:55:00", "2024-01-02 16:00:00",
        ])  # sorted ascending
        self.assertEqual(rows[0], ("2024-01-02 15:50:00", 100.0, 100.0))

    def test_rate_limit_note_raises(self):
        from alpha_pipeline.data import _parse_alphavantage

        with self.assertRaises(RuntimeError):
            _parse_alphavantage("IBM", {"Note": "5 calls/min limit reached"}, "5min")

    def test_error_message_raises(self):
        from alpha_pipeline.data import _parse_alphavantage

        with self.assertRaises(RuntimeError):
            _parse_alphavantage("BADSYM", {"Error Message": "invalid symbol"}, "5min")

    def test_missing_key_raises(self):
        from alpha_pipeline.data import load_alphavantage

        old = os.environ.pop("ALPHAVANTAGE_API_KEY", None)
        try:
            with self.assertRaises(RuntimeError):
                load_alphavantage(("IBM",), cache_dir=None)  # force a live fetch
        finally:
            if old is not None:
                os.environ["ALPHAVANTAGE_API_KEY"] = old


class TestAlphaVantageCache(unittest.TestCase):
    """Validate the on-disk cache offline by stubbing the live fetch, so repeated
    runs don't burn the API quota."""

    def setUp(self):
        import alpha_pipeline.data as data
        self.data = data
        self.cache = tempfile.mkdtemp()
        self.calls = []
        self._real_fetch = data._av_fetch

        def fake_fetch(ticker, interval, outputsize):
            self.calls.append(ticker)
            return {f"Time Series ({interval})": {
                "2024-01-02 10:00:00": {"4. close": "100.0", "5. volume": "10"},
                "2024-01-02 10:05:00": {"4. close": "101.0", "5. volume": "12"},
            }}

        data._av_fetch = fake_fetch
        self.addCleanup(setattr, data, "_av_fetch", self._real_fetch)

    def test_miss_then_hit_avoids_second_fetch(self):
        first = self.data.load_alphavantage(("IBM",), cache_dir=self.cache)
        second = self.data.load_alphavantage(("IBM",), cache_dir=self.cache)
        self.assertEqual(self.calls, ["IBM"])  # only one live fetch
        self.assertEqual(first, second)
        self.assertEqual(len(first["IBM"]), 2)

    def test_expired_ttl_refetches(self):
        self.data.load_alphavantage(("IBM",), cache_dir=self.cache)
        # ttl=0 makes any cached file immediately stale.
        self.data.load_alphavantage(("IBM",), cache_dir=self.cache, cache_ttl=0)
        self.assertEqual(self.calls, ["IBM", "IBM"])  # refetched

    def test_disabled_cache_always_fetches(self):
        self.data.load_alphavantage(("IBM",), cache_dir=None)
        self.data.load_alphavantage(("IBM",), cache_dir=None)
        self.assertEqual(self.calls, ["IBM", "IBM"])


if __name__ == "__main__":
    unittest.main()
