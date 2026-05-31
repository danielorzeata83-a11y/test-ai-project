"""Price loaders. Every loader returns the same neutral contract:

    {ticker: [(date, close, volume), ...]}  # ascending by date

This single shape is the seam that makes the pipeline asset- and frequency-
neutral: swap the loader (synthetic -> CSV -> yfinance -> intraday bars) and the
rest of the system is unchanged.
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict

# A price row: (date_str, close, volume)
PriceRow = tuple[str, float, float]
PriceData = dict[str, list[PriceRow]]

# S&P 100 constituents (OEX), for real-data runs via yfinance/Alpaca. Membership
# drifts over time; refresh from your data provider when you go live.
SP100 = (
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DHR", "DIS", "DOW", "DUK", "EMR", "FDX", "GD", "GE", "GILD",
    "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU", "JNJ",
    "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT",
    "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE",
    "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTX", "SBUX",
    "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TXN", "UNH", "UNP",
    "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM",
)


def load_synthetic(
    tickers: tuple[str, ...],
    n_days: int = 60,
    seed: int = 0,
    start: str = "2024-01-01",
) -> PriceData:
    """Deterministic geometric random walk. For tests and offline demos."""
    import datetime as _dt

    rng = random.Random(seed)
    start_date = _dt.date.fromisoformat(start)
    out: PriceData = {}
    for t in tickers:
        price = 100.0 * (1.0 + 0.5 * rng.random())
        rows: list[PriceRow] = []
        for i in range(n_days):
            ret = rng.gauss(0.0003, 0.012)
            price *= math.exp(ret)
            vol = 1_000_000 * (0.5 + rng.random())
            rows.append((str(start_date + _dt.timedelta(days=i)), price, vol))
        out[t] = rows
    return out


def load_synthetic_drift(
    tickers: tuple[str, ...],
    n_days: int = 300,
    seed: int = 0,
    start: str = "2024-01-01",
    drift_scale: float = 0.0015,
) -> PriceData:
    """Like load_synthetic but each ticker carries a fixed persistent drift, so
    cross-sectional momentum is a genuine (detectable) edge. For meaningful
    demos; pure random-walk data (load_synthetic) intentionally has no edge."""
    import datetime as _dt

    rng = random.Random(seed)
    start_date = _dt.date.fromisoformat(start)
    drifts = {t: rng.uniform(-drift_scale, drift_scale) for t in tickers}
    out: PriceData = {}
    for t in tickers:
        price = 100.0
        rows: list[PriceRow] = []
        for i in range(n_days):
            ret = drifts[t] + rng.gauss(0.0, 0.008)
            price *= math.exp(ret)
            vol = 1_000_000 * (0.5 + rng.random())
            rows.append((str(start_date + _dt.timedelta(days=i)), price, vol))
        out[t] = rows
    return out


def load_synthetic_intraday(
    tickers: tuple[str, ...], n_days: int = 30, bars_per_day: int = 78,
    seed: int = 0, start: str = "2024-01-01", drift_scale: float = 0.0015,
) -> PriceData:
    """Synthetic intraday bars with persistent per-ticker drift. 78 bars/day =
    5-minute bars over a 6.5h US session. For offline validation of the intraday
    path; timestamps are 'YYYY-MM-DD HH:MM' so they sort correctly."""
    import datetime as _dt

    rng = random.Random(seed)
    start_date = _dt.date.fromisoformat(start)
    drifts = {t: rng.uniform(-drift_scale, drift_scale) for t in tickers}
    per_bar_drift = {t: drifts[t] / bars_per_day for t in tickers}
    bar_vol = 0.008 / (bars_per_day ** 0.5)
    out: PriceData = {}
    for t in tickers:
        price = 100.0
        rows: list[PriceRow] = []
        for d in range(n_days):
            day = start_date + _dt.timedelta(days=d)
            for b in range(bars_per_day):
                minute = 9 * 60 + 30 + b * 5  # 09:30 + 5min steps
                ts = f"{day} {minute // 60:02d}:{minute % 60:02d}"
                price *= math.exp(per_bar_drift[t] + rng.gauss(0.0, bar_vol))
                rows.append((ts, price, 1e5 * (0.5 + rng.random())))
        out[t] = rows
    return out


def load_csv(path: str) -> PriceData:
    """Load from a CSV with columns: date, ticker, close, volume."""
    out: PriceData = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["ticker"]].append(
                (row["date"], float(row["close"]), float(row["volume"]))
            )
    for t in out:
        out[t].sort(key=lambda r: r[0])
    return dict(out)


def load_yfinance(tickers: tuple[str, ...], period: str = "1y") -> PriceData:
    """Load daily closes from yfinance. Lazy import so the package has no hard
    dependency on it; raises a clear error if it's not installed."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "yfinance is not installed; `pip install yfinance` or use a CSV/"
            "synthetic loader"
        ) from exc

    out: PriceData = {}
    data = yf.download(
        list(tickers), period=period, group_by="ticker", progress=False
    )
    for t in tickers:
        rows: list[PriceRow] = []
        sub = data[t] if len(tickers) > 1 else data
        for ts, r in sub.iterrows():
            close, vol = r.get("Close"), r.get("Volume", 0.0)
            if close is None or (isinstance(close, float) and math.isnan(close)):
                continue
            rows.append((str(ts.date()), float(close), float(vol)))
        out[t] = rows
    return out
