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
