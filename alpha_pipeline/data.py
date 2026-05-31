"""Price loaders. Every loader returns the same neutral contract:

    {ticker: [(date, close, volume), ...]}  # ascending by date

This single shape is the seam that makes the pipeline asset- and frequency-
neutral: swap the loader (synthetic -> CSV -> yfinance -> intraday bars) and the
rest of the system is unchanged.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
import urllib.parse
import urllib.request
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


def load_synthetic_fundamental(
    tickers: tuple[str, ...], n_days: int = 400, seed: int = 0,
    start: str = "2024-01-01",
) -> tuple[PriceData, dict[str, dict[str, float]]]:
    """Synthetic prices + per-ticker fundamentals for the quality alpha.

    Each ticker has TWO independent drivers of future return:
      - a momentum drift (random per ticker), and
      - a quality tilt derived from its fundamentals (earnings_yield, roe,
        profit_margin).
    Because the two drivers are drawn independently, the quality signal is
    genuinely predictive (clears the Gate) yet uncorrelated with momentum
    (clears the correlation test) — demonstrating real diversification, not just
    asserting it. Returns (prices, fundamentals).
    """
    import datetime as _dt

    rng = random.Random(seed)
    start_date = _dt.date.fromisoformat(start)

    fundamentals: dict[str, dict[str, float]] = {}
    mom_drift: dict[str, float] = {}
    qual_drift: dict[str, float] = {}
    for t in tickers:
        ey = rng.uniform(-1.0, 1.0)   # standardized earnings yield
        roe = rng.uniform(-1.0, 1.0)
        pm = rng.uniform(-1.0, 1.0)
        fundamentals[t] = {"earnings_yield": ey, "roe": roe, "profit_margin": pm}
        # Quality composite drives a small, persistent return tilt.
        qual_drift[t] = 0.0020 * (ey + roe + pm) / 3.0
        # Momentum drift is independent of fundamentals.
        mom_drift[t] = rng.uniform(-0.0015, 0.0015)

    out: PriceData = {}
    for t in tickers:
        price = 100.0
        rows: list[PriceRow] = []
        drift = mom_drift[t] + qual_drift[t]
        for i in range(n_days):
            price *= math.exp(drift + rng.gauss(0.0, 0.008))
            rows.append((str(start_date + _dt.timedelta(days=i)),
                         price, 1_000_000 * (0.5 + rng.random())))
        out[t] = rows
    return out, fundamentals


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


def load_yfinance_intraday(
    tickers: tuple[str, ...], period: str = "5d", interval: str = "5m"
) -> PriceData:
    """Load intraday bars from yfinance. interval e.g. '1m','5m','15m','60m';
    note yfinance limits 1m history to ~7 days. Same neutral contract, but the
    first field is a full timestamp string instead of a date."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("yfinance is not installed; `pip install yfinance`") from exc

    out: PriceData = {}
    data = yf.download(
        list(tickers), period=period, interval=interval,
        group_by="ticker", progress=False,
    )
    for t in tickers:
        rows: list[PriceRow] = []
        sub = data[t] if len(tickers) > 1 else data
        for ts, r in sub.iterrows():
            close, vol = r.get("Close"), r.get("Volume", 0.0)
            if close is None or (isinstance(close, float) and math.isnan(close)):
                continue
            rows.append((str(ts), float(close), float(vol)))
        out[t] = rows
    return out


def load_alpaca(
    tickers: tuple[str, ...], timeframe: str = "1Day", limit: int = 1000,
) -> PriceData:
    """Load bars from Alpaca. timeframe e.g. '1Day','1Hour','5Min'. Keys come
    from APCA_API_KEY_ID / APCA_API_SECRET_KEY. Lazy import of alpaca-py."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("alpaca-py is not installed; `pip install alpaca-py`") from exc

    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("set APCA_API_KEY_ID and APCA_API_SECRET_KEY")

    unit_map = {"Day": TimeFrameUnit.Day, "Hour": TimeFrameUnit.Hour,
                "Min": TimeFrameUnit.Minute}
    amount = int("".join(c for c in timeframe if c.isdigit()) or "1")
    unit = next((u for name, u in unit_map.items() if name in timeframe), None)
    if unit is None:
        raise ValueError(
            f"unrecognized timeframe {timeframe!r}; expected Day, Hour, or Min "
            "(e.g. '1Day', '1Hour', '5Min')"
        )

    client = StockHistoricalDataClient(key, secret)
    req = StockBarsRequest(
        symbol_or_symbols=list(tickers),
        timeframe=TimeFrame(amount, unit), limit=limit,
    )
    bars = client.get_stock_bars(req).data
    out: PriceData = {}
    for t in tickers:
        out[t] = [(str(b.timestamp), float(b.close), float(b.volume))
                  for b in bars.get(t, [])]
    return out


def load_alphavantage(
    tickers: tuple[str, ...], interval: str = "5min", outputsize: str = "compact",
    pause: float = 0.0, cache_dir: str | None = ".av_cache", cache_ttl: float = 3600.0,
) -> PriceData:
    """Load intraday bars from Alpha Vantage (TIME_SERIES_INTRADAY), stdlib only.

    The API key is read from the ALPHAVANTAGE_API_KEY environment variable, never
    passed in code. interval in {1min,5min,15min,30min,60min}; outputsize
    'compact' (latest 100 bars) or 'full'. Free tier is rate-limited (~5 req/min,
    25/day) — set `pause` (e.g. 15) to space out requests for multiple tickers.

    Caching: raw responses are cached per (ticker, interval, outputsize) under
    `cache_dir`. A cached file younger than `cache_ttl` seconds is reused instead
    of calling the API, so repeated runs don't burn the daily request quota. Set
    cache_dir=None to disable. Network calls (and pause) only happen on a miss.

    Same neutral contract: {ticker: [(timestamp, close, volume)]} ascending.
    """
    out: PriceData = {}
    fetched = 0
    for t in tickers:
        payload = _av_cache_read(cache_dir, t, interval, outputsize, cache_ttl)
        if payload is None:
            if fetched and pause:
                time.sleep(pause)  # respect free-tier rate limit between calls
            payload = _av_fetch(t, interval, outputsize)
            fetched += 1
            _av_cache_write(cache_dir, t, interval, outputsize, payload)
        out[t] = _parse_alphavantage(t, payload, interval)
    return out


def _av_cache_path(cache_dir: str, ticker: str, interval: str, outputsize: str) -> str:
    return os.path.join(cache_dir, f"{ticker}_{interval}_{outputsize}.json")


def _av_cache_read(cache_dir, ticker, interval, outputsize, ttl) -> dict | None:
    if not cache_dir:
        return None
    path = _av_cache_path(cache_dir, ticker, interval, outputsize)
    if not os.path.exists(path) or (time.time() - os.path.getmtime(path)) > ttl:
        return None
    with open(path) as f:
        return json.load(f)


def _av_cache_write(cache_dir, ticker, interval, outputsize, payload) -> None:
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    path = _av_cache_path(cache_dir, ticker, interval, outputsize)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)  # atomic; never leave a half-written cache file


def _av_fetch(ticker: str, interval: str, outputsize: str) -> dict:
    """One live API call. Key read from env at call time, never stored."""
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "set ALPHAVANTAGE_API_KEY (get a free key at "
            "https://www.alphavantage.co/support/#api-key)"
        )
    params = urllib.parse.urlencode({
        "function": "TIME_SERIES_INTRADAY",
        "symbol": ticker,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
    })
    url = f"https://www.alphavantage.co/query?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def _parse_alphavantage(ticker: str, payload: dict, interval: str) -> list[PriceRow]:
    """Turn an Alpha Vantage intraday response into ascending PriceRows. Raises a
    clear error on rate-limit / error notes instead of returning empty data."""
    if "Note" in payload or "Information" in payload:
        raise RuntimeError(
            f"Alpha Vantage throttled/limited for {ticker}: "
            f"{payload.get('Note') or payload.get('Information')}"
        )
    if "Error Message" in payload:
        raise RuntimeError(f"Alpha Vantage error for {ticker}: {payload['Error Message']}")
    series = payload.get(f"Time Series ({interval})")
    if not series:
        raise RuntimeError(f"Alpha Vantage returned no series for {ticker}: {payload}")
    rows = [
        (ts, float(bar["4. close"]), float(bar["5. volume"]))
        for ts, bar in series.items()
    ]
    rows.sort(key=lambda r: r[0])  # API returns newest-first; we want ascending
    return rows
