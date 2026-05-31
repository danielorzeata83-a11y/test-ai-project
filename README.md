# Alpha Pipeline — multi-agent paper-trading system

A modular, daily-first system that turns price data into disciplined buy/sell
decisions, built on the principle that **robustness comes from a few pieces that
understand each other perfectly** — not from clever, adaptive complexity.

> Research & paper-trading tool, **not investment advice**. Backtest performance
> does not guarantee future results. The system protects you from being fooled
> by a pretty backtest; it does not manufacture edge.

## Quick start

```bash
python run.py                 # synthetic paper-trading demo from config.yaml
python -m scripts.promote_alpha momentum   # run an alpha through the Gate
python -m unittest discover -s tests       # run the full test suite
```

No third-party dependencies — pure Python stdlib. (`yfinance` is an optional
lazy import, only needed for real market data.)

## The bots (a small trading desk)

A linear, one-way chain: **Analyst → Signal → Risk → Execution**, coordinated by
an Orchestrator and recorded by a Journal.

| Bot | Job | Key guarantee |
|-----|-----|---------------|
| **Analyst** (`analyst.py`) | Load + validate data, compute features | Never passes garbage downstream (fail-soft on data, fail-hard on bugs) |
| **Signal** (`signal.py`) | Run promoted alphas → long-short weights | Only trades alphas that passed the Gate |
| **Risk** (`risk.py`) | 7 controls + regime overlay | Every control only *reduces* exposure |
| **Execution** (`execution.py`) | Weights → simulated fills with costs | Stateless; replay-exact |
| **Orchestrator** (`orchestrator.py`) | Drive the daily cycle | Fail-closed: uncertainty → less risk |
| **Journal** (`journal.py`) | Append-only audit trail | Idempotent cycles, exact state replay |

## The Validation Gate (`gate.py`)

Six examiners; an alpha must pass all of them to be promoted:

1. **Statistician** — IC > 0, t-stat > 3.0
2. **Cost analyst** — IC doesn't decay to nothing
3. **Realist** — Sharpe survives 15 bps of cost
4. **Skeptic** — Deflated Sharpe Ratio (multiple-testing guard)
5. **Engineer** — CPCV positive across time splits
6. **Librarian** — correlation < 0.6 with promoted alphas

## Alphas — including a fundamental one

The library (`alphas.py`) has four signals: three price-based (`momentum`,
`reversal`, `low_vol`) and one fundamental — `quality` (earnings_yield + roe +
profit_margin). Quality is the diversifier: it reads fundamentals the Analyst
merges into the panel, so it falls for different reasons than the price alphas
(correlation ~0.12 with momentum on the demo data, well under the Gate's 0.6).

```bash
# Offline demo: both momentum and quality clear the Gate and trade together.
# (set source: synthetic_fundamental in config.yaml)
python run.py     # -> Promoted alphas: ('momentum', 'quality')
```

Live fundamentals come from Alpha Vantage OVERVIEW
(`fundamentals_source: alphavantage`). **Caveat:** OVERVIEW is a *current
snapshot*, not point-in-time — using it in a historical backtest leaks
future information (lookahead/survivorship bias). It's fine for live/paper
trading going forward; for research backtests, use point-in-time fundamentals.

## Regime overlay (`regime.py`)

A stdlib 2-state Gaussian HMM detects calm vs crisis volatility regimes
(~98% recovery on synthetic data, with anti-label-switching). The Risk bot cuts
exposure when crisis probability is high — the HMM **protects** alpha, it doesn't
generate it.

## Configuration (`config.yaml`)

Every tunable parameter lives in one versioned file: universe, Gate thresholds,
risk limits, costs. Edit there, not in code.

## Real data & intraday

The whole pipeline depends on one neutral data contract:
`{ticker: [(timestamp, close, volume)]}`. Swap the loader and the rest of the
system is unchanged — that's the seam from daily to intraday.

**Daily, real data (S&P 100 via yfinance):**

```bash
pip install yfinance
# config.yaml under universe:
#   names: (use the SP100 tuple from alpha_pipeline/data.py)
#   source: yfinance
#   period: 2y
python run.py
```

**Intraday (5-minute bars):**

```yaml
# config.yaml
universe:
  source: yfinance_intraday   # or: alpaca
  interval: 5m                # yfinance; for alpaca use timeframe: 5Min
  period: 5d
frequency:
  bars_per_day: 78            # 5-min bars over a 6.5h US session
```

```bash
# Alpaca (recommended: paper data + execution in one API)
pip install alpaca-py
export APCA_API_KEY_ID=...  APCA_API_SECRET_KEY=...
python run.py
```

**Alpha Vantage (intraday, no extra dependency — uses stdlib):**

```yaml
# config.yaml
universe:
  source: alphavantage
  interval: 5min              # 1min | 5min | 15min | 30min | 60min
  outputsize: full            # 'full' history or 'compact' (latest 100 bars)
  pause: 15                   # seconds between tickers (free tier ~5 req/min)
frequency:
  bars_per_day: 78
```

```bash
export ALPHAVANTAGE_API_KEY=...   # free key: alphavantage.co/support/#api-key
python run.py
```

Note the free tier is rate-limited (~5 requests/min, ~25/day), so a wide
universe needs either the `pause` spacing above or a premium key. Responses are
cached on disk under `.av_cache/` per (ticker, interval, outputsize); a run
within `cache_ttl` seconds (default 3600) reuses the cache instead of calling
the API, so repeated runs don't burn the daily quota. Set `cache_dir: null` to
disable.

**Why `bars_per_day` matters.** Every annualized number (Sharpe, Deflated
Sharpe, vol target) scales by `periods_per_year = 252 * bars_per_day`. Set it
correctly or intraday Sharpes come out wildly inflated. Daily = 1; 5-minute US
session = 78.

> Network note: the synthetic and CSV paths run fully offline and are covered by
> tests. The yfinance/Alpaca paths require network (and, for Alpaca, API keys);
> they're built to the same contract but validated on your machine, not in CI.

## Layout

```text
alpha_pipeline/   contracts, journal, analyst, signal, risk, regime,
                  execution, orchestrator, registry, gate, evaluate,
                  alphas, data, config, monitor
scripts/          promote_alpha.py
run.py            paper-trading runner
tests/            full suite
config.yaml       all parameters
```
