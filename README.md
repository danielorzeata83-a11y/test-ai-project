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

## Regime overlay (`regime.py`)

A stdlib 2-state Gaussian HMM detects calm vs crisis volatility regimes
(~98% recovery on synthetic data, with anti-label-switching). The Risk bot cuts
exposure when crisis probability is high — the HMM **protects** alpha, it doesn't
generate it.

## Configuration (`config.yaml`)

Every tunable parameter lives in one versioned file: universe, Gate thresholds,
risk limits, costs. Edit there, not in code.

## Toward intraday

The whole pipeline depends on one neutral data contract:
`{ticker: [(date, close, volume)]}`. Swap the loader (synthetic → CSV →
yfinance → intraday bars) and the rest of the system is unchanged. Daily is
built and tested; intraday is the same chain with a different loader and an
intraday-capable data source (e.g. Alpaca/Polygon).

## Layout

```
alpha_pipeline/   contracts, journal, analyst, signal, risk, regime,
                  execution, orchestrator, registry, gate, evaluate,
                  alphas, data, config, monitor
scripts/          promote_alpha.py
run.py            paper-trading runner
tests/            full suite
config.yaml       all parameters
```
