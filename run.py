"""Paper-trading runner: wires the whole system from config and runs the daily
cycle over a series of dates.

Usage:
    python run.py                      # synthetic demo from config.yaml
    python run.py --days 120 --config config.yaml

At daily frequency the loop advances one trading day per cycle; the same loop
carries intraday bars once the loader returns them (the data contract is
unchanged). This is research/paper-trading, not investment advice.
"""

from __future__ import annotations

import argparse
import os
import tempfile

from alpha_pipeline.analyst import AnalystBot
from alpha_pipeline.config import Config
from alpha_pipeline.contracts import PortfolioState
from alpha_pipeline import data as datamod
from alpha_pipeline.execution import ExecutionBot
from alpha_pipeline.gate import GateConfig
from alpha_pipeline.journal import Journal
from alpha_pipeline.monitor import health
from alpha_pipeline.orchestrator import Orchestrator
from alpha_pipeline.registry import Registry
from alpha_pipeline.risk import RiskBot, RiskConfig
from alpha_pipeline.signal import SignalBot


def make_loader(cfg: Config, days: int, seed: int):
    """Return a zero-arg callable producing PriceData for the configured source.
    Synthetic sources are reproducible offline; real sources need network/keys."""
    u = cfg.section("universe")
    universe = tuple(u["names"])
    source = u.get("source", "synthetic")
    bars = cfg.section("frequency").get("bars_per_day", 1)

    if source == "synthetic":
        return lambda: datamod.load_synthetic(universe, n_days=days, seed=seed)
    if source == "synthetic_drift":
        return lambda: datamod.load_synthetic_drift(universe, n_days=days, seed=seed)
    if source == "synthetic_intraday":
        return lambda: datamod.load_synthetic_intraday(
            universe, n_days=days, bars_per_day=bars, seed=seed)
    if source == "csv":
        return lambda: datamod.load_csv(u["csv_path"])
    if source == "yfinance":
        return lambda: datamod.load_yfinance(universe, period=u.get("period", "2y"))
    if source == "yfinance_intraday":
        return lambda: datamod.load_yfinance_intraday(
            universe, period=u.get("period", "5d"), interval=u.get("interval", "5m"))
    if source == "alpaca":
        return lambda: datamod.load_alpaca(
            universe, timeframe=u.get("timeframe", "1Day"))
    if source == "alphavantage":
        return lambda: datamod.load_alphavantage(
            universe, interval=u.get("interval", "5min"),
            outputsize=u.get("outputsize", "full"), pause=u.get("pause", 15.0))
    raise ValueError(f"unknown source {source!r}")


def periods_per_year(cfg: Config) -> int:
    return 252 * cfg.section("frequency").get("bars_per_day", 1)


def build_system(cfg: Config, prices, journal, registry):
    universe = tuple(cfg.section("universe")["names"])
    a = cfg.section("analyst")
    analyst = AnalystBot(
        universe, lambda: prices,
        min_coverage=a["min_coverage"], max_return_spike=a["max_return_spike"],
        min_history=a["min_history"],
    )
    signal = SignalBot(registry, top_frac=cfg.section("signal")["top_frac"])
    r = dict(cfg.section("risk"))
    r["periods_per_year"] = periods_per_year(cfg)
    risk = RiskBot(RiskConfig(**{k: r[k] for k in r if k in RiskConfig.__dataclass_fields__}))
    e = cfg.section("execution")
    execution = ExecutionBot(e["commission_bps"], e["slippage_bps"])
    return Orchestrator(journal, analyst, signal, risk, execution)


def run(cfg: Config, days: int, seed: int, workdir: str):
    universe = tuple(cfg.section("universe")["names"])
    prices = make_loader(cfg, days, seed)()
    dates = [r[0] for r in prices[universe[0]]]
    ppy = periods_per_year(cfg)

    registry = Registry(os.path.join(workdir, "registry.json"))
    # Auto-promote any library alpha that clears the Gate, so the demo trades.
    if not registry.active_names():
        from alpha_pipeline.alphas import ALPHAS
        from alpha_pipeline.gate import evaluate_alpha
        g = dict(cfg.section("gate"))
        g["periods_per_year"] = ppy
        gcfg = GateConfig(**{k: g[k] for k in g if k in GateConfig.__dataclass_fields__})
        for name, fn in ALPHAS.items():
            res = evaluate_alpha(prices, fn, name, registry, gcfg)
            if res.passed:
                registry.promote(name, {k: round(v, 6) for k, v in res.metrics.items()})

    # Analyst clamps min_history to >=64; mirror that for the warmup window.
    warmup = max(cfg.section("analyst")["min_history"], 64)
    initial = cfg.section("portfolio")["initial_cash"]
    if len(dates) <= warmup + 1:
        raise ValueError(
            f"need more than {warmup + 1} dates for warmup, got {len(dates)}"
        )
    with Journal(os.path.join(workdir, "journal.db")) as journal:
        orch = build_system(cfg, prices, journal, registry)
        state = PortfolioState.initial(dates[warmup], initial)
        # Start after the warmup window so features exist.
        for d in dates[warmup + 1:]:
            state, _ = orch.run_cycle(d, state)
            if state.halted:
                print(f"HALTED on {d}: {state.halt_reason}")
                break
        report = health(journal, ppy)
    print("Promoted alphas:", registry.active_names())
    print("Paper-trading result:", report)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run paper-trading")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--days", type=int, default=300)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--workdir", default=None)
    args = p.parse_args(argv)
    workdir = args.workdir or tempfile.mkdtemp(prefix="paper_")
    run(Config.load(args.config), args.days, args.seed, workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
