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
from alpha_pipeline.data import load_synthetic, load_synthetic_drift
from alpha_pipeline.execution import ExecutionBot
from alpha_pipeline.gate import GateConfig
from alpha_pipeline.journal import Journal
from alpha_pipeline.monitor import health
from alpha_pipeline.orchestrator import Orchestrator
from alpha_pipeline.registry import Registry
from alpha_pipeline.risk import RiskBot, RiskConfig
from alpha_pipeline.signal import SignalBot


def build_system(cfg: Config, prices, journal, registry):
    universe = tuple(cfg.section("universe")["names"])
    a = cfg.section("analyst")
    analyst = AnalystBot(
        universe, lambda: prices,
        min_coverage=a["min_coverage"], max_return_spike=a["max_return_spike"],
        min_history=a["min_history"],
    )
    signal = SignalBot(registry, top_frac=cfg.section("signal")["top_frac"])
    r = cfg.section("risk")
    risk = RiskBot(RiskConfig(**{k: r[k] for k in r if k in RiskConfig.__dataclass_fields__}))
    e = cfg.section("execution")
    execution = ExecutionBot(e["commission_bps"], e["slippage_bps"])
    return Orchestrator(journal, analyst, signal, risk, execution)


def run(cfg: Config, days: int, seed: int, workdir: str):
    universe = tuple(cfg.section("universe")["names"])
    source = cfg.section("universe").get("source", "synthetic")
    loader = load_synthetic_drift if source == "synthetic_drift" else load_synthetic
    prices = loader(universe, n_days=days, seed=seed)
    dates = [prices[universe[0]][i][0] for i in range(len(prices[universe[0]]))]

    registry = Registry(os.path.join(workdir, "registry.json"))
    # Auto-promote any library alpha that clears the Gate, so the demo trades.
    if not registry.active_names():
        from alpha_pipeline.alphas import ALPHAS
        from alpha_pipeline.gate import evaluate_alpha
        g = cfg.section("gate")
        gcfg = GateConfig(**{k: g[k] for k in g if k in GateConfig.__dataclass_fields__})
        for name, fn in ALPHAS.items():
            if evaluate_alpha(prices, fn, name, registry, gcfg).passed:
                registry.promote(name, {})

    initial = cfg.section("portfolio")["initial_cash"]
    with Journal(os.path.join(workdir, "journal.db")) as journal:
        orch = build_system(cfg, prices, journal, registry)
        state = PortfolioState.initial(dates[64], initial)
        # Start after the warmup window so features exist.
        for d in dates[65:]:
            state, _ = orch.run_cycle(d, state)
            if state.halted:
                print(f"HALTED on {d}: {state.halt_reason}")
                break
        report = health(journal)
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
