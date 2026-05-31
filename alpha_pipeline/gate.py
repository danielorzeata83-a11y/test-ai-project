"""Validation Gate: six examiners, all-or-nothing. An alpha must pass every
one to be promoted. Each looks for a different kind of lie.

  1. Statistician  — IC positive, IC_IR, t-stat > threshold.
  2. Cost analyst  — IC doesn't decay to nothing (IC[5d] >= frac * IC[1d]).
  3. Realist       — net-of-cost Sharpe survives.
  4. Skeptic       — Deflated Sharpe Ratio > threshold (multiple-testing guard).
  5. Engineer      — CPCV positive in enough paths (temporal robustness).
  6. Librarian     — correlation < threshold with already-promoted alphas.

Thresholds come from config; the defaults mirror the production values decided
in the design discussion.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import evaluate as ev
from .alphas import AlphaFn
from .registry import Registry


@dataclass
class GateConfig:
    min_ic: float = 0.005
    min_ic_ir: float = 0.05
    min_t_stat: float = 3.0
    min_decay_ratio: float = 0.3      # IC[5d] / IC[1d]
    cost_per_turnover: float = 0.0015  # 15 bps
    min_net_sharpe: float = 0.5
    min_dsr: float = 0.95
    n_trials: int = 20
    min_cpcv_fraction: float = 0.8
    max_correlation: float = 0.6
    periods_per_year: int = 252  # 252 daily; 252*bars_per_day for intraday


class GateResult:
    def __init__(self) -> None:
        self.checks: dict[str, bool] = {}
        self.metrics: dict[str, float] = {}

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        failed = [k for k, v in self.checks.items() if not v]
        return f"<GateResult {status} failed={failed} metrics={self.metrics}>"


def evaluate_alpha(
    prices, alpha: AlphaFn, name: str, registry: Registry, cfg: GateConfig,
    fundamentals=None,
) -> GateResult:
    res = GateResult()
    history = ev.build_history(prices, alpha, horizon=1, fundamentals=fundamentals)

    # 1. Statistician
    icr = ev.information_coefficient(history)
    res.metrics.update(ic=icr["ic"], ic_ir=icr["ic_ir"], t_stat=icr["t_stat"])
    res.checks["statistician"] = (
        icr["ic"] >= cfg.min_ic
        and icr["ic_ir"] >= cfg.min_ic_ir
        and icr["t_stat"] >= cfg.min_t_stat
    )

    # 2. Cost analyst (decay)
    decay = ev.ic_decay(prices, alpha, horizons=(1, 5), fundamentals=fundamentals)
    ic1, ic5 = decay[1], decay[5]
    ratio = (ic5 / ic1) if ic1 > 0 else 0.0
    res.metrics["decay_ratio"] = ratio
    res.checks["cost_analyst"] = ic1 > 0 and ratio >= cfg.min_decay_ratio

    # 3. Realist (net Sharpe)
    daily = ev.long_short_returns(history)
    turn = ev.turnover(history)
    net_sharpe = ev.sharpe(daily, cfg.cost_per_turnover, turn, cfg.periods_per_year)
    res.metrics.update(net_sharpe=net_sharpe, turnover=turn)
    res.checks["realist"] = net_sharpe >= cfg.min_net_sharpe

    # 4. Skeptic (Deflated Sharpe)
    gross_sharpe = ev.sharpe(daily, periods=cfg.periods_per_year)
    dsr = ev.deflated_sharpe(daily, gross_sharpe, cfg.n_trials, cfg.periods_per_year)
    res.metrics["dsr"] = dsr
    res.checks["skeptic"] = dsr >= cfg.min_dsr

    # 5. Engineer (CPCV)
    cpcv = ev.cpcv_positive_fraction(history)
    res.metrics["cpcv_fraction"] = cpcv
    res.checks["engineer"] = cpcv >= cfg.min_cpcv_fraction

    # 6. Librarian (correlation with promoted alphas)
    from .alphas import ALPHAS
    max_corr = 0.0
    for other in registry.active_names():
        if other == name or other not in ALPHAS:
            continue
        other_hist = ev.build_history(
            prices, ALPHAS[other], horizon=1, fundamentals=fundamentals)
        # Correlate the daily long-short return streams.
        other_daily = ev.long_short_returns(other_hist)
        m = min(len(daily), len(other_daily))
        if m >= 2:
            corr = abs(ev._pearson(daily[-m:], other_daily[-m:]))
            max_corr = max(max_corr, corr)
    res.metrics["max_correlation"] = max_corr
    res.checks["librarian"] = max_corr < cfg.max_correlation

    return res
