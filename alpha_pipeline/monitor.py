"""Monitoring: reads the Journal and reports portfolio health.

Pure reporting — no side effects, no trading. Reconstructs the NAV path from the
portfolio snapshots and derives the metrics you actually watch day to day.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .journal import Journal


@dataclass
class HealthReport:
    n_cycles: int
    nav: float
    total_return: float
    max_drawdown: float
    sharpe: float
    halted: bool
    n_critical_events: int

    def __str__(self) -> str:
        return (
            f"cycles={self.n_cycles} nav={self.nav:,.2f} "
            f"ret={self.total_return:+.2%} maxdd={self.max_drawdown:.2%} "
            f"sharpe={self.sharpe:.2f} halted={self.halted} "
            f"critical={self.n_critical_events}"
        )


def health(journal: Journal, periods_per_year: int = 252) -> HealthReport:
    rows = journal._conn.execute(
        "SELECT payload FROM portfolio_snaps ORDER BY date"
    ).fetchall()
    import json

    navs = [json.loads(r["payload"])["nav"] for r in rows]
    states = [json.loads(r["payload"]) for r in rows]

    if not navs:
        return HealthReport(0, 0.0, 0.0, 0.0, 0.0, False, 0)

    start, end = navs[0], navs[-1]
    total_return = end / start - 1.0 if start else 0.0

    peak = navs[0]
    max_dd = 0.0
    for v in navs:
        peak = max(peak, v)
        max_dd = min(max_dd, v / peak - 1.0)

    rets = [navs[i] / navs[i - 1] - 1.0 for i in range(1, len(navs))]
    if len(rets) >= 2 and statistics.pstdev(rets) > 0:
        sharpe = statistics.fmean(rets) / statistics.pstdev(rets) * math.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    n_crit = journal._conn.execute(
        "SELECT COUNT(*) c FROM events WHERE level = 'CRITICAL'"
    ).fetchone()["c"]

    return HealthReport(
        n_cycles=len(navs),
        nav=end,
        total_return=total_return,
        max_drawdown=max_dd,
        sharpe=sharpe,
        halted=states[-1]["halted"],
        n_critical_events=n_crit,
    )
