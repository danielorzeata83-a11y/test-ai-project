"""Analyst bot: the data-quality guardian, first in the chain.

Loads prices, validates them, computes per-ticker features, and emits a
FeaturePanel. It excludes individual bad tickers but keeps going; it refuses
the whole panel (valid=False) only when the data is too broken to trust.

Two failure modes, deliberately different:
  - fail-soft (data problems: stale, low coverage, loader error) -> invalid
    panel, the cycle is skipped, retry tomorrow. Never propagates.
  - fail-hard (programming bugs: TypeError etc.) -> propagates, Orchestrator
    halts. Those genuinely need a human.

Features are scalar-per-ticker (close + volume + momentum/vol windows) so the
panel matches the FeaturePanel contract. Signal combines them downstream.
"""

from __future__ import annotations

import math
from typing import Callable

from .contracts import FeaturePanel
from .data import PriceData

# A loader is any callable returning PriceData for the configured universe.
Loader = Callable[[], PriceData]


class AnalystBot:
    def __init__(
        self,
        universe: tuple[str, ...],
        loader: Loader,
        min_coverage: float = 0.8,
        max_return_spike: float = 0.5,
        min_history: int = 64,
    ) -> None:
        self.universe = universe
        self.loader = loader
        self.min_coverage = min_coverage
        self.max_return_spike = max_return_spike
        # Features need a 63-day window (closes[-64]); never go below that.
        self.min_history = max(min_history, 64)

    def run(self, date: str) -> FeaturePanel:
        # fail-soft on data-source problems (network, bad rows), fail-hard on
        # programmer bugs (TypeError etc.) so real defects surface, not hide.
        try:
            prices = self.loader()
        except (TypeError, AttributeError, NameError, KeyError, IndexError):
            raise  # programmer bug -> propagate -> Orchestrator halts
        except Exception as exc:
            return self._invalid(date, {"error": f"loader failed: {exc!r}"})

        data: dict[str, dict[str, float]] = {}
        excluded: dict[str, str] = {}

        for t in self.universe:
            rows = prices.get(t, [])
            # Keep only rows up to and including the requested date.
            rows = [r for r in rows if r[0] <= date]
            if len(rows) < self.min_history:
                excluded[t] = f"insufficient history ({len(rows)})"
                continue

            try:
                feats = self._features(rows)
            except (ZeroDivisionError, ValueError, TypeError) as exc:
                # One malformed ticker must not sink the whole panel.
                excluded[t] = f"feature extraction error: {exc}"
                continue
            if feats is None:
                excluded[t] = "non-finite feature"
                continue

            spike = abs(feats["ret_1d"])
            if spike > self.max_return_spike:
                excluded[t] = f"return spike |{spike:.1%}| exceeds {self.max_return_spike:.0%}"
                continue

            data[t] = feats

        coverage = len(data) / len(self.universe) if self.universe else 0.0
        meta = {
            "coverage": coverage,
            "excluded": excluded,
            "n_included": len(data),
        }

        if coverage < self.min_coverage:
            meta["error"] = (
                f"coverage {coverage:.0%} below threshold {self.min_coverage:.0%}"
            )
            missing = sorted(set(self.universe) - set(data))
            meta["missing"] = missing
            return FeaturePanel(date, tuple(sorted(data)), data, False, meta)

        return FeaturePanel(date, tuple(sorted(data)), data, True, meta)

    @staticmethod
    def _features(rows) -> dict[str, float] | None:
        """Compute scalar features from a price history (ascending by date)."""
        closes = [r[1] for r in rows]
        last = closes[-1]
        prev = closes[-2]
        ret_1d = last / prev - 1.0
        mom_21 = last / closes[-22] - 1.0
        mom_63 = last / closes[-64] - 1.0
        rets = [closes[i] / closes[i - 1] - 1.0 for i in range(-21, 0)]
        mean = sum(rets) / len(rets)
        var = sum((x - mean) ** 2 for x in rets) / len(rets)
        vol_21 = math.sqrt(var)
        feats = {
            "close": last,
            "volume": rows[-1][2],
            "ret_1d": ret_1d,
            "mom_21": mom_21,
            "mom_63": mom_63,
            "vol_21": vol_21,
        }
        if not all(math.isfinite(v) for v in feats.values()):
            return None
        return feats

    @staticmethod
    def _invalid(date: str, meta: dict) -> FeaturePanel:
        return FeaturePanel(date, (), {}, False, meta)
