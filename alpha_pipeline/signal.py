"""Signal bot: reads promoted alphas from the Registry, runs them on the clean
panel, combines them, and emits TargetWeights.

Combination is cross-sectional rank averaging: each alpha's scores are turned
into ranks, ranks are averaged across active alphas (equal weight), then the
top/bottom fraction become a dollar-neutral long-short book. Ranks make alphas
on different scales comparable and are robust to outliers.

No risk applied here — that is the Risk bot's job downstream.
"""

from __future__ import annotations

from .alphas import ALPHAS
from .contracts import FeaturePanel, TargetWeights
from .evaluate import _rank
from .registry import Registry


class SignalBot:
    def __init__(self, registry: Registry, top_frac: float = 0.2) -> None:
        self.registry = registry
        self.top_frac = top_frac

    def run(self, panel: FeaturePanel) -> TargetWeights:
        tickers = list(panel.tickers)
        active = [n for n in self.registry.active_names() if n in ALPHAS]

        if not active or len(tickers) < 2:
            return TargetWeights(panel.date, {})

        # Average rank across active alphas.
        agg = {t: 0.0 for t in tickers}
        for name in active:
            scores = ALPHAS[name](panel.data)
            vals = [scores[t] for t in tickers]
            ranks = _rank(vals)
            for t, r in zip(tickers, ranks):
                agg[t] += r
        combined = {t: agg[t] / len(active) for t in tickers}

        ranked = sorted(tickers, key=lambda t: combined[t])
        # Clamp to half the universe so longs and shorts never overlap.
        n = max(1, min(int(len(ranked) * self.top_frac), len(ranked) // 2))
        shorts, longs = ranked[:n], ranked[-n:]

        weights: dict[str, float] = {}
        for t in longs:
            weights[t] = 0.5 / n
        for t in shorts:
            weights[t] = -0.5 / n
        return TargetWeights(panel.date, weights)
