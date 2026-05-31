"""Alpha library: cross-sectional signal functions.

An alpha maps per-ticker features -> per-ticker score. Higher score = more
attractive to go long. They are pure functions of one cross-section (no
lookahead), so the same function runs live in the Signal bot and historically
in the evaluation/Gate.

ALPHAS is the full library; the Registry decides which are in production.
"""

from __future__ import annotations

from typing import Callable, Mapping

# features: {ticker: {feature_name: value}}  ->  {ticker: score}
Features = Mapping[str, Mapping[str, float]]
AlphaFn = Callable[[Features], dict]


def momentum(features: Features) -> dict:
    """12-1 style momentum: longer-horizon trend, skipping the last month."""
    return {t: f["mom_63"] - f["mom_21"] for t, f in features.items()}


def reversal(features: Features) -> dict:
    """Short-term mean reversion: yesterday's losers tend to bounce."""
    return {t: -f["ret_1d"] for t, f in features.items()}


def low_vol(features: Features) -> dict:
    """Low-volatility anomaly: prefer calmer names."""
    return {t: -f["vol_21"] for t, f in features.items()}


ALPHAS: dict[str, AlphaFn] = {
    "momentum": momentum,
    "reversal": reversal,
    "low_vol": low_vol,
}
