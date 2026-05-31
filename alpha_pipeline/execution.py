"""Execution bot: turns risk-approved weights into simulated fills.

Stateless — all state lives in PortfolioState. Given target weights (fractions
of NAV), the current portfolio, and current prices (close, from the panel), it
computes the trades needed to move from current to target, applies cost +
slippage, and returns the new state. Replay-safe: same inputs -> same outputs.
"""

from __future__ import annotations

from .contracts import FeaturePanel, PortfolioState, SafeWeights, Trade


class ExecutionBot:
    def __init__(self, commission_bps: float = 1.0, slippage_bps: float = 5.0) -> None:
        # Cost charged per unit of traded notional, in basis points.
        self._cost_rate = (commission_bps + slippage_bps) / 10_000.0

    @staticmethod
    def _prices(panel: FeaturePanel) -> dict[str, float]:
        prices = {}
        for t, feats in panel.data.items():
            if "close" not in feats:
                raise ValueError(f"panel missing close price for {t}")
            prices[t] = feats["close"]
        return prices

    def run(
        self, safe: SafeWeights, state: PortfolioState, panel: FeaturePanel
    ) -> tuple[tuple[Trade, ...], PortfolioState]:
        prices = self._prices(panel)
        nav = state.nav

        # Target shares per ticker; tickers held but no longer targeted go to 0.
        target_shares: dict[str, float] = {}
        for t, w in safe.weights.items():
            if t not in prices:
                raise ValueError(f"no price for targeted ticker {t}")
            target_shares[t] = (w * nav) / prices[t]
        for t in state.positions:
            target_shares.setdefault(t, 0.0)

        trades: list[Trade] = []
        for t, tgt in target_shares.items():
            qty = tgt - state.positions.get(t, 0.0)
            if abs(qty) < 1e-9:
                continue
            price = prices[t]
            cost = abs(qty * price) * self._cost_rate
            trades.append(Trade(safe.date, t, qty, price, cost))

        # Pass the full price set as marks so untouched holdings are valued at
        # today's prices, not zero.
        new_state = state.apply_trades(safe.date, tuple(trades), marks=prices)
        return tuple(trades), new_state
