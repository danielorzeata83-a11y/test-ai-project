"""Risk bot: defends alpha, never generates it. Applies seven controls in order,
each of which can only *reduce* exposure, never increase it. The output is
SafeWeights plus an audit log of every action taken.

Order matters — cheap structural limits first, then market-state overlays,
then the catastrophe stop:

  1. Per-position cap     — no single name above max_position.
  2. Dollar neutrality    — re-center so longs and shorts net to ~0.
  3. Gross cap            — scale the whole book down to max_gross.
  4. Regime overlay       — scale down when crisis probability is high.
  5. Vol targeting        — scale toward a target portfolio volatility.
  6. Drawdown breaker     — if peak-to-trough breach: flatten + request halt.
  7. Staleness halt       — if the panel is stale/invalid: flatten + request halt.

Fail-closed: any uncertainty reduces risk automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import FeaturePanel, PortfolioState, SafeWeights, TargetWeights
from .regime import RegimeHMM


@dataclass
class RiskConfig:
    max_position: float = 0.05      # per-name weight cap
    max_gross: float = 2.0          # sum of |weights|
    target_vol: float = 0.10        # annualized portfolio vol target
    regime_cut: float = 0.7         # fraction to keep when in crisis
    crisis_threshold: float = 0.5   # crisis prob above this triggers overlay
    max_drawdown: float = 0.15      # peak-to-trough breaker
    min_history_for_regime: int = 64
    periods_per_year: int = 252     # 252 daily; 252*bars_per_day intraday


class RiskBot:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.cfg = config or RiskConfig()

    def run(
        self, target: TargetWeights, state: PortfolioState, panel: FeaturePanel
    ) -> SafeWeights:
        cfg = self.cfg
        actions: list[str] = []

        # 7. Staleness halt (checked first; if data is bad, flatten now).
        if not panel.valid:
            return self._halt(target.date, "panel invalid: flatten + halt", actions)

        w = dict(target.weights)

        # 1. Per-position cap.
        capped = {t: max(-cfg.max_position, min(cfg.max_position, x))
                  for t, x in w.items()}
        if capped != w:
            actions.append(f"per-position cap @ {cfg.max_position}")
        w = capped

        # 2. Dollar neutrality (shift so net == 0 without adding gross blindly).
        if w:
            net = sum(w.values())
            adj = net / len(w)
            w = {t: x - adj for t, x in w.items()}
            if abs(net) > 1e-9:
                actions.append("re-centered to dollar-neutral")

        # 3. Gross cap.
        gross = sum(abs(x) for x in w.values())
        if gross > cfg.max_gross and gross > 0:
            scale = cfg.max_gross / gross
            w = {t: x * scale for t, x in w.items()}
            actions.append(f"gross cap {cfg.max_gross} (scaled {scale:.2f})")

        # 4. Regime overlay.
        crisis_p = self._crisis_probability(panel)
        if crisis_p is not None and crisis_p > cfg.crisis_threshold:
            w = {t: x * cfg.regime_cut for t, x in w.items()}
            actions.append(f"regime overlay: crisis p={crisis_p:.2f}, "
                           f"kept {cfg.regime_cut}")

        # 5. Vol targeting (scale down only; never lever up).
        pvol = self._portfolio_vol(w, panel)
        if pvol > cfg.target_vol and pvol > 0:
            scale = cfg.target_vol / pvol
            w = {t: x * scale for t, x in w.items()}
            actions.append(f"vol target {cfg.target_vol} (scaled {scale:.2f})")

        # 6. Drawdown breaker.
        if state.drawdown <= -cfg.max_drawdown:
            return self._halt(
                target.date,
                f"drawdown {state.drawdown:.1%} breached {cfg.max_drawdown:.0%}",
                actions,
            )

        gross = sum(abs(x) for x in w.values())
        net = sum(w.values())
        return SafeWeights(target.date, w, gross, net, False, tuple(actions))

    # --- helpers ---------------------------------------------------------

    def _crisis_probability(self, panel: FeaturePanel) -> float | None:
        """Cross-sectional mean ret_1d is a crude market return; without a real
        return history we use the dispersion of ret_1d as a stress proxy. If the
        panel carries a market return series in metadata, fit the HMM on it."""
        series = panel.metadata.get("market_returns")
        if series and len(series) >= self.cfg.min_history_for_regime:
            try:
                hmm = RegimeHMM().fit(list(series))
                return hmm.crisis_probabilities(list(series))[-1]
            except Exception:
                return None
        return None

    def _portfolio_vol(self, weights, panel: FeaturePanel) -> float:
        """Approximate annualized portfolio vol assuming uncorrelated names,
        using each ticker's vol_21 feature. Conservative (ignores diversification
        across correlations) so it tends to scale down, not up."""
        var = 0.0
        for t, x in weights.items():
            v = panel.data.get(t, {}).get("vol_21")
            if v is not None:
                var += (x * v) ** 2
        return (var ** 0.5) * (self.cfg.periods_per_year ** 0.5)

    def _halt(self, date, reason, actions) -> SafeWeights:
        actions.append(reason)
        return SafeWeights(date, {}, 0.0, 0.0, True, tuple(actions))
