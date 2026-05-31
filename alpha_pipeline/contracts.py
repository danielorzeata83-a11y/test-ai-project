"""Immutable message contracts passed between bots.

Every type is frozen, validates itself in __post_init__ (bad data fails at
creation, not later when it would cause damage), and serializes via to_dict()
so the Journal can persist it and replay it exactly.

The data contract for the whole pipeline is a per-(date, ticker) table of
features. At daily frequency `date` is the trading day; the same shape carries
intraday bars later by making `date` a bar timestamp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

# Tolerance for declared-vs-actual exposure checks (floating point slack).
_EXPOSURE_TOL = 1e-6


def _check_finite(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _check_ticker(t: str) -> None:
    if not isinstance(t, str) or not t.strip():
        raise ValueError(f"ticker must be a non-empty string, got {t!r}")


@dataclass(frozen=True)
class FeaturePanel:
    """Output of the Analyst bot: validated per-ticker features for one date.

    data maps ticker -> {feature_name: value}. `valid` is the Analyst's verdict;
    a False panel must stop the cycle before any signal is computed.
    """

    date: str
    tickers: tuple[str, ...]
    data: Mapping[str, Mapping[str, float]]
    valid: bool
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.date, str) or not self.date:
            raise ValueError("date must be a non-empty string")
        if not isinstance(self.tickers, tuple):
            raise ValueError("tickers must be a tuple")
        for t in self.tickers:
            _check_ticker(t)
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError("tickers must be unique")
        if set(self.data.keys()) != set(self.tickers):
            raise ValueError("data keys must match tickers exactly")
        for t, feats in self.data.items():
            for fname, fval in feats.items():
                _check_finite(f"{t}.{fname}", fval)

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "tickers": list(self.tickers),
            "data": {t: dict(f) for t, f in self.data.items()},
            "valid": self.valid,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "FeaturePanel":
        return cls(
            date=d["date"],
            tickers=tuple(d["tickers"]),
            data={t: dict(f) for t, f in d["data"].items()},
            valid=bool(d["valid"]),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass(frozen=True)
class TargetWeights:
    """Output of the Signal bot: desired portfolio weights, no risk applied.

    Convention: long-short, so weights may be negative. Positive = buy/long,
    negative = sell/short.
    """

    date: str
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.date, str) or not self.date:
            raise ValueError("date must be a non-empty string")
        for t, w in self.weights.items():
            _check_ticker(t)
            _check_finite(f"weight[{t}]", w)

    @property
    def gross(self) -> float:
        return sum(abs(w) for w in self.weights.values())

    @property
    def net(self) -> float:
        return sum(self.weights.values())

    def to_dict(self) -> dict:
        return {"date": self.date, "weights": dict(self.weights)}

    @classmethod
    def from_dict(cls, d: Mapping) -> "TargetWeights":
        return cls(date=d["date"], weights=dict(d["weights"]))


@dataclass(frozen=True)
class SafeWeights:
    """Output of the Risk bot: weights after limits, regime overlay, breakers.

    A "contract that doesn't lie": the declared gross_exposure must match the
    actual sum of |weights|. If the Risk bot miscomputes what it did, this fails
    immediately at construction rather than turning into wrong trades downstream.

    halt_request signals a controlled stop (e.g. drawdown breaker): Execution
    closes positions, Orchestrator halts future cycles.
    """

    date: str
    weights: Mapping[str, float]
    gross_exposure: float
    net_exposure: float
    halt_request: bool = False
    risk_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.date, str) or not self.date:
            raise ValueError("date must be a non-empty string")
        for t, w in self.weights.items():
            _check_ticker(t)
            _check_finite(f"weight[{t}]", w)
        _check_finite("gross_exposure", self.gross_exposure)
        _check_finite("net_exposure", self.net_exposure)
        if self.gross_exposure < 0:
            raise ValueError("gross_exposure must be non-negative")
        actual_gross = sum(abs(w) for w in self.weights.values())
        if abs(actual_gross - self.gross_exposure) > _EXPOSURE_TOL:
            raise ValueError(
                f"declared gross_exposure {self.gross_exposure} != actual "
                f"{actual_gross} (contract must not lie)"
            )
        actual_net = sum(self.weights.values())
        if abs(actual_net - self.net_exposure) > _EXPOSURE_TOL:
            raise ValueError(
                f"declared net_exposure {self.net_exposure} != actual {actual_net}"
            )
        if not isinstance(self.risk_actions, tuple):
            raise ValueError("risk_actions must be a tuple")

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "weights": dict(self.weights),
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "halt_request": self.halt_request,
            "risk_actions": list(self.risk_actions),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "SafeWeights":
        return cls(
            date=d["date"],
            weights=dict(d["weights"]),
            gross_exposure=d["gross_exposure"],
            net_exposure=d["net_exposure"],
            halt_request=bool(d.get("halt_request", False)),
            risk_actions=tuple(d.get("risk_actions", ())),
        )


@dataclass(frozen=True)
class Trade:
    """A single simulated fill. quantity is signed: positive buy, negative sell.

    cost bundles commission + slippage in cash terms and must be non-negative.
    """

    date: str
    ticker: str
    quantity: float
    price: float
    cost: float

    def __post_init__(self) -> None:
        if not isinstance(self.date, str) or not self.date:
            raise ValueError("date must be a non-empty string")
        _check_ticker(self.ticker)
        _check_finite("quantity", self.quantity)
        _check_finite("price", self.price)
        _check_finite("cost", self.cost)
        if self.price <= 0:
            raise ValueError("price must be positive")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "ticker": self.ticker,
            "quantity": self.quantity,
            "price": self.price,
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "Trade":
        return cls(
            date=d["date"],
            ticker=d["ticker"],
            quantity=d["quantity"],
            price=d["price"],
            cost=d["cost"],
        )


@dataclass(frozen=True)
class PortfolioState:
    """Immutable portfolio snapshot. Transitions return new instances so any
    historical state can be reconstructed exactly from the Journal (replay).

    nav is the last marked value; peak_nav is the running maximum used by the
    drawdown breaker. drawdown is peak-to-trough.
    """

    date: str
    cash: float
    positions: Mapping[str, float]
    nav: float
    peak_nav: float
    halted: bool = False
    halt_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.date, str) or not self.date:
            raise ValueError("date must be a non-empty string")
        _check_finite("cash", self.cash)
        _check_finite("nav", self.nav)
        _check_finite("peak_nav", self.peak_nav)
        for t, q in self.positions.items():
            _check_ticker(t)
            _check_finite(f"position[{t}]", q)
        if self.peak_nav < self.nav - _EXPOSURE_TOL:
            raise ValueError("peak_nav cannot be below nav")

    @property
    def drawdown(self) -> float:
        if self.peak_nav <= 0:
            return 0.0
        return (self.nav - self.peak_nav) / self.peak_nav

    @classmethod
    def initial(cls, date: str, cash: float) -> "PortfolioState":
        return cls(
            date=date, cash=cash, positions={}, nav=cash, peak_nav=cash
        )

    def apply_trades(
        self, date: str, trades: tuple[Trade, ...],
        marks: Mapping[str, float] | None = None,
    ) -> "PortfolioState":
        """Return a new state with trades applied. NAV is re-marked at the trade
        prices, plus `marks` for any held name that didn't trade this cycle.

        Fail-closed: if a resulting held position has neither a trade price nor a
        mark, raise rather than silently valuing it at zero (which would corrupt
        NAV, drawdown, and replay)."""
        new_cash = self.cash
        new_positions = dict(self.positions)
        prices: dict[str, float] = dict(marks or {})
        for tr in trades:
            new_cash -= tr.quantity * tr.price + tr.cost
            new_positions[tr.ticker] = new_positions.get(tr.ticker, 0.0) + tr.quantity
            prices[tr.ticker] = tr.price  # trade price overrides any stale mark
        new_positions = {t: q for t, q in new_positions.items() if abs(q) > 1e-12}
        missing = [t for t in new_positions if t not in prices]
        if missing:
            raise ValueError(f"no price/mark for held tickers {missing}")
        nav = new_cash + sum(q * prices[t] for t, q in new_positions.items())
        return PortfolioState(
            date=date,
            cash=new_cash,
            positions=new_positions,
            nav=nav,
            peak_nav=max(self.peak_nav, nav),
            halted=self.halted,
            halt_reason=self.halt_reason,
        )

    def mark_to_market(self, date: str, prices: Mapping[str, float]) -> "PortfolioState":
        """Return a new state revalued at the given prices."""
        for t in self.positions:
            if t not in prices:
                raise ValueError(f"missing price for held ticker {t}")
        nav = self.cash + sum(
            q * prices[t] for t, q in self.positions.items()
        )
        return PortfolioState(
            date=date,
            cash=self.cash,
            positions=dict(self.positions),
            nav=nav,
            peak_nav=max(self.peak_nav, nav),
            halted=self.halted,
            halt_reason=self.halt_reason,
        )

    def halt(self, reason: str) -> "PortfolioState":
        """Return a halted copy. Orchestrator refuses to start the next cycle
        while halted is True."""
        return PortfolioState(
            date=self.date,
            cash=self.cash,
            positions=dict(self.positions),
            nav=self.nav,
            peak_nav=self.peak_nav,
            halted=True,
            halt_reason=reason,
        )

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "cash": self.cash,
            "positions": dict(self.positions),
            "nav": self.nav,
            "peak_nav": self.peak_nav,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "PortfolioState":
        return cls(
            date=d["date"],
            cash=d["cash"],
            positions=dict(d["positions"]),
            nav=d["nav"],
            peak_nav=d["peak_nav"],
            halted=bool(d.get("halted", False)),
            halt_reason=d.get("halt_reason", ""),
        )
