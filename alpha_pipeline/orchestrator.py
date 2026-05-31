"""Orchestrator: drives one daily cycle, fail-closed.

Linear chain, one direction: Analyst -> Signal -> Risk -> Execution. Bots are
defined by Protocols (duck-typed .run() signatures), not inheritance, so any
object with the right method plugs in — including test mocks.

Fail-closed rules:
  - Invalid panel  -> stop the chain, cycle still "success" (ran, did nothing).
  - Bot exception  -> cycle "failed", state halted, CRITICAL event logged.
  - Halt requested -> state halted after this cycle (drawdown breaker etc.).
  - Already halted  -> skip the cycle entirely.
  - Cycle already logged -> skip (idempotent; a double-fired cron is harmless).
"""

from __future__ import annotations

from typing import Protocol

from .contracts import (
    FeaturePanel,
    PortfolioState,
    SafeWeights,
    TargetWeights,
    Trade,
)
from .journal import CycleAlreadyLogged, Journal


class AnalystProtocol(Protocol):
    def run(self, date: str) -> FeaturePanel: ...


class SignalProtocol(Protocol):
    def run(self, panel: FeaturePanel) -> TargetWeights: ...


class RiskProtocol(Protocol):
    def run(
        self, target: TargetWeights, state: PortfolioState, panel: FeaturePanel
    ) -> SafeWeights: ...


class ExecutionProtocol(Protocol):
    def run(
        self, safe: SafeWeights, state: PortfolioState, panel: FeaturePanel
    ) -> tuple[tuple[Trade, ...], PortfolioState]: ...


class Orchestrator:
    def __init__(
        self,
        journal: Journal,
        analyst: AnalystProtocol,
        signal: SignalProtocol,
        risk: RiskProtocol,
        execution: ExecutionProtocol,
    ) -> None:
        self.journal = journal
        self.analyst = analyst
        self.signal = signal
        self.risk = risk
        self.execution = execution

    def run_cycle(
        self, date: str, state: PortfolioState
    ) -> tuple[PortfolioState, tuple[Trade, ...]]:
        # Idempotency: a date already run is a no-op.
        if self.journal.has_cycle(date):
            self.journal.log_event("INFO", "cycle already run, skipping", date)
            return state, ()

        # Don't trade while halted; a human must reset first.
        if state.halted:
            self.journal.log_event(
                "WARNING", f"system halted ({state.halt_reason}), skipping", date
            )
            return state, ()

        try:
            with self.journal.cycle(date):
                panel = self.analyst.run(date)
                self.journal.log_panel(panel)
                if not panel.valid:
                    self.journal.log_event(
                        "WARNING", "panel invalid, no trading", date
                    )
                    return state, ()

                target = self.signal.run(panel)
                self.journal.log_target(target)

                safe = self.risk.run(target, state, panel)
                self.journal.log_safe(safe)

                trades, new_state = self.execution.run(safe, state, panel)
                self.journal.log_trades(trades)

                if safe.halt_request:
                    new_state = new_state.halt(
                        "risk halt_request (e.g. drawdown breaker)"
                    )
                    self.journal.log_event(
                        "CRITICAL", "halt requested by risk bot", date
                    )

                self.journal.log_state(new_state)
                return new_state, trades

        except CycleAlreadyLogged:
            raise
        except Exception as exc:  # any bot bug -> halt hard, demand attention
            halted = state.halt(f"exception in cycle {date}: {exc!r}")
            self.journal.log_state(halted)
            self.journal.log_event("CRITICAL", f"cycle failed: {exc!r}", date)
            return halted, ()
