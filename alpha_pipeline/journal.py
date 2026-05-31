"""Persistent, append-only journal — the system's audit trail and replay log.

SQLite in WAL mode. One row per artifact per cycle (date). The cycle() context
manager makes a date atomic and idempotent: a second attempt to run the same
date raises CycleAlreadyLogged, so a double-fired cron never duplicates trades.

Everything serializes through the contracts' to_dict()/from_dict(), so any past
PortfolioState can be recovered bit-for-bit.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Iterator

from .contracts import (
    FeaturePanel,
    PortfolioState,
    SafeWeights,
    TargetWeights,
    Trade,
)


class CycleAlreadyLogged(Exception):
    """Raised when a cycle for a date that already exists is started again."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    date TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS panels (
    date TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS targets (
    date TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS safe_weights (
    date TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS portfolio_snaps (
    date TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    ts REAL NOT NULL
);
"""


class Journal:
    def __init__(self, path: str) -> None:
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- cycle lifecycle -------------------------------------------------

    def has_cycle(self, date: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM cycles WHERE date = ?", (date,)
        ).fetchone()
        return row is not None

    @contextmanager
    def cycle(self, date: str) -> Iterator["Journal"]:
        """Mark a cycle atomically. Raises CycleAlreadyLogged if the date was
        already run. Marks 'success' on clean exit, 'failed' on exception."""
        if self.has_cycle(date):
            raise CycleAlreadyLogged(date)
        self._conn.execute(
            "INSERT INTO cycles (date, status, ts) VALUES (?, ?, ?)",
            (date, "running", time.time()),
        )
        self._conn.commit()
        try:
            yield self
        except Exception:
            self._conn.execute(
                "UPDATE cycles SET status = ? WHERE date = ?", ("failed", date)
            )
            self._conn.commit()
            raise
        else:
            self._conn.execute(
                "UPDATE cycles SET status = ? WHERE date = ?", ("success", date)
            )
            self._conn.commit()

    def cycle_status(self, date: str) -> str | None:
        row = self._conn.execute(
            "SELECT status FROM cycles WHERE date = ?", (date,)
        ).fetchone()
        return row["status"] if row else None

    # --- artifact logging ------------------------------------------------

    def log_panel(self, panel: FeaturePanel) -> None:
        self._conn.execute(
            "INSERT INTO panels (date, payload) VALUES (?, ?)",
            (panel.date, json.dumps(panel.to_dict())),
        )
        self._conn.commit()

    def log_target(self, target: TargetWeights) -> None:
        self._conn.execute(
            "INSERT INTO targets (date, payload) VALUES (?, ?)",
            (target.date, json.dumps(target.to_dict())),
        )
        self._conn.commit()

    def log_safe(self, safe: SafeWeights) -> None:
        self._conn.execute(
            "INSERT INTO safe_weights (date, payload) VALUES (?, ?)",
            (safe.date, json.dumps(safe.to_dict())),
        )
        self._conn.commit()

    def log_trades(self, trades: tuple[Trade, ...]) -> None:
        self._conn.executemany(
            "INSERT INTO trades (date, payload) VALUES (?, ?)",
            [(tr.date, json.dumps(tr.to_dict())) for tr in trades],
        )
        self._conn.commit()

    def log_state(self, state: PortfolioState) -> None:
        self._conn.execute(
            "INSERT INTO portfolio_snaps (date, payload) VALUES (?, ?)",
            (state.date, json.dumps(state.to_dict())),
        )
        self._conn.commit()

    def log_event(self, level: str, message: str, date: str | None = None) -> None:
        self._conn.execute(
            "INSERT INTO events (date, level, message, ts) VALUES (?, ?, ?, ?)",
            (date, level, message, time.time()),
        )
        self._conn.commit()

    # --- replay / recovery ----------------------------------------------

    def recover_state(self, date: str) -> PortfolioState | None:
        row = self._conn.execute(
            "SELECT payload FROM portfolio_snaps WHERE date = ?", (date,)
        ).fetchone()
        if row is None:
            return None
        return PortfolioState.from_dict(json.loads(row["payload"]))

    def latest_state(self) -> PortfolioState | None:
        row = self._conn.execute(
            "SELECT payload FROM portfolio_snaps ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return PortfolioState.from_dict(json.loads(row["payload"]))

    def read_trades(self, date: str) -> tuple[Trade, ...]:
        rows = self._conn.execute(
            "SELECT payload FROM trades WHERE date = ? ORDER BY id", (date,)
        ).fetchall()
        return tuple(Trade.from_dict(json.loads(r["payload"])) for r in rows)
