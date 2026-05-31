"""Alpha Registry: which alphas are in production, persisted to JSON.

Separates the alpha *library* (code in alphas.py) from what is actually live.
Survives restarts. The Signal bot reads only `promoted` names; the Gate is the
only thing that may add to it.
"""

from __future__ import annotations

import json
import os


class Registry:
    def __init__(self, path: str) -> None:
        self.path = path
        self.promoted: dict[str, dict] = {}
        if os.path.exists(path):
            with open(path) as f:
                self.promoted = json.load(f)

    def is_promoted(self, name: str) -> bool:
        return name in self.promoted

    def active_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.promoted))

    def promote(self, name: str, report: dict) -> None:
        self.promoted[name] = report
        self._save()

    def retire(self, name: str) -> None:
        self.promoted.pop(name, None)
        self._save()

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self.promoted, f, indent=2, sort_keys=True)
