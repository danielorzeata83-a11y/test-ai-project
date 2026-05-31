"""Minimal YAML loader (stdlib only) for config.yaml.

Supports the subset this project uses: nested mappings by indentation, scalars
(int/float/bool/str), and inline flow lists [a, b, c]. No anchors, no multi-line
strings — deliberately tiny so the system has no third-party dependency.
"""

from __future__ import annotations

from dataclasses import dataclass


def _scalar(s: str):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        return [_scalar(x) for x in inner.split(",")] if inner else []
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s.strip("'\"")


def load_yaml(path: str) -> dict:
    """Parse the supported YAML subset into nested dicts."""
    root: dict = {}
    stack = [(-1, root)]  # (indent, container)
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            key, _, val = line.strip().partition(":")
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if val.strip() == "":
                child: dict = {}
                parent[key.strip()] = child
                stack.append((indent, child))
            else:
                parent[key.strip()] = _scalar(val)
    return root


@dataclass
class Config:
    raw: dict

    @classmethod
    def load(cls, path: str = "config.yaml") -> "Config":
        return cls(load_yaml(path))

    def section(self, name: str) -> dict:
        return self.raw.get(name, {})
