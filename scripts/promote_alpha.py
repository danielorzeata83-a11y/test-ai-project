"""Promote an alpha into the registry by running it through the Validation Gate.

Usage:
    python -m scripts.promote_alpha momentum
    python -m scripts.promote_alpha momentum --config config.yaml

An alpha is added to the registry only if it passes all six examiners. This is
the only sanctioned way an alpha becomes live.
"""

from __future__ import annotations

import argparse
import sys

from alpha_pipeline.alphas import ALPHAS
from alpha_pipeline.config import Config
from alpha_pipeline.data import load_synthetic, load_synthetic_drift
from alpha_pipeline.gate import GateConfig, evaluate_alpha
from alpha_pipeline.registry import Registry


def _gate_config(cfg: Config) -> GateConfig:
    g = cfg.section("gate")
    return GateConfig(**{k: g[k] for k in g if k in GateConfig.__dataclass_fields__})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Promote an alpha via the Gate")
    p.add_argument("alpha", help="alpha name from the library")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--registry", default="registry.json")
    p.add_argument("--days", type=int, default=400)
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args(argv)

    if args.alpha not in ALPHAS:
        print(f"unknown alpha {args.alpha!r}; available: {sorted(ALPHAS)}")
        return 2

    cfg = Config.load(args.config)
    universe = tuple(cfg.section("universe")["names"])
    source = cfg.section("universe").get("source", "synthetic")
    loader = load_synthetic_drift if source == "synthetic_drift" else load_synthetic
    prices = loader(universe, n_days=args.days, seed=args.seed)

    registry = Registry(args.registry)
    result = evaluate_alpha(
        prices, ALPHAS[args.alpha], args.alpha, registry, _gate_config(cfg)
    )

    print(f"Gate result for {args.alpha!r}: {'PASS' if result.passed else 'FAIL'}")
    for examiner, ok in result.checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {examiner}")
    print("  metrics:", {k: round(v, 4) for k, v in result.metrics.items()})

    if result.passed:
        registry.promote(args.alpha, {k: round(v, 6) for k, v in result.metrics.items()})
        print(f"-> promoted {args.alpha!r}. Registry now: {registry.active_names()}")
        return 0
    print("-> not promoted.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
