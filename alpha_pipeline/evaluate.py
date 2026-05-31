"""Evaluation methodology — the layer that stops you being fooled by a pretty
backtest. Pure stdlib (math + statistics), no numpy.

Works on a "history": a list of daily cross-sections, each holding the alpha
scores and the realized forward returns for that date:

    history = [{"date": d, "scores": {tkr: s}, "fwd": {tkr: r}}, ...]

build_history() constructs this from PriceData + an alpha function, which keeps
the live path and the research path using the exact same alpha code.
"""

from __future__ import annotations

import math
import statistics
from typing import Callable, Mapping

from .alphas import AlphaFn

Cross = dict  # {"date", "scores": {t: s}, "fwd": {t: r}}


# --- building the history --------------------------------------------------

def _features_at(rows) -> dict[str, float] | None:
    """Replicates AnalystBot._features for a price history (ascending)."""
    closes = [r[1] for r in rows]
    if len(closes) < 64:
        return None
    last, prev = closes[-1], closes[-2]
    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(-21, 0)]
    mean = sum(rets) / len(rets)
    vol = math.sqrt(sum((x - mean) ** 2 for x in rets) / len(rets))
    feats = {
        "close": last,
        "ret_1d": last / prev - 1.0,
        "mom_21": last / closes[-22] - 1.0,
        "mom_63": last / closes[-64] - 1.0,
        "vol_21": vol,
    }
    return feats if all(math.isfinite(v) for v in feats.values()) else None


def build_history(prices: Mapping[str, list], alpha: AlphaFn, horizon: int = 1) -> list[Cross]:
    """Build daily cross-sections of (scores, forward returns) for an alpha."""
    tickers = list(prices)
    # Align on a common ascending date axis (assume loaders sort by date).
    n = min(len(prices[t]) for t in tickers)
    dates = [prices[tickers[0]][i][0] for i in range(n)]

    history: list[Cross] = []
    for i in range(64, n - horizon):
        feats = {}
        for t in tickers:
            f = _features_at(prices[t][: i + 1])
            if f is not None:
                feats[t] = f
        if len(feats) < 2:
            continue
        scores = alpha(feats)
        fwd = {}
        for t in feats:
            c_now = prices[t][i][1]
            c_fwd = prices[t][i + horizon][1]
            fwd[t] = c_fwd / c_now - 1.0
        history.append({"date": dates[i], "scores": scores, "fwd": fwd})
    return history


# --- correlations ----------------------------------------------------------

def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def spearman(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = [k for k in a if k in b]
    if len(keys) < 2:
        return 0.0
    return _pearson(_rank([a[k] for k in keys]), _rank([b[k] for k in keys]))


# --- metrics ---------------------------------------------------------------

def information_coefficient(history: list[Cross]) -> dict:
    """Mean rank-IC, its IR, and t-stat across all dates."""
    ics = [spearman(c["scores"], c["fwd"]) for c in history]
    ics = [x for x in ics if math.isfinite(x)]
    if len(ics) < 2:
        return {"ic": 0.0, "ic_ir": 0.0, "t_stat": 0.0, "n": len(ics)}
    mean = statistics.fmean(ics)
    sd = statistics.pstdev(ics)
    ic_ir = mean / sd if sd > 0 else 0.0
    t_stat = ic_ir * math.sqrt(len(ics))
    return {"ic": mean, "ic_ir": ic_ir, "t_stat": t_stat, "n": len(ics)}


def ic_decay(prices, alpha: AlphaFn, horizons=(1, 2, 5, 10)) -> dict:
    """IC at increasing horizons. A signal that dies fast is a cost trap."""
    return {h: information_coefficient(build_history(prices, alpha, h))["ic"]
            for h in horizons}


def long_short_returns(history: list[Cross], top_frac: float = 0.2) -> list[float]:
    """Daily return of an equal-weight long-short on score quintiles (gross)."""
    daily = []
    for c in history:
        ranked = sorted(c["scores"], key=lambda t: c["scores"][t])
        n = max(1, int(len(ranked) * top_frac))
        shorts, longs = ranked[:n], ranked[-n:]
        long_r = sum(c["fwd"][t] for t in longs) / len(longs)
        short_r = sum(c["fwd"][t] for t in shorts) / len(shorts)
        daily.append(long_r - short_r)
    return daily


def turnover(history: list[Cross], top_frac: float = 0.2) -> float:
    """Average fraction of the long book that changes each day."""
    prev: set | None = None
    changes = []
    for c in history:
        ranked = sorted(c["scores"], key=lambda t: c["scores"][t])
        n = max(1, int(len(ranked) * top_frac))
        longs = set(ranked[-n:])
        if prev is not None:
            changes.append(len(longs - prev) / n)
        prev = longs
    return statistics.fmean(changes) if changes else 0.0


def sharpe(daily: list[float], cost_per_turnover: float = 0.0, turn: float = 0.0,
           periods: int = 252) -> float:
    """Annualized Sharpe, optionally net of per-rebalance cost."""
    if len(daily) < 2:
        return 0.0
    net = [r - cost_per_turnover * turn for r in daily]
    mean, sd = statistics.fmean(net), statistics.pstdev(net)
    return (mean / sd) * math.sqrt(periods) if sd > 0 else 0.0


def deflated_sharpe(daily: list[float], observed_sharpe: float, n_trials: int) -> float:
    """Probability the true Sharpe > 0 after correcting for multiple testing
    and non-normal returns. Bailey & Lopez de Prado, simplified."""
    n = len(daily)
    if n < 4 or n_trials < 1:
        return 0.0
    sk = _skew(daily)
    ku = _kurt(daily)
    sr = observed_sharpe / math.sqrt(252)  # per-period

    # Expected max of n_trials standard-normal draws (benchmark to beat).
    emc = 0.5772156649
    z = lambda p: _norm_ppf(p)
    sr0_var = 1.0  # variance of trial Sharpes assumed ~1 in std units
    expected_max = math.sqrt(sr0_var) * (
        (1 - emc) * z(1 - 1.0 / n_trials)
        + emc * z(1 - 1.0 / (n_trials * math.e))
    )
    denom = math.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4.0 * sr * sr))
    stat = (sr - expected_max / math.sqrt(n)) * math.sqrt(n - 1) / denom
    return _norm_cdf(stat)


def cpcv_positive_fraction(history: list[Cross], n_groups: int = 6,
                           test_groups: int = 2) -> float:
    """Combinatorial purged CV: fraction of group-combinations whose held-out
    long-short Sharpe is positive. Robustness across time, not one lucky split."""
    from itertools import combinations
    n = len(history)
    if n < n_groups:
        return 0.0
    bounds = [round(i * n / n_groups) for i in range(n_groups + 1)]
    groups = [history[bounds[i]:bounds[i + 1]] for i in range(n_groups)]
    positive = 0
    total = 0
    for combo in combinations(range(n_groups), test_groups):
        test = [c for g in combo for c in groups[g]]
        s = sharpe(long_short_returns(test))
        positive += 1 if s > 0 else 0
        total += 1
    return positive / total if total else 0.0


# --- distribution helpers (stdlib only) ------------------------------------

def _skew(xs):
    n = len(xs); m = statistics.fmean(xs); sd = statistics.pstdev(xs)
    return sum(((x - m) / sd) ** 3 for x in xs) / n if sd > 0 else 0.0


def _kurt(xs):
    n = len(xs); m = statistics.fmean(xs); sd = statistics.pstdev(xs)
    return sum(((x - m) / sd) ** 4 for x in xs) / n if sd > 0 else 3.0


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_ppf(p):
    """Inverse normal CDF via Beasley-Springer-Moro approximation."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
