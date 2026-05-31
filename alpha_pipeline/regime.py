"""Regime detection: a 2-state Gaussian HMM on market returns, stdlib only.

The HMM doesn't predict price direction (hard); it estimates the volatility
*state* (calm vs crisis), which is persistent and easier to estimate. The Risk
bot uses it as an overlay: cut exposure when the crisis-state probability is
high. It protects alpha, it doesn't generate it.

Anti-label-switching: after fitting, states are sorted by variance so state 1
is always the high-vol/crisis state. Without this the "crisis" label can flip
between fits — the exact instability flagged in the Markov research.

Pure EM (Baum-Welch) with diagonal Gaussian emissions. Two states only, which
keeps it stable; more states overfit financial data badly.
"""

from __future__ import annotations

import math
import statistics

_SQRT_2PI = math.sqrt(2 * math.pi)


def _gauss_pdf(x: float, mu: float, var: float) -> float:
    var = max(var, 1e-12)
    return math.exp(-((x - mu) ** 2) / (2 * var)) / (_SQRT_2PI * math.sqrt(var))


class RegimeHMM:
    """2-state Gaussian HMM. State 0 = low vol, state 1 = high vol (crisis)."""

    def __init__(self, n_iter: int = 50, seed: int = 0) -> None:
        self.n_iter = n_iter
        self.seed = seed
        self.mu = [0.0, 0.0]
        self.var = [1.0, 1.0]
        self.trans = [[0.95, 0.05], [0.10, 0.90]]
        self.pi = [0.5, 0.5]
        self._fitted = False

    def fit(self, returns: list[float]) -> "RegimeHMM":
        x = list(returns)
        n = len(x)
        if n < 10:
            raise ValueError("need at least 10 observations to fit")

        # Init: split by median absolute return into calm/volatile clusters.
        med = statistics.median(abs(v) for v in x)
        calm = [v for v in x if abs(v) <= med]
        wild = [v for v in x if abs(v) > med] or calm
        self.mu = [statistics.fmean(calm), statistics.fmean(wild)]
        self.var = [max(statistics.pvariance(calm), 1e-8),
                    max(statistics.pvariance(wild), 1e-8)]

        for _ in range(self.n_iter):
            gamma, xi, ll = self._e_step(x)
            self._m_step(x, gamma, xi)
            if not math.isfinite(ll):
                break

        self._sort_states()
        self._fitted = True
        return self

    def _e_step(self, x):
        n = len(x)
        B = [[_gauss_pdf(x[t], self.mu[s], self.var[s]) for s in (0, 1)]
             for t in range(n)]
        # Forward (scaled).
        alpha = [[0.0, 0.0] for _ in range(n)]
        scale = [0.0] * n
        for s in (0, 1):
            alpha[0][s] = self.pi[s] * B[0][s]
        scale[0] = sum(alpha[0]) or 1e-300
        alpha[0] = [a / scale[0] for a in alpha[0]]
        for t in range(1, n):
            for s in (0, 1):
                alpha[t][s] = B[t][s] * sum(
                    alpha[t - 1][p] * self.trans[p][s] for p in (0, 1)
                )
            scale[t] = sum(alpha[t]) or 1e-300
            alpha[t] = [a / scale[t] for a in alpha[t]]
        # Backward (scaled).
        beta = [[0.0, 0.0] for _ in range(n)]
        beta[-1] = [1.0, 1.0]
        for t in range(n - 2, -1, -1):
            for s in (0, 1):
                beta[t][s] = sum(
                    self.trans[s][q] * B[t + 1][q] * beta[t + 1][q] for q in (0, 1)
                ) / scale[t + 1]
        # Gamma and xi.
        gamma = [[alpha[t][s] * beta[t][s] for s in (0, 1)] for t in range(n)]
        for t in range(n):
            z = sum(gamma[t]) or 1e-300
            gamma[t] = [g / z for g in gamma[t]]
        xi = [[[0.0, 0.0], [0.0, 0.0]] for _ in range(n - 1)]
        for t in range(n - 1):
            z = 0.0
            for s in (0, 1):
                for q in (0, 1):
                    xi[t][s][q] = (alpha[t][s] * self.trans[s][q]
                                   * B[t + 1][q] * beta[t + 1][q])
                    z += xi[t][s][q]
            z = z or 1e-300
            for s in (0, 1):
                for q in (0, 1):
                    xi[t][s][q] /= z
        ll = sum(math.log(c) for c in scale)
        return gamma, xi, ll

    def _m_step(self, x, gamma, xi):
        n = len(x)
        self.pi = list(gamma[0])
        for s in (0, 1):
            denom = sum(gamma[t][s] for t in range(n - 1)) or 1e-300
            for q in (0, 1):
                self.trans[s][q] = sum(xi[t][s][q] for t in range(n - 1)) / denom
            gsum = sum(gamma[t][s] for t in range(n)) or 1e-300
            self.mu[s] = sum(gamma[t][s] * x[t] for t in range(n)) / gsum
            self.var[s] = max(
                sum(gamma[t][s] * (x[t] - self.mu[s]) ** 2 for t in range(n)) / gsum,
                1e-10,
            )

    def _sort_states(self):
        """Force state 1 to be the high-variance (crisis) state."""
        if self.var[0] > self.var[1]:
            self.mu.reverse()
            self.var.reverse()
            self.pi.reverse()
            self.trans = [[self.trans[1][1], self.trans[1][0]],
                          [self.trans[0][1], self.trans[0][0]]]

    def crisis_probabilities(self, returns: list[float]) -> list[float]:
        """Smoothed P(state==crisis) per observation."""
        if not self._fitted:
            raise RuntimeError("fit() first")
        gamma, _, _ = self._e_step(list(returns))
        return [g[1] for g in gamma]

    def expected_durations(self) -> tuple[float, float]:
        """Expected dwell time in (calm, crisis) from the transition matrix."""
        d0 = 1.0 / (1.0 - self.trans[0][0]) if self.trans[0][0] < 1 else float("inf")
        d1 = 1.0 / (1.0 - self.trans[1][1]) if self.trans[1][1] < 1 else float("inf")
        return d0, d1
