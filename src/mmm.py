"""
Marketing Mix Model (MMM) built from first principles.

    sales_t = baseline_t + sum_c  beta_c * Hill( Adstock(spend_c,t ; decay_c) ; k_c, s_c ) + noise

* Adstock (carry-over): a share `decay` of last week's advertising effect carries into this week.
* Hill saturation (diminishing returns): each extra dollar buys a little less than the last.
* Baseline: intercept + trend + annual seasonality + price, promo and holiday effects.

Fitting uses *variable projection*: an outer global optimiser (differential evolution) searches the
non-linear media parameters (decay, k, s); for each candidate, the linear coefficients are solved
exactly with non-negative least squares, so every channel's effect is forced to be >= 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, nnls
from scipy.signal import lfilter

CHANNELS = ["tv", "search", "social", "display", "email"]
CONTROLS = ["trend", "sin52", "cos52", "price_index", "promo", "holiday"]


def adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """Geometric adstock: y_t = x_t + decay * y_{t-1}."""
    return lfilter([1.0], [1.0, -decay], x)


def hill(x: np.ndarray, k: float, s: float) -> np.ndarray:
    """Hill saturation curve, 0 -> 1. k = half-saturation point, s = shape."""
    x = np.maximum(x, 0.0)
    return x**s / (x**s + k**s + 1e-12)


def add_controls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t = np.arange(len(df))
    df["trend"] = t / 52.0
    df["sin52"] = np.sin(2 * np.pi * t / 52)
    df["cos52"] = np.cos(2 * np.pi * t / 52)
    return df


@dataclass
class MMM:
    channels: list[str] = field(default_factory=lambda: list(CHANNELS))
    controls: list[str] = field(default_factory=lambda: list(CONTROLS))
    seed: int = 7
    params_: dict = field(default_factory=dict, init=False)
    coef_: dict = field(default_factory=dict, init=False)
    scale_: dict = field(default_factory=dict, init=False)

    # ---- design matrix --------------------------------------------------
    def _media_features(self, df: pd.DataFrame, theta: np.ndarray) -> np.ndarray:
        cols = []
        for i, c in enumerate(self.channels):
            decay, k_rel, s = theta[3 * i : 3 * i + 3]
            x = adstock(df[f"{c}_spend"].to_numpy(float), decay)
            cols.append(hill(x, k_rel * self.scale_[c], s))
        return np.column_stack(cols)

    def _design(self, df: pd.DataFrame, theta: np.ndarray) -> np.ndarray:
        media = self._media_features(df, theta)
        ctrl = df[self.controls].to_numpy(float)
        ones = np.ones((len(df), 1))
        # controls may be +/-: split each into a positive and a negative column for NNLS
        return np.hstack([media, ones, ctrl, -ctrl])

    def _solve(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        w, _ = nnls(X / self._ys, y / self._ys, maxiter=2000)
        resid = y - X @ w
        return w, float(np.mean(resid**2))

    # ---- fitting --------------------------------------------------------
    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "MMM":
        self.scale_ = {c: float(df[f"{c}_spend"].mean()) for c in self.channels}
        self._ys = float(np.std(y))
        bounds = []
        for _ in self.channels:
            # k >= 0.6x mean spend stops a channel from saturating into a flat line that mimics the intercept
            bounds += [(0.0, 0.9), (0.6, 4.0), (0.8, 3.0)]  # decay, k (x mean spend), shape

        def loss(theta):
            return self._solve(self._design(df, theta), y)[1]

        res = differential_evolution(
            loss, bounds, seed=self.seed, maxiter=250, popsize=20, tol=1e-8,
            polish=True, updating="deferred", workers=1,
        )
        theta = res.x
        w, _ = self._solve(self._design(df, theta), y)

        n = len(self.channels)
        m = len(self.controls)
        self.theta_ = theta
        self.params_ = {
            c: {"decay": theta[3 * i], "k": theta[3 * i + 1] * self.scale_[c], "shape": theta[3 * i + 2]}
            for i, c in enumerate(self.channels)
        }
        self.coef_ = {c: w[i] for i, c in enumerate(self.channels)}
        self.coef_["intercept"] = w[n]
        for j, ctl in enumerate(self.controls):
            self.coef_[ctl] = w[n + 1 + j] - w[n + 1 + m + j]
        self.w_ = w
        return self

    # ---- inference ------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._design(df, self.theta_) @ self.w_

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        media = self._media_features(df, self.theta_)
        out = pd.DataFrame({c: media[:, i] * self.coef_[c] for i, c in enumerate(self.channels)})
        base = self.coef_["intercept"] + sum(df[ctl].to_numpy(float) * self.coef_[ctl] for ctl in self.controls)
        out.insert(0, "baseline", base)
        return out

    def response(self, channel: str, weekly_spend: np.ndarray) -> np.ndarray:
        """Steady-state weekly sales from a constant weekly spend (adstock settles at x / (1 - decay))."""
        p = self.params_[channel]
        return self.coef_[channel] * hill(np.asarray(weekly_spend) / (1 - p["decay"]), p["k"], p["shape"])

    def marginal_roi(self, channel: str, weekly_spend: float, eps: float = 100.0) -> float:
        return float((self.response(channel, weekly_spend + eps) - self.response(channel, weekly_spend)) / eps)
