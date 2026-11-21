"""Numerically safe financial ratio and time-series helpers.

Every function in this module is deliberately small, pure and dependency-light so that it can be
unit-tested and reused outside the feature extractor. All functions accept plain Python sequences
as well as NumPy arrays and pandas Series, return finite floats, and degrade to a documented
default instead of raising on empty or degenerate input.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

EPS = 1e-12


def _as_float_array(values: Sequence[float] | np.ndarray | Any) -> np.ndarray:
    """Convert input to a 1-D float array with non-finite values removed."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers, returning ``default`` when the denominator is zero or non-finite.

    Examples:
        >>> safe_div(10, 4)
        2.5
        >>> safe_div(10, 0)
        0.0
        >>> safe_div(10, 0, default=float("nan"))
        nan
    """
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(den) or abs(den) < EPS or not math.isfinite(num):
        return default
    result = num / den
    return result if math.isfinite(result) else default


def coefficient_of_variation(values: Sequence[float] | np.ndarray, default: float = 0.0) -> float:
    """Return the coefficient of variation (population standard deviation / |mean|).

    A value of 0 means perfectly stable; values above 1 indicate high dispersion relative to the
    mean. Returns ``default`` for empty input or a mean close to zero.
    """
    arr = _as_float_array(values)
    if arr.size < 2:
        return default
    mean = float(np.mean(arr))
    if abs(mean) < EPS:
        return default
    cv = float(np.std(arr, ddof=0) / abs(mean))
    return cv if math.isfinite(cv) else default


def hhi(values: Sequence[float] | np.ndarray, default: float = 0.0) -> float:
    """Herfindahl-Hirschman index of concentration, on a 0 to 1 scale.

    Input values are treated as non-negative weights (for example revenue by customer). The result
    is 1 for a single counterparty and approaches 0 for many equally sized counterparties.
    Negative weights are clipped to zero.
    """
    arr = _as_float_array(values)
    if arr.size == 0:
        return default
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total < EPS:
        return default
    shares = arr / total
    return float(np.sum(shares**2))


def normalized_entropy(values: Sequence[float] | np.ndarray, default: float = 0.0) -> float:
    """Shannon entropy normalised by ``log(n)``, on a 0 to 1 scale.

    A value of 1 means perfectly even distribution across all observed categories; 0 means all
    mass sits in one category. Returns ``default`` for fewer than two non-zero weights.
    """
    arr = _as_float_array(values)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if arr.size < 2 or total < EPS:
        return default
    shares = arr[arr > 0] / total
    entropy = float(-np.sum(shares * np.log(shares)))
    return float(entropy / math.log(arr.size))


def gini(values: Sequence[float] | np.ndarray, default: float = 0.0) -> float:
    """Gini coefficient of inequality for non-negative values, on a 0 to 1 scale.

    0 means perfect equality, values near 1 mean extreme concentration. This implementation uses
    the mean absolute difference formula and is O(n log n).
    """
    arr = _as_float_array(values)
    arr = np.clip(arr, 0.0, None)
    if arr.size < 2 or float(arr.sum()) < EPS:
        return default
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1, dtype=float)
    g = float((2.0 * np.sum(index * arr)) / (n * np.sum(arr)) - (n + 1.0) / n)
    return float(np.clip(g, 0.0, 1.0))


def linear_slope(values: Sequence[float] | np.ndarray, default: float = 0.0) -> float:
    """Ordinary least-squares slope of ``values`` against ``0..n-1``.

    Returns 0 for fewer than two points or a constant series.
    """
    arr = _as_float_array(values)
    if arr.size < 2:
        return default
    x = np.arange(arr.size, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered**2))
    if denominator < EPS:
        return default
    slope = float(np.sum(x_centered * (arr - arr.mean())) / denominator)
    return slope if math.isfinite(slope) else default


def normalized_slope(values: Sequence[float] | np.ndarray, default: float = 0.0) -> float:
    """OLS slope divided by the mean absolute level, making trends comparable across entities.

    The result is the fractional change per period (for example per month).
    """
    arr = _as_float_array(values)
    if arr.size < 2:
        return default
    scale = float(np.mean(np.abs(arr)))
    if scale < EPS:
        return default
    return safe_div(linear_slope(arr), scale, default=default)


def max_drawdown(values: Sequence[float] | np.ndarray, default: float = 0.0) -> float:
    """Largest peak-to-trough decline as a positive fraction of the running peak.

    For a balance series ``[100, 80, 90, 45]`` the result is ``0.55``. Returns 0 when the series
    never declines and ``default`` for empty input.
    """
    arr = _as_float_array(values)
    if arr.size == 0:
        return default
    peak = np.maximum.accumulate(arr)
    # Avoid division by zero or negative peaks; balances can legitimately be negative.
    scale = np.where(np.abs(peak) < EPS, np.nan, peak)
    drawdown = np.where(np.isnan(scale), 0.0, (peak - arr) / np.abs(scale))
    result = float(np.nanmax(drawdown)) if drawdown.size else default
    return float(np.clip(result, 0.0, 1.0)) if math.isfinite(result) else default


def seasonal_strength(values: Sequence[float], period: int = 12, default: float = 0.0) -> float:
    """Share of variance explained by repeating calendar positions (a simple seasonality score).

    ``values`` must be ordered oldest to newest with a fixed step (usually months). Each value is
    compared with the mean of its position modulo ``period``. The score is
    ``1 - var(residual) / var(values)`` clipped to 0..1. It is deliberately simple: it is a
    descriptive feature, not a statistical test.
    """
    arr = _as_float_array(values)
    # Two full cycles are needed before a repeating pattern is identifiable rather than an artefact
    # of seeing each calendar position only once.
    if arr.size < 2 * period or float(np.var(arr)) < EPS:
        return default
    positions = np.arange(arr.size) % period
    residual = np.empty_like(arr)
    for pos in np.unique(positions):
        mask = positions == pos
        residual[mask] = arr[mask] - float(np.mean(arr[mask]))
    total_var = float(np.var(arr))
    if total_var < EPS:
        return default
    score = 1.0 - float(np.var(residual)) / total_var
    return float(np.clip(score, 0.0, 1.0))


def seasonal_peak(values: Sequence[float], period: int = 12, default: float = 0.0) -> float:
    """Calendar position (1-based) with the highest average value, or ``default`` if undefined."""
    arr = _as_float_array(values)
    if arr.size < 2 * period:
        return default
    positions = np.arange(arr.size) % period
    means = [(pos + 1, float(np.mean(arr[positions == pos]))) for pos in np.unique(positions)]
    if not means:
        return default
    return float(max(means, key=lambda item: item[1])[0])


def trend_ratio(recent: float, prior: float, default: float = 0.0) -> float:
    """Relative change from ``prior`` to ``recent``, e.g. ``0.25`` for +25%.

    Returns ``default`` when the prior value is zero or either input is non-finite.
    """
    if not math.isfinite(float(prior)) or abs(float(prior)) < EPS:
        return default
    change = safe_div(float(recent) - float(prior), abs(float(prior)), default=default)
    return change


def cagr(first: float, last: float, periods: int, default: float = 0.0) -> float:
    """Compound growth rate per period, robust to non-positive endpoints."""
    if periods <= 0 or first <= 0 or last <= 0:
        return default
    rate = (last / first) ** (1.0 / periods) - 1.0
    return float(rate) if math.isfinite(rate) else default


def clipped_ratio(
    numerator: float, denominator: float, cap: float = 10.0, default: float = 0.0
) -> float:
    """``safe_div`` with an upper cap, useful for ratios that can explode near zero."""
    return float(np.clip(safe_div(numerator, denominator, default=default), -cap, cap))


def stability_score(values: Sequence[float], default: float = 0.0) -> float:
    """Map the coefficient of variation to a 0..1 stability score (1 is most stable)."""
    cv = coefficient_of_variation(values, default=float("nan"))
    if not math.isfinite(cv):
        return default
    return float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))


def weighted_mean(values: Sequence[float], weights: Sequence[float], default: float = 0.0) -> float:
    """Weighted mean with non-negative weights, safe for zero total weight."""
    arr = _as_float_array(values)
    w = _as_float_array(weights)
    if arr.size == 0 or arr.size != w.size:
        return default
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total < EPS:
        return default
    return float(np.sum(arr * w) / total)
