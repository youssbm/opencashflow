"""Tests for numerical ratio helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from opencashflow.features.ratios import (
    cagr,
    clipped_ratio,
    coefficient_of_variation,
    gini,
    hhi,
    linear_slope,
    max_drawdown,
    normalized_entropy,
    normalized_slope,
    safe_div,
    seasonal_peak,
    seasonal_strength,
    stability_score,
    trend_ratio,
)


def test_safe_div_handles_zero_and_non_finite():
    assert safe_div(10, 4) == 2.5
    assert safe_div(10, 0) == 0.0
    assert safe_div(10, 0, default=-1.0) == -1.0
    assert math.isnan(safe_div(10, 0, default=float("nan")))
    assert safe_div(float("inf"), 2) == 0.0


def test_coefficient_of_variation():
    assert coefficient_of_variation([5, 5, 5]) == 0.0
    assert coefficient_of_variation([0]) == 0.0
    assert coefficient_of_variation([1, 3]) == pytest.approx(0.5)
    assert coefficient_of_variation([1, -1]) == 0.0  # mean zero


def test_concentration_indices():
    assert hhi([10, 0, 0]) == 1.0
    assert hhi([1, 1, 1]) == pytest.approx(1 / 3)
    assert hhi([]) == 0.0
    assert normalized_entropy([1, 1, 1]) == pytest.approx(1.0)
    assert normalized_entropy([1, 0, 0]) == 0.0
    assert normalized_entropy([1]) == 0.0
    assert gini([1, 1, 1]) == 0.0
    assert gini([0, 0, 10]) > 0.6


def test_trend_helpers():
    assert linear_slope([1, 2, 3, 4]) == pytest.approx(1.0)
    assert linear_slope([5]) == 0.0
    assert normalized_slope([100, 110, 120]) == pytest.approx(10 / 110)
    assert trend_ratio(120, 100) == pytest.approx(0.2)
    assert trend_ratio(100, 0) == 0.0
    assert cagr(100, 121, 2) == pytest.approx(0.1)
    assert cagr(100, 0, 2) == 0.0


def test_max_drawdown():
    assert max_drawdown([100, 80, 90, 45]) == pytest.approx(0.55)
    assert max_drawdown([1, 2, 3]) == 0.0
    assert max_drawdown([]) == 0.0


def test_seasonality_requires_two_cycles():
    pattern = [0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0]
    values = pattern * 2
    assert seasonal_strength(values) == pytest.approx(1.0)
    assert seasonal_strength(pattern) == 0.0  # one cycle is not identifiable
    assert seasonal_peak(values) == 6.0
    assert seasonal_strength([5] * 24) == 0.0


def test_misc_ratios():
    assert clipped_ratio(100, 1, cap=10) == 10.0
    assert clipped_ratio(-100, 1, cap=10) == -10.0
    assert 0.0 <= stability_score([1, 1, 1]) <= 1.0
    assert stability_score([1, 1, 1]) == 1.0
    assert np.isfinite(stability_score([]))
