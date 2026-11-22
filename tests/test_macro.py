"""Tests for optional macro-context joining."""

from __future__ import annotations

import pandas as pd
import pytest

from opencashflow.data.macro import MacroFeatureJoiner, load_macro_csv


def _macro_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country": ["DE", "DE", "DE", "PL", "PL", "PL"],
            "period": [
                "2024-01-01",
                "2024-02-01",
                "2024-03-01",
                "2024-01-01",
                "2024-02-01",
                "2024-03-01",
            ],
            "indicator": ["confidence"] * 6,
            "value": [96.0, 95.0, 97.0, 98.0, 97.0, 99.0],
        }
    )


def test_macro_join_lags_and_country():
    joiner = MacroFeatureJoiner(lags=(0, 1), yoy_periods=None).fit(_macro_frame())
    features = pd.DataFrame({"revenue": [100.0, 200.0]}, index=["A", "B"])
    countries = pd.Series({"A": "DE", "B": "PL"})
    as_of = pd.Series({"A": "2024-03-15", "B": "2024-03-15"})
    result = joiner.transform(features, countries, as_of)
    assert result.loc["A", "macro_confidence_lag0"] == 95.0
    assert result.loc["A", "macro_confidence_lag1"] == 96.0
    assert result.loc["B", "macro_confidence_lag0"] == 97.0
    assert result.loc["B", "macro_confidence_lag1"] == 98.0
    assert result.loc["A", "macro_missing"] == 0.0


def test_macro_join_fills_missing_and_flags():
    joiner = MacroFeatureJoiner(lags=(0,), yoy_periods=None, fill_missing=-1.0).fit(_macro_frame())
    features = pd.DataFrame({"revenue": [100.0]}, index=["A"])
    countries = pd.Series({"A": "DE"})
    result = joiner.transform(features, countries, "2023-12-15")
    assert result.loc["A", "macro_confidence_lag0"] == -1.0
    assert result.loc["A", "macro_missing"] == 1.0


def test_macro_unknown_country_is_flagged():
    joiner = MacroFeatureJoiner(lags=(0,), yoy_periods=None, fill_missing=0.0).fit(_macro_frame())
    features = pd.DataFrame({"x": [1.0]}, index=["A"])
    result = joiner.transform(features, {"A": "US"}, "2024-02-15")
    assert result.loc["A", "macro_confidence_lag0"] == 0.0
    assert result.loc["A", "macro_missing"] == 1.0


def test_macro_yoy():
    macro = pd.DataFrame(
        {
            "country": ["DE"] * 13,
            "period": pd.date_range("2023-01-01", periods=13, freq="MS"),
            "indicator": ["confidence"] * 13,
            "value": [100.0] * 12 + [110.0],
        }
    )
    joiner = MacroFeatureJoiner(lags=(0,), yoy_periods=12).fit(macro)
    features = pd.DataFrame({"x": [1.0]}, index=["A"])
    result = joiner.transform(features, {"A": "DE"}, "2024-02-15")
    assert result.loc["A", "macro_confidence_yoy12"] == pytest.approx(0.10)


def test_macro_publication_lag():
    frame = _macro_frame()
    features = pd.DataFrame({"x": [1.0]}, index=["A"])
    countries = {"A": "DE"}
    latest = (
        MacroFeatureJoiner(lags=(0,), yoy_periods=None)
        .fit(frame)
        .transform(features, countries, "2024-02-15")
    )
    assert latest.loc["A", "macro_confidence_lag0"] == 96.0
    same_month = (
        MacroFeatureJoiner(lags=(0,), yoy_periods=None, publication_lag=0)
        .fit(frame)
        .transform(features, countries, "2024-02-15")
    )
    assert same_month.loc["A", "macro_confidence_lag0"] == 95.0
    shifted = (
        MacroFeatureJoiner(lags=(0, 1), yoy_periods=None, publication_lag=1)
        .fit(frame)
        .transform(features, countries, "2024-03-15")
    )
    assert shifted.loc["A", "macro_confidence_lag0"] == 95.0
    assert shifted.loc["A", "macro_confidence_lag1"] == 96.0
    too_old = (
        MacroFeatureJoiner(lags=(0,), yoy_periods=None, publication_lag=2, fill_missing=-1.0)
        .fit(frame)
        .transform(features, countries, "2024-02-15")
    )
    assert too_old.loc["A", "macro_confidence_lag0"] == -1.0
    assert too_old.loc["A", "macro_missing"] == 1.0


def test_macro_validation():
    with pytest.raises(ValueError, match="lag"):
        MacroFeatureJoiner(lags=())
    with pytest.raises(ValueError, match="publication_lag"):
        MacroFeatureJoiner(publication_lag=-1)
    with pytest.raises(ValueError, match="missing columns"):
        MacroFeatureJoiner().fit(pd.DataFrame({"country": ["DE"]}))
    with pytest.raises(RuntimeError, match="not fitted"):
        MacroFeatureJoiner().transform(pd.DataFrame(index=["A"]), {"A": "DE"}, "2024-01-01")


def test_load_macro_csv(tmp_path):
    path = tmp_path / "macro.csv"
    _macro_frame().to_csv(path, index=False)
    frame = load_macro_csv(str(path))
    assert {"country", "period", "indicator", "value"} <= set(frame.columns)
