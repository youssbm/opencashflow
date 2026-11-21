"""Tests for the cash-flow feature engine and its catalogue."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opencashflow.features.cashflow_features import (
    CashFlowFeatureExtractor,
    FeatureConfig,
    compute_features,
)
from opencashflow.features.specs import (
    FEATURE_NAMES,
    FEATURE_SPECS,
    feature_group_counts,
    feature_specs_frame,
)


def test_catalogue_is_complete_and_unique():
    assert len(FEATURE_SPECS) >= 100
    assert len(FEATURE_NAMES) == len(FEATURE_SPECS)
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert sum(feature_group_counts().values()) == len(FEATURE_SPECS)
    frame = feature_specs_frame()
    assert len(frame) == len(FEATURE_SPECS)
    assert set(frame.columns) == {"name", "group", "description", "higher_is_better", "formula"}
    assert frame["description"].str.len().min() > 10
    assert frame["higher_is_better"].dropna().isin([True, False]).all()


def test_single_entity_feature_matrix(company_transactions):
    extractor = CashFlowFeatureExtractor(config=FeatureConfig(window_days=540))
    features = extractor.fit_transform(company_transactions)
    assert features.shape == (1, len(FEATURE_NAMES))
    assert features.index.tolist() == ["SME0001"]
    assert list(features.columns) == list(FEATURE_NAMES)
    assert np.isfinite(features.to_numpy()).all()
    row = features.iloc[0]
    assert row["txn_count"] > 100
    assert row["months_observed"] >= 12
    assert 0 <= row["coverage_ratio"] <= 1
    assert row["has_balance_data"] == 1
    assert 0 <= row["cashflow_health_score"] <= 100
    assert row["revenue_total"] > 0
    assert row["expense_total"] > 0


def test_transform_is_deterministic(company_transactions):
    extractor = CashFlowFeatureExtractor(config=FeatureConfig(window_days=540))
    first = extractor.fit_transform(company_transactions)
    second = extractor.transform(company_transactions)
    pd.testing.assert_frame_equal(first, second)


def test_as_of_truncates_history(company_transactions):
    extractor = CashFlowFeatureExtractor(config=FeatureConfig(window_days=540))
    latest = extractor.fit_transform(company_transactions)
    early = extractor.transform(company_transactions, as_of="2024-01-31")
    assert early.iloc[0]["txn_count"] < latest.iloc[0]["txn_count"]
    assert early.iloc[0]["months_observed"] < latest.iloc[0]["months_observed"]


def test_multi_entity_alignment(portfolio, feature_config, feature_labels):
    extractor = CashFlowFeatureExtractor(config=feature_config)
    features = extractor.fit_transform(portfolio.transactions)
    assert len(features) == portfolio.labels["entity_id"].nunique()
    assert set(features.index) == set(feature_labels.index)
    assert np.isfinite(features.to_numpy()).all()


def test_transform_one(company_transactions):
    extractor = CashFlowFeatureExtractor(config=FeatureConfig(window_days=540))
    extractor.fit(company_transactions)
    row = extractor.transform_one(company_transactions)
    assert isinstance(row, pd.Series)
    assert len(row) == len(FEATURE_NAMES)


def test_compute_features_convenience(company_transactions):
    features = compute_features(company_transactions, FeatureConfig(window_days=540))
    assert features.shape[0] == 1
    assert features.shape[1] == len(FEATURE_NAMES)


def test_config_validation():
    with pytest.raises(ValueError, match="window_days"):
        FeatureConfig(window_days=10)
    with pytest.raises(ValueError, match="Unknown category"):
        FeatureConfig(expense_categories=("not_a_category",))
    config = FeatureConfig(include_other_inflow_as_revenue=True)
    assert "other_inflow" in config.revenue_keys


def test_missing_balance_is_flagged(company_transactions):
    without_balance = company_transactions.drop(columns=["balance"])
    features = CashFlowFeatureExtractor(config=FeatureConfig(window_days=540)).fit_transform(
        without_balance
    )
    assert features.iloc[0]["has_balance_data"] == 0
    assert np.isfinite(features.to_numpy()).all()


def test_unknown_narratives_are_flagged(company_transactions):
    mutated = company_transactions.copy()
    mutated["description"] = "ZZZ RANDOM NARRATIVE"
    mutated["counterparty"] = "ZZZ"
    features = CashFlowFeatureExtractor(config=FeatureConfig(window_days=540)).fit_transform(
        mutated
    )
    row = features.iloc[0]
    assert row["uncategorized_ratio"] == pytest.approx(1.0)
    assert row["revenue_total"] == 0
    assert np.isfinite(features.to_numpy()).all()


def test_longer_window_increases_revenue(company_transactions):
    short = CashFlowFeatureExtractor(config=FeatureConfig(window_days=365)).fit_transform(
        company_transactions
    )
    long = CashFlowFeatureExtractor(config=FeatureConfig(window_days=1_000)).fit_transform(
        company_transactions
    )
    assert long.iloc[0]["revenue_total"] > short.iloc[0]["revenue_total"]


def test_feature_specs_are_exported(company_transactions):
    extractor = CashFlowFeatureExtractor()
    spec = extractor.feature_spec("revenue_total")
    assert spec.group == "revenue"
    assert spec.higher_is_better is True
    with pytest.raises(KeyError):
        extractor.feature_spec("not_a_feature")
