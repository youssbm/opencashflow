"""Tests for the cash-flow feature engine and its catalogue."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from opencashflow.features.cashflow_features import (
    CashFlowFeatureExtractor,
    FeatureConfig,
    _daily_balance_series,
    compute_features,
)
from opencashflow.features.categorizer import CATEGORY_DESCRIPTIONS
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
    assert frame["formula"].str.len().min() > 0
    assert frame["higher_is_better"].dropna().isin([True, False]).all()


def test_every_feature_has_formula_and_description():
    for spec in FEATURE_SPECS.values():
        assert spec.description.strip(), f"Empty description for {spec.name}"
        assert spec.formula.strip(), f"Empty formula for {spec.name}"


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


def _ledger(rows: list[tuple[str, float, str, float]]) -> pd.DataFrame:
    """Build a small canonical ledger from (date, amount, category, balance) tuples."""
    return pd.DataFrame(
        {
            "entity_id": ["E1"] * len(rows),
            "transaction_id": [f"t{i}" for i in range(len(rows))],
            "date": pd.to_datetime([r[0] for r in rows]),
            "amount": [r[1] for r in rows],
            "currency": ["EUR"] * len(rows),
            "description": [r[2] for r in rows],
            "category": [r[2] if r[2] in CATEGORY_DESCRIPTIONS else pd.NA for r in rows],
            "counterparty": ["CP"] * len(rows),
            "balance": [r[3] for r in rows],
        }
    )


def test_negative_balance_days_count_days_not_transactions():
    rows = [
        ("2024-01-01", 100.0, "revenue", -50.0),
        ("2024-01-02", -10.0, "supplier", -60.0),
        ("2024-01-02", -5.0, "supplier", -65.0),
        ("2024-01-03", -10.0, "supplier", -75.0),
        ("2024-01-04", 200.0, "revenue", 125.0),
        ("2024-01-05", -5.0, "supplier", 120.0),
    ]
    config = FeatureConfig(window_days=30, as_of="2024-01-10")
    row = CashFlowFeatureExtractor(config=config).fit_transform(_ledger(rows)).iloc[0]
    assert row["negative_balance_days"] == 3
    assert row["negative_balance_ratio"] == pytest.approx(0.6)
    assert 0.0 <= row["negative_balance_ratio"] <= 1.0


def test_negative_balance_ratio_is_bounded_on_portfolio(portfolio, feature_config):
    X = CashFlowFeatureExtractor(config=feature_config).fit_transform(portfolio.transactions)
    assert X["negative_balance_ratio"].between(0.0, 1.0).all()
    assert (X["negative_balance_days"] <= X["txn_count"]).all()


def test_dscr_uses_principal_and_interest():
    rows = [
        ("2024-01-10", 100.0, "revenue", 100.0),
        ("2024-01-11", -40.0, "supplier", 60.0),
        ("2024-01-12", -20.0, "loan_repayment", 40.0),
        ("2024-01-13", -10.0, "interest_expense", 30.0),
    ]
    config = FeatureConfig(window_days=30, as_of="2024-01-31")
    row = CashFlowFeatureExtractor(config=config).fit_transform(_ledger(rows)).iloc[0]
    assert row["revenue_total"] == pytest.approx(100.0)
    assert row["expense_total"] == pytest.approx(50.0)
    assert row["interest_total"] == pytest.approx(10.0)
    assert row["loan_repayment_total"] == pytest.approx(20.0)
    assert row["dscr"] == pytest.approx(2.0)
    assert row["interest_coverage"] == pytest.approx(6.0)


def test_dscr_sentinel_without_debt_service():
    rows = [
        ("2024-01-10", 100.0, "revenue", 100.0),
        ("2024-01-11", -40.0, "supplier", 60.0),
    ]
    config = FeatureConfig(window_days=30, as_of="2024-01-31")
    row = CashFlowFeatureExtractor(config=config).fit_transform(_ledger(rows)).iloc[0]
    assert row["dscr"] == pytest.approx(10.0)


def test_daily_balance_sums_two_accounts():
    tx = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02", "2024-01-04"]),
            "amount": [100.0, 50.0, 200.0, -20.0],
            "balance": [100.0, 150.0, 200.0, 180.0],
            "account_id": ["A", "A", "B", "B"],
        }
    )
    start = pd.Timestamp("2024-01-01")
    as_of = pd.Timestamp("2024-01-04")
    result = _daily_balance_series(tx, start, as_of)
    expected = pd.Series([300.0, 300.0, 350.0, 330.0], index=pd.date_range(start, as_of))
    pd.testing.assert_series_equal(result, expected)


def test_daily_balance_single_account_unchanged():
    tx = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "amount": [100.0, 50.0],
            "balance": [100.0, 150.0],
            "account_id": ["A", "A"],
        }
    )
    start = pd.Timestamp("2024-01-01")
    as_of = pd.Timestamp("2024-01-04")
    result = _daily_balance_series(tx, start, as_of)
    expected = pd.Series(
        [100.0, 100.0, 150.0, 150.0], index=pd.date_range(start, as_of), name="balance"
    )
    pd.testing.assert_series_equal(result, expected)


def test_daily_balance_warns_for_account_without_balance():
    tx = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02", "2024-01-02"]
            ),
            "amount": [100.0, 50.0, 200.0, -20.0, -10.0],
            "balance": [100.0, 150.0, 200.0, 180.0, np.nan],
            "account_id": ["A", "A", "B", "B", "C"],
        }
    )
    with pytest.warns(UserWarning, match="1 account"):
        _daily_balance_series(tx, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))


def test_balance_warning_without_balance_column(company_transactions):
    without_balance = company_transactions.drop(columns=["balance"])
    config = FeatureConfig(window_days=540, as_of="2024-12-31")
    with pytest.warns(UserWarning, match="cumulative-flow proxies"):
        CashFlowFeatureExtractor(config=config).fit_transform(without_balance)


def test_no_balance_warning_with_synthetic_balances(company_transactions):
    config = FeatureConfig(window_days=540, as_of="2024-12-31")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CashFlowFeatureExtractor(config=config).fit_transform(company_transactions)
    assert not any("cumulative-flow proxies" in str(w.message) for w in caught)


def test_as_of_warning_without_fixed_date(company_transactions):
    with pytest.warns(UserWarning, match="No as_of date"):
        CashFlowFeatureExtractor(config=FeatureConfig(window_days=540)).fit_transform(
            company_transactions
        )


def test_no_as_of_warning_with_config_or_override(company_transactions):
    extractor = CashFlowFeatureExtractor(config=FeatureConfig(window_days=540, as_of="2024-12-31"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        extractor.transform(company_transactions)
    assert not any("No as_of date" in str(w.message) for w in caught)

    extractor = CashFlowFeatureExtractor(config=FeatureConfig(window_days=540))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        extractor.transform(company_transactions, as_of="2024-12-31")
    assert not any("No as_of date" in str(w.message) for w in caught)


def test_capital_injection_does_not_change_revenue_or_debt_totals():
    rows = [
        ("2024-01-10", 1000.0, "CAPITAL INJECTION SHAREHOLDER", 1000.0),
    ]
    config = FeatureConfig(window_days=30, as_of="2024-01-31")
    row = CashFlowFeatureExtractor(config=config).fit_transform(_ledger(rows)).iloc[0]
    assert row["revenue_total"] == pytest.approx(0.0)
    assert row["loan_proceeds_total"] == pytest.approx(0.0)
    assert row["net_cash_flow_total"] == pytest.approx(1000.0)


@pytest.mark.parametrize("description", ["PRELEVEMENT IMPAYE", "LASTSCHRIFT NICHT EINGELOEST"])
def test_returned_payment_features_count_nsf_rows(description):
    rows = [
        ("2024-01-10", -25.0, description, 100.0),
        ("2024-01-11", 500.0, "revenue", 600.0),
    ]
    config = FeatureConfig(window_days=30, as_of="2024-01-31")
    row = CashFlowFeatureExtractor(config=config).fit_transform(_ledger(rows)).iloc[0]
    assert row["nsf_count"] == 1
    assert row["returned_payment_count"] == 1
    assert row["returned_payment_rate"] > 0


def test_balance_warning_when_only_one_account_has_balances():
    tx = pd.DataFrame(
        {
            "entity_id": ["E1"] * 4,
            "transaction_id": ["t1", "t2", "t3", "t4"],
            "date": pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-05", "2024-01-06"]),
            "amount": [100.0, 50.0, 200.0, 30.0],
            "currency": ["EUR"] * 4,
            "description": ["INVOICE A", "INVOICE A", "INVOICE B", "INVOICE B"],
            "category": ["revenue"] * 4,
            "balance": [100.0, 150.0, np.nan, np.nan],
            "account_id": ["A", "A", "B", "B"],
        }
    )
    config = FeatureConfig(window_days=30, as_of="2024-01-31")
    with pytest.warns(UserWarning, match="1 account"):
        row = CashFlowFeatureExtractor(config=config).fit_transform(tx).iloc[0]
    assert row["txn_count"] == 4
    assert row["net_cash_flow_total"] == pytest.approx(380.0)
    assert row["revenue_total"] == pytest.approx(380.0)
