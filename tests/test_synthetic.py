"""Tests for the synthetic ledger generator and its labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opencashflow.data.synthetic import (
    TEMPLATES,
    CompanyProfile,
    _ledger_outcomes,
    _transaction,
    generate_company_transactions,
    generate_credit_dataset,
    generate_portfolio,
)
from opencashflow.features.cashflow_features import CashFlowFeatureExtractor, FeatureConfig
from opencashflow.features.categorizer import TransactionCategorizer


def test_transaction_schema_and_balance(company_transactions):
    expected = {
        "entity_id",
        "transaction_id",
        "date",
        "amount",
        "currency",
        "description",
        "counterparty",
        "balance",
    }
    assert set(company_transactions.columns) == expected
    assert len(company_transactions) > 100
    assert company_transactions["transaction_id"].is_unique
    assert (company_transactions["amount"].abs() > 0).all()
    assert company_transactions["date"].is_monotonic_increasing
    starting = 60_000.0
    assert company_transactions.iloc[-1]["balance"] == pytest.approx(
        starting + company_transactions["amount"].sum(), abs=0.02
    )


def test_generation_is_deterministic():
    first = generate_company_transactions(months=6, seed=99)
    second = generate_company_transactions(months=6, seed=99)
    pd.testing.assert_frame_equal(first, second)
    third = generate_company_transactions(months=6, seed=100)
    assert not first["amount"].equals(third["amount"])


def test_portfolio_labels_and_metadata(portfolio):
    assert len(portfolio.labels) == 30
    assert portfolio.transactions["entity_id"].nunique() == 30
    assert set(portfolio.labels["default_flag"].unique()) == {0, 1}
    for column in [
        "pd_true",
        "stress",
        "min_balance_ratio",
        "ending_balance_ratio",
        "negative_month_ratio",
        "inflow_growth",
        "nsf_count",
    ]:
        assert column in portfolio.labels.columns
    assert 0 < portfolio.labels["default_flag"].mean() < 1
    assert "companies" in portfolio.summary()
    assert portfolio.metadata["n_companies"] == 30
    assert portfolio.metadata["outcome_months"] == 12
    assert "observation_end" in portfolio.metadata
    assert " to " in portfolio.metadata["outcome_window"]
    assert len(portfolio.outcome_transactions) > 0


def test_to_features_alignment(portfolio):
    X, y = portfolio.to_features()
    assert len(X) == len(y) == 30
    assert X.index.equals(y.index)
    assert set(y.unique()) <= {0, 1}
    assert X["txn_count"].min() > 0


def test_to_features_pins_point_in_time_cutoff(portfolio):
    config = FeatureConfig(window_days=730)
    as_of = portfolio.metadata["observation_end"]
    expected = CashFlowFeatureExtractor(config=config).fit_transform(
        portfolio.transactions, as_of=as_of
    )
    X, y = portfolio.to_features()
    pd.testing.assert_frame_equal(X, expected.reindex(X.index))
    expected_y = (
        portfolio.labels.set_index("entity_id")["default_flag"].reindex(X.index).astype(int)
    )
    pd.testing.assert_series_equal(y, expected_y)


def test_credit_dataset_helper():
    X, y, portfolio = generate_credit_dataset(
        n_companies=12, months=10, default_rate=0.25, seed=5, window_days=310
    )
    assert len(X) == len(y) == 12
    assert X.shape[1] >= 100
    assert portfolio.labels["default_flag"].isin([0, 1]).all()


def test_profile_validation():
    with pytest.raises(ValueError, match="distress"):
        CompanyProfile(distress=1.5)
    with pytest.raises(ValueError, match="country"):
        CompanyProfile(country="XX")
    with pytest.raises(ValueError, match="revenue"):
        CompanyProfile(monthly_revenue=0)


@pytest.mark.parametrize("country", sorted(TEMPLATES))
def test_country_templates_categorise(country):
    profile = CompanyProfile(entity_id=f"TEST-{country}", country=country)
    transactions = generate_company_transactions(profile, months=6, seed=1)
    categorised = TransactionCategorizer().categorize_frame(transactions)
    other = categorised["category"].isin(["other_inflow", "other_outflow"])
    assert not other.any(), categorised.loc[other, ["description", "category"]].to_string()


def test_realised_default_rate_is_close_to_target():
    portfolio = generate_portfolio(n_companies=120, months=12, default_rate=0.25, seed=21)
    realised = float(portfolio.labels["default_flag"].mean())
    assert 0.10 <= realised <= 0.40
    assert portfolio.metadata["default_rate_realised"] == pytest.approx(realised)


def test_outcome_columns_are_informative(portfolio):
    labels = portfolio.labels
    defaults = labels[labels["default_flag"] == 1]
    non_defaults = labels[labels["default_flag"] == 0]
    assert defaults["negative_month_ratio"].mean() > non_defaults["negative_month_ratio"].mean()
    assert defaults["min_balance_ratio"].mean() < non_defaults["min_balance_ratio"].mean()


def test_no_nan_in_core_transaction_columns(company_transactions):
    for column in ["entity_id", "date", "amount", "description", "balance"]:
        assert company_transactions[column].notna().all()
    assert np.isfinite(company_transactions["amount"]).all()
    assert np.isfinite(company_transactions["balance"]).all()


def test_forward_window_split(portfolio):
    end = pd.Timestamp(portfolio.metadata["observation_end"])
    observable = portfolio.transactions
    outcome = portfolio.outcome_transactions
    assert len(outcome) > 0
    assert list(outcome.columns) == list(observable.columns)
    assert (observable["date"] < end + pd.Timedelta(days=1)).all()
    assert (outcome["date"] > end).all()
    assert outcome["date"].dt.to_period("M").nunique() == portfolio.metadata["outcome_months"]


def test_forward_window_literal_dates(portfolio):
    assert portfolio.metadata["observation_end"] == "2024-04-30"
    assert portfolio.metadata["outcome_window"] == "2024-05 to 2025-04"
    observable = portfolio.transactions
    outcome = portfolio.outcome_transactions
    assert observable["date"].max() <= pd.Timestamp("2024-04-30")
    assert outcome["date"].min() >= pd.Timestamp("2024-05-01")
    assert sorted(outcome["date"].dt.to_period("M").astype(str).unique()) == [
        "2024-05",
        "2024-06",
        "2024-07",
        "2024-08",
        "2024-09",
        "2024-10",
        "2024-11",
        "2024-12",
        "2025-01",
        "2025-02",
        "2025-03",
        "2025-04",
    ]


def test_labels_match_outcome_rows(portfolio):
    profiles = [CompanyProfile(entity_id=eid) for eid in portfolio.labels["entity_id"]]
    outcomes = _ledger_outcomes(portfolio.outcome_transactions, profiles)
    revenue = portfolio.labels["monthly_revenue"].to_numpy(dtype=float)
    np.testing.assert_allclose(
        outcomes["min_balance"].to_numpy(dtype=float) / revenue,
        portfolio.labels["min_balance_ratio"].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        outcomes["negative_month_ratio"].to_numpy(dtype=float),
        portfolio.labels["negative_month_ratio"].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        outcomes["nsf_count"].to_numpy(dtype=float),
        portfolio.labels["nsf_count"].to_numpy(dtype=float),
    )


def test_nsf_count_uses_exact_templates():
    date = pd.Timestamp("2024-01-01")
    descriptions = [
        "INTERNAL TRANSFER FROM OWN ACCOUNT",
        "TRANSFERENCIA RECIBIDA FACTURA 202401001 DE Casa Verde",
        *(TEMPLATES[country]["nsf"] for country in sorted(TEMPLATES)),
    ]
    rows = [
        _transaction(date + pd.Timedelta(days=i), -100.0, desc, "Bank", "EUR")
        for i, desc in enumerate(descriptions)
    ]
    ledger = pd.DataFrame(rows)
    ledger["entity_id"] = "TEST"
    ledger["transaction_id"] = [f"TEST-{i:03d}" for i in range(len(ledger))]
    ledger["balance"] = -100.0 * (np.arange(len(ledger)) + 1.0)
    outcomes = _ledger_outcomes(ledger, [CompanyProfile(entity_id="TEST")])
    assert outcomes.loc["TEST", "nsf_count"] == 6


def test_features_ignore_outcome_rows(portfolio):
    as_of = portfolio.metadata["observation_end"]
    config = FeatureConfig(window_days=5000)
    X_observable = CashFlowFeatureExtractor(config=config).fit_transform(
        portfolio.transactions, as_of=as_of
    )
    combined = pd.concat(
        [portfolio.transactions, portfolio.outcome_transactions], ignore_index=True
    )
    X_combined = CashFlowFeatureExtractor(config=config).fit_transform(combined, as_of=as_of)
    pd.testing.assert_frame_equal(X_observable, X_combined)


def test_portfolio_is_deterministic():
    first = generate_portfolio(n_companies=8, months=6, default_rate=0.25, seed=77)
    second = generate_portfolio(n_companies=8, months=6, default_rate=0.25, seed=77)
    pd.testing.assert_frame_equal(first.labels, second.labels)
    pd.testing.assert_frame_equal(first.transactions, second.transactions)
    pd.testing.assert_frame_equal(first.outcome_transactions, second.outcome_transactions)


@pytest.mark.parametrize("bad_outcome_months", [0, 2])
def test_outcome_months_validation(bad_outcome_months):
    with pytest.raises(ValueError, match="outcome_months"):
        generate_portfolio(n_companies=2, months=3, outcome_months=bad_outcome_months)
