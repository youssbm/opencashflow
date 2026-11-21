"""Tests for the synthetic ledger generator and its labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opencashflow.data.synthetic import (
    TEMPLATES,
    CompanyProfile,
    generate_company_transactions,
    generate_credit_dataset,
    generate_portfolio,
)
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


def test_to_features_alignment(portfolio):
    X, y = portfolio.to_features()
    assert len(X) == len(y) == 30
    assert X.index.equals(y.index)
    assert set(y.unique()) <= {0, 1}
    assert X["txn_count"].min() > 0


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
