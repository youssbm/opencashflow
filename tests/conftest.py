"""Shared pytest fixtures.

The synthetic portfolio is generated once per session so the feature-heavy tests stay fast. Fixtures
return copies where a test may mutate the frame.
"""

from __future__ import annotations

import pytest

from opencashflow.data.synthetic import (
    SyntheticPortfolio,
    generate_company_transactions,
    generate_portfolio,
)
from opencashflow.features.cashflow_features import CashFlowFeatureExtractor, FeatureConfig


@pytest.fixture(scope="session")
def company_transactions():
    """Eighteen months of German SME transactions."""
    return generate_company_transactions(months=18, seed=7)


@pytest.fixture(scope="session")
def portfolio() -> SyntheticPortfolio:
    """A small multi-country portfolio with both classes."""
    return generate_portfolio(n_companies=30, months=16, default_rate=0.20, seed=13)


@pytest.fixture(scope="session")
def feature_config() -> FeatureConfig:
    return FeatureConfig(window_days=16 * 31)


@pytest.fixture(scope="session")
def feature_bundle(portfolio, feature_config):
    """Feature matrix and aligned labels for the portfolio."""
    return portfolio.to_features(feature_config)


@pytest.fixture(scope="session")
def feature_frame(feature_bundle):
    return feature_bundle[0].copy()


@pytest.fixture(scope="session")
def feature_labels(feature_bundle):
    return feature_bundle[1].copy()


@pytest.fixture(scope="session")
def single_feature_frame(company_transactions):
    extractor = CashFlowFeatureExtractor(config=FeatureConfig(window_days=540))
    return extractor.fit_transform(company_transactions)
