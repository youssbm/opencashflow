"""Data loading, schema normalisation, synthetic ledgers and download helpers."""

from __future__ import annotations

from typing import Any

from opencashflow.data.loaders import (
    DATASET_REGISTRY,
    DatasetBundle,
    DatasetNotAvailableError,
    describe_datasets,
    download_dataset,
    get_data_dir,
    list_datasets,
    load_bangladesh_sme,
    load_dataset,
    load_german_credit,
    load_polish_bankruptcy,
    load_sba_loans,
    load_synthetic,
    load_transactions_csv,
)
from opencashflow.data.macro import MacroFeatureJoiner, load_macro_csv
from opencashflow.data.schema import (
    CANONICAL_COLUMNS,
    TransactionSchemaError,
    normalize_transactions,
    parse_amount,
    validate_transactions,
)

_SYNTHETIC_EXPORTS = {
    "CompanyProfile",
    "SyntheticPortfolio",
    "generate_company_transactions",
    "generate_credit_dataset",
    "generate_portfolio",
}

__all__ = [
    "CANONICAL_COLUMNS",
    "TransactionSchemaError",
    "normalize_transactions",
    "parse_amount",
    "validate_transactions",
    "DatasetBundle",
    "DatasetNotAvailableError",
    "DATASET_REGISTRY",
    "download_dataset",
    "get_data_dir",
    "load_polish_bankruptcy",
    "load_sba_loans",
    "load_german_credit",
    "load_bangladesh_sme",
    "load_synthetic",
    "load_transactions_csv",
    "load_dataset",
    "list_datasets",
    "describe_datasets",
    "MacroFeatureJoiner",
    "load_macro_csv",
    *_SYNTHETIC_EXPORTS,
]


def __getattr__(name: str) -> Any:
    if name in _SYNTHETIC_EXPORTS:
        from opencashflow.data import synthetic

        return getattr(synthetic, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
