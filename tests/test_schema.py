"""Tests for canonical transaction normalisation and amount parsing."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from opencashflow.data.schema import (
    TransactionSchemaError,
    entity_transaction_counts,
    normalize_transactions,
    parse_amount,
    validate_transactions,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,234.56", 1234.56),
        ("1.234,56", 1234.56),
        ("€ 1.234,56", 1234.56),
        ("-€ 12.50", -12.5),
        ("(42.00)", -42.0),
        ("1 234,56", 1234.56),
        ("12.5-", -12.5),
        ("not-a-number", math.nan),
        ("", math.nan),
        (None, math.nan),
        (12.5, 12.5),
    ],
)
def test_parse_amount(raw, expected):
    result = parse_amount(raw)
    if math.isnan(expected):
        assert math.isnan(result)
    else:
        assert result == pytest.approx(expected)


def test_normalize_unsigned_with_direction_column():
    raw = pd.DataFrame(
        {
            "account_id": ["A", "A"],
            "booking_date": ["2024-01-01", "2024-01-02"],
            "transaction_amount": ["1.234,56", "500.00"],
            "credit_debit": ["credit", "debit"],
            "narrative": ["SEPA CREDIT INVOICE", "SALARY"],
            "running_balance": ["1.234,56", "734,56"],
        }
    )
    result = normalize_transactions(raw)
    assert list(result["amount"]) == pytest.approx([1234.56, -500.0])
    assert list(result["direction"].astype(str)) == ["inflow", "outflow"]
    assert list(result["balance"]) == pytest.approx([1234.56, 734.56])
    assert result["entity_id"].tolist() == ["A", "A"]
    assert result["currency"].tolist() == ["EUR", "EUR"]
    assert result["transaction_id"].is_unique


def test_normalize_signed_amounts_and_aliases():
    raw = pd.DataFrame(
        {
            "entity_id": ["E1", "E1"],
            "date": ["2024-03-01", "2024-03-02"],
            "amount": [-100.0, 250.0],
            "description": ["RENT PAYMENT", "INVOICE PAID"],
            "counterparty": ["Landlord", "Customer"],
        }
    )
    result = normalize_transactions(raw)
    assert result["direction"].astype(str).tolist() == ["outflow", "inflow"]
    assert result.loc[0, "counterparty"] == "Landlord"
    assert result["transaction_id"].tolist() == ["txn_00000000", "txn_00000001"]


def test_normalize_dayfirst_and_defaults():
    raw = pd.DataFrame(
        {
            "date": ["02/01/2024"],
            "amount": [10],
            "description": ["TEST"],
        }
    )
    result = normalize_transactions(raw, dayfirst=True)
    assert result.loc[0, "date"] == pd.Timestamp("2024-01-02")
    assert result.loc[0, "entity_id"] == "entity_0"
    assert result.loc[0, "currency"] == "EUR"
    assert result.loc[0, "direction"] == "inflow"


def test_zero_amount_is_outflow():
    raw = pd.DataFrame({"date": ["2024-01-01"], "amount": [0.0], "description": ["ZERO"]})
    result = normalize_transactions(raw)
    assert result.loc[0, "direction"] == "outflow"


def test_missing_required_columns_raises():
    raw = pd.DataFrame({"date": ["2024-01-01"], "amount": [10]})
    with pytest.raises(TransactionSchemaError, match="Missing required"):
        normalize_transactions(raw)


def test_empty_table_raises():
    with pytest.raises(TransactionSchemaError, match="empty"):
        normalize_transactions(pd.DataFrame())


def test_invalid_rows_are_dropped_with_warning():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01", "not-a-date", "2024-01-03", "2024-01-04", "2024-01-05"],
            "amount": [10, 20, 30, 40, 50],
            "description": ["OK", "BAD", "C", "D", "E"],
        }
    )
    with pytest.warns(UserWarning, match="Dropped"):
        result = normalize_transactions(raw)
    assert len(result) == 4


def test_too_many_invalid_rows_raises():
    raw = pd.DataFrame(
        {
            "date": ["bad", "bad", "2024-01-01"],
            "amount": [1, 2, 3],
            "description": ["A", "B", "C"],
        }
    )
    with pytest.raises(TransactionSchemaError, match="invalid date or amount"):
        normalize_transactions(raw)


def test_unsigned_without_direction_raises():
    raw = pd.DataFrame({"date": ["2024-01-01"], "amount": [10.0], "description": ["NO SIGN"]})
    with pytest.raises(TransactionSchemaError, match="unsigned"):
        normalize_transactions(raw, amount_is_signed=False)


def test_duplicate_removal():
    raw = pd.DataFrame(
        {
            "entity_id": ["A", "A"],
            "date": ["2024-01-01", "2024-01-01"],
            "amount": [-10.0, -10.0],
            "description": ["RENT", "RENT"],
        }
    )
    result = normalize_transactions(raw, drop_duplicates=True)
    assert len(result) == 1


def test_validate_transactions():
    canonical = normalize_transactions(
        pd.DataFrame({"date": ["2024-01-01"], "amount": [1.0], "description": ["X"]})
    )
    validate_transactions(canonical)
    broken = canonical.copy()
    broken.loc[0, "date"] = pd.NaT
    with pytest.raises(TransactionSchemaError, match="NaT"):
        validate_transactions(broken)
    with pytest.raises(TransactionSchemaError, match="empty"):
        validate_transactions(pd.DataFrame())


def test_entity_counts():
    canonical = normalize_transactions(
        pd.DataFrame(
            {
                "entity_id": ["A", "A", "B"],
                "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
                "amount": [1.0, 2.0, 3.0],
                "description": ["X", "Y", "Z"],
            }
        )
    )
    counts = entity_transaction_counts(canonical)
    assert counts.loc["A"] == 2
    assert counts.loc["B"] == 1
