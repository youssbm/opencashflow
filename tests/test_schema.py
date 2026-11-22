"""Tests for canonical transaction normalisation and amount parsing."""

from __future__ import annotations

import math
import warnings

import pandas as pd
import pytest

from opencashflow.data.schema import (
    TransactionSchemaError,
    entity_transaction_counts,
    infer_entity_column,
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
        ("1.234", 1234.0),
        ("1,234", 1234.0),
        ("-1.250", -1250.0),
        ("1.234.567", 1234567.0),
        ("1,234,567", 1234567.0),
        ("1.234.56", math.nan),
        ("1.5", 1.5),
        ("1,5", 1.5),
        ("12.00", 12.0),
        ("12,50", 12.5),
        ("0.5", 0.5),
        ("0.001", 0.001),
        ("0,001", 0.001),
        ("12.5-", -12.5),
        ("abc", math.nan),
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


def test_dayfirst_explicit_false_does_not_warn():
    raw = pd.DataFrame({"date": ["02/01/2024"], "amount": [10.0], "description": ["X"]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = normalize_transactions(raw, dayfirst=False)
    assert result.loc[0, "date"] == pd.Timestamp("2024-02-01")
    assert not any("month-first" in str(w.message) for w in caught)


def test_dayfirst_inferred_day_first():
    raw = pd.DataFrame({"date": ["25/03/2024"], "amount": [10.0], "description": ["X"]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = normalize_transactions(raw)
    assert result.loc[0, "date"] == pd.Timestamp("2024-03-25")
    assert not any("month-first" in str(w.message) for w in caught)


def test_dayfirst_inferred_month_first():
    raw = pd.DataFrame({"date": ["03/25/2024"], "amount": [10.0], "description": ["X"]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = normalize_transactions(raw)
    assert result.loc[0, "date"] == pd.Timestamp("2024-03-25")
    assert not any("month-first" in str(w.message) for w in caught)


def test_dayfirst_ambiguous_warns_and_defaults_month_first():
    raw = pd.DataFrame({"date": ["03/04/2024"], "amount": [10.0], "description": ["X"]})
    with pytest.warns(UserWarning) as record:
        result = normalize_transactions(raw)
    messages = [str(w.message) for w in record]
    assert any("month-first" in m and "dayfirst=True" in m for m in messages)
    assert result.loc[0, "date"] == pd.Timestamp("2024-03-04")


def test_dayfirst_iso_does_not_warn():
    raw = pd.DataFrame({"date": ["2024-03-04"], "amount": [10.0], "description": ["X"]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = normalize_transactions(raw)
    assert result.loc[0, "date"] == pd.Timestamp("2024-03-04")
    assert not any("month-first" in str(w.message) for w in caught)


def test_company_and_account_columns():
    raw = pd.DataFrame(
        {
            "company_id": ["C1", "C1"],
            "account_id": ["A1", "A2"],
            "date": ["2024-01-01", "2024-01-02"],
            "amount": [10.0, 20.0],
            "description": ["X", "Y"],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = normalize_transactions(raw)
    assert result["entity_id"].tolist() == ["C1", "C1"]
    assert result["account_id"].tolist() == ["A1", "A2"]
    assert not any("company-level" in str(w.message) for w in caught)


def test_account_only_becomes_entity_with_warning():
    raw = pd.DataFrame(
        {
            "account_id": ["A1", "A1"],
            "date": ["2024-01-01", "2024-01-02"],
            "amount": [10.0, 20.0],
            "description": ["X", "Y"],
        }
    )
    with pytest.warns(UserWarning, match="using account column") as record:
        result = normalize_transactions(raw)
    assert result["entity_id"].tolist() == ["A1", "A1"]
    assert result["account_id"].tolist() == ["A1", "A1"]
    assert any("grouped per account" in str(w.message) for w in record)


def test_no_account_column_gives_all_na_account_id():
    raw = pd.DataFrame(
        {
            "entity_id": ["E1", "E1"],
            "date": ["2024-01-01", "2024-01-02"],
            "amount": [10.0, 20.0],
            "description": ["X", "Y"],
        }
    )
    result = normalize_transactions(raw)
    assert "account_id" in result.columns
    assert result["account_id"].isna().all()
    assert result["entity_id"].tolist() == ["E1", "E1"]


def test_mapping_accepts_account_id():
    raw = pd.DataFrame(
        {
            "firm": ["C1", "C1"],
            "bank_account": ["A1", "A1"],
            "when": ["2024-01-01", "2024-01-02"],
            "total": [10.0, 20.0],
            "what": ["X", "Y"],
        }
    )
    result = normalize_transactions(
        raw,
        mapping={
            "entity_id": "firm",
            "account_id": "bank_account",
            "date": "when",
            "amount": "total",
            "description": "what",
        },
    )
    assert result["entity_id"].tolist() == ["C1", "C1"]
    assert result["account_id"].tolist() == ["A1", "A1"]


def test_infer_entity_column_prefers_company_level():
    raw = pd.DataFrame({"account_id": ["A1"], "company_id": ["C1"]})
    assert infer_entity_column(raw) == "company_id"
    assert infer_entity_column(pd.DataFrame({"iban": ["DE1"]})) == "iban"


def test_validate_transactions_does_not_require_account_id():
    canonical = normalize_transactions(
        pd.DataFrame({"date": ["2024-01-01"], "amount": [1.0], "description": ["X"]})
    )
    validate_transactions(canonical.drop(columns=["account_id"]))


def test_two_entities_different_currencies_pass():
    raw = pd.DataFrame(
        {
            "entity_id": ["A", "B"],
            "date": ["2024-01-01", "2024-01-01"],
            "amount": [10.0, 20.0],
            "currency": ["EUR", "USD"],
            "description": ["X", "Y"],
        }
    )
    result = normalize_transactions(raw)
    assert sorted(result["currency"].tolist()) == ["EUR", "USD"]


def test_empty_currency_values_are_ignored():
    raw = pd.DataFrame(
        {
            "entity_id": ["A", "A"],
            "date": ["2024-01-01", "2024-01-02"],
            "amount": [10.0, 20.0],
            "currency": ["EUR", ""],
            "description": ["X", "Y"],
        }
    )
    result = normalize_transactions(raw)
    assert len(result) == 2


def test_mixed_currency_for_one_entity_raises():
    raw = pd.DataFrame(
        {
            "entity_id": ["A", "A", "B"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
            "amount": [10.0, 20.0, 30.0],
            "currency": ["EUR", "USD", "EUR"],
            "description": ["X", "Y", "Z"],
        }
    )
    with pytest.raises(TransactionSchemaError) as exc:
        normalize_transactions(raw)
    message = str(exc.value)
    assert "1 entity(ies) have more than one currency" in message
    assert "EUR" in message and "USD" in message
    assert (
        "Convert amounts to one currency before scoring. No FX conversion is performed." in message
    )


def test_invalid_row_warning_contains_count():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01", "not-a-date", "2024-01-03", "2024-01-04", "2024-01-05"],
            "amount": [10, 20, 30, 40, 50],
            "description": ["OK", "BAD", "C", "D", "E"],
        }
    )
    with pytest.warns(UserWarning) as record:
        normalize_transactions(raw)
    assert any("Dropped 1 of 5" in str(w.message) for w in record)


def test_mixed_iso_and_dayfirst_dates_keep_all_rows():
    raw = pd.DataFrame(
        {
            "date": ["25/03/2024", "2024-03-05", "26/03/2024", "2024-03-06"],
            "amount": [10.0, 20.0, 30.0, 40.0],
            "description": ["A", "B", "C", "D"],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = normalize_transactions(raw)
    assert len(result) == 4
    assert not any("Dropped" in str(w.message) for w in caught)
    dates = dict(zip(result["description"], result["date"]))
    assert dates == {
        "A": pd.Timestamp("2024-03-25"),
        "B": pd.Timestamp("2024-03-05"),
        "C": pd.Timestamp("2024-03-26"),
        "D": pd.Timestamp("2024-03-06"),
    }


def test_all_iso_dates_parse_unchanged():
    raw = pd.DataFrame(
        {
            "date": ["2024-03-05", "2024-03-06", "2024-03-07 12:30:00"],
            "amount": [10.0, 20.0, 30.0],
            "description": ["A", "B", "C"],
        }
    )
    result = normalize_transactions(raw)
    dates = dict(zip(result["description"], result["date"]))
    assert dates == {
        "A": pd.Timestamp("2024-03-05"),
        "B": pd.Timestamp("2024-03-06"),
        "C": pd.Timestamp("2024-03-07 12:30:00"),
    }


def test_all_day_month_dates_keep_order():
    raw = pd.DataFrame(
        {
            "date": ["25/03/2024", "26/03/2024", "27/03/2024"],
            "amount": [10.0, 20.0, 30.0],
            "description": ["A", "B", "C"],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = normalize_transactions(raw)
    assert not any("month-first" in str(w.message) for w in caught)
    dates = dict(zip(result["description"], result["date"]))
    assert dates == {
        "A": pd.Timestamp("2024-03-25"),
        "B": pd.Timestamp("2024-03-26"),
        "C": pd.Timestamp("2024-03-27"),
    }


def test_mixed_iso_and_month_first_dates_keep_all_rows():
    raw = pd.DataFrame(
        {
            "date": ["03/25/2024", "2024-03-26", "03/27/2024"],
            "amount": [10.0, 20.0, 30.0],
            "description": ["A", "B", "C"],
        }
    )
    result = normalize_transactions(raw)
    assert len(result) == 3
    dates = dict(zip(result["description"], result["date"]))
    assert dates == {
        "A": pd.Timestamp("2024-03-25"),
        "B": pd.Timestamp("2024-03-26"),
        "C": pd.Timestamp("2024-03-27"),
    }


def test_blank_company_falls_back_to_account():
    raw = pd.DataFrame(
        {
            "company_id": ["C1", "", float("nan")],
            "account_id": ["A1", "A2", "A3"],
            "date": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "amount": [10.0, 20.0, 30.0],
            "description": ["X", "Y", "Z"],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = normalize_transactions(raw)
    assert not any("company-level" in str(w.message) for w in caught)
    entities = dict(zip(result["description"], result["entity_id"]))
    accounts = dict(zip(result["description"], result["account_id"]))
    assert entities == {"X": "C1", "Y": "A2", "Z": "A3"}
    assert accounts == {"X": "A1", "Y": "A2", "Z": "A3"}


def test_none_values_are_treated_as_blank():
    raw = pd.DataFrame(
        {
            "company_id": ["C1", "", None],
            "account_id": ["A1", None, "A3"],
            "date": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "amount": [10.0, 20.0, 30.0],
            "description": ["X", "Y", "Z"],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = normalize_transactions(raw)
    entities = dict(zip(result["description"], result["entity_id"]))
    accounts = dict(zip(result["description"], result["account_id"]))
    # Row Y has a blank company and a None account: it falls back to the default entity.
    assert entities == {"X": "C1", "Y": "entity_0", "Z": "A3"}
    assert accounts["X"] == "A1" and accounts["Z"] == "A3"
    assert pd.isna(accounts["Y"])
    assert "None" not in set(result["entity_id"]) | set(result["account_id"].dropna())


def test_missing_currency_defaults_with_one_warning():
    raw = pd.DataFrame(
        {
            "entity_id": ["A", "A", "A"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "amount": [10.0, 20.0, 30.0],
            "currency": ["EUR", "", float("nan")],
            "description": ["X", "Y", "Z"],
        }
    )
    with pytest.warns(UserWarning) as record:
        result = normalize_transactions(raw)
    currency_warnings = [w for w in record if "had no currency" in str(w.message)]
    assert len(currency_warnings) == 1
    assert (
        str(currency_warnings[0].message)
        == "2 transaction row(s) had no currency; defaulted to 'EUR'."
    )
    assert result["currency"].tolist() == ["EUR", "EUR", "EUR"]
