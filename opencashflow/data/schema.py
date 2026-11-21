"""Canonical transaction schema and input normalisation.

Open-banking exports disagree on almost everything: column names, sign conventions, date formats
and whether amounts are signed. This module converts a wide range of bank and aggregator exports
into one predictable table so that downstream feature code has a single contract.

Canonical columns
-----------------
``entity_id``
    Borrower, company or account identifier. Features are computed per entity.
``transaction_id``
    Unique row identifier. Generated when absent.
``date``
    Booking or value date as ``datetime64[ns]``.
``amount``
    Signed amount in the account currency. Positive is a credit (money in), negative is a debit.
``currency``
    ISO-4217 code, default ``"EUR"``.
``description``
    Free-text narrative, upper-cased and stripped only at categorisation time.
``counterparty``
    Optional counterparty or merchant name.
``balance``
    Optional running balance after the transaction.
``direction``
    ``"inflow"`` or ``"outflow"``, derived from the signed amount.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

Direction = Literal["inflow", "outflow"]

CANONICAL_COLUMNS = [
    "entity_id",
    "transaction_id",
    "date",
    "amount",
    "currency",
    "description",
    "counterparty",
    "balance",
    "direction",
    "category",
]

REQUIRED_COLUMNS = ["entity_id", "date", "amount", "description"]

_ALIASES: dict[str, tuple[str, ...]] = {
    "entity_id": (
        "entity_id",
        "account_id",
        "account",
        "account_number",
        "iban",
        "customer_id",
        "company_id",
        "borrower_id",
        "business_id",
    ),
    "transaction_id": ("transaction_id", "txn_id", "id", "transaction_reference", "reference_id"),
    "date": (
        "date",
        "booking_date",
        "booked_at",
        "value_date",
        "transaction_date",
        "posted_at",
        "timestamp",
        "operation_date",
    ),
    "amount": ("amount", "transaction_amount", "value", "sum", "montant", "betrag", "importe"),
    "currency": ("currency", "currency_code", "ccy", "devise"),
    "description": (
        "description",
        "narrative",
        "memo",
        "details",
        "transaction_description",
        "purpose",
        "remittance_information",
        "reference",
        "libelle",
        "verwendungszweck",
    ),
    "counterparty": (
        "counterparty",
        "counterparty_name",
        "merchant",
        "merchant_name",
        "payee",
        "name",
        "creditor_name",
        "debtor_name",
        "beneficiary",
    ),
    "balance": ("balance", "running_balance", "available_balance", "balance_after", "solde"),
    "direction": (
        "direction",
        "credit_debit",
        "debit_credit",
        "credit_or_debit",
        "type",
        "transaction_type",
        "dc_indicator",
    ),
    "category": ("category", "category_name", "transaction_category", "label"),
}

_INFLOW_TOKENS = {"credit", "cr", "in", "inflow", "deposit", "c", "received", "incoming"}
_OUTFLOW_TOKENS = {"debit", "dr", "out", "outflow", "withdrawal", "d", "paid", "outgoing"}

_CURRENCY_SYMBOLS = "€$£¥₽₹₺₴₦₱₩"


class TransactionSchemaError(ValueError):
    """Raised when a transaction table cannot be normalised safely."""


def _find_column(df: pd.DataFrame, canonical: str, mapping: Mapping[str, str] | None) -> str | None:
    if mapping and canonical in mapping:
        source = mapping[canonical]
        if source not in df.columns:
            raise TransactionSchemaError(
                f"Mapping for {canonical!r} points to missing column {source!r}. "
                f"Available columns: {list(df.columns)}"
            )
        return source
    lower = {str(c).lower().strip(): c for c in df.columns}
    for alias in _ALIASES.get(canonical, ()):  # exact alias wins
        if alias in df.columns:
            return alias
        if alias in lower:
            return lower[alias]
    return None


def parse_amount(value: Any) -> float:
    """Parse a bank amount that may contain symbols, spaces or European separators.

    Handles ``"1,234.56"``, ``"1.234,56"``, ``"-€ 12.50"`` and numeric input. Returns ``nan`` for
    values that cannot be parsed. Parentheses are treated as negative, as in accounting exports.

    Examples:
        >>> parse_amount("€ 1.234,56")
        1234.56
        >>> parse_amount("(42.00)")
        -42.0
    """
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return float("nan")
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    if text.startswith("-") or text.endswith("-"):
        negative = True
    text = text.replace("-", "")
    for symbol in _CURRENCY_SYMBOLS:
        text = text.replace(symbol, "")
    text = re.sub(r"[A-Za-z]{2,}", "", text).strip()
    text = text.replace(" ", "").replace("\u00a0", "")

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
    try:
        result = float(text)
    except ValueError:
        return float("nan")
    return -result if negative else result


def _direction_from_value(value: Any) -> Direction | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in _INFLOW_TOKENS or text.startswith(("credit", "inflow", "deposit")):
        return "inflow"
    if text in _OUTFLOW_TOKENS or text.startswith(("debit", "outflow", "withdraw")):
        return "outflow"
    # Common signed indicator values.
    if text in {"1", "+1"}:
        return "inflow"
    if text in {"-1", "2"}:
        return "outflow"
    return None


def normalize_transactions(
    transactions: pd.DataFrame,
    mapping: Mapping[str, str] | None = None,
    *,
    amount_is_signed: bool | None = None,
    dayfirst: bool = False,
    default_currency: str = "EUR",
    default_entity_id: str = "entity_0",
    drop_invalid: bool = True,
    drop_duplicates: bool = False,
) -> pd.DataFrame:
    """Convert a transaction table into the canonical OpenCashFlow schema.

    Args:
        transactions: Raw transaction table. Must contain a date, an amount and a description;
            an entity/account identifier is strongly recommended.
        mapping: Optional mapping from canonical names to source columns. Aliases such as
            ``account_id``, ``booking_date``, ``narrative`` and ``running_balance`` are detected
            automatically when no mapping is supplied.
        amount_is_signed: ``True`` if positive amounts are credits and negative amounts are debits.
            ``False`` if all amounts are positive and a separate direction column carries the sign.
            ``None`` infers this from the data.
        dayfirst: Parse ambiguous dates as day-first (common in EU bank exports).
        default_currency: Currency assigned when no currency column exists.
        default_entity_id: Entity assigned when no account identifier exists.
        drop_invalid: Drop rows with unparseable dates or amounts instead of raising.
        drop_duplicates: Drop exact duplicate transactions on entity, date, amount, description.

    Returns:
        A new DataFrame with the canonical columns, sorted by entity and date. ``direction`` is
        ``"inflow"`` for positive amounts and ``"outflow"`` for negative amounts.

    Raises:
        TransactionSchemaError: If required columns are missing, the table is empty, or too many
            rows are invalid.
    """
    if not isinstance(transactions, pd.DataFrame):
        raise TransactionSchemaError(
            f"Expected a pandas DataFrame, got {type(transactions).__name__}."
        )
    if transactions.empty:
        raise TransactionSchemaError("Transaction table is empty.")

    df = transactions.copy()
    date_col = _find_column(df, "date", mapping)
    amount_col = _find_column(df, "amount", mapping)
    desc_col = _find_column(df, "description", mapping)
    missing = [
        name
        for name, col in (("date", date_col), ("amount", amount_col), ("description", desc_col))
        if col is None
    ]
    if missing:
        raise TransactionSchemaError(
            f"Missing required transaction column(s): {missing}. "
            f"Available columns: {list(transactions.columns)}. "
            "Pass mapping={'date': ..., 'amount': ..., 'description': ...} if names differ."
        )

    entity_col = _find_column(df, "entity_id", mapping)
    txn_col = _find_column(df, "transaction_id", mapping)
    currency_col = _find_column(df, "currency", mapping)
    counterparty_col = _find_column(df, "counterparty", mapping)
    balance_col = _find_column(df, "balance", mapping)
    direction_col = _find_column(df, "direction", mapping)
    category_col = _find_column(df, "category", mapping)

    out = pd.DataFrame(index=df.index)
    out["entity_id"] = (
        df[entity_col]
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan})
        .fillna(default_entity_id)
        if entity_col
        else default_entity_id
    )
    out["transaction_id"] = (
        df[txn_col].astype(str) if txn_col else [f"txn_{i:08d}" for i in range(len(df))]
    )
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=dayfirst)
    out["amount"] = df[amount_col].map(parse_amount)
    out["currency"] = (
        df[currency_col].astype(str).str.upper().str.strip().replace({"NAN": default_currency})
        if currency_col
        else default_currency
    )
    out["description"] = df[desc_col].fillna("").astype(str).str.strip()
    out["counterparty"] = (
        df[counterparty_col].fillna("").astype(str).str.strip() if counterparty_col else ""
    )
    out["balance"] = df[balance_col].map(parse_amount) if balance_col else np.nan
    # Preserve missing labels as NA so the extractor can categorise unlabelled rows.
    out["category"] = df[category_col] if category_col else pd.NA

    # Resolve the sign convention.
    raw_direction = (
        df[direction_col].map(_direction_from_value)
        if direction_col
        else pd.Series(None, index=df.index)
    )
    if amount_is_signed is None:
        has_negative = bool((out["amount"] < 0).any())
        amount_is_signed = bool(has_negative or direction_col is None)
    if not amount_is_signed:
        sign = raw_direction.map({"inflow": 1.0, "outflow": -1.0})
        if sign.isna().all():
            raise TransactionSchemaError(
                "Amounts are unsigned but no direction column could be interpreted. "
                "Provide mapping={'direction': ...} or signed amounts."
            )
        out["amount"] = out["amount"].abs() * sign.fillna(1.0)
    else:
        # A direction column can still repair missing signs in an otherwise signed export.
        if direction_col is not None and (out["amount"] >= 0).all():
            sign = raw_direction.map({"inflow": 1.0, "outflow": -1.0})
            out["amount"] = out["amount"].abs() * sign.fillna(1.0)

    out["direction"] = np.where(out["amount"] > 0, "inflow", "outflow")
    out["direction"] = pd.Categorical(out["direction"], categories=["inflow", "outflow"])

    invalid_date = out["date"].isna()
    invalid_amount = ~np.isfinite(out["amount"])
    invalid = invalid_date | invalid_amount
    n_invalid = int(invalid.sum())
    if n_invalid:
        share = n_invalid / len(out)
        if not drop_invalid or share > 0.25:
            raise TransactionSchemaError(
                f"{n_invalid} of {len(out)} rows ({share:.1%}) have an invalid date or amount. "
                "Fix the source data or set drop_invalid=True to discard them."
            )
        warnings.warn(
            f"Dropped {n_invalid} of {len(out)} transaction rows with invalid dates or amounts.",
            stacklevel=2,
        )
        out = out.loc[~invalid]
    if out.empty:
        raise TransactionSchemaError("No valid transactions remain after normalisation.")

    if drop_duplicates:
        subset = ["entity_id", "date", "amount", "description"]
        out = out.drop_duplicates(subset=subset, keep="first")

    out = out.sort_values(["entity_id", "date", "transaction_id"], kind="stable").reset_index(
        drop=True
    )
    ordered = [c for c in CANONICAL_COLUMNS if c in out.columns]
    return out.loc[:, ordered]


def validate_transactions(transactions: pd.DataFrame, *, require_balance: bool = False) -> None:
    """Validate an already-normalised transaction table and raise a helpful error if invalid."""
    if not isinstance(transactions, pd.DataFrame):
        raise TransactionSchemaError("Expected a pandas DataFrame.")
    if transactions.empty:
        raise TransactionSchemaError("Transaction table is empty.")
    missing = [c for c in REQUIRED_COLUMNS if c not in transactions.columns]
    if missing:
        raise TransactionSchemaError(
            f"Missing canonical column(s): {missing}. Call normalize_transactions() first."
        )
    if require_balance and (
        "balance" not in transactions.columns or transactions["balance"].isna().all()
    ):
        raise TransactionSchemaError("Balance is required for this operation but was not provided.")
    if transactions["date"].isna().any():
        raise TransactionSchemaError("Canonical date column contains NaT values.")
    amounts = pd.to_numeric(transactions["amount"], errors="coerce")
    if amounts.isna().any() or not np.isfinite(amounts).all():
        raise TransactionSchemaError("Canonical amount column contains non-finite values.")
    if transactions["entity_id"].isna().any():
        raise TransactionSchemaError("Canonical entity_id column contains missing values.")
    if (amounts == 0).all():
        warnings.warn(
            "All transaction amounts are zero; features will be degenerate.", stacklevel=2
        )


def infer_entity_column(df: pd.DataFrame) -> str | None:
    """Return the most likely entity/account column name, or ``None``."""
    return _find_column(df, "entity_id", None)


def entity_transaction_counts(transactions: pd.DataFrame) -> pd.Series:
    """Number of transactions per entity, sorted descending."""
    return transactions.groupby("entity_id", observed=True).size().sort_values(ascending=False)
