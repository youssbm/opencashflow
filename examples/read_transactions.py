"""Normalise and categorise a raw transaction export without any external data.

Shows the canonical schema, unsigned amount handling, European number parsing and feature
extraction for two borrowers.

Usage::

    python examples/read_transactions.py
"""

from __future__ import annotations

import pandas as pd

from opencashflow.data.schema import normalize_transactions
from opencashflow.features.cashflow_features import CashFlowFeatureExtractor, FeatureConfig


def build_raw_export() -> pd.DataFrame:
    """A tiny, deliberately awkward export: unsigned amounts and European decimals."""
    rows = []
    balance = 50_000.0
    for month in range(1, 13):
        entries = [
            (25_000.0, "credit", "SEPA GUTSCHRIFT RECHNUNG KUNDE ACME", "Acme GmbH"),
            (12_500.0, "debit", "LOHN MONAT", "Employees"),
            (2_000.0, "debit", "MIETE BUERO BERLIN", "Landlord"),
        ]
        for day, (amount, direction, narrative, counterparty) in enumerate(entries, start=1):
            signed = amount if direction == "credit" else -amount
            balance += signed
            rows.append(
                {
                    "account_number": "DE-001",
                    "booking_date": f"{day:02d}/{month:02d}/2024",
                    "transaction_amount": f"{amount:,.2f}".replace(",", "X")
                    .replace(".", ",")
                    .replace("X", "."),
                    "credit_debit": direction,
                    "narrative": narrative,
                    "merchant_name": counterparty,
                    "running_balance": f"{balance:,.2f}".replace(",", "X")
                    .replace(".", ",")
                    .replace("X", "."),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    raw = build_raw_export()
    transactions = normalize_transactions(
        raw,
        mapping={
            "entity_id": "account_number",
            "date": "booking_date",
            "amount": "transaction_amount",
            "description": "narrative",
            "counterparty": "merchant_name",
            "balance": "running_balance",
            "direction": "credit_debit",
        },
        dayfirst=True,
    )
    print(transactions.head().to_string())
    print("\nDirection counts:", transactions["direction"].value_counts().to_dict())

    from opencashflow.features.categorizer import TransactionCategorizer

    categorizer = TransactionCategorizer()
    categorised = categorizer.categorize_frame(transactions)
    print(
        "\nCategories:",
        categorizer.category_summary(categorised).set_index("category")["count"].to_dict(),
    )

    extractor = CashFlowFeatureExtractor(config=FeatureConfig(window_days=365))
    features = extractor.fit_transform(transactions, as_of="2024-12-31")
    print("\nFeature matrix:", features.shape)
    print(
        features[
            ["revenue_total", "expense_total", "net_margin_proxy", "cashflow_health_score"]
        ].to_string()
    )


if __name__ == "__main__":
    main()
