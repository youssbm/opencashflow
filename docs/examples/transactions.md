# Working with real transactions

OpenCashFlow expects a table of transactions. It can normalise most bank and aggregator exports, but
explicit column mapping produces the most reliable results.

## Canonical columns

| Column | Required | Notes |
|---|---|---|
| `entity_id` | Recommended | Company or borrower identifier; account-level identifiers go to `account_id` |
| `account_id` | Optional | Always present in the normalised output; NA when the source has no account column |
| `transaction_id` | Optional | Unique row id; generated when absent |
| `date` | Yes | Booking or value date |
| `amount` | Yes | Signed, positive = credit |
| `description` | Yes | Narrative, memo or remittance information |
| `counterparty` | Optional | Merchant, payer or payee |
| `balance` | Optional | Running balance after the transaction |
| `currency` | Optional | ISO code, default EUR |
| `direction` | Optional | `inflow` or `outflow`, derived from the signed amount |
| `category` | Optional | Pre-labelled category; NA triggers categorisation |

## Mapping a CSV

```python
import pandas as pd
from opencashflow.data.schema import normalize_transactions

raw = pd.read_csv("transactions.csv")
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
```

If `amount` is always positive and a separate column says credit/debit, pass
`amount_is_signed=False`. The direction column values may be `credit`/`debit`, `CR`/`DR`,
`inflow`/`outflow` or common numeric indicators.

## Dates

`dayfirst` defaults to `None`, which infers the date order. A first date component above 12 selects
day-first; a second component above 12 selects month-first. If neither applies, an ambiguous non-ISO
date is read month-first and a UserWarning is raised. ISO dates (YYYY-MM-DD) never warn. A column that
mixes ISO and day-first dates parses completely. Pass `dayfirst=True` for EU exports when ambiguous
dates are present.

## Amounts

`parse_amount` applies these separator rules:

* Both "." and "," present: the last one is the decimal separator. `1.234,56` and `1,234.56` both
  give 1234.56.
* One separator kind repeated: every group after the first must have three digits. `1.234.567`
  gives 1234567; `1.234.56` gives NaN.
* One separator once, followed by exactly three digits: thousands, except when the integer part is 0.
  `1.234` gives 1234; `0.001` gives 0.001.
* One separator once, otherwise: decimal. `12.00` gives 12.0; `12,50` gives 12.5.

Currency symbols and accounting parentheses for negatives are also handled.

Rows with an unparseable date or amount are dropped with a warning that gives the count. If more than
25 percent of rows are invalid, normalisation raises a `TransactionSchemaError` instead of dropping
them.

## Identifiers

Company-level identifiers take precedence for `entity_id`. A blank company cell falls back to the
account identifier for that row; `None`, NaN, the empty string, `nan` and `none` all count as blank.
If an export has only account-level identifiers, the account becomes the entity and a UserWarning is
raised.

## Currency

An entity must have one currency. An entity with more than one currency raises a
`TransactionSchemaError`. No FX conversion is performed, so convert amounts to one currency before
normalisation.

## Categorise

```python
from opencashflow.features.categorizer import TransactionCategorizer

categorizer = TransactionCategorizer()
labelled = categorizer.categorize_frame(transactions)
print(labelled["category"].value_counts())
print(categorizer.category_summary(labelled))
```

The default rules cover English, German, French, Polish, Italian, Dutch and Spanish narratives.
If a bank uses unusual wording, add a rule:

```python
from opencashflow.features.categorizer import CategoryRule, TransactionCategorizer

custom = CategoryRule(
    category="payroll",
    patterns=(r"\bSALARIO EMPRESA\b",),
    direction="outflow",
    priority=8,
    confidence=0.95,
    description="Payroll via the local processor",
)
categorizer = TransactionCategorizer(rules=[custom, *TransactionCategorizer().rules])
```

Lower `priority` numbers run first. Keep specific rules before generic ones.

You can also train the character n-gram fallback from labelled rows:

```python
categorizer.fit(labelled_transactions, category_col="category")
```

## Point-in-time features

Do not use future transactions when scoring a historical application:

```python
from opencashflow.features.cashflow_features import CashFlowFeatureExtractor, FeatureConfig

extractor = CashFlowFeatureExtractor(
    config=FeatureConfig(window_days=365, as_of="2024-03-31")
)
features = extractor.fit_transform(transactions)
```

For each entity, only rows on or before the cut-off are used. For production, store the cut-off with
the score so the decision can be reproduced.

## Multiple accounts per borrower

Use the same `entity_id` for every account belonging to the borrower. Internal transfers between
those accounts are categorised as `transfer_in` / `transfer_out` and excluded from revenue and
operating expenses. If your export gives one entity id per account, map accounts to a company id
before calling the extractor.

## Missing balances

If the ledger has no running balance, the extractor derives a cumulative net position and sets
`has_balance_data = 0`. Liquidity features are then weaker proxies. Prefer supplying balances for
production underwriting. Do not interpret a derived cumulative position as an available bank
balance.

## Macro context

```python
from opencashflow.data.macro import MacroFeatureJoiner
import pandas as pd

macro = pd.read_csv("macro.csv")  # country, period, indicator, value
joiner = MacroFeatureJoiner(lags=(0, 3, 12), yoy_periods=12).fit(macro)

features = joiner.transform(
    features,
    countries=entity_country_map,   # Series or dict: entity_id -> "DE"
    as_of=application_dates,        # scalar or Series
)
```

The joiner normalises periods to months, forward-fills within country, and lags values so only
information available at the scoring date is used. Record the macro extraction date and vintage.

## Sanity checks

Before modelling, check:

* `txn_count` and `months_observed` per entity;
* `uncategorized_ratio` (should be low; investigate high values);
* `coverage_ratio` (low values mean little history);
* `has_balance_data`;
* duplicate transactions;
* revenue concentration (`revenue_hhi`, `revenue_top1_share`);
* `nsf_rate` and `returned_payment_rate`;
* whether all amounts share one currency. Cross-currency ledgers need conversion before
  normalisation.
