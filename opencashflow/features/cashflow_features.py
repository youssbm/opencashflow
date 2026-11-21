"""Cash-flow feature engineering for SME credit underwriting.

The extractor turns a transaction ledger into one row of credit-relevant features per borrower.
It is designed for open-banking style data: signed amounts, free-text narratives, optional running
balances, and multiple accounts per company.

The feature set covers revenue quality, expense structure, liquidity, debt service, distress
events, behaviour and trends. Every feature is registered in
:mod:`opencashflow.features.specs`, so the catalogue, docs and model inputs stay aligned.

Example:
    >>> import pandas as pd
    >>> from opencashflow.data.synthetic import generate_company_transactions
    >>> from opencashflow.features.cashflow_features import CashFlowFeatureExtractor
    >>> tx = generate_company_transactions(months=18, seed=7)
    >>> features = CashFlowFeatureExtractor().fit_transform(tx)
    >>> features.shape[1] >= 100
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from opencashflow.data.schema import normalize_transactions, validate_transactions
from opencashflow.features.categorizer import (
    CATEGORY_DESCRIPTIONS,
    EXPENSE_CATEGORIES,
    OUTFLOW_CATEGORIES,
    TransactionCategorizer,
    normalize_text,
)
from opencashflow.features.ratios import (
    coefficient_of_variation,
    hhi,
    max_drawdown,
    normalized_entropy,
    normalized_slope,
    safe_div,
    seasonal_strength,
    stability_score,
    trend_ratio,
)
from opencashflow.features.specs import (
    FEATURE_NAMES,
    FEATURE_SPECS,
    FeatureSpec,
    feature_specs_frame,
)

DEFAULT_REVENUE_CATEGORIES: tuple[str, ...] = ("revenue",)
DEFAULT_EXPENSE_CATEGORIES: tuple[str, ...] = tuple(
    sorted(EXPENSE_CATEGORIES - {"interest_expense"})
)


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for :class:`CashFlowFeatureExtractor`.

    Args:
        window_days: Look-back window in calendar days.
        min_transactions: Minimum transactions required to score an entity. Entities below the
            threshold are still returned by default; set ``drop_entities_below_min=True`` to omit.
        as_of: Point-in-time cut-off. ``None`` uses each entity's latest transaction date, which is
            convenient for research but leaks future information if labels span a period. Pass a
            fixed date for production or out-of-time validation.
        include_other_inflow_as_revenue: Treat unmatched credits as revenue. Useful for raw bank
            exports where customer receipts lack invoice keywords; risky when internal transfers
            are common.
        seasonality_period: Calendar period for seasonality features, normally 12 months.
        clip_value: Absolute cap applied to every output feature to keep models numerically stable.
        drop_entities_below_min: Drop entities with fewer than ``min_transactions`` rows.
        revenue_categories: Category keys treated as revenue.
        expense_categories: Category keys treated as operating expenses (debt principal excluded).
    """

    window_days: int = 365
    min_transactions: int = 1
    as_of: str | pd.Timestamp | None = None
    include_other_inflow_as_revenue: bool = False
    seasonality_period: int = 12
    clip_value: float = 1e9
    drop_entities_below_min: bool = False
    revenue_categories: tuple[str, ...] = DEFAULT_REVENUE_CATEGORIES
    expense_categories: tuple[str, ...] = DEFAULT_EXPENSE_CATEGORIES

    def __post_init__(self) -> None:
        if self.window_days < 30:
            raise ValueError("window_days must be at least 30.")
        if self.min_transactions < 1:
            raise ValueError("min_transactions must be at least 1.")
        if self.seasonality_period < 2:
            raise ValueError("seasonality_period must be at least 2.")
        unknown = set(self.revenue_categories) | set(self.expense_categories)
        unknown -= set(CATEGORY_DESCRIPTIONS)
        if unknown:
            raise ValueError(f"Unknown category keys in FeatureConfig: {sorted(unknown)}")

    @property
    def revenue_keys(self) -> tuple[str, ...]:
        keys = set(self.revenue_categories)
        if self.include_other_inflow_as_revenue:
            keys.add("other_inflow")
        return tuple(sorted(keys))

    @property
    def expense_keys(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.expense_categories) | {"interest_expense"}))


@dataclass
class _EntityView:
    """Internal, per-entity pre-aggregation used by the feature functions."""

    tx: pd.DataFrame
    monthly: pd.DataFrame
    daily_balance: pd.Series
    as_of: pd.Timestamp
    start: pd.Timestamp
    days_observed: int
    has_balance_data: bool
    revenue_total: float
    outflow_total: float
    inflow_total: float
    expense_total: float
    months: pd.PeriodIndex = field(repr=False)

    def mcat(self, category: str) -> pd.Series:
        col = f"cat_{category}"
        if col in self.monthly.columns:
            return self.monthly[col].astype(float)
        return pd.Series(0.0, index=self.monthly.index, dtype=float)


class FeatureBuilder:
    """Collects feature values and enforces that every value is declared in the catalogue."""

    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def add(self, name: str, value: Any) -> None:
        if name not in FEATURE_SPECS:
            raise KeyError(f"Feature {name!r} is not declared in FEATURE_SPECS.")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        if not np.isfinite(numeric):
            numeric = 0.0
        self.values[name] = numeric

    def to_series(self) -> pd.Series:
        missing = [name for name in FEATURE_NAMES if name not in self.values]
        if missing:
            raise RuntimeError(f"Feature computation is incomplete; missing: {missing}")
        return pd.Series(self.values, dtype=float).reindex(FEATURE_NAMES)


def _as_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value).normalize()


def _counterparty_keys(df: pd.DataFrame) -> pd.Series:
    """Counterparty identifier, falling back to the first tokens of the narrative."""
    if df.empty:
        return pd.Series(dtype="object")
    cp = df["counterparty"].fillna("").astype(str).str.strip()
    if (cp != "").any():
        return cp.where(cp != "", "UNKNOWN")
    text = df["description"].fillna("").map(normalize_text)
    keys = text.str.split().str[:3].str.join(" ")
    return keys.replace("", "UNKNOWN")


def _known_keys(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    keys = _counterparty_keys(df)
    mask = keys.ne("UNKNOWN") & keys.ne("")
    return keys, df.loc[mask]


def _concentration(df: pd.DataFrame, amount_col: str) -> float:
    keys, known = _known_keys(df)
    if known.empty:
        return 0.0
    grouped = known.assign(_key=keys.loc[known.index]).groupby("_key")[amount_col].sum()
    return hhi(grouped.to_numpy())


def _top_share(df: pd.DataFrame, amount_col: str, k: int) -> float:
    keys, known = _known_keys(df)
    if known.empty:
        return 0.0
    grouped = (
        known.assign(_key=keys.loc[known.index])
        .groupby("_key")[amount_col]
        .sum()
        .sort_values(ascending=False)
    )
    total = float(grouped.sum())
    return safe_div(float(grouped.head(k).sum()), total, default=0.0)


def _diversity(df: pd.DataFrame, amount_col: str) -> float:
    keys, known = _known_keys(df)
    if known.empty:
        return 0.0
    grouped = known.assign(_key=keys.loc[known.index]).groupby("_key")[amount_col].sum()
    return normalized_entropy(grouped.to_numpy())


def _counterparty_count(df: pd.DataFrame) -> int:
    keys, known = _known_keys(df)
    return int(keys.loc[known.index].nunique()) if not known.empty else 0


def _growth(series: pd.Series, periods: int) -> float:
    """Recent ``periods`` mean versus the preceding ``periods`` mean."""
    arr = series.to_numpy(dtype=float)
    if arr.size < 2 * periods:
        return 0.0
    recent = float(np.mean(arr[-periods:]))
    prior = float(np.mean(arr[-2 * periods : -periods]))
    return trend_ratio(recent, prior)


def _consecutive_negative(series: pd.Series) -> int:
    longest = current = 0
    for value in series.to_numpy(dtype=float):
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _monthly_category_frame(tx: pd.DataFrame, months: pd.PeriodIndex) -> pd.DataFrame:
    """Aggregate each category into a complete monthly frame.

    Outflow categories are stored as positive expense amounts; inflow categories stay positive
    credits. ``end_balance`` is the last observed balance in the month, or cumulative net flow when
    the source has no balances.
    """
    work = tx.copy()
    work["month"] = work["date"].dt.to_period("M")
    pivot = work.pivot_table(
        index="month", columns="category", values="amount", aggfunc="sum", fill_value=0.0
    )
    monthly = pd.DataFrame(index=months)
    for category in CATEGORY_DESCRIPTIONS:
        if category not in pivot.columns:
            monthly[f"cat_{category}"] = 0.0
        elif category in OUTFLOW_CATEGORIES:
            monthly[f"cat_{category}"] = -pivot[category].reindex(months).fillna(0.0)
        else:
            monthly[f"cat_{category}"] = pivot[category].reindex(months).fillna(0.0)

    positive = work.loc[work["amount"] > 0].groupby("month")["amount"].sum()
    negative = work.loc[work["amount"] < 0].groupby("month")["amount"].sum().abs()
    monthly["inflow"] = positive.reindex(months).fillna(0.0)
    monthly["outflow"] = negative.reindex(months).fillna(0.0)
    monthly["net"] = monthly["inflow"] - monthly["outflow"]
    monthly["txn_count"] = work.groupby("month").size().reindex(months).fillna(0).astype(float)

    balances = work.dropna(subset=["balance"])
    if not balances.empty:
        end_balance = balances.groupby("month")["balance"].last().reindex(months)
        monthly["end_balance"] = end_balance.ffill().bfill()
    else:
        monthly["end_balance"] = (
            work.groupby("month")["amount"].sum().reindex(months).fillna(0.0).cumsum()
        )
    monthly["end_balance"] = monthly["end_balance"].fillna(0.0)
    return monthly


def _daily_balance_series(tx: pd.DataFrame, start: pd.Timestamp, as_of: pd.Timestamp) -> pd.Series:
    """Daily balance, carried forward. Uses observed balances when available."""
    index = pd.date_range(start=start, end=as_of, freq="D")
    balances = tx.dropna(subset=["balance"])
    if not balances.empty:
        observed = balances.set_index("date")["balance"].groupby(level=0).last().sort_index()
        series = observed.reindex(index).ffill().bfill()
        return series.fillna(0.0)
    daily_net = tx.set_index("date")["amount"].groupby(level=0).sum().sort_index()
    series = daily_net.reindex(index).fillna(0.0).cumsum()
    return series


class CashFlowFeatureExtractor:
    """Turn raw transactions into one feature row per entity.

    Args:
        config: Feature configuration. Defaults to a 365-day window and per-entity as-of dates.
        categorizer: Custom categoriser. A default rule-based categoriser is created otherwise.

    Attributes:
        feature_names_in_: Ordered feature names produced by :meth:`transform`.
        categorizer_: The categoriser used, including any trained ML fallback.
        n_features_in_: Number of features.

    Example:
        >>> from opencashflow.features.cashflow_features import CashFlowFeatureExtractor
        >>> extractor = CashFlowFeatureExtractor().fit(transactions)
        >>> X = extractor.transform(transactions)
    """

    def __init__(
        self,
        config: FeatureConfig | None = None,
        categorizer: TransactionCategorizer | None = None,
    ) -> None:
        self.config = config or FeatureConfig()
        self.categorizer = categorizer or TransactionCategorizer()
        self.categorizer_: TransactionCategorizer | None = None
        self.feature_names_in_: list[str] = list(FEATURE_NAMES)
        self.n_features_in_: int = len(FEATURE_NAMES)
        self.entities_: list[str] = []

    # ------------------------------------------------------------------ public API
    def fit(self, transactions: pd.DataFrame, y: Any = None) -> CashFlowFeatureExtractor:
        """Validate the input and fit the categoriser. ``y`` is accepted for sklearn compatibility."""
        del y
        prepared = self._prepare(transactions)
        self.entities_ = sorted(prepared["entity_id"].astype(str).unique().tolist())
        self.categorizer_ = self.categorizer
        return self

    def transform(
        self,
        transactions: pd.DataFrame,
        as_of: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Compute features for every entity in ``transactions``.

        Args:
            transactions: Raw or canonical transactions. Normalisation and categorisation are
                applied automatically.
            as_of: Optional override for the cut-off date. Use this for point-in-time scoring.

        Returns:
            DataFrame indexed by ``entity_id`` with one column per feature in
            :data:`opencashflow.features.specs.FEATURE_NAMES`. Values are finite floats.
        """
        prepared = self._prepare(transactions)
        override = _as_timestamp(as_of)
        rows: dict[str, pd.Series] = {}
        for entity_id, group in prepared.groupby("entity_id", observed=True, sort=True):
            rows[str(entity_id)] = self._compute_entity(group, override)
        if not rows:
            return pd.DataFrame(columns=FEATURE_NAMES, dtype=float)
        out = pd.DataFrame.from_dict(rows, orient="index")
        out.index.name = "entity_id"
        out = out.reindex(columns=FEATURE_NAMES)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out = out.clip(-self.config.clip_value, self.config.clip_value)
        return out.astype(float)

    def fit_transform(
        self, transactions: pd.DataFrame, y: Any = None, **kwargs: Any
    ) -> pd.DataFrame:
        """Fit and transform in one call."""
        return self.fit(transactions, y).transform(transactions, **kwargs)

    def transform_one(
        self, transactions: pd.DataFrame, as_of: str | pd.Timestamp | None = None
    ) -> pd.Series:
        """Compute a single feature row. The table must contain one entity."""
        features = self.transform(transactions, as_of=as_of)
        if len(features) != 1:
            raise ValueError(f"Expected one entity, found {len(features)}.")
        return features.iloc[0]

    def get_feature_names_out(self) -> np.ndarray:
        """Return the ordered feature names as a NumPy array."""
        return np.asarray(FEATURE_NAMES, dtype=object)

    @staticmethod
    def describe_features() -> pd.DataFrame:
        """Return the full feature dictionary as a DataFrame."""
        return feature_specs_frame()

    @staticmethod
    def feature_spec(name: str) -> FeatureSpec:
        """Look up one feature definition."""
        return FEATURE_SPECS[name]

    # ------------------------------------------------------------------ preparation
    def _prepare(self, transactions: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(transactions, pd.DataFrame):
            raise TypeError("transactions must be a pandas DataFrame.")
        prepared = normalize_transactions(transactions)
        validate_transactions(prepared)
        if "category" not in prepared.columns or prepared["category"].isna().all():
            prepared = self.categorizer.categorize_frame(prepared, overwrite=True)
        elif prepared["category"].isna().any():
            prepared = self.categorizer.categorize_frame(prepared, overwrite=False)
        prepared["category"] = prepared["category"].astype(str)
        prepared["date"] = pd.to_datetime(prepared["date"]).dt.normalize()
        prepared = prepared.sort_values(["entity_id", "date", "transaction_id"], kind="stable")
        return prepared.reset_index(drop=True)

    # ------------------------------------------------------------------ per entity
    def _compute_entity(self, tx: pd.DataFrame, override: pd.Timestamp | None) -> pd.Series:
        config = self.config
        tx = tx.sort_values(["date", "transaction_id"], kind="stable").copy()
        configured_as_of = (
            _as_timestamp(config.as_of) if config.as_of not in (None, "max") else None
        )
        as_of = override or configured_as_of or tx["date"].max()
        as_of = pd.Timestamp(as_of).normalize()
        tx = tx.loc[tx["date"] <= as_of]
        if tx.empty:
            return pd.Series(0.0, index=FEATURE_NAMES, dtype=float)

        first_date = tx["date"].min()
        window_start = as_of - pd.Timedelta(days=config.window_days - 1)
        start = max(window_start, first_date)
        tx = tx.loc[tx["date"] >= start].copy()
        if tx.empty:
            return pd.Series(0.0, index=FEATURE_NAMES, dtype=float)

        days_observed = max(int((as_of - start).days) + 1, 1)
        months = pd.period_range(start=start.to_period("M"), end=as_of.to_period("M"), freq="M")
        has_balance = bool(tx["balance"].notna().any())

        monthly = _monthly_category_frame(tx, months)
        daily_balance = _daily_balance_series(tx, start, as_of)

        revenue_keys = set(config.revenue_keys)
        expense_keys = set(config.expense_keys)
        revenue_monthly = monthly[[f"cat_{k}" for k in revenue_keys if f"cat_{k}" in monthly]].sum(
            axis=1
        )
        expense_monthly = monthly[[f"cat_{k}" for k in expense_keys if f"cat_{k}" in monthly]].sum(
            axis=1
        )
        revenue_total = float(revenue_monthly.sum())
        expense_total = float(expense_monthly.sum())
        inflow_total = float(monthly["inflow"].sum())
        outflow_total = float(monthly["outflow"].sum())

        view = _EntityView(
            tx=tx,
            monthly=monthly,
            daily_balance=daily_balance,
            as_of=as_of,
            start=start,
            days_observed=days_observed,
            has_balance_data=has_balance,
            revenue_total=revenue_total,
            outflow_total=outflow_total,
            inflow_total=inflow_total,
            expense_total=expense_total,
            months=months,
        )
        return self._features(view)

    # ------------------------------------------------------------------ feature functions
    def _features(self, view: _EntityView) -> pd.Series:
        b = FeatureBuilder()
        tx = view.tx
        monthly = view.monthly
        revenue = monthly[
            [f"cat_{k}" for k in self.config.revenue_keys if f"cat_{k}" in monthly]
        ].sum(axis=1)
        expense = monthly[
            [f"cat_{k}" for k in self.config.expense_keys if f"cat_{k}" in monthly]
        ].sum(axis=1)
        net = monthly["net"]
        balance = view.daily_balance
        months = max(len(monthly), 1)
        days = max(view.days_observed, 1)
        revenue_total = view.revenue_total
        expense_total = view.expense_total
        inflow_total = view.inflow_total
        outflow_total = view.outflow_total
        total_abs_flow = float(tx["amount"].abs().sum())

        abs_amounts = tx["amount"].abs()
        active_days = int(tx["date"].nunique())
        uncategorized = (
            (~tx["category"].isin(CATEGORY_DESCRIPTIONS))
            | tx["category"].isin(["other_inflow", "other_outflow", "uncategorized"])
        ).mean()

        # Coverage ---------------------------------------------------------------
        b.add("txn_count", len(tx))
        b.add("txn_count_monthly_avg", safe_div(len(tx), months))
        b.add("months_observed", months)
        b.add("days_observed", view.days_observed)
        b.add("coverage_ratio", safe_div(view.days_observed, self.config.window_days))
        b.add("active_days_ratio", safe_div(active_days, days))
        b.add("txn_amount_mean", abs_amounts.mean() if len(tx) else 0.0)
        b.add("txn_amount_median", abs_amounts.median() if len(tx) else 0.0)
        b.add("txn_amount_std", abs_amounts.std(ddof=0) if len(tx) > 1 else 0.0)
        b.add("txn_amount_cv", coefficient_of_variation(abs_amounts))
        b.add("uncategorized_ratio", uncategorized)
        b.add("has_balance_data", int(view.has_balance_data))

        # Revenue ---------------------------------------------------------------
        revenue_tx = tx.loc[
            tx["category"].isin(self.config.revenue_keys) & (tx["amount"] > 0)
        ].copy()
        revenue_monthly = revenue
        b.add("revenue_total", revenue_total)
        b.add("revenue_monthly_mean", revenue_monthly.mean())
        b.add("revenue_monthly_median", revenue_monthly.median())
        b.add("revenue_monthly_std", revenue_monthly.std(ddof=0))
        b.add("revenue_cv", coefficient_of_variation(revenue_monthly))
        b.add("revenue_growth_3m", _growth(revenue_monthly, 3))
        b.add("revenue_growth_6m", _growth(revenue_monthly, 6))
        b.add("revenue_trend_slope", normalized_slope(revenue_monthly))
        last_mean = float(revenue_monthly.iloc[-min(3, len(revenue_monthly)) :].mean())
        b.add("revenue_momentum", trend_ratio(float(revenue_monthly.iloc[-1]), last_mean))
        b.add("revenue_positive_month_ratio", float((revenue_monthly > 0).mean()))
        b.add("revenue_min_monthly", revenue_monthly.min())
        b.add("revenue_max_monthly", revenue_monthly.max())
        b.add("revenue_txn_count", len(revenue_tx))
        b.add("revenue_txn_mean", revenue_tx["amount"].mean() if len(revenue_tx) else 0.0)
        b.add(
            "revenue_seasonality_strength",
            seasonal_strength(revenue_monthly, self.config.seasonality_period),
        )
        b.add(
            "revenue_seasonality_peak_month",
            _peak_month(revenue_monthly)
            if len(revenue_monthly) >= 2 * self.config.seasonality_period
            else 0,
        )
        b.add("revenue_hhi", _concentration(revenue_tx, "amount"))
        b.add("revenue_top1_share", _top_share(revenue_tx, "amount", 1))
        b.add("revenue_top3_share", _top_share(revenue_tx, "amount", 3))
        b.add("revenue_entropy", _diversity(revenue_tx, "amount"))
        b.add("revenue_counterparty_count", _counterparty_count(revenue_tx))
        growth_series = revenue_monthly.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        b.add(
            "revenue_growth_volatility",
            growth_series.std(ddof=0) if len(growth_series) > 1 else 0.0,
        )
        b.add("revenue_last_month_share", safe_div(float(revenue_monthly.iloc[-1]), revenue_total))
        b.add("inflow_quality", safe_div(revenue_total, inflow_total))

        # Expenses --------------------------------------------------------------
        payroll = view.mcat("payroll")
        rent = view.mcat("rent")
        utilities = view.mcat("utilities")
        tax = view.mcat("tax")
        vat = view.mcat("vat")
        supplier = view.mcat("supplier")
        interest = view.mcat("interest_expense")
        fees = view.mcat("bank_fees")
        insurance = view.mcat("insurance")
        card = view.mcat("card_payment")
        cash = view.mcat("cash_withdrawal")
        gambling = view.mcat("gambling")
        loan_repayment = view.mcat("loan_repayment")
        loan_proceeds = view.mcat("loan_proceeds")
        refunds = view.mcat("refund")
        transfer_in = view.mcat("transfer_in")
        transfer_out = view.mcat("transfer_out")

        expense_tx = tx.loc[
            tx["category"].isin(self.config.expense_keys) & (tx["amount"] < 0)
        ].copy()
        expense_tx["exp_amount"] = -expense_tx["amount"]

        b.add("expense_total", expense_total)
        b.add("expense_monthly_mean", expense.mean())
        b.add("expense_monthly_std", expense.std(ddof=0))
        b.add("expense_cv", coefficient_of_variation(expense))
        b.add("expense_growth_3m", _growth(expense, 3))
        b.add("expense_to_revenue", safe_div(expense_total, revenue_total))
        b.add("payroll_total", payroll.sum())
        b.add("payroll_monthly_mean", payroll.mean())
        b.add("payroll_to_revenue", safe_div(float(payroll.sum()), revenue_total))
        active_payroll = payroll[payroll > 0]
        b.add("payroll_cv", coefficient_of_variation(active_payroll))
        b.add("payroll_regularity", stability_score(active_payroll))
        b.add("payroll_months_ratio", float((payroll > 0).mean()))
        b.add("rent_total", rent.sum())
        b.add("rent_to_revenue", safe_div(float(rent.sum()), revenue_total))
        b.add("rent_to_expense", safe_div(float(rent.sum()), expense_total))
        b.add("tax_total", tax.sum())
        b.add("tax_to_revenue", safe_div(float(tax.sum()), revenue_total))
        b.add("vat_total", vat.sum())
        b.add("vat_to_revenue", safe_div(float(vat.sum()), revenue_total))
        b.add("supplier_total", supplier.sum())
        b.add("supplier_to_revenue", safe_div(float(supplier.sum()), revenue_total))
        b.add("supplier_to_expense", safe_div(float(supplier.sum()), expense_total))
        b.add("utilities_total", utilities.sum())
        b.add("utilities_to_revenue", safe_div(float(utilities.sum()), revenue_total))
        fixed = payroll.sum() + rent.sum() + utilities.sum() + insurance.sum() + interest.sum()
        b.add("fixed_cost_ratio", safe_div(float(fixed), revenue_total))
        discretionary = card.sum() + cash.sum() + gambling.sum()
        b.add("discretionary_ratio", safe_div(float(discretionary), expense_total))
        b.add("expense_txn_count", len(expense_tx))
        b.add("expense_txn_mean", expense_tx["exp_amount"].mean() if len(expense_tx) else 0.0)
        b.add(
            "largest_expense_share",
            safe_div(
                float(expense_tx["exp_amount"].max()) if len(expense_tx) else 0.0, expense_total
            ),
        )
        b.add("expense_hhi", _concentration(expense_tx, "exp_amount"))
        supplier_tx = expense_tx.loc[expense_tx["category"] == "supplier"]
        b.add("supplier_count", _counterparty_count(supplier_tx))
        b.add("supplier_concentration", _concentration(supplier_tx, "exp_amount"))

        # Liquidity -------------------------------------------------------------
        monthly_balance = monthly["end_balance"].astype(float)
        b.add("balance_mean", balance.mean())
        b.add("balance_median", balance.median())
        b.add("balance_min", balance.min())
        b.add("balance_max", balance.max())
        b.add("balance_std", balance.std(ddof=0))
        b.add("balance_cv", coefficient_of_variation(monthly_balance))
        b.add("balance_trend_slope", normalized_slope(monthly_balance))
        b.add("balance_p10", float(balance.quantile(0.10)))
        b.add("balance_p90", float(balance.quantile(0.90)))
        b.add("balance_last", float(balance.iloc[-1]))
        last90 = balance.loc[balance.index >= view.as_of - pd.Timedelta(days=89)]
        b.add("balance_last_90d_mean", last90.mean() if len(last90) else float(balance.iloc[-1]))
        b.add("balance_min_last_90d", last90.min() if len(last90) else float(balance.min()))
        b.add("balance_drawdown_max", max_drawdown(balance))
        tx_balance = balance.reindex(pd.DatetimeIndex(tx["date"])).to_numpy(dtype=float)
        negative_days = int(np.sum(tx_balance < 0))
        b.add("negative_balance_days", negative_days)
        b.add("negative_balance_ratio", safe_div(negative_days, active_days))
        overdraft_days = int((balance < 0).sum())
        b.add("overdraft_days", overdraft_days)
        b.add("overdraft_ratio", safe_div(overdraft_days, days))
        b.add("days_cash", safe_div(float(balance.mean()), safe_div(expense_total, days)))
        b.add("runway_months", safe_div(float(balance.iloc[-1]), float(expense.mean())))
        b.add("liquidity_ratio", safe_div(inflow_total, outflow_total))
        net_total = inflow_total - outflow_total
        b.add("net_cash_flow_total", net_total)
        b.add("net_cash_flow_monthly_mean", net.mean())
        b.add("net_cash_flow_cv", coefficient_of_variation(net))
        negative_months = int((net < 0).sum())
        b.add("net_cash_flow_negative_months", negative_months)
        b.add("net_cash_flow_negative_ratio", safe_div(negative_months, months))
        b.add("consecutive_negative_months_max", _consecutive_negative(net))
        b.add("cash_flow_volatility", safe_div(float(net.std(ddof=0)), abs(float(revenue.mean()))))
        b.add("burn_rate", safe_div(max(0.0, -float(net.mean())), float(revenue.mean())))
        b.add("cash_buffer_months", safe_div(float(balance.quantile(0.10)), float(expense.mean())))
        b.add("net_margin_proxy", safe_div(revenue_total - expense_total, revenue_total))

        # Debt ------------------------------------------------------------------
        repayment_total = float(loan_repayment.sum())
        interest_total = float(interest.sum())
        proceeds_total = float(loan_proceeds.sum())
        cash_available = revenue_total - (expense_total - interest_total)
        b.add("loan_repayment_total", repayment_total)
        b.add("loan_repayment_monthly_mean", loan_repayment.mean())
        b.add("debt_service_ratio", safe_div(repayment_total, revenue_total))
        dscr = (
            10.0
            if repayment_total <= 0
            else min(max(safe_div(cash_available, repayment_total), 0.0), 10.0)
        )
        b.add("dscr", dscr)
        b.add("interest_total", interest_total)
        b.add("interest_to_revenue", safe_div(interest_total, revenue_total))
        interest_coverage = (
            20.0
            if interest_total <= 0
            else min(max(safe_div(cash_available, interest_total), 0.0), 20.0)
        )
        b.add("interest_coverage", interest_coverage)
        b.add("loan_proceeds_total", proceeds_total)
        b.add("loan_to_revenue", safe_div(proceeds_total, revenue_total))
        active_debt = loan_repayment[loan_repayment > 0]
        b.add("debt_service_months_ratio", float((loan_repayment > 0).mean()))
        b.add("debt_service_cv", coefficient_of_variation(active_debt))
        b.add("net_debt_flow", proceeds_total - repayment_total)

        # Distress --------------------------------------------------------------
        nsf_count = int((tx["category"] == "nsf").sum())
        b.add("nsf_count", nsf_count)
        b.add("nsf_rate", safe_div(nsf_count, len(tx)))
        returned = tx.loc[
            (tx["category"] == "nsf")
            & tx["description"]
            .map(normalize_text)
            .str.contains(r"RETURN|REJECT|DISHONOU?R|FAILED|UNPAID", regex=True)
        ]
        b.add("returned_payment_count", len(returned))
        b.add("returned_payment_rate", safe_div(len(returned), len(tx)))
        b.add("bank_fees_total", fees.sum())
        b.add("bank_fees_to_revenue", safe_div(float(fees.sum()), revenue_total))
        b.add("gambling_total", gambling.sum())
        b.add("gambling_to_expense", safe_div(float(gambling.sum()), expense_total))
        b.add("cash_withdrawal_total", cash.sum())
        b.add("cash_withdrawal_ratio", safe_div(float(cash.sum()), expense_total))
        b.add("card_payment_total", card.sum())
        b.add("card_payment_ratio", safe_div(float(card.sum()), expense_total))
        b.add("insurance_total", insurance.sum())
        b.add("insurance_to_revenue", safe_div(float(insurance.sum()), revenue_total))
        positive_balance = (balance >= 0).astype(int)
        events = int(((positive_balance.shift(1) == 1) & (positive_balance == 0)).sum())
        b.add("overdraft_events", events)

        revenue_growth_3m = _growth(revenue, 3)
        distress_flags = [
            nsf_count > 0,
            len(returned) > 0,
            overdraft_days > 0,
            safe_div(negative_months, months) > 0.5,
            revenue_growth_3m < -0.2,
            safe_div(float(gambling.sum()), expense_total) > 0.05,
            float(balance.min()) < 0,
            safe_div(repayment_total, revenue_total) > 0.3,
        ]
        b.add("distress_flag_count", sum(bool(f) for f in distress_flags))

        # Behaviour -------------------------------------------------------------
        amounts = abs_amounts
        b.add("round_amount_ratio", float(((amounts > 0) & (amounts % 100 == 0)).mean()))
        b.add("weekend_ratio", float((tx["date"].dt.dayofweek >= 5).mean()))
        b.add("month_end_ratio", float((tx["date"].dt.day >= 28).mean()))
        payroll_tx = tx.loc[tx["category"] == "payroll"]
        if len(payroll_tx) >= 3:
            day_std = float(payroll_tx["date"].dt.day.std(ddof=0))
            b.add("salary_day_consistency", 1.0 - min(day_std / 10.0, 1.0))
        else:
            b.add("salary_day_consistency", 0.0)
        if len(tx) >= 2:
            gaps = tx["date"].sort_values().diff().dt.days.dropna()
            b.add("dormant_days_max", gaps.max() if len(gaps) else view.days_observed)
        else:
            b.add("dormant_days_max", view.days_observed)
        b.add("txn_frequency_trend", normalized_slope(monthly["txn_count"]))
        b.add("inflow_outflow_correlation", _corr(monthly["inflow"], monthly["outflow"]))
        b.add(
            "internal_transfer_ratio",
            safe_div(float(transfer_in.sum() + transfer_out.sum()), total_abs_flow),
        )
        b.add("refund_ratio", safe_div(float(refunds.sum()), inflow_total))
        b.add("outflow_quality", safe_div(expense_total, outflow_total))
        b.add("recurring_counterparty_ratio", _recurring_share(expense_tx, expense_total))

        # Trends ----------------------------------------------------------------
        b.add("inflow_growth_3m", _growth(monthly["inflow"], 3))
        b.add("outflow_growth_3m", _growth(monthly["outflow"], 3))
        b.add("net_growth_3m", _growth(net, 3))
        b.add("inflow_trend_slope", normalized_slope(monthly["inflow"]))
        b.add("outflow_trend_slope", normalized_slope(monthly["outflow"]))
        b.add("net_trend_slope", normalized_slope(net))
        b.add(
            "seasonality_strength_expense",
            seasonal_strength(expense, self.config.seasonality_period),
        )
        b.add("volatility_ratio", _volatility_ratio(expense, revenue))
        b.add("revenue_expense_correlation", _corr(revenue, expense))

        # Composite -------------------------------------------------------------
        revenue_cv = coefficient_of_variation(revenue)
        net_negative_ratio = safe_div(negative_months, months)
        balance_cv = coefficient_of_variation(monthly_balance)
        stability = 100.0 * (
            0.30 * (1.0 - min(max(revenue_cv, 0.0), 1.0))
            + 0.20 * float((revenue > 0).mean())
            + 0.20 * (1.0 - min(max(net_negative_ratio, 0.0), 1.0))
            + 0.15 * (1.0 - min(max(balance_cv, 0.0), 2.0) / 2.0 if view.has_balance_data else 0.5)
            + 0.15 * stability_score(active_payroll)
        )
        b.add("cashflow_stability_score", stability)

        days_cash = safe_div(float(balance.mean()), safe_div(expense_total, days))
        runway = safe_div(float(balance.iloc[-1]), float(expense.mean()))
        liquidity_score = 100.0 * (
            0.35 * min(max(days_cash / 90.0, 0.0), 1.0)
            + 0.30 * min(max(runway / 6.0, 0.0), 1.0)
            + 0.20 * (1.0 - min(max(safe_div(overdraft_days, days), 0.0), 1.0))
            + 0.15 * (1.0 - min(max(safe_div(negative_days, active_days), 0.0), 1.0))
        )
        b.add("liquidity_health_score", liquidity_score)

        dsr = safe_div(repayment_total, revenue_total)
        leverage_score = 100.0 * (
            0.40 * (1.0 - min(max(dsr / 0.5, 0.0), 1.0))
            + 0.40 * min(max((dscr - 1.0) / 2.0, 0.0), 1.0)
            + 0.20 * min(max((interest_coverage - 1.0) / 4.0, 0.0), 1.0)
        )
        b.add("leverage_health_score", leverage_score)

        distress_score = 100.0 * (
            0.25 * (1.0 - min(max(safe_div(nsf_count, len(tx)) / 0.05, 0.0), 1.0))
            + 0.20 * (1.0 - min(max(safe_div(len(returned), len(tx)) / 0.05, 0.0), 1.0))
            + 0.20
            * (1.0 - min(max(safe_div(float(gambling.sum()), expense_total) / 0.10, 0.0), 1.0))
            + 0.15 * (1.0 - min(max(safe_div(float(fees.sum()), revenue_total) / 0.05, 0.0), 1.0))
            + 0.20 * (1.0 - min(max(net_negative_ratio, 0.0), 1.0))
        )
        b.add("distress_health_score", distress_score)
        b.add(
            "cashflow_health_score",
            0.25 * stability
            + 0.35 * liquidity_score
            + 0.20 * leverage_score
            + 0.20 * distress_score,
        )
        return b.to_series()


def _peak_month(series: pd.Series) -> int:
    """Calendar month with the highest average, using the actual PeriodIndex."""
    if not isinstance(series.index, pd.PeriodIndex) or series.empty:
        return 0
    by_month = series.groupby(series.index.month).mean()
    return int(by_month.idxmax()) if len(by_month) else 0


def _corr(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 3 or float(left.std(ddof=0)) < 1e-12 or float(right.std(ddof=0)) < 1e-12:
        return 0.0
    value = float(left.corr(right))
    return value if np.isfinite(value) else 0.0


def _volatility_ratio(expense: pd.Series, revenue: pd.Series) -> float:
    exp_cv = coefficient_of_variation(expense)
    rev_cv = coefficient_of_variation(revenue)
    if rev_cv < 1e-9:
        return 0.0 if exp_cv < 1e-9 else 10.0
    return float(min(exp_cv / rev_cv, 10.0))


def _recurring_share(expense_tx: pd.DataFrame, expense_total: float) -> float:
    """Share of expense value paid to counterparties seen in at least three distinct months."""
    if expense_tx.empty or expense_total <= 0:
        return 0.0
    work = expense_tx.copy()
    work["month"] = work["date"].dt.to_period("M")
    keys = _counterparty_keys(work)
    mask = keys.ne("UNKNOWN") & keys.ne("")
    if not mask.any():
        return 0.0
    work = work.loc[mask].assign(_key=keys.loc[mask])
    months_per_key = work.groupby("_key")["month"].nunique()
    recurring = months_per_key[months_per_key >= 3].index
    if len(recurring) == 0:
        return 0.0
    return safe_div(
        float(work.loc[work["_key"].isin(recurring), "exp_amount"].sum()), expense_total
    )


def compute_features(
    transactions: pd.DataFrame,
    config: FeatureConfig | None = None,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Convenience function: normalise, categorise and compute features in one call."""
    extractor = CashFlowFeatureExtractor(config=config)
    return extractor.fit_transform(transactions, as_of=as_of)


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SPECS",
    "CashFlowFeatureExtractor",
    "FeatureBuilder",
    "FeatureConfig",
    "compute_features",
]
