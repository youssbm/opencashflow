"""Feature catalogue for cash-flow underwriting.

Every feature produced by :class:`opencashflow.features.cashflow_features.CashFlowFeatureExtractor`
must be declared here. The extractor looks features up by name, so a missing declaration fails
fast, and documentation is generated from the same source of truth.

``higher_is_better`` is expressed from the lender's point of view: ``True`` means a higher value
is associated with lower credit risk (for example more months of history), ``False`` means higher
is riskier (for example revenue volatility), and ``None`` means the direction is contextual.
These signs are heuristics for explanation and monitoring, not causal claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

FeatureGroup = Literal[
    "coverage",
    "revenue",
    "expenses",
    "liquidity",
    "debt",
    "distress",
    "behaviour",
    "trends",
    "composite",
]


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for one engineered feature."""

    name: str
    group: FeatureGroup
    description: str
    higher_is_better: bool | None = None
    formula: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _spec(
    name: str,
    group: FeatureGroup,
    description: str,
    higher_is_better: bool | None = None,
    formula: str = "",
) -> FeatureSpec:
    return FeatureSpec(name, group, description, higher_is_better, formula)


FEATURE_SPECS: dict[str, FeatureSpec] = {
    spec.name: spec
    for spec in (
        # ----------------------------------------------------------------- coverage
        _spec("txn_count", "coverage", "Number of transactions in the observation window.", None),
        _spec(
            "txn_count_monthly_avg", "coverage", "Average transactions per observed month.", None
        ),
        _spec(
            "months_observed",
            "coverage",
            "Number of calendar months with an observed account span.",
            True,
        ),
        _spec(
            "days_observed",
            "coverage",
            "Calendar days between the first transaction and the as-of date.",
            True,
        ),
        _spec(
            "coverage_ratio",
            "coverage",
            "Observed days divided by the configured window length.",
            True,
        ),
        _spec(
            "active_days_ratio",
            "coverage",
            "Distinct transaction days divided by observed days.",
            None,
        ),
        _spec("txn_amount_mean", "coverage", "Mean absolute transaction amount.", None),
        _spec("txn_amount_median", "coverage", "Median absolute transaction amount.", None),
        _spec(
            "txn_amount_std",
            "coverage",
            "Standard deviation of absolute transaction amounts.",
            None,
        ),
        _spec(
            "txn_amount_cv",
            "coverage",
            "Coefficient of variation of absolute transaction amounts.",
            None,
        ),
        _spec(
            "uncategorized_ratio",
            "coverage",
            "Share of transactions that matched no category.",
            False,
        ),
        _spec(
            "has_balance_data",
            "coverage",
            "1 when the source includes running balances, else 0.",
            True,
        ),
        # ----------------------------------------------------------------- revenue
        _spec("revenue_total", "revenue", "Total customer receipts in the window.", True),
        _spec("revenue_monthly_mean", "revenue", "Mean monthly customer receipts.", True),
        _spec("revenue_monthly_median", "revenue", "Median monthly customer receipts.", True),
        _spec(
            "revenue_monthly_std",
            "revenue",
            "Standard deviation of monthly customer receipts.",
            False,
        ),
        _spec("revenue_cv", "revenue", "Coefficient of variation of monthly revenue.", False),
        _spec(
            "revenue_growth_3m", "revenue", "Last 3 months revenue versus the prior 3 months.", True
        ),
        _spec(
            "revenue_growth_6m", "revenue", "Last 6 months revenue versus the prior 6 months.", True
        ),
        _spec("revenue_trend_slope", "revenue", "Normalised OLS slope of monthly revenue.", True),
        _spec(
            "revenue_momentum", "revenue", "Latest month revenue versus the 3-month average.", True
        ),
        _spec(
            "revenue_positive_month_ratio",
            "revenue",
            "Share of months with positive revenue.",
            True,
        ),
        _spec("revenue_min_monthly", "revenue", "Lowest monthly revenue.", None),
        _spec("revenue_max_monthly", "revenue", "Highest monthly revenue.", None),
        _spec("revenue_txn_count", "revenue", "Number of customer receipt transactions.", None),
        _spec("revenue_txn_mean", "revenue", "Mean customer receipt size.", None),
        _spec(
            "revenue_seasonality_strength",
            "revenue",
            "Variance explained by month-of-year revenue patterns (0-1).",
            None,
        ),
        _spec(
            "revenue_seasonality_peak_month",
            "revenue",
            "Calendar month (1-12) with the highest average revenue.",
            None,
        ),
        _spec(
            "revenue_hhi",
            "revenue",
            "Customer concentration (Herfindahl-Hirschman index, 0-1).",
            False,
        ),
        _spec(
            "revenue_top1_share", "revenue", "Share of revenue from the largest customer.", False
        ),
        _spec(
            "revenue_top3_share",
            "revenue",
            "Share of revenue from the three largest customers.",
            False,
        ),
        _spec(
            "revenue_entropy", "revenue", "Normalised diversity of the customer base (0-1).", True
        ),
        _spec(
            "revenue_counterparty_count",
            "revenue",
            "Number of distinct revenue counterparties.",
            True,
        ),
        _spec(
            "revenue_growth_volatility",
            "revenue",
            "Standard deviation of month-on-month revenue growth.",
            False,
        ),
        _spec(
            "revenue_last_month_share",
            "revenue",
            "Latest month revenue as a share of total revenue.",
            None,
        ),
        _spec("inflow_quality", "revenue", "Customer receipts as a share of all inflows.", True),
        # ----------------------------------------------------------------- expenses
        _spec(
            "expense_total", "expenses", "Total operating outflows excluding debt principal.", False
        ),
        _spec("expense_monthly_mean", "expenses", "Mean monthly operating outflows.", False),
        _spec(
            "expense_monthly_std",
            "expenses",
            "Standard deviation of monthly operating outflows.",
            False,
        ),
        _spec(
            "expense_cv",
            "expenses",
            "Coefficient of variation of monthly operating outflows.",
            False,
        ),
        _spec(
            "expense_growth_3m",
            "expenses",
            "Last 3 months expenses versus the prior 3 months.",
            None,
        ),
        _spec("expense_to_revenue", "expenses", "Operating expenses divided by revenue.", False),
        _spec("payroll_total", "expenses", "Total payroll and social contributions.", None),
        _spec("payroll_monthly_mean", "expenses", "Mean monthly payroll.", None),
        _spec("payroll_to_revenue", "expenses", "Payroll divided by revenue.", None),
        _spec(
            "payroll_cv",
            "expenses",
            "Coefficient of variation of payroll in active payroll months.",
            False,
        ),
        _spec(
            "payroll_regularity", "expenses", "Stability score for payroll payments (0-1).", True
        ),
        _spec("payroll_months_ratio", "expenses", "Share of months with a payroll payment.", True),
        _spec("rent_total", "expenses", "Total rent and property lease payments.", None),
        _spec("rent_to_revenue", "expenses", "Rent divided by revenue.", False),
        _spec("rent_to_expense", "expenses", "Rent as a share of operating expenses.", None),
        _spec("tax_total", "expenses", "Total direct tax and social contribution payments.", None),
        _spec("tax_to_revenue", "expenses", "Direct tax divided by revenue.", None),
        _spec("vat_total", "expenses", "Total VAT and sales-tax payments.", None),
        _spec("vat_to_revenue", "expenses", "VAT divided by revenue.", None),
        _spec("supplier_total", "expenses", "Total supplier and inventory payments.", None),
        _spec("supplier_to_revenue", "expenses", "Supplier payments divided by revenue.", None),
        _spec(
            "supplier_to_expense",
            "expenses",
            "Supplier payments as a share of operating expenses.",
            None,
        ),
        _spec("utilities_total", "expenses", "Total utility and telecom payments.", None),
        _spec("utilities_to_revenue", "expenses", "Utilities divided by revenue.", None),
        _spec(
            "fixed_cost_ratio",
            "expenses",
            "Payroll, rent, utilities, insurance and interest divided by revenue.",
            False,
        ),
        _spec(
            "discretionary_ratio",
            "expenses",
            "Card, cash and gambling outflows as a share of expenses.",
            False,
        ),
        _spec("expense_txn_count", "expenses", "Number of operating expense transactions.", None),
        _spec("expense_txn_mean", "expenses", "Mean operating expense transaction size.", None),
        _spec(
            "largest_expense_share",
            "expenses",
            "Largest single expense as a share of total expenses.",
            False,
        ),
        _spec(
            "expense_hhi",
            "expenses",
            "Concentration of expenses across counterparties (0-1).",
            None,
        ),
        _spec("supplier_count", "expenses", "Number of distinct supplier counterparties.", None),
        _spec(
            "supplier_concentration", "expenses", "Concentration of supplier payments (0-1).", False
        ),
        # ----------------------------------------------------------------- liquidity
        _spec("balance_mean", "liquidity", "Mean running balance.", True),
        _spec("balance_median", "liquidity", "Median running balance.", True),
        _spec("balance_min", "liquidity", "Minimum running balance.", True),
        _spec("balance_max", "liquidity", "Maximum running balance.", None),
        _spec("balance_std", "liquidity", "Standard deviation of the running balance.", False),
        _spec("balance_cv", "liquidity", "Coefficient of variation of the running balance.", False),
        _spec(
            "balance_trend_slope", "liquidity", "Normalised OLS slope of the running balance.", True
        ),
        _spec("balance_p10", "liquidity", "10th percentile of the running balance.", True),
        _spec("balance_p90", "liquidity", "90th percentile of the running balance.", None),
        _spec("balance_last", "liquidity", "Most recent running balance.", True),
        _spec("balance_last_90d_mean", "liquidity", "Mean balance over the final 90 days.", True),
        _spec("balance_min_last_90d", "liquidity", "Minimum balance over the final 90 days.", True),
        _spec(
            "balance_drawdown_max",
            "liquidity",
            "Largest peak-to-trough balance decline (0-1).",
            False,
        ),
        _spec(
            "negative_balance_days", "liquidity", "Transaction days with a negative balance.", False
        ),
        _spec(
            "negative_balance_ratio",
            "liquidity",
            "Negative-balance transaction days divided by active days.",
            False,
        ),
        _spec("overdraft_days", "liquidity", "Calendar days with a negative balance.", False),
        _spec("overdraft_ratio", "liquidity", "Overdraft days divided by observed days.", False),
        _spec(
            "days_cash", "liquidity", "Mean balance divided by average daily operating spend.", True
        ),
        _spec(
            "runway_months", "liquidity", "Latest balance divided by mean monthly expenses.", True
        ),
        _spec("liquidity_ratio", "liquidity", "Total inflows divided by total outflows.", True),
        _spec("net_cash_flow_total", "liquidity", "Inflows minus outflows over the window.", True),
        _spec("net_cash_flow_monthly_mean", "liquidity", "Mean monthly net cash flow.", True),
        _spec(
            "net_cash_flow_cv",
            "liquidity",
            "Coefficient of variation of monthly net cash flow.",
            False,
        ),
        _spec(
            "net_cash_flow_negative_months",
            "liquidity",
            "Months with negative net cash flow.",
            False,
        ),
        _spec(
            "net_cash_flow_negative_ratio",
            "liquidity",
            "Share of months with negative net cash flow.",
            False,
        ),
        _spec(
            "consecutive_negative_months_max",
            "liquidity",
            "Longest run of consecutive negative cash-flow months.",
            False,
        ),
        _spec(
            "cash_flow_volatility",
            "liquidity",
            "Net cash-flow volatility relative to mean revenue.",
            False,
        ),
        _spec("burn_rate", "liquidity", "Average monthly net burn divided by mean revenue.", False),
        _spec(
            "cash_buffer_months",
            "liquidity",
            "10th-percentile balance divided by mean monthly expenses.",
            True,
        ),
        _spec(
            "net_margin_proxy",
            "liquidity",
            "Revenue minus operating expenses, divided by revenue.",
            True,
        ),
        # ----------------------------------------------------------------- debt
        _spec(
            "loan_repayment_total", "debt", "Total loan principal and instalment payments.", None
        ),
        _spec("loan_repayment_monthly_mean", "debt", "Mean monthly loan repayment.", None),
        _spec("debt_service_ratio", "debt", "Loan repayments divided by revenue.", False),
        _spec("dscr", "debt", "Debt-service coverage ratio, capped at 10.", True),
        _spec("interest_total", "debt", "Total interest and financing charges.", None),
        _spec("interest_to_revenue", "debt", "Interest expense divided by revenue.", False),
        _spec(
            "interest_coverage",
            "debt",
            "Operating cash available divided by interest, capped at 20.",
            True,
        ),
        _spec("loan_proceeds_total", "debt", "Total loan drawdowns and financing received.", None),
        _spec("loan_to_revenue", "debt", "Loan proceeds divided by revenue.", None),
        _spec(
            "debt_service_months_ratio",
            "debt",
            "Share of months with a debt-service payment.",
            None,
        ),
        _spec(
            "debt_service_cv",
            "debt",
            "Coefficient of variation of debt service in active months.",
            False,
        ),
        _spec("net_debt_flow", "debt", "Loan proceeds minus loan repayments.", None),
        # ----------------------------------------------------------------- distress
        _spec(
            "nsf_count",
            "distress",
            "Number of non-sufficient-funds or failed payment events.",
            False,
        ),
        _spec("nsf_rate", "distress", "NSF events divided by transaction count.", False),
        _spec(
            "returned_payment_count",
            "distress",
            "Number of explicitly returned or rejected payments.",
            False,
        ),
        _spec(
            "returned_payment_rate",
            "distress",
            "Returned payments divided by transaction count.",
            False,
        ),
        _spec("bank_fees_total", "distress", "Total bank and payment-provider fees.", False),
        _spec("bank_fees_to_revenue", "distress", "Bank fees divided by revenue.", False),
        _spec("gambling_total", "distress", "Total gambling and betting outflows.", False),
        _spec(
            "gambling_to_expense", "distress", "Gambling outflows as a share of expenses.", False
        ),
        _spec("cash_withdrawal_total", "distress", "Total cash withdrawals.", None),
        _spec(
            "cash_withdrawal_ratio", "distress", "Cash withdrawals as a share of expenses.", False
        ),
        _spec("card_payment_total", "distress", "Total card and point-of-sale payments.", None),
        _spec("card_payment_ratio", "distress", "Card payments as a share of expenses.", None),
        _spec("insurance_total", "distress", "Total insurance premiums.", None),
        _spec("insurance_to_revenue", "distress", "Insurance premiums divided by revenue.", None),
        _spec(
            "overdraft_events",
            "distress",
            "Transitions from a positive to a negative balance.",
            False,
        ),
        _spec(
            "distress_flag_count",
            "distress",
            "Count of triggered distress heuristics (0-8).",
            False,
        ),
        # ----------------------------------------------------------------- behaviour
        _spec(
            "round_amount_ratio",
            "behaviour",
            "Share of transactions with a round amount (multiple of 100).",
            None,
        ),
        _spec("weekend_ratio", "behaviour", "Share of transactions booked on a weekend.", None),
        _spec(
            "month_end_ratio",
            "behaviour",
            "Share of transactions booked in the last three days of a month.",
            None,
        ),
        _spec(
            "salary_day_consistency",
            "behaviour",
            "Consistency of the payroll day of month (0-1).",
            True,
        ),
        _spec(
            "dormant_days_max", "behaviour", "Longest gap between consecutive transactions.", False
        ),
        _spec(
            "txn_frequency_trend",
            "behaviour",
            "Normalised trend in monthly transaction count.",
            True,
        ),
        _spec(
            "inflow_outflow_correlation",
            "behaviour",
            "Correlation between monthly inflows and outflows.",
            None,
        ),
        _spec(
            "internal_transfer_ratio",
            "behaviour",
            "Internal transfers as a share of absolute cash flow.",
            False,
        ),
        _spec("refund_ratio", "behaviour", "Refunds as a share of total inflows.", None),
        _spec(
            "outflow_quality", "behaviour", "Operating expenses as a share of all outflows.", True
        ),
        _spec(
            "recurring_counterparty_ratio",
            "behaviour",
            "Outflows to counterparties seen in three or more months.",
            True,
        ),
        # ----------------------------------------------------------------- trends
        _spec(
            "inflow_growth_3m", "trends", "Last 3 months inflows versus the prior 3 months.", True
        ),
        _spec(
            "outflow_growth_3m", "trends", "Last 3 months outflows versus the prior 3 months.", None
        ),
        _spec(
            "net_growth_3m",
            "trends",
            "Last 3 months net cash flow versus the prior 3 months.",
            True,
        ),
        _spec("inflow_trend_slope", "trends", "Normalised OLS slope of monthly inflows.", True),
        _spec("outflow_trend_slope", "trends", "Normalised OLS slope of monthly outflows.", None),
        _spec("net_trend_slope", "trends", "Normalised OLS slope of monthly net cash flow.", True),
        _spec(
            "seasonality_strength_expense",
            "trends",
            "Variance explained by month-of-year expense patterns (0-1).",
            None,
        ),
        _spec(
            "volatility_ratio", "trends", "Expense volatility divided by revenue volatility.", False
        ),
        _spec(
            "revenue_expense_correlation",
            "trends",
            "Correlation between monthly revenue and expenses.",
            None,
        ),
        # ----------------------------------------------------------------- composite
        _spec(
            "cashflow_stability_score",
            "composite",
            "Heuristic stability score (0-100, higher is better).",
            True,
        ),
        _spec(
            "liquidity_health_score",
            "composite",
            "Heuristic liquidity score (0-100, higher is better).",
            True,
        ),
        _spec(
            "leverage_health_score",
            "composite",
            "Heuristic leverage score (0-100, higher is better).",
            True,
        ),
        _spec(
            "distress_health_score",
            "composite",
            "Heuristic distress score (0-100, higher is better).",
            True,
        ),
        _spec(
            "cashflow_health_score",
            "composite",
            "Weighted composite of the four health scores (0-100).",
            True,
        ),
    )
}

FEATURE_NAMES: tuple[str, ...] = tuple(FEATURE_SPECS)


def feature_specs_frame() -> object:
    """Return the feature catalogue as a pandas DataFrame, sorted by group then name."""
    import pandas as pd

    rows = [spec.to_dict() for spec in FEATURE_SPECS.values()]
    df = pd.DataFrame(rows)
    order = {
        "coverage": 0,
        "revenue": 1,
        "expenses": 2,
        "liquidity": 3,
        "debt": 4,
        "distress": 5,
        "behaviour": 6,
        "trends": 7,
        "composite": 8,
    }
    df["_order"] = df["group"].map(order)
    return df.sort_values(["_order", "name"]).drop(columns="_order").reset_index(drop=True)


def feature_group_counts() -> dict[str, int]:
    """Number of features per group."""
    counts: dict[str, int] = {}
    for spec in FEATURE_SPECS.values():
        counts[spec.group] = counts.get(spec.group, 0) + 1
    return counts
