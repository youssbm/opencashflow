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
        _spec(
            "txn_count",
            "coverage",
            "Number of transactions in the observation window.",
            formula="count of transactions in the window",
        ),
        _spec(
            "txn_count_monthly_avg",
            "coverage",
            "Average transactions per observed month.",
            formula="txn_count divided by months_observed",
        ),
        _spec(
            "months_observed",
            "coverage",
            "Number of calendar months with an observed account span.",
            True,
            "number of calendar months from the start month to the as-of month",
        ),
        _spec(
            "days_observed",
            "coverage",
            "Calendar days between the first transaction and the as-of date.",
            True,
            "as_of minus max(window start, first transaction date), in days, plus one",
        ),
        _spec(
            "coverage_ratio",
            "coverage",
            "Observed days divided by the configured window length.",
            True,
            "days_observed divided by window_days",
        ),
        _spec(
            "active_days_ratio",
            "coverage",
            "Distinct transaction days divided by observed days.",
            formula="distinct transaction dates divided by days_observed",
        ),
        _spec(
            "txn_amount_mean",
            "coverage",
            "Mean absolute transaction amount.",
            formula="mean of absolute transaction amounts",
        ),
        _spec(
            "txn_amount_median",
            "coverage",
            "Median absolute transaction amount.",
            formula="median of absolute transaction amounts",
        ),
        _spec(
            "txn_amount_std",
            "coverage",
            "Standard deviation of absolute transaction amounts.",
            formula="population standard deviation of absolute transaction amounts",
        ),
        _spec(
            "txn_amount_cv",
            "coverage",
            "Coefficient of variation of absolute transaction amounts.",
            formula="standard deviation of absolute amounts divided by their mean",
        ),
        _spec(
            "uncategorized_ratio",
            "coverage",
            "Share of transactions that matched no category.",
            False,
            "share of transactions with no known category",
        ),
        _spec(
            "has_balance_data",
            "coverage",
            "1 when the source includes running balances, else 0.",
            True,
            "1 if any transaction has a non-null balance, else 0",
        ),
        # ----------------------------------------------------------------- revenue
        _spec(
            "revenue_total",
            "revenue",
            "Total customer receipts in the window.",
            True,
            "sum of revenue-category inflows over the window",
        ),
        _spec(
            "revenue_monthly_mean",
            "revenue",
            "Mean monthly customer receipts.",
            True,
            "mean of monthly revenue",
        ),
        _spec(
            "revenue_monthly_median",
            "revenue",
            "Median monthly customer receipts.",
            True,
            "median of monthly revenue",
        ),
        _spec(
            "revenue_monthly_std",
            "revenue",
            "Standard deviation of monthly customer receipts.",
            False,
            "population standard deviation of monthly revenue",
        ),
        _spec(
            "revenue_cv",
            "revenue",
            "Coefficient of variation of monthly revenue.",
            False,
            "standard deviation of monthly revenue divided by its mean",
        ),
        _spec(
            "revenue_growth_3m",
            "revenue",
            "Last 3 months revenue versus the prior 3 months.",
            True,
            "mean of the last 3 months revenue relative to the mean of the prior 3 months",
        ),
        _spec(
            "revenue_growth_6m",
            "revenue",
            "Last 6 months revenue versus the prior 6 months.",
            True,
            "mean of the last 6 months revenue relative to the mean of the prior 6 months",
        ),
        _spec(
            "revenue_trend_slope",
            "revenue",
            "Normalised OLS slope of monthly revenue.",
            True,
            "OLS slope of monthly revenue divided by mean absolute revenue",
        ),
        _spec(
            "revenue_momentum",
            "revenue",
            "Latest month revenue versus the 3-month average.",
            True,
            "relative change from the 3-month mean revenue to the latest month revenue",
        ),
        _spec(
            "revenue_positive_month_ratio",
            "revenue",
            "Share of months with positive revenue.",
            True,
            "share of months with revenue above zero",
        ),
        _spec(
            "revenue_min_monthly",
            "revenue",
            "Lowest monthly revenue.",
            formula="minimum of monthly revenue",
        ),
        _spec(
            "revenue_max_monthly",
            "revenue",
            "Highest monthly revenue.",
            formula="maximum of monthly revenue",
        ),
        _spec(
            "revenue_txn_count",
            "revenue",
            "Number of customer receipt transactions.",
            formula="count of positive revenue-category transactions",
        ),
        _spec(
            "revenue_txn_mean",
            "revenue",
            "Mean customer receipt size.",
            formula="mean amount of positive revenue-category transactions",
        ),
        _spec(
            "revenue_seasonality_strength",
            "revenue",
            "Variance explained by month-of-year revenue patterns (0-1).",
            formula="1 minus residual variance over total variance of monthly revenue by month of year",
        ),
        _spec(
            "revenue_seasonality_peak_month",
            "revenue",
            "Calendar month (1-12) with the highest average revenue.",
            formula=(
                "calendar month with the highest mean revenue; 0 when there are fewer than "
                "two seasonal cycles"
            ),
        ),
        _spec(
            "revenue_hhi",
            "revenue",
            "Customer concentration (Herfindahl-Hirschman index, 0-1).",
            False,
            "Herfindahl index of revenue by counterparty",
        ),
        _spec(
            "revenue_top1_share",
            "revenue",
            "Share of revenue from the largest customer.",
            False,
            "revenue from the largest counterparty divided by total known revenue",
        ),
        _spec(
            "revenue_top3_share",
            "revenue",
            "Share of revenue from the three largest customers.",
            False,
            "revenue from the three largest counterparties divided by total known revenue",
        ),
        _spec(
            "revenue_entropy",
            "revenue",
            "Normalised diversity of the customer base (0-1).",
            True,
            "normalised Shannon entropy of revenue across counterparties",
        ),
        _spec(
            "revenue_counterparty_count",
            "revenue",
            "Number of distinct revenue counterparties.",
            True,
            "number of distinct revenue counterparties",
        ),
        _spec(
            "revenue_growth_volatility",
            "revenue",
            "Standard deviation of month-on-month revenue growth.",
            False,
            "standard deviation of month-on-month relative revenue changes",
        ),
        _spec(
            "revenue_last_month_share",
            "revenue",
            "Latest month revenue as a share of total revenue.",
            formula="latest month revenue divided by revenue_total",
        ),
        _spec(
            "inflow_quality",
            "revenue",
            "Customer receipts as a share of all inflows.",
            True,
            "revenue_total divided by total inflows",
        ),
        # ----------------------------------------------------------------- expenses
        _spec(
            "expense_total",
            "expenses",
            "Total operating outflows excluding debt principal.",
            False,
            "sum of expense-category outflows including interest, excluding debt principal",
        ),
        _spec(
            "expense_monthly_mean",
            "expenses",
            "Mean monthly operating outflows.",
            False,
            "mean of monthly operating expenses",
        ),
        _spec(
            "expense_monthly_std",
            "expenses",
            "Standard deviation of monthly operating outflows.",
            False,
            "population standard deviation of monthly operating expenses",
        ),
        _spec(
            "expense_cv",
            "expenses",
            "Coefficient of variation of monthly operating outflows.",
            False,
            "standard deviation of monthly expenses divided by their mean",
        ),
        _spec(
            "expense_growth_3m",
            "expenses",
            "Last 3 months expenses versus the prior 3 months.",
            formula="mean of the last 3 months expenses relative to the mean of the prior 3 months",
        ),
        _spec(
            "expense_to_revenue",
            "expenses",
            "Operating expenses divided by revenue.",
            False,
            "expense_total divided by revenue_total",
        ),
        _spec(
            "payroll_total",
            "expenses",
            "Total payroll and social contributions.",
            formula="sum of payroll-category outflows",
        ),
        _spec(
            "payroll_monthly_mean",
            "expenses",
            "Mean monthly payroll.",
            formula="mean of monthly payroll",
        ),
        _spec(
            "payroll_to_revenue",
            "expenses",
            "Payroll divided by revenue.",
            formula="payroll_total divided by revenue_total",
        ),
        _spec(
            "payroll_cv",
            "expenses",
            "Coefficient of variation of payroll in active payroll months.",
            False,
            "coefficient of variation of payroll in months with payroll",
        ),
        _spec(
            "payroll_regularity",
            "expenses",
            "Stability score for payroll payments (0-1).",
            True,
            "1 divided by (1 plus the payroll coefficient of variation)",
        ),
        _spec(
            "payroll_months_ratio",
            "expenses",
            "Share of months with a payroll payment.",
            True,
            "share of months with a positive payroll payment",
        ),
        _spec(
            "rent_total",
            "expenses",
            "Total rent and property lease payments.",
            formula="sum of rent-category outflows",
        ),
        _spec(
            "rent_to_revenue",
            "expenses",
            "Rent divided by revenue.",
            False,
            "rent_total divided by revenue_total",
        ),
        _spec(
            "rent_to_expense",
            "expenses",
            "Rent as a share of operating expenses.",
            formula="rent_total divided by expense_total",
        ),
        _spec(
            "tax_total",
            "expenses",
            "Total direct tax payments.",
            formula="sum of direct-tax-category outflows",
        ),
        _spec(
            "tax_to_revenue",
            "expenses",
            "Direct tax divided by revenue.",
            formula="tax_total divided by revenue_total",
        ),
        _spec(
            "vat_total",
            "expenses",
            "Total VAT and sales-tax payments.",
            formula="sum of VAT and sales-tax outflows",
        ),
        _spec(
            "vat_to_revenue",
            "expenses",
            "VAT divided by revenue.",
            formula="vat_total divided by revenue_total",
        ),
        _spec(
            "supplier_total",
            "expenses",
            "Total supplier and inventory payments.",
            formula="sum of supplier and inventory outflows",
        ),
        _spec(
            "supplier_to_revenue",
            "expenses",
            "Supplier payments divided by revenue.",
            formula="supplier_total divided by revenue_total",
        ),
        _spec(
            "supplier_to_expense",
            "expenses",
            "Supplier payments as a share of operating expenses.",
            formula="supplier_total divided by expense_total",
        ),
        _spec(
            "utilities_total",
            "expenses",
            "Total utility and telecom payments.",
            formula="sum of utility and telecom outflows",
        ),
        _spec(
            "utilities_to_revenue",
            "expenses",
            "Utilities divided by revenue.",
            formula="utilities_total divided by revenue_total",
        ),
        _spec(
            "fixed_cost_ratio",
            "expenses",
            "Payroll, rent, utilities, insurance and interest divided by revenue.",
            False,
            "(payroll + rent + utilities + insurance + interest) divided by revenue_total",
        ),
        _spec(
            "discretionary_ratio",
            "expenses",
            "Card, cash and gambling outflows as a share of expenses.",
            False,
            "(card + cash withdrawal + gambling) divided by expense_total",
        ),
        _spec(
            "expense_txn_count",
            "expenses",
            "Number of operating expense transactions.",
            formula="count of expense-category outflow transactions",
        ),
        _spec(
            "expense_txn_mean",
            "expenses",
            "Mean operating expense transaction size.",
            formula="mean amount of expense-category outflow transactions",
        ),
        _spec(
            "largest_expense_share",
            "expenses",
            "Largest single expense as a share of total expenses.",
            False,
            "largest single expense divided by expense_total",
        ),
        _spec(
            "expense_hhi",
            "expenses",
            "Concentration of expenses across counterparties (0-1).",
            formula="Herfindahl index of expenses by counterparty",
        ),
        _spec(
            "supplier_count",
            "expenses",
            "Number of distinct supplier counterparties.",
            formula="number of distinct supplier counterparties",
        ),
        _spec(
            "supplier_concentration",
            "expenses",
            "Concentration of supplier payments (0-1).",
            False,
            "Herfindahl index of supplier payments by counterparty",
        ),
        # ----------------------------------------------------------------- liquidity
        _spec(
            "balance_mean",
            "liquidity",
            "Mean running balance.",
            True,
            "mean of the daily balance series",
        ),
        _spec(
            "balance_median",
            "liquidity",
            "Median running balance.",
            True,
            "median of the daily balance series",
        ),
        _spec(
            "balance_min",
            "liquidity",
            "Minimum running balance.",
            True,
            "minimum of the daily balance series",
        ),
        _spec(
            "balance_max",
            "liquidity",
            "Maximum running balance.",
            formula="maximum of the daily balance series",
        ),
        _spec(
            "balance_std",
            "liquidity",
            "Standard deviation of the running balance.",
            False,
            "population standard deviation of the daily balance series",
        ),
        _spec(
            "balance_cv",
            "liquidity",
            "Coefficient of variation of the running balance.",
            False,
            "coefficient of variation of monthly end balances",
        ),
        _spec(
            "balance_trend_slope",
            "liquidity",
            "Normalised OLS slope of the running balance.",
            True,
            "OLS slope of monthly end balances divided by mean absolute end balance",
        ),
        _spec(
            "balance_p10",
            "liquidity",
            "10th percentile of the running balance.",
            True,
            "10th percentile of the daily balance series",
        ),
        _spec(
            "balance_p90",
            "liquidity",
            "90th percentile of the running balance.",
            formula="90th percentile of the daily balance series",
        ),
        _spec(
            "balance_last",
            "liquidity",
            "Most recent running balance.",
            True,
            "last value of the daily balance series",
        ),
        _spec(
            "balance_last_90d_mean",
            "liquidity",
            "Mean balance over the final 90 days.",
            True,
            "mean daily balance over the final 90 days",
        ),
        _spec(
            "balance_min_last_90d",
            "liquidity",
            "Minimum balance over the final 90 days.",
            True,
            "minimum daily balance over the final 90 days",
        ),
        _spec(
            "balance_drawdown_max",
            "liquidity",
            "Largest peak-to-trough balance decline (0-1).",
            False,
            "largest peak-to-trough decline of the daily balance, as a fraction of the peak",
        ),
        _spec(
            "negative_balance_days",
            "liquidity",
            "Distinct active days with a negative end-of-day balance.",
            False,
            "number of active days with a negative end-of-day balance",
        ),
        _spec(
            "negative_balance_ratio",
            "liquidity",
            "Active days with a negative end-of-day balance divided by all active days.",
            False,
            "negative_balance_days divided by the number of active days",
        ),
        _spec(
            "overdraft_days",
            "liquidity",
            "Calendar days with a negative balance.",
            False,
            "number of calendar days with a negative daily balance",
        ),
        _spec(
            "overdraft_ratio",
            "liquidity",
            "Overdraft days divided by observed days.",
            False,
            "overdraft_days divided by days_observed",
        ),
        _spec(
            "days_cash",
            "liquidity",
            "Mean balance divided by average daily operating spend.",
            True,
            "mean daily balance divided by (expense_total divided by days_observed)",
        ),
        _spec(
            "runway_months",
            "liquidity",
            "Latest balance divided by mean monthly expenses.",
            True,
            "last daily balance divided by mean monthly expenses",
        ),
        _spec(
            "liquidity_ratio",
            "liquidity",
            "Total inflows divided by total outflows.",
            True,
            "total inflows divided by total outflows",
        ),
        _spec(
            "net_cash_flow_total",
            "liquidity",
            "Inflows minus outflows over the window.",
            True,
            "total inflows minus total outflows",
        ),
        _spec(
            "net_cash_flow_monthly_mean",
            "liquidity",
            "Mean monthly net cash flow.",
            True,
            "mean of monthly net cash flow",
        ),
        _spec(
            "net_cash_flow_cv",
            "liquidity",
            "Coefficient of variation of monthly net cash flow.",
            False,
            "coefficient of variation of monthly net cash flow",
        ),
        _spec(
            "net_cash_flow_negative_months",
            "liquidity",
            "Months with negative net cash flow.",
            False,
            "number of months with negative net cash flow",
        ),
        _spec(
            "net_cash_flow_negative_ratio",
            "liquidity",
            "Share of months with negative net cash flow.",
            False,
            "negative cash-flow months divided by months_observed",
        ),
        _spec(
            "consecutive_negative_months_max",
            "liquidity",
            "Longest run of consecutive negative cash-flow months.",
            False,
            "longest run of consecutive months with negative net cash flow",
        ),
        _spec(
            "cash_flow_volatility",
            "liquidity",
            "Net cash-flow volatility relative to mean revenue.",
            False,
            "standard deviation of monthly net cash flow divided by mean revenue",
        ),
        _spec(
            "burn_rate",
            "liquidity",
            "Average monthly net burn divided by mean revenue.",
            False,
            "max(0, minus mean monthly net cash flow) divided by mean revenue",
        ),
        _spec(
            "cash_buffer_months",
            "liquidity",
            "10th-percentile balance divided by mean monthly expenses.",
            True,
            "10th percentile of the daily balance divided by mean monthly expenses",
        ),
        _spec(
            "net_margin_proxy",
            "liquidity",
            "Revenue minus operating expenses, divided by revenue.",
            True,
            "(revenue_total minus expense_total) divided by revenue_total",
        ),
        # ----------------------------------------------------------------- debt
        _spec(
            "loan_repayment_total",
            "debt",
            "Total loan principal and instalment payments.",
            formula="sum of loan principal and instalment outflows",
        ),
        _spec(
            "loan_repayment_monthly_mean",
            "debt",
            "Mean monthly loan repayment.",
            formula="mean of monthly loan repayments",
        ),
        _spec(
            "debt_service_ratio",
            "debt",
            "Loan repayments divided by revenue.",
            False,
            "loan_repayment_total divided by revenue_total",
        ),
        _spec(
            "dscr",
            "debt",
            "Cash available before interest divided by principal plus interest, capped at 10.",
            True,
            "CFADS / (loan_repayment_total + interest_total), clipped to 0 to 10, with CFADS = "
            "revenue minus (expenses minus interest); sentinel 10 when the denominator is at most 0",
        ),
        _spec(
            "interest_total",
            "debt",
            "Total interest and financing charges.",
            formula="sum of interest and financing-charge outflows",
        ),
        _spec(
            "interest_to_revenue",
            "debt",
            "Interest expense divided by revenue.",
            False,
            "interest_total divided by revenue_total",
        ),
        _spec(
            "interest_coverage",
            "debt",
            "Operating cash available divided by interest, capped at 20.",
            True,
            "CFADS / interest_total, clipped to 0 to 20, with CFADS = revenue minus "
            "(expenses minus interest); sentinel 20 when interest is at most 0",
        ),
        _spec(
            "loan_proceeds_total",
            "debt",
            "Total loan drawdowns and financing received.",
            formula="sum of loan drawdown inflows",
        ),
        _spec(
            "loan_to_revenue",
            "debt",
            "Loan proceeds divided by revenue.",
            formula="loan_proceeds_total divided by revenue_total",
        ),
        _spec(
            "debt_service_months_ratio",
            "debt",
            "Share of months with a debt-service payment.",
            formula="share of months with a positive loan repayment",
        ),
        _spec(
            "debt_service_cv",
            "debt",
            "Coefficient of variation of debt service in active months.",
            False,
            "coefficient of variation of loan repayments in active months",
        ),
        _spec(
            "net_debt_flow",
            "debt",
            "Loan proceeds minus loan repayments.",
            formula="loan_proceeds_total minus loan_repayment_total",
        ),
        # ----------------------------------------------------------------- distress
        _spec(
            "nsf_count",
            "distress",
            "Number of non-sufficient-funds or failed payment events.",
            False,
            "count of nsf-category transactions",
        ),
        _spec(
            "nsf_rate",
            "distress",
            "NSF events divided by transaction count.",
            False,
            "nsf_count divided by transaction count",
        ),
        _spec(
            "returned_payment_count",
            "distress",
            "Number of explicitly returned or rejected payments.",
            False,
            "count of nsf transactions whose normalised text matches the returned-payment pattern",
        ),
        _spec(
            "returned_payment_rate",
            "distress",
            "Returned payments divided by transaction count.",
            False,
            "returned_payment_count divided by transaction count",
        ),
        _spec(
            "bank_fees_total",
            "distress",
            "Total bank and payment-provider fees.",
            False,
            "sum of bank-fee outflows",
        ),
        _spec(
            "bank_fees_to_revenue",
            "distress",
            "Bank fees divided by revenue.",
            False,
            "bank_fees_total divided by revenue_total",
        ),
        _spec(
            "gambling_total",
            "distress",
            "Total gambling and betting outflows.",
            False,
            "sum of gambling outflows",
        ),
        _spec(
            "gambling_to_expense",
            "distress",
            "Gambling outflows as a share of expenses.",
            False,
            "gambling_total divided by expense_total",
        ),
        _spec(
            "cash_withdrawal_total",
            "distress",
            "Total cash withdrawals.",
            formula="sum of cash-withdrawal outflows",
        ),
        _spec(
            "cash_withdrawal_ratio",
            "distress",
            "Cash withdrawals as a share of expenses.",
            False,
            "cash_withdrawal_total divided by expense_total",
        ),
        _spec(
            "card_payment_total",
            "distress",
            "Total card and point-of-sale payments.",
            formula="sum of card-payment outflows",
        ),
        _spec(
            "card_payment_ratio",
            "distress",
            "Card payments as a share of expenses.",
            formula="card_payment_total divided by expense_total",
        ),
        _spec(
            "insurance_total",
            "distress",
            "Total insurance premiums.",
            formula="sum of insurance-premium outflows",
        ),
        _spec(
            "insurance_to_revenue",
            "distress",
            "Insurance premiums divided by revenue.",
            formula="insurance_total divided by revenue_total",
        ),
        _spec(
            "overdraft_events",
            "distress",
            "Transitions from a non-negative to a negative balance.",
            False,
            "count of transitions from a non-negative to a negative daily balance",
        ),
        _spec(
            "distress_flag_count",
            "distress",
            "Count of triggered distress heuristics (0-8).",
            False,
            "number of eight predefined distress heuristics that are true",
        ),
        # ----------------------------------------------------------------- behaviour
        _spec(
            "round_amount_ratio",
            "behaviour",
            "Share of transactions with a round amount (multiple of 100).",
            formula="share of transactions with a strictly positive absolute amount divisible by 100",
        ),
        _spec(
            "weekend_ratio",
            "behaviour",
            "Share of transactions booked on a weekend.",
            formula="share of transactions booked on a Saturday or Sunday",
        ),
        _spec(
            "month_end_ratio",
            "behaviour",
            "Share of transactions booked in the last three days of a month.",
            formula="share of transactions booked on day 28 to 31 of a month",
        ),
        _spec(
            "salary_day_consistency",
            "behaviour",
            "Consistency of the payroll day of month (0-1).",
            True,
            "1 minus min(payroll day-of-month standard deviation divided by 10, 1); 0 when "
            "there are fewer than three payroll transactions",
        ),
        _spec(
            "dormant_days_max",
            "behaviour",
            "Longest gap between consecutive transactions.",
            False,
            "maximum gap in days between consecutive transaction dates",
        ),
        _spec(
            "txn_frequency_trend",
            "behaviour",
            "Normalised trend in monthly transaction count.",
            True,
            "OLS slope of monthly transaction counts divided by mean absolute count",
        ),
        _spec(
            "inflow_outflow_correlation",
            "behaviour",
            "Correlation between monthly inflows and outflows.",
            formula="Pearson correlation of monthly inflows and outflows",
        ),
        _spec(
            "internal_transfer_ratio",
            "behaviour",
            "Internal transfers as a share of absolute cash flow.",
            False,
            "(transfer_in plus transfer_out) divided by total absolute flow",
        ),
        _spec(
            "refund_ratio",
            "behaviour",
            "Refunds as a share of total inflows.",
            formula="refunds divided by total inflows",
        ),
        _spec(
            "outflow_quality",
            "behaviour",
            "Operating expenses as a share of all outflows.",
            True,
            "expense_total divided by total outflows",
        ),
        _spec(
            "recurring_counterparty_ratio",
            "behaviour",
            "Outflows to counterparties seen in three or more months.",
            True,
            "share of expense value paid to counterparties seen in at least three months",
        ),
        # ----------------------------------------------------------------- trends
        _spec(
            "inflow_growth_3m",
            "trends",
            "Last 3 months inflows versus the prior 3 months.",
            True,
            "mean of the last 3 months inflows relative to the mean of the prior 3 months",
        ),
        _spec(
            "outflow_growth_3m",
            "trends",
            "Last 3 months outflows versus the prior 3 months.",
            formula="mean of the last 3 months outflows relative to the mean of the prior 3 months",
        ),
        _spec(
            "net_growth_3m",
            "trends",
            "Last 3 months net cash flow versus the prior 3 months.",
            True,
            "mean of the last 3 months net cash flow relative to the mean of the prior 3 months",
        ),
        _spec(
            "inflow_trend_slope",
            "trends",
            "Normalised OLS slope of monthly inflows.",
            True,
            "OLS slope of monthly inflows divided by mean absolute inflow",
        ),
        _spec(
            "outflow_trend_slope",
            "trends",
            "Normalised OLS slope of monthly outflows.",
            formula="OLS slope of monthly outflows divided by mean absolute outflow",
        ),
        _spec(
            "net_trend_slope",
            "trends",
            "Normalised OLS slope of monthly net cash flow.",
            True,
            "OLS slope of monthly net cash flow divided by mean absolute net flow",
        ),
        _spec(
            "seasonality_strength_expense",
            "trends",
            "Variance explained by month-of-year expense patterns (0-1).",
            formula="1 minus residual variance over total variance of monthly expenses by month of year",
        ),
        _spec(
            "volatility_ratio",
            "trends",
            "Expense volatility divided by revenue volatility.",
            False,
            "coefficient of variation of expenses divided by coefficient of variation of revenue, "
            "clipped to 0 to 10",
        ),
        _spec(
            "revenue_expense_correlation",
            "trends",
            "Correlation between monthly revenue and expenses.",
            formula="Pearson correlation of monthly revenue and expenses",
        ),
        # ----------------------------------------------------------------- composite
        _spec(
            "cashflow_stability_score",
            "composite",
            "Heuristic stability score (0-100, higher is better).",
            True,
            "weighted score of revenue stability, positive months, net-flow stability, balance "
            "stability and payroll regularity",
        ),
        _spec(
            "liquidity_health_score",
            "composite",
            "Heuristic liquidity score (0-100, higher is better).",
            True,
            "weighted score of days_cash, runway, overdraft days and negative balance days",
        ),
        _spec(
            "leverage_health_score",
            "composite",
            "Heuristic leverage score (0-100, higher is better).",
            True,
            "weighted score of debt-service ratio, dscr and interest coverage",
        ),
        _spec(
            "distress_health_score",
            "composite",
            "Heuristic distress score (0-100, higher is better).",
            True,
            "weighted score of nsf rate, returned rate, gambling, fees and negative months",
        ),
        _spec(
            "cashflow_health_score",
            "composite",
            "Weighted composite of the four health scores (0-100).",
            True,
            "0.25 stability + 0.35 liquidity + 0.20 leverage + 0.20 distress",
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
