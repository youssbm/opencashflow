# Feature dictionary

OpenCashFlow computes **151 features** per borrower from a transaction ledger. This page is
generated from `opencashflow/features/specs.py`; do not edit it by hand. Run
`python scripts/generate_feature_docs.py` after changing the catalogue.

All features are finite floats. Features whose direction is marked **higher is better** are
protective from the lender's point of view; **higher is worse** features are risk drivers.
The sign is a heuristic for explanation and monitoring, not a causal claim and not enforced
by the models.

## Groups

| Group | Features | What it captures |
|---|---:|---|
| `coverage` | 12 | Coverage and data quality |
| `revenue` | 24 | Revenue quality and concentration |
| `expenses` | 32 | Expense structure |
| `liquidity` | 30 | Liquidity and cash flow |
| `debt` | 12 | Debt service |
| `distress` | 16 | Distress events |
| `behaviour` | 11 | Behavioural signals |
| `trends` | 9 | Trends and seasonality |
| `composite` | 5 | Composite health scores |
| **Total** | **151** | |

## Usage

```python
from opencashflow.features.cashflow_features import CashFlowFeatureExtractor

extractor = CashFlowFeatureExtractor()
features = extractor.fit_transform(transactions)
print(features.shape)  # (n_entities, 151)
```

Set `FeatureConfig(window_days=730)` to use two years of history, or pass `as_of` to
score at a fixed point in time. Monthly features include partial boundary months;
`months_observed` and `coverage_ratio` describe how much history is available.

## Coverage and data quality

| Feature | Description | Direction | Formula |
|---|---|---|---|
| `active_days_ratio` | Distinct transaction days divided by observed days. | contextual |  |
| `coverage_ratio` | Observed days divided by the configured window length. | higher is better |  |
| `days_observed` | Calendar days between the first transaction and the as-of date. | higher is better |  |
| `has_balance_data` | 1 when the source includes running balances, else 0. | higher is better |  |
| `months_observed` | Number of calendar months with an observed account span. | higher is better |  |
| `txn_amount_cv` | Coefficient of variation of absolute transaction amounts. | contextual |  |
| `txn_amount_mean` | Mean absolute transaction amount. | contextual |  |
| `txn_amount_median` | Median absolute transaction amount. | contextual |  |
| `txn_amount_std` | Standard deviation of absolute transaction amounts. | contextual |  |
| `txn_count` | Number of transactions in the observation window. | contextual |  |
| `txn_count_monthly_avg` | Average transactions per observed month. | contextual |  |
| `uncategorized_ratio` | Share of transactions that matched no category. | higher is worse |  |

## Revenue quality and concentration

| Feature | Description | Direction | Formula |
|---|---|---|---|
| `inflow_quality` | Customer receipts as a share of all inflows. | higher is better |  |
| `revenue_counterparty_count` | Number of distinct revenue counterparties. | higher is better |  |
| `revenue_cv` | Coefficient of variation of monthly revenue. | higher is worse |  |
| `revenue_entropy` | Normalised diversity of the customer base (0-1). | higher is better |  |
| `revenue_growth_3m` | Last 3 months revenue versus the prior 3 months. | higher is better |  |
| `revenue_growth_6m` | Last 6 months revenue versus the prior 6 months. | higher is better |  |
| `revenue_growth_volatility` | Standard deviation of month-on-month revenue growth. | higher is worse |  |
| `revenue_hhi` | Customer concentration (Herfindahl-Hirschman index, 0-1). | higher is worse |  |
| `revenue_last_month_share` | Latest month revenue as a share of total revenue. | contextual |  |
| `revenue_max_monthly` | Highest monthly revenue. | contextual |  |
| `revenue_min_monthly` | Lowest monthly revenue. | contextual |  |
| `revenue_momentum` | Latest month revenue versus the 3-month average. | higher is better |  |
| `revenue_monthly_mean` | Mean monthly customer receipts. | higher is better |  |
| `revenue_monthly_median` | Median monthly customer receipts. | higher is better |  |
| `revenue_monthly_std` | Standard deviation of monthly customer receipts. | higher is worse |  |
| `revenue_positive_month_ratio` | Share of months with positive revenue. | higher is better |  |
| `revenue_seasonality_peak_month` | Calendar month (1-12) with the highest average revenue. | contextual |  |
| `revenue_seasonality_strength` | Variance explained by month-of-year revenue patterns (0-1). | contextual |  |
| `revenue_top1_share` | Share of revenue from the largest customer. | higher is worse |  |
| `revenue_top3_share` | Share of revenue from the three largest customers. | higher is worse |  |
| `revenue_total` | Total customer receipts in the window. | higher is better |  |
| `revenue_trend_slope` | Normalised OLS slope of monthly revenue. | higher is better |  |
| `revenue_txn_count` | Number of customer receipt transactions. | contextual |  |
| `revenue_txn_mean` | Mean customer receipt size. | contextual |  |

## Expense structure

| Feature | Description | Direction | Formula |
|---|---|---|---|
| `discretionary_ratio` | Card, cash and gambling outflows as a share of expenses. | higher is worse |  |
| `expense_cv` | Coefficient of variation of monthly operating outflows. | higher is worse |  |
| `expense_growth_3m` | Last 3 months expenses versus the prior 3 months. | contextual |  |
| `expense_hhi` | Concentration of expenses across counterparties (0-1). | contextual |  |
| `expense_monthly_mean` | Mean monthly operating outflows. | higher is worse |  |
| `expense_monthly_std` | Standard deviation of monthly operating outflows. | higher is worse |  |
| `expense_to_revenue` | Operating expenses divided by revenue. | higher is worse |  |
| `expense_total` | Total operating outflows excluding debt principal. | higher is worse |  |
| `expense_txn_count` | Number of operating expense transactions. | contextual |  |
| `expense_txn_mean` | Mean operating expense transaction size. | contextual |  |
| `fixed_cost_ratio` | Payroll, rent, utilities, insurance and interest divided by revenue. | higher is worse |  |
| `largest_expense_share` | Largest single expense as a share of total expenses. | higher is worse |  |
| `payroll_cv` | Coefficient of variation of payroll in active payroll months. | higher is worse |  |
| `payroll_monthly_mean` | Mean monthly payroll. | contextual |  |
| `payroll_months_ratio` | Share of months with a payroll payment. | higher is better |  |
| `payroll_regularity` | Stability score for payroll payments (0-1). | higher is better |  |
| `payroll_to_revenue` | Payroll divided by revenue. | contextual |  |
| `payroll_total` | Total payroll and social contributions. | contextual |  |
| `rent_to_expense` | Rent as a share of operating expenses. | contextual |  |
| `rent_to_revenue` | Rent divided by revenue. | higher is worse |  |
| `rent_total` | Total rent and property lease payments. | contextual |  |
| `supplier_concentration` | Concentration of supplier payments (0-1). | higher is worse |  |
| `supplier_count` | Number of distinct supplier counterparties. | contextual |  |
| `supplier_to_expense` | Supplier payments as a share of operating expenses. | contextual |  |
| `supplier_to_revenue` | Supplier payments divided by revenue. | contextual |  |
| `supplier_total` | Total supplier and inventory payments. | contextual |  |
| `tax_to_revenue` | Direct tax divided by revenue. | contextual |  |
| `tax_total` | Total direct tax and social contribution payments. | contextual |  |
| `utilities_to_revenue` | Utilities divided by revenue. | contextual |  |
| `utilities_total` | Total utility and telecom payments. | contextual |  |
| `vat_to_revenue` | VAT divided by revenue. | contextual |  |
| `vat_total` | Total VAT and sales-tax payments. | contextual |  |

## Liquidity and cash flow

| Feature | Description | Direction | Formula |
|---|---|---|---|
| `balance_cv` | Coefficient of variation of the running balance. | higher is worse |  |
| `balance_drawdown_max` | Largest peak-to-trough balance decline (0-1). | higher is worse |  |
| `balance_last` | Most recent running balance. | higher is better |  |
| `balance_last_90d_mean` | Mean balance over the final 90 days. | higher is better |  |
| `balance_max` | Maximum running balance. | contextual |  |
| `balance_mean` | Mean running balance. | higher is better |  |
| `balance_median` | Median running balance. | higher is better |  |
| `balance_min` | Minimum running balance. | higher is better |  |
| `balance_min_last_90d` | Minimum balance over the final 90 days. | higher is better |  |
| `balance_p10` | 10th percentile of the running balance. | higher is better |  |
| `balance_p90` | 90th percentile of the running balance. | contextual |  |
| `balance_std` | Standard deviation of the running balance. | higher is worse |  |
| `balance_trend_slope` | Normalised OLS slope of the running balance. | higher is better |  |
| `burn_rate` | Average monthly net burn divided by mean revenue. | higher is worse |  |
| `cash_buffer_months` | 10th-percentile balance divided by mean monthly expenses. | higher is better |  |
| `cash_flow_volatility` | Net cash-flow volatility relative to mean revenue. | higher is worse |  |
| `consecutive_negative_months_max` | Longest run of consecutive negative cash-flow months. | higher is worse |  |
| `days_cash` | Mean balance divided by average daily operating spend. | higher is better |  |
| `liquidity_ratio` | Total inflows divided by total outflows. | higher is better |  |
| `negative_balance_days` | Transaction days with a negative balance. | higher is worse |  |
| `negative_balance_ratio` | Negative-balance transaction days divided by active days. | higher is worse |  |
| `net_cash_flow_cv` | Coefficient of variation of monthly net cash flow. | higher is worse |  |
| `net_cash_flow_monthly_mean` | Mean monthly net cash flow. | higher is better |  |
| `net_cash_flow_negative_months` | Months with negative net cash flow. | higher is worse |  |
| `net_cash_flow_negative_ratio` | Share of months with negative net cash flow. | higher is worse |  |
| `net_cash_flow_total` | Inflows minus outflows over the window. | higher is better |  |
| `net_margin_proxy` | Revenue minus operating expenses, divided by revenue. | higher is better |  |
| `overdraft_days` | Calendar days with a negative balance. | higher is worse |  |
| `overdraft_ratio` | Overdraft days divided by observed days. | higher is worse |  |
| `runway_months` | Latest balance divided by mean monthly expenses. | higher is better |  |

## Debt service

| Feature | Description | Direction | Formula |
|---|---|---|---|
| `debt_service_cv` | Coefficient of variation of debt service in active months. | higher is worse |  |
| `debt_service_months_ratio` | Share of months with a debt-service payment. | contextual |  |
| `debt_service_ratio` | Loan repayments divided by revenue. | higher is worse |  |
| `dscr` | Debt-service coverage ratio, capped at 10. | higher is better |  |
| `interest_coverage` | Operating cash available divided by interest, capped at 20. | higher is better |  |
| `interest_to_revenue` | Interest expense divided by revenue. | higher is worse |  |
| `interest_total` | Total interest and financing charges. | contextual |  |
| `loan_proceeds_total` | Total loan drawdowns and financing received. | contextual |  |
| `loan_repayment_monthly_mean` | Mean monthly loan repayment. | contextual |  |
| `loan_repayment_total` | Total loan principal and instalment payments. | contextual |  |
| `loan_to_revenue` | Loan proceeds divided by revenue. | contextual |  |
| `net_debt_flow` | Loan proceeds minus loan repayments. | contextual |  |

## Distress events

| Feature | Description | Direction | Formula |
|---|---|---|---|
| `bank_fees_to_revenue` | Bank fees divided by revenue. | higher is worse |  |
| `bank_fees_total` | Total bank and payment-provider fees. | higher is worse |  |
| `card_payment_ratio` | Card payments as a share of expenses. | contextual |  |
| `card_payment_total` | Total card and point-of-sale payments. | contextual |  |
| `cash_withdrawal_ratio` | Cash withdrawals as a share of expenses. | higher is worse |  |
| `cash_withdrawal_total` | Total cash withdrawals. | contextual |  |
| `distress_flag_count` | Count of triggered distress heuristics (0-8). | higher is worse |  |
| `gambling_to_expense` | Gambling outflows as a share of expenses. | higher is worse |  |
| `gambling_total` | Total gambling and betting outflows. | higher is worse |  |
| `insurance_to_revenue` | Insurance premiums divided by revenue. | contextual |  |
| `insurance_total` | Total insurance premiums. | contextual |  |
| `nsf_count` | Number of non-sufficient-funds or failed payment events. | higher is worse |  |
| `nsf_rate` | NSF events divided by transaction count. | higher is worse |  |
| `overdraft_events` | Transitions from a positive to a negative balance. | higher is worse |  |
| `returned_payment_count` | Number of explicitly returned or rejected payments. | higher is worse |  |
| `returned_payment_rate` | Returned payments divided by transaction count. | higher is worse |  |

## Behavioural signals

| Feature | Description | Direction | Formula |
|---|---|---|---|
| `dormant_days_max` | Longest gap between consecutive transactions. | higher is worse |  |
| `inflow_outflow_correlation` | Correlation between monthly inflows and outflows. | contextual |  |
| `internal_transfer_ratio` | Internal transfers as a share of absolute cash flow. | higher is worse |  |
| `month_end_ratio` | Share of transactions booked in the last three days of a month. | contextual |  |
| `outflow_quality` | Operating expenses as a share of all outflows. | higher is better |  |
| `recurring_counterparty_ratio` | Outflows to counterparties seen in three or more months. | higher is better |  |
| `refund_ratio` | Refunds as a share of total inflows. | contextual |  |
| `round_amount_ratio` | Share of transactions with a round amount (multiple of 100). | contextual |  |
| `salary_day_consistency` | Consistency of the payroll day of month (0-1). | higher is better |  |
| `txn_frequency_trend` | Normalised trend in monthly transaction count. | higher is better |  |
| `weekend_ratio` | Share of transactions booked on a weekend. | contextual |  |

## Trends and seasonality

| Feature | Description | Direction | Formula |
|---|---|---|---|
| `inflow_growth_3m` | Last 3 months inflows versus the prior 3 months. | higher is better |  |
| `inflow_trend_slope` | Normalised OLS slope of monthly inflows. | higher is better |  |
| `net_growth_3m` | Last 3 months net cash flow versus the prior 3 months. | higher is better |  |
| `net_trend_slope` | Normalised OLS slope of monthly net cash flow. | higher is better |  |
| `outflow_growth_3m` | Last 3 months outflows versus the prior 3 months. | contextual |  |
| `outflow_trend_slope` | Normalised OLS slope of monthly outflows. | contextual |  |
| `revenue_expense_correlation` | Correlation between monthly revenue and expenses. | contextual |  |
| `seasonality_strength_expense` | Variance explained by month-of-year expense patterns (0-1). | contextual |  |
| `volatility_ratio` | Expense volatility divided by revenue volatility. | higher is worse |  |

## Composite health scores

| Feature | Description | Direction | Formula |
|---|---|---|---|
| `cashflow_health_score` | Weighted composite of the four health scores (0-100). | higher is better |  |
| `cashflow_stability_score` | Heuristic stability score (0-100, higher is better). | higher is better |  |
| `distress_health_score` | Heuristic distress score (0-100, higher is better). | higher is better |  |
| `leverage_health_score` | Heuristic leverage score (0-100, higher is better). | higher is better |  |
| `liquidity_health_score` | Heuristic liquidity score (0-100, higher is better). | higher is better |  |

## Notes

* Revenue means customer receipts (`revenue` category). Other credits are tracked
  separately as `other_inflow`, `transfer_in`, `loan_proceeds`, `refund` and
  `interest_income`.
* Operating expenses exclude loan principal. Debt service is tracked in the `debt` group.
* If the source has no running balance, the extractor derives a cumulative net position
  and sets `has_balance_data = 0`. Treat liquidity features with extra caution in that
  case.
* Segment features by sector before comparing raw currency amounts. Ratios and scores are
  more comparable across companies.
