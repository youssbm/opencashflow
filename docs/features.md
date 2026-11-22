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
| `active_days_ratio` | Distinct transaction days divided by observed days. | contextual | distinct transaction dates divided by days_observed |
| `coverage_ratio` | Observed days divided by the configured window length. | higher is better | days_observed divided by window_days |
| `days_observed` | Calendar days between the first transaction and the as-of date. | higher is better | as_of minus max(window start, first transaction date), in days, plus one |
| `has_balance_data` | 1 when the source includes running balances, else 0. | higher is better | 1 if any transaction has a non-null balance, else 0 |
| `months_observed` | Number of calendar months with an observed account span. | higher is better | number of calendar months from the start month to the as-of month |
| `txn_amount_cv` | Coefficient of variation of absolute transaction amounts. | contextual | standard deviation of absolute amounts divided by their mean |
| `txn_amount_mean` | Mean absolute transaction amount. | contextual | mean of absolute transaction amounts |
| `txn_amount_median` | Median absolute transaction amount. | contextual | median of absolute transaction amounts |
| `txn_amount_std` | Standard deviation of absolute transaction amounts. | contextual | population standard deviation of absolute transaction amounts |
| `txn_count` | Number of transactions in the observation window. | contextual | count of transactions in the window |
| `txn_count_monthly_avg` | Average transactions per observed month. | contextual | txn_count divided by months_observed |
| `uncategorized_ratio` | Share of transactions that matched no category. | higher is worse | share of transactions with no known category |

## Revenue quality and concentration

| Feature | Description | Direction | Formula |
|---|---|---|---|
| `inflow_quality` | Customer receipts as a share of all inflows. | higher is better | revenue_total divided by total inflows |
| `revenue_counterparty_count` | Number of distinct revenue counterparties. | higher is better | number of distinct revenue counterparties |
| `revenue_cv` | Coefficient of variation of monthly revenue. | higher is worse | standard deviation of monthly revenue divided by its mean |
| `revenue_entropy` | Normalised diversity of the customer base (0-1). | higher is better | normalised Shannon entropy of revenue across counterparties |
| `revenue_growth_3m` | Last 3 months revenue versus the prior 3 months. | higher is better | mean of the last 3 months revenue relative to the mean of the prior 3 months |
| `revenue_growth_6m` | Last 6 months revenue versus the prior 6 months. | higher is better | mean of the last 6 months revenue relative to the mean of the prior 6 months |
| `revenue_growth_volatility` | Standard deviation of month-on-month revenue growth. | higher is worse | standard deviation of month-on-month relative revenue changes |
| `revenue_hhi` | Customer concentration (Herfindahl-Hirschman index, 0-1). | higher is worse | Herfindahl index of revenue by counterparty |
| `revenue_last_month_share` | Latest month revenue as a share of total revenue. | contextual | latest month revenue divided by revenue_total |
| `revenue_max_monthly` | Highest monthly revenue. | contextual | maximum of monthly revenue |
| `revenue_min_monthly` | Lowest monthly revenue. | contextual | minimum of monthly revenue |
| `revenue_momentum` | Latest month revenue versus the 3-month average. | higher is better | relative change from the 3-month mean revenue to the latest month revenue |
| `revenue_monthly_mean` | Mean monthly customer receipts. | higher is better | mean of monthly revenue |
| `revenue_monthly_median` | Median monthly customer receipts. | higher is better | median of monthly revenue |
| `revenue_monthly_std` | Standard deviation of monthly customer receipts. | higher is worse | population standard deviation of monthly revenue |
| `revenue_positive_month_ratio` | Share of months with positive revenue. | higher is better | share of months with revenue above zero |
| `revenue_seasonality_peak_month` | Calendar month (1-12) with the highest average revenue. | contextual | calendar month with the highest mean revenue; 0 when there are fewer than two seasonal cycles |
| `revenue_seasonality_strength` | Variance explained by month-of-year revenue patterns (0-1). | contextual | 1 minus residual variance over total variance of monthly revenue by month of year |
| `revenue_top1_share` | Share of revenue from the largest customer. | higher is worse | revenue from the largest counterparty divided by total known revenue |
| `revenue_top3_share` | Share of revenue from the three largest customers. | higher is worse | revenue from the three largest counterparties divided by total known revenue |
| `revenue_total` | Total customer receipts in the window. | higher is better | sum of revenue-category inflows over the window |
| `revenue_trend_slope` | Normalised OLS slope of monthly revenue. | higher is better | OLS slope of monthly revenue divided by mean absolute revenue |
| `revenue_txn_count` | Number of customer receipt transactions. | contextual | count of positive revenue-category transactions |
| `revenue_txn_mean` | Mean customer receipt size. | contextual | mean amount of positive revenue-category transactions |

## Expense structure

| Feature | Description | Direction | Formula |
|---|---|---|---|
| `discretionary_ratio` | Card, cash and gambling outflows as a share of expenses. | higher is worse | (card + cash withdrawal + gambling) divided by expense_total |
| `expense_cv` | Coefficient of variation of monthly operating outflows. | higher is worse | standard deviation of monthly expenses divided by their mean |
| `expense_growth_3m` | Last 3 months expenses versus the prior 3 months. | contextual | mean of the last 3 months expenses relative to the mean of the prior 3 months |
| `expense_hhi` | Concentration of expenses across counterparties (0-1). | contextual | Herfindahl index of expenses by counterparty |
| `expense_monthly_mean` | Mean monthly operating outflows. | higher is worse | mean of monthly operating expenses |
| `expense_monthly_std` | Standard deviation of monthly operating outflows. | higher is worse | population standard deviation of monthly operating expenses |
| `expense_to_revenue` | Operating expenses divided by revenue. | higher is worse | expense_total divided by revenue_total |
| `expense_total` | Total operating outflows excluding debt principal. | higher is worse | sum of expense-category outflows including interest, excluding debt principal |
| `expense_txn_count` | Number of operating expense transactions. | contextual | count of expense-category outflow transactions |
| `expense_txn_mean` | Mean operating expense transaction size. | contextual | mean amount of expense-category outflow transactions |
| `fixed_cost_ratio` | Payroll, rent, utilities, insurance and interest divided by revenue. | higher is worse | (payroll + rent + utilities + insurance + interest) divided by revenue_total |
| `largest_expense_share` | Largest single expense as a share of total expenses. | higher is worse | largest single expense divided by expense_total |
| `payroll_cv` | Coefficient of variation of payroll in active payroll months. | higher is worse | coefficient of variation of payroll in months with payroll |
| `payroll_monthly_mean` | Mean monthly payroll. | contextual | mean of monthly payroll |
| `payroll_months_ratio` | Share of months with a payroll payment. | higher is better | share of months with a positive payroll payment |
| `payroll_regularity` | Stability score for payroll payments (0-1). | higher is better | 1 divided by (1 plus the payroll coefficient of variation) |
| `payroll_to_revenue` | Payroll divided by revenue. | contextual | payroll_total divided by revenue_total |
| `payroll_total` | Total payroll and social contributions. | contextual | sum of payroll-category outflows |
| `rent_to_expense` | Rent as a share of operating expenses. | contextual | rent_total divided by expense_total |
| `rent_to_revenue` | Rent divided by revenue. | higher is worse | rent_total divided by revenue_total |
| `rent_total` | Total rent and property lease payments. | contextual | sum of rent-category outflows |
| `supplier_concentration` | Concentration of supplier payments (0-1). | higher is worse | Herfindahl index of supplier payments by counterparty |
| `supplier_count` | Number of distinct supplier counterparties. | contextual | number of distinct supplier counterparties |
| `supplier_to_expense` | Supplier payments as a share of operating expenses. | contextual | supplier_total divided by expense_total |
| `supplier_to_revenue` | Supplier payments divided by revenue. | contextual | supplier_total divided by revenue_total |
| `supplier_total` | Total supplier and inventory payments. | contextual | sum of supplier and inventory outflows |
| `tax_to_revenue` | Direct tax divided by revenue. | contextual | tax_total divided by revenue_total |
| `tax_total` | Total direct tax payments. | contextual | sum of direct-tax-category outflows |
| `utilities_to_revenue` | Utilities divided by revenue. | contextual | utilities_total divided by revenue_total |
| `utilities_total` | Total utility and telecom payments. | contextual | sum of utility and telecom outflows |
| `vat_to_revenue` | VAT divided by revenue. | contextual | vat_total divided by revenue_total |
| `vat_total` | Total VAT and sales-tax payments. | contextual | sum of VAT and sales-tax outflows |

## Liquidity and cash flow

| Feature | Description | Direction | Formula |
|---|---|---|---|
| `balance_cv` | Coefficient of variation of the running balance. | higher is worse | coefficient of variation of monthly end balances |
| `balance_drawdown_max` | Largest peak-to-trough balance decline (0-1). | higher is worse | largest peak-to-trough decline of the daily balance, as a fraction of the peak |
| `balance_last` | Most recent running balance. | higher is better | last value of the daily balance series |
| `balance_last_90d_mean` | Mean balance over the final 90 days. | higher is better | mean daily balance over the final 90 days |
| `balance_max` | Maximum running balance. | contextual | maximum of the daily balance series |
| `balance_mean` | Mean running balance. | higher is better | mean of the daily balance series |
| `balance_median` | Median running balance. | higher is better | median of the daily balance series |
| `balance_min` | Minimum running balance. | higher is better | minimum of the daily balance series |
| `balance_min_last_90d` | Minimum balance over the final 90 days. | higher is better | minimum daily balance over the final 90 days |
| `balance_p10` | 10th percentile of the running balance. | higher is better | 10th percentile of the daily balance series |
| `balance_p90` | 90th percentile of the running balance. | contextual | 90th percentile of the daily balance series |
| `balance_std` | Standard deviation of the running balance. | higher is worse | population standard deviation of the daily balance series |
| `balance_trend_slope` | Normalised OLS slope of the running balance. | higher is better | OLS slope of monthly end balances divided by mean absolute end balance |
| `burn_rate` | Average monthly net burn divided by mean revenue. | higher is worse | max(0, minus mean monthly net cash flow) divided by mean revenue |
| `cash_buffer_months` | 10th-percentile balance divided by mean monthly expenses. | higher is better | 10th percentile of the daily balance divided by mean monthly expenses |
| `cash_flow_volatility` | Net cash-flow volatility relative to mean revenue. | higher is worse | standard deviation of monthly net cash flow divided by mean revenue |
| `consecutive_negative_months_max` | Longest run of consecutive negative cash-flow months. | higher is worse | longest run of consecutive months with negative net cash flow |
| `days_cash` | Mean balance divided by average daily operating spend. | higher is better | mean daily balance divided by (expense_total divided by days_observed) |
| `liquidity_ratio` | Total inflows divided by total outflows. | higher is better | total inflows divided by total outflows |
| `negative_balance_days` | Distinct active days with a negative end-of-day balance. | higher is worse | number of active days with a negative end-of-day balance |
| `negative_balance_ratio` | Active days with a negative end-of-day balance divided by all active days. | higher is worse | negative_balance_days divided by the number of active days |
| `net_cash_flow_cv` | Coefficient of variation of monthly net cash flow. | higher is worse | coefficient of variation of monthly net cash flow |
| `net_cash_flow_monthly_mean` | Mean monthly net cash flow. | higher is better | mean of monthly net cash flow |
| `net_cash_flow_negative_months` | Months with negative net cash flow. | higher is worse | number of months with negative net cash flow |
| `net_cash_flow_negative_ratio` | Share of months with negative net cash flow. | higher is worse | negative cash-flow months divided by months_observed |
| `net_cash_flow_total` | Inflows minus outflows over the window. | higher is better | total inflows minus total outflows |
| `net_margin_proxy` | Revenue minus operating expenses, divided by revenue. | higher is better | (revenue_total minus expense_total) divided by revenue_total |
| `overdraft_days` | Calendar days with a negative balance. | higher is worse | number of calendar days with a negative daily balance |
| `overdraft_ratio` | Overdraft days divided by observed days. | higher is worse | overdraft_days divided by days_observed |
| `runway_months` | Latest balance divided by mean monthly expenses. | higher is better | last daily balance divided by mean monthly expenses |

## Debt service

| Feature | Description | Direction | Formula |
|---|---|---|---|
| `debt_service_cv` | Coefficient of variation of debt service in active months. | higher is worse | coefficient of variation of loan repayments in active months |
| `debt_service_months_ratio` | Share of months with a debt-service payment. | contextual | share of months with a positive loan repayment |
| `debt_service_ratio` | Loan repayments divided by revenue. | higher is worse | loan_repayment_total divided by revenue_total |
| `dscr` | Cash available before interest divided by principal plus interest, capped at 10. | higher is better | CFADS / (loan_repayment_total + interest_total), clipped to 0 to 10, with CFADS = revenue minus (expenses minus interest); sentinel 10 when the denominator is at most 0 |
| `interest_coverage` | Operating cash available divided by interest, capped at 20. | higher is better | CFADS / interest_total, clipped to 0 to 20, with CFADS = revenue minus (expenses minus interest); sentinel 20 when interest is at most 0 |
| `interest_to_revenue` | Interest expense divided by revenue. | higher is worse | interest_total divided by revenue_total |
| `interest_total` | Total interest and financing charges. | contextual | sum of interest and financing-charge outflows |
| `loan_proceeds_total` | Total loan drawdowns and financing received. | contextual | sum of loan drawdown inflows |
| `loan_repayment_monthly_mean` | Mean monthly loan repayment. | contextual | mean of monthly loan repayments |
| `loan_repayment_total` | Total loan principal and instalment payments. | contextual | sum of loan principal and instalment outflows |
| `loan_to_revenue` | Loan proceeds divided by revenue. | contextual | loan_proceeds_total divided by revenue_total |
| `net_debt_flow` | Loan proceeds minus loan repayments. | contextual | loan_proceeds_total minus loan_repayment_total |

## Distress events

| Feature | Description | Direction | Formula |
|---|---|---|---|
| `bank_fees_to_revenue` | Bank fees divided by revenue. | higher is worse | bank_fees_total divided by revenue_total |
| `bank_fees_total` | Total bank and payment-provider fees. | higher is worse | sum of bank-fee outflows |
| `card_payment_ratio` | Card payments as a share of expenses. | contextual | card_payment_total divided by expense_total |
| `card_payment_total` | Total card and point-of-sale payments. | contextual | sum of card-payment outflows |
| `cash_withdrawal_ratio` | Cash withdrawals as a share of expenses. | higher is worse | cash_withdrawal_total divided by expense_total |
| `cash_withdrawal_total` | Total cash withdrawals. | contextual | sum of cash-withdrawal outflows |
| `distress_flag_count` | Count of triggered distress heuristics (0-8). | higher is worse | number of eight predefined distress heuristics that are true |
| `gambling_to_expense` | Gambling outflows as a share of expenses. | higher is worse | gambling_total divided by expense_total |
| `gambling_total` | Total gambling and betting outflows. | higher is worse | sum of gambling outflows |
| `insurance_to_revenue` | Insurance premiums divided by revenue. | contextual | insurance_total divided by revenue_total |
| `insurance_total` | Total insurance premiums. | contextual | sum of insurance-premium outflows |
| `nsf_count` | Number of non-sufficient-funds or failed payment events. | higher is worse | count of nsf-category transactions |
| `nsf_rate` | NSF events divided by transaction count. | higher is worse | nsf_count divided by transaction count |
| `overdraft_events` | Transitions from a non-negative to a negative balance. | higher is worse | count of transitions from a non-negative to a negative daily balance |
| `returned_payment_count` | Number of explicitly returned or rejected payments. | higher is worse | count of nsf transactions whose normalised text matches the returned-payment pattern |
| `returned_payment_rate` | Returned payments divided by transaction count. | higher is worse | returned_payment_count divided by transaction count |

## Behavioural signals

| Feature | Description | Direction | Formula |
|---|---|---|---|
| `dormant_days_max` | Longest gap between consecutive transactions. | higher is worse | maximum gap in days between consecutive transaction dates |
| `inflow_outflow_correlation` | Correlation between monthly inflows and outflows. | contextual | Pearson correlation of monthly inflows and outflows |
| `internal_transfer_ratio` | Internal transfers as a share of absolute cash flow. | higher is worse | (transfer_in plus transfer_out) divided by total absolute flow |
| `month_end_ratio` | Share of transactions booked in the last three days of a month. | contextual | share of transactions booked on day 28 to 31 of a month |
| `outflow_quality` | Operating expenses as a share of all outflows. | higher is better | expense_total divided by total outflows |
| `recurring_counterparty_ratio` | Outflows to counterparties seen in three or more months. | higher is better | share of expense value paid to counterparties seen in at least three months |
| `refund_ratio` | Refunds as a share of total inflows. | contextual | refunds divided by total inflows |
| `round_amount_ratio` | Share of transactions with a round amount (multiple of 100). | contextual | share of transactions with a strictly positive absolute amount divisible by 100 |
| `salary_day_consistency` | Consistency of the payroll day of month (0-1). | higher is better | 1 minus min(payroll day-of-month standard deviation divided by 10, 1); 0 when there are fewer than three payroll transactions |
| `txn_frequency_trend` | Normalised trend in monthly transaction count. | higher is better | OLS slope of monthly transaction counts divided by mean absolute count |
| `weekend_ratio` | Share of transactions booked on a weekend. | contextual | share of transactions booked on a Saturday or Sunday |

## Trends and seasonality

| Feature | Description | Direction | Formula |
|---|---|---|---|
| `inflow_growth_3m` | Last 3 months inflows versus the prior 3 months. | higher is better | mean of the last 3 months inflows relative to the mean of the prior 3 months |
| `inflow_trend_slope` | Normalised OLS slope of monthly inflows. | higher is better | OLS slope of monthly inflows divided by mean absolute inflow |
| `net_growth_3m` | Last 3 months net cash flow versus the prior 3 months. | higher is better | mean of the last 3 months net cash flow relative to the mean of the prior 3 months |
| `net_trend_slope` | Normalised OLS slope of monthly net cash flow. | higher is better | OLS slope of monthly net cash flow divided by mean absolute net flow |
| `outflow_growth_3m` | Last 3 months outflows versus the prior 3 months. | contextual | mean of the last 3 months outflows relative to the mean of the prior 3 months |
| `outflow_trend_slope` | Normalised OLS slope of monthly outflows. | contextual | OLS slope of monthly outflows divided by mean absolute outflow |
| `revenue_expense_correlation` | Correlation between monthly revenue and expenses. | contextual | Pearson correlation of monthly revenue and expenses |
| `seasonality_strength_expense` | Variance explained by month-of-year expense patterns (0-1). | contextual | 1 minus residual variance over total variance of monthly expenses by month of year |
| `volatility_ratio` | Expense volatility divided by revenue volatility. | higher is worse | coefficient of variation of expenses divided by coefficient of variation of revenue, clipped to 0 to 10 |

## Composite health scores

| Feature | Description | Direction | Formula |
|---|---|---|---|
| `cashflow_health_score` | Weighted composite of the four health scores (0-100). | higher is better | 0.25 stability + 0.35 liquidity + 0.20 leverage + 0.20 distress |
| `cashflow_stability_score` | Heuristic stability score (0-100, higher is better). | higher is better | weighted score of revenue stability, positive months, net-flow stability, balance stability and payroll regularity |
| `distress_health_score` | Heuristic distress score (0-100, higher is better). | higher is better | weighted score of nsf rate, returned rate, gambling, fees and negative months |
| `leverage_health_score` | Heuristic leverage score (0-100, higher is better). | higher is better | weighted score of debt-service ratio, dscr and interest coverage |
| `liquidity_health_score` | Heuristic liquidity score (0-100, higher is better). | higher is better | weighted score of days_cash, runway, overdraft days and negative balance days |

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
