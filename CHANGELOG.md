# Changelog

All notable changes to OpenCashFlow are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses semantic versioning.

## [Unreleased]

## [0.1.0] - 2026-11-22

First public release.

### Added

* Canonical transaction schema with alias detection, European number parsing, signed and unsigned amount handling, and direction repair. `account_id` is always present and is NA when the source has no account column. Company-level identifiers take precedence over account-level identifiers.
* Rule-first multilingual transaction categoriser (English, German, French, Polish, Italian, Dutch and Spanish) with confidence, method and matched-rule metadata, plus a TF-IDF character n-gram fallback. An `equity_injection` category covers capital injections and is excluded from revenue and debt totals.
* 151 cash-flow features across coverage, revenue, expenses, liquidity, debt, distress, behaviour, trends and composite health, with a generated feature dictionary.
* Synthetic SME ledger generator for six European countries with a documented, ledger-grounded default process. Labels come from a forward outcome window (`outcome_months`, default 12). `transactions` holds the observable rows and `outcome_transactions` holds the outcome window.
* LimiX-2-1016 wrapper for classification, regression and imputation, including CPU and Windows triton and nvtx shims, availability diagnostics, categorical encoding and lazy pickle behaviour.
* LightGBM, XGBoost, logistic regression and heuristic scorecard baselines. `LogisticBaseline` takes `numeric_transform`, with `quantile` as the default and `standard` as the alternative.
* Credit-risk metrics: ROC-AUC, KS, Gini, PR-AUC, Brier, log loss, calibration tables, ratings, expected loss, lift and cost-sensitive thresholds.
* TreeSHAP, occlusion and additive explanations with adverse-action style reason codes.
* End-to-end `CashFlowPipeline` with policy decisions, expected loss, reason codes and joblib persistence.
* Loaders for Polish Companies Bankruptcy, German Credit, Bangladesh SME and the 899k-row U.S. SBA National loan dataset, with licences, citations, leakage protection and share-alike notices. The SBA loader takes inclusive `approval_fy_min` and `approval_fy_max` vintage filters.
* Optional macro-context joiner with country lags (`publication_lag`, default 1 month) and year-on-year changes.
* Benchmark harness comparing LimiX-2 with tree, linear and heuristic baselines on stratified folds, with repeats, an out-of-time split, feature exclusion, variants, interval and `predict_seconds` columns. The CLI exposes the same options.
* Fast test suite (`pytest -q`) that passes on a clean checkout. Dataset-dependent tests and five LimiX integration tests skip unless their data or checkpoint is present (`OPENCASHFLOW_RUN_LIMIX=1` runs the LimiX tests).
* Three notebooks: feature engineering, LimiX scoring, and benchmarks.
* Documentation: feature dictionary, dataset notes, LimiX guide, architecture, quickstart and transaction guide.
* Project metadata and URLs point to https://github.com/youssbm/opencashflow. `CITATION.cff` cites the SBA study.
* Apache-2.0 licence, contribution guide, code of conduct, security policy and citation metadata.

### Changed

* Synthetic labels are a 12-month forward outcome. The NSF term counts only exact narrative templates. Benchmark results were regenerated with the published configuration (repeats 2), including the SBA sensitivity runs.
* Feature extraction emits a `UserWarning` when `as_of` is absent; it then scores each entity at its latest transaction date. Without a balance column, a warning says balance-derived features are cumulative-flow proxies. With several accounts, daily balances are summed per account.
* Transaction normalization: `dayfirst` defaults to `None` and infers from the values, with a warning on ambiguous dates. Blank company identifiers fall back to the account identifier. Mixed currencies for one entity raise `TransactionSchemaError`.
* Features: `negative_balance_ratio` is in [0, 1]; `dscr` divides cash available by principal plus interest; returned payments are matched in seven languages; social contributions are payroll. All 151 features carry a one-line formula in the generated dictionary.
* Baselines default to `class_weight=None`, so probabilities are not reweighted.
* `LimiXScorer.softmax_temperature` defaults to `None`, which resolves to 0.9 for classification and 1.0 for regression. The LimiX dataset name includes a SHA-256 digest of the training data.
* Occlusion explanations select features per row by standardised deviation from the reference and use the training matrix as background. Reason codes list only risk-increasing drivers, largest first.
* Benchmark defaults are 300 companies and 20 months. ROC-AUC intervals use the t critical value. An out-of-time split is available. `predict_seconds` is reported in the summary tables.
* The benchmark CLI writes files only when `--output-dir` is supplied.

### Fixed

* Direction-specific categorisation rules are consulted when no direction is supplied; `direction="any"` no longer skips them.
* LimiX imputation replaces only missing cells and returns observed cells unchanged. A cell LimiX leaves as NaN falls back to the training median with a `UserWarning`.
* `parse_amount` reads a single separator after a zero integer part as decimal and rejects repeated separators unless every group after the first has three digits. Mixed-format date columns are parsed per value with explicit formats, so ISO and day-first dates both survive and the result does not depend on the pandas version. Missing currency values default to the configured currency with one warning.
* `LogisticBaseline` keeps all-NaN numeric and categorical columns as constant columns during imputation instead of dropping them.
* The missing-balance-account warning is emitted when an account has transactions but no balance rows, even if only one account of the entity has balances.
* Occlusion gives a NaN contribution to features whose reference value is NaN; reason codes ignore it.

### Known limitations

* Public SME datasets are tabular accounting or application data, not transaction ledgers; only the synthetic portfolio exercises the full cash-flow pipeline end to end. The SBA dataset is U.S. government-guaranteed lending and carries CC BY-SA 4.0 share-alike terms.
* LimiX-2 weights are non-commercial and are not bundled.
* The policy thresholds are illustrative, not calibrated to a lender's risk appetite.
* Explanations are model-relative and not causal.
* The benchmark uses one LimiX ensemble pipeline on CPU; results are not directly comparable to published leaderboard runs with 32 members on GPU.
* SBA Term is not verified as known at approval, so SBA results must not be read as approval-time PD.
* LimiX was not run out of time; out-of-time results cover the tree and logistic baselines only.
* `interest_coverage` is capped at 20 and the cap binds for 49% of synthetic firms, so the feature carries no information above 20 for about half the sample.
* Categoriser accuracy is not measured; the synthetic generator does not write true categories.
* No calibration step is shipped. Check Brier score and log loss, and use `calibration_table` to inspect probabilities. Calibration is on the roadmap.
* Verified locally on Python 3.13 only. The CI matrix covers Python 3.10 to 3.13, but the 3.10 to 3.12 runs were not done locally.
