# Changelog

All notable changes to OpenCashFlow are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses semantic versioning.

## [0.1.0] - 2026-11-21

### Added

* Canonical transaction schema with alias detection, European number parsing, signed/unsigned
  amount handling and direction repair.
* Rule-first multilingual transaction categoriser (English, German, French, Polish, Italian, Dutch,
  Spanish) with confidence, method and matched-rule metadata, plus a TF-IDF character n-gram
  fallback.
* 151 cash-flow features across coverage, revenue, expenses, liquidity, debt, distress, behaviour,
  trends and composite health, with a generated feature dictionary.
* Synthetic SME ledger generator for six European countries with a documented, ledger-grounded
  default process and a multi-company portfolio API.
* LimiX-2-1016 wrapper for classification, regression and imputation, including CPU/Windows triton
  and nvtx shims, availability diagnostics, categorical encoding and lazy pickle behaviour.
* LightGBM, XGBoost, logistic regression and heuristic scorecard baselines.
* Credit-risk metrics: ROC-AUC, KS, Gini, PR-AUC, Brier, log loss, calibration tables, ratings,
  expected loss, lift and cost-sensitive thresholds.
* TreeSHAP, occlusion and additive explanations with adverse-action style reason codes.
* End-to-end `CashFlowPipeline` with policy decisions, expected loss, reason codes and joblib
  persistence.
* Loaders for Polish Companies Bankruptcy, German Credit, Bangladesh SME and the 899k-row U.S. SBA
  National loan dataset, with licences, citations, leakage protection and share-alike notices.
* Optional macro-context joiner with country lags and year-on-year changes.
* Benchmark harness comparing LimiX-2 with tree, linear and heuristic baselines on stratified folds.
* 200+ tests, including opt-in LimiX integration tests.
* Three notebooks: feature engineering, LimiX scoring, and benchmarks.
* Documentation: feature dictionary, dataset notes, LimiX guide, architecture, quickstart and
  transaction guide.
* Apache-2.0 licence, contribution guide, code of conduct, security policy and citation metadata.

### Known limitations

* Public SME datasets are tabular accounting/application data, not transaction ledgers; only the
  synthetic portfolio exercises the full cash-flow pipeline end to end. The SBA dataset is U.S.
  government-guaranteed lending and carries CC BY-SA 4.0 share-alike terms.
* LimiX-2 weights are non-commercial and are not bundled.
* The policy thresholds are illustrative, not calibrated to a lender's risk appetite.
* Explanations are model-relative and not causal.
* The benchmark uses one LimiX ensemble pipeline on CPU; results are not directly comparable to
  published leaderboard runs with 32 members on GPU.
