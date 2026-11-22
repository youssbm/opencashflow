# Datasets

OpenCashFlow ships loaders for four public credit-risk datasets and one generator for
transaction-level SME ledgers. Raw data is downloaded on demand and never redistributed with the
repository.

| Key | Dataset | Rows | Features | Default rate | Licence | Citation |
|---|---|---|---:|---:|---|---|
| `polish` | Polish Companies Bankruptcy | 7,027 (1year) to 10,503 (3year) | 64 financial ratios | 3.9% to 6.9% | CC BY 4.0 | [10.24432/C5F600](https://doi.org/10.24432/C5F600) |
| `german` | Statlog German Credit | 1,000 | 20 application attributes | 30.0% | CC BY 4.0 | [10.24432/C5NC77](https://doi.org/10.24432/C5NC77) |
| `bangladesh_sme` | Bangladesh SME Credit | 1,200 | 19 tabular (plus 4 text) | 25.3% | CC BY 4.0 data, MIT code | [10.5281/zenodo.18525278](https://doi.org/10.5281/zenodo.18525278) |
| `sba` | SBA National Loan Data | 897,167 labelled | 21 features | 17.6% | CC BY-SA 4.0 | [Kaggle](https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied) and [Li et al. (2018)](https://doi.org/10.1080/10691898.2018.1434342) |
| `synthetic` | OpenCashFlow synthetic portfolio | 300 companies (published) | 151 cash-flow features; 20-month history, 12-month forward label | 11.3% (published) | Apache-2.0 | this repository |

## Quick download

```bash
python data/download.py --list
python data/download.py --dataset polish german bangladesh_sme
python data/download.py --dataset sba
python data/download.py --dataset all --data-dir ./data/raw
```

Or through the package CLI:

```bash
opencashflow download polish german bangladesh_sme
```

The default landing zone is `data/raw`, overridable with `OPENCASHFLOW_DATA_DIR`.

## Load in Python

```python
from opencashflow.data.loaders import load_dataset

polish = load_dataset("polish", variant="1year")
german = load_dataset("german", decode=True)
bangladesh = load_dataset("bangladesh_sme")
sba = load_dataset("sba", max_rows=20_000)
synthetic = load_dataset("synthetic", n_companies=300, months=20)

print(polish.summary())
print(polish.X.shape, polish.y.mean())
```

Every loader returns a `DatasetBundle` with `X`, `y`, `categorical_features`, `source`, `license`,
`citation`, `metadata` and, where available, the provided `split`.

## Polish Companies Bankruptcy

* **Source:** UCI Machine Learning Repository, dataset 365.
* **Citation:** Tomczak, S. (2016). *Polish Companies Bankruptcy*. UCI.
  https://doi.org/10.24432/C5F600. CC BY 4.0.
* **Content:** 64 financial ratios computed from EMIS financial statements. Bankrupt companies
  span 2000-2012; operating companies span 2007-2013.
* **Horizons:** `1year` (UCI "1stYear" file) uses first-year ratios and labels bankruptcy five years
  later; `5year` (UCI "5thYear" file) labels bankruptcy one year later.
  The loader supports `1year` to `5year` and `all` (concatenated with a `horizon_years` column).
* **Primary benchmark:** `1year`, 7,027 rows, 3.9% bankrupt. See
  [the main results](../benchmarks/results/benchmark_summary.md).
* **Duplicate rows:** 164 rows belong to exact duplicate feature groups, all non-bankrupt. They are
  kept, and the count is reported in the bundle metadata as `duplicate_feature_rows`.
* **Caveat:** these are accounting ratios, not bank transactions. They are a useful external
  benchmark for the modelling layer, not a test of the cash-flow feature engine.

## Statlog German Credit

* **Source:** UCI Machine Learning Repository, dataset 144.
* **Citation:** Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI.
  https://doi.org/10.24432/C5NC77. CC BY 4.0.
* **Content:** 1,000 loan applications, 20 coded attributes, target 1 = good, 2 = bad.
* **Decoding:** `load_german_credit(decode=True)` maps codes such as `A34` to readable labels.
  The original `german.data` file is preserved; the numeric variant is available with
  `numeric=True`.
* **Cost matrix:** approving a bad borrower costs 5; rejecting a good borrower costs 1. Use
  `cost_sensitive_threshold(..., cost_false_positive=1, cost_false_negative=5)`.
* **Protected attributes:** `personal_status_sex`, `age_years` and `foreign_worker` can act as
  proxies for sex, age and nationality. Pass `exclude_protected=True` to drop them. Review before
  any lending use.
* **Caveat:** 1,000 rows and 30% bad rate make this a small, high-variance benchmark. Report
  confidence intervals or repeated folds.

## Bangladesh SME Credit

* **Source:** Zenodo record 18525278.
* **Version DOI:** https://doi.org/10.5281/zenodo.18525278.
* **Citation:** the Zenodo record lists Shoumya Chowdhury and Anmita Das as creators, and the
  citation follows it. The upstream main README and the `zenodo` branch disagree on the author list
  and the data period, so the record is treated as the authority.
* **Content:** 1,200 anonymised SME loan applications from Bangladesh with structured features,
  text narratives and a binary `target_missed_payment` label. A `split` column marks the authors'
  train/test split.
* **Target:** "missed any scheduled payment in the 12 months after disbursement", as described in
  the upstream README. This is broader than a 90-days-past-due default.
* **Leakage warning:** `missed_payment_last_12m` is an exact restatement of the target
  (897 "No" rows are all 0; 303 "Yes" rows are all 1). The loader drops it by default. Without this
  fix, every model scores AUC 1.0. Pass `include_leakage=True` only for teaching.
* **Text:** four narrative columns are dropped by default. Set `include_text=True` for NLP
  experiments.
* **Mirror:** Zenodo may reject automated requests with HTTP 403. The loader falls back to the
  authors' GitHub mirror and records the fallback in `data/raw/bangladesh_sme_SOURCE.txt`.

## SBA National loan data

* **Source:** Kaggle upload by Mirbek Toktogaraev, built from U.S. Small Business Administration
  loan data.
* **Citations:**
  * Toktogaraev, M. (2020). *Should This Loan be Approved or Denied?* [Dataset]. Kaggle.
    https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied
  * Li, M., Mickel, A. & Taylor, S. (2018). "Should This Loan be Approved or Denied?: A Large Dataset
    with Class Assignment Guidelines". *Journal of Statistics Education* 26(1), 55 to 66.
    https://doi.org/10.1080/10691898.2018.1434342. These details come from search listings, not
    the full text.
* **Licence:** CC BY-SA 4.0. This is a **share-alike** licence. The OpenCashFlow code remains
  Apache-2.0, but derived datasets may need to be shared under CC BY-SA 4.0. Consult your legal team
  before redistributing feature matrices built from these rows.
* **Content:** 899,164 SBA-guaranteed loans, 27 raw columns, approval dates from 1966-05-18 to
  2014-06-25. The target is `MIS_Status`: `CHGOFF` = 1 (default), `P I F` = 0 (paid in full). 1,997
  rows have a missing target and are dropped, leaving 897,167 rows and a 17.6% default rate.
* **Features:** the loader keeps 21 columns: approval fiscal year and month, term, employees, jobs
  created/retained, gross and SBA-approved amounts, SBA guarantee ratio, loan per employee, jobs per
  employee, state, bank state, NAICS sector, new/existing business, urban/rural, revolving line of
  credit, LowDoc flag and franchise flag.
* **Leakage removal:** five post-approval or post-default columns are excluded from the feature
  matrix:
  * `ChgOffDate`: the charge-off date, recorded after the loan outcome; never loaded for
    approval-time models.
  * `DisbursementDate`: disbursement happens after approval; never loaded for approval-time models.
  * `DisbursementGross`: the disbursed amount, known only after approval.
  * `BalanceGross`: the outstanding balance, known only after approval or during servicing.
  * `ChgOffPrinGr`: the principal charged off, recorded only after default.
  The last three are loaded only with `include_post_approval=True`, which is for servicing
  research, never for approval-time scoring.
* **Identifiers:** borrower name, city, zip, bank and loan id are excluded from `X` by default.
  Use `include_identifiers=True` for auditing, not modelling.
* **Size:** 899k rows x 21 features is too large for CPU in-context LimiX. Use a stratified cap:

  ```python
  bundle = load_dataset("sba", max_rows=20_000, random_state=42)
  ```

  The cap is applied after the target and leakage cleaning, so the default rate is preserved.
* **Date handling:** the raw dates are `DD-Mon-YY`. The loader uses a fixed century window so `68`
  becomes 1968, not 2068. `ApprovalFY` values such as `1976A` fall back to the parsed date.
* **Matured bound:** set `approval_fy_max=2008` (reported as variant `matured-2008`) to restrict
  approvals to 2008 or earlier and to score on approvals from 2004 to 2008, which gives the loans
  time to mature. The saved run trains on 12,206 rows and tests on 7,794 rows.
* **Term sensitivity:** most random-CV performance depends on `term_months` and `term_years`. Out of
  time without Term, AUC is about 0.6. Whether Term is fixed at approval has NOT been verified, so
  SBA results must not be described as approval-time PD until it is.
* **Caveats:** this is U.S. government-guaranteed lending, not European open-banking data. It is a
  useful external benchmark for credit-risk modelling, not a test of SEPA categorisation or
  cash-flow features.

### SBA benchmark results

Mean ROC-AUC. Bracketed values are the approximate interval from repeated cross-validation; the
out-of-time runs use a single split and have no interval. LimiX is not run out of time because it is
too slow at these sample sizes.

| Setup | LimiX | XGBoost | LightGBM | Logistic | Result file |
|---|---|---|---|---|---|
| Random CV, 21 features, 5,000 rows | 0.976 [0.971, 0.981] | 0.960 [0.950, 0.969] | 0.958 [0.948, 0.969] | 0.854 [0.843, 0.864] | [sba](../benchmarks/results/sba/benchmark_summary.md) |
| Random CV, no Term, 19 features | 0.823 [0.808, 0.839] | 0.802 [0.788, 0.816] | 0.799 [0.784, 0.815] | 0.734 [0.717, 0.751] | [sba_no_term](../benchmarks/results/sba_no_term/benchmark_summary.md) |
| Out of time, 21 features | not run | 0.947 | 0.948 | 0.784 | [sba_oot](../benchmarks/results/sba_oot/benchmark_summary.md) |
| Out of time, no Term | not run | 0.620 | 0.631 | 0.586 | [sba_oot_no_term](../benchmarks/results/sba_oot_no_term/benchmark_summary.md) |
| Matured bound, 21 features | not run | 0.957 | 0.957 | 0.803 | [sba_oot_matured](../benchmarks/results/sba_oot_matured/benchmark_summary.md) |
| Matured bound, no Term | not run | 0.631 | 0.639 | 0.581 | [sba_oot_matured_no_term](../benchmarks/results/sba_oot_matured_no_term/benchmark_summary.md) |

## Synthetic SME ledgers

```python
from opencashflow.data.synthetic import generate_portfolio

portfolio = generate_portfolio(n_companies=300, months=20, seed=42)
print(portfolio.summary())
print(portfolio.metadata["observation_end"])
X, y = portfolio.to_features()
```

The generator produces localised narratives for Germany, France, Poland, Italy, the Netherlands and
Spain. Each company has customer receipts, payroll, rent, utilities, supplier invoices, VAT, profit
tax, debt service, bank fees, card spending, cash withdrawals, occasional refunds and internal
transfers. Distressed companies additionally show failed payments, gambling, loan drawdowns and
capital injections.

`months` sets the observable history that features see. `outcome_months` sets the forward window
used for labels and defaults to 12, so the label is a 12-month forward outcome.
`portfolio.transactions` holds observable rows only; `portfolio.outcome_transactions` holds the
forward rows, and `portfolio.metadata["observation_end"]` is the last observable day.

A latent `stress` variable drives both the ledger and the default probability. It is not a feature.
The label is sampled from a logistic function of post-ledger outcomes:

* minimum and ending balance relative to monthly revenue;
* share of months with negative net cash flow;
* NSF / returned-payment count, which counts only exact NSF narrative templates (a substring match
  previously counted `TRANSFER` as an NSF event);
* debt-service ratio;
* inflow growth.

The published configuration is 300 companies, 20 months, seed 42, with `observation_end` at
2024-08-31 and a realised default rate of 11.3%. The oracle AUC, the true default probability
against the realised label, is 0.899 at this configuration. After the NSF fix, country default
rates are not significantly different (chi-square p = 0.547): the earlier Spanish gap came from the
NSF bug, and the earlier Polish gap was within sampling noise.

The untrained heuristic scorecard is not an independent baseline on synthetic data because its risk
drivers overlap with the label's. The data-generating process is a benchmark, not a calibrated model
of any real bank portfolio.

## Benchmark protocol

The saved results in `benchmarks/results/` use:

* stratified 3-fold cross-validation with seed 42; the published tables set `repeats=2`, so each
  model has 6 runs (2 repeats times 3 folds); repeat i reshuffles the folds with seed + i - 1;
* the same folds for every model; preprocessing fitted inside each training fold; LimiX sees only
  the training fold as its reference set; one LimiX ensemble pipeline on CPU (`n_pipelines=1`);
* intervals are the mean plus or minus t(0.975, n-1) times sd over the square root of n, computed
  over the runs. Folds share training rows, so the interval is approximate and understates
  uncertainty;
* out-of-time runs use one split with no repeats: training rows have `approval_fy` up to the cutoff
  and test rows have `approval_fy` after it. LimiX is not run out of time;
* `--exclude-features` removes named columns (used for the no-Term variants) and `--variant` names
  the saved output (for example `no-term` or `matured-2008`);
* `predict_seconds` is reported per fold;
* for the 899k-row SBA data, a stratified sample is used rather than the full ledger: 5,000 rows
  for random CV and 20,000 rows for the out-of-time runs;
* no test-fold information used for feature selection or calibration.

Reproduce with:

```bash
python benchmarks/run_benchmark.py \
  --datasets synthetic,german,bangladesh_sme,polish \
  --models heuristic,logistic,lightgbm,xgboost,limix \
  --cv 3 --repeats 2 --synthetic-companies 300 --synthetic-months 20 \
  --limix-pipelines 1 --output-dir benchmarks/results
```

For SBA random CV:

```bash
python benchmarks/run_benchmark.py --datasets sba \
  --models logistic,lightgbm,xgboost,limix --cv 3 --repeats 2 \
  --max-rows 5000 --limix-pipelines 1 --output-dir benchmarks/results/sba
```

The no-Term variant adds `--exclude-features term_months,term_years --variant no-term` and writes to
`benchmarks/results/sba_no_term`. Out-of-time runs set `--split oot --time-column approval_fy
--oot-cutoff 2003 --max-rows 20000`; the matured variant adds `--sba-approval-fy-max 2008`.

Saved results: [main](../benchmarks/results/benchmark_summary.md),
[sba](../benchmarks/results/sba/benchmark_summary.md),
[sba_no_term](../benchmarks/results/sba_no_term/benchmark_summary.md),
[sba_oot](../benchmarks/results/sba_oot/benchmark_summary.md),
[sba_oot_no_term](../benchmarks/results/sba_oot_no_term/benchmark_summary.md),
[sba_oot_matured](../benchmarks/results/sba_oot_matured/benchmark_summary.md) and
[sba_oot_matured_no_term](../benchmarks/results/sba_oot_matured_no_term/benchmark_summary.md).

Metrics: ROC-AUC, KS, PR-AUC, Brier score and log loss. AUC is the primary ranking metric; Brier and
log loss expose calibration problems that AUC hides.

## Optional macro context

Eurostat and EBA aggregate indicators can be joined as external columns. The toolkit does not
bundle a macro database because country coverage, vintages and revision policies differ by use case.
Use `opencashflow.data.macro.MacroFeatureJoiner` with a tidy table:

| country | period | indicator | value |
|---|---|---|---|
| DE | 2024-01-01 | business_confidence | 96.2 |
| DE | 2024-02-01 | business_confidence | 95.4 |
| PL | 2024-01-01 | business_confidence | 98.1 |

Eurostat publishes the original series through its REST API; EBA publishes risk indicators through
its data portal. Record the extraction date and vintage alongside the features. Macro features must
be lagged to the observation date to avoid look-ahead bias. `publication_lag` defaults to 1 month;
lag 0 means the latest value published by the scoring date.

## Attribution checklist

When publishing results, include:

1. dataset name, version/variant and DOI;
2. licence;
3. row count, feature count and default rate;
4. fold count and random seed;
5. model name and, for LimiX, the checkpoint and code commits plus the non-commercial licence;
6. any preprocessing that differs from the loaders in this repository;
7. for SBA, the stratified row cap and the CC BY-SA 4.0 share-alike notice;
8. for SBA, the Li, Mickel and Taylor (2018) reference
   (https://doi.org/10.1080/10691898.2018.1434342) and a note that Term's availability at approval
   is unverified.
