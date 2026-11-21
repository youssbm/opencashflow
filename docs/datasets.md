# Datasets

OpenCashFlow ships loaders for three public credit-risk datasets and one generator for
transaction-level SME ledgers. Raw data is downloaded on demand and never redistributed with the
repository.

| Key | Dataset | Rows | Features | Default rate | Licence | Citation |
|---|---|---:|---:|---:|---|---|
| `polish` | Polish Companies Bankruptcy | 7,027 (1year) to 10,503 (3year) | 64 financial ratios | 3.9% to 6.9% | CC BY 4.0 | [10.24432/C5F600](https://doi.org/10.24432/C5F600) |
| `german` | Statlog German Credit | 1,000 | 20 application attributes | 30.0% | CC BY 4.0 | [10.24432/C5NC77](https://doi.org/10.24432/C5NC77) |
| `bangladesh_sme` | Bangladesh SME Credit | 1,200 | 19 tabular (plus 4 text) | 25.25% | CC BY 4.0 data, MIT code | [10.5281/zenodo.18525278](https://doi.org/10.5281/zenodo.18525278) |
| `synthetic` | OpenCashFlow synthetic portfolio | configurable | 151 cash-flow features | configurable | Apache-2.0 | this repository |

## Quick download

```bash
python data/download.py --list
python data/download.py --dataset polish german bangladesh_sme
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
* **Horizons:** `1year` predicts bankruptcy within five years; `5year` predicts within one year.
  The loader supports `1year` to `5year` and `all` (concatenated with a `horizon_years` column).
* **Primary benchmark:** `1year`, 7,027 rows, 271 bankruptcies.
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
* **Caveat:** 1,000 rows and 30% bad rate make this a small, high-variance benchmark. Report
  confidence intervals or repeated folds.

## Bangladesh SME Credit

* **Source:** Zenodo record 18525278.
* **Citation:** Chowdhury, S., Das, A., & Paul, S. (2026). *Structured Data Suffices: Machine
  Learning for SME Credit Risk Assessment in an Emerging Market - Dataset and Analysis Code*.
  Zenodo. https://doi.org/10.5281/zenodo.18525278. Data CC BY 4.0, code MIT.
* **Content:** 1,200 anonymised SME loan applications from Bangladesh with structured features,
  text narratives and a binary `target_missed_payment` label. A `split` column marks the authors'
  train/test split.
* **Leakage warning:** `missed_payment_last_12m` is an exact restatement of the target
  (897 "No" rows are all 0; 303 "Yes" rows are all 1). The loader drops it by default. Without this
  fix, every model scores AUC 1.0. Pass `include_leakage=True` only for teaching.
* **Text:** four narrative columns are dropped by default. Set `include_text=True` for NLP
  experiments.
* **Mirror:** Zenodo may reject automated requests with HTTP 403. The loader falls back to the
  authors' GitHub mirror and records the fallback in `data/raw/bangladesh_sme_SOURCE.txt`.

## Synthetic SME ledgers

```python
from opencashflow.data.synthetic import CompanyProfile, generate_portfolio

portfolio = generate_portfolio(n_companies=300, months=24, default_rate=0.12, seed=42)
print(portfolio.summary())
X, y = portfolio.to_features()
```

The generator produces localised narratives for Germany, France, Poland, Italy, the Netherlands and
Spain. Each company has customer receipts, payroll, rent, utilities, supplier invoices, VAT, profit
tax, debt service, bank fees, card spending, cash withdrawals, occasional refunds and internal
transfers. Distressed companies additionally show failed payments, gambling, loan drawdowns and
capital injections.

Default labels are sampled from a logistic function of **post-ledger outcomes**:

* minimum and ending balance relative to monthly revenue,
* share of months with negative net cash flow,
* NSF / returned-payment count,
* debt-service ratio,
* inflow growth.

The latent `stress` variable is **not** a feature. This keeps the benchmark learnable from
observable cash-flow behaviour rather than from a directly leaked label driver. The data-generating
process is a benchmark, not a calibrated model of any real bank portfolio.

## Benchmark protocol

The saved results in `benchmarks/results/` use:

* stratified 3-fold cross-validation, seed 42;
* the same folds for every model;
* preprocessing fitted inside each training fold;
* LimiX-2 with one ensemble pipeline on CPU (`n_pipelines=1`) to keep the benchmark reproducible;
* no test-fold information used for feature selection or calibration.

Reproduce with:

```bash
python benchmarks/run_benchmark.py \
  --datasets synthetic,german,bangladesh_sme,polish \
  --models heuristic,logistic,lightgbm,xgboost,limix \
  --cv 3 --synthetic-companies 300 --synthetic-months 20 --limix-pipelines 1
```

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
be lagged to the observation date to avoid look-ahead bias.

## Attribution checklist

When publishing results, include:

1. dataset name, version/variant and DOI;
2. licence;
3. row count, feature count and default rate;
4. fold count and random seed;
5. model name and, for LimiX, the checkpoint and code commits plus the non-commercial licence;
6. any preprocessing that differs from the loaders in this repository.
