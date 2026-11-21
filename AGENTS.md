# AGENTS.md - OpenCashFlow

This file is the operating manual for AI agents and maintainers working in this repository. Read it
before changing code. It records the contracts that keep the project honest, reproducible and
usable by credit-risk practitioners.

## 1. What this project is

OpenCashFlow is an open-source cash-flow underwriting toolkit for B2B / SME credit risk. It turns
raw open-banking-style transactions into credit-relevant features and a probability of default
(PD), with LimiX-2 as the primary tabular foundation model and LightGBM / XGBoost / logistic
regression as baselines.

The project must stay:

* **Original** - do not copy code or prose from closed commercial products.
* **Auditable** - every decision and feature must be explainable.
* **Reproducible** - fixed seeds, pinned model/config versions, no hidden state.
* **Honest about limitations** - especially leakage, small samples, and the non-commercial LimiX
  weights.
* **EU-aware** - EUR defaults, SEPA-style narratives, GDPR/PSD2 framing where relevant.

## 2. Non-negotiable constraints

### LimiX version

* Use **`stable-ai/LimiX-2-1016`** weights at commit `bbbf26c`.
* Use code at commit `a917fa0` ("resolve randomness in inference").
* Use the V2 configs: `cls_default_noretrieval_v2.json`, `reg_default_noretrieval_v2.json`,
  `reg_default_noretrieval_MVI_v2.json`.
* Never use the withdrawn `stable-ai/LimiX-2` checkpoint for leaderboard claims. It was
  contaminated (TabArena issue #648).
* Regression leaderboard reproduction requires `softmax_temperature=1.0`.

### LimiX licence

The weights are under the **StableAI LimiX Non-Commercial License v1.0**. Commercial use requires a
separate agreement. Disclose the model name "LimiX" when publishing results. The code in this
repository is Apache-2.0; the weights are not.

### Data leakage

* Never train on a column that restates the target.
* The Bangladesh SME dataset contains `missed_payment_last_12m`, which is an exact copy of
  `target_missed_payment`. The loader drops it by default. Do not "improve" the model by adding it
  back.
* For transaction features, respect point-in-time scoring. Do not use the future: `as_of` must be
  at or before the label observation date.
* For synthetic labels, do not expose the latent `stress` variable as a feature. The generator
  intentionally labels from post-ledger outcomes.

### Scope discipline

* Do not add dependencies that are not needed for the task.
* Do not refactor unrelated code.
* Do not delete user work or data files.
* Do not commit raw datasets, checkpoints, or credentials.

## 3. Repository map

```
opencashflow/
  __init__.py                 Public namespace and lazy optional imports
  py.typed                    PEP 561 marker
  pipeline.py                 CashFlowPipeline, UnderwritingPolicy, decisions, persistence
  benchmark.py                Cross-validated benchmark harness
  cli.py                      `opencashflow` command line
  features/
    ratios.py                 Pure numerical helpers (safe_div, HHI, entropy, slope, drawdown)
    categorizer.py            Rule-first multilingual categoriser + TF-IDF fallback
    specs.py                  Single source of truth for all 151 feature definitions
    cashflow_features.py      FeatureConfig, CashFlowFeatureExtractor, FeatureBuilder
  models/
    metrics.py                AUC, KS, Gini, Brier, calibration, rating, expected loss
    baselines.py              TreeBaseline, LogisticBaseline, HeuristicScorer, factory
    limix_scorer.py           LimiX-2 wrapper (classification, regression, imputation)
    explain.py                TreeSHAP, occlusion, additive explanations, reason codes
  data/
    schema.py                 Canonical transaction schema and normalisation
    synthetic.py              Synthetic SME ledger and portfolio generator
    loaders.py                UCI / Zenodo / synthetic dataset loaders
    macro.py                  Optional lagged Eurostat / EBA macro joiner
    download.py               Download CLI
tests/                        pytest suite, LimiX tests are opt-in
benchmarks/                   Benchmark CLI and saved results
docs/                         Feature dictionary, dataset notes, LimiX guide, examples
notebooks/                    End-to-end demos
data/                         Download script and raw-data landing zone (gitignored)
```

## 4. Environment setup

### Core

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[trees,dev]"
```

### LimiX on this Windows CPU machine

The working local install is:

* code: `C:\Users\youss\Documents\Code\tabularFM_tests\limix_src` at `a917fa0`
* checkpoint: `C:\Users\youss\Documents\Code\tabularFM_tests\cache_limix\LimiX-2-1016.ckpt`
* configs: `C:\Users\youss\Documents\Code\tabularFM_tests\limix_src\config`

Set these before running LimiX tests or notebooks:

```powershell
$env:LIMIX_MODEL_PATH="C:\Users\youss\Documents\Code\tabularFM_tests\cache_limix\LimiX-2-1016.ckpt"
$env:LIMIX_SRC="C:\Users\youss\Documents\Code\tabularFM_tests\limix_src"
```

LimiX imports `triton` even on CPU. `opencashflow.models.limix_scorer` installs a no-op shim when
triton is absent. Do not remove it unless the target machine has a GPU and matching triton/flash-attn.

`nvtx` is optional and also shimmed when missing.

## 5. Canonical transaction contract

`normalize_transactions` accepts aliases and returns:

| Column | Meaning |
|---|---|
| `entity_id` | Borrower/account identifier |
| `transaction_id` | Unique row id |
| `date` | Normalised booking date |
| `amount` | Signed amount, positive = credit |
| `currency` | ISO code, default EUR |
| `description` | Free-text narrative |
| `counterparty` | Optional merchant/counterparty |
| `balance` | Optional running balance |
| `direction` | `inflow` for amount > 0, otherwise `outflow` |
| `category` | Optional pre-existing label; NA triggers categorisation |

Rules:

* Zero amounts are `outflow` for direction purposes and are dropped by the synthetic generator.
* Unsigned exports require a direction column. If neither signed amounts nor a direction column are
  present, raise rather than guessing.
* `parse_amount` handles `1.234,56`, `1,234.56`, currency symbols and accounting parentheses.
* Always pass `as_of` for production scoring and out-of-time validation.

## 6. Feature contract

* **All features live in `features/specs.py`.** `FeatureBuilder.add(name, value)` raises `KeyError`
  if a name is not declared. `to_series()` raises if any declared feature was not computed.
* The current catalogue has **151 features** in nine groups: coverage, revenue, expenses,
  liquidity, debt, distress, behaviour, trends, composite.
* Feature names are stable API. Renaming requires a deprecation note and a CHANGELOG entry.
* All outputs are finite floats; `inf` and `nan` become 0, then values are clipped.
* `higher_is_better` is from the lender's perspective and is for explanation/monitoring. It is not
  a causal claim and not enforced by the models.
* Monthly features include partial boundary months. `months_observed` and `coverage_ratio` tell the
  model how much history exists. Do not silently drop partial months.
* `include_other_inflow_as_revenue` is off by default. Enable it only for raw bank exports where
  customer receipts lack invoice keywords and internal transfers are rare.
* To add a feature:
  1. Add a `FeatureSpec` in `features/specs.py`.
  2. Add `builder.add("name", value)` in `features/cashflow_features.py`.
  3. Add a test assertion or fixture check.
  4. Regenerate `docs/features.md` with `python scripts/generate_feature_docs.py`.
  5. Run `pytest tests/test_features.py`.

## 7. Model contract

Every supervised scorer exposes:

```python
model.fit(X, y)
model.predict_proba(X)   # (n, n_classes), rows sum to 1
model.predict(X)         # labels
model.predict_pd(X)      # P(positive class), binary only
```

* The positive class is the second class in sorted order. Encode default as `1`.
* `TreeBaseline` ordinal-encodes categorical columns and passes them to LightGBM as categorical.
  XGBoost receives ordinal codes; document this limitation.
* `LogisticBaseline` imputes, scales, and one-hot encodes inside a scikit-learn pipeline. Stronger
  regularisation is needed for p >> n; the benchmark uses larger samples.
* `HeuristicScorer` is a fixed, untrained scorecard. It only knows cash-flow feature names and is
  skipped on non-transaction datasets.
* `LimiXScorer.fit` does **not** train. It stores the in-context reference set. `predict` loads the
  checkpoint on first use and keeps it in memory until `close()`.

## 8. LimiX wrapper runbook

```python
from opencashflow.models.limix_scorer import LimiXScorer

LimiXScorer.availability()          # check code, checkpoint and configs without loading weights
scorer = LimiXScorer(n_pipelines=1) # fast CPU configuration
scorer.fit(X_train, y_train)        # in-context reference set
pd_values = scorer.predict_pd(X_test)
proba = scorer.predict_proba(X_test)
imputed = scorer.impute(X_with_nan)
scorer.close()
```

Implementation notes:

* `n_pipelines=1` for tests; `4` is the default; the released config has `32`.
* The first `predict` loads the 1.6 GB checkpoint (about 5 s on this machine). Subsequent calls
  reuse it.
* `n_pipelines=1` is about 1 s per small dataset on CPU. Do not run 32 members on CPU for tests.
* Classification returns `(n, n_classes)` probabilities in sorted class order.
* `predict_pd` requires exactly two classes and returns the column for class `1` when present.
* Imputation uses `reg_default_noretrieval_MVI_v2.json`, not the classification config. LimiX
  returns train rows followed by query rows; the wrapper slices the query block.
* Categorical columns are ordinal-encoded using categories learned during `fit`. Unseen categories
  map to `-1`. This is deliberate; do not one-hot encode into hundreds of columns before LimiX.
* The predictor is dropped from pickles via `__getstate__`; it reloads lazily. The checkpoint is not
  bundled.
* If `availability()` is false, read its `reasons` list. Set `LIMIX_MODEL_PATH`, `LIMIX_CONFIG_DIR`
  or `LIMIX_SRC` as needed.

## 9. Data rules

| Dataset | Loader | Licence | Notes |
|---|---|---|---|
| Polish Companies Bankruptcy | `load_polish_bankruptcy` | CC BY 4.0 | 64 ratios, 5 horizons, DOI 10.24432/C5F600 |
| German Credit | `load_german_credit` | CC BY 4.0 | 20 attributes, cost matrix 5:1, DOI 10.24432/C5NC77 |
| Bangladesh SME | `load_bangladesh_sme` | CC BY 4.0 (data), MIT (code) | 1,200 rows, DOI 10.5281/zenodo.18525278, leakage column dropped |
| Synthetic | `load_synthetic` | Apache-2.0 | generated locally, documented distress process |

* Download with `python data/download.py --dataset all` or the `opencashflow download` CLI.
* Raw data goes to `data/raw` (or `OPENCASHFLOW_DATA_DIR`) and is gitignored.
* Zenodo blocks some automated downloads with HTTP 403. The loader falls back to the authors'
  GitHub mirror and writes `bangladesh_sme_SOURCE.txt`; the canonical DOI remains the citation.
* Do not redistribute raw datasets in this repository. Point to the source instead.
* When adding a dataset: implement a loader, add a `DATASET_REGISTRY` entry with licence and
  citation, add a fixture-based test, and update `docs/datasets.md`.
* Macro indicators must be lagged to the scoring date. Use `MacroFeatureJoiner` rather than joining
  raw contemporaneous values.

## 10. Testing

Fast suite:

```powershell
python -m pytest -q
```

LimiX integration suite (loads the checkpoint):

```powershell
$env:OPENCASHFLOW_RUN_LIMIX="1"
python -m pytest tests/test_limix.py -q -m limix
```

Lint:

```powershell
python -m ruff check opencashflow tests benchmarks
```

Benchmark:

```powershell
python benchmarks/run_benchmark.py --datasets synthetic,german,bangladesh_sme,polish `
  --models heuristic,logistic,lightgbm,xgboost,limix --cv 3 --limix-pipelines 1
```

Rules:

* Every new behaviour needs a test. Prefer fast fixtures over downloading data.
* Do not assert specific AUC values on 20-30 row fixtures. Test mechanics; use the benchmark for
  quality.
* LimiX tests are opt-in so CI and contributors without the non-commercial weights can still work.
* Notebook execution is not part of the fast suite; run it before release.

## 11. Code style

* Python 3.10+, type hints on public functions, concise docstrings with inputs/outputs.
* Use `safe_div` and the ratio helpers instead of raw division in feature code.
* Never return silent `inf`/`nan` from features. Fill and clip.
* Keep the package importable without LightGBM, XGBoost, SHAP or LimiX installed. Import optional
  dependencies inside the function that needs them.
* Keep `data/__init__.py` lazy for synthetic exports to avoid the features/data import cycle.
* Do not use em dashes in user-facing prose. Use commas, colons or parentheses.
* Prefer explicit column names and stable feature names over clever abstractions.
* Match the existing style; do not reformat unrelated files.

## 12. Release checklist

1. `pytest -q` passes.
2. LimiX tests pass on a machine with the checkpoint.
3. `ruff check` passes.
4. Benchmarks are regenerated and `benchmarks/results/benchmark_summary.md` is committed.
5. `docs/features.md` is regenerated.
6. README numbers match the saved benchmark output.
7. CHANGELOG updated.
8. Version bumped in `pyproject.toml` and `opencashflow/__init__.py`.
