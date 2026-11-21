# Contributing to OpenCashFlow

Thanks for considering a contribution. This project is intended to be useful to researchers and
practitioners, so clarity, reproducibility and honest evaluation matter more than clever code.

## Before you start

1. Read [`AGENTS.md`](AGENTS.md). It contains the engineering contract and the LimiX runbook.
2. Search existing issues and discussions to avoid duplicate work.
3. For substantial changes, open an issue with the problem, proposed approach and evaluation plan.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[trees,dev]"
```

Run the fast suite:

```bash
python -m pytest -q
python -m ruff check opencashflow tests benchmarks
```

LimiX tests are opt-in because they need the non-commercial checkpoint:

```bash
export OPENCASHFLOW_RUN_LIMIX=1
export LIMIX_MODEL_PATH=/path/to/LimiX-2-1016.ckpt
export LIMIX_SRC=/path/to/LimiX
python -m pytest tests/test_limix.py -q -m limix
```

## What to contribute

Good first contributions:

* new country-specific transaction rules with tests;
* additional feature specs with a clear definition, formula and test;
* dataset provenance improvements and loader tests;
* notebook clarity and reproducibility;
* documentation fixes, especially around limitations and licences.

Please avoid:

* adding raw datasets, checkpoints or secrets;
* adding dependencies without a strong need;
* changing feature names without a deprecation path;
* reporting a single AUC on a tiny sample as a quality claim;
* using the withdrawn `stable-ai/LimiX-2` checkpoint.

## Adding a feature

1. Add a `FeatureSpec` to `opencashflow/features/specs.py` with a group, description,
   `higher_is_better` and formula.
2. Add the computation in `opencashflow/features/cashflow_features.py` with
   `builder.add("feature_name", value)`.
3. Use `safe_div` and the ratio helpers instead of raw division.
4. Add a focused test in `tests/test_features.py`.
5. Regenerate the dictionary:

   ```bash
   python scripts/generate_feature_docs.py
   ```

6. Run `pytest tests/test_features.py`.

Keep the feature name stable. If a rename is unavoidable, document it in `CHANGELOG.md`.

## Adding a dataset

1. Implement a loader returning `DatasetBundle`.
2. Add a `DATASET_REGISTRY` entry with source, DOI/URL, licence, citation and expected shape.
3. Add a fixture-based test that does not require the network.
4. Add an opt-in real-data test guarded by `skipif`.
5. Update `docs/datasets.md` and, if relevant, the README table.
6. Do not commit the raw data.

## Adding a model

Implement `fit`, `predict_proba`, `predict` and `predict_pd`. The positive class is the second sorted
class; encode default as `1`. Add a test for probability shape and finite values. If the model has no
TreeSHAP support, `OcclusionExplainer` will work as long as `predict_pd` exists.

## Benchmark changes

The benchmark protocol is part of the project's credibility. If you change it:

* keep stratified folds and a fixed seed;
* fit preprocessing inside each fold;
* do not tune on the test fold;
* report the number of LimiX ensemble members and the device;
* regenerate `benchmarks/results/` and update the README numbers;
* explain any change in `CHANGELOG.md`.

## Pull requests

* Keep changes focused and describe the verification you ran.
* Include tests for new behaviour.
* Update docs when the public API or feature catalogue changes.
* Do not reformat unrelated files.
* Use clear commit messages.

## Licence

By contributing, you agree that your contribution is licensed under Apache-2.0. You retain copyright
to your work. Do not contribute code or data that you do not have the right to license.
