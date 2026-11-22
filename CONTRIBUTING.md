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

Run the fast suite and lint. The fast suite has 347 tests and should pass locally. The lint command
uses the CI scope, `opencashflow`, `tests` and `benchmarks`:

```bash
python -m pytest -q
python -m ruff check opencashflow tests benchmarks
```

There are 5 LimiX integration tests. They are skipped in the fast suite unless
`OPENCASHFLOW_RUN_LIMIX=1` is set and the checkpoint and source are present. LimiX tests are opt-in
because they need the non-commercial checkpoint:

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

## Reproducing the published benchmarks

These are the published configurations. All of them pass `--output-dir` so the results are written
to files. Without `--output-dir` the script prints results and writes nothing.

Main results:

```bash
python benchmarks/run_benchmark.py --datasets synthetic,german,bangladesh_sme,polish --models heuristic,logistic,lightgbm,xgboost,limix --cv 3 --repeats 2 --synthetic-companies 300 --synthetic-months 20 --limix-pipelines 1 --output-dir benchmarks/results
```

SBA with all 21 features:

```bash
python benchmarks/run_benchmark.py --datasets sba --models logistic,lightgbm,xgboost,limix --cv 3 --repeats 2 --max-rows 5000 --limix-pipelines 1 --output-dir benchmarks/results/sba
```

SBA without `term_months` and `term_years`:

```bash
python benchmarks/run_benchmark.py --datasets sba --models logistic,lightgbm,xgboost,limix --cv 3 --repeats 2 --max-rows 5000 --limix-pipelines 1 --exclude-features term_months,term_years --variant no-term --output-dir benchmarks/results/sba_no_term
```

SBA out of time:

```bash
python benchmarks/run_benchmark.py --datasets sba --models logistic,lightgbm,xgboost --split oot --time-column approval_fy --oot-cutoff 2003 --max-rows 20000 --output-dir benchmarks/results/sba_oot
```

The remaining SBA variants add flags to the out-of-time command: `sba_oot_no_term` adds
`--exclude-features term_months,term_years --variant no-term --output-dir benchmarks/results/sba_oot_no_term`;
`sba_oot_matured` adds `--sba-approval-fy-max 2008 --variant matured-2008 --output-dir benchmarks/results/sba_oot_matured`;
`sba_oot_matured_no_term` adds both sets with `--output-dir benchmarks/results/sba_oot_matured_no_term`.

## Rebuilding the notebooks

Notebooks are generated from a script, then executed in place. Notebook execution is not part of the
fast suite; run it before release.

The notebooks need `ipykernel` in the same environment. Install it once, then register the kernel the notebooks use:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name opencashflow --display-name "OpenCashFlow Python"
```

Then rebuild and execute the notebooks in order. The timeout flag lets the LimiX cells run longer than nbconvert's default per-cell limit:

```bash
python scripts/build_notebooks.py
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/01_feature_engineering.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/02_limix_scoring.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/03_benchmarks.ipynb
```

## Release checklist

1. `python -m pytest -q` passes (347 tests).
2. LimiX tests pass on a machine with the checkpoint (`OPENCASHFLOW_RUN_LIMIX=1`).
3. `python -m ruff check opencashflow tests benchmarks` passes.
4. Regenerate the benchmarks and commit the results under `benchmarks/results/`, including
   `benchmarks/results/benchmark_summary.md`.
5. Regenerate `docs/features.md` with `python scripts/generate_feature_docs.py`.
6. README numbers match the saved benchmark output.
7. Update `CHANGELOG.md` with an `[Unreleased]` section for the changes.
8. Bump the version in `pyproject.toml` and `opencashflow/__init__.py`.

## Pull requests

* Keep changes focused and describe the verification you ran.
* Include tests for new behaviour.
* Update docs when the public API or feature catalogue changes.
* Do not reformat unrelated files.
* Use clear commit messages.

## Licence

By contributing, you agree that your contribution is licensed under Apache-2.0. You retain copyright
to your work. Do not contribute code or data that you do not have the right to license.
