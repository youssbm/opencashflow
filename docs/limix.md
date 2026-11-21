# LimiX-2 integration

LimiX-2 is a 400M-parameter tabular foundation model from Stable AI. One checkpoint handles
classification, regression and missing-value imputation through in-context learning: you provide a
labelled reference set, and the model predicts the query set in a single forward pass. There is no
gradient training in this repository.

## Licence: read this first

The LimiX-2 **weights** are released under the **StableAI LimiX Non-Commercial License v1.0**.
Commercial use requires a separate agreement with Stable AI. If you publish results, disclose the
model name "LimiX" as required by the licence.

* Code in this repository: Apache-2.0.
* LimiX weights: non-commercial.
* Pretrained model card: https://huggingface.co/stable-ai/LimiX-2-1016

## Pinned versions

| Component | Version / commit |
|---|---|
| Weights | `stable-ai/LimiX-2-1016` @ `bbbf26c` |
| Code | https://github.com/limix-ldm-ai/LimiX @ `a917fa0` |
| Configs | `*_v2.json` |
| Contamination note | https://github.com/autogluon/tabarena/issues/648 |

Do not use the older `stable-ai/LimiX-2` checkpoint for leaderboard claims. It was withdrawn after
contamination was found.

## Installation

```bash
git clone https://github.com/limix-ldm-ai/LimiX
cd LimiX
git checkout a917fa0
pip install -e .
```

Download `LimiX-2-1016.ckpt` from Hugging Face and point the wrapper at it:

```bash
export LIMIX_MODEL_PATH=/path/to/LimiX-2-1016.ckpt
export LIMIX_CONFIG_DIR=/path/to/LimiX/config   # optional if auto-discovered
export LIMIX_SRC=/path/to/LimiX                 # optional if `limix` is installed
```

On Windows PowerShell:

```powershell
$env:LIMIX_MODEL_PATH="C:\models\LimiX-2-1016.ckpt"
$env:LIMIX_SRC="C:\code\LimiX"
```

## CPU and Windows notes

LimiX imports `triton` even when its default path is pure PyTorch. On a CPU-only machine the wrapper
installs a no-op `triton` shim before importing `inference.predictor`. It also shims `nvtx` when the
profiling package is absent. No GPU is required.

The released config uses **32 ensemble pipelines**. On CPU that is slow. Use:

* `n_pipelines=1` for tests and quick experiments;
* `n_pipelines=4` for a reasonable quality/speed trade-off;
* `n_pipelines=32` on a GPU when reproducing leaderboard-quality results.

The first prediction loads the 1.6 GB checkpoint (about 5 seconds on the development machine).
Subsequent predictions reuse it. Call `close()` to release memory.

## Classification

```python
from opencashflow.models.limix_scorer import LimiXScorer

scorer = LimiXScorer(n_pipelines=1, seed=0)
scorer.fit(X_train, y_train)          # in-context reference set
pd_values = scorer.predict_pd(X_test)
proba = scorer.predict_proba(X_test)  # (n, n_classes)
scorer.close()
```

`fit` only stores and validates the reference set. It does not train. The positive class is the
second class in sorted order; encode default as `1`. `predict_pd` returns the column for class `1`
when present.

## Regression

```python
scorer = LimiXScorer(task="regression", n_pipelines=4)
scorer.fit(X_train, y_train)
predictions = scorer.predict_score(X_test)  # original target scale
```

For leaderboard-equivalent regression, set `softmax_temperature=1.0`.

## Imputation

```python
imputed = scorer.impute(X_with_nan)
```

Imputation uses `reg_default_noretrieval_MVI_v2.json`, not the classification config. LimiX returns
the reference rows followed by the query rows; the wrapper slices the query block and returns the
same shape as the input. If the scorer is not fitted, the input itself is used as the reference set.

## Categorical features

Categorical columns are ordinal-encoded using categories learned during `fit`; unseen categories map
to `-1`. This is intentional. High-cardinality one-hot expansion would make the in-context set wide
and slow. For high-cardinality categories, use target encoding or frequency encoding before LimiX,
and fit that encoding inside the training fold.

## What the wrapper checks

`LimiXScorer.availability()` returns a dictionary without loading weights:

```python
{
    "available": True,
    "model_path": "...",
    "config_dir": "...",
    "limix_src": "...",
    "limix_installed": True,
    "reasons": [],
}
```

If `available` is false, the `reasons` list names the missing piece. The wrapper searches
`LIMIX_MODEL_PATH`, `OPENCASHFLOW_LIMIX_CKPT`, `cache_limix/`, `models/` and the Hugging Face cache.
Config discovery uses `LIMIX_CONFIG_DIR`, `LIMIX_SRC/config`, `limix_src/config`, and the installed
`limix` package.

## Persistence

`LimiXScorer` drops the in-memory predictor from pickles. A saved `CashFlowPipeline` stores the
checkpoint path and configuration, and reloads the weights lazily on the next prediction. The
checkpoint itself is never embedded in the pickle.

## Benchmarks

The repository's benchmark uses one pipeline on CPU and stratified 3-fold cross-validation. See
[`benchmarks/results/benchmark_summary.md`](../benchmarks/results/benchmark_summary.md) for the
latest saved numbers. Run:

```bash
python benchmarks/run_benchmark.py \
  --datasets synthetic,german,bangladesh_sme,polish \
  --models heuristic,logistic,lightgbm,xgboost,limix \
  --cv 3 --limix-pipelines 1
```

Do not compare a 1-pipeline CPU run with a 32-pipeline GPU run without saying so. Report the number
of ensemble members, the device and the fold count.

## Troubleshooting

**`Could not import LimiX`**
Set `LIMIX_SRC` to the folder containing `inference/`, or install the repository with `pip install -e .`.

**`LimiX checkpoint not found`**
Set `LIMIX_MODEL_PATH` to the `.ckpt` file. The wrapper does not download weights automatically
because of the licence and size.

**`bad escape (end of pattern)` or triton import errors**
Use the wrapper, which installs the CPU shim. If you import `inference.predictor` directly on
Windows CPU, install the shim first or set `sys.modules["triton"]` as shown in
`opencashflow/models/limix_scorer.py`.

**`Found array with 0 feature(s)`**
This occurred when imputing with an unfitted scorer. The wrapper now initialises feature names from
the query matrix. Upgrade to the current version.

**Very slow CPU runs**
Reduce `n_pipelines`. With one member, a small dataset takes about a second after the checkpoint is
loaded. With 32 members, expect roughly 32 times the compute.

## Citation

```bibtex
@misc{limix2_1016,
  title  = {LimiX-2-1016},
  author = {Stable AI},
  year   = {2026},
  note   = {Tabular foundation model, non-commercial weights},
  url    = {https://huggingface.co/stable-ai/LimiX-2-1016}
}
```

Also cite the LimiX code repository and disclose the checkpoint commit used.
