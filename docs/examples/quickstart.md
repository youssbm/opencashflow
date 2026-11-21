# Quickstart

This walkthrough runs in a few minutes on a laptop and does not need any downloaded data.

## 1. Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[trees,dev]"
```

## 2. Generate a portfolio

```python
from opencashflow.data.synthetic import generate_portfolio

portfolio = generate_portfolio(
    n_companies=120,
    months=20,
    default_rate=0.15,
    seed=42,
)
print(portfolio.summary())
```

`portfolio.transactions` is a transaction ledger. `portfolio.labels` contains the binary
`default_flag`, the latent `pd_true`, and the post-ledger outcomes used to generate the label.

## 3. Score with the no-training scorecard

```python
from opencashflow.pipeline import CashFlowPipeline

pipeline = CashFlowPipeline(scorer="heuristic")
pipeline.fit(portfolio.transactions, portfolio.labels)
scored = pipeline.predict(portfolio.transactions, top_k=3)

print(scored[["pd", "rating", "decision", "reason_1"]].head(10))
print(pipeline.portfolio_summary(scored))
```

The heuristic scorer is a fixed, explainable cash-flow scorecard. It needs no labels, which makes it
useful for a first look and for demos.

## 4. Fit a tree baseline

```python
pipeline = CashFlowPipeline(scorer="lightgbm")
pipeline.fit(portfolio.transactions, portfolio.labels)
scored = pipeline.predict(portfolio.transactions, top_k=3, ead=25_000)
```

`ead` adds an expected-loss column using PD x LGD (45% by default) x EAD. The result is sorted by
descending PD.

> **Evaluate on a borrower-level holdout.** Running `predict` on the same transactions used in `fit`
> is in-sample. Split by `entity_id`, keep test borrowers out of the training transactions, and only
> then compute AUC or Brier score. This matters even more for LimiX, which can memorise its
> in-context reference set.

## 5. Inspect the drivers

```python
explanations = pipeline.explain(portfolio.transactions, max_rows=3)
for explanation in explanations:
    print(explanation.entity_id, round(explanation.prediction, 3))
    for reason in explanation.reason_codes(3):
        print("  -", reason)
```

Tree models use exact TreeSHAP. LimiX and logistic models use occlusion, which replaces one feature
at a time with a reference value and measures the PD change. Reason codes are model-relative, not
causal.

## 6. Use LimiX-2

```bash
export LIMIX_MODEL_PATH=/path/to/LimiX-2-1016.ckpt
export LIMIX_SRC=/path/to/LimiX
```

```python
pipeline = CashFlowPipeline(scorer="limix")
pipeline.fit(portfolio.transactions, portfolio.labels)
scored = pipeline.predict(portfolio.transactions, top_k=3)
```

On CPU, use `LimiXScorer(n_pipelines=1)` for speed. See [LimiX integration](../limix.md) for the
licence, pinned versions and troubleshooting.

## 7. Save and reload

```python
path = pipeline.save("artifacts/pipeline.joblib")
restored = CashFlowPipeline.load(path)
restored.predict(portfolio.transactions.head(200), explain=False)
```

LimiX weights are not embedded in the pickle. The checkpoint path is stored and the predictor
reloads lazily.

## 8. Benchmark

```bash
python benchmarks/run_benchmark.py \
  --datasets synthetic,german,bangladesh_sme,polish \
  --models heuristic,logistic,lightgbm,xgboost,limix \
  --cv 3 --synthetic-companies 300 --synthetic-months 20 --limix-pipelines 1
```

Results are written to `benchmarks/results/`. Read the dataset notes before interpreting them:
[Datasets](../datasets.md).

## Next steps

* [Feature dictionary](../features.md) for the 151 features.
* [Transaction ingestion](transactions.md) for CSV mapping, custom rules and point-in-time scoring.
* [Architecture](../architecture.md) for the layer contracts and design decisions.
