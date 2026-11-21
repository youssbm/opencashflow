# OpenCashFlow

**Cash-flow underwriting for SME credit risk, built in the open.**

OpenCashFlow turns raw open-banking-style transactions into credit-relevant features, a probability
of default, an illustrative policy decision and human-readable reason codes. It uses LimiX-2 as the
primary tabular foundation model, with LightGBM, XGBoost, logistic regression and an untrained
cash-flow scorecard as baselines.

<div style="border:1px solid rgba(127,127,127,.28);border-radius:12px;padding:16px;background:rgba(127,127,127,.06)">
<strong>Why it exists.</strong> SME credit tooling is dominated by closed products. Public SME data is
mostly accounting ratios, not bank ledgers. OpenCashFlow provides a transparent feature engine, a
documented synthetic ledger generator and honest benchmarks so researchers and developers can build
and compare cash-flow underwriting models without a data-sharing agreement.
</div>

## Highlights

* **151 cash-flow features** across revenue, expenses, liquidity, debt, distress, behaviour, trends
  and composite health.
* **Multilingual transaction categoriser** for English, German, French, Polish, Italian, Dutch and
  Spanish narratives, with an optional learned fallback.
* **LimiX-2 wrapper** for classification, regression and imputation, including CPU/Windows support
  and the non-commercial licence notice.
* **Tree and linear baselines** with SHAP explanations, calibration metrics and cost-sensitive
  thresholds.
* **End-to-end pipeline**: transactions -> features -> PD -> rating -> decision -> reasons.
* **Three public datasets** plus a synthetic portfolio generator, with licences and citations.
* **Tests, notebooks and a reproducible benchmark suite**.

## Quick example

```python
from opencashflow.data.synthetic import generate_portfolio
from opencashflow.pipeline import CashFlowPipeline

portfolio = generate_portfolio(n_companies=120, months=20, seed=42)
pipeline = CashFlowPipeline(scorer="lightgbm").fit(
    portfolio.transactions, portfolio.labels
)
scored = pipeline.predict(portfolio.transactions, top_k=3)
print(scored[["pd", "rating", "decision", "reason_1"]].head())
```

## Documentation

* [Quickstart](examples/quickstart.md)
* [Feature dictionary](features.md)
* [Datasets and licences](datasets.md)
* [LimiX-2 integration](limix.md)
* [Architecture](architecture.md)
* [Working with real transactions](examples/transactions.md)

## Licence

Code: Apache-2.0. Public datasets keep their own licences (CC BY 4.0). **LimiX-2 weights are
non-commercial** under the StableAI LimiX Non-Commercial License v1.0 and require a separate
agreement for commercial use.
