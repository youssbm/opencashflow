"""End-to-end quickstart.

Generates a synthetic SME portfolio, fits an untrained scorecard and a LightGBM baseline, and prints
portfolio-level results. Runs in well under a minute on a laptop.

Usage::

    python examples/quickstart.py
"""

from __future__ import annotations

from opencashflow.data.synthetic import generate_portfolio
from opencashflow.features.cashflow_features import FeatureConfig
from opencashflow.pipeline import CashFlowPipeline


def main() -> None:
    portfolio = generate_portfolio(
        n_companies=100,
        months=18,
        default_rate=0.15,
        seed=42,
    )
    print(portfolio.summary())
    as_of = portfolio.metadata.get("observation_end") or portfolio.transactions["date"].max()

    for scorer in ("heuristic", "lightgbm"):
        pipeline = CashFlowPipeline(scorer=scorer, config=FeatureConfig(as_of=as_of))
        pipeline.fit(portfolio.transactions, portfolio.labels)
        scored = pipeline.predict(portfolio.transactions, as_of=as_of, top_k=3, ead=25_000)
        summary = pipeline.portfolio_summary(scored)
        print(f"\n{scorer}: {summary}")
        columns = ["pd", "rating", "decision", "reason_1"]
        print(scored[columns].head(5).to_string())

    explanations = pipeline.explain(portfolio.transactions, as_of=as_of, max_rows=2)
    for explanation in explanations:
        print(f"\nEntity {explanation.entity_id}: PD={explanation.prediction:.3f}")
        for reason in explanation.reason_codes(3):
            print("  -", reason)


if __name__ == "__main__":
    main()
