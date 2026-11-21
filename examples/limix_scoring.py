"""Optional LimiX-2 example.

Requires the LimiX code and checkpoint. If they are not available, the script explains what to set
and exits cleanly.

Usage::

    export LIMIX_MODEL_PATH=/path/to/LimiX-2-1016.ckpt
    export LIMIX_SRC=/path/to/LimiX
    python examples/limix_scoring.py
"""

from __future__ import annotations

import sys

from sklearn.model_selection import train_test_split

from opencashflow.data.synthetic import generate_portfolio
from opencashflow.models.limix_scorer import LimiXScorer
from opencashflow.models.metrics import evaluate_binary
from opencashflow.pipeline import CashFlowPipeline


def main() -> int:
    info = LimiXScorer.availability()
    if not info["available"]:
        print("LimiX is not available:")
        for reason in info["reasons"]:
            print("  -", reason)
        print("\nSee docs/limix.md for setup instructions.")
        return 0

    portfolio = generate_portfolio(n_companies=200, months=20, default_rate=0.15, seed=7)
    labels = portfolio.labels.set_index("entity_id")["default_flag"]
    train_ids, test_ids = train_test_split(
        labels.index, test_size=0.30, random_state=7, stratify=labels
    )
    train_ids = set(train_ids)
    test_ids = set(test_ids)

    train_transactions = portfolio.transactions[portfolio.transactions["entity_id"].isin(train_ids)]
    test_transactions = portfolio.transactions[portfolio.transactions["entity_id"].isin(test_ids)]
    train_labels = portfolio.labels[portfolio.labels["entity_id"].isin(train_ids)]

    pipeline = CashFlowPipeline(scorer="limix")
    pipeline.fit(train_transactions, train_labels)
    scored = pipeline.predict(test_transactions, top_k=3, explain=False)

    y = portfolio.labels.set_index("entity_id")["default_flag"].reindex(scored.index).astype(int)
    metrics = evaluate_binary(y, scored["pd"])
    print(portfolio.summary())
    print(f"Holdout: {len(train_ids)} train / {len(test_ids)} test entities")
    print(f"LimiX holdout ROC-AUC: {metrics['roc_auc']:.3f}, KS: {metrics['ks']:.3f}")
    print(scored[["pd", "rating", "decision"]].head(8).to_string())

    explanations = pipeline.explain(test_transactions, max_rows=1, max_features=20)
    if explanations:
        print("\nTop driver for the riskiest holdout application:")
        for reason in explanations[0].reason_codes(3):
            print("  -", reason)

    print("\nLicence:", LimiXScorer.license_notice())
    return 0


if __name__ == "__main__":
    sys.exit(main())
