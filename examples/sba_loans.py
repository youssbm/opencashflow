"""SBA National loan-data example.

Loads a stratified 5,000-row sample of the 899,164-row U.S. SBA dataset, fits a tree and a linear
baseline, and reports a borrower-level holdout AUC.

Usage::

    python examples/sba_loans.py

The data is CC BY-SA 4.0. See docs/datasets.md before redistributing derived datasets.
"""

from __future__ import annotations

from sklearn.model_selection import train_test_split

from opencashflow.data.loaders import load_sba_loans
from opencashflow.models.baselines import LogisticBaseline, TreeBaseline
from opencashflow.models.metrics import evaluate_binary


def main() -> None:
    bundle = load_sba_loans(max_rows=5_000, random_state=42)
    print(bundle.summary())
    print("\nColumns:", list(bundle.X.columns))
    print("\nCategorical:", bundle.categorical_features)
    print("\nDropped leakage columns:", bundle.metadata["dropped_leakage_columns"])
    print("\nSample:")
    print(bundle.X.head(3).to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=0.30,
        random_state=42,
        stratify=bundle.y,
    )
    for name, model in [
        ("logistic", LogisticBaseline()),
        ("lightgbm", TreeBaseline("lightgbm", n_estimators=200)),
    ]:
        model.fit(X_train, y_train)
        metrics = evaluate_binary(y_test, model.predict_pd(X_test))
        print(
            f"\n{name}: ROC-AUC={metrics['roc_auc']:.3f} "
            f"KS={metrics['ks']:.3f} Brier={metrics['brier']:.3f}"
        )

    print("\nLicence:", bundle.license)
    print("Share-alike:", bundle.metadata["share_alike"])


if __name__ == "__main__":
    main()
