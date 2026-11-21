"""OpenCashFlow: cash-flow underwriting for SME credit risk.

The package is organised around four layers:

* ``opencashflow.data`` - dataset loaders, schema normalisation and synthetic ledgers.
* ``opencashflow.features`` - transaction categorisation and 151 cash-flow features.
* ``opencashflow.models`` - LimiX-2, tree baselines, metrics and explanations.
* ``opencashflow.pipeline`` - end-to-end underwriting from raw transactions to a decision.

The top-level namespace exposes the most common entry points. Heavier optional dependencies
(LimiX, XGBoost, SHAP) are imported lazily so that the core install stays light.
"""

from __future__ import annotations

__version__ = "0.1.0"

from opencashflow.data import (
    DatasetBundle,
    download_dataset,
    generate_company_transactions,
    generate_credit_dataset,
    generate_portfolio,
    list_datasets,
    load_bangladesh_sme,
    load_dataset,
    load_german_credit,
    load_polish_bankruptcy,
    load_sba_loans,
    load_synthetic,
    load_transactions_csv,
    normalize_transactions,
)
from opencashflow.features import (
    CashFlowFeatureExtractor,
    CategoryMatch,
    CategoryRule,
    FeatureConfig,
    TransactionCategorizer,
    compute_features,
)
from opencashflow.models import (
    Explanation,
    HeuristicScorer,
    LimiXScorer,
    LogisticBaseline,
    TreeBaseline,
    evaluate_binary,
    explain_model,
    make_baseline,
)
from opencashflow.pipeline import CashFlowPipeline, UnderwritingPolicy

__all__ = [
    "CashFlowFeatureExtractor",
    "CashFlowPipeline",
    "CategoryMatch",
    "CategoryRule",
    "DatasetBundle",
    "Explanation",
    "FeatureConfig",
    "HeuristicScorer",
    "LimiXScorer",
    "LogisticBaseline",
    "TransactionCategorizer",
    "TreeBaseline",
    "UnderwritingPolicy",
    "__version__",
    "compute_features",
    "download_dataset",
    "evaluate_binary",
    "explain_model",
    "generate_company_transactions",
    "generate_credit_dataset",
    "generate_portfolio",
    "list_datasets",
    "load_bangladesh_sme",
    "load_dataset",
    "load_german_credit",
    "load_polish_bankruptcy",
    "load_sba_loans",
    "load_synthetic",
    "load_transactions_csv",
    "make_baseline",
    "normalize_transactions",
]
