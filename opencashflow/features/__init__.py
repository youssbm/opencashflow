"""Feature engineering: categorisation, ratios and cash-flow features."""

from opencashflow.features.cashflow_features import (
    CashFlowFeatureExtractor,
    FeatureConfig,
    compute_features,
)
from opencashflow.features.categorizer import (
    CATEGORY_DESCRIPTIONS,
    CategoryMatch,
    CategoryRule,
    TransactionCategorizer,
)
from opencashflow.features.ratios import (
    coefficient_of_variation,
    gini,
    hhi,
    max_drawdown,
    normalized_entropy,
    safe_div,
    seasonal_strength,
    trend_ratio,
)
from opencashflow.features.specs import (
    FEATURE_NAMES,
    FEATURE_SPECS,
    FeatureSpec,
    feature_specs_frame,
)

__all__ = [
    "CATEGORY_DESCRIPTIONS",
    "FEATURE_NAMES",
    "FEATURE_SPECS",
    "CashFlowFeatureExtractor",
    "CategoryMatch",
    "CategoryRule",
    "FeatureConfig",
    "FeatureSpec",
    "TransactionCategorizer",
    "coefficient_of_variation",
    "compute_features",
    "feature_specs_frame",
    "gini",
    "hhi",
    "max_drawdown",
    "normalized_entropy",
    "safe_div",
    "seasonal_strength",
    "trend_ratio",
]
