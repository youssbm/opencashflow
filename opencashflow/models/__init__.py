"""Scoring engines, baselines, metrics and explanations."""

from opencashflow.models.baselines import (
    HEURISTIC_WEIGHTS,
    HeuristicScorer,
    LogisticBaseline,
    TreeBaseline,
    available_backends,
    make_baseline,
)
from opencashflow.models.explain import (
    AdditiveExplainer,
    Explanation,
    OcclusionExplainer,
    ShapExplainer,
    explain_model,
    explanations_to_frame,
    permutation_importance,
)
from opencashflow.models.limix_scorer import LICENSE_NOTICE, LimiXScorer, LimiXUnavailableError
from opencashflow.models.metrics import (
    approval_rate_at_pd,
    calibration_table,
    cost_sensitive_threshold,
    evaluate_binary,
    expected_loss,
    gini_from_auc,
    ks_statistic,
    lift_at_k,
    rating_from_pd,
)

__all__ = [
    "HEURISTIC_WEIGHTS",
    "LICENSE_NOTICE",
    "AdditiveExplainer",
    "Explanation",
    "HeuristicScorer",
    "LimiXScorer",
    "LimiXUnavailableError",
    "LogisticBaseline",
    "OcclusionExplainer",
    "ShapExplainer",
    "TreeBaseline",
    "approval_rate_at_pd",
    "available_backends",
    "calibration_table",
    "cost_sensitive_threshold",
    "evaluate_binary",
    "expected_loss",
    "explain_model",
    "explanations_to_frame",
    "gini_from_auc",
    "ks_statistic",
    "lift_at_k",
    "make_baseline",
    "permutation_importance",
    "rating_from_pd",
]
