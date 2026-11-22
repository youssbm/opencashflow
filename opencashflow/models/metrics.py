"""Credit-risk metrics and rating helpers.

The functions are deliberately explicit about edge cases: a single-class validation set has an
undefined AUC, and returning ``nan`` is safer than silently reporting 0.5. Use
:func:`evaluate_binary` for a complete metric dictionary.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

RATING_BOUNDS: tuple[tuple[float, str], ...] = (
    (0.02, "A"),
    (0.05, "B"),
    (0.10, "C"),
    (0.20, "D"),
    (0.35, "E"),
    (0.50, "F"),
)


def _finite_or_nan(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else float("nan")


def ks_statistic(y_true: Any, y_score: Any) -> float:
    """Kolmogorov-Smirnov separation between score distributions of good and bad borrowers.

    Returns ``nan`` when only one class is present.
    """
    y = np.asarray(y_true).astype(int).reshape(-1)
    s = np.asarray(y_score, dtype=float).reshape(-1)
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(np.abs(tpr - fpr)))


def gini_from_auc(auc: float) -> float:
    """Gini coefficient corresponding to an AUC (``2 * AUC - 1``)."""
    return float(2.0 * auc - 1.0) if math.isfinite(auc) else float("nan")


def evaluate_binary(
    y_true: Any,
    y_score: Any,
    *,
    threshold: float = 0.5,
    sample_weight: Any = None,
) -> dict[str, float]:
    """Compute a standard credit-risk metric panel.

    Args:
        y_true: Binary labels; the positive class is ``1`` (default).
        y_score: Predicted probability of the positive class.
        threshold: Decision threshold used for accuracy, precision, recall and F1.
        sample_weight: Optional observation weights.

    Returns:
        Dictionary with ROC-AUC, PR-AUC, KS, Gini, Brier, log loss and threshold metrics.
        Undefined values are ``nan`` rather than misleading defaults.
    """
    y = np.asarray(y_true).astype(int).reshape(-1)
    s = np.asarray(y_score, dtype=float).reshape(-1)
    if y.shape != s.shape:
        raise ValueError(
            f"y_true and y_score must have the same length, got {y.shape} and {s.shape}."
        )
    if y.size == 0:
        raise ValueError("Cannot evaluate an empty array.")
    if not np.isfinite(s).all():
        raise ValueError("y_score contains non-finite values.")
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    present = np.unique(y)
    binary = present.size == 2
    pred = (s >= threshold).astype(int)

    auc = float(roc_auc_score(y, s, sample_weight=weights)) if binary else float("nan")
    pr_auc = float(average_precision_score(y, s, sample_weight=weights)) if binary else float("nan")
    brier = float(brier_score_loss(y, s, sample_weight=weights)) if binary else float("nan")
    logloss = (
        float(log_loss(y, np.clip(s, 1e-9, 1 - 1e-9), sample_weight=weights))
        if binary
        else float("nan")
    )
    cm = confusion_matrix(y, pred, labels=[0, 1], sample_weight=weights)
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    result = {
        "n": float(y.size),
        "n_positive": float(int(y.sum())),
        "default_rate": float(np.average(y, weights=weights))
        if weights is not None
        else float(y.mean()),
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "ks": ks_statistic(y, s),
        "gini": gini_from_auc(auc),
        "brier": brier,
        "log_loss": logloss,
        "accuracy": float(accuracy_score(y, pred, sample_weight=weights)),
        "precision": float(precision_score(y, pred, sample_weight=weights, zero_division=0)),
        "recall": float(recall_score(y, pred, sample_weight=weights, zero_division=0)),
        "f1": float(f1_score(y, pred, sample_weight=weights, zero_division=0)),
        "threshold": float(threshold),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
    return {
        key: (value if isinstance(value, str) else _finite_or_nan(value))
        for key, value in result.items()
    }


def calibration_table(
    y_true: Any,
    y_score: Any,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Reliability table with equal-width probability bins.

    Returns columns ``bin_lower``, ``bin_upper``, ``count``, ``mean_predicted``, ``observed`` and
    ``gap``. Empty bins are omitted.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    frame = pd.DataFrame(
        {
            "y": np.asarray(y_true, dtype=int).reshape(-1),
            "p": np.asarray(y_score, dtype=float).reshape(-1),
        }
    )
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    frame["bin"] = pd.cut(frame["p"], bins=edges, include_lowest=True, right=True)
    table = (
        frame.groupby("bin", observed=True)
        .agg(count=("y", "size"), mean_predicted=("p", "mean"), observed=("y", "mean"))
        .reset_index()
    )
    table["bin_lower"] = [interval.left for interval in table["bin"]]
    table["bin_upper"] = [interval.right for interval in table["bin"]]
    table["gap"] = table["observed"] - table["mean_predicted"]
    return table[["bin_lower", "bin_upper", "count", "mean_predicted", "observed", "gap"]]


def rating_from_pd(probability: float | np.ndarray) -> str | np.ndarray:
    """Map a probability of default to a letter rating (A best, G worst).

    The cut-offs are illustrative and should be replaced with a calibrated master scale before
    production use.
    """
    values = np.asarray(probability, dtype=float)
    flat = values.reshape(-1)
    labels: list[str] = []
    for value in flat:
        label = "G"
        for bound, grade in RATING_BOUNDS:
            if value <= bound:
                label = grade
                break
        labels.append(label)
    result = np.asarray(labels, dtype=object).reshape(values.shape)
    return str(result) if result.ndim == 0 else result


def expected_loss(
    probability: float | np.ndarray,
    *,
    lgd: float = 0.45,
    ead: float = 1.0,
) -> float | np.ndarray:
    """Expected loss = PD x LGD x EAD.

    Args:
        probability: Probability of default in ``[0, 1]``.
        lgd: Loss given default, as a fraction (default 45%, a common unsecured SME assumption).
        ead: Exposure at default in currency units, or a normalised 1.0.
    """
    pd_values = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    if not 0.0 <= lgd <= 1.0:
        raise ValueError("lgd must be between 0 and 1.")
    result = pd_values * lgd * ead
    return float(result) if result.ndim == 0 else result


def cost_sensitive_threshold(
    y_true: Any,
    y_score: Any,
    *,
    cost_false_positive: float = 1.0,
    cost_false_negative: float = 5.0,
    grid: int = 200,
) -> tuple[float, float]:
    """Threshold that minimises the observed misclassification cost on the supplied scores.

    The default 1:5 ratio mirrors the classic German Credit cost matrix: ``cost_false_positive=1``
    for rejecting a good borrower and ``cost_false_negative=5`` for approving a bad one.

    Returns:
        ``(threshold, mean_cost_per_observation)``, with the cost averaged over ``y_true``.
    """
    if cost_false_positive < 0 or cost_false_negative < 0:
        raise ValueError("Costs must be non-negative.")
    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(y_score, dtype=float).reshape(-1)
    thresholds = np.linspace(0.001, 0.999, grid)
    costs = []
    for threshold in thresholds:
        pred = (s >= threshold).astype(int)
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        costs.append((fp * cost_false_positive + fn * cost_false_negative) / max(len(y), 1))
    best = int(np.argmin(costs))
    return float(thresholds[best]), float(costs[best])


def lift_at_k(y_true: Any, y_score: Any, k: float = 0.1) -> float:
    """Lift in the riskiest ``k`` fraction of the portfolio versus the overall default rate."""
    if not 0.0 < k <= 1.0:
        raise ValueError("k must be in (0, 1].")
    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(y_score, dtype=float).reshape(-1)
    if y.size == 0:
        return float("nan")
    n_top = max(1, math.ceil(k * y.size))
    order = np.argsort(-s)
    top_rate = float(np.mean(y[order[:n_top]]))
    overall = float(np.mean(y))
    return float(top_rate / overall) if overall > 0 else float("nan")


def approval_rate_at_pd(pd_values: Any, max_pd: float) -> float:
    """Share of applicants with predicted PD at or below a policy cut-off."""
    values = np.asarray(pd_values, dtype=float).reshape(-1)
    return float(np.mean(values <= max_pd)) if values.size else float("nan")


__all__ = [
    "approval_rate_at_pd",
    "calibration_table",
    "cost_sensitive_threshold",
    "evaluate_binary",
    "expected_loss",
    "gini_from_auc",
    "ks_statistic",
    "lift_at_k",
    "rating_from_pd",
]
