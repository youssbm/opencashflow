"""Model explanations and adverse-action reason codes.

Two complementary explainers are provided:

* :class:`ShapExplainer` for tree ensembles, using SHAP's exact TreeSHAP values.
* :class:`OcclusionExplainer` for any model with ``predict_pd``, including LimiX-2. It replaces one
  feature at a time with a reference value and measures the change in predicted PD. This is slower
  than SHAP but model-agnostic and easy to audit.

Explanations are expressed as contributions to the model output. For tree models these are
log-odds; for occlusion they are probability changes. Always report the method alongside the
numbers. The reason-code helper turns contributions into short, human-readable statements, but it
is not a substitute for a regulated adverse-action process.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from opencashflow.features.specs import FEATURE_SPECS


@dataclass
class Explanation:
    """One borrower-level explanation.

    Attributes:
        entity_id: Borrower identifier, if known.
        base_value: Model output before feature contributions.
        prediction: Predicted PD for this borrower.
        contributions: Signed contribution per feature. Positive means the feature increased the
            model output (risk); negative means it reduced it.
        method: ``"tree_shap"``, ``"occlusion"`` or ``"additive"``.
        feature_values: Original feature values for the borrower.
        metadata: Optional extra information.
    """

    entity_id: str | None
    base_value: float
    prediction: float
    contributions: pd.Series
    method: str
    feature_values: pd.Series | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def top(self, k: int = 5, *, absolute: bool = True) -> pd.DataFrame:
        """Return the top ``k`` drivers as a DataFrame.

        Columns: ``feature``, ``value``, ``contribution``, ``direction``, ``group`` and
        ``description``. ``direction`` is "increases risk" or "reduces risk".
        """
        if k < 1:
            raise ValueError("k must be at least 1.")
        series = self.contributions.dropna()
        order = (
            series.abs().sort_values(ascending=False).index
            if absolute
            else series.sort_values(ascending=False).index
        )
        rows = []
        for feature in order[:k]:
            contribution = float(series[feature])
            spec = FEATURE_SPECS.get(str(feature))
            rows.append(
                {
                    "feature": str(feature),
                    "value": float(self.feature_values[feature])
                    if self.feature_values is not None and feature in self.feature_values
                    else np.nan,
                    "contribution": contribution,
                    "direction": "increases risk" if contribution > 0 else "reduces risk",
                    "group": spec.group if spec else "unknown",
                    "description": spec.description if spec else "",
                }
            )
        return pd.DataFrame(rows)

    def reason_codes(self, k: int = 3) -> list[str]:
        """Return adverse-action style statements for risk-increasing features only.

        Only contributions greater than zero are kept, ordered from largest to smallest. At most
        ``k`` statements are produced; an explanation with no risk-increasing feature returns an
        empty list. Use :meth:`top` with ``absolute=True`` for the full audit view. These are
        illustrative statements, not regulated adverse-action reason codes.
        """
        if k < 1:
            raise ValueError("k must be at least 1.")
        positive = self.contributions.dropna()
        positive = positive[positive > 0].sort_values(ascending=False)
        reasons: list[str] = []
        for feature in positive.index[:k]:
            spec = FEATURE_SPECS.get(str(feature))
            label = spec.description.rstrip(".") if spec else str(feature)
            reasons.append(f"{label} ({feature}) increased predicted risk.")
        return reasons

    def to_dict(self) -> dict[str, Any]:
        """Serialise the explanation to JSON-friendly primitives."""
        return {
            "entity_id": self.entity_id,
            "base_value": float(self.base_value),
            "prediction": float(self.prediction),
            "method": self.method,
            "contributions": {str(k): float(v) for k, v in self.contributions.items()},
            "feature_values": (
                {str(k): float(v) for k, v in self.feature_values.items()}
                if self.feature_values is not None
                else {}
            ),
            "reason_codes": self.reason_codes(),
        }


class ShapExplainer:
    """Exact TreeSHAP explanations for LightGBM and XGBoost baselines."""

    def __init__(self, model: Any, *, background: pd.DataFrame | None = None) -> None:
        try:
            import shap
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "SHAP is not installed. Run: pip install 'opencashflow[trees]'"
            ) from exc
        backend = getattr(model, "backend", None)
        if backend not in {"lightgbm", "xgboost"} or not hasattr(model, "model_"):
            raise TypeError("ShapExplainer supports TreeBaseline models with a fitted model_.")
        self.model = model
        self.explainer_ = shap.TreeExplainer(model.model_)
        self.background = background

    def explain(self, X: pd.DataFrame) -> list[Explanation]:
        if self.model.model_ is None:
            raise RuntimeError("The underlying model is not fitted.")
        Xp = self.model.prepare_features(X)
        raw = self.explainer_.shap_values(Xp)
        values, base = _extract_positive_shap(raw, self.explainer_.expected_value)
        predictions = self.model.predict_pd(X)
        feature_names = list(self.model.feature_names_in_)
        explanations: list[Explanation] = []
        for i, (_, row) in enumerate(X.iterrows()):
            contributions = pd.Series(values[i], index=feature_names, dtype=float)
            explanations.append(
                Explanation(
                    entity_id=str(row.name) if row.name is not None else None,
                    base_value=float(base),
                    prediction=float(predictions[i]),
                    contributions=contributions,
                    method="tree_shap",
                    feature_values=pd.Series(row.to_numpy(dtype=float), index=feature_names),
                )
            )
        return explanations


def _extract_positive_shap(raw: Any, expected: Any) -> tuple[np.ndarray, float]:
    """Normalise SHAP output across versions and model types to ``(n, features), base``."""
    values = raw
    base = expected
    if isinstance(values, list):
        # Older SHAP returns [class0, class1].
        values = values[1] if len(values) > 1 else values[0]
        if isinstance(base, (list, np.ndarray)):
            base = np.asarray(base).reshape(-1)
            base = float(base[1] if base.size > 1 else base[0])
    values = np.asarray(values)
    if values.ndim == 3:
        # (n, features, classes)
        values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
    if values.ndim != 2:
        raise RuntimeError(f"Unexpected SHAP value shape: {values.shape}")
    base_array = np.asarray(base, dtype=float).reshape(-1)
    base_value = float(base_array[1] if base_array.size > 1 else base_array[0])
    return values, base_value


class OcclusionExplainer:
    """Model-agnostic occlusion explainer for any scorer with ``predict_pd``.

    For each feature, the value is replaced by a reference and the change in PD is recorded.
    Positive contributions mean the observed value increased risk relative to the reference. This
    is useful for LimiX and logistic models where TreeSHAP is unavailable.

    The default reference is the median of ``X``. When ``X`` holds a single row, that reference is
    the row itself and the contributions are zero; a ``UserWarning`` is emitted. Pass ``reference``
    (or ``background`` to :func:`explain_model`) to compare against a representative sample.

    With ``max_features`` set, features are chosen per row by standardised deviation from the
    reference, ``abs(x - ref) / scale``, where ``scale`` is the reference standard deviation; 1.0
    where that is zero, NaN or not finite. Unselected features get ``NaN`` contributions. A feature
    whose reference value is NaN is also not explained: it gets a ``NaN`` contribution in both the
    implicit and explicit reference paths, and reason codes ignore it.

    Args:
        model: Any object with ``predict_pd``.
        reference: Optional reference row or frame. Defaults to the median of ``X``.
        max_features: Maximum number of features explained per row. Selected by standardised
            deviation when the model has more. Set ``None`` to explain all features.
        random_state: Reserved for future sampling strategies.
    """

    def __init__(
        self,
        model: Any,
        *,
        reference: pd.DataFrame | pd.Series | None = None,
        max_features: int | None = 30,
        random_state: int = 42,
    ) -> None:
        if not hasattr(model, "predict_pd"):
            raise TypeError("OcclusionExplainer requires a model with predict_pd().")
        self.model = model
        self.reference = reference
        self.max_features = max_features
        self.random_state = random_state

    def _reference_row(self, X: pd.DataFrame) -> pd.Series:
        if isinstance(self.reference, pd.DataFrame):
            return self.reference.median(numeric_only=True).reindex(X.columns)
        if isinstance(self.reference, pd.Series):
            return self.reference.reindex(X.columns)
        if len(X) == 1:
            warnings.warn(
                "OcclusionExplainer has no explicit reference and X has one row, so the reference "
                "is that row and the contributions will be zero. Pass reference, or background to "
                "explain_model, to compare against a representative sample.",
                UserWarning,
                stacklevel=2,
            )
        numeric = X.select_dtypes(include=[np.number])
        reference = pd.Series(index=X.columns, dtype=float)
        if not numeric.empty:
            reference.loc[numeric.columns] = numeric.median()
        for column in X.columns:
            if (
                pd.isna(reference.get(column))
                and not numeric.empty
                and column not in numeric.columns
            ):
                modes = X[column].mode(dropna=True)
                reference.loc[column] = modes.iloc[0] if len(modes) else 0
        return reference

    def explain(self, X: pd.DataFrame) -> list[Explanation]:
        if X.empty:
            return []
        reference = self._reference_row(X)
        features = list(X.columns)
        n_rows = len(X)
        scale_source = self.reference if isinstance(self.reference, pd.DataFrame) else X
        scale = scale_source.apply(pd.to_numeric, errors="coerce").std().reindex(features)
        scale = scale.where(np.isfinite(scale) & (scale > 0), 1.0)
        numeric_X = X.apply(pd.to_numeric, errors="coerce")
        deviation = (numeric_X - pd.to_numeric(reference, errors="coerce")).abs().fillna(0.0)
        scores = (deviation / scale).reindex(columns=features).fillna(0.0)
        score_values = scores.to_numpy(dtype=float)
        n_select = len(features) if self.max_features is None else max(1, self.max_features)
        n_select = min(n_select, len(features))
        order = np.argsort(-score_values, axis=1)[:, :n_select]
        pairs: list[tuple[int, str]] = [
            (i, features[int(position)]) for i in range(n_rows) for position in order[i]
        ]
        feature_positions = {name: position for position, name in enumerate(features)}

        baseline = np.asarray(self.model.predict_pd(X), dtype=float).reshape(-1)
        variants: list[pd.DataFrame] = []
        for row_index, feature in pairs:
            frame = X.iloc[[row_index]].copy()
            frame.loc[:, feature] = reference.loc[feature]
            variants.append(frame)
        occluded = np.asarray(
            self.model.predict_pd(pd.concat(variants, ignore_index=True)), dtype=float
        ).reshape(-1)

        contributions = np.full((n_rows, len(features)), np.nan, dtype=float)
        for variant_index, (row_index, feature) in enumerate(pairs):
            contributions[row_index, feature_positions[feature]] = (
                baseline[row_index] - occluded[variant_index]
            )
        for feature in features:
            if pd.isna(reference.get(feature)):
                contributions[:, feature_positions[feature]] = np.nan
        explanations: list[Explanation] = []
        for i, (index, row) in enumerate(X.iterrows()):
            occluded_pd = [occluded[k] for k, (r, _) in enumerate(pairs) if r == i]
            explanations.append(
                Explanation(
                    entity_id=str(index) if index is not None else None,
                    base_value=float(np.mean(occluded_pd)) if occluded_pd else float(baseline[i]),
                    prediction=float(baseline[i]),
                    contributions=pd.Series(contributions[i], index=features, dtype=float),
                    method="occlusion",
                    feature_values=pd.Series(row.to_numpy(dtype=float), index=features),
                )
            )
        return explanations


class AdditiveExplainer:
    """Explainer for additive log-odds models such as :class:`HeuristicScorer`."""

    def __init__(self, model: Any) -> None:
        if not hasattr(model, "feature_contributions"):
            raise TypeError("AdditiveExplainer requires a model with feature_contributions().")
        self.model = model

    def explain(self, X: pd.DataFrame) -> list[Explanation]:
        contributions = self.model.feature_contributions(X)
        predictions = np.asarray(self.model.predict_pd(X), dtype=float).reshape(-1)
        base = float(getattr(self.model, "intercept", 0.0))
        explanations: list[Explanation] = []
        for i, (_, row) in enumerate(X.iterrows()):
            explanations.append(
                Explanation(
                    entity_id=str(row.name) if row.name is not None else None,
                    base_value=base,
                    prediction=float(predictions[i]),
                    contributions=contributions.iloc[i].astype(float),
                    method="additive",
                    feature_values=row.astype(float),
                )
            )
        return explanations


def explain_model(
    model: Any,
    X: pd.DataFrame,
    *,
    background: pd.DataFrame | None = None,
    max_features: int | None = 30,
) -> list[Explanation]:
    """Choose the best available explainer for ``model`` and explain every row of ``X``.

    ``background`` is forwarded as the occlusion reference for models without SHAP or additive
    contributions. It is otherwise unused by those explainers.
    """
    if getattr(model, "backend", None) in {"lightgbm", "xgboost"}:
        return ShapExplainer(model, background=background).explain(X)
    if hasattr(model, "feature_contributions"):
        return AdditiveExplainer(model).explain(X)
    return OcclusionExplainer(model, reference=background, max_features=max_features).explain(X)


def permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: Any,
    *,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.Series:
    """AUC drop when each feature is randomly permuted, averaged over repeats.

    Unlike SHAP, this measures predictive reliance on a feature and is model-agnostic. It is slower
    because it requires ``n_features * n_repeats`` predictions.
    """
    from sklearn.metrics import roc_auc_score

    y_array = np.asarray(y).astype(int).reshape(-1)
    if len(np.unique(y_array)) < 2:
        raise ValueError("permutation_importance needs both classes.")
    rng = np.random.default_rng(random_state)
    baseline = float(roc_auc_score(y_array, model.predict_pd(X)))
    scores: dict[str, float] = {}
    for column in X.columns:
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            try:
                auc = float(roc_auc_score(y_array, model.predict_pd(shuffled)))
            except ValueError:
                auc = baseline
            drops.append(baseline - auc)
        scores[str(column)] = float(np.mean(drops))
    return pd.Series(scores, name="auc_drop").sort_values(ascending=False)


def explanations_to_frame(explanations: list[Explanation], k: int = 3) -> pd.DataFrame:
    """Flatten explanations into one row per borrower with top reason codes."""
    rows = []
    for explanation in explanations:
        row: dict[str, Any] = {
            "entity_id": explanation.entity_id,
            "prediction": explanation.prediction,
            "method": explanation.method,
        }
        for rank, reason in enumerate(explanation.reason_codes(k), start=1):
            row[f"reason_{rank}"] = reason
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "AdditiveExplainer",
    "Explanation",
    "OcclusionExplainer",
    "ShapExplainer",
    "explain_model",
    "explanations_to_frame",
    "permutation_importance",
]
