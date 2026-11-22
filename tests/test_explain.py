"""Tests for SHAP, occlusion and additive explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opencashflow.models.baselines import HeuristicScorer, LogisticBaseline, available_backends
from opencashflow.models.explain import (
    AdditiveExplainer,
    Explanation,
    OcclusionExplainer,
    ShapExplainer,
    explain_model,
    explanations_to_frame,
    permutation_importance,
)


def test_manual_explanation():
    explanation = Explanation(
        entity_id="E1",
        base_value=-1.0,
        prediction=0.4,
        contributions=pd.Series({"revenue_cv": 0.5, "cash_buffer_months": -0.2}),
        method="manual",
        feature_values=pd.Series({"revenue_cv": 1.2, "cash_buffer_months": 3.0}),
    )
    top = explanation.top(2)
    assert list(top["feature"]) == ["revenue_cv", "cash_buffer_months"]
    assert top.iloc[0]["direction"] == "increases risk"
    reasons = explanation.reason_codes(2)
    assert len(reasons) == 1
    assert "revenue_cv" in reasons[0]
    assert "increased predicted risk." in reasons[0]
    payload = explanation.to_dict()
    assert payload["entity_id"] == "E1"
    assert set(payload["contributions"]) == {"revenue_cv", "cash_buffer_months"}
    assert payload["reason_codes"] == reasons


def test_reason_codes_only_positive_and_nan_safe():
    explanation = Explanation(
        entity_id="E2",
        base_value=0.0,
        prediction=0.2,
        contributions=pd.Series(
            {"risk_up": 0.3, "mild_up": 0.1, "not_selected": np.nan, "risk_down": -0.8}
        ),
        method="manual",
        feature_values=pd.Series({"risk_up": 1.0, "mild_up": 2.0, "risk_down": 3.0}),
    )
    reasons = explanation.reason_codes(3)
    assert len(reasons) == 2
    assert "risk_up" in reasons[0]
    assert "mild_up" in reasons[1]
    assert all("increased predicted risk." in reason for reason in reasons)
    assert all("reduced" not in reason for reason in reasons)
    assert explanation.to_dict()["reason_codes"] == reasons
    all_negative = Explanation(
        entity_id="E3",
        base_value=0.0,
        prediction=0.1,
        contributions=pd.Series({"risk_down": -0.5}),
        method="manual",
    )
    assert all_negative.reason_codes(3) == []
    frame = explanations_to_frame([all_negative], k=2)
    assert "reason_1" not in frame.columns


def test_additive_explainer(feature_frame, feature_labels):
    model = HeuristicScorer().fit(feature_frame, feature_labels)
    explanations = AdditiveExplainer(model).explain(feature_frame.head(2))
    assert len(explanations) == 2
    assert all(e.method == "additive" for e in explanations)
    assert len(explanations[0].contributions) == len(model.weights)


def test_occlusion_explainer(feature_frame, feature_labels):
    model = HeuristicScorer().fit(feature_frame, feature_labels)
    explainer = OcclusionExplainer(model, max_features=5)
    explanations = explainer.explain(feature_frame.head(3))
    assert len(explanations) == 3
    assert all(len(e.contributions) == feature_frame.shape[1] for e in explanations)
    assert all(int(e.contributions.notna().sum()) == 5 for e in explanations)
    assert all(e.method == "occlusion" for e in explanations)
    assert all(0 <= e.prediction <= 1 for e in explanations)
    assert all(isinstance(e.reason_codes(2), list) for e in explanations)


def test_occlusion_background_reference_and_single_row_warning(feature_frame, feature_labels):
    model = LogisticBaseline().fit(feature_frame, feature_labels)
    background = feature_frame.head(20)
    row = feature_frame.iloc[[0]]
    explained = OcclusionExplainer(model, reference=background, max_features=None).explain(row)[0]
    assert explained.contributions.notna().all()
    assert explained.contributions.abs().sum() > 0
    with pytest.warns(UserWarning, match="reference"):
        single = OcclusionExplainer(model, max_features=None).explain(row)[0]
    assert single.contributions.abs().sum() == pytest.approx(0.0, abs=1e-12)


class _ToyModel:
    """Deterministic scorer for controlling which occluded feature changes the PD."""

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        ratio = pd.to_numeric(X["ratio"]).to_numpy(dtype=float)
        eur_total = pd.to_numeric(X["eur_total"]).to_numpy(dtype=float)
        return np.clip(0.5 * ratio + 1e-7 * eur_total, 0.0, 1.0)


class _AdditiveModel:
    """Deterministic additive scorer over arbitrary numeric columns."""

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        numeric = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return np.clip(0.01 * numeric.sum(axis=1).to_numpy(dtype=float), 0.0, 1.0)


def test_occlusion_selects_by_standardised_deviation():
    background = pd.DataFrame(
        {
            "ratio": [0.49, 0.50, 0.50, 0.51, 0.50],
            "eur_total": [1.0e6, 1.1e6, 0.9e6, 1.05e6, 0.95e6],
        }
    )
    row = pd.DataFrame({"ratio": [1.0], "eur_total": [2.0e6]})
    explained = OcclusionExplainer(_ToyModel(), reference=background, max_features=1).explain(row)[
        0
    ]
    assert int(explained.contributions.notna().sum()) == 1
    assert pd.notna(explained.contributions["ratio"])
    assert np.isnan(explained.contributions["eur_total"])
    assert explained.contributions["ratio"] > 0
    assert "ratio" in explained.reason_codes(3)[0]


def test_occlusion_zero_variance_reference_uses_unit_scale():
    # A zero-variance reference column uses scale 1.0, so its raw deviation is still scored.
    # flat: |2.02 - 2.0| / 1.0 = 0.02. noisy: |0.5 - 0| / 100 ~= 0.005. flat is selected.
    background = pd.DataFrame(
        {
            "flat": [2.0, 2.0, 2.0, 2.0, 2.0],
            "noisy": [-100.0, 100.0, -100.0, 100.0, 0.0],
        }
    )
    row = pd.DataFrame({"flat": [2.02], "noisy": [0.5]})
    explained = OcclusionExplainer(_AdditiveModel(), reference=background, max_features=1).explain(
        row
    )[0]
    assert pd.notna(explained.contributions["flat"])
    assert np.isnan(explained.contributions["noisy"])


def test_occlusion_explicit_nan_reference_not_explained():
    background = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 2.0, 2.0],
            "b": [1.0, 1.0, 1.0, 1.0, 1.0],
            "c": [4.0, 5.0, 6.0, 5.0, 5.0],
        }
    )
    background["b"] = np.nan
    row = pd.DataFrame({"a": [5.0], "b": [2.0], "c": [9.0]})
    explained = OcclusionExplainer(
        _AdditiveModel(), reference=background, max_features=None
    ).explain(row)[0]
    assert np.isnan(explained.contributions["b"])
    assert np.isfinite(explained.contributions[["a", "c"]]).all()
    assert all("b" not in reason for reason in explained.reason_codes(10))


def test_occlusion_implicit_nan_reference_not_explained():
    X = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [np.nan, np.nan, np.nan],
            "c": [4.0, 5.0, 6.0],
        }
    )
    explained = OcclusionExplainer(_AdditiveModel(), max_features=None).explain(X)[0]
    assert np.isnan(explained.contributions["b"])
    assert np.isfinite(explained.contributions[["a", "c"]]).all()


def test_explain_model_dispatch(feature_frame, feature_labels):
    model = HeuristicScorer().fit(feature_frame, feature_labels)
    explanations = explain_model(model, feature_frame.head(2))
    assert all(e.method == "additive" for e in explanations)
    frame = explanations_to_frame(explanations, k=2)
    assert {"entity_id", "prediction", "reason_1", "reason_2"} <= set(frame.columns)


@pytest.mark.skipif(not available_backends()["lightgbm"], reason="LightGBM not installed")
def test_shap_explainer(feature_frame, feature_labels):
    from opencashflow.models.baselines import TreeBaseline

    model = TreeBaseline("lightgbm", n_estimators=40).fit(feature_frame, feature_labels)
    explanations = ShapExplainer(model).explain(feature_frame.head(2))
    assert len(explanations) == 2
    for explanation in explanations:
        assert explanation.method == "tree_shap"
        assert len(explanation.contributions) == feature_frame.shape[1]
        reasons = explanation.reason_codes(2)
        assert len(reasons) <= 2
        assert all("increased predicted risk." in reason for reason in reasons)
        assert 0 <= explanation.prediction <= 1


def test_permutation_importance(feature_frame, feature_labels):
    model = LogisticBaseline().fit(feature_frame, feature_labels)
    importance = permutation_importance(
        model, feature_frame.head(20), feature_labels.head(20), n_repeats=2
    )
    assert len(importance) == feature_frame.shape[1]
    assert importance.index[0] in feature_frame.columns
