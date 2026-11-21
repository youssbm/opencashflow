"""Tests for SHAP, occlusion and additive explanations."""

from __future__ import annotations

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
    assert len(reasons) == 2
    payload = explanation.to_dict()
    assert payload["entity_id"] == "E1"
    assert set(payload["contributions"]) == {"revenue_cv", "cash_buffer_months"}


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
    assert all(len(e.contributions) == 5 for e in explanations)
    assert all(e.method == "occlusion" for e in explanations)
    assert all(0 <= e.prediction <= 1 for e in explanations)


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
        assert explanation.reason_codes(2)
        assert 0 <= explanation.prediction <= 1


def test_permutation_importance(feature_frame, feature_labels):
    model = LogisticBaseline().fit(feature_frame, feature_labels)
    importance = permutation_importance(
        model, feature_frame.head(20), feature_labels.head(20), n_repeats=2
    )
    assert len(importance) == feature_frame.shape[1]
    assert importance.index[0] in feature_frame.columns
