"""Tests for baselines, metrics and the untrained scorecard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from opencashflow.models.baselines import (
    HeuristicScorer,
    LogisticBaseline,
    TreeBaseline,
    available_backends,
    make_baseline,
)
from opencashflow.models.metrics import (
    calibration_table,
    cost_sensitive_threshold,
    evaluate_binary,
    expected_loss,
    gini_from_auc,
    ks_statistic,
    lift_at_k,
    rating_from_pd,
)


def test_metrics_on_perfect_scores():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = evaluate_binary(y, scores)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["ks"] == pytest.approx(1.0)
    assert metrics["gini"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["brier"] < 0.05
    assert metrics["n"] == 4
    assert metrics["n_positive"] == 2


def test_metrics_single_class_returns_nan_auc():
    metrics = evaluate_binary([0, 0, 0], [0.1, 0.2, 0.3])
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["ks"])
    assert np.isnan(metrics["gini"])


def test_metric_helpers():
    assert gini_from_auc(0.75) == pytest.approx(0.5)
    assert np.isnan(gini_from_auc(float("nan")))
    assert ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert expected_loss(0.1, lgd=0.5, ead=1000) == pytest.approx(50.0)
    assert rating_from_pd(0.01) == "A"
    assert rating_from_pd(0.9) == "G"
    assert list(rating_from_pd(np.array([0.01, 0.9]))) == ["A", "G"]
    assert lift_at_k([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], k=0.5) == pytest.approx(2.0)


def test_calibration_table():
    table = calibration_table([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_bins=5)
    assert {"bin_lower", "bin_upper", "count", "mean_predicted", "observed", "gap"} <= set(
        table.columns
    )
    assert table["count"].sum() == 4


def test_cost_sensitive_threshold():
    threshold, cost = cost_sensitive_threshold(
        [0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], cost_false_positive=1, cost_false_negative=5
    )
    assert 0 < threshold < 1
    assert cost >= 0


def test_heuristic_scorecard(feature_frame, feature_labels):
    model = HeuristicScorer().fit(feature_frame, feature_labels)
    pd_values = model.predict_pd(feature_frame)
    proba = model.predict_proba(feature_frame)
    assert pd_values.shape == (len(feature_frame),)
    assert ((pd_values >= 0) & (pd_values <= 1)).all()
    assert np.allclose(proba.sum(axis=1), 1.0)
    contributions = model.feature_contributions(feature_frame)
    assert contributions.shape[0] == len(feature_frame)
    assert "cashflow_health_score" in contributions.columns
    metrics = evaluate_binary(feature_labels, pd_values)
    assert metrics["roc_auc"] > 0.5


def test_default_baselines_preserve_base_rate():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(1500, 5)), columns=[f"x{i}" for i in range(5)])
    y = (rng.random(1500) < 0.20).astype(int)
    logistic = LogisticBaseline().fit(X, y)
    assert abs(float(logistic.predict_pd(X).mean()) - 0.20) < 0.05
    assert logistic.class_weight is None
    if available_backends()["lightgbm"]:
        tree = TreeBaseline("lightgbm").fit(X, y)
        assert abs(float(tree.predict_pd(X).mean()) - 0.20) < 0.05
        assert tree.class_weight is None


def test_logistic_numeric_transform_options():
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=[f"x{i}" for i in range(4)])
    y = (X["x0"] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    for option in ("quantile", "standard"):
        model = LogisticBaseline(numeric_transform=option).fit(X, y)
        proba = model.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.allclose(proba, model.predict_proba(X))
    with pytest.raises(ValueError, match="numeric_transform"):
        LogisticBaseline(numeric_transform="rank")


def test_logistic_standard_matches_manual_pipeline():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(13)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=[f"x{i}" for i in range(4)])
    y = (X["x0"] > 0).astype(int)
    model = LogisticBaseline(numeric_transform="standard").fit(X, y)
    manual = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(C=1.0, class_weight=None, max_iter=3_000, random_state=42),
            ),
        ]
    ).fit(X, y)
    actual = model.model_.named_steps["model"].coef_.reshape(-1)
    expected = manual.named_steps["model"].coef_.reshape(-1)
    assert np.allclose(actual, expected, atol=1e-6)


def test_logistic_baseline(feature_frame, feature_labels):
    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame, feature_labels, test_size=0.3, random_state=1, stratify=feature_labels
    )
    model = LogisticBaseline().fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert model.coefficients_.shape[0] > 0
    # With only 30 companies and 151 features, the linear model is data-starved; this is a
    # mechanics test, not a quality claim. The benchmark suite covers quality on larger samples.
    metrics = evaluate_binary(y_test, model.predict_pd(X_test))
    assert np.isfinite(metrics["roc_auc"])
    assert 0.0 <= metrics["roc_auc"] <= 1.0


@pytest.mark.skipif(not available_backends()["lightgbm"], reason="LightGBM not installed")
def test_lightgbm_baseline(feature_frame, feature_labels):
    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame, feature_labels, test_size=0.3, random_state=2, stratify=feature_labels
    )
    model = TreeBaseline("lightgbm", n_estimators=20, min_child_samples=2, num_leaves=7).fit(
        X_train, y_train
    )
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    metrics = evaluate_binary(y_test, model.predict_pd(X_test))
    assert np.isfinite(metrics["roc_auc"])
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    importance = model.feature_importances_
    assert importance.sum() == pytest.approx(1.0)
    assert len(importance) == len(model.feature_names_in_)


@pytest.mark.skipif(not available_backends()["xgboost"], reason="XGBoost not installed")
def test_xgboost_baseline(feature_frame, feature_labels):
    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame, feature_labels, test_size=0.3, random_state=3, stratify=feature_labels
    )
    model = TreeBaseline("xgboost", n_estimators=60).fit(X_train, y_train)
    assert model.predict_proba(X_test).shape == (len(X_test), 2)
    metrics = evaluate_binary(y_test, model.predict_pd(X_test))
    assert np.isfinite(metrics["roc_auc"])


def test_categorical_handling():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "num": rng.normal(size=30),
            "cat": ["a", "b", "c"] * 10,
            "flag": [True, False] * 15,
        }
    )
    labels = pd.Series((frame["cat"] == "a").astype(int))
    logistic = LogisticBaseline().fit(frame, labels)
    assert logistic.predict_proba(frame).shape == (30, 2)
    if available_backends()["lightgbm"]:
        tree = TreeBaseline("lightgbm", n_estimators=20).fit(frame, labels)
        assert tree.predict_proba(frame).shape == (30, 2)
        assert tree.categorical_features_ == ["cat", "flag"]


def test_factory_and_unknown_backend():
    assert isinstance(make_baseline("logistic"), LogisticBaseline)
    assert isinstance(make_baseline("heuristic"), HeuristicScorer)
    with pytest.raises(ValueError, match="Unknown baseline"):
        make_baseline("magic")


def test_binary_class_validation():
    with pytest.raises(ValueError, match="two classes"):
        LogisticBaseline().fit(pd.DataFrame({"x": [1, 2, 3]}), [0, 0, 0])


@pytest.mark.parametrize("numeric_transform", ["quantile", "standard"])
def test_logistic_all_nan_numeric_with_categorical(numeric_transform):
    rng = np.random.default_rng(42)
    n = 20
    X = pd.DataFrame(
        {
            "empty": np.full(n, np.nan),
            "cat": ["a", "b"] * (n // 2),
        }
    )
    y = np.array([0, 1] * (n // 2))
    rng.shuffle(y)
    model = LogisticBaseline(numeric_transform=numeric_transform).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0)
    new_rows = X.iloc[:4].copy()
    new_rows["empty"] = np.nan
    new_proba = model.predict_proba(new_rows)
    assert new_proba.shape == (4, 2)
    assert np.isfinite(new_proba).all()
    assert np.allclose(new_proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("numeric_transform", ["quantile", "standard"])
def test_logistic_only_all_nan_numeric(numeric_transform):
    rng = np.random.default_rng(42)
    n = 20
    X = pd.DataFrame({"empty": np.full(n, np.nan)})
    y = np.array([0, 1] * (n // 2))
    rng.shuffle(y)
    model = LogisticBaseline(numeric_transform=numeric_transform).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0)
    new_proba = model.predict_proba(pd.DataFrame({"empty": [np.nan, np.nan]}))
    assert new_proba.shape == (2, 2)
    assert np.isfinite(new_proba).all()
    assert np.allclose(new_proba.sum(axis=1), 1.0)
