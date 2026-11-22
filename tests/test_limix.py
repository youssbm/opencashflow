"""Tests for the LimiX-2 wrapper.

The integration tests load the 1.6 GB checkpoint and are opt-in. Set the environment before running::

    $env:OPENCASHFLOW_RUN_LIMIX="1"
    $env:LIMIX_MODEL_PATH="C:\\path\\to\\LimiX-2-1016.ckpt"
    pytest tests/test_limix.py -q -m limix

The unit tests need no checkpoint and run in the normal suite.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import pytest

RUN_LIMIX = os.getenv("OPENCASHFLOW_RUN_LIMIX") == "1"


def limix_test(func):
    """Apply the opt-in markers to a test that needs the LimiX checkpoint."""
    marked = func
    for mark in (
        pytest.mark.limix,
        pytest.mark.slow,
        pytest.mark.skipif(not RUN_LIMIX, reason="Set OPENCASHFLOW_RUN_LIMIX=1 to run LimiX tests"),
    ):
        marked = mark(marked)
    return marked


# --------------------------------------------------------------------- unit tests, no checkpoint


def test_softmax_temperature_defaults_by_task():
    from opencashflow.models.limix_scorer import LimiXScorer

    classification = LimiXScorer()
    regression = LimiXScorer(task="regression")
    assert classification.softmax_temperature == 0.9
    assert regression.softmax_temperature == 1.0
    assert classification.describe()["softmax_temperature"] == 0.9
    assert regression.describe()["softmax_temperature"] == 1.0


def test_softmax_temperature_explicit_value_is_kept():
    from opencashflow.models.limix_scorer import LimiXScorer

    scorer = LimiXScorer(task="regression", softmax_temperature=0.7)
    assert scorer.softmax_temperature == 0.7


def test_dataset_name_is_content_addressed():
    from opencashflow.models.limix_scorer import _make_dataset_name

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = [0, 1]
    name = _make_dataset_name(0, X, y)
    assert name == _make_dataset_name(0, X.copy(), [0, 1])
    assert name.startswith("opencashflow-0-")
    assert len(name.split("-")[-1]) == 16
    assert _make_dataset_name(0, X + 1.0, y) != name
    assert _make_dataset_name(0, X, [1, 0]) != name
    assert _make_dataset_name(1, X, y) != name


def test_categorical_encoding_is_stable_and_unseen_maps_to_minus_one():
    from opencashflow.models.limix_scorer import LimiXScorer

    X = pd.DataFrame({"cat": ["b", "a", "b", "a"], "num": [1.0, 2.0, 3.0, 4.0]})
    y = [0, 1, 0, 1]
    first = LimiXScorer(n_pipelines=1, seed=0, verbose=False).fit(X, y)
    second = LimiXScorer(n_pipelines=1, seed=0, verbose=False).fit(X, y)

    assert first.categories_["cat"].tolist() == ["a", "b"]
    encoded_first = first._prepare_X(X, fit=False)
    encoded_second = second._prepare_X(X, fit=False)
    np.testing.assert_array_equal(encoded_first, encoded_second)
    assert encoded_first[:, 0].tolist() == [1.0, 0.0, 1.0, 0.0]

    unseen = pd.DataFrame({"cat": ["c", "a"], "num": [1.0, 2.0]})
    encoded_unseen = first._prepare_X(unseen, fit=False)
    assert encoded_unseen[0, 0] == -1.0
    assert encoded_unseen[1, 0] == 0.0


def test_impute_replaces_only_missing_cells(monkeypatch):
    """The imputation path must send NaN to LimiX and keep observed cells untouched."""
    from opencashflow.models.limix_scorer import LimiXScorer

    frame = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "b": [4.0, 5.0, np.nan],
            "c": [7.0, 8.0, 9.0],
        }
    )
    observed = frame.to_numpy(dtype=float)
    missing = np.isnan(observed)

    class FakePredictor:
        def __init__(self) -> None:
            self.query: np.ndarray | None = None
            self.kwargs: dict = {}

        def predict(self, ref, y_train, query, **kwargs):
            self.query = np.asarray(query, dtype=float).copy()
            self.kwargs = kwargs
            # Reconstruct every cell, like the real predictor. A constant value makes the
            # observed-cell restoration a real check: without it, observed cells become 99.
            reconstructed = np.full_like(self.query, 99.0)
            return np.vstack([np.asarray(ref, dtype=float), reconstructed])

    model = LimiXScorer(n_pipelines=1, seed=0, verbose=False)
    fake = FakePredictor()
    monkeypatch.setattr(model, "_ensure_imputation_predictor", lambda: fake)

    result = model.impute(frame)

    assert isinstance(result, pd.DataFrame)
    result_values = result.to_numpy(dtype=float)
    np.testing.assert_array_equal(result_values[~missing], observed[~missing])
    np.testing.assert_array_equal(result_values[missing], np.full(int(missing.sum()), 99.0))
    assert not np.isnan(result_values).any()
    assert fake.query is not None
    assert np.isnan(fake.query[missing]).all()
    assert fake.kwargs["task_type"] == "Feature_imputation"
    assert fake.kwargs["unique_dataset_name"].startswith("opencashflow-0-")


def test_impute_falls_back_for_nan_from_limix(monkeypatch):
    """A NaN left by LimiX is median-filled and raises exactly one UserWarning."""
    from opencashflow.models.limix_scorer import LimiXScorer

    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})

    class FakePredictor:
        def predict(self, ref, y_train, query, **kwargs):
            reconstructed = np.full_like(np.asarray(query, dtype=float), 99.0)
            reconstructed[np.isnan(query)] = np.nan
            return np.vstack([np.asarray(ref, dtype=float), reconstructed])

    model = LimiXScorer(n_pipelines=1, seed=0, verbose=False)
    monkeypatch.setattr(model, "_ensure_imputation_predictor", lambda: FakePredictor())

    with pytest.warns(UserWarning) as record:
        result = model.impute(frame)

    assert len(record) == 1
    assert "1" in str(record[0].message)
    assert "median" in str(record[0].message).lower()
    values = result.to_numpy(dtype=float)
    assert not np.isnan(values).any()
    assert values[1, 0] == pytest.approx(model.fill_values_[0])
    np.testing.assert_array_equal(values[0], [1.0, 2.0])
    assert values[1, 1] == 3.0


def test_impute_explicit_reference_uses_zero_labels(monkeypatch):
    """An explicit reference is unlabelled; the predictor sees zeros of length len(ref)."""
    from opencashflow.models.limix_scorer import LimiXScorer

    rng = np.random.default_rng(0)
    train = pd.DataFrame(rng.normal(size=(10, 2)), columns=["a", "b"])
    y = np.array([0, 1] * 5)
    reference = train.head(7).copy()
    query = train.head(2).copy()
    query.iloc[0, 0] = np.nan

    class FakePredictor:
        def __init__(self) -> None:
            self.y_train: np.ndarray | None = None
            self.n_ref: int = -1

        def predict(self, ref, y_train, query, **kwargs):
            self.y_train = np.asarray(y_train, dtype=float).copy()
            self.n_ref = np.asarray(ref).shape[0]
            query_block = np.full_like(np.asarray(query, dtype=float), 99.0)
            return np.vstack([np.asarray(ref, dtype=float), query_block])

    model = LimiXScorer(n_pipelines=1, seed=0, verbose=False).fit(train, y)
    fake = FakePredictor()
    monkeypatch.setattr(model, "_ensure_imputation_predictor", lambda: fake)

    model.impute(query, reference=reference)

    assert fake.n_ref == 7
    assert fake.y_train is not None
    assert len(fake.y_train) == 7
    np.testing.assert_array_equal(fake.y_train, np.zeros(7))


# --------------------------------------------------------------------- opt-in integration tests


@pytest.fixture(scope="module")
def scorer():
    from opencashflow.models.limix_scorer import LimiXScorer

    info = LimiXScorer.availability()
    if not info["available"]:
        pytest.skip(f"LimiX unavailable: {info['reasons']}")
    model = LimiXScorer(n_pipelines=1, seed=0, verbose=False)
    yield model
    model.close()


@limix_test
def test_availability_and_license():
    from opencashflow.models.limix_scorer import LimiXScorer

    info = LimiXScorer.availability()
    assert info["available"] is True
    assert info["model_path"]
    assert info["config_dir"]
    assert "Non-Commercial" in LimiXScorer.license_notice()


@limix_test
def test_classification_end_to_end(scorer, feature_frame, feature_labels):
    """Mechanics on an out-of-sample split; no quality threshold is asserted."""
    cut = int(0.7 * len(feature_frame))
    scorer.fit(feature_frame.iloc[:cut], feature_labels.iloc[:cut])
    query = feature_frame.iloc[cut:]

    proba = scorer.predict_proba(query)
    assert proba.shape == (len(query), 2)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0)

    pd_values = scorer.predict_pd(query)
    assert pd_values.shape == (len(query),)
    assert np.isfinite(pd_values).all()
    assert ((pd_values >= 0) & (pd_values <= 1)).all()


@limix_test
def test_imputation_preserves_observed_cells(scorer, feature_frame, feature_labels):
    """Mask ten observed cells and confirm LimiX touches only those cells."""
    scorer.fit(feature_frame, feature_labels)
    query = feature_frame.head(10).copy()
    values = query.to_numpy(dtype=float, copy=True)
    rng = np.random.default_rng(0)
    flat_index = rng.choice(np.flatnonzero(~np.isnan(values)), size=10, replace=False)
    masked = np.unravel_index(flat_index, values.shape)
    true_values = values[masked].copy()
    values[masked] = np.nan
    query = pd.DataFrame(values, columns=query.columns, index=query.index)

    imputed = scorer.impute(query)
    result_values = imputed.to_numpy(dtype=float)
    observed = ~np.isnan(values)

    training_medians = np.nanmedian(feature_frame.to_numpy(dtype=float), axis=0)
    median_values = training_medians[masked[1]]
    mae_limix = float(np.mean(np.abs(result_values[masked] - true_values)))
    mae_median = float(np.mean(np.abs(median_values - true_values)))
    print(f"IMPUTATION_MAE_LIMIX_VS_TRUTH={mae_limix:.6f}")
    print(f"IMPUTATION_MAE_MEDIAN_VS_TRUTH={mae_median:.6f}")
    print(f"IMPUTATION_MASKED_TRUE={np.round(true_values, 3).tolist()}")
    print(f"IMPUTATION_MASKED_PREDICTED={np.round(result_values[masked], 3).tolist()}")

    np.testing.assert_array_equal(result_values[observed], values[observed])
    assert not np.isnan(result_values).any()


@limix_test
def test_pickle_roundtrip(scorer, feature_frame, feature_labels):
    scorer.fit(feature_frame, feature_labels)
    restored = pickle.loads(pickle.dumps(scorer))
    assert restored.feature_names_in_ == scorer.feature_names_in_
    assert restored.predict_pd(feature_frame.head(3)).shape == (3,)


@limix_test
def test_imputation_requires_no_fit(feature_frame):
    from opencashflow.models.limix_scorer import LimiXScorer

    if not LimiXScorer.availability()["available"]:
        pytest.skip("LimiX unavailable")
    model = LimiXScorer(n_pipelines=1, seed=0, verbose=False)
    query = feature_frame.head(3).copy()
    query.iloc[0, 1] = np.nan
    imputed = model.impute(query)
    assert imputed.shape == query.shape
    model.close()
