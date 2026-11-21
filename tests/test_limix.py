"""Opt-in integration tests for the local LimiX-2 checkpoint.

Run with::

    $env:OPENCASHFLOW_RUN_LIMIX="1"
    $env:LIMIX_MODEL_PATH="C:\\path\\to\\LimiX-2-1016.ckpt"
    pytest tests/test_limix.py -m limix

These tests load the 1.6 GB checkpoint and take a few seconds. They are skipped by default so the
normal test suite stays fast and works without the non-commercial weights.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from opencashflow.models.metrics import evaluate_binary

RUN_LIMIX = os.getenv("OPENCASHFLOW_RUN_LIMIX") == "1"

pytestmark = [
    pytest.mark.limix,
    pytest.mark.slow,
    pytest.mark.skipif(not RUN_LIMIX, reason="Set OPENCASHFLOW_RUN_LIMIX=1 to run LimiX tests"),
]


@pytest.fixture(scope="module")
def scorer():
    from opencashflow.models.limix_scorer import LimiXScorer

    info = LimiXScorer.availability()
    if not info["available"]:
        pytest.skip(f"LimiX unavailable: {info['reasons']}")
    model = LimiXScorer(n_pipelines=1, seed=0, verbose=False)
    yield model
    model.close()


def test_availability_and_license():
    from opencashflow.models.limix_scorer import LimiXScorer

    info = LimiXScorer.availability()
    assert info["available"] is True
    assert info["model_path"]
    assert info["config_dir"]
    assert "Non-Commercial" in LimiXScorer.license_notice()


def test_classification_end_to_end(scorer, feature_frame, feature_labels):
    scorer.fit(feature_frame, feature_labels)
    proba = scorer.predict_proba(feature_frame.head(10))
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    pd_values = scorer.predict_pd(feature_frame)
    assert pd_values.shape == (len(feature_frame),)
    assert ((pd_values >= 0) & (pd_values <= 1)).all()
    metrics = evaluate_binary(feature_labels, pd_values)
    assert metrics["roc_auc"] > 0.4


def test_imputation(scorer, feature_frame, feature_labels):
    scorer.fit(feature_frame, feature_labels)
    query = feature_frame.head(5).copy()
    query.iloc[0, 0] = np.nan
    query.iloc[2, 3] = np.nan
    imputed = scorer.impute(query)
    assert isinstance(imputed, type(query))
    assert imputed.shape == query.shape
    assert not imputed.isna().any().any()


def test_pickle_roundtrip(scorer, feature_frame, feature_labels):
    scorer.fit(feature_frame, feature_labels)
    restored = pickle.loads(pickle.dumps(scorer))
    assert restored.feature_names_in_ == scorer.feature_names_in_
    assert restored.predict_pd(feature_frame.head(3)).shape == (3,)


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
