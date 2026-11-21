"""Tests for the benchmark harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opencashflow.benchmark import _to_markdown_table, run_benchmark


def test_markdown_table():
    frame = pd.DataFrame({"model": ["a", "b"], "auc": [0.8123, np.nan]})
    text = _to_markdown_table(frame)
    assert "| model | auc |" in text
    assert "0.812" in text
    assert "n/a" in text


def test_empty_markdown():
    assert _to_markdown_table(pd.DataFrame()) == "_No results._"


@pytest.mark.slow
def test_run_benchmark_synthetic(tmp_path):
    report = run_benchmark(
        datasets=["synthetic"],
        models=["heuristic", "logistic"],
        cv=2,
        seed=7,
        synthetic_companies=20,
        synthetic_months=10,
        limix_pipelines=1,
        output_dir=tmp_path,
        verbose=False,
    )
    assert len(report.results) == 4  # 2 models x 2 folds
    assert {"dataset", "model", "roc_auc", "ks", "brier"} <= set(report.summary.columns)
    assert set(report.summary["model"]) == {"heuristic", "logistic"}
    assert report.summary["roc_auc"].between(0.0, 1.0).all()
    assert (tmp_path / "benchmark_results.csv").is_file()
    assert (tmp_path / "benchmark_summary.md").is_file()
    assert "synthetic" in report.to_markdown()


def test_benchmark_rejects_bad_cv():
    with pytest.raises(ValueError, match="cv"):
        run_benchmark(datasets=["synthetic"], models=["heuristic"], cv=1)
