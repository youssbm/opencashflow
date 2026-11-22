"""Tests for the benchmark harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import opencashflow.benchmark as benchmark_module
from opencashflow.benchmark import _to_markdown_table, run_benchmark
from opencashflow.data.loaders import DatasetBundle


def _toy_bundle(labels: list[int] | None = None) -> DatasetBundle:
    """Tiny table with a numeric time column for split tests."""
    X = pd.DataFrame(
        {
            "approval_fy": [2001, 2001, 2002, 2002, 2003, 2003, 2004, 2004],
            "feat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    y = pd.Series(labels if labels is not None else [0, 1, 0, 1, 0, 1, 0, 1])
    return DatasetBundle(name="toy", X=X, y=y)


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


def test_repeats_add_rows_and_ci():
    report = run_benchmark(
        datasets=["synthetic"],
        models=["heuristic", "logistic"],
        cv=3,
        repeats=2,
        seed=7,
        synthetic_companies=30,
        synthetic_months=12,
        verbose=False,
    )
    assert len(report.results) == 12
    assert report.results.groupby("model").size().eq(6).all()
    assert set(report.results["repeat"]) == {1, 2}
    assert report.summary["n_runs"].eq(6).all()
    assert report.summary["roc_auc_ci_low"].notna().all()
    text = report.to_markdown()
    for header in ("variant", "split", "predict_seconds", "roc_auc"):
        assert header in text
    assert "[" in text
    assert report.results["time_cutoff"].isna().all()


def test_oot_split_counts_and_single_run_ci(monkeypatch):
    monkeypatch.setattr(
        benchmark_module, "_load_benchmark_dataset", lambda *args, **kwargs: _toy_bundle()
    )
    report = run_benchmark(
        datasets=["toy"],
        models=["logistic"],
        cv=2,
        split="oot",
        time_column="approval_fy",
        oot_cutoff=2002,
        variant="toy_variant",
        verbose=False,
    )
    assert len(report.results) == 1
    row = report.results.iloc[0]
    assert (row["n_train"], row["n_test"]) == (4, 4)
    assert (row["fold"], row["repeat"]) == (1, 1)
    assert row["time_cutoff"] == 2002.0
    assert row["split"] == "oot"
    assert row["variant"] == "toy_variant"
    assert report.summary["n_runs"].eq(1).all()
    assert report.summary["roc_auc_ci_low"].isna().all()
    assert report.summary["roc_auc_ci_high"].isna().all()


def test_oot_requires_both_classes(monkeypatch):
    monkeypatch.setattr(
        benchmark_module,
        "_load_benchmark_dataset",
        lambda *args, **kwargs: _toy_bundle([0, 0, 0, 0, 1, 1, 1, 1]),
    )
    with pytest.raises(ValueError, match="both classes"):
        run_benchmark(
            datasets=["toy"],
            models=["logistic"],
            cv=2,
            split="oot",
            time_column="approval_fy",
            oot_cutoff=2002,
            verbose=False,
        )


def test_exclude_features_reduces_n_features(monkeypatch):
    monkeypatch.setattr(
        benchmark_module, "_load_benchmark_dataset", lambda *args, **kwargs: _toy_bundle()
    )
    base = run_benchmark(datasets=["toy"], models=["logistic"], cv=2, verbose=False)
    reduced = run_benchmark(
        datasets=["toy"],
        models=["logistic"],
        cv=2,
        exclude_features=("feat", "not_a_column"),
        verbose=False,
    )
    assert reduced.results["n_features"].iloc[0] == base.results["n_features"].iloc[0] - 1


def test_summary_ci_uses_t_critical():
    """Six known ROC-AUCs pin the interval to a t quantile, not a normal one."""
    aucs = [0.80, 0.82, 0.78, 0.85, 0.79, 0.81]
    results = pd.DataFrame(
        {
            "dataset": ["toy"] * 6,
            "model": ["logistic"] * 6,
            "variant": ["all"] * 6,
            "split": ["stratified"] * 6,
            "n": [8] * 6,
            "n_features": [2] * 6,
            "default_rate": [0.5] * 6,
            "roc_auc": aucs,
            "ks": [0.5] * 6,
            "pr_auc": [0.5] * 6,
            "brier": [0.5] * 6,
            "log_loss": [0.5] * 6,
            "fit_seconds": [0.01] * 6,
            "predict_seconds": [0.01] * 6,
        }
    )
    row = benchmark_module._summarize_results(results).iloc[0]
    assert row["n_runs"] == 6
    assert row["roc_auc"] == pytest.approx(0.808333, abs=1e-6)
    assert row["roc_auc_std"] == pytest.approx(0.024833, abs=1e-6)
    assert stats.t.ppf(0.975, 5) == pytest.approx(2.570582, abs=1e-6)
    assert row["roc_auc_ci_low"] == pytest.approx(0.782273, abs=1e-6)
    assert row["roc_auc_ci_high"] == pytest.approx(0.834394, abs=1e-6)


def _load_script_main():
    """Load ``benchmarks/run_benchmark.py`` main from its file path."""
    script = Path(__file__).resolve().parents[1] / "benchmarks" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_script_default_prints_without_writing(monkeypatch, tmp_path, capsys):
    """The CLI makes no files unless --output-dir is supplied."""
    main = _load_script_main()
    monkeypatch.chdir(tmp_path)
    code = main(
        [
            "--datasets",
            "synthetic",
            "--models",
            "heuristic",
            "--cv",
            "2",
            "--synthetic-companies",
            "20",
            "--synthetic-months",
            "8",
            "--quiet",
        ]
    )
    assert code == 0
    assert not (tmp_path / "benchmarks").exists()
    assert not list(tmp_path.rglob("*.csv"))
    captured = capsys.readouterr()
    assert "No --output-dir given: results were printed, not saved." in captured.out
    assert "synthetic" in captured.out


def test_sba_vintage_bound_is_passed_to_loader(monkeypatch):
    import pandas as pd

    import opencashflow.benchmark as benchmark_module
    from opencashflow.data.loaders import DatasetBundle

    captured: dict = {}

    def fake_load_dataset(name, **kwargs):
        captured.update(kwargs)
        return DatasetBundle(name="sba", X=pd.DataFrame({"a": [1.0, 2.0]}), y=pd.Series([0, 1]))

    monkeypatch.setattr(benchmark_module, "load_dataset", fake_load_dataset)
    benchmark_module._load_benchmark_dataset(
        "sba",
        seed=1,
        synthetic_companies=10,
        synthetic_months=6,
        max_rows=100,
        sba_approval_fy_max=2005,
    )
    assert captured["approval_fy_max"] == 2005
    assert captured["max_rows"] == 100
