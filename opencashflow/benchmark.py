"""Benchmark suite: LimiX-2 versus tree and linear baselines.

The benchmark uses stratified K-fold cross-validation and reports ROC-AUC, KS, PR-AUC, Brier score
and log loss. It is deliberately conservative:

* every model sees the same folds;
* preprocessing is fitted inside each fold;
* LimiX receives only the training fold as its in-context reference set;
* no test-fold information is used for feature selection or calibration.

Example:
    >>> from opencashflow.benchmark import run_benchmark
    >>> report = run_benchmark(
    ...     datasets=("german",), models=("logistic", "lightgbm"), cv=3
    ... )
    >>> set(report.summary["model"]) >= {"logistic", "lightgbm"}
    True
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from opencashflow.data.loaders import DatasetBundle, load_dataset, load_synthetic
from opencashflow.models.baselines import available_backends, make_baseline
from opencashflow.models.limix_scorer import LimiXScorer
from opencashflow.models.metrics import evaluate_binary

DEFAULT_DATASETS = ("synthetic", "german", "bangladesh_sme", "polish")
DEFAULT_MODELS = ("heuristic", "logistic", "lightgbm", "xgboost", "limix")
METRIC_COLUMNS = ("roc_auc", "ks", "pr_auc", "brier", "log_loss")


def _to_markdown_table(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown table writer."""
    if frame.empty:
        return "_No results._"
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.3f}" if np.isfinite(value) else "n/a")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


@dataclass
class BenchmarkReport:
    """Long-form fold results and an aggregated summary table."""

    results: pd.DataFrame
    summary: pd.DataFrame
    output_dir: Path | None = None

    def to_markdown(self) -> str:
        """Render the summary as a Markdown table."""
        columns = [
            "dataset",
            "model",
            "n",
            "n_features",
            "default_rate",
            "roc_auc",
            "roc_auc_std",
            "ks",
            "brier",
            "fit_seconds",
        ]
        available = [c for c in columns if c in self.summary.columns]
        return _to_markdown_table(self.summary[available].copy())

    def save(self, output_dir: str | Path) -> Path:
        """Save fold results, summary CSV and Markdown summary."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(directory / "benchmark_results.csv", index=False)
        self.summary.to_csv(directory / "benchmark_summary.csv", index=False)
        (directory / "benchmark_summary.md").write_text(
            "# OpenCashFlow benchmark\n\n" + self.to_markdown() + "\n", encoding="utf-8"
        )
        self.output_dir = directory
        return directory


def _load_benchmark_dataset(
    name: str,
    *,
    seed: int,
    synthetic_companies: int,
    synthetic_months: int,
    max_rows: int | None,
) -> DatasetBundle:
    key = name.lower().strip()
    if key == "synthetic":
        bundle = load_synthetic(
            n_companies=synthetic_companies,
            months=synthetic_months,
            seed=seed,
        )
        assert isinstance(bundle, DatasetBundle)
    else:
        bundle = load_dataset(key)
    if max_rows is not None and len(bundle.X) > max_rows:
        X, _, y, _ = train_test_split(
            bundle.X,
            bundle.y,
            train_size=max_rows,
            random_state=seed,
            stratify=bundle.y,
        )
        bundle = DatasetBundle(
            name=bundle.name,
            X=X,
            y=y,
            target_name=bundle.target_name,
            categorical_features=[c for c in bundle.categorical_features if c in X.columns],
            source=bundle.source,
            license=bundle.license,
            citation=bundle.citation,
            metadata={**bundle.metadata, "subsampled_rows": max_rows},
        )
    return bundle


def _build_model(name: str, *, seed: int, limix_pipelines: int) -> Any:
    key = name.lower().strip()
    if key == "limix":
        return LimiXScorer(n_pipelines=limix_pipelines, seed=seed, verbose=False)
    if key in {"heuristic", "scorecard", "rules"}:
        return make_baseline(key)
    if key in {"lightgbm", "lgbm", "xgboost", "xgb"}:
        return make_baseline(key, random_state=seed, n_estimators=300)
    return make_baseline(key, random_state=seed)


def _requested_models(models: Iterable[str]) -> list[str]:
    requested = [m.lower().strip() for m in models]
    backends = available_backends()
    available: list[str] = []
    for name in requested:
        if name == "limix":
            info = LimiXScorer.availability()
            if not info["available"]:
                warnings.warn("Skipping LimiX: " + "; ".join(info["reasons"]), stacklevel=2)
                continue
        elif name in {"lightgbm", "lgbm"} and not backends["lightgbm"]:
            warnings.warn("Skipping LightGBM: not installed.", stacklevel=2)
            continue
        elif name in {"xgboost", "xgb"} and not backends["xgboost"]:
            warnings.warn("Skipping XGBoost: not installed.", stacklevel=2)
            continue
        available.append(name)
    return available


def run_benchmark(
    *,
    datasets: Iterable[str] = DEFAULT_DATASETS,
    models: Iterable[str] = DEFAULT_MODELS,
    cv: int = 3,
    seed: int = 42,
    max_rows: int | None = None,
    synthetic_companies: int = 160,
    synthetic_months: int = 18,
    limix_pipelines: int = 1,
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> BenchmarkReport:
    """Run stratified cross-validated benchmarks.

    Args:
        datasets: Dataset names from :data:`opencashflow.data.loaders.DATASET_REGISTRY`.
        models: ``heuristic``, ``logistic``, ``lightgbm``, ``xgboost`` and/or ``limix``.
        cv: Number of stratified folds.
        seed: Random seed used for folds, model initialisation and synthetic data.
        max_rows: Optional stratified row cap per dataset, useful for quick runs.
        synthetic_companies: Portfolio size for the synthetic dataset.
        synthetic_months: Months of history for the synthetic dataset.
        limix_pipelines: Number of LimiX ensemble members. One is fast on CPU.
        output_dir: When set, the report is written to this directory.
        verbose: Print progress.

    Returns:
        A :class:`BenchmarkReport` with long-form fold results and an aggregated summary.
    """
    if cv < 2:
        raise ValueError("cv must be at least 2.")
    selected_models = _requested_models(models)
    if not selected_models:
        raise ValueError("No available models were selected.")
    rows: list[dict[str, Any]] = []

    for dataset_name in datasets:
        if verbose:
            print(f"[dataset] {dataset_name}")
        bundle = _load_benchmark_dataset(
            dataset_name,
            seed=seed,
            synthetic_companies=synthetic_companies,
            synthetic_months=synthetic_months,
            max_rows=max_rows,
        )
        X, y = bundle.X, bundle.y
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        folds = list(splitter.split(X, y))
        for model_name in selected_models:
            if model_name == "heuristic" and dataset_name != "synthetic":
                if verbose:
                    print("  [skip] heuristic: needs cash-flow features")
                continue
            if verbose:
                print(f"  [model] {model_name}")
            shared_limix: LimiXScorer | None = None
            try:
                for fold, (train_idx, test_idx) in enumerate(folds, start=1):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    if model_name == "limix":
                        if shared_limix is None:
                            shared_limix = _build_model(
                                model_name, seed=seed, limix_pipelines=limix_pipelines
                            )
                        model = shared_limix
                    else:
                        model = _build_model(model_name, seed=seed, limix_pipelines=limix_pipelines)
                    fit_start = time.perf_counter()
                    model.fit(X_train, y_train)
                    fit_seconds = time.perf_counter() - fit_start
                    predict_start = time.perf_counter()
                    probabilities = model.predict_pd(X_test)
                    predict_seconds = time.perf_counter() - predict_start
                    metrics = evaluate_binary(y_test, probabilities)
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "fold": fold,
                            "n": len(y),
                            "n_train": len(y_train),
                            "n_test": len(y_test),
                            "n_features": X.shape[1],
                            "default_rate": float(y.mean()),
                            "fit_seconds": fit_seconds,
                            "predict_seconds": predict_seconds,
                            **{metric: metrics[metric] for metric in METRIC_COLUMNS},
                        }
                    )
            finally:
                if shared_limix is not None:
                    shared_limix.close()

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("Benchmark produced no results. Check dataset and model names.")
    summary = (
        results.groupby(["dataset", "model"], as_index=False)
        .agg(
            n=("n", "max"),
            n_features=("n_features", "max"),
            default_rate=("default_rate", "mean"),
            roc_auc=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            ks=("ks", "mean"),
            pr_auc=("pr_auc", "mean"),
            brier=("brier", "mean"),
            log_loss=("log_loss", "mean"),
            fit_seconds=("fit_seconds", "mean"),
            predict_seconds=("predict_seconds", "mean"),
        )
        .sort_values(["dataset", "roc_auc"], ascending=[True, False])
        .reset_index(drop=True)
    )
    report = BenchmarkReport(results=results, summary=summary)
    if output_dir is not None:
        report.save(output_dir)
    return report


__all__ = ["DEFAULT_DATASETS", "DEFAULT_MODELS", "BenchmarkReport", "run_benchmark"]
