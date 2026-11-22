"""Benchmark suite: LimiX-2 versus tree and linear baselines.

The benchmark uses repeated stratified K-fold cross-validation or a single out-of-time split and
reports ROC-AUC, KS, PR-AUC, Brier score and log loss. It is deliberately conservative:

* every model sees the same folds;
* preprocessing is fitted inside each fold;
* LimiX receives only the training fold as its in-context reference set;
* no test-fold information is used for feature selection or calibration.

The ROC-AUC confidence interval uses a t quantile over folds times repeats. Folds share training
rows, so the interval understates uncertainty and is an approximation.

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
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
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


def _format_auc_ci(mean: float, low: float, high: float) -> str:
    """Format mean ROC-AUC and interval as ``0.928 [0.905, 0.951]``."""
    point = f"{mean:.3f}" if np.isfinite(mean) else "n/a"
    if np.isfinite(low) and np.isfinite(high):
        return f"{point} [{low:.3f}, {high:.3f}]"
    return f"{point} [n/a, n/a]"


@dataclass
class BenchmarkReport:
    """Long-form fold results and an aggregated summary table."""

    results: pd.DataFrame
    summary: pd.DataFrame
    output_dir: Path | None = None

    def to_markdown(self) -> str:
        """Render the summary as a Markdown table with ROC-AUC confidence intervals."""
        if self.summary.empty:
            return "_No results._"
        display = self.summary.copy()
        low = display.get("roc_auc_ci_low", pd.Series(np.nan, index=display.index))
        high = display.get("roc_auc_ci_high", pd.Series(np.nan, index=display.index))
        display["roc_auc_display"] = [
            _format_auc_ci(mean, lo, hi) for mean, lo, hi in zip(display["roc_auc"], low, high)
        ]
        columns = [
            "dataset",
            "model",
            "variant",
            "split",
            "n",
            "n_features",
            "default_rate",
            "roc_auc_display",
            "roc_auc_std",
            "ks",
            "pr_auc",
            "brier",
            "log_loss",
            "fit_seconds",
            "predict_seconds",
            "n_runs",
        ]
        available = [c for c in columns if c in display.columns]
        return _to_markdown_table(display[available].rename(columns={"roc_auc_display": "roc_auc"}))

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
    sba_approval_fy_max: int | None = None,
) -> DatasetBundle:
    key = name.lower().strip()
    if key == "synthetic":
        bundle = load_synthetic(
            n_companies=synthetic_companies,
            months=synthetic_months,
            seed=seed,
        )
        assert isinstance(bundle, DatasetBundle)
    elif key == "sba":
        bundle = load_dataset(key, max_rows=max_rows, approval_fy_max=sba_approval_fy_max)
    else:
        bundle = load_dataset(key)
    if max_rows is not None and key != "sba" and len(bundle.X) > max_rows:
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


def _summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate long-form fold results with t-based ROC-AUC confidence intervals.

    Args:
        results: Long-form fold results, one row per model fit on one split.

    Returns:
        Summary table with one row per dataset, model, variant and split. The ROC-AUC interval
        columns are NaN when a group has fewer than two runs.
    """
    group_keys = ["dataset", "model", "variant", "split"]
    summary = results.groupby(group_keys, as_index=False).agg(
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
        n_runs=("roc_auc", "size"),
    )
    n_runs = summary["n_runs"].to_numpy(dtype=float)
    auc_mean = summary["roc_auc"].to_numpy(dtype=float)
    auc_std = summary["roc_auc_std"].to_numpy(dtype=float)
    half_width = np.full(len(summary), np.nan)
    valid = n_runs >= 2
    if valid.any():
        t_critical = stats.t.ppf(0.975, n_runs[valid] - 1)
        half_width[valid] = t_critical * auc_std[valid] / np.sqrt(n_runs[valid])
    summary["roc_auc_ci_low"] = auc_mean - half_width
    summary["roc_auc_ci_high"] = auc_mean + half_width
    summary = summary[
        [
            *group_keys,
            "n",
            "n_features",
            "default_rate",
            "roc_auc",
            "roc_auc_std",
            "roc_auc_ci_low",
            "roc_auc_ci_high",
            "ks",
            "pr_auc",
            "brier",
            "log_loss",
            "fit_seconds",
            "predict_seconds",
            "n_runs",
        ]
    ]
    summary = summary.sort_values(
        [*group_keys, "roc_auc"], ascending=[True, True, True, True, False]
    ).reset_index(drop=True)
    return summary


def run_benchmark(
    *,
    datasets: Iterable[str] = DEFAULT_DATASETS,
    models: Iterable[str] = DEFAULT_MODELS,
    cv: int = 3,
    seed: int = 42,
    max_rows: int | None = None,
    sba_approval_fy_max: int | None = None,
    synthetic_companies: int = 300,
    synthetic_months: int = 20,
    limix_pipelines: int = 1,
    repeats: int = 1,
    split: Literal["stratified", "oot"] = "stratified",
    time_column: str | None = None,
    oot_cutoff: float | None = None,
    exclude_features: tuple[str, ...] = (),
    variant: str = "all",
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> BenchmarkReport:
    """Run cross-validated benchmarks.

    Args:
        datasets: Dataset names from :data:`opencashflow.data.loaders.DATASET_REGISTRY`.
        models: ``heuristic``, ``logistic``, ``lightgbm``, ``xgboost`` and/or ``limix``.
        cv: Number of stratified folds.
        seed: Random seed for folds, models and synthetic data. Repeat ``i`` reshuffles folds with
            seed + i - 1.
        max_rows: Optional stratified row cap per dataset, useful for quick runs.
        sba_approval_fy_max: SBA only. Keep loans approved in or before this fiscal year,
            which restricts the sample to vintages with time to mature. None keeps all years.
        synthetic_companies: Portfolio size for the synthetic dataset.
        synthetic_months: Months of history for the synthetic dataset.
        limix_pipelines: Number of LimiX ensemble members. One is fast on CPU.
        repeats: Number of stratified CV repeats. Ignored for out-of-time splits, which run once.
        split: ``"stratified"`` for repeated K-fold, ``"oot"`` for one out-of-time split.
        time_column: Numeric column of X used for ``split="oot"``, for example ``"approval_fy"``.
            It is required for out-of-time splits and stays in the model features unless excluded.
        oot_cutoff: For ``split="oot"``, rows with ``time_column <= cutoff`` train and the rest are
            tested. Both sides must contain both classes.
        exclude_features: Columns dropped from X before modelling. Unknown names are ignored.
        variant: Label written to results, used to compare feature sets on one dataset.
        output_dir: When set, the report is written to this directory.
        verbose: Print progress.

    Returns:
        A :class:`BenchmarkReport` with long-form fold results and an aggregated summary.

    Notes:
        The ROC-AUC confidence interval is ``mean +/- t(0.975, n_runs - 1) * sd / sqrt(n_runs)``
        and is NaN when ``n_runs < 2``. Folds share training rows, so the interval understates
        uncertainty and is an approximation.
    """
    if cv < 2:
        raise ValueError("cv must be at least 2.")
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if split not in {"stratified", "oot"}:
        raise ValueError("split must be 'stratified' or 'oot'.")
    if split == "oot":
        if time_column is None:
            raise ValueError("time_column is required for split='oot'.")
        if oot_cutoff is None:
            raise ValueError("oot_cutoff is required for split='oot'.")
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
            sba_approval_fy_max=sba_approval_fy_max,
        )
        y = bundle.y
        X_full = bundle.X
        if split == "oot":
            assert time_column is not None and oot_cutoff is not None
            if time_column not in X_full.columns:
                raise ValueError(
                    f"time_column {time_column!r} is not a feature of {dataset_name!r}."
                )
            if not pd.api.types.is_numeric_dtype(X_full[time_column]):
                raise ValueError(f"time_column {time_column!r} must be numeric.")
            cutoff = float(oot_cutoff)
            in_train = (X_full[time_column] <= cutoff).to_numpy()
            train_idx = np.flatnonzero(in_train)
            test_idx = np.flatnonzero(~in_train)
            if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].nunique() < 2:
                raise ValueError(
                    "Both sides of an out-of-time split must contain both classes; "
                    f"train classes={sorted(y.iloc[train_idx].unique())}, "
                    f"test classes={sorted(y.iloc[test_idx].unique())}."
                )
            splits: list[tuple[int, int, np.ndarray, np.ndarray]] = [(1, 1, train_idx, test_idx)]
            time_cutoff: float = cutoff
        else:
            splits = []
            for repeat in range(1, repeats + 1):
                splitter = StratifiedKFold(
                    n_splits=cv, shuffle=True, random_state=seed + repeat - 1
                )
                for fold, (train_idx, test_idx) in enumerate(splitter.split(X_full, y), start=1):
                    splits.append((repeat, fold, np.asarray(train_idx), np.asarray(test_idx)))
            time_cutoff = float("nan")

        drop_columns = [c for c in exclude_features if c in X_full.columns]
        X = X_full.drop(columns=drop_columns) if drop_columns else X_full

        for model_name in selected_models:
            if model_name == "heuristic" and dataset_name != "synthetic":
                if verbose:
                    print("  [skip] heuristic: needs cash-flow features")
                continue
            if verbose:
                print(f"  [model] {model_name}")
            shared_limix: LimiXScorer | None = None
            try:
                for repeat, fold, train_idx, test_idx in splits:
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
                            "variant": variant,
                            "split": split,
                            "repeat": repeat,
                            "fold": fold,
                            "n": len(y),
                            "n_train": len(y_train),
                            "n_test": len(y_test),
                            "n_features": X.shape[1],
                            "default_rate": float(y.mean()),
                            "fit_seconds": fit_seconds,
                            "predict_seconds": predict_seconds,
                            "time_cutoff": time_cutoff,
                            **{metric: metrics[metric] for metric in METRIC_COLUMNS},
                        }
                    )
            finally:
                if shared_limix is not None:
                    shared_limix.close()

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("Benchmark produced no results. Check dataset and model names.")
    summary = _summarize_results(results)
    report = BenchmarkReport(results=results, summary=summary)
    if output_dir is not None:
        report.save(output_dir)
    return report


__all__ = ["DEFAULT_DATASETS", "DEFAULT_MODELS", "BenchmarkReport", "run_benchmark"]
