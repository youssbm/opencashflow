"""Baseline probability-of-default models.

Three baselines are provided:

* :class:`LogisticBaseline` - regularised logistic regression with median imputation, a numeric
  transform (quantile by default) and one-hot encoding. It is the transparent reference model.
* :class:`TreeBaseline` - LightGBM or XGBoost with native missing-value handling.
* :class:`HeuristicScorer` - a fixed, explainable cash-flow scorecard that needs no training. It is
  useful for demos, sanity checks and as a fallback when labels are unavailable.

All models expose ``fit``, ``predict_proba``, ``predict`` and ``predict_pd``. The positive class is
the second class in sorted order, which is ``1`` for the standard ``0/1`` default encoding.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from opencashflow.models.metrics import expected_loss


def _is_categorical(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
    )


class _BinaryModelMixin:
    """Shared binary classification behaviour."""

    classes_: np.ndarray
    feature_names_in_: list[str]
    n_features_in_: int

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_pd(self, X: Any) -> np.ndarray:
        """Probability of the positive (second) class, i.e. probability of default."""
        proba = self.predict_proba(X)
        if proba.shape[1] != 2:
            raise ValueError("predict_pd requires a binary classifier.")
        return proba[:, 1]

    def predict_expected_loss(self, X: Any, *, lgd: float = 0.45, ead: float = 1.0) -> np.ndarray:
        """Expected loss per observation using the model PD."""
        return np.asarray(expected_loss(self.predict_pd(X), lgd=lgd, ead=ead), dtype=float)

    def _store_classes(self, y: Any) -> np.ndarray:
        y_array = np.asarray(y).reshape(-1)
        classes = np.unique(y_array)
        if classes.size != 2:
            raise ValueError(
                f"Binary classification needs exactly two classes, got {classes.tolist()}."
            )
        self.classes_ = classes
        return (y_array == classes[1]).astype(int)


class TreeBaseline(_BinaryModelMixin):
    """LightGBM or XGBoost gradient-boosted baseline.

    Predicted probabilities are not reweighted by default (``class_weight=None``), so they are
    directly interpretable as probabilities of default. Pass ``class_weight="balanced"`` to
    reweight the training loss; that shifts the outputs away from the observed base rate.

    Args:
        backend: ``"lightgbm"`` or ``"xgboost"``.
        n_estimators: Number of boosting rounds.
        learning_rate: Shrinkage.
        max_depth: Maximum tree depth. ``-1`` means no limit for LightGBM.
        num_leaves: LightGBM leaf count.
        subsample: Row subsampling fraction.
        colsample_bytree: Feature subsampling fraction.
        reg_lambda: L2 regularisation.
        class_weight: ``"balanced"`` or ``None``.
        random_state: Seed.
        n_jobs: Parallel threads, ``-1`` for all.
        **params: Extra parameters passed to the underlying estimator.
    """

    def __init__(
        self,
        backend: Literal["lightgbm", "xgboost"] = "lightgbm",
        *,
        n_estimators: int = 400,
        learning_rate: float = 0.03,
        max_depth: int = -1,
        num_leaves: int = 31,
        subsample: float = 0.9,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        class_weight: str | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
        **params: Any,
    ) -> None:
        if backend not in {"lightgbm", "xgboost"}:
            raise ValueError("backend must be 'lightgbm' or 'xgboost'.")
        self.backend = backend
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.params = params
        self.model_: Any = None
        self.categories_: dict[str, pd.Index] = {}
        self.categorical_features_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any) -> TreeBaseline:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X))
        X = X.copy()
        X.columns = [str(c) for c in X.columns]
        y_bin = self._store_classes(y)
        if np.bincount(y_bin).min() == 0:
            raise ValueError("Both classes must be present in y.")
        Xp = self._fit_preprocessor(X)
        self.model_ = self._build_model(y_bin)
        if self.backend == "lightgbm":
            self.model_.fit(
                Xp,
                y_bin,
                categorical_feature=self.categorical_features_ or "auto",
            )
        else:
            self.model_.fit(Xp, y_bin)
        return self

    def _build_model(self, y_bin: np.ndarray) -> Any:
        if self.backend == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "LightGBM is not installed. Run: pip install 'opencashflow[trees]'"
                ) from exc
            return LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_lambda=self.reg_lambda,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1,
                **self.params,
            )
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "XGBoost is not installed. Run: pip install 'opencashflow[trees]'"
            ) from exc
        positives = max(int(y_bin.sum()), 1)
        negatives = max(int(len(y_bin) - positives), 1)
        scale = negatives / positives if self.class_weight == "balanced" else 1.0
        return XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=None if self.max_depth < 0 else self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=scale,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method="hist",
            eval_metric="logloss",
            **self.params,
        )

    def _fit_preprocessor(self, X: pd.DataFrame) -> pd.DataFrame:
        self.feature_names_in_ = [str(c) for c in X.columns]
        self.n_features_in_ = len(self.feature_names_in_)
        self.categorical_features_ = [c for c in X.columns if _is_categorical(X[c])]
        self.categories_ = {}
        data: dict[str, Any] = {}
        for column in X.columns:
            series = X[column]
            if column in self.categorical_features_:
                categories = pd.Index(sorted(series.dropna().astype(str).unique()))
                self.categories_[column] = categories
                data[column] = pd.Categorical(
                    series.astype(str), categories=categories
                ).codes.astype(float)
            else:
                data[column] = pd.to_numeric(series, errors="coerce")
        return pd.DataFrame(data, index=X.index)

    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted categorical encoding and numeric coercion to new data."""
        if self.model_ is None:
            raise RuntimeError("The model is not fitted.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X), columns=self.feature_names_in_)
        X = X.copy()
        X.columns = [str(c) for c in X.columns]
        data: dict[str, Any] = {}
        for column in self.feature_names_in_:
            if column not in X.columns:
                data[column] = np.nan
                continue
            series = X[column]
            if column in self.categorical_features_:
                categories = self.categories_[column]
                data[column] = pd.Categorical(
                    series.astype(str), categories=categories
                ).codes.astype(float)
            else:
                data[column] = pd.to_numeric(series, errors="coerce")
        return pd.DataFrame(data, index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("The model is not fitted.")
        Xp = self.prepare_features(X)
        proba = np.asarray(self.model_.predict_proba(Xp), dtype=float)
        return proba

    @property
    def feature_importances_(self) -> pd.Series:
        """Normalised gain-based feature importance."""
        if self.model_ is None:
            raise RuntimeError("The model is not fitted.")
        if self.backend == "lightgbm":
            values = np.asarray(
                self.model_.booster_.feature_importance(importance_type="gain"), dtype=float
            )
        else:
            values = np.asarray(self.model_.feature_importances_, dtype=float)
        total = values.sum()
        if total > 0:
            values = values / total
        return pd.Series(values, index=self.feature_names_in_, name="importance").sort_values(
            ascending=False
        )


class LogisticBaseline(_BinaryModelMixin):
    """Regularised logistic regression with imputation, a numeric transform and one-hot encoding.

    Numeric columns are median-imputed, then transformed with a quantile transformer by default
    (``numeric_transform="quantile"``) or a standard scaler (``numeric_transform="standard"``).
    Predicted probabilities are not reweighted by default (``class_weight=None``), so they are
    directly interpretable as probabilities of default. Pass ``class_weight="balanced"`` to reweight
    the training loss; that shifts the outputs away from the observed base rate.

    Args:
        C: Inverse regularisation strength.
        class_weight: Class weights passed to scikit-learn. ``None`` keeps probabilities calibrated
            to the base rate; ``"balanced"`` reweights them.
        max_iter: Maximum solver iterations.
        random_state: Seed.
        max_categories: Maximum number of one-hot categories per categorical column.
        numeric_transform: ``"quantile"`` for a normal quantile transform, ``"standard"`` for a
            standard scaler. The transform is fitted inside the pipeline, hence within each fold.
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        class_weight: str | dict[Any, float] | None = None,
        max_iter: int = 3_000,
        random_state: int = 42,
        max_categories: int = 30,
        numeric_transform: Literal["quantile", "standard"] = "quantile",
    ) -> None:
        if numeric_transform not in {"quantile", "standard"}:
            raise ValueError("numeric_transform must be 'quantile' or 'standard'.")
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.max_categories = max_categories
        self.numeric_transform = numeric_transform
        self.model_: Any = None
        self.feature_names_in_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any) -> LogisticBaseline:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X))
        X = X.copy()
        X.columns = [str(c) for c in X.columns]
        for column in X.columns:
            if _is_categorical(X[column]):
                X[column] = X[column].astype(object)
        y_bin = self._store_classes(y)
        self.feature_names_in_ = [str(c) for c in X.columns]
        self.n_features_in_ = len(self.feature_names_in_)
        categorical = [c for c in X.columns if _is_categorical(X[c])]
        numeric = [c for c in X.columns if c not in categorical]
        numeric_steps: list[tuple[str, Any]] = [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
        ]
        if self.numeric_transform == "quantile":
            n_quantiles = max(10, min(1000, len(X)))
            numeric_steps.append(
                (
                    "scale",
                    QuantileTransformer(output_distribution="normal", n_quantiles=n_quantiles),
                )
            )
        else:
            numeric_steps.append(("scale", StandardScaler()))
        numeric_pipe = Pipeline(numeric_steps)
        categorical_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                (
                    "encode",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        min_frequency=0.01,
                        max_categories=self.max_categories,
                    ),
                ),
            ]
        )
        preprocessor = ColumnTransformer(
            [
                *([("num", numeric_pipe, numeric)] if numeric else []),
                *([("cat", categorical_pipe, categorical)] if categorical else []),
            ],
            remainder="drop",
        )
        self.model_ = Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        C=self.C,
                        class_weight=self.class_weight,
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.model_.fit(X, y_bin)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("The model is not fitted.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X), columns=self.feature_names_in_)
        X = X.copy()
        X.columns = [str(c) for c in X.columns]
        for column in X.columns:
            if _is_categorical(X[column]):
                X[column] = X[column].astype(object)
        return np.asarray(self.model_.predict_proba(X), dtype=float)

    @property
    def coefficients_(self) -> pd.Series:
        """Signed coefficients on the encoded feature space, sorted by absolute value."""
        if self.model_ is None:
            raise RuntimeError("The model is not fitted.")
        names = self.model_.named_steps["preprocess"].get_feature_names_out()
        coefs = self.model_.named_steps["model"].coef_.reshape(-1)
        return pd.Series(coefs, index=names, name="coefficient").sort_values(
            key=np.abs, ascending=False
        )


#: Weights for the untrained cash-flow scorecard. Positive values increase PD; negative values
#: reduce it. The weights are illustrative and intentionally conservative.
HEURISTIC_WEIGHTS: dict[str, float] = {
    "revenue_cv": 1.8,
    "net_cash_flow_negative_ratio": 2.0,
    "overdraft_ratio": 1.5,
    "debt_service_ratio": 2.2,
    "nsf_rate": 8.0,
    "gambling_to_expense": 2.0,
    "revenue_growth_3m": -0.8,
    "cash_buffer_months": -0.06,
    "runway_months": -0.04,
    "cashflow_health_score": -0.035,
}

HEURISTIC_CLIPS: dict[str, tuple[float, float]] = {
    "revenue_cv": (0.0, 2.0),
    "net_cash_flow_negative_ratio": (0.0, 1.0),
    "overdraft_ratio": (0.0, 1.0),
    "debt_service_ratio": (0.0, 1.0),
    "nsf_rate": (0.0, 0.2),
    "gambling_to_expense": (0.0, 0.2),
    "revenue_growth_3m": (-1.0, 1.0),
    "cash_buffer_months": (-6.0, 12.0),
    "runway_months": (-6.0, 24.0),
    "cashflow_health_score": (0.0, 100.0),
}


class HeuristicScorer(_BinaryModelMixin):
    """A fixed, explainable cash-flow scorecard that requires no training.

    The score is a logistic function of a small set of cash-flow features with hand-set weights.
    It is not a calibrated credit model; use it for demos, baseline comparisons and as a sensible
    prior when no labelled defaults are available.
    """

    def __init__(self, intercept: float = -2.1, weights: dict[str, float] | None = None) -> None:
        self.intercept = intercept
        self.weights = dict(HEURISTIC_WEIGHTS if weights is None else weights)
        self.classes_ = np.array([0, 1])
        self.feature_names_in_: list[str] = list(self.weights)

    def fit(self, X: pd.DataFrame, y: Any = None) -> HeuristicScorer:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = [c for c in self.weights if c in X.columns]
        self.n_features_in_ = len(self.feature_names_in_)
        if y is not None:
            self._store_classes(y)
        return self

    def _logit(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X), columns=self.feature_names_in_)
        z = np.full(len(X), self.intercept, dtype=float)
        for feature, weight in self.weights.items():
            if feature not in X.columns:
                continue
            values = pd.to_numeric(X[feature], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            low, high = HEURISTIC_CLIPS.get(feature, (-np.inf, np.inf))
            z += weight * np.clip(values, low, high)
        return z

    def feature_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature additive log-odds contributions."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X), columns=self.feature_names_in_)
        contributions = {}
        for feature, weight in self.weights.items():
            if feature not in X.columns:
                values = np.zeros(len(X), dtype=float)
            else:
                values = (
                    pd.to_numeric(X[feature], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                )
            low, high = HEURISTIC_CLIPS.get(feature, (-np.inf, np.inf))
            contributions[feature] = weight * np.clip(values, low, high)
        return pd.DataFrame(contributions, index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pd_values = 1.0 / (1.0 + np.exp(-self._logit(X)))
        return np.column_stack([1.0 - pd_values, pd_values])


def available_backends() -> dict[str, bool]:
    """Report which optional baseline backends are importable."""
    result = {"logistic": True, "heuristic": True, "lightgbm": False, "xgboost": False}
    try:
        import lightgbm  # noqa: F401

        result["lightgbm"] = True
    except ImportError:
        pass
    try:
        import xgboost  # noqa: F401

        result["xgboost"] = True
    except ImportError:
        pass
    return result


def make_baseline(name: str, **kwargs: Any) -> Any:
    """Factory for baseline models.

    Accepted names: ``"lightgbm"``, ``"lgbm"``, ``"xgboost"``, ``"xgb"``, ``"logistic"``, ``"heuristic"``.
    """
    key = name.lower().strip()
    if key in {"lightgbm", "lgbm"}:
        return TreeBaseline("lightgbm", **kwargs)
    if key in {"xgboost", "xgb"}:
        return TreeBaseline("xgboost", **kwargs)
    if key in {"logistic", "logreg", "lr"}:
        return LogisticBaseline(**kwargs)
    if key in {"heuristic", "scorecard", "rules"}:
        kwargs.pop("random_state", None)
        kwargs.pop("n_estimators", None)
        return HeuristicScorer(**kwargs)
    raise ValueError(f"Unknown baseline {name!r}. Choose lightgbm, xgboost, logistic or heuristic.")


__all__ = [
    "HEURISTIC_WEIGHTS",
    "HeuristicScorer",
    "LogisticBaseline",
    "TreeBaseline",
    "available_backends",
    "make_baseline",
]
