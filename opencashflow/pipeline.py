"""End-to-end cash-flow underwriting pipeline.

The pipeline composes the four layers of OpenCashFlow:

``raw transactions`` -> ``categorisation`` -> ``151 cash-flow features`` -> ``PD model`` ->
``rating, policy decision and reason codes``.

Example:
    >>> from opencashflow.data.synthetic import generate_portfolio
    >>> from opencashflow.pipeline import CashFlowPipeline
    >>> portfolio = generate_portfolio(n_companies=80, months=18, seed=7)
    >>> pipeline = CashFlowPipeline(scorer="lightgbm").fit(
    ...     portfolio.transactions, portfolio.labels
    ... )
    >>> result = pipeline.predict(portfolio.transactions)
    >>> {"pd", "rating", "decision"}.issubset(result.columns)
    True

For a no-training demo, use ``scorer="heuristic"``. For LimiX-2, use ``scorer="limix"`` and set
``LIMIX_MODEL_PATH`` and ``LIMIX_CONFIG_DIR`` as described in :mod:`opencashflow.models.limix_scorer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from opencashflow.features.cashflow_features import CashFlowFeatureExtractor, FeatureConfig
from opencashflow.features.categorizer import TransactionCategorizer
from opencashflow.models.baselines import HeuristicScorer, make_baseline
from opencashflow.models.explain import Explanation, explain_model, explanations_to_frame
from opencashflow.models.limix_scorer import LimiXScorer
from opencashflow.models.metrics import expected_loss, rating_from_pd

Decision = Literal["accept", "review", "decline", "refer"]


@dataclass(frozen=True)
class UnderwritingPolicy:
    """Illustrative credit policy thresholds.

    These defaults are for demonstration only. A real policy must be calibrated to the lender's
    risk appetite, pricing, capital and fairness constraints.

    Args:
        accept_max_pd: Maximum PD for automatic approval.
        review_max_pd: Maximum PD for manual review. Above this, the application is declined.
        min_months_history: Minimum months of transaction history; below this the case is referred.
        min_transactions: Minimum transaction count; below this the case is referred.
    """

    accept_max_pd: float = 0.05
    review_max_pd: float = 0.20
    min_months_history: int = 3
    min_transactions: int = 20

    def __post_init__(self) -> None:
        if not 0 <= self.accept_max_pd <= 1:
            raise ValueError("accept_max_pd must be between 0 and 1.")
        if not 0 <= self.review_max_pd <= 1:
            raise ValueError("review_max_pd must be between 0 and 1.")
        if self.accept_max_pd > self.review_max_pd:
            raise ValueError("accept_max_pd cannot exceed review_max_pd.")

    def decide(
        self, probability: float, months_history: float, transaction_count: float
    ) -> Decision:
        """Map a PD and data-sufficiency checks to a decision."""
        if months_history < self.min_months_history or transaction_count < self.min_transactions:
            return "refer"
        if probability <= self.accept_max_pd:
            return "accept"
        if probability <= self.review_max_pd:
            return "review"
        return "decline"


def _coerce_labels(labels: Any, label_col: str) -> pd.Series:
    """Convert a Series, DataFrame or dict into a label Series indexed by entity_id."""
    if isinstance(labels, pd.Series):
        series = labels.copy()
    elif isinstance(labels, pd.DataFrame):
        if label_col in labels.columns:
            if "entity_id" in labels.columns:
                series = labels.set_index("entity_id")[label_col]
            else:
                series = labels[label_col]
        elif labels.shape[1] == 1:
            series = labels.iloc[:, 0]
        else:
            raise ValueError(
                f"labels DataFrame must contain {label_col!r} or exactly one column. "
                f"Found: {list(labels.columns)}"
            )
    elif isinstance(labels, dict):
        series = pd.Series(labels)
    else:
        raise TypeError("labels must be a pandas Series, DataFrame or dict.")
    series.index = series.index.astype(str)
    return pd.to_numeric(series, errors="coerce")


class CashFlowPipeline:
    """Raw transactions to credit decision.

    Args:
        config: Feature configuration.
        scorer: ``"limix"``, ``"lightgbm"``, ``"xgboost"``, ``"logistic"``, ``"heuristic"`` or a
            fitted/unfitted estimator exposing ``fit`` and ``predict_pd`` (or ``predict_proba``).
        categorizer: Optional custom transaction categoriser.
        policy: Underwriting policy. Defaults to illustrative thresholds.
        random_state: Seed forwarded to baselines.
        explainer: Reserved for a custom explainer; when ``None`` the best available explainer is
            selected automatically.

    Attributes:
        extractor: The fitted :class:`CashFlowFeatureExtractor`.
        scorer_: The fitted scoring model.
        feature_names_in_: Feature columns used by the model.
        training_summary_: Small dictionary describing the fitted portfolio.
    """

    def __init__(
        self,
        *,
        config: FeatureConfig | None = None,
        scorer: str | Any = "lightgbm",
        categorizer: TransactionCategorizer | None = None,
        policy: UnderwritingPolicy | None = None,
        random_state: int = 42,
        explainer: Any = None,
    ) -> None:
        self.config = config or FeatureConfig()
        self.scorer = scorer
        self.policy = policy or UnderwritingPolicy()
        self.random_state = random_state
        self.explainer = explainer
        self.extractor: CashFlowFeatureExtractor = CashFlowFeatureExtractor(
            config=self.config, categorizer=categorizer
        )
        self.scorer_: Any = None
        self.feature_names_in_: list[str] = []
        self.training_summary_: dict[str, Any] = {}
        self._fitted = False

    # ------------------------------------------------------------------ construction helpers
    def _build_scorer(self) -> Any:
        if isinstance(self.scorer, str):
            key = self.scorer.lower().strip()
            if key == "limix":
                return LimiXScorer(seed=self.random_state)
            if key in {"heuristic", "scorecard", "rules"}:
                return HeuristicScorer()
            return make_baseline(key, random_state=self.random_state)
        return self.scorer

    @property
    def is_limix(self) -> bool:
        """Whether the active scorer is a :class:`LimiXScorer`."""
        return isinstance(self.scorer_, LimiXScorer) or (
            isinstance(self.scorer, str) and self.scorer.lower().strip() == "limix"
        )

    # ------------------------------------------------------------------ fit / transform
    def fit(
        self,
        transactions: pd.DataFrame,
        labels: Any = None,
        *,
        label_col: str = "default_flag",
    ) -> CashFlowPipeline:
        """Fit the categoriser, feature extractor and scorer.

        Args:
            transactions: Raw or canonical transaction table.
            labels: Binary default labels. Accepted forms: Series indexed by ``entity_id``, a
                DataFrame with ``entity_id`` and ``label_col``, a one-column DataFrame, or a dict.
                May be ``None`` only for the heuristic scorecard.
            label_col: Label column name.

        Returns:
            ``self``.
        """
        features = self.extractor.fit_transform(transactions)
        self.feature_names_in_ = list(features.columns)
        scorer = self._build_scorer()

        if labels is None and label_col in transactions.columns:
            labels = (
                transactions.assign(entity_id=transactions["entity_id"].astype(str))
                .groupby("entity_id", observed=True)[label_col]
                .first()
            )
        if labels is None:
            if isinstance(scorer, HeuristicScorer):
                scorer.fit(features)
                self.scorer_ = scorer
                self._fitted = True
                self.training_summary_ = {
                    "n_entities": len(features),
                    "n_features": int(features.shape[1]),
                    "scorer": "heuristic",
                    "supervised": False,
                }
                return self
            raise ValueError(
                "labels are required for a supervised scorer. Pass a Series, a DataFrame with "
                "'entity_id' and 'default_flag', or a dict; or use scorer='heuristic'."
            )

        y = _coerce_labels(labels, label_col)
        joined = pd.concat([features, y.rename("__label__")], axis=1, join="inner")
        joined = joined.loc[joined["__label__"].notna()]
        if joined.empty:
            raise ValueError(
                "No entities matched between the feature matrix and labels. Check that the label "
                "index uses the same entity identifiers as the transaction table."
            )
        y_train = joined["__label__"].astype(int)
        if y_train.nunique() < 2:
            raise ValueError(
                "Training labels contain a single class. Provide defaults and non-defaults, or use "
                "the heuristic scorecard."
            )
        if len(y_train) < 10:
            raise ValueError(f"Need at least 10 labelled entities, got {len(y_train)}.")
        X_train = joined.drop(columns="__label__")
        scorer.fit(X_train, y_train)
        self.scorer_ = scorer
        self._fitted = True
        self.training_summary_ = {
            "n_entities": len(features),
            "n_labelled": len(y_train),
            "n_features": int(features.shape[1]),
            "default_rate": float(y_train.mean()),
            "scorer": type(scorer).__name__,
            "supervised": True,
        }
        return self

    def transform(
        self, transactions: pd.DataFrame, as_of: str | pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Compute the feature matrix for new transactions."""
        return self.extractor.transform(transactions, as_of=as_of)

    def fit_predict(
        self,
        transactions: pd.DataFrame,
        labels: Any = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fit the pipeline and score the same transactions."""
        return self.fit(transactions, labels).predict(transactions, **kwargs)

    # ------------------------------------------------------------------ scoring
    def predict(
        self,
        transactions: pd.DataFrame,
        *,
        as_of: str | pd.Timestamp | None = None,
        explain: bool = True,
        top_k: int = 3,
        explain_max_rows: int = 20,
        lgd: float = 0.45,
        ead: float | pd.Series | None = None,
        include_features: bool = False,
    ) -> pd.DataFrame:
        """Score a transaction table and return one row per borrower.

        Scoring the same borrowers that were passed to :meth:`fit` is an in-sample evaluation and
        will overstate performance, especially for in-context models such as LimiX. Use a
        borrower-level holdout or cross-validation for measurement.

        Args:
            transactions: Raw or canonical transactions.
            as_of: Optional point-in-time cut-off.
            explain: Add reason codes and explanation metadata.
            top_k: Number of reason codes per borrower.
            explain_max_rows: Maximum number of borrowers to explain, starting with the highest PD.
                Set to 0 to disable explanations even when ``explain=True``. Occlusion-based
                explanations for LimiX are expensive; the default is deliberately small.
            lgd: Loss given default used for expected loss.
            ead: Exposure at default, scalar or Series indexed by entity. When omitted, no expected
                loss column is produced.
            include_features: Append the engineered feature columns to the result.

        Returns:
            DataFrame indexed by ``entity_id`` with ``pd``, ``rating``, ``decision`` and, when
            requested, expected loss and reason codes.
        """
        if not self._fitted:
            if isinstance(self.scorer, str) and self.scorer.lower().strip() in {
                "heuristic",
                "scorecard",
                "rules",
            }:
                features = self.extractor.fit_transform(transactions)
                self.scorer_ = self._build_scorer().fit(features)
                self._fitted = True
                self.feature_names_in_ = list(features.columns)
            else:
                raise RuntimeError(
                    "The pipeline is not fitted. Call fit(transactions, labels) first."
                )
        features = self.extractor.transform(transactions, as_of=as_of)
        if features.empty:
            return pd.DataFrame(columns=["pd", "rating", "decision"])
        pd_values = self._predict_pd(features)
        months = features.get("months_observed", pd.Series(0.0, index=features.index))
        counts = features.get("txn_count", pd.Series(0.0, index=features.index))
        decisions = [
            self.policy.decide(float(p), float(m), float(c))
            for p, m, c in zip(pd_values, months, counts)
        ]
        result = pd.DataFrame(
            {
                "pd": pd_values,
                "rating": rating_from_pd(pd_values),
                "decision": decisions,
            },
            index=features.index,
        )
        result.index.name = "entity_id"
        if ead is not None:
            ead_values = (
                pd.Series(ead, index=features.index)
                if np.isscalar(ead)
                else pd.Series(ead).reindex(features.index)
            )
            result["expected_loss"] = expected_loss(pd_values, lgd=lgd, ead=ead_values.to_numpy())
        if include_features:
            result = result.join(features)
        if explain and explain_max_rows > 0:
            explanations = self._explain(features, pd_values, explain_max_rows)
            if explanations:
                reason_frame = explanations_to_frame(explanations, k=top_k).set_index("entity_id")
                reason_frame.index.name = "entity_id"
                reason_frame = reason_frame.reindex(result.index)
                for column in [f"reason_{i}" for i in range(1, top_k + 1)]:
                    if column in reason_frame.columns:
                        result[column] = reason_frame[column]
                if "method" in reason_frame.columns:
                    result["explanation_method"] = reason_frame["method"]
        return result.sort_values("pd", ascending=False)

    def score(self, transactions: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Alias for :meth:`predict`."""
        return self.predict(transactions, **kwargs)

    def explain(
        self,
        transactions: pd.DataFrame,
        *,
        as_of: str | pd.Timestamp | None = None,
        max_rows: int | None = None,
        max_features: int = 30,
    ) -> list[Explanation]:
        """Explain individual predictions, ordered by descending PD."""
        if not self._fitted:
            raise RuntimeError("The pipeline is not fitted.")
        features = self.extractor.transform(transactions, as_of=as_of)
        pd_values = self._predict_pd(features)
        order = np.argsort(-pd_values)
        if max_rows is not None:
            order = order[:max_rows]
        selected = features.iloc[order]
        return explain_model(self.scorer_, selected, max_features=max_features)

    def _predict_pd(self, features: pd.DataFrame) -> np.ndarray:
        if hasattr(self.scorer_, "predict_pd"):
            return np.asarray(self.scorer_.predict_pd(features), dtype=float).reshape(-1)
        if hasattr(self.scorer_, "predict_proba"):
            proba = np.asarray(self.scorer_.predict_proba(features), dtype=float)
            if proba.ndim != 2 or proba.shape[1] != 2:
                raise ValueError("predict_proba must return two columns for PD scoring.")
            return proba[:, 1]
        raise TypeError("The scorer must expose predict_pd() or predict_proba().")

    def _explain(
        self, features: pd.DataFrame, pd_values: np.ndarray, max_rows: int
    ) -> list[Explanation]:
        # Occlusion for LimiX scales with rows x features; keep the default demo bounded.
        if self.is_limix:
            max_rows = min(max_rows, 10)
        if len(features) > max_rows:
            order = np.argsort(-pd_values)[:max_rows]
            selected = features.iloc[order]
        else:
            selected = features
        return explain_model(self.scorer_, selected, max_features=30)

    # ------------------------------------------------------------------ reporting / persistence
    def portfolio_summary(self, scored: pd.DataFrame) -> dict[str, Any]:
        """Aggregate a scored portfolio into decision counts, mean PD and expected loss."""
        summary: dict[str, Any] = {
            "n_entities": len(scored),
            "mean_pd": float(scored["pd"].mean()) if len(scored) else float("nan"),
            "approval_rate": float((scored["decision"] == "accept").mean())
            if len(scored)
            else float("nan"),
        }
        if "decision" in scored.columns:
            summary["decisions"] = scored["decision"].value_counts().to_dict()
        if "expected_loss" in scored.columns:
            summary["expected_loss_total"] = float(scored["expected_loss"].sum())
        return summary

    def feature_importance(self) -> pd.Series:
        """Native feature importance when the scorer exposes one."""
        if self.scorer_ is None:
            raise RuntimeError("The pipeline is not fitted.")
        if hasattr(self.scorer_, "feature_importances_"):
            return self.scorer_.feature_importances_
        if hasattr(self.scorer_, "coefficients_"):
            return self.scorer_.coefficients_
        raise ValueError(
            "This scorer has no native feature importance. Use "
            "opencashflow.models.explain.permutation_importance with a labelled set."
        )

    def save(self, path: str | Path) -> Path:
        """Persist the fitted pipeline with joblib.

        LimiX predictors are not serialised; the checkpoint path and configuration are stored and
        the predictor is reloaded lazily on the next prediction.
        """
        import joblib

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> CashFlowPipeline:
        """Load a pipeline saved with :meth:`save`."""
        import joblib

        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected a CashFlowPipeline, got {type(loaded).__name__}.")
        return loaded

    def summary(self) -> dict[str, Any]:
        """Return a compact description of the fitted pipeline."""
        return {
            "scorer": type(self.scorer_).__name__ if self.scorer_ is not None else None,
            "feature_config": {
                "window_days": self.config.window_days,
                "as_of": str(self.config.as_of) if self.config.as_of is not None else None,
                "include_other_inflow_as_revenue": self.config.include_other_inflow_as_revenue,
            },
            "policy": {
                "accept_max_pd": self.policy.accept_max_pd,
                "review_max_pd": self.policy.review_max_pd,
                "min_months_history": self.policy.min_months_history,
                "min_transactions": self.policy.min_transactions,
            },
            "training": self.training_summary_,
        }


__all__ = ["CashFlowPipeline", "Decision", "UnderwritingPolicy"]
