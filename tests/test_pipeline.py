"""End-to-end pipeline, policy and persistence tests."""

from __future__ import annotations

import pandas as pd
import pytest

from opencashflow.data.synthetic import generate_company_transactions
from opencashflow.pipeline import CashFlowPipeline, UnderwritingPolicy


def test_policy_decisions():
    policy = UnderwritingPolicy(accept_max_pd=0.05, review_max_pd=0.20)
    assert policy.decide(0.01, 12, 100) == "accept"
    assert policy.decide(0.10, 12, 100) == "review"
    assert policy.decide(0.40, 12, 100) == "decline"
    assert policy.decide(0.01, 1, 100) == "refer"
    assert policy.decide(0.01, 12, 2) == "refer"
    with pytest.raises(ValueError, match="accept_max_pd"):
        UnderwritingPolicy(accept_max_pd=0.5, review_max_pd=0.2)


def test_heuristic_pipeline_end_to_end(portfolio):
    pipeline = CashFlowPipeline(scorer="heuristic")
    pipeline.fit(portfolio.transactions, portfolio.labels)
    scored = pipeline.predict(portfolio.transactions, top_k=2, explain_max_rows=10)
    assert {"pd", "rating", "decision", "reason_1", "reason_2"} <= set(scored.columns)
    assert len(scored) == 30
    assert scored["pd"].between(0, 1).all()
    assert set(scored["decision"]) <= {"accept", "review", "decline", "refer"}
    summary = pipeline.portfolio_summary(scored)
    assert summary["n_entities"] == 30
    assert 0 <= summary["mean_pd"] <= 1
    assert "decisions" in summary


def test_logistic_pipeline_and_persistence(portfolio, tmp_path):
    pipeline = CashFlowPipeline(scorer="logistic")
    pipeline.fit(portfolio.transactions, portfolio.labels)
    scored = pipeline.predict(
        portfolio.transactions, explain=False, ead=10_000, include_features=True
    )
    assert "expected_loss" in scored.columns
    assert (scored["expected_loss"] >= 0).all()
    assert "revenue_total" in scored.columns
    importance = pipeline.feature_importance()
    assert len(importance) > 0

    path = pipeline.save(tmp_path / "pipeline.joblib")
    loaded = CashFlowPipeline.load(path)
    entity_ids = scored.index[:2].tolist()
    subset = portfolio.transactions[portfolio.transactions["entity_id"].isin(entity_ids)]
    result = loaded.predict(subset, explain=False)
    assert len(result) == 2
    assert result["pd"].between(0, 1).all()


def test_fit_without_labels_heuristic(portfolio):
    pipeline = CashFlowPipeline(scorer="heuristic").fit(portfolio.transactions)
    first_entity = portfolio.transactions["entity_id"].iloc[0]
    subset = portfolio.transactions[portfolio.transactions["entity_id"] == first_entity]
    scored = pipeline.predict(subset, explain=False)
    assert len(scored) == 1
    assert 0 <= scored.iloc[0]["pd"] <= 1


def test_unfitted_supervised_pipeline_raises(portfolio):
    pipeline = CashFlowPipeline(scorer="logistic")
    with pytest.raises(RuntimeError, match="not fitted"):
        pipeline.predict(portfolio.transactions)


def test_single_class_labels_raise(portfolio):
    labels = portfolio.labels.copy()
    labels["default_flag"] = 0
    with pytest.raises(ValueError, match="single class"):
        CashFlowPipeline(scorer="heuristic").fit(portfolio.transactions, labels)


@pytest.mark.parametrize("form", ["series", "frame", "dict"])
def test_label_input_forms(portfolio, form):
    entity_ids = portfolio.labels["entity_id"].head(12).tolist()
    transactions = portfolio.transactions[portfolio.transactions["entity_id"].isin(entity_ids)]
    labels = portfolio.labels.set_index("entity_id")["default_flag"].loc[entity_ids]
    if form == "frame":
        labels_input = labels.rename_axis("entity_id").reset_index()
    elif form == "dict":
        labels_input = labels.to_dict()
    else:
        labels_input = labels
    pipeline = CashFlowPipeline(scorer="heuristic")
    pipeline.fit(transactions, labels_input)
    assert pipeline.training_summary_["n_labelled"] == 12


def test_short_history_is_referred():
    transactions = generate_company_transactions(months=1, seed=4)
    pipeline = CashFlowPipeline(scorer="heuristic", policy=UnderwritingPolicy(min_months_history=3))
    result = pipeline.fit(transactions).predict(transactions, explain=False)
    assert result.iloc[0]["decision"] == "refer"


def test_explain_returns_ordered_explanations(portfolio):
    pipeline = CashFlowPipeline(scorer="heuristic").fit(portfolio.transactions, portfolio.labels)
    explanations = pipeline.explain(portfolio.transactions, max_rows=3)
    assert len(explanations) == 3
    predictions = [e.prediction for e in explanations]
    assert predictions == sorted(predictions, reverse=True)


def test_summary_shape(portfolio):
    pipeline = CashFlowPipeline(scorer="heuristic").fit(portfolio.transactions, portfolio.labels)
    summary = pipeline.summary()
    assert summary["scorer"] == "HeuristicScorer"
    assert "policy" in summary
    assert "feature_config" in summary


def test_pipeline_rejects_mismatched_labels(company_transactions):
    labels = pd.Series({"unknown_entity": 1})
    with pytest.raises(ValueError, match="No entities matched"):
        CashFlowPipeline(scorer="heuristic").fit(company_transactions, labels)
