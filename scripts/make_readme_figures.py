"""Regenerate the figures used in the README.

Run from the repository root with the dev and tree extras installed::

    python scripts/make_readme_figures.py

The benchmark chart reads the saved summary in ``benchmarks/results``. The other charts use the
published synthetic configuration (300 companies, 20 months, seed 42) and a 120-company LightGBM
run with the same seed as the README quickstart. The script writes PNG files to ``docs/assets``
and prints the numbers quoted in the README. It does not change any saved result.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opencashflow.data.synthetic import generate_portfolio
from opencashflow.features.cashflow_features import (
    CashFlowFeatureExtractor,
    FeatureConfig,
)
from opencashflow.features.specs import feature_group_counts
from opencashflow.pipeline import CashFlowPipeline

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
SUMMARY = ROOT / "benchmarks" / "results" / "benchmark_summary.csv"

MODELS = {
    "limix": ("LimiX-2", "#1b6ca8"),
    "xgboost": ("XGBoost", "#6fa3c7"),
    "lightgbm": ("LightGBM", "#a9c9de"),
    "logistic": ("Logistic", "#bdbdbd"),
    "heuristic": ("Heuristic (untrained)", "#e0a458"),
}
DATASETS = {
    "bangladesh_sme": "Bangladesh SME",
    "german": "German credit",
    "polish": "Polish bankruptcy",
    "synthetic": "Synthetic ledger",
}


def benchmark_chart(path: Path) -> None:
    """Grouped ROC-AUC bars with the saved 95% intervals."""
    summary = pd.read_csv(SUMMARY)
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=120)
    width = 0.16
    x = np.arange(len(DATASETS))
    for i, (key, (label, color)) in enumerate(MODELS.items()):
        means, lows, highs, xs = [], [], [], []
        for j, dataset in enumerate(DATASETS):
            row = summary[(summary["dataset"] == dataset) & (summary["model"] == key)]
            if row.empty:
                continue
            r = row.iloc[0]
            xs.append(x[j] + (i - 2) * width)
            means.append(r["roc_auc"])
            lows.append(r["roc_auc"] - r["roc_auc_ci_low"])
            highs.append(r["roc_auc_ci_high"] - r["roc_auc"])
        bars = ax.bar(xs, means, width, color=color, label=label, edgecolor="white")
        ax.errorbar(xs, means, yerr=[lows, highs], fmt="none", ecolor="#333333", capsize=3, lw=1)
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.515,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7.5,
                color="#1a1a1a",
            )
    ax.set_xticks(x, list(DATASETS.values()))
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC by dataset: mean and 95% interval over 6 runs", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def feature_group_chart(path: Path) -> None:
    """Horizontal bars: number of documented features per group."""
    counts = feature_group_counts()
    items = sorted(counts.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=120)
    ax.barh([name.capitalize() for name, _ in items], [n for _, n in items], color="#1b6ca8")
    for k, (_, n) in enumerate(items):
        ax.text(n + 0.4, k, str(n), va="center", fontsize=9)
    ax.set_xlabel("features")
    ax.set_title(f"{sum(counts.values())} documented features in {len(counts)} groups", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def outcome_chart(path: Path, portfolio) -> dict[str, float]:
    """Two features measured before the outcome window, split by default outcome."""
    config = FeatureConfig(as_of=portfolio.metadata["observation_end"])
    features = CashFlowFeatureExtractor(config=config).fit_transform(portfolio.transactions)
    flags = portfolio.labels.set_index("entity_id")["default_flag"].reindex(features.index)
    panels = [
        ("cashflow_health_score", "Cash-flow health score (0-100)"),
        ("net_cash_flow_negative_ratio", "Share of months with negative net cash flow"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=120)
    stats: dict[str, float] = {}
    for ax, (feature, title) in zip(axes, panels):
        groups = [features.loc[flags == 0, feature], features.loc[flags == 1, feature]]
        bp = ax.boxplot(groups, widths=0.5, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], ["#a9c9de", "#d9534f"]):
            patch.set_facecolor(color)
        ax.set_xticks([1, 2], ["No default", "Default"])
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        stats[f"{feature}_no_default_median"] = float(groups[0].median())
        stats[f"{feature}_default_median"] = float(groups[1].median())
    fig.suptitle("Features measured before the outcome window separate defaulters", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return stats


def reason_code_chart(path: Path) -> dict[str, object]:
    """Top TreeSHAP contributions for the riskiest company in the 120-company quickstart."""
    portfolio = generate_portfolio(n_companies=120, months=20, seed=42)
    config = FeatureConfig(as_of=portfolio.metadata["observation_end"])
    pipeline = CashFlowPipeline(config=config, scorer="lightgbm")
    pipeline.fit(portfolio.transactions, portfolio.labels)
    scored = pipeline.predict(portfolio.transactions, explain=False)
    top_id = scored.sort_values("pd", ascending=False).index[0]
    top_pd = float(scored.loc[top_id, "pd"])
    explanation = pipeline.explain(portfolio.transactions[portfolio.transactions["entity_id"] == top_id])[0]
    top = explanation.top(6).iloc[::-1]
    labels = [f"{f} = {v:.2f}" for f, v in zip(top["feature"], top["value"])]
    colors = ["#d9534f" if c > 0 else "#1b6ca8" for c in top["contribution"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=120)
    ax.barh(labels, top["contribution"], color=colors)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel("contribution to the model's risk score (TreeSHAP)")
    ax.set_title(f"Why {top_id} is high risk (PD {top_pd:.2f}): top drivers", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"entity": top_id, "pd": round(top_pd, 4), "reason_codes": explanation.reason_codes(3)}


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    benchmark_chart(ASSETS / "benchmark_auc.png")
    feature_group_chart(ASSETS / "feature_groups.png")
    portfolio = generate_portfolio(n_companies=300, months=20, seed=42)
    print("outcome medians:", outcome_chart(ASSETS / "outcome_separation.png", portfolio))
    print("default rate:", round(float(portfolio.labels["default_flag"].mean()), 4))
    print("top risk:", reason_code_chart(ASSETS / "reason_codes.png"))
    print("wrote", sorted(p.name for p in ASSETS.glob("*.png")))


if __name__ == "__main__":
    main()
