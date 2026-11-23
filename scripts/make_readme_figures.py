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
from matplotlib.patches import FancyBboxPatch

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


def policy_chart(path: Path) -> None:
    """Decision bands and rating bands, drawn from the cut-offs in the code.

    The two example markers are the lowest and highest PD businesses in the 120-company worked
    example on the methodology page.
    """
    from opencashflow.models.metrics import RATING_BOUNDS
    from opencashflow.pipeline import UnderwritingPolicy

    policy = UnderwritingPolicy()
    top = 0.6
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=120)
    decisions = [
        (0.0, policy.accept_max_pd, "#4caf50", "accept"),
        (policy.accept_max_pd, policy.review_max_pd, "#f0ad4e", "review"),
        (policy.review_max_pd, top, "#d9534f", "decline"),
    ]
    for start, end, color, label in decisions:
        ax.barh(1.2, end - start, left=start, height=0.5, color=color, edgecolor="white")
        ax.text(
            (start + end) / 2,
            1.2,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
        )
    starts = [0.0] + [bound for bound, _ in RATING_BOUNDS]
    ends = [bound for bound, _ in RATING_BOUNDS] + [top]
    grades = [grade for _, grade in RATING_BOUNDS] + ["G"]
    shades = matplotlib.colormaps["Blues"](np.linspace(0.2, 0.9, len(grades)))
    for start, end, grade, shade in zip(starts, ends, grades, shades):
        ax.barh(0.2, end - start, left=start, height=0.5, color=shade, edgecolor="white")
        ax.text(
            (start + end) / 2,
            0.2,
            grade,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white" if shade[0] < 0.5 else "#1a1a1a",
        )
    examples = [(0.0028, "lowest PD example: 0.28%"), (0.594, "highest PD example: 59.4%")]
    for pd_value, text in examples:
        ax.axvline(pd_value, color="#333333", lw=0.8, ls="--")
        offset = 0.008 if pd_value < 0.3 else -0.008
        ax.text(pd_value + offset, 1.75, text, ha="left" if pd_value < 0.3 else "right", fontsize=8.5)
    ticks = np.round(np.arange(0.0, top + 1e-9, 0.1), 2)
    ax.set_xlim(0, top)
    ax.set_ylim(-0.1, 1.95)
    ax.set_xticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_yticks([1.2, 0.2], ["Decision", "Rating"])
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("From probability of default to decision and rating", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


FLOW_STYLES = {
    "plain": ("#eaf2fb", "#1b6ca8", "-"),
    "question": ("#fff4e0", "#f0ad4e", "-"),
    "accept": ("#e6f4ea", "#4caf50", "-"),
    "review": ("#fff4e0", "#f0ad4e", "-"),
    "decline": ("#fbe9e8", "#d9534f", "-"),
    "refer": ("#eeeeee", "#777777", "-"),
    "dashed": ("#f7f7f7", "#777777", "--"),
}


def draw_flow(path: Path, boxes: list[dict], arrows: list[tuple], dpi: int = 120) -> None:
    """Draw labelled boxes and arrows. Each box is sized to its text, so no label is cut off.

    A box is a dict with ``id``, ``text``, ``x`` and ``y`` (its centre in inches, y pointing up)
    and an optional ``style`` (a key of FLOW_STYLES). An arrow is a tuple
    ``(src, dst, src_side, dst_side, label, dashed)``. A side is ``left``, ``right``, ``top`` or
    ``bottom``. The function raises if two boxes overlap.
    """
    pad_x, pad_y = 0.18 * dpi, 0.14 * dpi
    probe = plt.figure(dpi=dpi)
    renderer = probe.canvas.get_renderer()
    rects: dict[str, list[float]] = {}
    for box in boxes:
        probe_text = probe.text(
            0, 0, box["text"], fontsize=10.5, linespacing=1.25, ha="center", va="center"
        )
        bbox = probe_text.get_window_extent(renderer=renderer)
        width, height = bbox.width + 2 * pad_x, bbox.height + 2 * pad_y
        rects[box["id"]] = [box["x"] * dpi - width / 2, box["y"] * dpi - height / 2, width, height]
    plt.close(probe)

    names = list(rects)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            ax1, ay1, aw, ah = rects[first]
            bx1, by1, bw, bh = rects[second]
            if ax1 < bx1 + bw and bx1 < ax1 + aw and ay1 < by1 + bh and by1 < ay1 + ah:
                raise ValueError(f"boxes {first!r} and {second!r} overlap")

    margin = 0.3 * dpi
    shift_x = margin - min(r[0] for r in rects.values())
    shift_y = margin - min(r[1] for r in rects.values())
    width_px = max(r[0] + r[2] for r in rects.values()) + shift_x + margin
    height_px = max(r[1] + r[3] for r in rects.values()) + shift_y + margin

    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_px)
    ax.set_ylim(0, height_px)
    ax.set_axis_off()
    centres: dict[str, tuple[float, float, float, float]] = {}
    for box in boxes:
        x0, y0, w, h = rects[box["id"]]
        x0, y0 = x0 + shift_x, y0 + shift_y
        face, edge, line = FLOW_STYLES[box.get("style", "plain")]
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=12",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.3,
                linestyle=line,
            )
        )
        cx, cy = x0 + w / 2, y0 + h / 2
        centres[box["id"]] = (cx, cy, w, h)
        ax.text(
            cx,
            cy,
            box["text"],
            ha="center",
            va="center",
            fontsize=10.5,
            linespacing=1.25,
            color="#1a1a1a",
        )

    def anchor(box_id: str, side: str) -> tuple[float, float]:
        cx, cy, w, h = centres[box_id]
        points = {
            "left": (cx - w / 2, cy),
            "right": (cx + w / 2, cy),
            "top": (cx, cy + h / 2),
            "bottom": (cx, cy - h / 2),
        }
        return points[side]

    for src, dst, src_side, dst_side, label, dashed in arrows:
        start, end = anchor(src, src_side), anchor(dst, dst_side)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#333333",
                "lw": 1.2,
                "shrinkA": 0,
                "shrinkB": 0,
                "linestyle": "--" if dashed else "-",
            },
        )
        if label:
            mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
            ax.text(
                mid_x,
                mid_y,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="#333333",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none"},
            )
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def pipeline_flow_chart(path: Path) -> None:
    """From bank transactions to the decision, rating, expected loss and reason codes."""
    middle = 2.75
    boxes = [
        {"id": "bank", "text": "Bank transactions", "x": 0.0, "y": middle},
        {"id": "cat", "text": "Categorise\n(seven languages)", "x": 2.6, "y": middle},
        {"id": "feat", "text": "151 cash-flow features\n(point in time)", "x": 5.2, "y": middle},
        {"id": "pd", "text": "Probability of default\n(PD)", "x": 7.8, "y": middle},
        {"id": "dec", "text": "Decision\naccept, review, decline\nor refer", "x": 10.6, "y": 4.4},
        {"id": "rat", "text": "Rating\nA (safest) to G (riskiest)", "x": 10.6, "y": 3.3},
        {"id": "el", "text": "Expected loss\nPD x LGD x EAD", "x": 10.6, "y": 2.2},
        {"id": "rc", "text": "Reason codes\nfeatures that raised PD", "x": 10.6, "y": 1.1},
    ]
    arrows = [
        ("bank", "cat", "right", "left", "", False),
        ("cat", "feat", "right", "left", "", False),
        ("feat", "pd", "right", "left", "", False),
        ("pd", "dec", "right", "left", "", False),
        ("pd", "rat", "right", "left", "", False),
        ("pd", "el", "right", "left", "", False),
        ("pd", "rc", "right", "left", "", False),
    ]
    draw_flow(path, boxes, arrows)


def decision_flow_chart(path: Path) -> None:
    """The history check, then the PD cut-offs, ending in accept, review, refer or decline."""
    boxes = [
        {
            "id": "history",
            "text": "Enough history?\nAt least 3 months and\n20 transactions",
            "x": 0.0,
            "y": 4.4,
            "style": "question",
        },
        {"id": "refer", "text": "Refer to an underwriter", "x": 3.9, "y": 4.4, "style": "refer"},
        {"id": "pd5", "text": "PD at most 5%?", "x": 0.0, "y": 2.9, "style": "question"},
        {"id": "accept", "text": "Accept", "x": 3.9, "y": 2.9, "style": "accept"},
        {"id": "pd20", "text": "PD at most 20%?", "x": 0.0, "y": 1.4, "style": "question"},
        {"id": "review", "text": "Review by an underwriter", "x": 3.9, "y": 1.4, "style": "review"},
        {"id": "decline", "text": "Decline", "x": 0.0, "y": -0.1, "style": "decline"},
    ]
    arrows = [
        ("history", "refer", "right", "left", "No", False),
        ("history", "pd5", "bottom", "top", "Yes", False),
        ("pd5", "accept", "right", "left", "Yes", False),
        ("pd5", "pd20", "bottom", "top", "No", False),
        ("pd20", "review", "right", "left", "Yes", False),
        ("pd20", "decline", "bottom", "top", "No", False),
    ]
    draw_flow(path, boxes, arrows)


def architecture_flow_chart(path: Path) -> None:
    """The layers of the system, from the open-banking export to the decision and reasons."""
    boxes = [
        {"id": "export", "text": "Open-banking export\n(CSV or API)", "x": 0.0, "y": 9.0},
        {"id": "schema", "text": "Canonical transaction schema", "x": 0.0, "y": 7.7},
        {"id": "cat", "text": "Rule-first categoriser", "x": 0.0, "y": 6.4},
        {"id": "feat", "text": "151 cash-flow features", "x": 0.0, "y": 5.1},
        {
            "id": "macro",
            "text": "Eurostat / EBA macro table\n(optional, lagged)",
            "x": -3.9,
            "y": 5.1,
            "style": "dashed",
        },
        {"id": "seg", "text": "Segment and portfolio\nanalytics", "x": 3.9, "y": 5.1},
        {"id": "score", "text": "Scoring engine", "x": 0.0, "y": 3.8},
        {
            "id": "models",
            "text": "Models: LimiX-2, LightGBM,\nXGBoost, logistic regression,\nheuristic scorecard",
            "x": 3.9,
            "y": 3.8,
        },
        {"id": "pd", "text": "Probability of default", "x": 0.0, "y": 2.5},
        {"id": "rating", "text": "Rating and policy decision", "x": 0.0, "y": 1.2},
        {"id": "reason", "text": "Reason codes, SHAP\nand occlusion", "x": 0.0, "y": -0.1},
    ]
    arrows = [
        ("export", "schema", "bottom", "top", "", False),
        ("schema", "cat", "bottom", "top", "", False),
        ("cat", "feat", "bottom", "top", "", False),
        ("feat", "score", "bottom", "top", "", False),
        ("macro", "feat", "right", "left", "", True),
        ("feat", "seg", "right", "left", "", False),
        ("models", "score", "left", "right", "", False),
        ("score", "pd", "bottom", "top", "", False),
        ("pd", "rating", "bottom", "top", "", False),
        ("rating", "reason", "bottom", "top", "", False),
    ]
    draw_flow(path, boxes, arrows)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    benchmark_chart(ASSETS / "benchmark_auc.png")
    feature_group_chart(ASSETS / "feature_groups.png")
    portfolio = generate_portfolio(n_companies=300, months=20, seed=42)
    print("outcome medians:", outcome_chart(ASSETS / "outcome_separation.png", portfolio))
    print("default rate:", round(float(portfolio.labels["default_flag"].mean()), 4))
    print("top risk:", reason_code_chart(ASSETS / "reason_codes.png"))
    policy_chart(ASSETS / "policy_bands.png")
    pipeline_flow_chart(ASSETS / "flow_pipeline.png")
    decision_flow_chart(ASSETS / "flow_decision.png")
    architecture_flow_chart(ASSETS / "flow_architecture.png")
    print("wrote", sorted(p.name for p in ASSETS.glob("*.png")))


if __name__ == "__main__":
    main()
