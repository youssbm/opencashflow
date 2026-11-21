"""Generate the feature dictionary from the single source of truth in `features/specs.py`.

Usage::

    python scripts/generate_feature_docs.py
    python scripts/generate_feature_docs.py --output docs/features.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from opencashflow.features.specs import FEATURE_SPECS, feature_group_counts, feature_specs_frame

GROUP_TITLES = {
    "coverage": "Coverage and data quality",
    "revenue": "Revenue quality and concentration",
    "expenses": "Expense structure",
    "liquidity": "Liquidity and cash flow",
    "debt": "Debt service",
    "distress": "Distress events",
    "behaviour": "Behavioural signals",
    "trends": "Trends and seasonality",
    "composite": "Composite health scores",
}


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render() -> str:
    counts = feature_group_counts()
    frame = feature_specs_frame()
    lines = [
        "# Feature dictionary",
        "",
        "OpenCashFlow computes **151 features** per borrower from a transaction ledger. This page is",
        "generated from `opencashflow/features/specs.py`; do not edit it by hand. Run",
        "`python scripts/generate_feature_docs.py` after changing the catalogue.",
        "",
        "All features are finite floats. Features whose direction is marked **higher is better** are",
        "protective from the lender's point of view; **higher is worse** features are risk drivers.",
        "The sign is a heuristic for explanation and monitoring, not a causal claim and not enforced",
        "by the models.",
        "",
        "## Groups",
        "",
        "| Group | Features | What it captures |",
        "|---|---:|---|",
    ]
    for group, title in GROUP_TITLES.items():
        lines.append(f"| `{group}` | {counts.get(group, 0)} | {title} |")
    lines.extend(
        [
            f"| **Total** | **{len(FEATURE_SPECS)}** | |",
            "",
            "## Usage",
            "",
            "```python",
            "from opencashflow.features.cashflow_features import CashFlowFeatureExtractor",
            "",
            "extractor = CashFlowFeatureExtractor()",
            "features = extractor.fit_transform(transactions)",
            "print(features.shape)  # (n_entities, 151)",
            "```",
            "",
            "Set `FeatureConfig(window_days=730)` to use two years of history, or pass `as_of` to",
            "score at a fixed point in time. Monthly features include partial boundary months;",
            "`months_observed` and `coverage_ratio` describe how much history is available.",
            "",
        ]
    )
    for group, title in GROUP_TITLES.items():
        group_frame = frame[frame["group"] == group]
        if group_frame.empty:
            continue
        lines.extend(
            [
                f"## {title}",
                "",
                "| Feature | Description | Direction | Formula |",
                "|---|---|---|---|",
            ]
        )
        for row in group_frame.itertuples(index=False):
            if row.higher_is_better is True:
                direction = "higher is better"
            elif row.higher_is_better is False:
                direction = "higher is worse"
            else:
                direction = "contextual"
            formula = _escape(row.formula) if row.formula else ""
            lines.append(f"| `{row.name}` | {_escape(row.description)} | {direction} | {formula} |")
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "* Revenue means customer receipts (`revenue` category). Other credits are tracked",
            "  separately as `other_inflow`, `transfer_in`, `loan_proceeds`, `refund` and",
            "  `interest_income`.",
            "* Operating expenses exclude loan principal. Debt service is tracked in the `debt` group.",
            "* If the source has no running balance, the extractor derives a cumulative net position",
            "  and sets `has_balance_data = 0`. Treat liquidity features with extra caution in that",
            "  case.",
            "* Segment features by sector before comparing raw currency amounts. Ratios and scores are",
            "  more comparable across companies.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "features.md",
    )
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(), encoding="utf-8")
    print(f"Wrote {args.output} ({len(FEATURE_SPECS)} features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
