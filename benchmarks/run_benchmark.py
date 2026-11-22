"""Run the OpenCashFlow benchmark from the command line.

Without ``--output-dir``, results are printed and no files are written. Pass ``--output-dir``
explicitly to save the CSV and Markdown outputs.

Examples::

    python benchmarks/run_benchmark.py --datasets german --models logistic,lightgbm
    python benchmarks/run_benchmark.py --datasets synthetic,german,bangladesh_sme,polish `
        --cv 3 --limix-pipelines 1 --output-dir benchmarks/results
    python benchmarks/run_benchmark.py --datasets sba --models logistic,lightgbm,xgboost,limix `
        --cv 3 --max-rows 5000 --limix-pipelines 1 --output-dir benchmarks/results/sba
    python benchmarks/run_benchmark.py --datasets sba --split oot --time-column approval_fy `
        --oot-cutoff 2003 --exclude-features term_months,term_years `
        --output-dir benchmarks/results/sba

The out-of-time command is an example only. Adjust the cutoff to the scoring date of interest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opencashflow.benchmark import DEFAULT_DATASETS, DEFAULT_MODELS, run_benchmark


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark LimiX-2 against tree and linear credit-risk baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets", default=",".join(DEFAULT_DATASETS), help="Comma-separated dataset names."
    )
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model names."
    )
    parser.add_argument("--cv", type=int, default=3, help="Number of stratified folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--repeats", type=int, default=1, help="Stratified CV repeats.")
    parser.add_argument(
        "--split", choices=["stratified", "oot"], default="stratified", help="Split strategy."
    )
    parser.add_argument("--time-column", type=str, default=None, help="Numeric column for oot.")
    parser.add_argument("--oot-cutoff", type=float, default=None, help="Cutoff for oot split.")
    parser.add_argument(
        "--exclude-features", type=str, default=None, help="Comma-separated columns to drop."
    )
    parser.add_argument("--variant", type=str, default="all", help="Result variant label.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional stratified row cap.")
    parser.add_argument(
        "--sba-approval-fy-max",
        type=int,
        default=None,
        help="SBA only: keep loans approved in or before this fiscal year (matured-vintage filter).",
    )
    parser.add_argument(
        "--synthetic-companies", type=int, default=300, help="Synthetic portfolio size."
    )
    parser.add_argument(
        "--synthetic-months", type=int, default=20, help="Synthetic history length."
    )
    parser.add_argument("--limix-pipelines", type=int, default=1, help="LimiX ensemble members.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV and Markdown outputs. Print only when omitted.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = _split(args.datasets)
    output_dir = args.output_dir
    report = run_benchmark(
        datasets=datasets,
        models=_split(args.models),
        cv=args.cv,
        seed=args.seed,
        max_rows=args.max_rows,
        sba_approval_fy_max=args.sba_approval_fy_max,
        synthetic_companies=args.synthetic_companies,
        synthetic_months=args.synthetic_months,
        limix_pipelines=args.limix_pipelines,
        repeats=args.repeats,
        split=args.split,
        time_column=args.time_column,
        oot_cutoff=args.oot_cutoff,
        exclude_features=tuple(_split(args.exclude_features)) if args.exclude_features else (),
        variant=args.variant,
        output_dir=output_dir,
        verbose=not args.quiet,
    )
    print("\n" + report.to_markdown())
    if output_dir is None:
        print("\nNo --output-dir given: results were printed, not saved.")
    else:
        print(f"\nSaved results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
