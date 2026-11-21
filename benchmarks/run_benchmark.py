"""Run the OpenCashFlow benchmark from the command line.

Examples::

    python benchmarks/run_benchmark.py --datasets german --models logistic,lightgbm
    python benchmarks/run_benchmark.py --datasets synthetic,german,bangladesh_sme,polish
    python benchmarks/run_benchmark.py --datasets polish --models limix --limix-pipelines 1
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
    parser.add_argument("--max-rows", type=int, default=None, help="Optional stratified row cap.")
    parser.add_argument(
        "--synthetic-companies", type=int, default=160, help="Synthetic portfolio size."
    )
    parser.add_argument(
        "--synthetic-months", type=int, default=18, help="Synthetic history length."
    )
    parser.add_argument("--limix-pipelines", type=int, default=1, help="LimiX ensemble members.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for CSV and Markdown outputs.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir if str(args.output_dir) else None
    report = run_benchmark(
        datasets=_split(args.datasets),
        models=_split(args.models),
        cv=args.cv,
        seed=args.seed,
        max_rows=args.max_rows,
        synthetic_companies=args.synthetic_companies,
        synthetic_months=args.synthetic_months,
        limix_pipelines=args.limix_pipelines,
        output_dir=output_dir,
        verbose=not args.quiet,
    )
    print("\n" + report.to_markdown())
    if output_dir is not None:
        print(f"\nSaved results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
