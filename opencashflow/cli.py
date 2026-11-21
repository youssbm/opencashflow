"""OpenCashFlow command-line interface.

Commands::

    opencashflow demo --companies 80 --scorer heuristic
    opencashflow features --transactions transactions.csv --output features.csv
    opencashflow download --dataset polish german bangladesh_sme
    opencashflow benchmark --datasets german --models logistic,lightgbm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _demo(args: argparse.Namespace) -> int:
    from opencashflow.data.synthetic import generate_portfolio
    from opencashflow.pipeline import CashFlowPipeline

    portfolio = generate_portfolio(
        n_companies=args.companies,
        months=args.months,
        default_rate=args.default_rate,
        seed=args.seed,
    )
    pipeline = CashFlowPipeline(scorer=args.scorer)
    pipeline.fit(portfolio.transactions, portfolio.labels)
    scored = pipeline.predict(portfolio.transactions, top_k=3)
    print(portfolio.summary())
    print(json.dumps(pipeline.portfolio_summary(scored), indent=2, default=str))
    columns = ["pd", "rating", "decision", "reason_1", "reason_2"]
    print(scored.head(10)[[c for c in columns if c in scored.columns]].to_string())
    if args.output:
        scored.to_csv(args.output)
        print(f"Saved scores to {args.output}")
    return 0


def _features(args: argparse.Namespace) -> int:
    from opencashflow.data.schema import normalize_transactions
    from opencashflow.features.cashflow_features import CashFlowFeatureExtractor, FeatureConfig

    frame = pd.read_csv(args.transactions)
    mapping = {
        key: value
        for key, value in {
            "entity_id": args.entity_column,
            "date": args.date_column,
            "amount": args.amount_column,
            "description": args.description_column,
            "counterparty": args.counterparty_column,
            "balance": args.balance_column,
        }.items()
        if value
    }
    transactions = normalize_transactions(frame, mapping=mapping or None)
    extractor = CashFlowFeatureExtractor(config=FeatureConfig(window_days=args.window_days))
    features = extractor.fit_transform(transactions)
    if args.output:
        features.to_csv(args.output)
        print(f"Saved {features.shape[0]} entities x {features.shape[1]} features to {args.output}")
    else:
        print(features.to_string())
    return 0


def _download(args: argparse.Namespace) -> int:
    from opencashflow.data.download import main as download_main

    argv = ["--dataset", *args.dataset]
    if args.data_dir:
        argv.extend(["--data-dir", str(args.data_dir)])
    if args.force:
        argv.append("--force")
    return download_main(argv)


def _benchmark(args: argparse.Namespace) -> int:
    from opencashflow.benchmark import run_benchmark

    report = run_benchmark(
        datasets=[item.strip() for item in args.datasets.split(",") if item.strip()],
        models=[item.strip() for item in args.models.split(",") if item.strip()],
        cv=args.cv,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(report.to_markdown())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opencashflow",
        description="Cash-flow underwriting for SME credit risk.",
    )
    parser.add_argument("--version", action="version", version="opencashflow 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Generate a synthetic portfolio and score it.")
    demo.add_argument("--companies", type=int, default=60)
    demo.add_argument("--months", type=int, default=18)
    demo.add_argument("--default-rate", type=float, default=0.15)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument(
        "--scorer", default="heuristic", help="heuristic, logistic, lightgbm, xgboost or limix"
    )
    demo.add_argument("--output", type=Path, default=None, help="Optional CSV output for scores.")
    demo.set_defaults(func=_demo)

    features = subparsers.add_parser(
        "features", help="Compute cash-flow features from a transaction CSV."
    )
    features.add_argument("--transactions", type=Path, required=True)
    features.add_argument("--output", type=Path, default=None)
    features.add_argument("--window-days", type=int, default=365)
    features.add_argument("--entity-column", default=None)
    features.add_argument("--date-column", default=None)
    features.add_argument("--amount-column", default=None)
    features.add_argument("--description-column", default=None)
    features.add_argument("--counterparty-column", default=None)
    features.add_argument("--balance-column", default=None)
    features.set_defaults(func=_features)

    download = subparsers.add_parser("download", help="Download benchmark datasets.")
    download.add_argument("dataset", nargs="+", help="Dataset names, or 'all'.")
    download.add_argument("--data-dir", type=Path, default=None)
    download.add_argument("--force", action="store_true")
    download.set_defaults(func=_download)

    benchmark = subparsers.add_parser("benchmark", help="Run the model benchmark suite.")
    benchmark.add_argument(
        "--datasets",
        default="synthetic,german,bangladesh_sme,polish",
        help="Comma-separated dataset names.",
    )
    benchmark.add_argument(
        "--models",
        default="heuristic,logistic,lightgbm,xgboost,limix",
        help="Comma-separated model names.",
    )
    benchmark.add_argument("--cv", type=int, default=3)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--output-dir", type=Path, default=None)
    benchmark.set_defaults(func=_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
