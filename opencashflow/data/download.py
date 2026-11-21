"""Command-line downloader for the public benchmark datasets.

Usage::

    python -m opencashflow.data.download --list
    python -m opencashflow.data.download --dataset polish german
    python -m opencashflow.data.download --dataset all --data-dir ./data/raw

The canonical sources are UCI and Zenodo. Bangladesh SME has a documented GitHub mirror because
Zenodo aggressively rate-limits automated downloads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from opencashflow.data.loaders import (
    DATASET_REGISTRY,
    download_dataset,
    get_data_dir,
    list_datasets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download OpenCashFlow benchmark datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=[*DATASET_REGISTRY, "all"],
        default=["all"],
        help="Dataset(s) to download.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Raw-data directory.")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    parser.add_argument("--list", action="store_true", help="List datasets and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print(list_datasets().to_string(index=False))
        return 0
    root = get_data_dir(args.data_dir)
    print(f"Data directory: {root}")
    selected = list(DATASET_REGISTRY) if "all" in args.dataset else args.dataset
    failures = 0
    for name in selected:
        if name == "synthetic":
            print(f"[skip] {name}: generated locally, no download needed")
            continue
        try:
            path = download_dataset(name, data_dir=root, force=args.force)
            print(f"[ok]   {name}: {path}")
        except Exception as exc:
            failures += 1
            print(f"[fail] {name}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
