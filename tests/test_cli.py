"""Smoke tests for the command-line interface."""

from __future__ import annotations

import pytest

from opencashflow.cli import build_parser


def test_demo_parser():
    parser = build_parser()
    args = parser.parse_args(["demo", "--companies", "12", "--scorer", "heuristic"])
    assert args.command == "demo"
    assert args.companies == 12
    assert args.scorer == "heuristic"


def test_features_parser():
    parser = build_parser()
    args = parser.parse_args(
        ["features", "--transactions", "transactions.csv", "--window-days", "180"]
    )
    assert args.command == "features"
    assert args.window_days == 180


def test_version_exits():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0


def test_download_uses_positional_datasets():
    parser = build_parser()
    args = parser.parse_args(["download", "polish", "german", "bangladesh_sme"])
    assert args.dataset == ["polish", "german", "bangladesh_sme"]


def test_benchmark_parser_defaults():
    args = build_parser().parse_args(["benchmark"])
    assert args.synthetic_companies == 300
    assert args.synthetic_months == 20
    assert args.repeats == 1
    assert args.limix_pipelines == 1
    assert args.split == "stratified"
    assert args.time_column is None
    assert args.oot_cutoff is None
    assert args.exclude_features is None
    assert args.variant == "all"
    assert args.max_rows is None


def test_benchmark_parser_custom_options():
    args = build_parser().parse_args(
        [
            "benchmark",
            "--repeats",
            "2",
            "--split",
            "oot",
            "--time-column",
            "approval_fy",
            "--oot-cutoff",
            "2003",
            "--exclude-features",
            "term_months,term_years",
            "--variant",
            "no_term",
            "--max-rows",
            "5000",
            "--synthetic-companies",
            "50",
            "--synthetic-months",
            "12",
            "--limix-pipelines",
            "2",
        ]
    )
    assert args.repeats == 2
    assert args.split == "oot"
    assert args.time_column == "approval_fy"
    assert args.oot_cutoff == 2003.0
    assert args.exclude_features == "term_months,term_years"
    assert args.variant == "no_term"
    assert args.max_rows == 5000
    assert args.synthetic_companies == 50
    assert args.synthetic_months == 12
    assert args.limix_pipelines == 2
