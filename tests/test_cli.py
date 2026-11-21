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
