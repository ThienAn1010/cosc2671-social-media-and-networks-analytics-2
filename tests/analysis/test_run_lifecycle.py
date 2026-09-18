from __future__ import annotations

import pytest

from src.analysis.run import build_parser


def test_analysis_commands_require_an_explicit_mode():
    assert build_parser().parse_args(["scope", "--check"]).check is True
    assert build_parser().parse_args(["scope", "--build"]).build is True
    with pytest.raises(SystemExit):
        build_parser().parse_args(["scope"])


def test_hypothesis_has_separate_build_execute_and_check_modes():
    parser = build_parser()
    assert parser.parse_args(["hypothesis", "--id", "H1", "--build"]).build is True
    assert parser.parse_args(["hypothesis", "--id", "H1", "--execute"]).execute is True
    assert parser.parse_args(["hypothesis", "--id", "H1", "--check"]).check is True


def test_validation_sample_accepts_an_explicit_revision():
    args = build_parser().parse_args(["validation", "sample", "--build", "--revision", "cycle-v3"])
    assert args.revision == "cycle-v3"
