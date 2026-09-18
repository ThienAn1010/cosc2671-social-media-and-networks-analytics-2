from __future__ import annotations

import json

import pytest

from src.analysis.scope import CORE_WINDOW_IDS, check_scope, validate_scope, write_scope_artifact


def test_scope_registry_is_proposal_aligned():
    payload = validate_scope()
    assert tuple(row["case_window_id"] for row in payload["core_case_windows"]) == CORE_WINDOW_IDS
    assert payload["point_event_registry"][-1]["event_id"] == "E5"
    assert payload["point_event_registry"][-1]["event_date_utc"] == "2026-03-09"
    assert payload["approval_gate"]["without_receipt"] == "repository_only_exploratory"


def test_scope_artifact_is_replayable(tmp_path):
    path = tmp_path / "scope_manifest.json"
    write_scope_artifact(path)
    first = json.loads(path.read_text(encoding="utf-8"))
    result = check_scope(path)
    second = json.loads(path.read_text(encoding="utf-8"))
    assert result["status"] == "valid"
    assert first == second


def test_scope_check_is_read_only_when_artifact_is_missing(tmp_path):
    path = tmp_path / "missing-scope.json"
    with pytest.raises(FileNotFoundError):
        check_scope(path)
    assert not path.exists()
