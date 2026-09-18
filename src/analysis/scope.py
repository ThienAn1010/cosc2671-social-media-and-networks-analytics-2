"""Scope contracts for the frozen Age-Gate Paradox analysis.

The proposal case windows and the point-event calendar answer different
questions. Keeping them in separate registries prevents an E1--E5 overlay from
silently becoming the assessed case-window definition.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from src.shared.case_windows import REDDIT_EVENT_BY_CASE, load_case_windows as load_shared_case_windows
from src.shared.events import load_events

REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_WINDOWS_PATH = REPO_ROOT / "config" / "case_windows.csv"
SCOPE_ARTIFACT_PATH = REPO_ROOT / "data" / "analysis" / "scope" / "scope_manifest.json"

CORE_WINDOW_IDS = ("AU_LEGISLATION", "UK_ENFORCEMENT_CLUSTER", "AU_IMPLEMENTATION")
POINT_EVENT_STATUS = {
    "E1": "context_negative_control",
    "E2": "core_anchor",
    "E3": "core_anchor",
    "E4": "approval_required_legislative_extension",
    "E5": "approval_required_extension",
}
CORE_HYPOTHESES = ("H1", "H2", "H3", "H4")
EXTENDED_ANALYSES = (
    "google_trends_timing",
    "change_points",
    "granger_predictive_tests",
    "forecasting",
)
AUTHORITY_FILES = (
    "verify_me_not_analysis_plan_revised.md",
    "docs/age_gate_paradox_project_proposal.md",
    "ASSESSMENT_SPEC.md",
    "docs/data_handling.md",
    "config/events.csv",
    "config/case_windows.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_case_windows(path: Path = CASE_WINDOWS_PATH) -> list[dict[str, str]]:
    return [
        {
            "case_window_id": window.case_window_id,
            "start_utc": window.start.isoformat(),
            "end_utc": window.end.isoformat(),
            "anchor_event_id": window.anchor_event_id,
            "scope_status": window.scope_status,
            "purpose": window.purpose,
        }
        for window in load_shared_case_windows(path)
    ]


def _parse_day(value: str) -> date:
    return date.fromisoformat(value)


def validate_scope(
    case_windows_path: Path = CASE_WINDOWS_PATH,
    events_path: Path | None = None,
) -> dict[str, Any]:
    windows = load_case_windows(case_windows_path)
    if tuple(row["case_window_id"] for row in windows) != CORE_WINDOW_IDS:
        raise ValueError("case-window registry must contain exactly the three proposal core windows")
    for row in windows:
        start, end = _parse_day(row["start_utc"]), _parse_day(row["end_utc"])
        if start >= end or row["scope_status"] != "core":
            raise ValueError(f"invalid core case window: {row['case_window_id']}")
    for previous, current in zip(windows, windows[1:]):
        if _parse_day(previous["end_utc"]) > _parse_day(current["start_utc"]):
            raise ValueError("proposal case windows overlap")

    events = load_events(events_path or REPO_ROOT / "config" / "events.csv")
    event_ids = {event.event_id for event in events}
    if set(event_ids) != set(POINT_EVENT_STATUS):
        raise ValueError(f"point-event registry mismatch: {sorted(event_ids)}")
    event_dates = {event.event_id: event.event_date.isoformat() for event in events}
    if event_dates["E5"] != "2026-03-09":
        raise ValueError("E5 must use the authoritative 2026-03-09 date")
    if any(POINT_EVENT_STATUS[event.event_id].startswith("approval_required") for event in events if event.event_id in {"E4", "E5"}) is False:
        raise ValueError("approval-gated point-event status is missing")

    uk = next(row for row in windows if row["case_window_id"] == "UK_ENFORCEMENT_CLUSTER")
    au_impl = next(row for row in windows if row["case_window_id"] == "AU_IMPLEMENTATION")
    if not (_parse_day(uk["start_utc"]) <= _parse_day(event_dates["E2"]) < _parse_day(uk["end_utc"])):
        raise ValueError("E2 is not inside the UK core case window")
    if not (_parse_day(au_impl["start_utc"]) <= _parse_day(event_dates["E3"]) < _parse_day(au_impl["end_utc"])):
        raise ValueError("E3 is not inside the AU implementation case window")

    return {
        "schema_version": "scope.v1",
        "scope_version": "age-gate-paradox-core-2026-09-15",
        "core_case_windows": windows,
        "point_event_registry": [
            {
                "event_id": event.event_id,
                "event_date_utc": event.event_date.isoformat(),
                "role": POINT_EVENT_STATUS[event.event_id],
                "event_type": event.event_type,
                "jurisdiction": event.jurisdiction,
            }
            for event in events
        ],
        "core_hypotheses": list(CORE_HYPOTHESES),
        "extended_analyses": list(EXTENDED_ANALYSES),
        "approval_gate": {
            "receipt_path": "data/analysis/scope/lecturer_approval.json",
            "required_for": ["E4", "E5", *EXTENDED_ANALYSES],
            "without_receipt": "repository_only_exploratory",
        },
        "authority_checksums": {
            relative: _sha256(REPO_ROOT / relative) for relative in AUTHORITY_FILES
        },
    }


def write_scope_artifact(path: Path = SCOPE_ARTIFACT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = validate_scope()
    payload["created_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def check_scope(path: Path = SCOPE_ARTIFACT_PATH) -> dict[str, Any]:
    expected = validate_scope()
    if not path.exists():
        raise FileNotFoundError(f"scope artifact is missing: {path}; run the explicit scope build first")
    actual = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"scope artifact is stale or inconsistent at {path}: {key}")
    return {"status": "valid", "artifact": str(path), "scope_version": actual["scope_version"]}
