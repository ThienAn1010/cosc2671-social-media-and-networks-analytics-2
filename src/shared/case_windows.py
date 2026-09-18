"""Proposal case windows, separate from the E1--E5 point-event calendar."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_WINDOWS_PATH = REPO_ROOT / "config" / "case_windows.csv"

# Source-event names are the stable bridge for Reddit's prepared corpus. This
# is separate from the E1--E5 point-event calendar and from case-window dates.
REDDIT_EVENT_BY_CASE = {
    "AU_LEGISLATION": "australia_legislation",
    "UK_ENFORCEMENT_CLUSTER": "uk_eu_policy_cluster",
    "AU_IMPLEMENTATION": "australia_implementation",
}
REDDIT_CASE_BY_EVENT = {event: case for case, event in REDDIT_EVENT_BY_CASE.items()}


@dataclass(frozen=True)
class CaseWindow:
    case_window_id: str
    start: date
    end: date
    anchor_event_id: str
    scope_status: str
    purpose: str

    def contains(self, day: date) -> bool:
        return self.start <= day < self.end


def load_case_windows(path: Path = DEFAULT_CASE_WINDOWS_PATH) -> list[CaseWindow]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        CaseWindow(
            case_window_id=row["case_window_id"],
            start=date.fromisoformat(row["start_utc"]),
            end=date.fromisoformat(row["end_utc"]),
            anchor_event_id=row["anchor_event_id"],
            scope_status=row["scope_status"],
            purpose=row["purpose"],
        )
        for row in sorted(rows, key=lambda item: item["start_utc"])
    ]


def case_window_for_date(day: date, windows: list[CaseWindow] | None = None) -> str:
    matches = [window.case_window_id for window in (windows or load_case_windows()) if window.contains(day)]
    if len(matches) > 1:
        raise ValueError(f"date belongs to overlapping case windows: {matches}")
    return matches[0] if matches else "none"
