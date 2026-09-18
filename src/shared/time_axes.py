"""Platform-specific time-axis contracts used by analysis modules."""

from __future__ import annotations

from datetime import date
from typing import Mapping

from src.shared.events import load_events
from src.shared.case_windows import case_window_for_date, load_case_windows


def reddit_case_window(day: date) -> str:
    return case_window_for_date(day, load_case_windows())


def bluesky_day(row: Mapping[str, object]) -> date:
    value = row.get("day")
    if value in (None, ""):
        raise ValueError("Bluesky analysis rows must use processed day")
    return date.fromisoformat(str(value))


def validate_event_calendar() -> None:
    events = {event.event_id: event for event in load_events()}
    if events.get("E5") is None or events["E5"].event_date.isoformat() != "2026-03-09":
        raise ValueError("superseded E5 date detected; expected 2026-03-09")


def reject_superseded_event_date(event_id: str, event_date: str) -> None:
    if event_id == "E5" and event_date != "2026-03-09":
        raise ValueError(f"superseded {event_id} date artifact rejected: {event_date}")
