# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

"""Event calendar helpers: load config/events.csv and tag dates with event windows (docs/06 §4)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS_PATH = REPO_ROOT / "config" / "events.csv"
# Window [date - 7, date + 21) keeps the five event windows from overlapping (DECISIONS D-005).
WINDOW_DAYS_BEFORE = 7
WINDOW_DAYS_AFTER = 21
NO_WINDOW = "none"


@dataclass(frozen=True)
class Event:
    event_id: str
    event_date: date
    jurisdiction: str
    event_type: str
    role: str
    control_jurisdiction: str
    name: str
    source_url: str

    # First day inside the window.
    @property
    def window_start(self) -> date:
        return self.event_date - timedelta(days=WINDOW_DAYS_BEFORE)

    # First day after the window (half-open end).
    @property
    def window_end(self) -> date:
        return self.event_date + timedelta(days=WINDOW_DAYS_AFTER)


def load_events(path: Path = DEFAULT_EVENTS_PATH) -> list[Event]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    events = [
        Event(
            event_id=row["event_id"],
            event_date=date.fromisoformat(row["event_date_utc"]),
            jurisdiction=row["jurisdiction"],
            event_type=row["event_type"],
            role=row["role"],
            control_jurisdiction=row["control_jurisdiction"],
            name=row["name"],
            source_url=row["source_url"],
        )
        for row in rows
    ]
    return sorted(events, key=lambda event: event.event_date)


# Event ID whose window contains the day, or "none".
def event_window(day: date, events: list[Event]) -> str:
    for event in events:
        if event.window_start <= day < event.window_end:
            return event.event_id
    return NO_WINDOW


# Nearest event by absolute distance in days; on a tie the earlier event wins so the result is deterministic.
def nearest_event(day: date, events: list[Event]) -> tuple[Event, int]:
    event = min(events, key=lambda item: (abs((day - item.event_date).days), item.event_date))
    return event, (day - event.event_date).days


def _midnight_utc(day: date) -> str:
    return datetime.combine(day, time(0), tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# RFC 3339 bounds for YouTube search: publishedAfter = window start, publishedBefore = window end.
def window_rfc3339(event: Event) -> tuple[str, str]:
    return _midnight_utc(event.window_start), _midnight_utc(event.window_end)
