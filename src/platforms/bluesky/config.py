# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

"""Load and validate `config/bluesky/config.yaml`, and read credentials from `.env`.

Why this file exists: every script in this project needs the same queries,
dates and paths. Putting that in one place means changing a query is a one-line
edit to a YAML file, not a hunt through four Python scripts.

Everything here works in **UTC**. Mixing local and UTC time is the single most
common way to end up with off-by-one-day gaps in a corpus -- and a gap looks
exactly like a genuinely quiet day, so you would never notice.

Adapted from the Assignment 1 loader. The Reddit half is gone (another team
member owns that platform), and three things are new: the two-stratum query
structure, the baseline/denominator queries, and the event calendar.
"""

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.shared.events import load_events as load_team_events

# `__file__` is src/platforms/bluesky/config.py, so three levels up is the repository root.
# Deriving it this way means scripts work no matter which directory you run
# them from.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Event:
    """One dated policy shock from the event calendar."""

    id: str
    date: date
    kind: str          # event_type from the shared calendar
    jurisdiction: str
    name: str
    role: str = "treatment"        # "treatment" | "placebo" -- the DiD design
    control_jurisdiction: str = "none"
    source_url: str = ""


@dataclass(frozen=True)
class Config:
    """Validated settings from config/bluesky/config.yaml.

    `frozen=True` makes instances read-only. Config that mutates halfway
    through a long collection run is a debugging nightmare, so we forbid it.
    """

    queries_a: list[str]
    queries_b: list[str]
    baseline_queries: list[str]
    date_start: date
    date_end: date
    events: list[Event]
    raw_dir: Path
    checkpoint_path: Path
    enrich_checkpoint_path: Path
    profiles_dir: Path
    profile_checkpoint_path: Path
    follows_dir: Path
    follows_checkpoint_path: Path
    follows_min_posts: int
    follows_max_seeds: int
    follows_max_per_user: int
    baseline_path: Path
    baseline_checkpoint_path: Path
    interim_dir: Path
    processed_path: Path
    figures_dir: Path
    sleep_seconds: float
    max_pages_per_cell: int
    enrich_min_reply_count: int
    enrich_depth: int
    enrich_max_replies: int
    profile_batch_size: int

    # --- queries ---------------------------------------------------------

    @property
    def queries(self) -> list[str]:
        """Every search query, both strata, in a stable order.

        Stratum A first so that a run killed early has collected the
        trustworthy backbone rather than a random half of both strata.
        """
        return self.queries_a + self.queries_b

    def stratum_of(self, query: str) -> str:
        """Which stratum a query belongs to: "A" or "B".

        Written into every raw record's envelope. It cannot be recovered later
        if config/bluesky/config.yaml changes, and the whole frame-analysis design depends on
        being able to separate the neutral backbone from the biased
        supplement -- so it is recorded at collection time, not inferred.
        """
        return "A" if query in self.queries_a else "B"

    # --- dates -----------------------------------------------------------

    def days(self) -> list[date]:
        """Every day in the collection window, inclusive of both ends."""
        span = (self.date_end - self.date_start).days
        return [self.date_start + timedelta(days=i) for i in range(span + 1)]

    def day_window(self, day: date) -> tuple[str, str]:
        """The (since, until) bounds for one day, as ISO-8601 UTC strings.

        Returns midnight on `day` and midnight on the *following* day.

        Next-midnight rather than 23:59:59 means that if the API treats `until`
        as inclusive we collect the boundary post twice, and if exclusive,
        exactly once. A duplicate is harmless -- normalisation dedups on
        doc_id. A one-second gap every single day would not be harmless, and
        would be invisible. Always prefer the error you can clean up over the
        error you cannot see.
        """
        return (_iso_z(day), _iso_z(day + timedelta(days=1)))

    def cells(self) -> list[tuple[str, date]]:
        """The full (query x day) grid -- one unit of collection work each.

        Narrow windows are deliberate. The SDK warns that a cursor "may not
        necessarily allow scrolling through the entire result set", so one deep
        paginated run over 18 months can silently truncate. A single day of a
        single query is shallow enough to exhaust, and gives a natural unit for
        checkpointing.
        """
        return [(q, d) for q in self.queries for d in self.days()]

    def baseline_cells(self) -> list[tuple[str, date]]:
        """The (baseline term x day) grid. Counts only, never posts."""
        return [(q, d) for q in self.baseline_queries for d in self.days()]

    def ensure_dirs(self) -> None:
        """Create the output directories if they don't exist yet."""
        for path in (self.raw_dir, self.profiles_dir, self.follows_dir, self.figures_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.processed_path, self.baseline_path):
            path.parent.mkdir(parents=True, exist_ok=True)


def _iso_z(day: date) -> str:
    """Format a date as midnight UTC in the `Z` form the API expects."""
    return f"{day.isoformat()}T00:00:00Z"


def _parse_date(value: object, field_name: str) -> date:
    """Parse a YYYY-MM-DD string, failing loudly on anything else.

    We validate rather than trust because a malformed `since`/`until` may be
    silently IGNORED by the search API rather than rejected. The run would then
    return unbounded results that look perfectly fine, and the date filtering
    you thought you applied simply never happened.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"config/bluesky/config.yaml: `{field_name}` must be a quoted YYYY-MM-DD string, got {value!r}"
        )
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"config/bluesky/config.yaml: `{field_name}` is not a valid YYYY-MM-DD date: {value!r}"
        ) from exc


def _load_events(date_start: date, date_end: date) -> list[Event]:
    """Read the team's shared event calendar and keep the ones in our window.

    The calendar is `config/events.csv`, owned by Duy and consumed by every
    workstream through `src.shared.events`. This workstream does NOT keep its
    own copy: a second list would drift, and a Bluesky series dated off one
    calendar cannot be compared with a YouTube or Reddit series dated off
    another.

    Adopting it corrected a date we had wrong. Our `config/bluesky/config.yaml` carried E5 at
    2026-03-11 with the note "DATE UNVERIFIED placeholder"; the shared calendar
    has **2026-03-09**, with a primary source. Anything computed against the old
    date needs re-running.

    The calendar also carries two fields we had no equivalent for -- `role`
    (treatment / placebo) and `control_jurisdiction` -- which encode a
    difference-in-differences design. E1 and E4 are PLACEBO events. That matters
    for reading our own results: we flagged E4 as showing weak movement on
    Bluesky and treated it as a problem, when a weak response is the expected
    outcome for a placebo.

    Events do not drive collection -- the backfill is continuous and covers the
    whole window regardless -- so a calendar change never invalidates collected
    data, only the analysis layered on it.
    """
    kept = []
    for event in load_team_events():
        if not (date_start <= event.event_date <= date_end):
            continue
        kept.append(Event(
            id=event.event_id,
            date=event.event_date,
            kind=event.event_type,
            jurisdiction=event.jurisdiction,
            name=event.name,
            role=event.role,
            control_jurisdiction=event.control_jurisdiction,
            source_url=event.source_url,
        ))
    if not kept:
        raise ValueError(
            f"config/events.csv has no events inside {date_start}..{date_end}. "
            f"Either the window is wrong or the calendar moved; both are worth "
            f"knowing before an event study is run."
        )
    return kept


def load_config(path: Path | None = None) -> Config:
    """Read config/bluesky/config.yaml, validate it, and return a Config."""
    path = path or PROJECT_ROOT / "config" / "bluesky" / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    queries = raw.get("queries") or {}
    queries_a = queries.get("stratum_a") or []
    queries_b = queries.get("stratum_b") or []
    if not queries_a:
        raise ValueError(
            "config/bluesky/config.yaml: `queries.stratum_a` is empty. Stratum A is the neutral "
            "backbone that frame analysis rests on -- collecting only stratum B "
            "would produce a corpus whose frame distribution is an artefact of "
            "the query list."
        )
    overlap = set(queries_a) & set(queries_b)
    if overlap:
        raise ValueError(
            f"config/bluesky/config.yaml: {sorted(overlap)} appears in BOTH strata. `stratum_of()` "
            f"would silently report 'A', and the biased supplement would contaminate "
            f"the neutral backbone."
        )

    baseline_queries = raw.get("baseline_queries") or []
    if not baseline_queries:
        raise ValueError(
            "config/bluesky/config.yaml: `baseline_queries` is empty. Without a denominator series, "
            "discourse volume cannot be separated from Bluesky's own activity trend, "
            "and every timing result is confounded by platform drift."
        )

    date_start = _parse_date(raw.get("date_start"), "date_start")
    date_end = _parse_date(raw.get("date_end"), "date_end")
    if date_end < date_start:
        raise ValueError(
            f"config/bluesky/config.yaml: date_end ({date_end}) is before date_start ({date_start})."
        )

    # Refuse to collect today or later. Today is still in progress, so it would
    # land in the corpus as a half-empty day and read as a sudden collapse in
    # posting volume -- a data-quality bug that looks like a finding.
    today_utc = datetime.now(timezone.utc).date()
    if date_end >= today_utc:
        raise ValueError(
            f"config/bluesky/config.yaml: date_end ({date_end}) must be a COMPLETE day in the past. "
            f"Today (UTC) is {today_utc}; set date_end to "
            f"{today_utc - timedelta(days=1)} or earlier."
        )

    paths = raw.get("paths") or {}

    def _path(key: str) -> Path:
        if key not in paths:
            raise ValueError(f"config/bluesky/config.yaml: `paths.{key}` is missing.")
        return PROJECT_ROOT / paths[key]

    enrich = raw.get("enrich") or {}
    profiles = raw.get("profiles") or {}
    follows = raw.get("follows") or {}

    batch_size = int(profiles.get("batch_size", 25))
    if not 1 <= batch_size <= 25:
        raise ValueError(
            f"config/bluesky/config.yaml: `profiles.batch_size` is {batch_size}; the getProfiles "
            f"endpoint accepts at most 25 actors per call."
        )

    return Config(
        queries_a=queries_a,
        queries_b=queries_b,
        baseline_queries=baseline_queries,
        date_start=date_start,
        date_end=date_end,
        events=_load_events(date_start, date_end),
        raw_dir=_path("raw_dir"),
        checkpoint_path=_path("checkpoint"),
        enrich_checkpoint_path=_path("enrich_checkpoint"),
        profiles_dir=_path("profiles_dir"),
        profile_checkpoint_path=_path("profile_checkpoint"),
        follows_dir=_path("follows_dir"),
        follows_checkpoint_path=_path("follows_checkpoint"),
        follows_min_posts=int(follows.get("min_posts", 3)),
        follows_max_seeds=int(follows.get("max_seeds", 5000)),
        follows_max_per_user=int(follows.get("max_follows_per_user", 2000)),
        baseline_path=_path("baseline_path"),
        baseline_checkpoint_path=_path("baseline_checkpoint"),
        interim_dir=_path("interim_dir"),
        processed_path=_path("processed"),
        figures_dir=_path("figures"),
        sleep_seconds=float(raw.get("sleep_seconds", 0.05)),
        max_pages_per_cell=int(raw.get("max_pages_per_cell", 40)),
        enrich_min_reply_count=int(enrich.get("min_reply_count", 3)),
        enrich_depth=int(enrich.get("depth", 2)),
        enrich_max_replies=int(enrich.get("max_replies_per_thread", 200)),
        profile_batch_size=batch_size,
    )


def load_credentials() -> tuple[str, str]:
    """Read Bluesky credentials from `.env`.

    Credentials never appear in code or in config/bluesky/config.yaml -- config/bluesky/config.yaml is shared
    with the team, `.env` is not. Keeping the two separate is the whole reason
    this project has both files.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    handle = os.environ.get("BSKY_HANDLE")
    password = os.environ.get("BSKY_APP_PASSWORD")

    missing = [n for n, v in (("BSKY_HANDLE", handle), ("BSKY_APP_PASSWORD", password)) if not v]
    if missing:
        raise RuntimeError(
            f"Missing {' and '.join(missing)} in .env\n"
            f"Use an App Password (Bluesky -> Settings -> Privacy and Security -> "
            f"App Passwords), not your account password. App passwords are revocable "
            f"and cannot change account settings, so a leaked one is recoverable."
        )
    return handle, password
