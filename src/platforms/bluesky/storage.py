# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

"""Reading and writing the raw layer: JSONL files and the resume checkpoint.

The raw layer is **append-only and immutable**. Nothing in this project ever
edits or deletes a raw file. The reason is that collection is *unrepeatable*:
iOS 27 is a live topic, posts get deleted, engagement counts drift upward every
hour. If the schema turns out to be wrong you can fix normalisation
(report_part1_data.ipynb section 3) and re-run it in minutes -- but you cannot
re-collect June.

So raw is a **log, not a set**. It records what the API returned, when, and for
which query. It may contain the same post more than once (the same post can
match several queries, and an interrupted cell re-appends on resume). That is
fine and intentional -- deduplication happens during normalisation, where it is
cheap and reversible.

Nothing here is Bluesky-specific: the ID field is passed in as a function, so
the same code would serve any source.
"""

import json
import logging
import os
import tempfile
import time
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path
from typing import Any

log = logging.getLogger("storage")

# Windows raises PermissionError when another process holds a handle to a file.
# Cloud-sync clients (Google Drive, OneDrive, Dropbox) do exactly that, briefly,
# whenever a watched file changes -- and this project writes its checkpoint after
# every single unit of work, which makes it a frequent target. The lock clears in
# milliseconds, so a short backoff rides it out.
LOCK_ATTEMPTS = 6


def _retry_while_locked(action: Callable[[], Any], what: str) -> Any:
    """Run `action`, retrying briefly if the OS says the file is locked."""
    for attempt in range(LOCK_ATTEMPTS):
        try:
            return action()
        except PermissionError:
            if attempt == LOCK_ATTEMPTS - 1:
                raise
            wait = 0.1 * (2 ** attempt)  # 0.1s .. 3.2s, ~6s total
            log.debug("%s locked, retrying in %.1fs (attempt %s/%s)",
                      what, wait, attempt + 1, LOCK_ATTEMPTS)
            time.sleep(wait)


# --------------------------------------------------------------------------
# JSONL
# --------------------------------------------------------------------------

def raw_path_for_day(raw_dir: Path, day: date | str, prefix: str = "posts") -> Path:
    """One file per day, e.g. `posts_2026-06-08.jsonl`.

    Splitting by day (rather than one giant file) means you can open a single
    day in a text editor to eyeball it, and a corrupted write damages one day
    instead of the whole corpus.

    `prefix` separates the two collection passes on disk:
      * `posts_*.jsonl`  - found by searching for a query
      * `thread_*.jsonl` - found by walking a collected post's reply tree

    Both are read back together by `iter_raw_records`; the split is for your
    eyes, not for the code. `day` accepts a date or an ISO "YYYY-MM-DD" string.
    """
    stamp = day.isoformat() if hasattr(day, "isoformat") else str(day)
    return raw_dir / f"{prefix}_{stamp}.jsonl"


def append_records(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Append records to a JSONL file, one JSON object per line.

    JSONL rather than a single JSON array because appending to an array means
    rewriting the whole file; appending a line does not. That is what makes an
    interrupted run resumable instead of destructive.

    Returns the number of records written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialise first, so a retry never writes a half-formed batch.
    # ensure_ascii=False keeps emoji and non-Latin text readable in the file
    # rather than escaped into \uXXXX noise.
    lines = [json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records]
    if not lines:
        return 0

    def _write() -> int:
        with open(path, "a", encoding="utf-8") as fh:
            fh.writelines(lines)
        return len(lines)

    return _retry_while_locked(_write, path.name)


def iter_raw_records(raw_dir: Path) -> Iterable[dict[str, Any]]:
    """Yield every record from every JSONL file in `raw_dir`, oldest day first.

    Reads every `*.jsonl` file, so both collection passes (`posts_*` from
    search, `thread_*` from reply enrichment) come back as one stream. The
    checkpoints are `.json`, not `.jsonl`, so they are excluded automatically.

    A truncated final line (from a process killed mid-write) raises -- we would
    rather know than silently drop data.
    """
    for path in sorted(raw_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON at {path.name}:{line_no}") from exc


def load_seen_ids(raw_dir: Path, id_of: Callable[[dict[str, Any]], str]) -> set[str]:
    """Collect the IDs already present in the raw files.

    `id_of` extracts the ID from a record, so this function does not need to
    know anything about the shape of a post.

    Used by the collector to avoid re-appending posts it already has, which is
    what makes a second run produce zero new rows.
    """
    if not raw_dir.exists():
        return set()
    return {id_of(record) for record in iter_raw_records(raw_dir)}


# --------------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------------

class Checkpoint:
    """Tracks which units of work have been completed, so a run can resume.

    On restart, completed units are skipped. This is what turns "killed
    halfway through, start again from June 8th" into "killed halfway through,
    pick up where it stopped".

    Stored as nested JSON, readable by eye. Both passes use it:

        collector:  {"iOS 27": {"2026-06-08": {"posts": 12, "pages": 1}}}
        enrichment: {"thread": {"at://did:.../post/abc": {"replies": 7}}}

    The inner key is a date for the collector and a post URI for enrichment,
    so it accepts either a date or a plain string.
    """

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, dict[str, Any]] = {}
        if path.exists():
            try:
                with open(path, encoding="utf-8") as fh:
                    self._data = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                # A truncated checkpoint must not be fatal. Starting over costs
                # duplicated API calls, and the duplicate rows they append are
                # removed during normalisation -- whereas refusing to start would
                # strand a half-finished collection with no way to resume.
                log.warning("Checkpoint %s is unreadable (%s). Starting fresh -- "
                            "already-collected work will be repeated and "
                            "deduplicated during normalisation.", path.name, exc)
                self._data = {}

    @staticmethod
    def _key(unit: date | str) -> str:
        return unit.isoformat() if hasattr(unit, "isoformat") else str(unit)

    def is_done(self, group: str, unit: date | str) -> bool:
        return self._key(unit) in self._data.get(group, {})

    def mark_done(self, group: str, unit: date | str, **stats: Any) -> None:
        """Record a unit as complete and immediately persist.

        Saving after every unit (rather than once at the end) means a crash
        loses at most one unit of work.
        """
        self._data.setdefault(group, {})[self._key(unit)] = stats
        self.save()

    def completed_count(self) -> int:
        return sum(len(days) for days in self._data.values())

    def save(self) -> None:
        """Write atomically: to a temp file first, then replace.

        If the process is killed mid-write, a plain `open(...,'w')` leaves a
        half-written, unparseable checkpoint -- and the next run would crash on
        startup or, worse, decide nothing had been collected. Writing to a temp
        file and then renaming means the checkpoint is always either the old
        complete version or the new complete version, never a broken one.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._data, indent=2, ensure_ascii=False)

        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            # os.replace is atomic on Windows and POSIX alike, but on Windows it
            # fails outright if anything else holds the destination open. A
            # cloud-sync client does exactly that for a few milliseconds after
            # each change, so retry rather than abandoning a long collection run.
            _retry_while_locked(lambda: os.replace(tmp, self.path), self.path.name)
        except PermissionError:
            # Sync client is holding the file harder than expected. Falling back
            # to an in-place write gives up atomicity, which is survivable here:
            # a torn checkpoint is detected on load and the run simply restarts,
            # whereas crashing loses the entire in-progress collection.
            Path(tmp).unlink(missing_ok=True)
            log.warning("Could not atomically replace %s (file locked, most likely "
                        "by a cloud-sync client). Writing in place instead.",
                        self.path.name)
            self.path.write_text(payload, encoding="utf-8")
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
