# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

"""Collect the denominator series: daily Bluesky activity on neutral terms.

Run it (order does not matter relative to collect_bluesky.py):
    python -m src.platforms.bluesky.collect_baseline

Writes one JSONL line per (term, day) to `paths.baseline_path`. Safe to kill
and re-run; completed cells are checkpointed.


Why this exists
---------------
Bluesky's own activity level is not constant across an eighteen-month window.
So a rise in the raw daily count of age-verification posts may be the platform
changing rather than the topic growing, and any interrupted time series or
lead-lag analysis computed on raw counts would be partly measuring Bluesky
instead of the discourse.

This was measured, not assumed. Across seven sampled months, everyday terms
DECLINE: "coffee" falls from 7,502 posts/day in Jan 2025 to 4,445 in Jul 2026;
"tired" from 6,319 to 4,583. General English chatter fell over exactly the
period in which this topic rose -- so raw counts UNDERSTATE the real increase.
The correction matters in both directions, and without it the headline timing
result is attackable in one sentence.


Why it collects counts and not posts
------------------------------------
The denominator needs a NUMBER per day, not a corpus. `hits_total` gives that
in a single request, so this whole pass costs one request per (term, day)
rather than full pagination -- about 2,730 requests against the topic
backfill's ~11,000, for a series that is just as important.

The catch, and the reason the terms in config/bluesky/config.yaml look arbitrary: `hits_total`
SATURATES AT 10,000. A high-frequency term returns the cap instead of a count,
which is useless as a denominator and, worse, looks like a perfectly flat
series. "music" returns exactly 10,000 in every month sampled. The baseline
terms were chosen to sit comfortably below that ceiling while staying
seasonally stable -- see the selection notes in config/bluesky/config.yaml.
"""

import logging
import sys
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from atproto import Client

from src.platforms.bluesky.api import call_with_retry, login_with_retry
from src.platforms.bluesky.config import Config, load_config, load_credentials
from src.platforms.bluesky.storage import Checkpoint, append_records

log = logging.getLogger("baseline")

# The server-side cap on `hits_total`, measured 2026-09-13. A value at or above
# this is a ceiling, not a count, and must not be used as a denominator.
HITS_TOTAL_CAP = 10_000


def fetch_count(client: Client, cfg: Config, term: str, day: date) -> tuple[int | None, bool]:
    """One request. Returns (hits_total, saturated).

    `limit=1` because the posts are discarded -- only the server's own count of
    matching posts is wanted. Asking for 100 would cost the same request but
    transfer a hundred times the payload for nothing.
    """
    since, until = cfg.day_window(day)
    response = call_with_retry(client.app.bsky.feed.search_posts, {
        "q": term, "limit": 1, "sort": "latest", "since": since, "until": until,
    })
    hits = getattr(response, "hits_total", None)
    return hits, bool(hits is not None and hits >= HITS_TOTAL_CAP)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    cfg.ensure_dirs()
    handle, password = load_credentials()

    client = Client()
    login_with_retry(client, handle, password)
    log.info("Logged in as %s", handle)

    cells = cfg.baseline_cells()
    checkpoint = Checkpoint(cfg.baseline_checkpoint_path)
    log.info("%s terms x %s days = %s cells (%s already done)",
             len(cfg.baseline_queries), len(cfg.days()), f"{len(cells):,}",
             checkpoint.completed_count())

    saturated_cells: list[str] = []
    missing_cells: list[str] = []
    written = 0
    started = time.monotonic()

    for i, (term, day) in enumerate(cells, start=1):
        if checkpoint.is_done(term, day):
            continue

        hits, saturated = fetch_count(client, cfg, term, day)

        if saturated:
            saturated_cells.append(f"{term} / {day}")
        if hits is None:
            # The endpoint does not guarantee hits_total. A missing value is
            # not zero, and must never be silently treated as zero -- that
            # would put an artificial trough in the denominator and an
            # artificial spike in every ratio computed from it.
            missing_cells.append(f"{term} / {day}")

        append_records(cfg.baseline_path, [{
            "term": term,
            "day": day.isoformat(),
            "hits_total": hits,
            "saturated": saturated,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }])
        written += 1

        if i % 200 == 0 or i == len(cells):
            elapsed = time.monotonic() - started
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(cells) - i) / rate / 60 if rate > 0 else 0
            log.info("[%5d/%5d] %-12s %s  hits=%-8s eta=%3.0fm",
                     i, len(cells), term, day,
                     f"{hits:,}" if hits is not None else "None", eta)

        checkpoint.mark_done(term, day, hits=hits)
        time.sleep(cfg.sleep_seconds)

    log.info("-" * 70)
    log.info("Done in %.1f min. %s rows appended to %s",
             (time.monotonic() - started) / 60, f"{written:,}", cfg.baseline_path.name)

    if saturated_cells:
        # This is a correctness problem, not a warning to note and move past.
        # A saturated cell is a ceiling reported as a measurement, and averaging
        # it into a denominator silently flattens the series.
        log.warning("%s cell(s) SATURATED the %s hits_total cap and are unusable "
                    "as denominators:", len(saturated_cells), f"{HITS_TOTAL_CAP:,}")
        for cell in saturated_cells[:10]:
            log.warning("    %s", cell)
        # Advice scaled to how bad it is. A term that saturates on most days is
        # unusable and should go; a term that clips on two days out of 608 is
        # fine with those two cells masked, and dropping the whole term would
        # throw away 606 good measurements to fix 2 bad ones.
        worst = max(Counter(c.split(" / ")[0] for c in saturated_cells).values())
        share = worst / len(cfg.days())
        if share > 0.05:
            log.warning("Worst term saturates on %.0f%% of days -- drop it from "
                        "`baseline_queries` in config/bluesky/config.yaml and re-run.", share * 100)
        else:
            log.warning("Worst term saturates on only %.1f%% of its days. Keep the "
                        "term; MASK THESE CELLS when building the denominator "
                        "(they are ceilings, not counts). Do not treat them as "
                        "real values and do not drop the whole term.", share * 100)

    if missing_cells:
        log.warning("%s cell(s) returned no hits_total at all. Treat as MISSING, "
                    "never as zero.", len(missing_cells))

    log.info("Next: src/platforms/bluesky/enrich_threads.py")


if __name__ == "__main__":
    main()
