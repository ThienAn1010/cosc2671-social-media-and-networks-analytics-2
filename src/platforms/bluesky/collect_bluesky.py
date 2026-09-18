# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

"""Collect age-verification posts from Bluesky into the raw layer.

Walks a grid of (query x day) cells. For each cell it pages through
`app.bsky.feed.search_posts` until the result set is exhausted, wraps each post
in a metadata envelope, and appends it to that day's JSONL file.

Run it:
    python -m src.platforms.bluesky.collect_bluesky
    python -m src.platforms.bluesky.collect_bluesky --days 3      # short test
    python -m src.platforms.bluesky.collect_bluesky --stratum A   # backbone only

It is safe to kill this and re-run it. Completed cells are checkpointed, so a
restart resumes rather than starting over.

Adapted from the Assignment 1 collector, which ran the same grid design over a
seven-week window. Three things changed for this project, all consequences of
running over eighteen months on a policy topic rather than seven weeks on a
product launch. They are marked CHANGED FROM A1 below.


Two design decisions worth understanding
----------------------------------------

**1. Why daily windows instead of one long paginated run?**

The SDK warns that a cursor "may not necessarily allow scrolling through the
entire result set" -- a single deep run over eighteen months can stop early and
return a partial corpus that looks complete. One day of one query is a shallow
enough result set to exhaust reliably, and it gives a natural unit to
checkpoint.

**2. Why NOT skip posts already collected under a different query?**

The obvious optimisation is to load every URI already collected and skip them.
It is wrong here. The same post can match "age verification", "Online Safety
Act" and "verify your age", and knowing WHICH queries matched is what lets the
yield-by-query analysis tell you which query earns its place and which is
dragging in false positives. More importantly, it is what keeps stratum A
separable from stratum B -- and the entire frame analysis rests on that
separation.

So we deduplicate WITHIN a cell (page boundaries can repeat a post) and let the
same post appear once per matching query across cells. Normalisation merges
those into one row with a list of matched queries. Raw is a log, not a set.
"""

import argparse
import collections
import logging
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

from atproto import Client

from src.platforms.bluesky.api import call_with_retry, login_with_retry, to_dict
from src.platforms.bluesky.config import Config, load_config, load_credentials
from src.platforms.bluesky.storage import Checkpoint, append_records, raw_path_for_day

log = logging.getLogger("collect")

# CHANGED FROM A1: the shortfall test now requires the gap to be large in BOTH
# relative and absolute terms.
#
# A1 warned whenever a cell returned less than 90% of the hits the server
# reported. That works on a consumer-launch topic, where every cell is large.
# It misfires constantly here: this topic has genuinely quiet days, and on a
# 4-hit cell a single deleted post is a 25% shortfall. The first probe run
# emitted nine such warnings ("got 3 of 4 reported hits"), all noise -- and a
# warning that cries wolf is worse than no warning, because it trains you to
# scroll past the real one.
#
# Measured attrition on the live API is 1-3% per cell (deleted, blocked or
# moderated posts the index still counts), so 0.90 remains the right relative
# threshold. The absolute floor is what suppresses small-cell noise.
SHORTFALL_RATIO = 0.90
SHORTFALL_MIN_GAP = 10

# MEASURED 2026-09-13 on the completed backfill. An earlier reading of this
# said `hits_total` was unreliable for the query "social media ban". THAT WAS
# WRONG, and the corrected version matters because it changes what the gap is:
#
# The shortfall is CELL-specific and perfectly DETERMINISTIC, not query-wide and
# not transient. Same query, two different days, three consecutive runs each:
#
#     social media ban / 2025-10-24 -> 53 of 264 hits (20%), three times out of three
#     social media ban / 2025-12-10 -> 1,621 of 1,711  (95%), three times out of three
#
# On the affected cells the backend serves ONE short page and then withdraws the
# cursor, declaring the result set exhausted while still reporting a much larger
# `hits_total`. Our page cap is never reached, so this is not truncation we
# caused, and because it reproduces exactly it is not something a re-run fixes.
# Those posts are simply not servable.
#
# Scale on the finished corpus: 150 of 7,273 cells (2.1%) tripped the test, and
# roughly 3% of indexed posts across the run could not be retrieved.
#
# The practical rule: a shortfall is a DIAGNOSTIC, not a survivorship estimate.
# Before treating any single gap as data loss, re-run that one cell -- most
# large cells re-verify at 95-97%, which is ordinary attrition (posts deleted,
# blocked or moderated between indexing and retrieval). Attrition is therefore
# reported per query, so a concentrated anomaly stays visible instead of being
# averaged into one plausible-looking headline number.
SUSPECT_RATIO = 0.75


def fetch_cell(client: Client, cfg: Config, query: str, day: date) -> tuple[list, int, int | None, bool]:
    """Page through one (query, day) cell.

    Returns (posts, pages_fetched, hits_total, truncated).
    """
    since, until = cfg.day_window(day)
    posts: list = []
    seen_in_cell: set[str] = set()
    cursor, pages, hits_total = None, 0, None

    while pages < cfg.max_pages_per_cell:
        params = {
            "q": query,
            "limit": 100,          # max page size; cost is per request, not per post
            "sort": "latest",      # NOT "top" -- see below
            "since": since,
            "until": until,
        }
        # "top" is engagement-ranked. Sampling by engagement would bias the
        # corpus toward whichever sentiment travels furthest, which is exactly
        # the variable we are trying to measure. "latest" is chronological and
        # therefore unbiased with respect to sentiment.
        if cursor:
            params["cursor"] = cursor

        response = call_with_retry(client.app.bsky.feed.search_posts, params)
        pages += 1
        batch = response.posts or []

        if hits_total is None:
            hits_total = getattr(response, "hits_total", None)

        # Page boundaries can repeat a post; keep the first occurrence.
        for post in batch:
            if post.uri not in seen_in_cell:
                seen_in_cell.add(post.uri)
                posts.append(post)

        cursor = getattr(response, "cursor", None)
        if not cursor or not batch:
            return posts, pages, hits_total, False

        time.sleep(cfg.sleep_seconds)

    return posts, pages, hits_total, True


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect age-verification posts from Bluesky.")
    parser.add_argument("--days", type=int, default=None,
                        help="Only collect the first N days. For short test runs.")
    parser.add_argument("--stratum", choices=["A", "B"], default=None,
                        help="Only collect one stratum. 'A' is the neutral backbone.")
    args = parser.parse_args()

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

    days = cfg.days()[: args.days] if args.days else cfg.days()
    queries = {"A": cfg.queries_a, "B": cfg.queries_b}.get(args.stratum, cfg.queries)
    cells = [(q, d) for q in queries for d in days]
    checkpoint = Checkpoint(cfg.checkpoint_path)

    log.info("%s queries x %s days = %s cells (%s already done)",
             len(queries), len(days), f"{len(cells):,}", checkpoint.completed_count())
    log.info("Window %s .. %s", cfg.date_start, cfg.date_end)

    written_total = 0
    unavailable = collections.Counter()   # per query: indexed but not served
    reported = collections.Counter()      # per query: what the server claimed
    empty_cells: list[str] = []
    truncated_cells: list[str] = []
    started = time.monotonic()

    for i, (query, day) in enumerate(cells, start=1):
        if checkpoint.is_done(query, day):
            continue

        since, until = cfg.day_window(day)
        posts, pages, hits_total, truncated = fetch_cell(client, cfg, query, day)

        # Wrap each post in an envelope. The payload is verbatim; the envelope
        # records the circumstances of collection.
        #
        # This is not decoration. `collected_at`, `query` and `stratum` cannot
        # be recovered later -- normalisation runs days afterwards and has no
        # way to know when a post was fetched or which query found it. Without
        # the envelope, those fields would have to be invented at normalisation
        # time, and normalisation would stop being reproducible.
        collected_at = datetime.now(timezone.utc).isoformat()
        records = [
            {
                "_meta": {
                    "collected_at": collected_at,
                    # "search" = this post's own text matched `query`.
                    # enrich_threads.py writes "thread" for posts found by
                    # walking a reply tree, which matched no query at all.
                    # Keeping the two separable is what lets the analysis
                    # measure how much the sampling method shaped the result.
                    "discovery": "search",
                    "query": query,
                    # CHANGED FROM A1: the stratum is recorded per record.
                    # It is derivable from `query` plus config/bluesky/config.yaml today, but
                    # config/bluesky/config.yaml will change and raw files must stay
                    # self-describing. The frame analysis depends on being able
                    # to isolate stratum A years from now.
                    "stratum": cfg.stratum_of(query),
                    "since": since,
                    "until": until,
                },
                "post": to_dict(post),
            }
            for post in posts
        ]
        written = append_records(raw_path_for_day(cfg.raw_dir, day), records)
        written_total += written

        # Every cell is logged, including empty ones. A zero-result day that
        # scrolls past silently is indistinguishable from a day that was never
        # attempted -- and that is the bug you find three weeks later.
        flag = ""
        if not posts:
            empty_cells.append(f"{query} / {day}")
            flag = "   <- EMPTY"
        # Two very different situations, only one of which is a problem.
        #
        # Hitting the page cap IS truncation: pagination stopped before the
        # result set did, and posts are missing.
        #
        # A small gap between hits_total and what we collected is NOT. The
        # search index counts posts the API then declines to serve -- deleted,
        # blocked or moderated between indexing and retrieval.
        if truncated:
            truncated_cells.append(f"{query} / {day}: hit the {cfg.max_pages_per_cell}-page cap")
            flag += f"   <- TRUNCATED at {cfg.max_pages_per_cell} pages"
        elif (hits_total and len(posts) < hits_total * SHORTFALL_RATIO
                and hits_total - len(posts) >= SHORTFALL_MIN_GAP):
            truncated_cells.append(
                f"{query} / {day}: got {len(posts)} of {hits_total} reported hits")
            flag += f"   <- SHORTFALL: only {len(posts)}/{hits_total} hits retrieved"
        if hits_total:
            reported[query] += hits_total
            if hits_total > len(posts):
                # Counted per query for the survivorship figure, not warned.
                unavailable[query] += hits_total - len(posts)

        # CHANGED FROM A1: an ETA. A1's grid was 350 cells and finished while
        # you watched; this one is 6,552 and runs for over an hour, which makes
        # "is this progressing or wedged?" a real question.
        done = i
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed > 0 else 0
        eta_min = (len(cells) - done) / rate / 60 if rate > 0 else 0
        log.info("[%5d/%5d] %-26s %s  posts=%4d  pages=%2d  eta=%3.0fm%s",
                 i, len(cells), query, day, len(posts), pages, eta_min, flag)

        checkpoint.mark_done(query, day, posts=len(posts), pages=pages, at=collected_at)
        time.sleep(cfg.sleep_seconds)

    log.info("-" * 70)
    log.info("Done in %.1f min. %s records appended to %s",
             (time.monotonic() - started) / 60, f"{written_total:,}", cfg.raw_dir)
    log.info("Cells completed: %s", checkpoint.completed_count())

    if empty_cells:
        log.warning("%s cell(s) returned nothing:", len(empty_cells))
        for cell in empty_cells[:20]:
            log.warning("    %s", cell)
        if len(empty_cells) > 20:
            log.warning("    ... and %s more", len(empty_cells) - 20)

    if reported:
        # Evidence, not an error. The search index counted these posts but the
        # API would not serve them -- deleted, blocked or moderated before we
        # got there. This is survivorship bias with a number attached.
        #
        # Reported per query, never as one total: see SUSPECT_RATIO above.
        log.info("Retrieval against what the server reported, by query:")
        suspect = []
        for query in sorted(reported, key=lambda q: -reported[q]):
            claimed = reported[query]
            got = claimed - unavailable[query]
            ratio = got / claimed if claimed else 1.0
            flag = ""
            if ratio < SUSPECT_RATIO:
                flag = "   <- concentrated shortfall; re-run a cell before believing it"
                suspect.append(query)
            log.info("    %-26s %9s / %9s  = %3.0f%%%s",
                     query, f"{got:,}", f"{claimed:,}", ratio * 100, flag)
        trusted_claimed = sum(reported[q] for q in reported if q not in suspect)
        trusted_got = trusted_claimed - sum(unavailable[q] for q in unavailable
                                            if q not in suspect)
        if trusted_claimed:
            log.info("Attrition across the queries whose hits_total is trustworthy: "
                     "%.1f%% of indexed posts could not be retrieved (deleted, "
                     "blocked or moderated since indexing). THIS is the figure to "
                     "quote in limitations.",
                     100 * (1 - trusted_got / trusted_claimed))
        if suspect:
            log.warning("EXCLUDED from that figure as anomalous: %s. Re-run one of "
                        "their shortfall cells to tell a server-side pagination "
                        "failure (reproduces exactly) from ordinary attrition "
                        "(re-verifies at 95-97%%).", ", ".join(suspect))

    if truncated_cells:
        log.warning("%s cell(s) returned materially less than the server reported:",
                    len(truncated_cells))
        for cell in truncated_cells[:20]:
            log.warning("    %s", cell)
        log.warning("Raise max_pages_per_cell in config/bluesky/config.yaml, or split those "
                    "queries into narrower windows.")

    log.info("Next: src/platforms/bluesky/collect_baseline.py, then src/platforms/bluesky/enrich_threads.py, "
             "then src/platforms/bluesky/collect_profiles.py")


if __name__ == "__main__":
    main()
