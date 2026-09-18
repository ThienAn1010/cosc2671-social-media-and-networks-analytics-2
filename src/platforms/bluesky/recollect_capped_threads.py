# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

r"""Repair pass: re-fetch the threads that hit the reply cap during enrichment.

    python -m src.platforms.bluesky.recollect_capped_threads --dry-run
    python -m src.platforms.bluesky.recollect_capped_threads

Why this exists
---------------
`enrich_threads.py` stops walking a reply tree at `enrich.max_replies_per_thread`
(200). The 2026-09-14 data-quality audit found 33 threads that hit it. Those
reply counts are floors, not measurements.

The bias is not random, which is what makes it worth a repair pass rather than a
footnote. A thread only exceeds 200 replies by being large and contentious --
exactly the threads a conflict, polarisation or circumvention-diffusion analysis
cares about most. Capping them truncates the tail of the distribution the
project is trying to describe, and it does so silently: a capped thread and a
thread with exactly 200 replies look identical once the data is flat.

Repairing it costs 33 API calls. Documenting around it costs a caveat in every
network result. So it gets repaired.

What it does
------------
Re-fetches each capped thread with the cap effectively lifted, then appends ONLY
the replies not already on disk for that thread. Raw stays append-only -- nothing
is rewritten or deleted -- but the thread files end up holding the union rather
than 200 plus a duplicate 200.

Two honest notes about what comes back:

* Some new replies will be genuinely NEW, posted between the original
  enrichment (2026-09-13/14) and this pass, not merely previously-capped. They
  are indistinguishable, and both are legitimate replies to a collected post.
  Rows written here carry `_meta.recollect_of_capped = true` so the analysis can
  isolate them if that distinction ever matters.

* `getPostThread` applies its own response-size limits at a given depth. If a
  thread comes back still at or near the new cap, the run says so rather than
  assuming the repair worked.
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from atproto import Client

from src.platforms.bluesky.api import call_with_retry, is_gone, login_with_retry, to_dict
from src.platforms.bluesky.config import load_config, load_credentials
from src.platforms.bluesky.storage import (
    Checkpoint,
    append_records,
    iter_raw_records,
    raw_path_for_day,
)
from src.platforms.bluesky.enrich_threads import find_candidates, walk_replies

log = logging.getLogger("recollect")

# Effectively "no cap". Kept finite so a pathological thread cannot spin forever.
UNCAPPED = 10_000


def capped_uris(checkpoint: Checkpoint, cap: int) -> list[str]:
    """Thread URIs whose recorded reply count reached the cap.

    `>=` rather than `==` because a later run with a different cap would
    otherwise be missed.

    Threads already repaired are excluded, and that exclusion is load-bearing: a
    repaired thread legitimately holds MORE than the cap (one now holds 493), so
    a count-only test would re-select the very threads it just fixed and the
    pass would never reach a fixed point. `recollected` is the only signal that
    distinguishes "truncated at 200" from "genuinely larger than 200".
    """
    done = checkpoint._data.get("thread", {})
    return sorted(uri for uri, info in done.items()
                  if isinstance(info, dict)
                  and (info.get("replies") or 0) >= cap
                  and not info.get("recollected"))


def existing_replies_for(cfg, parents: set[str]) -> dict[str, set[str]]:
    """Reply URIs already on disk, per parent, for the parents we care about.

    Scanning the raw layer once is cheaper than re-reading it per thread, and it
    is what lets this pass append only novel rows instead of a second copy of
    the first 200.
    """
    found: dict[str, set[str]] = {p: set() for p in parents}
    for record in iter_raw_records(cfg.raw_dir):
        meta = record["_meta"]
        if meta.get("discovery") != "thread":
            continue
        parent = meta.get("parent_uri")
        if parent in found:
            uri = record["post"].get("uri")
            if uri:
                found[parent].add(uri)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-fetch threads that were truncated at the reply cap.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report which threads would be re-fetched, then stop.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only repair the first N capped threads.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    cfg = load_config()
    cfg.ensure_dirs()

    checkpoint = Checkpoint(cfg.enrich_checkpoint_path)
    targets = capped_uris(checkpoint, cfg.enrich_max_replies)
    if args.limit:
        targets = targets[: args.limit]

    if not targets:
        log.info("No threads are sitting at the %s-reply cap. Nothing to repair.",
                 cfg.enrich_max_replies)
        return

    log.info("%s thread(s) hit the %s-reply cap and will be re-fetched with the "
             "cap lifted to %s.", len(targets), cfg.enrich_max_replies, UNCAPPED)

    # Routing metadata (day / query / stratum) comes from the same function the
    # original pass used, so repaired rows land in the same files under the same
    # provenance as the rows they extend.
    log.info("Scanning the raw layer for thread routing and existing replies...")
    candidates = find_candidates(cfg)
    existing = existing_replies_for(cfg, set(targets))

    missing = [u for u in targets if u not in candidates]
    if missing:
        log.warning("%s capped thread(s) are no longer search-discovered "
                    "candidates and will be skipped: %s", len(missing), missing[:3])
        targets = [u for u in targets if u in candidates]

    for uri in targets:
        log.info("  %s  on disk: %3d replies", uri[-24:], len(existing.get(uri, ())))

    if args.dry_run:
        log.info("Dry run -- nothing fetched, nothing written.")
        return

    handle, password = load_credentials()
    client = Client()
    login_with_retry(client, handle, password)
    log.info("Logged in as %s", handle)

    added_total = 0
    still_capped = 0
    gone = 0

    for i, uri in enumerate(targets, start=1):
        info = candidates[uri]
        try:
            response = call_with_retry(
                client.app.bsky.feed.get_post_thread,
                {"uri": uri, "depth": cfg.enrich_depth, "parent_height": 0},
            )
        except Exception as exc:
            if not is_gone(exc):
                raise
            gone += 1
            log.info("[%2d/%2d] %s  -> root gone since enrichment", i, len(targets), uri[-24:])
            continue

        thread = getattr(response, "thread", None)
        if thread is None or getattr(thread, "post", None) is None:
            gone += 1
            log.info("[%2d/%2d] %s  -> root gone since enrichment", i, len(targets), uri[-24:])
            continue

        replies, tombstones = walk_replies(thread, UNCAPPED)
        if len(replies) >= UNCAPPED:
            still_capped += 1

        seen = existing.get(uri, set())
        fresh = [r for r in replies if getattr(r, "uri", None) not in seen]

        collected_at = datetime.now(timezone.utc).isoformat()
        records = [
            {
                "_meta": {
                    "collected_at": collected_at,
                    "discovery": "thread",
                    "parent_uri": uri,
                    "parent_query": info["query"],
                    "parent_stratum": info.get("stratum"),
                    "depth": cfg.enrich_depth,
                    # Provenance: this row came from the repair pass, not the
                    # original enrichment. Lets the analysis separate
                    # previously-capped replies from replies posted since.
                    "recollect_of_capped": True,
                },
                "post": to_dict(reply),
            }
            for reply in fresh
        ]
        written = append_records(
            raw_path_for_day(cfg.raw_dir, info["day"], prefix="thread"), records)
        added_total += written

        total_now = len(seen) + written
        log.info("[%2d/%2d] %s  had %3d, fetched %4d, added %4d new -> %4d total%s",
                 i, len(targets), uri[-24:], len(seen), len(replies), written, total_now,
                 f"   ({tombstones} deleted/blocked)" if tombstones else "")

        checkpoint.mark_done("thread", uri, replies=total_now,
                             tombstones=tombstones, recollected=True)
        time.sleep(cfg.sleep_seconds)

    log.info("-" * 60)
    log.info("Repaired %s thread(s); %s new replies appended.",
             len(targets) - gone, f"{added_total:,}")
    if gone:
        log.info("%s root post(s) had been deleted since enrichment.", gone)
    if still_capped:
        log.warning("%s thread(s) still returned %s replies -- the server's own "
                    "response limit, not ours. Their counts remain floors.",
                    still_capped, UNCAPPED)
    else:
        log.info("No thread hit the lifted cap, so every repaired thread is now "
                 "complete to depth %s.", cfg.enrich_depth)


if __name__ == "__main__":
    main()
