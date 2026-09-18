# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

r"""Second collection pass: fetch the replies to posts we already collected.

Run it AFTER collect_bluesky.py:
    python -m src.platforms.bluesky.enrich_threads
    python -m src.platforms.bluesky.enrich_threads --limit 20   # short test run

Why this pass exists
--------------------
`search_posts` returns posts whose OWN text matches a query. So this happens:

    [top-level] "New age checks start today, here is what changes"  -> collected
       |- reply "just use a VPN, takes thirty seconds"              -> INVISIBLE
       |- reply "so I have to send my passport to watch anything?"  -> INVISIBLE

In Assignment 1 that cost sentiment coverage, because replies carry reaction
while top-level posts skew announcement-flavoured.

HERE IT COSTS THE NETWORK ITSELF. A reply graph built from search results alone
has almost no edges, because the replies that FORM the edges are exactly the
posts search cannot see. Both circumvention claims the project wants to make --
how bypass knowledge spreads, and whether privacy and child-safety communities
actually talk to each other -- are claims about edges. Without this pass there
is no graph to analyse, only a pile of disconnected nodes.

This pass retrieves posts by their POSITION IN A CONVERSATION rather than by
their text, which is why it reaches wording no query could match.

It must run at collection time. Replies are not in the search response and
cannot be recovered later, unlike anything already inside a saved payload.

What it costs you
-----------------
It introduces a bias of its own, in the opposite direction: only posts that
already have replies contribute any, so busy threads get amplified. That is why
every row it writes is tagged `discovery="thread"` -- so the analysis can
measure the effect instead of inheriting it silently. A bias you can name and
quantify beats a hole in the sample you cannot see.

A second, subtler gap this pass CANNOT close: a threadgated post restricts who
may reply at all. Its missing edges are a deliberate act by the author, not a
sampling failure, and the two are indistinguishable once the data is flat. The
`threadgate` field is preserved verbatim in the raw payload by collect_bluesky
so the graph builder can tell them apart.
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

log = logging.getLogger("enrich")


def find_candidates(cfg) -> dict[str, dict]:
    """Pick the collected posts worth fetching a reply tree for.

    Returns {uri: {"reply_count", "day", "query"}}.

    Only search-discovered posts are considered. Enriching a post that was
    itself found by enrichment would walk outward through the network
    indefinitely and quietly turn a topic corpus into a general Bluesky sample.
    """
    candidates: dict[str, dict] = {}

    for record in iter_raw_records(cfg.raw_dir):
        meta = record["_meta"]
        # Rows written before `discovery` existed are search rows by definition.
        if meta.get("discovery", "search") != "search":
            continue

        post = record["post"]
        uri = post.get("uri")
        if not uri:
            continue

        reply_count = post.get("reply_count") or 0
        existing = candidates.get(uri)
        # The same post appears once per query that matched it, and its reply
        # count drifts upward between those collections. Keep the highest --
        # it is the most recent view of how busy the thread is.
        if existing is None or reply_count > existing["reply_count"]:
            candidates[uri] = {
                "reply_count": reply_count,
                # `since` is the day window this post was collected in. Using it
                # avoids parsing any date out of the payload, keeping this pass
                # free of interpretation, same as the collector. The fallback
                # only affects which file a reply lands in, never its content --
                # normalisation re-derives every date from the payload anyway.
                "day": (meta.get("since") or "unknown")[:10],
                "query": meta.get("query"),
                # Carried so an enriched reply can be traced back to the
                # stratum of the post that surfaced it. A reply reached through
                # a stratum B (frame-flavoured) parent inherits that selection
                # bias, and the frame analysis has to be able to see that.
                "stratum": meta.get("stratum"),
            }

    return {
        uri: info
        for uri, info in candidates.items()
        if info["reply_count"] >= cfg.enrich_min_reply_count
    }


def walk_replies(thread, max_replies: int) -> tuple[list, int]:
    """Flatten a reply tree into a list of posts, breadth-first.

    Returns (posts, tombstones).

    The tree is a union type: each node is a real ThreadViewPost, or a
    NotFoundPost (deleted), or a BlockedPost (author blocked the viewer). Only
    the first has a `.post`, so the other two are counted and skipped -- reading
    `.post` off them would crash.

    Those tombstones are worth counting rather than discarding: they are a
    direct, measurable read on the survivorship bias in this corpus. You can
    state exactly what share of replies had vanished by collection time.

    Breadth-first (rather than depth-first) matters because of `max_replies`:
    if a huge thread gets truncated, we keep the direct replies -- the ones
    closest to the topic -- and drop the deep tangents.
    """
    posts: list = []
    tombstones = 0
    queue = list(getattr(thread, "replies", None) or [])

    while queue and len(posts) < max_replies:
        node = queue.pop(0)
        post = getattr(node, "post", None)
        if post is None:
            tombstones += 1
            continue
        posts.append(post)
        queue.extend(getattr(node, "replies", None) or [])

    return posts, tombstones


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch replies to collected age-verification posts.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only enrich the first N threads. For short test runs.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    cfg.ensure_dirs()

    if not any(cfg.raw_dir.glob("*.jsonl")):
        raise SystemExit(f"No raw data in {cfg.raw_dir}. Run src/platforms/bluesky/collect_bluesky.py first.")

    candidates = find_candidates(cfg)
    # Sorted so a resumed run processes threads in the same order as the first.
    ordered = sorted(candidates.items())
    total_candidates = len(ordered)
    if args.limit:
        ordered = ordered[: args.limit]

    checkpoint = Checkpoint(cfg.enrich_checkpoint_path)
    # Report the true candidate count, not the post-`--limit` count -- otherwise
    # "20 posts have >= 3 replies" is just the limit echoed back, and tells you
    # nothing about how much work the full run actually faces.
    log.info("%s posts have >= %s replies; enriching %s this run (%s already done)",
             f"{total_candidates:,}", cfg.enrich_min_reply_count,
             f"{len(ordered):,}", checkpoint.completed_count())

    if not ordered:
        log.warning("Nothing to enrich. Either the corpus is small or "
                    "enrich.min_reply_count (%s) is set too high.",
                    cfg.enrich_min_reply_count)
        return

    handle, password = load_credentials()
    client = Client()
    login_with_retry(client, handle, password)
    log.info("Logged in as %s", handle)

    total_replies, total_tombstones, capped_threads = 0, 0, 0
    gone_count = 0

    for i, (uri, info) in enumerate(ordered, start=1):
        if checkpoint.is_done("thread", uri):
            continue

        # A deleted root arrives one of two ways, and BOTH are normal:
        #   * as a tombstone in the response (handled just below), or
        #   * as a 400 NotFound (handled here).
        # The second form killed three runs before it was caught, because an
        # exception looks like a bug while a tombstone looks like data.
        try:
            response = call_with_retry(
                client.app.bsky.feed.get_post_thread,
                {"uri": uri, "depth": cfg.enrich_depth, "parent_height": 0},
            )
        except Exception as exc:
            if not is_gone(exc):
                raise
            gone_count += 1
            log.info("[%4d/%4d] %s  -> root gone (404 at fetch time)", i, len(ordered), uri[-24:])
            checkpoint.mark_done("thread", uri, replies=0, gone=True)
            continue

        thread = getattr(response, "thread", None)
        # The root itself can be a tombstone: deleted between collection and now.
        if thread is None or getattr(thread, "post", None) is None:
            gone_count += 1
            log.info("[%4d/%4d] %s  -> root gone (deleted or blocked)", i, len(ordered), uri[-24:])
            checkpoint.mark_done("thread", uri, replies=0, gone=True)
            continue

        replies, tombstones = walk_replies(thread, cfg.enrich_max_replies)
        total_tombstones += tombstones
        capped = len(replies) >= cfg.enrich_max_replies
        if capped:
            capped_threads += 1

        collected_at = datetime.now(timezone.utc).isoformat()
        records = [
            {
                "_meta": {
                    "collected_at": collected_at,
                    # These posts matched NO query -- they were found by position
                    # in a conversation. Normalisation leaves query_matched empty
                    # for them, so they never contaminate the per-query analysis.
                    "discovery": "thread",
                    "parent_uri": uri,
                    "parent_query": info["query"],
                    "parent_stratum": info.get("stratum"),
                    "depth": cfg.enrich_depth,
                },
                "post": to_dict(reply),
            }
            for reply in replies
        ]
        written = append_records(raw_path_for_day(cfg.raw_dir, info["day"], prefix="thread"), records)
        total_replies += written

        flag = ""
        if capped:
            flag = f"   <- CAPPED at {cfg.enrich_max_replies}"
        if tombstones:
            flag += f"   ({tombstones} deleted/blocked)"
        log.info("[%4d/%4d] %s  replies=%3d  (thread had %s)%s",
                 i, len(ordered), uri[-24:], written, info["reply_count"], flag)

        checkpoint.mark_done("thread", uri, replies=written, tombstones=tombstones)
        time.sleep(cfg.sleep_seconds)

    log.info("-" * 60)
    log.info("Done. %s replies appended to %s", f"{total_replies:,}", cfg.raw_dir)
    log.info("Threads enriched: %s", checkpoint.completed_count())

    if gone_count:
        # Root posts that vanished between collection and enrichment. Evidence,
        # not error: this is survivorship bias measured at the thread level,
        # and it is the figure to pair with the post-level attrition rate.
        log.info("%s root post(s) had been deleted by enrichment time and could "
                 "not be fetched. Quote alongside the post-level attrition "
                 "figure in limitations.", f"{gone_count:,}")

    if total_tombstones:
        # Not an error -- evidence. Quote this figure in the limitations section.
        log.info("%s replies were already deleted or blocked at collection time. "
                 "That is your survivorship bias, measured.", f"{total_tombstones:,}")
    if capped_threads:
        log.warning("%s thread(s) hit the %s-reply cap and were truncated.",
                    capped_threads, cfg.enrich_max_replies)

    log.info("Next: src/platforms/bluesky/collect_profiles.py")


if __name__ == "__main__":
    main()
