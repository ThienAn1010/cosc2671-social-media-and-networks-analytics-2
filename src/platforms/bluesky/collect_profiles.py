# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

r"""Third collection pass: fetch the profile record of every author collected.

Run it AFTER collect_bluesky.py and enrich_threads.py:
    python -m src.platforms.bluesky.collect_profiles
    python -m src.platforms.bluesky.collect_profiles --limit 100   # short test

Writes `profiles_*.jsonl` into the raw layer. Safe to kill and re-run.


Why this pass exists
--------------------
Bluesky has no country field. Not on the post, not in the profile record, not
anywhere. That is a problem for this project specifically, because the research
design compares the UK against Australia with Ireland and New Zealand as
controls -- and Reddit gets that split for free from national subreddits while
Bluesky cannot.

Measured on 6,699 distinct authors during the 2026-09-13 probe, the two signals
that ARE free in the search payload are far too sparse to carry it:

    country-TLD handles (.uk, .au, ...)   1.5% of authors
    regional language tags (en-GB/en-AU)  11 posts out of 8,438 English ones

The profile record has no location field either. What it does have is a bio:
24 of 25 sampled authors had a non-empty one. Bio text and the follow graph are
therefore the only two viable routes to jurisdiction, and this pass collects
the first of them.

The team has DELIBERATELY DEFERRED the question of how (or whether) to assign
jurisdiction, until the pooled data from all platforms can be looked at
together. That deferral is only safe if collection preserves the raw material
for every option that might later be chosen. Profiles are not in the search
payload, so skipping this pass would quietly close the option for good -- and
it would close it silently, which is the worst way for an option to disappear.

This is the general rule the collection phase runs on: collect anything that
cannot be recovered later, decide later what to do with it.


What this pass does NOT settle
------------------------------
A bio is self-reported, unverified, frequently a joke, and often absent. "Based
in London" in a bio is evidence, not ground truth, and anyone using these
fields for jurisdiction must validate them against a hand-labelled sample and
report the error rate. Collecting the field commits nobody to trusting it.

One genuine caveat for the writeup: a profile is fetched TODAY, not as it stood
when the post was written. Someone who moved, rebranded, or rewrote their bio
in the intervening eighteen months is recorded as they are now. `collected_at`
is stamped on every row so the gap is at least visible. Post payloads do not
have this problem -- they were captured as they were.
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from atproto import Client

from src.platforms.bluesky.api import call_with_retry, is_gone, login_with_retry, to_dict
from src.platforms.bluesky.config import Config, load_config, load_credentials
from src.platforms.bluesky.storage import Checkpoint, append_records, iter_raw_records

log = logging.getLogger("profiles")


def find_authors(cfg: Config) -> dict[str, dict]:
    """Every distinct author DID in the raw layer, with how we met them.

    Returns {did: {"handle", "n_posts", "discovery"}}.

    BOTH collection passes are included -- search-discovered and
    thread-discovered. Enrichment-only authors are exactly the people who
    replied without ever posting a query-matching post of their own, and on a
    reply graph they are a large share of the nodes. Excluding them would leave
    most of the network without profiles.

    `discovery` records which pass first surfaced each author, so any later
    analysis can check whether reply-only participants differ systematically
    from people who post about the topic unprompted.
    """
    authors: dict[str, dict] = {}

    for record in iter_raw_records(cfg.raw_dir):
        post = record["post"]
        author = post.get("author") or {}
        did = author.get("did")
        if not did:
            continue

        discovery = record.get("_meta", {}).get("discovery", "search")
        existing = authors.get(did)
        if existing is None:
            authors[did] = {
                "handle": author.get("handle"),
                "n_posts": 1,
                "discovery": discovery,
            }
        else:
            existing["n_posts"] += 1
            # "search" wins if the author was ever found by search, since that
            # is the stronger statement about them: they posted about the topic
            # themselves rather than only replying to someone who did.
            if discovery == "search":
                existing["discovery"] = "search"

    return authors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch profile records for every collected author.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only fetch the first N authors. For short test runs.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    cfg.ensure_dirs()

    if not any(cfg.raw_dir.glob("posts_*.jsonl")):
        raise SystemExit(f"No raw posts in {cfg.raw_dir}. Run src/platforms/bluesky/collect_bluesky.py first.")

    log.info("Scanning the raw layer for authors ...")
    authors = find_authors(cfg)
    # Sorted so a resumed run processes authors in the same order as the first.
    ordered = sorted(authors.items())
    total = len(ordered)
    if args.limit:
        ordered = ordered[: args.limit]

    checkpoint = Checkpoint(cfg.profile_checkpoint_path)
    # Report the TRUE author count, not the post-`--limit` count. Otherwise the
    # number is just the limit echoed back and says nothing about the real job.
    log.info("%s distinct authors in the raw layer; fetching %s this run (%s already done)",
             f"{total:,}", f"{len(ordered):,}", checkpoint.completed_count())

    pending = [(did, info) for did, info in ordered if not checkpoint.is_done("profile", did)]
    if not pending:
        log.info("Nothing to do -- every author already has a profile.")
        return

    handle, password = load_credentials()
    client = Client()
    login_with_retry(client, handle, password)
    log.info("Logged in as %s", handle)

    batch_size = cfg.profile_batch_size
    n_batches = (len(pending) + batch_size - 1) // batch_size
    log.info("%s authors to fetch in %s batches of %s",
             f"{len(pending):,}", f"{n_batches:,}", batch_size)

    written, missing_total = 0, 0
    started = time.monotonic()

    for b in range(n_batches):
        batch = pending[b * batch_size : (b + 1) * batch_size]
        dids = [did for did, _ in batch]

        # getProfiles normally omits actors it cannot resolve, and the loop
        # below already records those as tombstones. But if the whole batch is
        # rejected because one actor in it is gone, treat the batch as all
        # tombstones rather than losing 25 authors to a crash.
        try:
            response = call_with_retry(client.app.bsky.actor.get_profiles, {"actors": dids})
            profiles = getattr(response, "profiles", None) or []
        except Exception as exc:
            if not is_gone(exc):
                raise
            log.info("batch %s rejected (an actor in it no longer exists); "
                     "recording all %s as tombstones", b + 1, len(dids))
            profiles = []
        by_did = {p.did: p for p in profiles}

        collected_at = datetime.now(timezone.utc).isoformat()
        records = []
        for did, info in batch:
            profile = by_did.get(did)
            if profile is None:
                # The account was deleted, deactivated or suspended between
                # posting and now. Recorded as an explicit tombstone rather
                # than omitted: a silently absent author is indistinguishable
                # from one this pass never reached, and the count of them is a
                # measured survivorship figure worth quoting.
                missing_total += 1
                records.append({
                    "_meta": {
                        "collected_at": collected_at,
                        "kind": "profile",
                        "n_posts_in_corpus": info["n_posts"],
                        "first_seen_via": info["discovery"],
                        "gone": True,
                    },
                    "profile": {"did": did, "handle": info.get("handle")},
                })
            else:
                records.append({
                    "_meta": {
                        "collected_at": collected_at,
                        "kind": "profile",
                        # Carried from the post layer so the profile file can be
                        # read on its own. How much of the corpus an author
                        # wrote is what decides whose profile is worth manual
                        # inspection during any jurisdiction validation.
                        "n_posts_in_corpus": info["n_posts"],
                        "first_seen_via": info["discovery"],
                        "gone": False,
                    },
                    "profile": to_dict(profile),
                })

        written += append_records(cfg.profiles_dir / "profiles.jsonl", records)
        for did, _ in batch:
            checkpoint.mark_done("profile", did)

        if (b + 1) % 20 == 0 or b + 1 == n_batches:
            elapsed = time.monotonic() - started
            rate = (b + 1) / elapsed if elapsed > 0 else 0
            eta = (n_batches - b - 1) / rate / 60 if rate > 0 else 0
            log.info("[%5d/%5d batches] %s profiles written  eta=%3.0fm",
                     b + 1, n_batches, f"{written:,}", eta)

        time.sleep(cfg.sleep_seconds)

    log.info("-" * 70)
    log.info("Done in %.1f min. %s profile rows appended.",
             (time.monotonic() - started) / 60, f"{written:,}")

    if missing_total:
        # Evidence, not an error -- the same survivorship story the post-level
        # attrition figure tells, measured at the author level instead.
        log.info("%s author(s) no longer exist (deleted, deactivated or suspended "
                 "since posting). Recorded as tombstones; quote the figure in "
                 "limitations.", f"{missing_total:,}")

    log.info("Raw layer now holds posts, replies and profiles. "
             "Follow-graph collection is the remaining pass.")


if __name__ == "__main__":
    main()
