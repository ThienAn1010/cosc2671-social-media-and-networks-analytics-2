# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

r"""Fourth collection pass: the follow graph of the corpus's active authors.

Run it AFTER collect_bluesky.py (and ideally after enrich_threads.py, so that
reply-only participants are in the seed pool too):

    python -m src.platforms.bluesky.collect_follows
    python -m src.platforms.bluesky.collect_follows --limit 50    # short test
    python -m src.platforms.bluesky.collect_follows --dry-run     # size it first

Writes `follows.jsonl` -- one record per seed author, holding that author's
outbound follow edges. Safe to kill and re-run.


Why a follow graph as well as a reply graph
-------------------------------------------
They are not the same network and they answer different questions.

A reply edge records that two people ARGUED. On a contested policy topic that
is at least as much a record of conflict as of affinity: the densest reply
edges in a corpus like this one will often join people who despise each other.
Reply structure alone therefore cannot distinguish "these communities are
segregated" from "these communities fight constantly", which are opposite
findings.

A follow edge records that someone chose to LISTEN. It is deliberate, durable,
and survives the argument being over.

The pair is what makes the echo-chamber question answerable. Segregated follow
structure with heavy cross-community replies is a world where people know the
other side and shout at it. Segregated in both is a world where they never meet.
Either is a result; neither graph alone can tell you which one you are in.


Why getFollows and not getFollowers
-----------------------------------
Following is an act the user chose. Being followed is not, and is heavily
confounded by fame. A popular account's follower list is also unbounded -- it
can run to millions -- while its follow list is bounded by its own behaviour.


The cap, and why it is a methodological choice rather than a budget one
----------------------------------------------------------------------
Follow counts among active authors in this corpus are savagely skewed.
Measured over 200 of them:

    median 133,  mean 3,647,  p90 2,258,  p99 23,123,  max 540,838

Uncapped, that is a mean of 37 requests per user -- about 34 hours for 5,000
seeds, essentially all of it spent on a handful of accounts.

    cap 1,000 -> 4.0 requests/user, 21% of users truncated
    cap 2,000 -> 5.6 requests/user, 12% truncated
    cap 5,000 -> 7.6 requests/user,  4% truncated

The cap is set at 2,000 in config/bluesky/config.yaml. Affordability is the obvious reason,
but not the good one: an account that follows 540,838 others is an aggregator
or a bot, not a participant in a community. Its edges connect everything to
everything, and in a community-detection run they smear exactly the structure
the analysis exists to find. Truncating that tail makes the graph better as
well as cheaper.

The bias it introduces is recorded rather than hidden. Every seed stores the
follow count the server reported next to how many edges were actually kept,
plus a `truncated` flag, so the affected share is a number you can quote.
`getFollows` returns most-recent-first, so a truncated seed keeps its recent
follows and loses its oldest.

Getting that reported count needs a detour worth knowing about. The obvious
source is the `subject` profile that rides along on every getFollows page --
but that is a ProfileView, and `follows_count` exists only on the
ProfileViewDetailed that getProfiles returns. Reading it off the subject
silently yields None for every seed, which leaves truncation detectable but not
measurable. So the counts are prefetched in batches of 25 before the walk
begins: about 200 extra requests against roughly 28,000, for the difference
between "12% of seeds were truncated" and "12% were truncated and here is how
much they lost".


What is stored per edge, and what is deliberately not
-----------------------------------------------------
Only identity: `did`, `handle`, `display_name`.

`getFollows` actually returns a full profile view per followed account,
including the bio. Keeping those verbatim would be the usual raw-layer
instinct, and here it is the wrong call: 5,000 seeds x up to 2,000 follows is
up to ten million records, and at a few hundred bytes each that is gigabytes of
data duplicated thousands of times over -- the same popular accounts appear in
thousands of follow lists.

Identity is enough to build the graph, and anything richer is recoverable:
`collect_profiles.py` can be pointed at the distinct followed DIDs later if
bio-based homophily turns out to be wanted for jurisdiction inference. A
followed account's profile is a current-state query that can be re-run at any
time, unlike a post, which is why this deviation from verbatim-raw is safe.
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from atproto import Client

from src.platforms.bluesky.api import call_with_retry, is_gone, login_with_retry, to_dict
from src.platforms.bluesky.config import Config, load_config, load_credentials
from src.platforms.bluesky.storage import Checkpoint, append_records, iter_raw_records

log = logging.getLogger("follows")

# getFollows page size. 100 is the documented maximum.
PAGE_SIZE = 100


def select_seeds(cfg: Config) -> tuple[list[tuple[str, str, int]], Counter]:
    """Choose whose follow lists to fetch.

    Returns ([(did, handle, n_posts)], the full posts-per-author distribution).

    Seeds are the most active authors in the corpus, because cost scales
    linearly with the seed count and activity is the best cheap proxy for
    "participates in this discourse". Measured on the first ~108,000 rows,
    a >=3-post threshold selects 14% of authors but covers 48% of all rows.

    A one-post author is not excluded because their opinion matters less. They
    are excluded because a single post is not participation in a community, and
    including them would multiply the cost of this pass severalfold to add
    nodes that mostly dangle off the graph with one edge.

    BOTH discovery passes count. An author found only by thread enrichment
    replied to the topic without ever posting about it unprompted -- on a
    conversation graph they are a real participant, and dropping them would
    bias the seed set toward people who broadcast over people who converse.
    """
    counts: Counter = Counter()
    handles: dict[str, str] = {}

    for record in iter_raw_records(cfg.raw_dir):
        author = (record["post"].get("author") or {})
        did = author.get("did")
        if not did:
            continue
        counts[did] += 1
        if did not in handles:
            handles[did] = author.get("handle") or ""

    eligible = [(d, handles.get(d, ""), n) for d, n in counts.most_common()
                if n >= cfg.follows_min_posts]
    return eligible[: cfg.follows_max_seeds], counts


def fetch_follow_counts(client: Client, cfg: Config, dids: list[str]) -> dict[str, int]:
    """Prefetch `follows_count` for the seeds, 25 at a time.

    Needed because getFollows' own `subject` is a ProfileView, which does not
    carry follow counts -- only getProfiles' ProfileViewDetailed does. Without
    this, truncation is detectable but not quantifiable: you would know a seed
    hit the cap, but not whether it lost ten follows or half a million.

    Costs about 200 requests for 5,000 seeds, against ~28,000 for the walk.
    """
    counts: dict[str, int] = {}
    batch_size = 25
    n_batches = (len(dids) + batch_size - 1) // batch_size
    for b in range(n_batches):
        batch = dids[b * batch_size : (b + 1) * batch_size]
        response = call_with_retry(client.app.bsky.actor.get_profiles, {"actors": batch})
        for profile in (getattr(response, "profiles", None) or []):
            prof = to_dict(profile)
            if prof.get("did") is not None and prof.get("follows_count") is not None:
                counts[prof["did"]] = prof["follows_count"]
        if (b + 1) % 40 == 0 or b + 1 == n_batches:
            log.info("  prefetched follow counts: %s/%s batches", b + 1, n_batches)
        time.sleep(cfg.sleep_seconds)
    return counts


def fetch_follows(client: Client, cfg: Config, did: str) -> tuple[list[dict], bool]:
    """Page through one author's follow list.

    Returns (edges, truncated).

    Only identity is kept per edge -- see the module docstring for why the full
    profile view returned by the API is deliberately discarded.
    """
    edges: list[dict] = []
    seen: set[str] = set()
    cursor = None

    while len(edges) < cfg.follows_max_per_user:
        params = {"actor": did, "limit": PAGE_SIZE}
        if cursor:
            params["cursor"] = cursor

        # Same exposure as the enrichment pass: a seed author who deleted or
        # deactivated their account between collection and now returns a 400
        # NotFound rather than an empty list. An empty follow list is the
        # correct record for them -- the node exists in the corpus, it simply
        # has no outbound edges we can observe.
        try:
            response = call_with_retry(client.app.bsky.graph.get_follows, params)
        except Exception as exc:
            if not is_gone(exc):
                raise
            return edges, False
        batch = getattr(response, "follows", None) or []
        for profile in batch:
            p = to_dict(profile)
            followed_did = p.get("did")
            # Page boundaries can repeat an account, and a follow list can
            # shift under us mid-walk if the user follows someone while we
            # page. Keeping the first occurrence makes the result stable.
            if followed_did and followed_did not in seen:
                seen.add(followed_did)
                edges.append({
                    "did": followed_did,
                    "handle": p.get("handle"),
                    "display_name": p.get("display_name"),
                })

        cursor = getattr(response, "cursor", None)
        if not cursor or not batch:
            return edges, False

        time.sleep(cfg.sleep_seconds)

    # Left the loop on the cap rather than on an exhausted cursor.
    return edges[: cfg.follows_max_per_user], True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect the follow graph of the corpus's active authors.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only fetch the first N seeds. For short test runs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the seed set and projected cost, fetch nothing.")
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

    log.info("Selecting seeds from the raw layer ...")
    seeds, distribution = select_seeds(cfg)

    eligible_total = sum(1 for n in distribution.values() if n >= cfg.follows_min_posts)
    log.info("%s distinct authors; %s have >= %s posts; taking the top %s",
             f"{len(distribution):,}", f"{eligible_total:,}",
             cfg.follows_min_posts, f"{len(seeds):,}")
    if eligible_total > cfg.follows_max_seeds:
        # Not a warning about a bug -- a statement of what was left out, so the
        # writeup can say "the follow graph covers the N most active authors"
        # rather than implying it covers everyone.
        log.info("%s eligible authors are NOT seeded (max_seeds cap). The follow "
                 "graph covers the most active %s, not the whole corpus.",
                 f"{eligible_total - cfg.follows_max_seeds:,}", f"{len(seeds):,}")

    if seeds:
        log.info("Seed activity: most active %s posts, least active %s posts",
                 seeds[0][2], seeds[-1][2])

    if args.dry_run:
        # 5.6 requests/user is the measured mean at a 2,000 cap; scale it if the
        # cap has been changed, since cost is dominated by the capped tail.
        est_per_user = 5.6 * (cfg.follows_max_per_user / 2000)
        est_requests = len(seeds) * est_per_user
        log.info("DRY RUN. Projected: ~%s requests, ~%.1f hours at the measured "
                 "1.5 req/s.", f"{est_requests:,.0f}", est_requests / 1.5 / 3600)
        log.info("Turn `follows.max_seeds` down in config/bluesky/config.yaml to shorten this.")
        return

    ordered = seeds[: args.limit] if args.limit else seeds
    checkpoint = Checkpoint(cfg.follows_checkpoint_path)
    pending = [s for s in ordered if not checkpoint.is_done("follows", s[0])]

    log.info("Fetching %s seed(s) this run (%s already done)",
             f"{len(pending):,}", checkpoint.completed_count())
    if not pending:
        log.info("Nothing to do.")
        return

    handle, password = load_credentials()
    client = Client()
    login_with_retry(client, handle, password)
    log.info("Logged in as %s", handle)

    log.info("Prefetching follow counts for %s seeds (needed to measure "
             "truncation) ...", f"{len(pending):,}")
    reported_counts = fetch_follow_counts(client, cfg, [seed[0] for seed in pending])
    log.info("Got counts for %s of %s seeds.",
             f"{len(reported_counts):,}", f"{len(pending):,}")

    total_edges = 0
    truncated_seeds = 0
    lost_edges = 0
    empty_seeds = 0
    started = time.monotonic()

    for i, (did, seed_handle, n_posts) in enumerate(pending, start=1):
        edges, truncated = fetch_follows(client, cfg, did)
        follows_count = reported_counts.get(did)

        record = {
            "_meta": {
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "kind": "follows",
                "seed_did": did,
                "seed_handle": seed_handle,
                "n_posts_in_corpus": n_posts,
                # The server's own count, kept alongside what we actually
                # stored. The gap between them IS the truncation bias, with a
                # number attached, rather than something to be estimated later.
                "follows_count_reported": follows_count,
                "n_collected": len(edges),
                "truncated": truncated,
                "max_follows_per_user": cfg.follows_max_per_user,
            },
            "follows": edges,
        }
        total_edges += append_records(cfg.follows_dir / "follows.jsonl", [record]) and len(edges)

        if truncated:
            truncated_seeds += 1
            if follows_count:
                lost_edges += max(0, follows_count - len(edges))
        if not edges:
            # A real state, not an error: plenty of accounts follow nobody, and
            # a suspended account returns nothing. Recorded either way so the
            # seed is not silently missing from the graph.
            empty_seeds += 1

        if i % 50 == 0 or i == len(pending):
            elapsed = time.monotonic() - started
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(pending) - i) / rate / 60 if rate > 0 else 0
            log.info("[%5d/%5d] %-28s edges=%5d  total=%s  eta=%3.0fm",
                     i, len(pending), (seed_handle or did)[:28], len(edges),
                     f"{total_edges:,}", eta)

        checkpoint.mark_done("follows", did, edges=len(edges), truncated=truncated)
        time.sleep(cfg.sleep_seconds)

    log.info("-" * 70)
    log.info("Done in %.1f min. %s edges from %s seeds -> %s",
             (time.monotonic() - started) / 60, f"{total_edges:,}",
             f"{len(pending):,}", cfg.follows_dir / "follows.jsonl")

    if truncated_seeds:
        log.info("%s seed(s) (%.0f%%) hit the %s-follow cap, dropping %s edges "
                 "between them. Quote both figures in limitations: the lost "
                 "follows are the OLDEST ones, since getFollows returns "
                 "most-recent-first.",
                 f"{truncated_seeds:,}", 100 * truncated_seeds / len(pending),
                 f"{cfg.follows_max_per_user:,}", f"{lost_edges:,}")
    if empty_seeds:
        log.info("%s seed(s) returned no follows (follow nobody, or the account "
                 "is gone).", f"{empty_seeds:,}")

    log.info("Raw layer now holds posts, replies, profiles and follow edges.")


if __name__ == "__main__":
    main()
