# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

"""Pre-flight check. Run this BEFORE building the real collector.

    python -m src.platforms.bluesky.probe

It writes nothing to data/. It only reads.

Adapted from the Assignment 1 probe, which asked two questions. This project
needs four, because it differs from A1 in one decisive way: A1 collected a
SEVEN-WEEK window that had ended days earlier, while this project needs
EIGHTEEN MONTHS reaching back to January 2025. Everything below follows from
that difference.

1. HOW FAR BACK DOES SEARCH ACTUALLY REACH?
   Bluesky's search backend has been rebuilt more than once and has had
   indexing gaps. If `since=2025-01-01` quietly returns a thin result set
   rather than a complete one, every downstream time series inherits a hole
   that looks exactly like "nobody was talking about it yet". This is the
   single biggest feasibility risk in the Bluesky workstream, and it decides
   whether events E1 and E2 are reachable at all.

   The test: run one high-volume, temporally stable control term at monthly
   intervals backwards. A control term is used rather than a topic term
   precisely because its true volume should be roughly flat -- so a cliff in
   the measured series is an INDEX artefact, not news. A topic term cannot
   distinguish the two.

   MEASURED 2026-09-13, and the result changed how this test should be read:
   `hits_total` SATURATES AT EXACTLY 10,000. Every month from 2024-11 to
   2026-08 reported 10,000 for the control term, which is a server-side cap on
   the reported count, not a measurement. So this test cannot detect a gradual
   decline -- it degenerates into a floor check: "were there at least 10,000
   indexed posts that day?". That is still worth having (a genuinely missing
   month would fall below the cap), but it is much weaker than designed, and
   the real evidence for index depth comes from `probe_early_window()` below,
   which paginates topic queries to exhaustion on early-2025 days and gets
   small, precise, non-capped counts back.

2. IS THERE ENOUGH TOPIC VOLUME, AND WHERE?
   Per candidate query, sampled around each of the five policy events plus
   quiet baseline days. Age verification is a policy story, not a consumer
   tech story, so volume is not guaranteed the way iOS 27's was.

3. ARE THE FIELD NAMES STILL WHAT THE A1 FLATTENERS EXPECT?
   This project runs atproto 0.0.72; A1's flatteners were written against
   0.0.69. The SDK returns snake_case (`like_count`) while the HTTP API docs
   show camelCase (`likeCount`), so guessing wrong produces a corpus of nulls.
   We print a real object and read the names off it.

4. WHAT JURISDICTION SIGNAL EXISTS, IF ANY?
   Bluesky has no country field. Section 3.1 of the planning discussion left
   the jurisdiction-tagging METHOD open, to be decided at analysis time -- but
   that is only safe if collection preserves the raw material for every option.
   This probe measures how much of that material actually exists: how many
   authors have a custom-domain handle, whether profile records carry anything
   location-shaped, and what the declared language tags look like.
"""

import json
import re
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from atproto import Client
from dotenv import load_dotenv

from src.platforms.bluesky.api import call_with_retry, login_with_retry, to_dict
from src.platforms.bluesky.config import PROJECT_ROOT

# Pacing. Bluesky allows 3000 requests / 5 min per IP = 10/sec. A1 measured
# ~0.30s of network time per request and settled on a 0.05s sleep (~2.9 req/s,
# 3.5x under the limit) with the 429 backoff in src/platforms/bluesky/api.py as the real
# safety net. No reason to be greedier here -- this is a probe, not the backfill.
SLEEP = 0.05
MAX_PAGES = 40

# --- Question 1: index depth -----------------------------------------------
# A term that is common, boring, and temporally stable. Its true daily volume
# on an English-language platform should not swing much month to month, so any
# cliff in the measured series is the index, not the world.
INDEX_PROBE_TERM = "the"
# Walk back from a recent complete month to before the study window opens.
INDEX_PROBE_MONTHS = 22

# --- Question 2: topic volume ----------------------------------------------
# Sample days, chosen to span the event timeline rather than to flatter it.
# Each event gets its own day plus a day shortly after (discourse peaks are
# rarely same-day), and two deliberately quiet days establish a baseline.
# If the quiet days are near zero, the continuous daily series that the timing
# analysis needs will be mostly zeros, which is worth knowing now.
SAMPLE_DAYS = [
    ("E1  Paxton ruling",        date(2025, 6, 27)),
    ("E1+ day after",            date(2025, 6, 28)),
    ("baseline  quiet Sep 2025", date(2025, 9, 17)),
    ("E2  UK OSA enforcement",   date(2025, 7, 25)),
    ("E2+ day after",            date(2025, 7, 26)),
    ("E3  AU under-16 ban",      date(2025, 12, 10)),
    ("E3+ day after",            date(2025, 12, 11)),
    ("E4  Lords VPN vote",       date(2026, 1, 21)),
    ("baseline  quiet Feb 2026", date(2026, 2, 18)),
    ("E5  AU adult content",     date(2026, 3, 11)),
]

# Two deliberately separate strata, mirroring the A1 design.
#
# STRATUM A is the neutral policy backbone: statute names, institutions, and
# the mechanism itself. It names no frame, so whatever people choose to say
# about age verification is what comes back. Frame-share analysis must rest on
# THIS stratum alone, or the finding is an artefact of the query list.
#
# STRATUM B is frame- and bypass-flavoured. It exists to GUARANTEE COVERAGE of
# the privacy and circumvention conversations, not to measure how common they
# are. Querying "surveillance" and then reporting that the surveillance frame
# is prevalent would be circular. Every row records which query found it, so
# the two strata stay separable downstream.
QUERIES_A = [
    "age verification",
    "age assurance",
    "Online Safety Act",
    "age check",
    "under-16 ban",
    "Ofcom",
    "eSafety",
]
QUERIES_B = [
    "age verification VPN",
    "age verification privacy",
    "porn ID check",
]

# --- Question 4: jurisdiction signal ---------------------------------------
# Handle suffixes worth counting. A custom domain is self-assigned and
# unverified, but it is free and already in every payload.
COUNTRY_TLDS = (".uk", ".au", ".ie", ".nz", ".scot", ".wales", ".london")


def iso_z(day: date) -> str:
    return f"{day.isoformat()}T00:00:00Z"


def day_window(day: date) -> tuple[str, str]:
    """Midnight on `day` to midnight on the following day, UTC.

    Next-midnight rather than 23:59:59 on purpose: if the API treats `until` as
    inclusive we collect a boundary post twice (harmless, deduplicated later),
    and if exclusive we collect it exactly once. A one-second gap every day
    would be neither harmless nor visible. Prefer the error you can clean up
    over the error you cannot see.
    """
    return iso_z(day), iso_z(day + timedelta(days=1))


def fetch_cell(client, query, since, until, max_pages=MAX_PAGES):
    """Page through one (query, window) cell to exhaustion.

    Returns (posts, pages, hit_page_cap, hits_total).

    `hits_total` is the server's own count of matching posts. When present it
    is a direct truncation check: collecting 100 posts when the server reports
    340 means pagination stopped early and two thirds of the cell was lost
    silently.
    """
    posts, cursor, pages, hits_total = [], None, 0, None

    while pages < max_pages:
        params = {
            "q": query,
            "limit": 100,
            # NOT "top". Engagement ranking would bias the sample toward
            # whichever sentiment travels furthest -- the variable this project
            # is trying to measure. "latest" is chronological and therefore
            # neutral with respect to it.
            "sort": "latest",
            "since": since,
            "until": until,
        }
        if cursor:
            params["cursor"] = cursor

        response = call_with_retry(client.app.bsky.feed.search_posts, params)
        pages += 1
        batch = response.posts or []
        posts.extend(batch)

        if hits_total is None:
            hits_total = getattr(response, "hits_total", None)

        cursor = getattr(response, "cursor", None)
        if not cursor or not batch:
            return posts, pages, False, hits_total
        time.sleep(SLEEP)

    return posts, pages, True, hits_total


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Question 3: field names
# ---------------------------------------------------------------------------

def probe_fields(client):
    rule("Q3.  FIELD NAMES  (does atproto 0.0.72 still match the A1 flatteners?)")
    sample = client.app.bsky.feed.search_posts({"q": "age verification", "limit": 1})
    if not sample.posts:
        print("WARNING: search returned nothing at all. Check the session or the query.")
        return None

    post = to_dict(sample.posts[0])
    print(json.dumps(post, indent=2, ensure_ascii=False, default=str)[:2600])

    # The exact field paths flatten_bluesky() reads. Checking them explicitly
    # beats eyeballing a JSON dump, because a renamed field shows up as a null
    # column three weeks later rather than as an error today.
    record = post.get("record") or {}
    expected = {
        "post.uri": post.get("uri"),
        "post.author.did": (post.get("author") or {}).get("did"),
        "post.author.handle": (post.get("author") or {}).get("handle"),
        "post.indexed_at": post.get("indexed_at"),
        "post.like_count": post.get("like_count"),
        "post.reply_count": post.get("reply_count"),
        "post.repost_count": post.get("repost_count"),
        "post.quote_count": post.get("quote_count"),
        "record.text": record.get("text"),
        "record.created_at": record.get("created_at"),
        "record.langs": record.get("langs"),
    }
    print()
    print("  A1 flattener field check:")
    for path, value in expected.items():
        state = "MISSING" if value is None else "ok"
        shown = "" if value is None else f"  {str(value)[:52]}"
        print(f"    [{state:>7}]  {path:<24}{shown}")
    return sample.posts[0]


# ---------------------------------------------------------------------------
# Question 1: index depth
# ---------------------------------------------------------------------------

def probe_index_depth(client):
    rule("Q1.  INDEX DEPTH  (how far back does search actually reach?)")
    print(f"Control term {INDEX_PROBE_TERM!r}, one full day sampled per month, walking backwards.")
    print("True volume should be roughly FLAT. A cliff is the index, not the news.\n")
    print(f"  {'month':<10}{'day sampled':<14}{'hits_total':>12}{'retrieved':>11}")
    print("  " + "-" * 47)

    # Anchor on the most recent complete month, then step back.
    today = datetime.now(timezone.utc).date()
    anchor = date(today.year, today.month, 1) - timedelta(days=1)

    results = []
    for i in range(INDEX_PROBE_MONTHS):
        month_end = anchor
        for _ in range(i):
            month_end = date(month_end.year, month_end.month, 1) - timedelta(days=1)
        # The 15th of that month: away from month boundaries and from the
        # 1st-of-month spikes that scheduled/automated posting creates.
        day = date(month_end.year, month_end.month, 15)

        since, until = day_window(day)
        # One page is enough: hits_total is the measure here, not the posts.
        posts, _, _, hits_total = fetch_cell(client, INDEX_PROBE_TERM, since, until, max_pages=1)
        results.append((day, hits_total, len(posts)))
        shown = f"{hits_total:,}" if hits_total is not None else "(none given)"
        print(f"  {day.strftime('%Y-%m'):<10}{day.isoformat():<14}{shown:>12}{len(posts):>11}")
        time.sleep(SLEEP)

    counts = [(d, h) for d, h, _ in results if h]
    if len(counts) >= 3:
        print()
        # `hits_total` saturates at 10,000 (measured), so this is a FLOOR check,
        # not a cliff detector. Saying so in the output matters: a reader who
        # takes "no cliff detected" as proof of full index depth would be
        # drawing a stronger conclusion than the data supports.
        CAP = 10_000
        capped = [d for d, h in counts if h >= CAP]
        below = [(d, h) for d, h in counts if h < CAP]
        if len(capped) == len(counts):
            print(f"  All {len(counts)} months report >= {CAP:,} hits -- the reported-count")
            print( "  CAP, not a real measurement. Read this as: no month is missing from")
            print( "  the index. It does NOT rule out thinning. See the early-window check.")
        else:
            print(f"  {len(capped)} of {len(counts)} months saturate the {CAP:,} cap.")
            print( "  Months reporting BELOW the cap (candidate index gaps):")
            for d, h in below:
                print(f"    {d.strftime('%Y-%m')}  {h:,}")
    return results


def probe_early_window(client):
    """Paginate real topic queries to exhaustion on early-2025 days.

    This is the test that actually answers Q1. The control-term probe above is
    capped and therefore blunt; here the counts come back small and precise
    (326, 16, 2 ...), which means they are genuine exhaustive counts rather than
    a truncated page.

    The distinction that matters: a quiet pre-enforcement period and a thin
    index look identical in a single series. They are told apart by SHAPE --
    a quiet topic still shows day-to-day variation driven by news, whereas a
    failing index decays smoothly toward zero the further back you go.
    """
    rule("Q1b.  EARLY-WINDOW REALITY CHECK  (is there real content before E1?)")
    print("  Full pagination, so these counts are exhaustive rather than one page.\n")
    print(f"  {'day':<13}{'query':<20}{'posts':>8}{'pages':>7}{'hits_total':>12}")
    print("  " + "-" * 58)

    days = [date(2025, m, 15) for m in range(1, 7)]
    for day in days:
        for query in ("age verification", "Online Safety Act"):
            since, until = day_window(day)
            posts, pages, _, hits_total = fetch_cell(client, query, since, until)
            print(f"  {day.isoformat():<13}{query:<20}{len(posts):>8}{pages:>7}"
                  f"{(f'{hits_total:,}' if hits_total is not None else '-'):>12}")
            time.sleep(SLEEP)


# ---------------------------------------------------------------------------
# Question 2: topic volume
# ---------------------------------------------------------------------------

def probe_volume(client):
    rule("Q2.  TOPIC VOLUME  (per query, across the event timeline)")

    queries = [(q, "A") for q in QUERIES_A] + [(q, "B") for q in QUERIES_B]
    header = f"  {'query':<26}{'str':<5}" + "".join(f"{lab.split()[0]:>9}" for lab, _ in SAMPLE_DAYS) + f"{'mean':>8}"
    print("  Columns are the sample days below, in this order:")
    for label, day in SAMPLE_DAYS:
        print(f"    {label.split()[0]:<6} {day.isoformat()}  {label}")
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))

    per_query = {}
    all_posts = []
    for query, stratum in queries:
        counts = []
        for _, day in SAMPLE_DAYS:
            since, until = day_window(day)
            posts, pages, capped, hits_total = fetch_cell(client, query, since, until)
            counts.append(len(posts))
            all_posts.extend(posts)
            if capped:
                print(f"    ! {query} / {day}: hit the {MAX_PAGES}-page cap -- truncated")
            # A small gap between hits_total and what we retrieved is normal
            # attrition: the index counts posts the API then declines to serve
            # (deleted, blocked, moderated since indexing). A1 measured this at
            # 1-3% per cell regardless of size. Only a large gap means
            # pagination actually gave out.
            #
            # A PROPORTIONAL TEST ALONE IS WRONG ON SMALL CELLS, which this
            # project has far more of than A1 did (policy topics have quiet
            # days; a consumer product launch does not). "3 of 4 hits" is a
            # single deleted post and trips a 10% threshold instantly. The
            # first run of this probe emitted nine such warnings, all noise.
            # So the gap must be large in BOTH relative and absolute terms.
            if hits_total and (len(posts) < hits_total * 0.90
                               and hits_total - len(posts) >= 10):
                print(f"    ! {query} / {day}: got {len(posts)} of {hits_total} "
                      f"reported hits -- pagination fell short")
            time.sleep(SLEEP)

        mean = sum(counts) / len(counts)
        per_query[query] = (stratum, counts, mean)
        print(f"  {query:<26}{stratum:<5}" + "".join(f"{c:>9}" for c in counts) + f"{mean:>8.1f}")

    return per_query, all_posts


def extrapolate(per_query):
    rule("Q2b.  EXTRAPOLATION  (what does the full backfill look like?)")
    # Study window per the proposal: Jan 2025 - Jun 2026.
    n_days = (date(2026, 6, 30) - date(2025, 1, 1)).days + 1

    a_means = [m for q, (s, c, m) in per_query.items() if s == "A"]
    all_means = [m for _, (_, _, m) in per_query.items()]

    # Summing across queries overstates the total, because one post matching
    # three queries is counted three times. Deduplication removes much of that.
    # So: sum = upper bound, single best query = lower bound.
    upper = sum(all_means) * n_days
    lower = max(all_means) * n_days if all_means else 0
    backbone = sum(a_means) * n_days

    print(f"  Window Jan 2025 - Jun 2026 = {n_days} days")
    print(f"  Sample days are EVENT-WEIGHTED (7 of 10 are event days), so these")
    print(f"  figures are an OPTIMISTIC ceiling, not an expectation:")
    print()
    print(f"    Stratum A backbone only : ~{backbone:,.0f} posts (pre-dedup)")
    print(f"    Lower bound (best query): ~{lower:,.0f} posts")
    print(f"    Upper bound (all sum)   : ~{upper:,.0f} posts (pre-dedup)")
    print()

    n_cells = len(per_query) * n_days
    # Per-cell cost: at least one request, plus the inter-cell sleep. A1
    # measured ~0.30s of network time per request on this connection.
    est_hours = n_cells * (0.30 + SLEEP) / 3600
    print(f"  Collection cost: {len(per_query)} queries x {n_days} days = {n_cells:,} cells")
    print(f"                   ~{est_hours:.1f} hours minimum, more for multi-page cells.")
    print( "                   Resumable and checkpointed, so it runs unattended.")

    if upper < 3000:
        print()
        print("  VERDICT: THIN. Widen stratum A before the backfill, or narrow the")
        print("           window to event periods and drop the continuous-series methods.")
    else:
        print()
        print("  VERDICT: adequate. Safe to proceed to the full collector.")


# ---------------------------------------------------------------------------
# Question 2c: query precision
# ---------------------------------------------------------------------------

# Policy-adjacent vocabulary. Deliberately broad: the aim is to separate "this
# post is about the age-verification debate" from "this post happens to contain
# those two words", not to classify frames. A post about the debate will almost
# always contain at least one of these.
POLICY_RE = re.compile(
    r"law|legal|bill|act\b|ban\b|regulat|ofcom|esafety|government|minor|"
    r"under.?1[68]|child|kid|teen|porn|adult site|id check|upload.{0,10}id|"
    r"face scan|vpn|privacy|surveillanc|censor|platform|social media|"
    r"age.?gat|verif|assuran",
    re.I,
)

PRECISION_CELLS = [
    ("age verification", date(2025, 1, 15)),
    ("age verification", date(2025, 7, 25)),
    ("age verification", date(2025, 9, 17)),
    ("age verification", date(2026, 3, 11)),
    ("age check", date(2025, 7, 25)),
    ("age check", date(2025, 9, 17)),
]


def probe_precision(client):
    """How much of what a query returns is actually about the topic?

    Recall is what the volume probe measures. Precision is the other half, and
    it is the half that decides whether a query belongs in the neutral backbone.
    A backbone query that drags in unrelated posts does not merely add noise --
    it silently changes the denominator of every share-based statistic computed
    from that stratum.
    """
    rule("Q2c.  QUERY PRECISION  (is what comes back actually on-topic?)")
    print("  `policy%` = share of returned posts containing any policy-adjacent term.")
    print("  Low precision on a STRATUM A query is disqualifying: the backbone is")
    print("  supposed to be the trustworthy sample.\n")
    print(f"  {'day':<13}{'query':<20}{'n':>7}{'policy%':>10}")
    print("  " + "-" * 50)

    off_topic = []
    for query, day in PRECISION_CELLS:
        since, until = day_window(day)
        posts, _, _, _ = fetch_cell(client, query, since, until)
        texts = [(to_dict(p).get("record") or {}).get("text", "") or "" for p in posts]
        if not texts:
            continue
        on = sum(1 for t in texts if POLICY_RE.search(t))
        print(f"  {day.isoformat():<13}{query:<20}{len(texts):>7}{on / len(texts):>9.0%}")
        off_topic += [(query, str(day), t) for t in texts if not POLICY_RE.search(t)][:3]
        time.sleep(SLEEP)

    print("\n  Posts matching NO policy term (inspect before trusting the number --")
    print("  a link-only post whose substance sits in the link card will land here")
    print("  and is NOT actually off-topic; the card text is collected separately):")
    for query, day, text in off_topic[:12]:
        print(f"    [{query} {day}] {text[:88].replace(chr(10), ' / ') or '(empty)'}")


# ---------------------------------------------------------------------------
# Question 4: jurisdiction signal
# ---------------------------------------------------------------------------

def probe_jurisdiction(client, posts):
    rule("Q4.  JURISDICTION SIGNAL  (what raw material exists for country tagging?)")
    if not posts:
        print("  No posts sampled; skipping.")
        return

    authors = {}
    langs = Counter()
    for post in posts:
        p = to_dict(post)
        author = p.get("author") or {}
        did = author.get("did")
        if did and did not in authors:
            authors[did] = author.get("handle") or ""
        for lang in (p.get("record") or {}).get("langs") or []:
            langs[lang] += 1

    print(f"  Sampled {len(posts):,} posts from {len(authors):,} distinct authors.\n")

    # --- signal 1: handle domain (free, already in every payload) ---
    # `.bsky.social` is the default handle; anything else is a custom domain
    # the user deliberately set, which is why a country TLD carries some signal.
    default = sum(1 for h in authors.values() if h.endswith(".bsky.social"))
    custom = len(authors) - default
    tld_hits = Counter()
    for handle in authors.values():
        for tld in COUNTRY_TLDS:
            if handle.endswith(tld):
                tld_hits[tld] += 1
    print(f"  Handle domains:")
    print(f"    default .bsky.social : {default:,} ({default / len(authors):.1%})")
    print(f"    custom domain        : {custom:,} ({custom / len(authors):.1%})")
    if tld_hits:
        print(f"    of which country-TLD : {sum(tld_hits.values()):,}")
        for tld, n in tld_hits.most_common():
            print(f"        {tld:<10} {n:,}")
    else:
        print( "    of which country-TLD : 0  -- this signal is near-useless on its own")

    # --- signal 2: declared language tag ---
    # en-GB vs en-AU would be a jurisdiction hint. A1 found the lang tag
    # unreliable in both directions (it is the posting CLIENT's declaration,
    # not a detection), so this measures whether regional variants even appear.
    print(f"\n  Declared language tags (top 8):")
    for lang, n in langs.most_common(8):
        print(f"    {lang:<10} {n:,}")
    regional = {l: n for l, n in langs.items() if "-" in l}
    if regional:
        print(f"    regional variants present: {regional}")
    else:
        print( "    no regional variants (en-GB / en-AU) present -- clients emit bare 'en'")

    # --- signal 3: does the profile record carry anything location-shaped? ---
    # This is the one that decides whether profile collection is worth a step in
    # the pipeline. getProfiles takes up to 25 DIDs per call.
    print(f"\n  Profile records (app.bsky.actor.getProfiles, sample of 25):")
    sample_dids = list(authors)[:25]
    if not sample_dids:
        return
    profiles = call_with_retry(client.app.bsky.actor.get_profiles, {"actors": sample_dids})
    plist = getattr(profiles, "profiles", None) or []
    if not plist:
        print("    none returned.")
        return

    first = to_dict(plist[0])
    print(f"    fields available: {sorted(first.keys())}")
    has_location_field = any("location" in k.lower() or "country" in k.lower() for k in first)
    print(f"    explicit location/country field: {'YES' if has_location_field else 'NO'}")

    with_desc = [to_dict(p) for p in plist]
    n_desc = sum(1 for p in with_desc if (p.get("description") or "").strip())
    print(f"    have a non-empty bio: {n_desc}/{len(with_desc)}")
    print( "    bio samples (first line, truncated):")
    for p in with_desc[:6]:
        desc = (p.get("description") or "").strip().replace("\n", " / ")
        print(f"      @{(p.get('handle') or '')[:30]:<32}{desc[:60] or '(empty)'}")


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    import os
    handle = os.environ.get("BSKY_HANDLE")
    password = os.environ.get("BSKY_APP_PASSWORD")
    if not handle or not password:
        raise SystemExit(
            "Missing BSKY_HANDLE / BSKY_APP_PASSWORD in .env\n"
            "Use an App Password (Settings -> Privacy and Security -> App Passwords), "
            "not the account password."
        )

    client = Client()
    login_with_retry(client, handle, password)
    print(f"Logged in as {handle}")
    print(f"Probe started {datetime.now(timezone.utc).isoformat(timespec='seconds')}")

    probe_fields(client)
    probe_index_depth(client)
    probe_early_window(client)
    per_query, posts = probe_volume(client)
    extrapolate(per_query)
    probe_precision(client)
    probe_jurisdiction(client, posts)

    rule("PROBE COMPLETE")
    print("  Nothing was written to data/. Decisions this informs:")
    print("    Q1 -> the start date in config/bluesky/config.yaml")
    print("    Q2 -> the final query list, and whether the window can stay daily")
    print("    Q2c -> which queries belong in the neutral backbone vs the supplement")
    print("    Q3 -> whether the A1 flatteners port unchanged")
    print("    Q4 -> whether profile collection is worth its own pipeline step")


if __name__ == "__main__":
    main()
