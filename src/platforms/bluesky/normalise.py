# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

r"""Normalisation pass: turn the raw append-only log into analysis tables.

    python -m src.platforms.bluesky.normalise
    python -m src.platforms.bluesky.normalise --no-langdetect   # skip pass 3

Reads `data/raw/` and writes Parquet into `data/interim/`. Raw is never touched.

What this pass is and is not
----------------------------
It ANNOTATES. It does not exclude. Every flag it computes -- meme, spam,
duplicate text, bot, phrase-exactness -- is a column, never a deleted row.

That is deliberate and it is the core design rule of this phase. Different
analyses need different exclusions: the age-verification meme is noise for an
interrupted time series but is the object of study for a virality analysis.
Deleting it here would force one analysis's judgement onto every other, and
irreversibly. Worse, our own first meme regex caught a tenth of the wave; had
that pass deleted rows, the error would have been permanent and undetectable.

So: this script decides nothing that the team has not already agreed. It makes
the decisions POSSIBLE by measuring them. See `docs/bluesky/data_quality_audit.md` for
what each flag is for, and `docs/bluesky/collection_and_cleaning_log.md` for status.

Why four tables instead of one
------------------------------
The raw layer holds a many-to-many relation flattened into a log: one row per
(post, query that matched it). 338,637 rows describe 298,843 distinct posts,
because 32,225 posts matched more than one query.

Flattening that into a single table forces an unwinnable choice. Deduplicate by
URI and the per-query series dies, because a surviving row keeps only one query.
Keep every row and every volume, sentiment or author statistic silently
multiplies posts by how many queries happened to match them.

Splitting the relation solves it structurally rather than by convention:

    posts         one row per unique post        -> volume, sentiment, authors
    post_query    one row per (post, query)      -> per-query and per-stratum series
    thread_edges  one row per (reply, parent)    -> the reply graph
    authors       one row per account            -> account-level attributes

Join `posts` to `post_query` on `uri` when you need both. Never count rows in
`post_query` as posts.

Time axis
---------
Group by `day`. Three candidate timestamps exist and only one is sound.

`created_at` is set by the posting CLIENT and can be any value at all: 899 posts
here are dated before Bluesky existed, the earliest to 2004. A daily series
built on it would scatter those across two decades.

`indexed_at` is the platform's own index time, and is the obvious fix -- but it
is not stable. 1,750 search posts (0.59%) carry an `indexed_at` outside the very
cell window that retrieved them, one of them nine days past the end of the
study window, because the AppView re-indexes some posts after the fact. Using it
would silently move posts between days and put a 609th day in a 608-day window.

`day` therefore uses the COLLECTION CELL for search posts -- the exact window
the search predicate matched on, so the series is consistent with how the data
was gathered and lands on exactly 608 days -- and falls back to the index day
for replies, which have no cell of their own.

All three are kept (`day`, `cell_day`, `index_day`, `created_at`) along with
`index_day_drift` and `created_at_drift`, so the choice stays auditable.

Passes
------
1. Accumulate global counters (duplicate-text counts, per-author activity,
   per-day meme share). Keeps no post text, so memory stays flat.
2. Stream again and emit the four tables, deduplicating by URI on the fly.
3. Resolve language (detect where untagged, verify where the text is long
   enough), then set `is_english` -- the project works in English only.
"""

import argparse
import csv
import hashlib
import json
import logging
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.platforms.bluesky.config import load_config
from src.platforms.bluesky.storage import iter_raw_records

log = logging.getLogger("normalise")

BATCH = 50_000

# The meme template. Anchored at the start because the payload varies in every
# post ("Age verification? I bought Drakengard 1 at Blockbuster"), so exact-text
# clustering cannot find it and a substring search cannot separate it from the
# topic -- 32% of the corpus contains the phrase "age verification" somewhere,
# which measures nothing. Matching only the colon form catches a tenth of it:
# the dominant punctuation is "?".
MEME = re.compile(r"^\s*age\s*verification\b", re.I)

# A meme WAVE needs both a high share and a high volume, and the second
# condition is not padding.
#
# Share alone conflates two different things. On 2026-08-22 the meme is 33.8% of
# the day -- but the day holds 296 posts, 0.7x the median. The meme is present;
# it is not distorting anything. On 2025-07-11 it is 80.2% of 9,084 posts, 23x
# the median. That is the case the flag exists for: an interrupted time series
# would read a day like that as a real surge in discourse.
#
# So `meme_wave_day` means "this day's VOLUME is inflated by the meme", not
# "the meme appears here". Ordinary months sit at 0.3-0.9% share, so 20% is
# ~25x the floor; 2x the median day is the smallest spike worth calling a spike.
# Both underlying quantities are emitted per post (`day_meme_share`,
# `day_volume_ratio`) so the team can re-threshold without re-running this pass.
MEME_WAVE_SHARE = 0.20
MEME_WAVE_VOLUME_RATIO = 2.0

# An account repeating the SAME text this many times is broadcasting, not
# conversing. Set from the observed distribution: ordinary accounts sit in the
# low single digits, the spam account posted one string 2,361 times, and news
# bots repeat their boilerplate in the hundreds.
REPEAT_BURST_MIN = 20

# Posts dated before Bluesky's public launch cannot be genuine `created_at`
# values from this platform.
BLUESKY_EPOCH = "2023-01-01"

# How far `created_at` may sit from `indexed_at` before it is suspect. One day
# absorbs ordinary timezone and midnight-boundary effects.
CREATED_DRIFT_DAYS = 1

# Shortest text on which langdetect may CONTRADICT a declared language tag.
#
# MEASURED on 25,000 posts declaring `en`, disagreement between the tag and
# detection by text length:
#
#     <20 chars   49.1%      detection is noise
#     20-40       19.8%      still unreliable
#     40-60        3.6%
#     60-80        1.3%      <- elbow
#     80-120       0.7%
#     120+         0.3-0.5%  stable floor
#
# Below 60 characters the disagreement is the detector failing, not the tag
# lying: "cy", "af", "so", "tl" dominate, which is what langdetect returns for
# short English strings. Above it the residual ~0.5% is real -- the longest
# disagreements are genuinely Turkish, Spanish, Norwegian and German posts that
# declared `en`, because many clients send `en` by default.
#
# So detection overrules a declared tag only where detection is trustworthy, and
# the tag wins everywhere else. Neither signal is reliable enough alone.
LANG_VERIFY_MIN_CHARS = 60

# ...and detection must also be CONFIDENT before it may overrule a tag.
#
# MEASURED on the 1,519 posts that length alone would have overridden: the
# top-1 probability is sharply bimodal -- 618 sit below p=0.90 (median 82
# chars), 897 sit above p=0.999 (median 170 chars), and only 4 fall in between.
# Reading samples from each side settles which is which:
#
#   p >= 0.999  genuinely Japanese, Finnish, Chinese, Portuguese -- override
#   p <  0.999  "denmark to ban social media for under-15s" scored Danish,
#               "[ai + swearing + online safety act] #granny" scored Welsh
#
# Probability separates them where length does not: a genuinely Danish post of
# 98 characters scores 1.000, while an English one of 123 scores 0.857.
LANG_CONFIDENT_PROB = 0.999


def nfkc(text: str) -> str:
    """Fold styled and fullwidth Unicode to plain characters.

    368 posts use mathematical-alphanumeric or fullwidth letters. They are
    invisible to tokenisers and keyword filters, and they cluster in exactly the
    campaign and engagement-bait posts a discourse analysis wants to see.
    """
    return unicodedata.normalize("NFKC", text)


def norm_text(text: str) -> str:
    return " ".join(nfkc(text).lower().split())


def text_hash(normalised: str) -> str:
    return hashlib.md5(normalised.encode("utf-8")).hexdigest() if normalised else ""


def match_haystack(post: dict) -> str:
    """Text plus embed metadata, alphanumeric-folded, for phrase testing.

    Bluesky search matches against link titles and descriptions too, not only
    the post text, so a phrase test that reads `record.text` alone would report
    false negatives.
    """
    record = post.get("record") or {}
    parts = [record.get("text") or ""]
    embed = post.get("embed")
    if embed:
        parts.append(json.dumps(embed, ensure_ascii=False))
    return re.sub(r"[^a-z0-9]+", " ", nfkc(" ".join(parts)).lower())


def fold_query(query: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()


def day_of(ts):
    return ts[:10] if ts else None


def filed_day_of(post: dict, fallback: str) -> str:
    """The platform's index date, which is the axis every series should use."""
    return day_of(post.get("indexed_at")) or fallback


def source_of(path: Path) -> str:
    return "search" if path.name.startswith("posts_") else "thread"


def iter_layer(raw_dir: Path, prefix: str):
    """Yield (record, file_day) for one layer, in day order."""
    for fp in sorted(raw_dir.glob(f"{prefix}_*.jsonl")):
        file_day = fp.name[len(prefix) + 1:-6]
        with fp.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line), file_day


# --------------------------------------------------------------------- pass 1
def pass1(cfg):
    """Global counters. Deliberately keeps no post text."""
    seen_source = {}            # uri -> bitmask 1=search 2=thread
    hash_count = Counter()      # text_hash -> occurrences (unique posts only)
    author_hash = Counter()     # (did, text_hash) -> occurrences
    author_posts = Counter()
    author_replies = Counter()
    author_days = defaultdict(set)
    day_total = Counter()
    day_meme = Counter()
    no_indexed_at = 0
    rows = 0

    for prefix, bit in (("posts", 1), ("thread", 2)):
        for record, file_day in iter_layer(cfg.raw_dir, prefix):
            rows += 1
            post = record["post"]
            uri = post.get("uri")
            if not uri:
                continue
            first_time = uri not in seen_source
            seen_source[uri] = seen_source.get(uri, 0) | bit
            if not first_time:
                continue

            if not post.get("indexed_at"):
                no_indexed_at += 1
            fday = file_day if bit == 1 else filed_day_of(post, file_day)
            author = (post.get("author") or {}).get("did")
            text = (post.get("record") or {}).get("text") or ""
            normalised = norm_text(text)
            h = text_hash(normalised)
            if h:
                hash_count[h] += 1
                if author:
                    author_hash[(author, h)] += 1
            if author:
                author_days[author].add(fday)
                if bit == 1:
                    author_posts[author] += 1
                else:
                    author_replies[author] += 1
            # Meme share is a property of the searched corpus, not of replies.
            if bit == 1:
                day_total[fday] += 1
                if MEME.match(text):
                    day_meme[fday] += 1

    median_volume = statistics.median(day_total.values()) if day_total else 0
    day_share = {d: (day_meme[d] / n if n else 0.0) for d, n in day_total.items()}
    day_ratio = {d: (n / median_volume if median_volume else 0.0)
                 for d, n in day_total.items()}
    wave_days = {d for d in day_total
                 if day_share[d] >= MEME_WAVE_SHARE
                 and day_ratio[d] >= MEME_WAVE_VOLUME_RATIO}
    meme_present = {d for d in day_total if day_share[d] >= MEME_WAVE_SHARE}
    burst_pairs = {k for k, n in author_hash.items() if n >= REPEAT_BURST_MIN}

    log.info("pass 1: %s raw rows, %s unique posts, %s rows without indexed_at",
             f"{rows:,}", f"{len(seen_source):,}", f"{no_indexed_at:,}")
    log.info("pass 1: median day volume %s; %s day(s) at >=%.0f%% meme share, of "
             "which %s also exceed %.0fx median volume and are flagged as waves",
             f"{median_volume:,.0f}", len(meme_present), 100 * MEME_WAVE_SHARE,
             len(wave_days), MEME_WAVE_VOLUME_RATIO)
    for d in sorted(wave_days):
        log.info("    wave  %s  %6s posts  %5.1f%% meme  %5.1fx median",
                 d, f"{day_total[d]:,}", 100 * day_share[d], day_ratio[d])
    for d in sorted(meme_present - wave_days):
        log.info("    (meme present, volume normal)  %s  %5s posts  %5.1f%%  %.1fx",
                 d, f"{day_total[d]:,}", 100 * day_share[d], day_ratio[d])
    log.info("pass 1: %s (author, text) pair(s) repeated >=%s times",
             f"{len(burst_pairs):,}", REPEAT_BURST_MIN)
    return {
        "seen_source": seen_source, "hash_count": hash_count,
        "author_hash": author_hash, "author_posts": author_posts,
        "author_replies": author_replies, "author_days": author_days,
        "wave_days": wave_days, "burst_pairs": burst_pairs,
        "day_total": day_total, "day_meme": day_meme,
        "day_share": day_share, "day_ratio": day_ratio,
    }


# --------------------------------------------------------------------- pass 2
POSTS_SCHEMA = pa.schema([
    ("uri", pa.string()), ("cid", pa.string()), ("author_did", pa.string()),
    ("day", pa.string()), ("cell_day", pa.string()), ("index_day", pa.string()),
    ("index_day_drift", pa.bool_()),
    ("indexed_at", pa.string()), ("created_at", pa.string()),
    ("created_day", pa.string()),
    ("created_at_drift", pa.bool_()), ("created_at_backdated", pa.bool_()),
    ("collected_at", pa.string()), ("age_at_collection_days", pa.int32()),
    ("text", pa.string()), ("text_norm", pa.string()), ("text_hash", pa.string()),
    ("has_text", pa.bool_()), ("n_chars", pa.int32()),
    ("langs_declared", pa.string()), ("lang_norm", pa.string()),
    ("lang_source", pa.string()), ("lang_detected", pa.string()),
    ("lang_detect_prob", pa.float64()), ("declares_english", pa.bool_()),
    ("is_english", pa.bool_()), ("english_basis", pa.string()),
    ("is_reply", pa.bool_()), ("reply_parent_uri", pa.string()),
    ("reply_root_uri", pa.string()),
    ("discovered_via", pa.string()),
    ("in_search", pa.bool_()), ("in_thread", pa.bool_()),
    ("like_count", pa.int32()), ("repost_count", pa.int32()),
    ("reply_count", pa.int32()), ("quote_count", pa.int32()),
    ("is_meme", pa.bool_()), ("meme_wave_day", pa.bool_()),
    ("day_meme_share", pa.float64()), ("day_volume_ratio", pa.float64()),
    ("dup_text_n", pa.int32()), ("is_dup_text", pa.bool_()),
    ("is_repeat_burst", pa.bool_()),
    ("embed_type", pa.string()), ("labels", pa.string()),
])

PQ_SCHEMA = pa.schema([
    ("uri", pa.string()), ("query", pa.string()), ("stratum", pa.string()),
    ("phrase_exact", pa.bool_()), ("all_terms", pa.bool_()),
    ("day", pa.string()), ("cell_day", pa.string()),
    ("collected_at", pa.string()),
])

EDGE_SCHEMA = pa.schema([
    ("reply_uri", pa.string()), ("parent_uri", pa.string()),
    ("reply_author_did", pa.string()), ("parent_query", pa.string()),
    ("parent_stratum", pa.string()), ("depth", pa.int32()),
    ("from_repair_pass", pa.bool_()), ("reply_day", pa.string()),
])


class Writer:
    """Batched Parquet writer, so no table is ever fully held in memory."""

    def __init__(self, path: Path, schema: pa.Schema):
        self.path = path
        self.schema = schema
        self.rows: list[dict] = []
        self.n = 0
        self._w = pq.ParquetWriter(path, schema, compression="zstd")

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= BATCH:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        cols = {f.name: [r.get(f.name) for r in self.rows] for f in self.schema}
        self._w.write_table(pa.Table.from_pydict(cols, schema=self.schema))
        self.n += len(self.rows)
        self.rows = []

    def close(self) -> int:
        self.flush()
        self._w.close()
        return self.n


def pass2(cfg, state):
    out = cfg.interim_dir
    out.mkdir(parents=True, exist_ok=True)
    posts_w = Writer(out / "posts.parquet", POSTS_SCHEMA)
    pq_w = Writer(out / "post_query.parquet", PQ_SCHEMA)
    edge_w = Writer(out / "thread_edges.parquet", EDGE_SCHEMA)

    seen_source = state["seen_source"]
    hash_count = state["hash_count"]
    burst_pairs = state["burst_pairs"]
    wave_days = state["wave_days"]
    day_share = state["day_share"]
    day_ratio = state["day_ratio"]
    emitted: set[str] = set()
    needs_lang: list[str] = []
    stats = Counter()

    for prefix, bit in (("posts", 1), ("thread", 2)):
        for record, file_day in iter_layer(cfg.raw_dir, prefix):
            meta, post = record["_meta"], record["post"]
            uri = post.get("uri")
            if not uri:
                continue
            fday = filed_day_of(post, file_day)

            if bit == 1:
                hay = match_haystack(post)
                q = meta.get("query") or ""
                qf = fold_query(q)
                terms = qf.split()
                pq_w.add({
                    "uri": uri, "query": q, "stratum": meta.get("stratum"),
                    "phrase_exact": bool(qf and qf in hay),
                    "all_terms": bool(terms and all(t in hay for t in terms)),
                    "day": file_day,
                    "cell_day": day_of(meta.get("since")) or file_day,
                    "collected_at": meta.get("collected_at"),
                })
            else:
                edge_w.add({
                    "reply_uri": uri, "parent_uri": meta.get("parent_uri"),
                    "reply_author_did": (post.get("author") or {}).get("did"),
                    "parent_query": meta.get("parent_query"),
                    "parent_stratum": meta.get("parent_stratum"),
                    "depth": meta.get("depth"),
                    "from_repair_pass": bool(meta.get("recollect_of_capped")),
                    "reply_day": fday,
                })

            if uri in emitted:
                continue
            emitted.add(uri)

            axis_day = file_day if bit == 1 else fday
            rec = post.get("record") or {}
            text = rec.get("text") or ""
            normalised = norm_text(text)
            h = text_hash(normalised)
            author = (post.get("author") or {}).get("did")
            created = rec.get("created_at")
            cday = day_of(created)

            drift = False
            if cday and axis_day:
                try:
                    d1 = datetime.strptime(cday, "%Y-%m-%d")
                    d2 = datetime.strptime(axis_day, "%Y-%m-%d")
                    drift = abs((d1 - d2).days) > CREATED_DRIFT_DAYS
                except ValueError:
                    drift = True

            age_days = None
            collected = meta.get("collected_at")
            if created and collected:
                try:
                    c1 = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    c2 = datetime.fromisoformat(collected)
                    age_days = int((c2 - c1).total_seconds() // 86400)
                except ValueError:
                    age_days = None

            langs = rec.get("langs") or []
            lang_norm = (langs[0].split("-")[0].lower() if langs else None)
            if not langs and normalised:
                needs_lang.append(uri)

            reply = rec.get("reply") or {}
            src = seen_source.get(uri, 0)
            dup_n = hash_count.get(h, 0) if h else 0

            posts_w.add({
                "uri": uri, "cid": post.get("cid"), "author_did": author,
                "day": (file_day if bit == 1 else fday),
                "cell_day": file_day if bit == 1 else None,
                "index_day": fday,
                "index_day_drift": bool(bit == 1 and fday != file_day),
                "indexed_at": post.get("indexed_at"), "created_at": created,
                "created_day": cday,
                "created_at_drift": drift,
                "created_at_backdated": bool(created and created < BLUESKY_EPOCH),
                "collected_at": collected, "age_at_collection_days": age_days,
                "text": text, "text_norm": normalised, "text_hash": h,
                "has_text": bool(normalised), "n_chars": len(text),
                "langs_declared": ",".join(langs) if langs else None,
                "lang_norm": lang_norm,
                "lang_source": "declared" if langs else ("pending" if normalised else "none"),
                "lang_detected": None,          # filled by pass 3
                "lang_detect_prob": None,
                "declares_english": any(l.lower().startswith("en") for l in langs),
                "is_english": None,             # filled by pass 3
                "english_basis": None,
                "is_reply": bool(reply),
                "reply_parent_uri": ((reply.get("parent") or {}).get("uri")
                                     if reply else None),
                "reply_root_uri": ((reply.get("root") or {}).get("uri")
                                   if reply else None),
                "discovered_via": {1: "search", 2: "thread", 3: "both"}.get(src),
                "in_search": bool(src & 1), "in_thread": bool(src & 2),
                "like_count": post.get("like_count") or 0,
                "repost_count": post.get("repost_count") or 0,
                "reply_count": post.get("reply_count") or 0,
                "quote_count": post.get("quote_count") or 0,
                "is_meme": bool(MEME.match(text)),
                "meme_wave_day": axis_day in wave_days,
                "day_meme_share": round(day_share.get(axis_day, 0.0), 4),
                "day_volume_ratio": round(day_ratio.get(axis_day, 0.0), 3),
                "dup_text_n": dup_n, "is_dup_text": dup_n > 1,
                "is_repeat_burst": bool(author and h and (author, h) in burst_pairs),
                "embed_type": (post.get("embed") or {}).get("py_type"),
                "labels": ",".join(sorted({l.get("val") for l in (post.get("labels") or [])
                                           if l.get("val")})) or None,
            })
            stats[{1: "search", 2: "thread", 3: "both"}.get(src, "?")] += 1

    n_posts = posts_w.close()
    n_pq = pq_w.close()
    n_edges = edge_w.close()
    log.info("pass 2: posts=%s  post_query=%s  thread_edges=%s",
             f"{n_posts:,}", f"{n_pq:,}", f"{n_edges:,}")
    log.info("pass 2: discovery -> %s", dict(stats))
    log.info("pass 2: %s post(s) need language detection", f"{len(needs_lang):,}")
    return {"posts": n_posts, "post_query": n_pq, "thread_edges": n_edges,
            "needs_lang": len(needs_lang)}


# --------------------------------------------------------------------- pass 3
def pass3_language(cfg):
    """Resolve a language for every post, then decide what counts as English.

    Two jobs, because they need the same expensive detection sweep.

    1. DETECT where Bluesky carries no `langs` tag at all -- a sixth of the
       corpus. A naive `langs == "en"` filter drops all of it silently, along
       with every post tagged `en-GB` or `en-US`.

    2. VERIFY the declared tag where detection is both long enough and confident
       enough to be trusted (LANG_VERIFY_MIN_CHARS, LANG_CONFIDENT_PROB). The tag
       is set by the posting CLIENT, which generally reports the user's own
       interface language rather than the language they typed in -- so a German
       user posting in English is tagged `de`, and many clients default to `en`
       regardless. Both directions of error are real and both are corrected.

    `is_english` is the project's English-only decision made explicit, and
    `english_basis` records WHY each post was included or excluded so the rule
    can be audited or revisited without re-running the detector.
    """
    import pandas as pd
    from langdetect import DetectorFactory, detect_langs
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0  # langdetect is non-deterministic unless seeded

    path = cfg.interim_dir / "posts.parquet"
    df = pd.read_parquet(path)

    has_text = df["text_norm"].str.len() > 0
    undeclared = (df["lang_source"] == "pending") & has_text
    verifiable = (df["lang_source"] == "declared") & (df["n_chars"] >= LANG_VERIFY_MIN_CHARS)
    todo = undeclared | verifiable
    log.info("pass 3: detecting on %s post(s) (%s undeclared, %s long enough to "
             "verify a declared tag)...",
             f"{int(todo.sum()):,}", f"{int(undeclared.sum()):,}",
             f"{int(verifiable.sum()):,}")

    detected, probs = [], []
    for txt in df.loc[todo, "text_norm"]:
        try:
            best = detect_langs(txt)[0]
            detected.append(best.lang.split("-")[0])
            probs.append(float(best.prob))
        except (LangDetectException, IndexError):
            detected.append(None)
            probs.append(0.0)
    df.loc[todo, "lang_detected"] = detected
    df.loc[todo, "lang_detect_prob"] = probs

    # Undeclared posts take the detected language as their only language.
    resolved = df.loc[undeclared, "lang_detected"]
    df.loc[undeclared, "lang_norm"] = resolved
    df.loc[undeclared, "lang_source"] = resolved.map(
        lambda d: "detected" if d else "undetectable")

    # ---------------------------------------------------------- is_english
    #
    # The rule is deliberately ASYMMETRIC: a post leaves the English corpus only
    # on confident evidence, and a declared tag is trusted otherwise.
    #
    # That asymmetry is not timidity, it is the error costs being unequal. The
    # corpus is ~95% English, so a wrongly-dropped post is a silent loss that
    # correlates with short, informal, emoji-heavy text -- exactly the register a
    # discourse study cares about -- while a few hundred stray non-English posts
    # are negligible noise in 400,000. Dropping on a weak signal would bias the
    # corpus in a direction nobody could see afterwards.
    det = df["lang_detected"]
    prob = df["lang_detect_prob"].fillna(0.0)
    declares_en = df["declares_english"].fillna(False)
    was_declared = df["langs_declared"].notna()

    confident = (prob >= LANG_CONFIDENT_PROB) & (df["n_chars"] >= LANG_VERIFY_MIN_CHARS)
    confident_other = confident & det.notna() & (det != "en")
    confident_en = confident & (det == "en")

    basis = pd.Series("non_english", index=df.index, dtype=object)
    english = pd.Series(False, index=df.index)

    # No text, no language. Excluded from the English corpus but recoverable
    # with `~has_text`: these are link shares, and a link-diffusion analysis
    # wants them even though no text model can use them.
    basis[~has_text] = "no_text"

    # --- posts carrying a declared tag ---------------------------------
    dec = has_text & was_declared
    # Declared English stands unless detection confidently disagrees.
    keep = dec & declares_en & ~confident_other
    english[keep] = True
    basis[keep & (det == "en")] = "declared_confirmed"
    basis[keep & (det != "en")] = "declared"
    basis[dec & declares_en & confident_other] = "declared_overridden"
    # Declared non-English is reclaimed only on a confident English reading --
    # the mirror of the above, so the rule has no thumb on it.
    reclaim = dec & ~declares_en & confident_en
    english[reclaim] = True
    basis[reclaim] = "detected_override"
    basis[dec & ~declares_en & ~confident_en] = "declared_other"

    # --- posts with no declared tag ------------------------------------
    # Detection is the only evidence there is, so it is used as it stands.
    und = has_text & ~was_declared
    english[und & (det == "en")] = True
    basis[und & (det == "en")] = "detected"
    basis[und & det.notna() & (det != "en")] = "detected_other"
    basis[und & det.isna()] = "undetectable"

    df["is_english"] = english
    df["english_basis"] = basis
    df.to_parquet(path, compression="zstd", index=False)

    log.info("pass 3: language -> %s", dict(df["lang_source"].value_counts()))
    log.info("pass 3: is_english = %s of %s posts (%.1f%%)",
             f"{int(english.sum()):,}", f"{len(df):,}", 100 * english.mean())
    for k, v in basis.value_counts().items():
        log.info("    %-22s %8s", k, f"{v:,}")
    return int(english.sum())


# ------------------------------------------------------------------- authors
def build_authors(cfg, state):
    """One row per account, with the signals D5 classifies on kept alongside."""
    profiles = {}
    with (cfg.profiles_dir / "profiles.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            p = r.get("profile") or {}
            did = p.get("did")
            if did:
                profiles[did] = (p, bool((r.get("_meta") or {}).get("gone")))

    author_posts = state["author_posts"]
    author_replies = state["author_replies"]
    author_days = state["author_days"]
    author_hash = state["author_hash"]

    per_author_repeat = Counter()
    for (did, _), n in author_hash.items():
        if n > 1:
            per_author_repeat[did] += n

    rows = []
    classes = Counter()
    for did in set(author_posts) | set(author_replies) | set(profiles):
        profile, gone = profiles.get(did, ({}, False))
        handle = profile.get("handle") or ""
        n_posts = author_posts.get(did, 0)
        n_replies = author_replies.get(did, 0)
        total = n_posts + n_replies
        n_days = len(author_days.get(did, ()))
        labels = {l.get("val") for l in (profile.get("labels") or []) if l.get("val")}
        bridged = "brid.gy" in handle or any("bridged-from" in (l or "") for l in labels)
        rate = (total / n_days) if n_days else 0.0
        repeat_ratio = (per_author_repeat.get(did, 0) / total) if total else 0.0

        # Priority order: the strongest evidence wins. Platform labels first
        # (someone else adjudicated), then structural facts, then behaviour.
        if "spam" in labels:
            klass = "labelled_spam"
        elif "bot" in labels:
            klass = "labelled_bot"
        elif bridged:
            klass = "bridged"
        elif total >= REPEAT_BURST_MIN and repeat_ratio >= 0.5:
            klass = "repeater"
        elif rate >= 10 and total >= 50:
            klass = "high_volume"
        else:
            klass = "ordinary"
        classes[klass] += 1

        rows.append({
            "did": did, "handle": handle or None,
            "n_posts": n_posts, "n_replies": n_replies,
            "n_active_days": n_days, "posts_per_active_day": round(rate, 3),
            "repeat_text_ratio": round(repeat_ratio, 3),
            "followers_count": profile.get("followers_count"),
            "follows_count": profile.get("follows_count"),
            "account_created_at": profile.get("created_at"),
            "has_bio": bool((profile.get("description") or "").strip()),
            "is_bridged": bridged,
            "profile_labels": ",".join(sorted(labels)) or None,
            "profile_missing": did not in profiles,
            "profile_gone": gone,
            "account_class": klass,
        })

    schema = pa.schema([
        ("did", pa.string()), ("handle", pa.string()),
        ("n_posts", pa.int32()), ("n_replies", pa.int32()),
        ("n_active_days", pa.int32()), ("posts_per_active_day", pa.float64()),
        ("repeat_text_ratio", pa.float64()),
        ("followers_count", pa.int64()), ("follows_count", pa.int64()),
        ("account_created_at", pa.string()), ("has_bio", pa.bool_()),
        ("is_bridged", pa.bool_()), ("profile_labels", pa.string()),
        ("profile_missing", pa.bool_()), ("profile_gone", pa.bool_()),
        ("account_class", pa.string()),
    ])
    cols = {f.name: [r.get(f.name) for r in rows] for f in schema}
    pq.write_table(pa.Table.from_pydict(cols, schema=schema),
                   cfg.interim_dir / "authors.parquet", compression="zstd")
    log.info("authors: %s accounts -> %s", f"{len(rows):,}", dict(classes))
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build interim analysis tables.")
    parser.add_argument("--no-langdetect", action="store_true",
                        help="Skip language resolution (pass 3). Leaves "
                             "is_english unset.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config()
    cfg.ensure_dirs()

    state = pass1(cfg)
    counts = pass2(cfg, state)
    n_authors = build_authors(cfg, state)
    if not args.no_langdetect:
        pass3_language(cfg)

    log.info("-" * 60)
    log.info("Wrote %s", cfg.interim_dir)
    for name in ("posts", "post_query", "thread_edges", "authors"):
        p = cfg.interim_dir / f"{name}.parquet"
        if p.exists():
            log.info("  %-14s %8.1f MB", name + ".parquet", p.stat().st_size / 1e6)
    log.info("Nothing was excluded. Every filter is a column; see "
             "docs/bluesky/data_quality_audit.md for what each one is for.")


if __name__ == "__main__":
    main()
