# Student name: Nguyen Hoang Thien An
# Student ID: s3825455
# Assignment 1 - Social Media and Networks Analytics

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import platform
import re
import secrets
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera import Check
from pandera.errors import SchemaErrors

try:
    from src.collect_arctic_shift_core import (
        AGE_ASSURANCE_QUERIES as COLLECTOR_AGE_ASSURANCE_QUERIES,
        API_BASE as COLLECTOR_API_BASE,
        EVENT_WINDOWS as COLLECTOR_EVENT_WINDOWS,
        RELEVANCE_SCREENING,
        TARGET_SUBREDDITS as COLLECTOR_TARGET_SUBREDDITS,
        SUBREDDIT_SCOPE,
        STUDY_NAME,
        STUDY_SCOPE,
    )
    from src.prepare_corpus_text import (
        BOT_AUTHORS,
        DOMAIN_KEEPWORDS,
        LANGUAGE_CONFIDENCE_THRESHOLD,
        NEAR_DUPLICATE_HAMMING_THRESHOLD,
        NEAR_DUPLICATE_MIN_TOKENS,
        NEGATION_KEEPWORDS,
        SPACY_MODEL_NAME,
        URL_RE,
        clean_text_for_sentiment,
        clean_text_light,
        clean_text_for_topic,
        enrich_rows,
        language_set_names,
        package_version,
        promo_hits as promo_keyword_hits,
        sha256_file,
        stable_hash,
        text_presence_status,
        timestamp_fields,
    )
except ModuleNotFoundError:
    from collect_arctic_shift_core import (
        AGE_ASSURANCE_QUERIES as COLLECTOR_AGE_ASSURANCE_QUERIES,
        API_BASE as COLLECTOR_API_BASE,
        EVENT_WINDOWS as COLLECTOR_EVENT_WINDOWS,
        RELEVANCE_SCREENING,
        TARGET_SUBREDDITS as COLLECTOR_TARGET_SUBREDDITS,
        SUBREDDIT_SCOPE,
        STUDY_NAME,
        STUDY_SCOPE,
    )
    from prepare_corpus_text import (
        BOT_AUTHORS,
        DOMAIN_KEEPWORDS,
        LANGUAGE_CONFIDENCE_THRESHOLD,
        NEAR_DUPLICATE_HAMMING_THRESHOLD,
        NEAR_DUPLICATE_MIN_TOKENS,
        NEGATION_KEEPWORDS,
        SPACY_MODEL_NAME,
        URL_RE,
        clean_text_for_sentiment,
        clean_text_light,
        clean_text_for_topic,
        enrich_rows,
        language_set_names,
        package_version,
        promo_hits as promo_keyword_hits,
        sha256_file,
        stable_hash,
        text_presence_status,
        timestamp_fields,
    )


# Name patterns that indicate an automated account.
BOT_AUTHOR_RE = re.compile(r"(^|[_-])bot($|[_-])|automod|moderator", re.IGNORECASE)
# Overall bounds are used only as a preprocessing guard; each raw row also
# carries its event window in the collector provenance.
RESEARCH_START_EPOCH = min(
    int(datetime.fromisoformat(window["start_utc"].replace("Z", "+00:00")).timestamp())
    for window in COLLECTOR_EVENT_WINDOWS.values()
)
RESEARCH_END_EPOCH = max(
    int(datetime.fromisoformat(window["end_utc"].replace("Z", "+00:00")).timestamp())
    for window in COLLECTOR_EVENT_WINDOWS.values()
)
TARGET_SUBREDDITS = set(COLLECTOR_TARGET_SUBREDDITS)
AGE_ASSURANCE_QUERIES = set(COLLECTOR_AGE_ASSURANCE_QUERIES)
SOURCE_CORPORA = {"reddit"}
DEFAULT_AUTHOR_HASH_SALT = secrets.token_bytes(32)


# Preserve Reddit's type prefix so post and comment IDs cannot collide in joins.
def canonical_reddit_id(value: Any, thing: str | None = None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.match(r"^t[13]_", text):
        return text
    prefix = {"post": "t3_", "comment": "t1_"}.get(thing or "")
    return f"{prefix}{text}" if prefix else text


# A top-level comment points to its post; replies point to another comment.
def canonical_parent_id(value: Any, thread_id: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.match(r"^t[13]_", text):
        return text
    bare_thread_id = thread_id.split("_", 1)[1] if "_" in thread_id else thread_id
    return thread_id if text == bare_thread_id else f"t1_{text}"


# Backward-compatible name for callers that used the old helper.
def normalize_reddit_id(value: Any, thing: str | None = None) -> str:
    return canonical_reddit_id(value, thing)


# Keep a run-scoped actor key for within-corpus network analysis.
# HMAC prevents a public hash dictionary attack; the salt is never written to outputs.
def pseudonymous_author_hash(data: dict[str, Any], salt: bytes | None = None) -> str:
    fullname = str(data.get("author_fullname") or "").strip().casefold()
    author = str(data.get("author") or "").strip().casefold()
    identity = fullname or author
    if not identity or identity in {"[deleted]", "deleted", "deleted by user"}:
        return ""
    effective_salt = salt if salt is not None else DEFAULT_AUTHOR_HASH_SALT
    return hmac.new(effective_salt, f"reddit|{identity}".encode("utf-8"), hashlib.sha256).hexdigest()[:24]


# Flag a row as failing schema checks, recording why, instead of deleting it.
def mark_invalid_schema(row: dict[str, Any], reason: str) -> None:
    row["schema_valid"] = False
    row["exclusion_status"] = "invalid_schema"
    row["exclusion_reason"] = reason


# 64-bit hash of one token, memoised because tokens repeat heavily across records.
def token_hash64(token: str, cache: dict[str, int]) -> int:
    value = cache.get(token)
    if value is None:
        value = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")
        cache[token] = value
    return value


# SimHash fingerprint: each bit is set when its column sums positive across token hashes.
def simhash(tokens_text: str, bits: int = 64, token_hash_cache: dict[str, int] | None = None) -> int:
    tokens = tokens_text.split()
    if not tokens:
        return 0
    if token_hash_cache is None:
        token_hash_cache = {}
    vector = [0] * bits
    for token in tokens:
        token_hash = token_hash64(token, token_hash_cache)
        for bit in range(bits):
            vector[bit] += 1 if token_hash & (1 << bit) else -1
    value = 0
    for bit, weight in enumerate(vector):
        if weight > 0:
            value |= 1 << bit
    return value


# Number of differing bits; small distance means near-identical text.
def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


# Exact duplicates via SHA-256 of the case-folded light text; first occurrence is marked.
def duplicate_flags(rows: list[dict[str, Any]]) -> None:
    hashes = [stable_hash(str(row.get("text_light") or "").casefold()) for row in rows]
    counts = Counter(hashes)
    first_seen: set[str] = set()
    for row, text_hash in zip(rows, hashes):
        count = counts[text_hash]
        row["text_hash"] = text_hash
        row["duplicate_text_count"] = count
        row["is_exact_duplicate_text"] = count > 1
        row["is_first_text_instance"] = text_hash not in first_seen
        first_seen.add(text_hash)


# Near duplicates via SimHash banding, comparing only records that share a band.
def near_duplicate_flags(rows: list[dict[str, Any]]) -> None:
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    near_indices: set[int] = set()
    token_hash_cache: dict[str, int] = {}
    simhash_cache: dict[str, int] = {}
    for index, row in enumerate(rows):
        token_count = int(row.get("token_count") or 0)
        tokens_text = str(row.get("tokens_topic") or "")
        if (
            token_count < NEAR_DUPLICATE_MIN_TOKENS
            or not tokens_text
            or (bool(row.get("is_exact_duplicate_text")) and not bool(row.get("is_first_text_instance")))
        ):
            row["simhash"] = ""
            row["is_near_duplicate_text"] = False
            continue

        value = simhash_cache.get(tokens_text)
        if value is None:
            value = simhash(tokens_text, token_hash_cache=token_hash_cache)
            simhash_cache[tokens_text] = value
        row["simhash"] = f"{value:016x}"
        # Band the 64-bit hash into four 16-bit keys; only same-band records are compared.
        for band in range(4):
            key = (band, (value >> (band * 16)) & 0xFFFF)
            for candidate_index, candidate_hash in buckets.get(key, []):
                if rows[candidate_index].get("text_hash") == row.get("text_hash"):
                    continue
                if hamming_distance(value, candidate_hash) <= NEAR_DUPLICATE_HAMMING_THRESHOLD:
                    near_indices.add(candidate_index)
                    near_indices.add(index)
            buckets.setdefault(key, []).append((index, value))

    for index, row in enumerate(rows):
        row["is_near_duplicate_text"] = index in near_indices


# Attach duplicate, bot, spam and author flags to every row.
def add_corpus_flags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    duplicate_flags(rows)
    near_duplicate_flags(rows)

    author_counts = Counter(str(row.get("author_hash") or "") for row in rows if row.get("author_hash"))
    for row in rows:
        author_lower = str(row.get("_author") or "").strip().lower()
        distinguished = str(row.get("_distinguished") or "").strip().lower()
        token_count = int(row.get("token_count") or 0)
        urls = URL_RE.findall(str(row.get("text_raw") or ""))
        url_count = len(urls)
        # A link with almost no text is the common spam shape here.
        is_link_heavy = url_count >= 2 or (url_count >= 1 and token_count <= 4)
        has_author_metadata = bool(row.get("has_author_metadata"))
        is_automoderator = author_lower == "automoderator"
        is_deleted_author = has_author_metadata and author_lower in {"", "[deleted]", "deleted"}
        is_mod_distinguished = distinguished in {"moderator", "admin", "special"}
        is_likely_bot_author = (
            is_automoderator
            or author_lower in BOT_AUTHORS
            or bool(BOT_AUTHOR_RE.search(author_lower))
        )
        promo = promo_keyword_hits(str(row.get("text_light") or ""))
        spam_reasons = []
        if promo and url_count >= 1:
            spam_reasons.append("promo_link")
        if url_count >= 3:
            spam_reasons.append("many_urls")
        if is_likely_bot_author and is_link_heavy:
            spam_reasons.append("bot_link_pattern")

        row["author_post_comment_count"] = author_counts.get(str(row.get("author_hash") or ""), 0)
        row["url_count"] = url_count
        row["promo_keyword_hits"] = "|".join(promo)
        row["is_link_heavy"] = is_link_heavy
        row["is_automoderator"] = is_automoderator
        row["is_deleted_author"] = is_deleted_author
        row["is_mod_distinguished"] = is_mod_distinguished
        row["is_likely_bot_author"] = is_likely_bot_author
        row["is_likely_spam"] = bool(spam_reasons)
        row["spam_reason"] = "|".join(spam_reasons)
        row["unique_text_only"] = bool(row.get("is_first_text_instance"))
    return rows


# Derive the boolean masks each analysis selects on; nothing is removed here.
def finalize_analysis_masks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        exclusion_status = str(row.get("exclusion_status") or "")
        sentiment_text = str(row.get("text_sentiment") or "").strip()
        age_related = bool(row.get("age_assurance_related"))
        manual_language_override = str(row.get("manual_language_override") or "").strip().lower()
        if manual_language_override not in {"", "english", "non_english", "exclude"}:
            manual_language_override = ""
        row["manual_language_override"] = manual_language_override
        row["language_review_required"] = (
            exclusion_status == "language_uncertain"
            and age_related
            and bool(sentiment_text)
            and not bool(row.get("is_url_only"))
            and not bool(row.get("is_no_substantive_text"))
        )
        # Inclusive mask keeps uncertain-language records that still hold real text.
        row["sentiment_review_candidate"] = bool(row["language_review_required"])
        is_substantive_sentiment_text = (
            bool(sentiment_text)
            and not bool(row.get("is_url_only"))
            and not bool(row.get("is_no_substantive_text"))
        )
        row["sentiment_eligible_strict"] = (
            is_substantive_sentiment_text
            and age_related
            and exclusion_status == "eligible"
            and bool(row.get("is_english"))
            and not bool(row.get("is_language_uncertain"))
            and manual_language_override not in {"exclude", "non_english"}
        )
        row["sentiment_eligible"] = (
            is_substantive_sentiment_text
            and age_related
            and (
                row["sentiment_eligible_strict"]
                or row["language_review_required"]
                or (manual_language_override == "english" and exclusion_status in {"eligible", "language_uncertain"})
            )
            and manual_language_override != "exclude"
            and manual_language_override != "non_english"
        )
        row["topic_eligible"] = (
            exclusion_status == "eligible"
            and age_related
            and bool(row.get("is_english"))
            and int(row.get("token_count") or 0) >= 3
            and not bool(row.get("is_likely_spam"))
            and not bool(row.get("is_likely_bot_author"))
            and not bool(row.get("is_url_only"))
            and not bool(row.get("is_no_substantive_text"))
        )
        # Bots and spam are excluded from every reported sentiment population.
        row["human_only"] = (
            not bool(row.get("is_likely_bot_author"))
            and not bool(row.get("is_likely_spam"))
        )
        row["sentiment_analysis_eligible"] = bool(row["sentiment_eligible"] and row["human_only"])
        row["sentiment_analysis_eligible_strict"] = bool(row["sentiment_eligible_strict"] and row["human_only"])
    return rows


# Read optional manual language corrections, rejecting unknown values.
def load_language_overrides(path: Path | None) -> dict[tuple[str, str], str]:
    if path is None:
        return {}
    allowed = {"english", "non_english", "exclude"}
    overrides: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}
        missing = {"thing", "record_id"} - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"language override CSV missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            thing = str(row.get("thing") or "").strip()
            record_id = str(row.get("record_id") or "").strip()
            value = str(row.get("manual_language_override") or row.get("language_override") or "").strip().lower()
            if not thing or not record_id or not value:
                continue
            if value not in allowed:
                raise SystemExit(f"invalid language override {value!r} at {path} line {line_number}")
            overrides[(thing, record_id)] = value
    return overrides


# Apply manual language decisions and report how many rows were touched.
def apply_language_overrides(rows: list[dict[str, Any]], overrides: dict[tuple[str, str], str]) -> int:
    applied = 0
    for row in rows:
        thing = str(row.get("thing") or "")
        record_id = str(row.get("record_id") or "")
        value = overrides.get((thing, record_id))
        if value is None:
            value = overrides.get((thing, canonical_reddit_id(record_id, thing)))
        if value is None and record_id.startswith(("t1_", "t3_")):
            value = overrides.get((thing, record_id.split("_", 1)[1]))
        if value:
            row["manual_language_override"] = value
            applied += 1
    return applied


# Turn one raw JSONL wrapper into the fixed schema, validating provenance as it goes.
def build_base_row(
    wrapper: dict[str, Any],
    thing: str,
    text_raw: str,
    status_parts: tuple[str, str],
    research_start_epoch: int = RESEARCH_START_EPOCH,
    research_end_epoch: int = RESEARCH_END_EPOCH,
    author_hash_salt: bytes | None = None,
) -> dict[str, Any]:
    data = wrapper.get("data")
    collection = wrapper.get("collection")
    schema_valid = isinstance(data, dict) and isinstance(collection, dict)
    row = {
        "record_id": canonical_reddit_id(data.get("id"), thing) if isinstance(data, dict) else "",
        "platform": "reddit",
        "thing": thing,
        "source_corpus": collection.get("source_corpus") if isinstance(collection, dict) else "",
        # Reddit returns display-case names such as "Parenting"; the study configuration is lowercase.
        "subreddit": str(
            (data.get("subreddit") if isinstance(data, dict) else "")
            or (collection.get("subreddit") if isinstance(collection, dict) else "")
        ).strip().lower(),
        "query": collection.get("query") if isinstance(collection, dict) else "",
        "matched_queries": "|".join(
            sorted(
                str(value)
                for value in ((collection.get("matched_queries") if isinstance(collection, dict) else []) or [])
                if value
            )
        ),
        "acquisition_match_count": int(collection.get("acquisition_match_count") or 0)
        if isinstance(collection, dict)
        else 0,
        "event_id": collection.get("event_id") if isinstance(collection, dict) else "",
        "jurisdiction": collection.get("jurisdiction") if isinstance(collection, dict) else "",
        "window_start_utc": collection.get("window_start_utc") if isinstance(collection, dict) else "",
        "window_end_utc": collection.get("window_end_utc") if isinstance(collection, dict) else "",
        "subreddit_scope": (
            (collection.get("subreddit_scope") if isinstance(collection, dict) else "") or SUBREDDIT_SCOPE
        ),
        "jurisdiction_proxy": (
            (collection.get("subreddit_scope") if isinstance(collection, dict) else "") or SUBREDDIT_SCOPE
        ),
        "score": data.get("score") if isinstance(data, dict) else None,
        "text_raw": text_raw,
        "author_hash": pseudonymous_author_hash(data, salt=author_hash_salt) if isinstance(data, dict) else "",
        "_author": str(data.get("author") or "") if isinstance(data, dict) else "",
        "_author_fullname": str(data.get("author_fullname") or "") if isinstance(data, dict) else "",
        "_author_flair_text": str(data.get("author_flair_text") or "") if isinstance(data, dict) else "",
        "_distinguished": data.get("distinguished") if isinstance(data, dict) else "",
        "has_author_metadata": bool(
            isinstance(data, dict)
            and any(
                data.get(key) not in (None, "")
                for key in ("author", "author_fullname", "author_flair_text", "distinguished")
            )
        ),
        "schema_valid": schema_valid,
        "parent_is_root": thing == "post",
        "exclusion_status": status_parts[0],
        "exclusion_reason": status_parts[1],
        "source_file": "",
        "source_line": None,
    }
    timestamp_ok, time_values, timestamp_reason = timestamp_fields(data.get("created_utc") if isinstance(data, dict) else None)
    row.update(time_values)
    if not schema_valid:
        row["exclusion_status"] = "invalid_schema"
        row["exclusion_reason"] = "wrapper missing collection or data object"
    elif not row["record_id"] or row["thing"] not in {"post", "comment"}:
        row["schema_valid"] = False
        row["exclusion_status"] = "invalid_schema"
        row["exclusion_reason"] = "missing record_id or invalid thing"
    elif not timestamp_ok:
        row["schema_valid"] = False
        row["exclusion_status"] = "invalid_timestamp"
        row["exclusion_reason"] = timestamp_reason
    elif int(row["created_utc"]) < research_start_epoch or int(row["created_utc"]) >= research_end_epoch:
        mark_invalid_schema(row, "created_utc outside research window")
    elif row["source_corpus"] not in SOURCE_CORPORA:
        mark_invalid_schema(row, "invalid source_corpus")
    elif row["subreddit"] not in TARGET_SUBREDDITS:
        mark_invalid_schema(row, "row subreddit is outside the configured Reddit communities")
    elif row["query"] not in AGE_ASSURANCE_QUERIES:
        mark_invalid_schema(row, "row query is missing or outside the configured age-assurance query set")
    elif row["event_id"] not in COLLECTOR_EVENT_WINDOWS:
        mark_invalid_schema(row, "row event_id is outside the configured event windows")
    elif any(
        row[field_name] != COLLECTOR_EVENT_WINDOWS[row["event_id"]][window_key]
        for field_name, window_key in (
            ("jurisdiction", "jurisdiction"),
            ("window_start_utc", "start_utc"),
            ("window_end_utc", "end_utc"),
        )
    ):
        mark_invalid_schema(row, "event provenance does not match the configured event window")
    elif row["jurisdiction_proxy"] != SUBREDDIT_SCOPE:
        mark_invalid_schema(row, "subreddit scope provenance is invalid")
    elif data.get("score") is not None:
        try:
            int(data.get("score"))
        except (TypeError, ValueError):
            mark_invalid_schema(row, "score not integer-like")

    row["text_light"] = clean_text_light(text_raw)
    row["text_sentiment"] = clean_text_for_sentiment(text_raw)
    row["text_topic"] = clean_text_for_topic(text_raw)
    row["thread_id"] = ""
    row["parent_record_id"] = ""
    return row


# Parse a submission; title and body are joined because the title often carries the argument.
def unwrap_post(
    wrapper: dict[str, Any],
    nlp=None,
    detector=None,
    stopword_set: set[str] | None = None,
    research_start_epoch: int = RESEARCH_START_EPOCH,
    research_end_epoch: int = RESEARCH_END_EPOCH,
    author_hash_salt: bytes | None = None,
) -> dict[str, Any]:
    data = wrapper.get("data", {}) if isinstance(wrapper.get("data"), dict) else {}
    title = "" if data.get("title") is None else str(data.get("title"))
    selftext = "" if data.get("selftext") is None else str(data.get("selftext"))
    text_raw = title if not selftext else f"{title}\n\n{selftext}" if title else selftext
    row = build_base_row(
        wrapper,
        "post",
        text_raw,
        text_presence_status(title, selftext),
        research_start_epoch,
        research_end_epoch,
        author_hash_salt,
    )
    row.update(
        {
            "title": title,
            "selftext": selftext,
            "body": "",
            "num_comments": data.get("num_comments"),
            "url": data.get("url") or "",
            "link_id": "",
            "parent_id": "",
            "thread_id": canonical_reddit_id(data.get("id"), "post"),
            "parent_record_id": "",
            "parent_is_root": True,
        }
    )
    if row["schema_valid"]:
        try:
            if row["num_comments"] is not None and int(row["num_comments"]) < 0:
                mark_invalid_schema(row, "num_comments negative")
        except (TypeError, ValueError):
            mark_invalid_schema(row, "num_comments not integer-like")
        if "title" not in data or "selftext" not in data:
            mark_invalid_schema(row, "post missing title or selftext field")
    if nlp is not None and detector is not None and stopword_set is not None:
        enrich_rows([row], nlp, detector, stopword_set)
    return row


# Parse a comment, keeping link_id and parent_id for thread context.
def unwrap_comment(
    wrapper: dict[str, Any],
    nlp=None,
    detector=None,
    stopword_set: set[str] | None = None,
    research_start_epoch: int = RESEARCH_START_EPOCH,
    research_end_epoch: int = RESEARCH_END_EPOCH,
    author_hash_salt: bytes | None = None,
) -> dict[str, Any]:
    data = wrapper.get("data", {}) if isinstance(wrapper.get("data"), dict) else {}
    body = "" if data.get("body") is None else str(data.get("body"))
    row = build_base_row(
        wrapper,
        "comment",
        body,
        text_presence_status(body),
        research_start_epoch,
        research_end_epoch,
        author_hash_salt,
    )
    thread_id = canonical_reddit_id(data.get("link_id"), "post")
    parent_record_id = canonical_parent_id(data.get("parent_id"), thread_id)
    row.update(
        {
            "title": "",
            "selftext": "",
            "body": body,
            "num_comments": None,
            "url": "",
            "link_id": thread_id,
            "parent_id": parent_record_id,
            "thread_id": thread_id,
            "parent_record_id": parent_record_id,
            "parent_is_root": not bool(parent_record_id),
        }
    )
    if row["schema_valid"] and ("body" not in data or "link_id" not in data or "parent_id" not in data):
        mark_invalid_schema(row, "comment missing body, link_id or parent_id field")
    elif row["schema_valid"] and not thread_id:
        mark_invalid_schema(row, "comment missing non-empty link_id")
    if nlp is not None and detector is not None and stopword_set is not None:
        enrich_rows([row], nlp, detector, stopword_set)
    return row


# Read JSONL line by line, recording source file and line number on each row.
def read_jsonl(
    path: Path,
    unwrap,
    research_start_epoch: int = RESEARCH_START_EPOCH,
    research_end_epoch: int = RESEARCH_END_EPOCH,
    author_hash_salt: bytes | None = None,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    parsed_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                wrapper = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} line {line_number}") from exc
            row = unwrap(
                wrapper,
                research_start_epoch=research_start_epoch,
                research_end_epoch=research_end_epoch,
                author_hash_salt=author_hash_salt,
            )
            row["source_file"] = str(path)
            row["source_line"] = line_number
            rows.append(row)
            parsed_count += 1
    return rows, parsed_count


# Remove raw author fields before writing; only the opaque network key remains.
def drop_private_columns(frame: pd.DataFrame) -> pd.DataFrame:
    private_columns = [
        "_author",
        "_author_fullname",
        "_author_flair_text",
        "_distinguished",
        "author",
        "author_fullname",
        "author_flair_text",
    ]
    return frame.drop(columns=[column for column in private_columns if column in frame.columns], errors="ignore")


# Pandera schema: required columns, types and the allowed values of each category.
def build_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "record_id": pa.Column(str, nullable=False),
            "platform": pa.Column(str, checks=Check.isin(["reddit"]), nullable=False),
            "thing": pa.Column(str, checks=Check.isin(["post", "comment"]), nullable=False),
            "author_hash": pa.Column(str, nullable=False),
            "jurisdiction_proxy": pa.Column(str, nullable=False),
            "thread_id": pa.Column(str, nullable=False),
            "parent_record_id": pa.Column(str, nullable=False),
            "parent_is_root": pa.Column(bool, nullable=False),
            "relevance_frames": pa.Column(str, nullable=False),
            "schema_valid": pa.Column(bool, nullable=False),
            "exclusion_status": pa.Column(
                str,
                checks=Check.isin(
                    [
                        "eligible",
                        "deleted",
                        "removed",
                        "empty",
                        "invalid_schema",
                        "invalid_timestamp",
                        "non_english",
                        "language_uncertain",
                    ]
                ),
                nullable=False,
            ),
            "text_raw": pa.Column(str, nullable=False),
            "text_light": pa.Column(str, nullable=False),
            "text_sentiment": pa.Column(str, nullable=False),
            "text_topic": pa.Column(str, nullable=False),
            "tokens_topic": pa.Column(str, nullable=False),
            "has_author_metadata": pa.Column(bool, nullable=False),
            "is_low_information": pa.Column(bool, nullable=False),
            "is_url_only": pa.Column(bool, nullable=False),
            "is_no_substantive_text": pa.Column(bool, nullable=False),
            "language": pa.Column(str, nullable=False),
            "language_confidence": pa.Column(float, nullable=False),
            "is_english": pa.Column(bool, nullable=False),
            "is_language_uncertain": pa.Column(bool, nullable=False),
            "language_review_required": pa.Column(bool, nullable=False),
            "sentiment_review_candidate": pa.Column(bool, nullable=False),
            "manual_language_override": pa.Column(str, checks=Check.isin(["", "english", "non_english", "exclude"]), nullable=False),
            "sentiment_neg": pa.Column(float, nullable=False),
            "sentiment_neu": pa.Column(float, nullable=False),
            "sentiment_pos": pa.Column(float, nullable=False),
            "sentiment_compound": pa.Column(float, nullable=False),
            "sentiment_label": pa.Column(str, checks=Check.isin(["", "negative", "neutral", "positive"]), nullable=False),
            "age_assurance_related": pa.Column(bool, nullable=False),
            "relevance_level": pa.Column(
                str,
                checks=Check.isin(
                    [
                        "age_policy",
                        "child_safety",
                        "privacy_surveillance",
                        "governance",
                        "circumvention_autonomy",
                        "unrelated",
                    ]
                ),
                nullable=False,
            ),
            "sentiment_eligible": pa.Column(bool, nullable=False),
            "sentiment_eligible_strict": pa.Column(bool, nullable=False),
            "sentiment_analysis_eligible": pa.Column(bool, nullable=False),
            "sentiment_analysis_eligible_strict": pa.Column(bool, nullable=False),
            "topic_eligible": pa.Column(bool, nullable=False),
            "human_only": pa.Column(bool, nullable=False),
            "unique_text_only": pa.Column(bool, nullable=False),
        },
        strict=False,
        coerce=False,
    )


# Validate the frame and stop the run on failure, writing the failing cases to CSV.
def validate_frame(frame: pd.DataFrame, out_root: Path | None = None) -> None:
    try:
        build_schema().validate(frame, lazy=True)
    except SchemaErrors as exc:
        failure_root = out_root or Path("data/processed")
        failure_path = failure_root / "schema_failures.csv"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        exc.failure_cases.to_csv(failure_path, index=False)
        raise SystemExit(f"schema validation failed. details: {failure_path}") from exc

    # A repeated (thing, record_id) would mean the collector deduplication failed.
    valid_ids = frame["schema_valid"] & frame["record_id"].astype(str).str.len().gt(0)
    duplicated = frame.loc[valid_ids].duplicated(subset=["record_id"], keep=False)
    if duplicated.any():
        raise SystemExit("schema validation failed. duplicate (thing, record_id) values present")


# Build the DataFrame and force consistent dtypes so Parquet and CSV agree.
def frame_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame = drop_private_columns(frame)
    numeric_columns = [
        "created_utc",
        "score",
        "num_comments",
        "token_count",
        "char_count",
        "duplicate_text_count",
        "author_post_comment_count",
        "url_count",
        "emoji_count",
        "source_line",
    ]
    float_columns = [
        "language_confidence",
        "sentiment_neg",
        "sentiment_neu",
        "sentiment_pos",
        "sentiment_compound",
    ]
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in float_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    bool_columns = [
        "schema_valid",
        "parent_is_root",
        "age_assurance_related",
        "has_author_metadata",
        "is_english",
        "is_language_uncertain",
        "language_review_required",
        "sentiment_review_candidate",
        "is_automoderator",
        "is_deleted_author",
        "is_mod_distinguished",
        "is_likely_bot_author",
        "is_low_information",
        "is_url_only",
        "is_no_substantive_text",
        "is_link_heavy",
        "is_likely_spam",
        "is_exact_duplicate_text",
        "is_near_duplicate_text",
        "is_first_text_instance",
        "sentiment_eligible",
        "sentiment_eligible_strict",
        "sentiment_analysis_eligible",
        "sentiment_analysis_eligible_strict",
        "topic_eligible",
        "human_only",
        "unique_text_only",
        "has_emoji",
    ]
    for column in bool_columns:
        if column in frame:
            frame[column] = frame[column].fillna(False).astype(bool)
    for column in frame.columns:
        if column not in set(numeric_columns + float_columns + bool_columns):
            frame[column] = frame[column].fillna("").astype("string")
    return frame


# Validate, then write both Parquet (for code) and CSV (for inspection).
def write_outputs(rows: list[dict[str, Any]], stem: str, out_root: Path) -> tuple[Path, Path, pd.DataFrame]:
    out_root.mkdir(parents=True, exist_ok=True)
    parquet_path = out_root / f"{stem}.parquet"
    csv_path = out_root / f"{stem}.csv"
    frame = frame_from_rows(rows)
    validate_frame(frame, out_root)
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    return parquet_path, csv_path, frame


# Write a representative export without raw text or Reddit identifiers.
def write_submission_sample(frame: pd.DataFrame, out_path: Path, sample_size: int = 500) -> Path:
    if sample_size < 1:
        raise ValueError("submission sample size must be positive")
    safe_frame = frame.loc[frame["schema_valid"].eq(True)].copy()
    if len(safe_frame) > sample_size:
        safe_frame = safe_frame.sample(n=sample_size, random_state=42)
    safe_frame = safe_frame.sort_values(["thing", "record_id"]).reset_index(drop=True)
    node_ids = {record_id: f"node_{index + 1:06d}" for index, record_id in enumerate(safe_frame["record_id"])}
    safe_frame["submission_node_id"] = safe_frame["record_id"].map(node_ids)
    safe_frame["submission_thread_id"] = safe_frame["thread_id"].map(node_ids).fillna("")
    safe_frame["submission_parent_id"] = safe_frame["parent_record_id"].map(node_ids).fillna("")
    safe_frame = safe_frame.drop(
        columns=[
            "record_id",
            "thread_id",
            "parent_record_id",
            "link_id",
            "parent_id",
            "text_raw",
            "text_light",
            "text_sentiment",
            "text_topic",
            "title",
            "selftext",
            "body",
            "url",
            "source_file",
            "source_line",
            "text_hash",
            "simhash",
        ],
        errors="ignore",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe_frame.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    return out_path


# Per-output counts of exclusion status, language and relevance, for the manifest.
def summarize_rows(frame: pd.DataFrame) -> dict[str, Any]:
    frame_counts: Counter[str] = Counter()
    for value in frame["relevance_frames"].astype(str):
        frame_counts.update(label for label in value.split("|") if label)
    return {
        "rows": int(len(frame)),
        "exclusion_counts": {str(k): int(v) for k, v in frame["exclusion_status"].value_counts(dropna=False).items()},
        "language_counts": {str(k): int(v) for k, v in frame["language"].value_counts(dropna=False).items()},
        "relevance_counts": {str(k): int(v) for k, v in frame["relevance_level"].value_counts(dropna=False).items()},
        "frame_counts": dict(sorted(frame_counts.items())),
    }


# Read the manifest written by the collector, if it exists.
def load_collection_manifest(raw_root: Path) -> dict[str, Any]:
    manifest_path = raw_root / "collection_manifest.json"
    if not manifest_path.exists():
        return {"status": "missing", "path": str(manifest_path)}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


# Prevent preprocessing from accepting a complete manifest from another study.
def validate_collection_manifest_identity(manifest: dict[str, Any]) -> None:
    mismatches: list[str] = []
    if manifest.get("study_name") != STUDY_NAME:
        mismatches.append("study_name")
    if manifest.get("api_base") != COLLECTOR_API_BASE:
        mismatches.append("api_base")
    if set(manifest.get("target_subreddits") or []) != TARGET_SUBREDDITS:
        mismatches.append("target_subreddits")
    if set(manifest.get("age_assurance_queries") or []) != AGE_ASSURANCE_QUERIES:
        mismatches.append("age_assurance_queries")
    if manifest.get("relevance_screening") != RELEVANCE_SCREENING:
        mismatches.append("relevance_screening")
    expected_events = [{"event_id": event_id, **window} for event_id, window in COLLECTOR_EVENT_WINDOWS.items()]
    if manifest.get("event_windows") != expected_events:
        mismatches.append("event_windows")
    if mismatches:
        raise SystemExit("collection manifest does not match this study: " + ", ".join(mismatches))


# Refuse to preprocess a partial collection, since it would produce plausible but wrong totals.
def require_complete_collection_manifest(raw_root: Path, allow_partial: bool) -> dict[str, Any]:
    manifest = load_collection_manifest(raw_root)
    status = manifest.get("completion_status", manifest.get("status", "missing"))
    if status != "missing":
        validate_collection_manifest_identity(manifest)
    if status != "complete" and not allow_partial:
        raise SystemExit(
            "collection manifest status must be complete before preprocessing. "
            f"found {status!r} at {raw_root / 'collection_manifest.json'}. "
            "rerun collection or pass --allow-partial"
        )
    return manifest


# Use the collector's actual window so CLI overrides cannot invalidate good rows.
def research_window_from_manifest(manifest: dict[str, Any]) -> tuple[int, int]:
    start_value = manifest.get("start_utc")
    end_value = manifest.get("end_utc")
    if not start_value or not end_value:
        return RESEARCH_START_EPOCH, RESEARCH_END_EPOCH
    try:
        start = datetime.fromisoformat(str(start_value).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(end_value).replace("Z", "+00:00"))
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("collection window timestamps must include timezone")
        start_epoch = int(start.timestamp())
        end_epoch = int(end.timestamp())
    except (TypeError, ValueError) as exc:
        raise SystemExit("collection manifest has invalid start_utc or end_utc") from exc
    if end_epoch <= start_epoch:
        raise SystemExit("collection manifest end_utc must be after start_utc")
    return start_epoch, end_epoch


# Record inputs, hashes, package versions, thresholds and counts for reproducibility.
def build_prepare_manifest(
    *,
    raw_root: Path,
    posts_frame: pd.DataFrame,
    comments_frame: pd.DataFrame,
    analysis_frame: pd.DataFrame,
    parsed_counts: dict[str, int],
    collection_manifest: dict[str, Any],
    nltk_resources: dict[str, str],
    spacy_model,
    stopword_set: set[str],
    submission_sample: Path | None = None,
    submission_sample_size: int | None = None,
) -> dict[str, Any]:
    raw_inputs = {
        "posts_raw": {
            "path": str(raw_root / "posts_raw.jsonl"),
            "sha256": sha256_file(raw_root / "posts_raw.jsonl"),
        },
        "comments_raw": {
            "path": str(raw_root / "comments_raw.jsonl"),
            "sha256": sha256_file(raw_root / "comments_raw.jsonl"),
        },
    }
    acquisition_routes = raw_root / "acquisition_routes.jsonl"
    if acquisition_routes.exists():
        raw_inputs["acquisition_routes"] = {
            "path": str(acquisition_routes),
            "sha256": sha256_file(acquisition_routes),
        }
    manifest = {
        "study_name": STUDY_NAME,
        "study_scope": STUDY_SCOPE,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "raw_root": str(raw_root),
        "raw_inputs": raw_inputs,
        "outputs": {
            "posts_clean": ["posts_clean.parquet", "posts_clean.csv"],
            "comments_clean": ["comments_clean.parquet", "comments_clean.csv"],
            "analysis_corpus": ["analysis_corpus.parquet", "analysis_corpus.csv"],
        },
        "collection_manifest_status": collection_manifest.get("completion_status", "missing"),
        "collection_manifest_path": str(raw_root / "collection_manifest.json"),
        "relevance_screening": RELEVANCE_SCREENING,
        "package_versions": {
            "python": platform.python_version(),
            "pandas": package_version("pandas"),
            "pyarrow": package_version("pyarrow"),
            "scikit-learn": package_version("scikit-learn"),
            "nltk": package_version("nltk"),
            "pandera": package_version("pandera"),
            "spacy": package_version("spacy"),
            "emoji": package_version("emoji"),
            "lingua-language-detector": package_version("lingua-language-detector"),
            "ftfy": package_version("ftfy"),
        },
        "author_hashing": {
            "algorithm": "HMAC-SHA256",
            "scope": "one preprocessing run",
            "salt_stored": False,
        },
        "nltk_resources": nltk_resources,
        "spacy_model": {
            "name": SPACY_MODEL_NAME if spacy_model is not None else "",
            "version": spacy_model.meta.get("version", "") if hasattr(spacy_model, "meta") else "",
        },
        "language_detector": {
            "library": "lingua-language-detector",
            "configuration": "from_languages + preloaded models",
            "languages": language_set_names(),
            "english_threshold": LANGUAGE_CONFIDENCE_THRESHOLD,
        },
        "stopwords": {
            "source": "nltk english stopwords union scikit-learn ENGLISH_STOP_WORDS",
            "negation_keepwords": sorted(NEGATION_KEEPWORDS),
            "domain_overrides": sorted(DOMAIN_KEEPWORDS),
            "final_size": len(stopword_set),
        },
        "thresholds": {
            "language_confidence_threshold": LANGUAGE_CONFIDENCE_THRESHOLD,
            "near_duplicate_hamming_threshold": NEAR_DUPLICATE_HAMMING_THRESHOLD,
            "near_duplicate_min_tokens": NEAR_DUPLICATE_MIN_TOKENS,
        },
        "counts": {
            "parsed_rows": parsed_counts,
            "output_rows": {
                "posts_clean": int(len(posts_frame)),
                "comments_clean": int(len(comments_frame)),
                "analysis_corpus": int(len(analysis_frame)),
            },
            "before_after": {
                "raw_parsed_total": int(sum(parsed_counts.values())),
                "processed_total": int(len(analysis_frame)),
            },
            "exclusion_counts": {str(k): int(v) for k, v in analysis_frame["exclusion_status"].value_counts(dropna=False).items()},
            "language_counts": {str(k): int(v) for k, v in analysis_frame["language"].value_counts(dropna=False).items()},
            "relevance_counts": {str(k): int(v) for k, v in analysis_frame["relevance_level"].value_counts(dropna=False).items()},
            "bot_spam_duplicate_counts": {
                "is_likely_bot_author": int(analysis_frame["is_likely_bot_author"].sum()),
                "is_mod_distinguished": int(analysis_frame["is_mod_distinguished"].sum()),
                "is_likely_spam": int(analysis_frame["is_likely_spam"].sum()),
                "is_exact_duplicate_text": int(analysis_frame["is_exact_duplicate_text"].sum()),
                "is_near_duplicate_text": int(analysis_frame["is_near_duplicate_text"].sum()),
            },
            "analysis_mask_counts": {
                "sentiment_eligible": int(analysis_frame["sentiment_eligible"].sum()),
                "sentiment_eligible_strict": int(analysis_frame["sentiment_eligible_strict"].sum()),
                "sentiment_analysis_eligible": int(analysis_frame["sentiment_analysis_eligible"].sum()),
                "sentiment_analysis_eligible_strict": int(analysis_frame["sentiment_analysis_eligible_strict"].sum()),
                "topic_eligible": int(analysis_frame["topic_eligible"].sum()),
                "human_only": int(analysis_frame["human_only"].sum()),
                "unique_text_only": int(analysis_frame["unique_text_only"].sum()),
                "language_review_required": int(analysis_frame["language_review_required"].sum()),
            },
        },
        "summaries": {
            "posts_clean": summarize_rows(posts_frame),
            "comments_clean": summarize_rows(comments_frame),
            "analysis_corpus": summarize_rows(analysis_frame),
        },
    }
    if submission_sample is not None:
        manifest["outputs"]["submission_sample"] = [str(submission_sample)]
        manifest["submission_export"] = {
            "path": str(submission_sample),
            "sample_size": submission_sample_size,
            "redactions": [
                "raw text and titles",
                "Reddit record IDs and relation IDs",
                "raw author fields",
                "post URLs",
            ],
        }
    return manifest


# Write prepare_manifest.json beside the processed outputs.
def write_manifest(out_root: Path, manifest: dict[str, Any]) -> Path:
    path = out_root / "prepare_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path
