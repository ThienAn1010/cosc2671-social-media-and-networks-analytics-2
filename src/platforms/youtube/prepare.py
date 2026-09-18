# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

"""Prepare one YouTube collection run into the shared documents table (task Y7).

Rules C1-C17 are documented in docs/youtube/cleaning_decisions.md; the schema is documented in docs/youtube/data_dictionary.md.
Rows are never dropped: every rule sets a flag, and exclusion_status names the highest-priority reason.
Output: <out-root>/documents.parquet and <out-root>/prepare_manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib.metadata
import json
import os
import re
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import ftfy
import pandas as pd
from dotenv import load_dotenv
from lingua import Language, LanguageDetectorBuilder

from src.shared.events import DEFAULT_EVENTS_PATH, Event, event_window, load_events, nearest_event
from src.shared.ids import author_hash, make_doc_id
from src.platforms.youtube.storage import REPO_ROOT, load_manifest, read_jsonl, sha256_file, utc_now_iso

SEED_PATH = REPO_ROOT / "config" / "youtube" / "seed_videos.csv"
LEXICONS_PATH = REPO_ROOT / "config" / "youtube" / "lexicons.json"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed" / "youtube"
RAW_FILES = ("videos.jsonl", "comment_threads.jsonl", "replies.jsonl")

# Same language thresholds as the Reddit pipeline (src/prepare_corpus_text.py), so English filtering matches across platforms.
LANGUAGE_CONFIDENCE_THRESHOLD = 0.80
SHORT_UNCERTAIN_TEXT_CHARS = 24
# C9: one author repeating a long comment is a duplicate; C8: texts shared by many authors are counted, not excluded (D-016).
DUPLICATE_MIN_CHARS = 30
COPY_PASTE_MIN_CHARS = 20
LATE_COMMENT_DAYS = 90
PHONE_MIN_DIGITS = 9
# Highest priority first: structural, then ethics, then non-human, then repetition, then language.
EXCLUSION_PRIORITY = ["empty", "sensitive", "spam", "bot_or_owner", "duplicate", "non_english", "language_uncertain"]
EXCLUSION_STATUSES = set(EXCLUSION_PRIORITY) | {"eligible"}
PRIVATE_COLUMNS = {"authorDisplayName", "authorProfileImageUrl", "authorChannelUrl", "author_channel_id", "author_display_name"}

DOCUMENT_COLUMNS = [
    "doc_id", "platform", "thing", "native_id", "parent_doc_id", "root_doc_id", "container_id", "author_hash",
    "created_utc", "created_date_utc", "text_raw", "text_clean", "lang", "lang_confidence", "is_english",
    "engagement", "reply_count", "event_id_nearest", "days_from_event", "event_window", "jurisdiction_hint",
    "jurisdiction_source", "exclusion_status", "is_exact_duplicate", "is_near_duplicate", "is_bot_or_owner",
    "is_spam", "is_url_only", "is_sensitive_content", "bypass_term_hits", "collection_run_id",
    # YouTube-specific fields and diagnostics that explain each flag.
    "yt_video_id", "yt_video_type", "yt_video_event_id", "yt_days_from_video_event", "yt_is_edited",
    "yt_is_late_comment", "yt_like_count", "yt_source", "lang_en_confidence", "is_language_uncertain",
    "has_leading_mention", "url_count", "pii_count", "spam_reason", "copy_paste_author_count",
]

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
HANDLE_RE = re.compile(r"(?<![\w@])@[\w-]+(?:\.[\w-]+)*")
LEAKED_HANDLE_RE = re.compile(r"(?<![\w@])@(?!user\b)[\w-]+")
LEADING_HANDLE_RE = re.compile(r"^\s*@[\w-]+(?:\.[\w-]+)*[\s,:]*")
ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿]")
WHITESPACE_RE = re.compile(r"\s+")
PLACEHOLDER_RE = re.compile(r"<URL>|<PII>|@user")
NON_WORD_RE = re.compile(r"[\W_]+")


def compile_terms(terms: list[str]) -> re.Pattern[str]:
    # Longest terms first so "use a vpn" wins over "vpn"; boundaries stop "vpn" matching inside other words.
    alternation = "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9])(?:{alternation})(?![a-z0-9])", re.IGNORECASE)


def term_hits(pattern: re.Pattern[str], text: str) -> list[str]:
    return sorted({match.lower() for match in pattern.findall(text)})


# C2: repair mojibake, normalise Unicode, unescape HTML entities, drop zero-width characters.
def clean_light(text: str | None) -> str:
    text = ftfy.fix_text(text or "")
    text = unicodedata.normalize("NFC", html.unescape(text))
    return WHITESPACE_RE.sub(" ", ZERO_WIDTH_RE.sub("", text)).strip()


def _mask_phone(match: re.Match[str]) -> str:
    # Only long digit runs are phone numbers; "2025-2026" must survive.
    return "<PII>" if sum(char.isdigit() for char in match.group(0)) >= PHONE_MIN_DIGITS else match.group(0)


# Handles, emails and phone numbers identify people, so they never reach processed text (RULES R-B4).
def mask_people(text: str) -> tuple[str, int]:
    text = EMAIL_RE.sub("<PII>", text)
    text = PHONE_RE.sub(_mask_phone, text)
    return HANDLE_RE.sub("@user", text), text.count("<PII>")


# C4 + C5: NLP text keeps case, punctuation and emoji; drops the leading reply address and masks URLs and personal data.
def make_text_clean(light: str, is_reply: bool) -> tuple[str, int, int, bool]:
    has_leading_mention = bool(is_reply and LEADING_HANDLE_RE.match(light))
    text = LEADING_HANDLE_RE.sub("", light, count=1) if has_leading_mention else light
    text, url_count = URL_RE.subn("<URL>", text)
    text, pii_count = mask_people(text)
    return WHITESPACE_RE.sub(" ", text).strip(), url_count, pii_count, has_leading_mention


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


# C6: top language + confidence; short or low-confidence text is "uncertain", which is kept apart from non-English.
def detect_languages(texts: list[str], detector: Any) -> list[dict[str, Any]]:
    results = []
    for text, values in zip(texts, detector.compute_language_confidence_values_in_parallel(texts)):
        if not text.strip() or not values:
            results.append({"lang": "unknown", "lang_confidence": 0.0, "lang_en_confidence": 0.0, "is_english": False, "is_language_uncertain": True})
            continue
        top = values[0]
        english = next((float(value.value) for value in values if value.language == Language.ENGLISH), 0.0)
        results.append({
            "lang": top.language.name.lower(),
            "lang_confidence": round(float(top.value), 6),
            "lang_en_confidence": round(english, 6),
            "is_english": top.language == Language.ENGLISH,
            "is_language_uncertain": float(top.value) < LANGUAGE_CONFIDENCE_THRESHOLD or len(text.strip()) < SHORT_UNCERTAIN_TEXT_CHARS,
        })
    return results


# C12-C14: two views of time (calendar-nearest event, and the event the video was sampled for) plus jurisdiction hint.
def event_fields(day: date, events: list[Event], video_event: Event | None) -> dict[str, Any]:
    nearest, offset = nearest_event(day, events)
    fields: dict[str, Any] = {"event_id_nearest": nearest.event_id, "days_from_event": offset, "event_window": event_window(day, events)}
    if video_event is None:
        return {**fields, "yt_video_event_id": "", "yt_days_from_video_event": None, "yt_is_late_comment": False,
                "jurisdiction_hint": "unknown", "jurisdiction_source": "none"}
    days = (day - video_event.event_date).days
    return {**fields, "yt_video_event_id": video_event.event_id, "yt_days_from_video_event": days,
            "yt_is_late_comment": days > LATE_COMMENT_DAYS, "jurisdiction_hint": video_event.jurisdiction,
            "jurisdiction_source": "discovery_query"}


# C1: top-level comments, embedded replies and fetched replies, one record per comment ID.
def comment_records(threads: list[dict[str, Any]], reply_wrappers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    records: dict[str, dict[str, Any]] = {}
    for thread in threads:
        snippet = thread["snippet"]
        top = snippet["topLevelComment"]
        records[top["id"]] = {"thing": "comment", "item": top, "video_id": snippet["videoId"], "parent_id": None,
                              "reply_count": snippet.get("totalReplyCount", 0), "source": "commentThreads.list"}
        for reply in thread.get("replies", {}).get("comments", []):
            records.setdefault(reply["id"], {"thing": "reply", "item": reply, "video_id": snippet["videoId"],
                                             "parent_id": reply["snippet"]["parentId"], "reply_count": None,
                                             "source": "commentThreads.list:inline"})
    superseded = 0
    for wrapper in reply_wrappers:
        reply = wrapper["data"]
        superseded += reply["id"] in records
        # The comments.list copy wins: it is the complete, paginated source for that thread.
        records[reply["id"]] = {"thing": "reply", "item": reply, "video_id": wrapper["collection"]["video_id"],
                                "parent_id": reply["snippet"]["parentId"], "reply_count": None, "source": "comments.list"}
    return list(records.values()), superseded


def make_row(
    *, thing: str, native_id: str, parent_doc_id: str, video: dict[str, Any], seed_row: dict[str, str],
    original_text: str, author_channel_id: str, published_at: str, updated_at: str | None, like_count: Any,
    reply_count: Any, source: str, events: list[Event], events_by_id: dict[str, Event],
    patterns: dict[str, re.Pattern[str]], salt: str, run_id: str,
) -> dict[str, Any]:
    channel_id = video["snippet"]["channelId"]
    created = parse_time(published_at)
    light = clean_light(original_text)
    text_clean, url_count, pii_count, leading_mention = make_text_clean(light, is_reply=thing == "reply")
    return {
        "doc_id": make_doc_id("youtube", thing, native_id),
        "platform": "youtube",
        "thing": thing,
        "native_id": native_id,
        "parent_doc_id": parent_doc_id,
        "root_doc_id": make_doc_id("youtube", "video", video["id"]),
        "container_id": f"yt:channel:{channel_id}",
        "author_hash": author_hash(salt, "youtube", author_channel_id),
        "created_utc": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "created_date_utc": created.date().isoformat(),
        "text_raw": mask_people(original_text or "")[0],
        "text_clean": text_clean,
        "engagement": int(like_count) if like_count not in (None, "") else None,
        "reply_count": int(reply_count) if reply_count not in (None, "") else None,
        **event_fields(created.date(), events, events_by_id.get(seed_row.get("event_id", ""))),
        "is_url_only": url_count > 0 and not NON_WORD_RE.sub("", PLACEHOLDER_RE.sub("", text_clean)),
        # C7: the channel that published the video commenting on it is the publisher, not public discourse.
        "is_bot_or_owner": thing != "video" and author_channel_id == channel_id,
        # C10: adult terms or platform names only count when a comment shares a link; talking about them is policy discourse (D-014).
        # Videos are exempt: they passed title screening and news descriptions routinely contain links.
        "is_sensitive_content": thing != "video" and url_count > 0 and bool(term_hits(patterns["sensitive"], light) or term_hits(patterns["adult_platform"], light)),
        "bypass_term_hits": "|".join(term_hits(patterns["bypass"], text_clean)),
        "collection_run_id": run_id,
        "yt_video_id": video["id"],
        "yt_video_type": seed_row.get("video_type", ""),
        "yt_is_edited": bool(updated_at) and updated_at != published_at,
        "yt_like_count": int(like_count) if like_count not in (None, "") else None,
        "yt_source": source,
        "has_leading_mention": leading_mention,
        "url_count": url_count,
        "pii_count": pii_count,
        "_spam_terms": term_hits(patterns["spam"], light),
        "_norm_text": WHITESPACE_RE.sub(" ", text_clean.casefold()).strip(),
    }


# C8 + C9 need the whole corpus: copy-paste across authors, and repeats by the same author (first copy kept eligible).
def add_group_flags(rows: list[dict[str, Any]]) -> None:
    comments = [row for row in rows if row["thing"] != "video"]
    authors_by_text: dict[str, set[str]] = defaultdict(set)
    for row in comments:
        if len(row["_norm_text"]) >= COPY_PASTE_MIN_CHARS:
            authors_by_text[row["_norm_text"]].add(row["author_hash"])
    seen: set[tuple[str, str]] = set()
    for row in sorted(comments, key=lambda item: (item["created_utc"], item["doc_id"])):
        reasons = []
        if row["is_url_only"]:
            reasons.append("url_only")
        if row["_spam_terms"] and (row["url_count"] or row["pii_count"]):
            reasons.append("promo_with_contact")
        # Recorded, not excluded: the same slogan or quote from many authors is shared discourse (D-016).
        row["copy_paste_author_count"] = len(authors_by_text.get(row["_norm_text"], ()))
        row["is_spam"] = bool(reasons)
        row["spam_reason"] = "|".join(reasons)
        key = (row["author_hash"], row["_norm_text"])
        row["is_exact_duplicate"] = len(row["_norm_text"]) >= DUPLICATE_MIN_CHARS and key in seen
        seen.add(key)
    for row in rows:
        if row["thing"] == "video":
            row.update({"is_spam": False, "spam_reason": "", "is_exact_duplicate": False, "copy_paste_author_count": 0})


def exclusion_status(row: dict[str, Any]) -> str:
    checks = {
        "empty": not row["text_clean"],
        "sensitive": row["is_sensitive_content"],
        "spam": row["is_spam"],
        "bot_or_owner": row["is_bot_or_owner"],
        "duplicate": row["is_exact_duplicate"],
        "non_english": not row["is_english"] and not row["is_language_uncertain"],
        "language_uncertain": row["is_language_uncertain"],
    }
    return next((status for status in EXCLUSION_PRIORITY if checks[status]), "eligible")


def build_documents(
    videos: dict[str, dict[str, Any]], threads: list[dict[str, Any]], reply_wrappers: list[dict[str, Any]],
    seed: list[dict[str, str]], events: list[Event], lexicons: dict[str, Any], salt: str, detector: Any, run_id: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    events_by_id = {event.event_id: event for event in events}
    seed_by_video = {row["video_id"]: row for row in seed}
    patterns = {
        name: compile_terms(lexicons[key])
        for name, key in (("spam", "spam_patterns"), ("sensitive", "sensitive_terms"), ("adult_platform", "adult_platform_terms"), ("bypass", "bypass_terms"))
    }
    common = {"events": events, "events_by_id": events_by_id, "patterns": patterns, "salt": salt, "run_id": run_id}

    rows = []
    for video in videos.values():
        snippet = video["snippet"]
        # C17: the video itself is a document (title + description) and the root node of its comment tree.
        rows.append(make_row(
            thing="video", native_id=video["id"], parent_doc_id="", video=video, seed_row=seed_by_video.get(video["id"], {}),
            original_text=f"{snippet.get('title', '')}\n\n{snippet.get('description', '')}", author_channel_id=snippet["channelId"],
            published_at=snippet["publishedAt"], updated_at=None, like_count=video.get("statistics", {}).get("likeCount"),
            reply_count=video.get("statistics", {}).get("commentCount"), source="videos.list", **common,
        ))
    records, superseded = comment_records(threads, reply_wrappers)
    orphans = 0
    for record in records:
        video = videos.get(record["video_id"])
        if video is None:
            orphans += 1
            continue
        snippet = record["item"]["snippet"]
        parent_doc_id = (make_doc_id("youtube", "comment", record["parent_id"]) if record["parent_id"]
                         else make_doc_id("youtube", "video", record["video_id"]))
        rows.append(make_row(
            thing=record["thing"], native_id=record["item"]["id"], parent_doc_id=parent_doc_id, video=video,
            seed_row=seed_by_video.get(record["video_id"], {}),
            original_text=snippet.get("textOriginal") or snippet.get("textDisplay", ""),
            author_channel_id=snippet.get("authorChannelId", {}).get("value", ""), published_at=snippet["publishedAt"],
            updated_at=snippet.get("updatedAt"), like_count=snippet.get("likeCount"), reply_count=record["reply_count"],
            source=record["source"], **common,
        ))

    add_group_flags(rows)
    languages = detect_languages([WHITESPACE_RE.sub(" ", PLACEHOLDER_RE.sub(" ", row["text_clean"])).strip() for row in rows], detector)
    for row, language in zip(rows, languages):
        row.update(language)
        row["exclusion_status"] = exclusion_status(row)

    frame = pd.DataFrame(rows)
    # Near-duplicates are not part of the YouTube cleaning rules, so the column is unknown rather than False.
    frame["is_near_duplicate"] = pd.array([pd.NA] * len(frame), dtype="boolean")
    for column in ("engagement", "reply_count", "yt_days_from_video_event", "yt_like_count"):
        frame[column] = frame[column].astype("Int64")
    return frame[DOCUMENT_COLUMNS], {"inline_replies_superseded_by_comments_list": superseded, "comments_without_video_skipped": orphans}


def validate_documents(frame: pd.DataFrame) -> None:
    problems = []
    missing = [column for column in DOCUMENT_COLUMNS if column not in frame.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    if not frame["doc_id"].is_unique:
        problems.append(f"{int(frame['doc_id'].duplicated().sum())} duplicate doc_id values")
    if PRIVATE_COLUMNS & set(frame.columns):
        problems.append(f"private columns present: {sorted(PRIVATE_COLUMNS & set(frame.columns))}")
    unknown = set(frame["exclusion_status"]) - EXCLUSION_STATUSES
    if unknown:
        problems.append(f"unknown exclusion_status values: {sorted(unknown)}")
    for column in ("text_raw", "text_clean"):
        leaked = int(frame[column].str.contains(LEAKED_HANDLE_RE).sum())
        if leaked:
            problems.append(f"{leaked} rows with an unmasked @handle in {column}")
    if problems:
        raise ValueError("documents failed validation: " + "; ".join(problems))


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.items()}


def build_manifest(args: argparse.Namespace, frame: pd.DataFrame, collection: dict[str, Any], extra_counts: dict[str, int], output: Path) -> dict[str, Any]:
    comments = frame[frame["thing"] != "video"]
    by_event = comments.groupby("yt_video_event_id")["exclusion_status"]
    return {
        "created_at_utc": utc_now_iso(),
        "platform": "youtube",
        "stage": "prepare",
        "script": "src.platforms.youtube.prepare",
        "input": {
            "collect_run": args.collect_run.name,
            "collection_completion_status": collection.get("completion_status"),
            "allow_partial": args.allow_partial,
            "raw_sha256": {name: sha256_file(args.collect_run / name) for name in RAW_FILES if (args.collect_run / name).exists()},
            "seed_sha256": sha256_file(args.seed),
            "lexicons_sha256": sha256_file(args.lexicons),
            "events_sha256": sha256_file(args.events_csv),
        },
        "package_versions": {name: importlib.metadata.version(name) for name in ("pandas", "pyarrow", "lingua-language-detector", "ftfy")},
        "thresholds": {
            "language_confidence": LANGUAGE_CONFIDENCE_THRESHOLD, "short_uncertain_text_chars": SHORT_UNCERTAIN_TEXT_CHARS,
            "duplicate_min_chars": DUPLICATE_MIN_CHARS, "copy_paste_min_chars": COPY_PASTE_MIN_CHARS,
            "late_comment_days": LATE_COMMENT_DAYS,
            "phone_min_digits": PHONE_MIN_DIGITS, "exclusion_priority": EXCLUSION_PRIORITY,
        },
        "counts": {
            "rows": len(frame),
            "by_thing": _counts(frame["thing"].value_counts()),
            "by_exclusion_status": _counts(frame["exclusion_status"].value_counts()),
            "by_thing_and_status": {thing: _counts(group.value_counts()) for thing, group in frame.groupby("thing")["exclusion_status"]},
            "by_event_window_and_thing": {window: _counts(group.value_counts()) for window, group in frame.groupby("event_window")["thing"]},
            "comments_eligible_share_by_video_event": {event: round(float((group == "eligible").mean()), 4) for event, group in by_event},
            "spam_reasons": _counts(comments["spam_reason"][comments["spam_reason"] != ""].str.split("|").explode().value_counts()),
            "comments_shared_by_3plus_authors": int((comments["copy_paste_author_count"] >= 3).sum()),
            **extra_counts,
        },
        "notes": [
            "is_near_duplicate is null: near-duplicate detection is not part of the YouTube cleaning rules.",
            "text_topic is not produced here; the NLP lead derives it from text_clean.",
        ],
        "output": {"documents_parquet": str(output.relative_to(REPO_ROOT)) if output.is_relative_to(REPO_ROOT) else str(output), "sha256": sha256_file(output)},
    }


def load_salt() -> str:
    load_dotenv(REPO_ROOT / ".env")
    salt = os.environ.get("PSEUDONYM_SALT", "").strip()
    if not salt:
        raise SystemExit("PSEUDONYM_SALT missing: add it to SMFR/.env (docs/10-youtube-api-setup-guide.md)")
    return salt


def run_prepare(args: argparse.Namespace, salt: str | None = None, detector: Any = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    collection = load_manifest(args.collect_run)
    if collection.get("completion_status") != "complete" and not args.allow_partial:
        raise SystemExit(f"collection run is {collection.get('completion_status')!r}; pass --allow-partial to prepare it anyway")
    output = args.out_root / "documents.parquet"
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; pass --overwrite to replace it")

    videos = {row["data"]["id"]: row["data"] for row in read_jsonl(args.collect_run / "videos.jsonl")}
    threads = [row["data"] for row in read_jsonl(args.collect_run / "comment_threads.jsonl")]
    reply_wrappers = list(read_jsonl(args.collect_run / "replies.jsonl"))
    with args.seed.open(newline="", encoding="utf-8") as handle:
        seed = list(csv.DictReader(handle))
    lexicons = json.loads(args.lexicons.read_text(encoding="utf-8"))
    detector = detector or LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()

    frame, extra_counts = build_documents(videos, threads, reply_wrappers, seed, load_events(args.events_csv), lexicons,
                                          salt or load_salt(), detector, args.collect_run.name)
    validate_documents(frame)
    args.out_root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    manifest = build_manifest(args, frame, collection, extra_counts, output)
    (args.out_root / "prepare_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return frame, manifest


def run_self_test() -> None:
    assert make_text_clean("@john.doe hi see https://x.com", is_reply=True)[:2] == ("hi see <URL>", 1)
    assert mask_people("mail a@b.com or +61 412 345 678, years 2025-2026")[0] == "mail <PII> or <PII>, years 2025-2026"
    assert clean_light("Tom &amp; Jerry​") == "Tom & Jerry"
    assert term_hits(compile_terms(["vpn", "use a vpn"]), "Just USE A VPN") == ["use a vpn"]
    print("self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collect-run", type=Path, help="Raw collection run folder (data/raw/youtube/<run_id>)")
    parser.add_argument("--seed", type=Path, default=SEED_PATH)
    parser.add_argument("--events-csv", type=Path, default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--lexicons", type=Path, default=LEXICONS_PATH)
    parser.add_argument("--out-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing processed outputs")
    parser.add_argument("--allow-partial", action="store_true", help="Prepare a collection run that is not complete yet")
    parser.add_argument("--self-test", action="store_true", help="Run offline checks and exit")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if not args.collect_run:
        parser.error("--collect-run is required")
    frame, manifest = run_prepare(args)
    counts = manifest["counts"]
    print(json.dumps({key: counts[key] for key in ("rows", "by_thing", "by_exclusion_status", "comments_eligible_share_by_video_event")}, indent=2))
    print(f"output: {manifest['output']['documents_parquet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
