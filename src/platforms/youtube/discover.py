# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

"""Discover candidate videos for each event window and write a screening sheet (task Y2).

1. search.list for every event x query x scope, restricted to the event window.
2. videos.list to hydrate statistics for every unique video found.
3. A candidates CSV with auto-flags for manual screening.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.shared.events import DEFAULT_EVENTS_PATH, NO_WINDOW, Event, event_window, load_events, window_rfc3339
from src.platforms.youtube.api import ApiError, BudgetExceededError, Ledger, QuotaExceededError, api_get, iter_pages, load_api_key
from src.platforms.youtube.storage import (
    RAW_ROOT,
    REPO_ROOT,
    append_jsonl,
    git_commit,
    load_ids,
    load_manifest,
    new_run_dir,
    read_jsonl,
    sha256_file,
    utc_now_iso,
    wrap,
    write_manifest,
)

QUERIES_PATH = REPO_ROOT / "config" / "youtube" / "queries.json"
LEXICONS_PATH = REPO_ROOT / "config" / "youtube" / "lexicons.json"
SCREENING_ROOT = REPO_ROOT / "data" / "interim" / "youtube" / "screening"
SEARCH_CALLS_PER_DAY = 100
VIDEOS_PER_CALL = 50
# Videos with fewer comments than this give too little text or structure to be worth quota.
MIN_COMMENTS = 20
SCREENING_COLUMNS = [
    "video_id", "event_id", "matched_queries", "scope", "title", "channel_title", "channel_id", "published_at",
    "view_count", "comment_count", "duration_s", "default_audio_language", "keyword_score", "auto_flags",
    "include_suggested", "video_type_suggested", "reason_suggested", "include", "video_type", "reviewer_note",
]
DURATION_RE = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")


@dataclass(frozen=True)
class SearchTask:
    event_id: str
    query_id: str
    q: str
    scope: str
    region_code: str | None

    @property
    def key(self) -> str:
        return f"{self.event_id}|{self.query_id}|{self.scope}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# Every query runs twice: restricted to the event's country, and without a region for international coverage.
def plan_search_tasks(queries_cfg: dict[str, Any], event_ids: list[str] | None = None) -> list[SearchTask]:
    tasks = []
    for event_id, spec in queries_cfg["events"].items():
        if event_ids and event_id not in event_ids:
            continue
        for query in spec["queries"]:
            for scope in queries_cfg["scopes"]:
                region = spec["region"] if scope == "region" else None
                tasks.append(SearchTask(event_id, query["query_id"], query["q"], scope, region))
    return tasks


def search_params(task: SearchTask, event: Event, defaults: dict[str, Any]) -> dict[str, Any]:
    published_after, published_before = window_rfc3339(event)
    params = {**defaults, "part": "snippet", "q": task.q, "publishedAfter": published_after, "publishedBefore": published_before}
    if task.region_code:
        params["regionCode"] = task.region_code
    return params


# ISO 8601 duration such as "PT1H2M3S" -> seconds.
def parse_duration_seconds(value: str | None) -> int | None:
    match = DURATION_RE.match(value or "")
    if not value or not match:
        return None
    days, hours, minutes, seconds = (int(part) if part else 0 for part in match.groups())
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


# Number of distinct topic keywords present; a cheap relevance signal for screening, not a classifier.
def keyword_score(text: str, keywords: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def _to_int(value: Any) -> int | None:
    return int(value) if value not in (None, "") else None


def build_candidates(
    search_rows: list[dict[str, Any]],
    video_items: list[dict[str, Any]],
    events: list[Event],
    topic_keywords: list[str],
) -> pd.DataFrame:
    matches: dict[str, dict[str, set[str]]] = {}
    for row in search_rows:
        video_id = row["data"].get("id", {}).get("videoId")
        if not video_id:
            continue
        match = matches.setdefault(video_id, {"events": set(), "queries": set(), "scopes": set()})
        match["events"].add(row["collection"]["event_id"])
        match["queries"].add(row["collection"]["query_id"])
        match["scopes"].add(row["collection"]["scope"])

    videos_by_id = {item["id"]: item for item in video_items}
    records = []
    for video_id, match in matches.items():
        video = videos_by_id.get(video_id, {})
        snippet = video.get("snippet", {})
        statistics = video.get("statistics", {})
        published_at = snippet.get("publishedAt", "")
        # The event is decided by publish date, so a video can never belong to two event windows.
        window = event_window(date.fromisoformat(published_at[:10]), events) if published_at else NO_WINDOW
        comment_count = _to_int(statistics.get("commentCount"))
        title = snippet.get("title", "")
        score = keyword_score(f"{title}\n{snippet.get('description', '')}", topic_keywords)
        language = snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage") or ""

        flags = []
        if not video:
            flags.append("not_hydrated")
        if window == NO_WINDOW:
            flags.append("out_of_window")
        if comment_count is None:
            flags.append("comments_hidden")
        elif comment_count < MIN_COMMENTS:
            flags.append("low_comments")
        if score == 0:
            flags.append("no_topic_keyword")
        if language and not language.lower().startswith("en"):
            flags.append("non_english_audio")

        records.append({
            "video_id": video_id,
            "event_id": window if window != NO_WINDOW else sorted(match["events"])[0],
            "matched_queries": "|".join(sorted(match["queries"])),
            "scope": "|".join(sorted(match["scopes"])),
            "title": title,
            "channel_title": snippet.get("channelTitle", ""),
            "channel_id": snippet.get("channelId", ""),
            "published_at": published_at,
            "view_count": _to_int(statistics.get("viewCount")),
            "comment_count": comment_count,
            "duration_s": parse_duration_seconds(video.get("contentDetails", {}).get("duration")),
            "default_audio_language": language,
            "keyword_score": score,
            "auto_flags": "|".join(flags),
        })

    frame = pd.DataFrame(records, columns=SCREENING_COLUMNS)
    for column in ("view_count", "comment_count", "duration_s"):
        frame[column] = frame[column].astype("Int64")
    for column in ("include_suggested", "video_type_suggested", "reason_suggested", "include", "video_type", "reviewer_note"):
        frame[column] = ""
    return frame.sort_values(["event_id", "view_count"], ascending=[True, False], na_position="last").reset_index(drop=True)


def hydrate_videos(run_dir: Path, search_path: Path, videos_path: Path, ledger: Ledger, api_key: str) -> None:
    wanted = sorted({row["data"].get("id", {}).get("videoId") for row in read_jsonl(search_path)} - {None})
    already = load_ids(videos_path)
    missing = [video_id for video_id in wanted if video_id not in already]
    for start in range(0, len(missing), VIDEOS_PER_CALL):
        batch = missing[start:start + VIDEOS_PER_CALL]
        payload = api_get("videos", {"part": "snippet,statistics,contentDetails", "id": ",".join(batch)}, ledger, api_key)
        collection = {"run_id": run_dir.name, "stage": "discover", "endpoint": "videos.list", "fetched_at_utc": utc_now_iso()}
        append_jsonl(videos_path, [wrap(item, collection) for item in payload.get("items", [])])


def run_discovery(args: argparse.Namespace, api_key: str | None = None) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    queries_cfg = load_json(args.queries)
    lexicons = load_json(args.lexicons)
    events = load_events(args.events_csv)
    events_by_id = {event.event_id: event for event in events}
    tasks = plan_search_tasks(queries_cfg, args.events)
    api_key = api_key or load_api_key()

    run_dir = args.resume if args.resume else new_run_dir(args.out_root)
    previous = load_manifest(run_dir)
    task_results = {task["key"]: task for task in previous.get("tasks", [])}
    search_path = run_dir / "search_results.jsonl"
    videos_path = run_dir / "videos.jsonl"
    ledger = Ledger(max_units=args.max_units)
    started_at = utc_now_iso()
    stop_reason = ""
    hydration_status = "pending"

    try:
        for task in tasks:
            if task_results.get(task.key, {}).get("status") == "complete":
                continue
            result = task_results[task.key] = {
                "key": task.key, "event_id": task.event_id, "query_id": task.query_id, "scope": task.scope,
                "status": "partial", "items": 0, "pages": 0,
            }
            params = search_params(task, events_by_id[task.event_id], queries_cfg["defaults"])
            for page, (token, payload) in enumerate(iter_pages("search", params, ledger, api_key, max_pages=args.pages)):
                collection = {
                    "run_id": run_dir.name, "stage": "discover", "endpoint": "search.list", "event_id": task.event_id,
                    "query_id": task.query_id, "q": task.q, "scope": task.scope, "region_code": task.region_code,
                    "page": page, "page_token": token, "fetched_at_utc": utc_now_iso(),
                }
                result["items"] += append_jsonl(search_path, [wrap(item, collection) for item in payload.get("items", [])])
                result["pages"] += 1
            result["status"] = "complete"
        hydrate_videos(run_dir, search_path, videos_path, ledger, api_key)
        hydration_status = "complete"
    except QuotaExceededError:
        stop_reason = "quota"
    except BudgetExceededError:
        stop_reason = "budget"
    except ApiError as exc:
        stop_reason = f"api_error:{exc.status}:{exc.reason}"

    search_rows = list(read_jsonl(search_path))
    video_items = [row["data"] for row in read_jsonl(videos_path)]
    candidates = build_candidates(search_rows, video_items, events, lexicons["topic_keywords"])
    args.screening_root.mkdir(parents=True, exist_ok=True)
    screening_csv = args.screening_root / f"candidates_{run_dir.name}.csv"
    candidates.to_csv(screening_csv, index=False)

    all_tasks_complete = all(task_results.get(task.key, {}).get("status") == "complete" for task in tasks)
    if stop_reason.startswith("api_error"):
        completion_status = "failed"
    elif stop_reason or not all_tasks_complete or hydration_status != "complete":
        completion_status = "partial"
    else:
        completion_status = "complete"

    manifest = {
        **previous,
        "created_at_utc": previous.get("created_at_utc", started_at),
        "platform": "youtube",
        "stage": "discover",
        "run_id": run_dir.name,
        "script": "src.platforms.youtube.discover",
        "git_commit": git_commit(),
        "config_sha256": {
            "queries": sha256_file(args.queries), "lexicons": sha256_file(args.lexicons), "events": sha256_file(args.events_csv),
        },
        "event_windows": {event.event_id: window_rfc3339(event) for event in events},
        "snapshot_note": "search.list returns a ranked sample at collection time; results are not exhaustive or reproducible",
        "tasks": sorted(task_results.values(), key=lambda task: task["key"]),
        "hydration_status": hydration_status,
        "sessions": previous.get("sessions", []) + [{
            "started_at_utc": started_at, "ended_at_utc": utc_now_iso(), "usage": ledger.as_dict(), "stop_reason": stop_reason,
            "args": {key: str(value) for key, value in vars(args).items()},
        }],
        "stop_reason": stop_reason,
        "completion_status": completion_status,
        "counts": {
            "search_items": len(search_rows),
            "unique_videos": int(candidates["video_id"].nunique()),
            "hydrated_videos": len(video_items),
            "unflagged_candidates": int((candidates["auto_flags"] == "").sum()),
        },
        "screening_csv": str(screening_csv.relative_to(REPO_ROOT)) if screening_csv.is_relative_to(REPO_ROOT) else str(screening_csv),
    }
    write_manifest(run_dir, manifest)
    return run_dir, manifest, candidates


def print_plan(tasks: list[SearchTask], pages: int) -> None:
    for event_id, count in sorted(Counter(task.event_id for task in tasks).items()):
        print(f"{event_id}: {count} search tasks x {pages} page(s) = {count * pages} search calls")
    total = len(tasks) * pages
    print(f"total search calls: {total} (daily search bucket: {SEARCH_CALLS_PER_DAY})")
    print(f"videos.list units: <= {math.ceil(total * 50 / VIDEOS_PER_CALL)} (upper bound; duplicates across queries reduce it)")
    if total > SEARCH_CALLS_PER_DAY:
        print("WARNING: plan exceeds the daily search bucket")


def print_summary(candidates: pd.DataFrame) -> None:
    for event_id, group in candidates.groupby("event_id"):
        clean = group[group["auto_flags"] == ""]
        channels = ", ".join(f"{name} ({count})" for name, count in clean["channel_title"].value_counts().head(5).items())
        print(
            f"{event_id}: candidates={len(group)} unflagged={len(clean)} "
            f"unflagged_comment_count={int(clean['comment_count'].fillna(0).sum())} top_channels: {channels}"
        )


def run_self_test() -> None:
    assert parse_duration_seconds("PT1H2M3S") == 3723
    assert parse_duration_seconds("PT45S") == 45
    assert parse_duration_seconds("P0D") == 0
    assert parse_duration_seconds("bad") is None
    assert keyword_score("Online Safety Act: VPN surge", ["online safety act", "vpn", "ofcom"]) == 2
    cfg = {"scopes": ["region", "global"], "events": {"E2": {"region": "GB", "queries": [{"query_id": "E2-q1", "q": "x"}]}}}
    assert [task.region_code for task in plan_search_tasks(cfg)] == ["GB", None]
    print("self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--lexicons", type=Path, default=LEXICONS_PATH)
    parser.add_argument("--events-csv", type=Path, default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--events", nargs="+", help="Limit to these event IDs, e.g. E2 E3")
    parser.add_argument("--pages", type=int, default=1, help="search.list pages per task (50 results per page)")
    parser.add_argument("--out-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--screening-root", type=Path, default=SCREENING_ROOT)
    parser.add_argument("--resume", type=Path, help="Continue an existing discovery run folder")
    parser.add_argument("--max-units", type=int, help="Stop after this many requests (smoke runs)")
    parser.add_argument("--plan-only", action="store_true", help="Print planned calls and exit without using quota")
    parser.add_argument("--self-test", action="store_true", help="Run offline checks and exit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return 0
    tasks = plan_search_tasks(load_json(args.queries), args.events)
    if args.plan_only:
        print_plan(tasks, args.pages)
        return 0

    run_dir, manifest, candidates = run_discovery(args)
    print_summary(candidates)
    print(f"usage this session: {manifest['sessions'][-1]['usage']}")
    print(f"completion_status: {manifest['completion_status']} stop_reason: {manifest['stop_reason'] or '-'}")
    print(f"run folder: {run_dir}")
    print(f"screening sheet: {manifest['screening_csv']}")
    return 0 if manifest["completion_status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
