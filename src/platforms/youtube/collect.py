# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

"""Collect every comment thread and reply for the frozen seed videos (tasks Y4-Y5).

Phase threads: videos.list (fresh commentCount) + commentThreads.list for every seed video.
Phase replies: comments.list for threads whose totalReplyCount exceeds the replies embedded in the
thread, largest threads first so the most network-relevant threads are fetched before quota runs out.
Raw output is append-only JSONL; --resume skips IDs already on disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

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

SEED_PATH = REPO_ROOT / "config" / "youtube" / "seed_videos.csv"
VIDEOS_FILE = "videos.jsonl"
THREADS_FILE = "comment_threads.jsonl"
REPLIES_FILE = "replies.jsonl"
REPLIES_DONE_FILE = "replies_done.tsv"
PAGE_SIZE = 100
VIDEOS_PER_CALL = 50
# API errors that describe the video/thread rather than a bug: record the status and move on.
VIDEO_ERROR_STATUS = {"commentsDisabled": "comments_disabled", "videoNotFound": "not_found", "forbidden": "not_accessible"}
THREAD_ERROR_STATUS = {"commentThreadNotFound": "not_found", "commentNotFound": "not_found", "forbidden": "not_accessible"}
FINAL_VIDEO_STATUSES = {"complete", *VIDEO_ERROR_STATUS.values()}
MANIFEST_EVERY_THREADS = 50


def load_seed(path: Path, video_ids: set[str] | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = {"video_id", "event_id"} - set(rows[0].keys() if rows else [])
    if missing:
        raise SystemExit(f"seed file {path} is empty or missing columns: {sorted(missing)}")
    seed, seen = [], set()
    for row in rows:
        if (video_ids and row["video_id"] not in video_ids) or row["video_id"] in seen:
            continue
        seen.add(row["video_id"])
        seed.append(row)
    return seed


# Works with both the frozen seed list and a raw screening sheet (used for smoke runs).
def seed_comment_count(row: dict[str, str]) -> int:
    value = row.get("comment_count_at_screening") or row.get("comment_count") or "0"
    return int(float(value))


def inline_reply_count(thread: dict[str, Any]) -> int:
    return len(thread.get("replies", {}).get("comments", []))


# Only threads with replies missing from the embedded subset need comments.list; biggest first.
def threads_needing_replies(threads: list[dict[str, Any]], video_ids: set[str]) -> list[tuple[str, str, int]]:
    queue = []
    for thread in threads:
        snippet = thread.get("snippet", {})
        total = int(snippet.get("totalReplyCount", 0))
        if snippet.get("videoId") in video_ids and total > inline_reply_count(thread):
            queue.append((thread["id"], snippet["videoId"], total))
    return sorted(queue, key=lambda item: (-item[2], item[0]))


def load_replies_done(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {parts[0]: parts[1] for parts in (line.rstrip("\n").split("\t") for line in handle) if len(parts) >= 2}


def estimate_units(seed: list[dict[str, str]], run_dir: Path | None = None) -> dict[str, Any]:
    per_event: dict[str, dict[str, int]] = {}
    for row in seed:
        count = seed_comment_count(row)
        event = per_event.setdefault(row["event_id"], {"videos": 0, "comment_count": 0, "threads_units_upper_bound": 0})
        event["videos"] += 1
        event["comment_count"] += count
        event["threads_units_upper_bound"] += max(1, math.ceil(count / PAGE_SIZE))
    estimate: dict[str, Any] = {
        "videos_list_units": math.ceil(len(seed) / VIDEOS_PER_CALL),
        "threads_units_upper_bound": sum(event["threads_units_upper_bound"] for event in per_event.values()),
        "per_event": per_event,
        "replies_units_estimate": "unknown until the threads phase has run (re-run --plan-only with --resume)",
    }
    if run_dir and (run_dir / THREADS_FILE).exists():
        threads = [row["data"] for row in read_jsonl(run_dir / THREADS_FILE)]
        done = load_replies_done(run_dir / REPLIES_DONE_FILE)
        remaining = [item for item in threads_needing_replies(threads, {row["video_id"] for row in seed}) if item[0] not in done]
        estimate["replies_threads_remaining"] = len(remaining)
        estimate["replies_units_estimate"] = sum(max(1, math.ceil(total / PAGE_SIZE)) for _, _, total in remaining)
    return estimate


def hydrate_seed_videos(run_dir: Path, seed: list[dict[str, str]], ledger: Ledger, api_key: str) -> None:
    path = run_dir / VIDEOS_FILE
    already = load_ids(path)
    missing = [row["video_id"] for row in seed if row["video_id"] not in already]
    for start in range(0, len(missing), VIDEOS_PER_CALL):
        batch = missing[start:start + VIDEOS_PER_CALL]
        payload = api_get("videos", {"part": "snippet,statistics,contentDetails", "id": ",".join(batch)}, ledger, api_key)
        collection = {"run_id": run_dir.name, "stage": "collect", "endpoint": "videos.list", "fetched_at_utc": utc_now_iso()}
        append_jsonl(path, [wrap(item, collection) for item in payload.get("items", [])])


def collect_threads(run_dir: Path, seed: list[dict[str, str]], manifest: dict[str, Any], ledger: Ledger, api_key: str) -> None:
    path = run_dir / THREADS_FILE
    seen = load_ids(path)
    videos = manifest.setdefault("videos", {})
    for row in seed:
        video_id = row["video_id"]
        state = videos.setdefault(video_id, {"event_id": row["event_id"], "threads_status": "pending", "threads_written": 0, "error": ""})
        if state["threads_status"] in FINAL_VIDEO_STATUSES:
            continue
        # A video interrupted mid-way is re-read from page 1; already-stored threads are skipped by ID.
        state["threads_status"] = "partial"
        params = {"part": "snippet,replies", "videoId": video_id, "order": "time", "textFormat": "plainText", "maxResults": PAGE_SIZE}
        try:
            for page, (token, payload) in enumerate(iter_pages("commentThreads", params, ledger, api_key)):
                new_items = [item for item in payload.get("items", []) if item.get("id") and item["id"] not in seen]
                seen.update(item["id"] for item in new_items)
                collection = {
                    "run_id": run_dir.name, "stage": "collect", "endpoint": "commentThreads.list", "video_id": video_id,
                    "event_id": row["event_id"], "page": page, "page_token": token, "fetched_at_utc": utc_now_iso(),
                }
                state["threads_written"] += append_jsonl(path, [wrap(item, collection) for item in new_items])
            state["threads_status"] = "complete"
        except QuotaExceededError:
            raise
        except ApiError as exc:
            state["threads_status"] = VIDEO_ERROR_STATUS.get(exc.reason, "failed")
            state["error"] = f"HTTP {exc.status} {exc.reason}"
        finally:
            write_manifest(run_dir, manifest)


def collect_replies(run_dir: Path, seed: list[dict[str, str]], manifest: dict[str, Any], ledger: Ledger, api_key: str) -> None:
    replies_path = run_dir / REPLIES_FILE
    done_path = run_dir / REPLIES_DONE_FILE
    threads = [row["data"] for row in read_jsonl(run_dir / THREADS_FILE)]
    queue = threads_needing_replies(threads, {row["video_id"] for row in seed})
    event_by_video = {row["video_id"]: row["event_id"] for row in seed}
    done = load_replies_done(done_path)
    seen = load_ids(replies_path)
    summary = manifest.setdefault("replies", {})
    summary["threads_needing_replies"] = len(queue)
    processed = 0
    try:
        for thread_id, video_id, _total in queue:
            if thread_id in done:
                continue
            written = 0
            params = {"part": "snippet", "parentId": thread_id, "textFormat": "plainText", "maxResults": PAGE_SIZE}
            try:
                for page, (token, payload) in enumerate(iter_pages("comments", params, ledger, api_key)):
                    new_items = [item for item in payload.get("items", []) if item.get("id") and item["id"] not in seen]
                    seen.update(item["id"] for item in new_items)
                    collection = {
                        "run_id": run_dir.name, "stage": "collect", "endpoint": "comments.list", "video_id": video_id,
                        "event_id": event_by_video[video_id], "parent_id": thread_id, "page": page, "page_token": token,
                        "fetched_at_utc": utc_now_iso(),
                    }
                    written += append_jsonl(replies_path, [wrap(item, collection) for item in new_items])
                status = "complete"
            except QuotaExceededError:
                raise
            except ApiError as exc:
                status = THREAD_ERROR_STATUS.get(exc.reason, "failed")
            # Mark the thread done only after all its pages are stored, so an interrupted thread is retried.
            with done_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{thread_id}\t{status}\t{written}\n")
            done[thread_id] = status
            processed += 1
            if processed % MANIFEST_EVERY_THREADS == 0:
                summary["status_counts"] = dict(Counter(done.get(item[0], "pending") for item in queue))
                write_manifest(run_dir, manifest)
    finally:
        summary["status_counts"] = dict(Counter(done.get(item[0], "pending") for item in queue))
        summary["replies_on_disk"] = len(seen)
        write_manifest(run_dir, manifest)


def compute_status(manifest: dict[str, Any], seed: list[dict[str, str]], stop_reason: str) -> str:
    video_states = [manifest.get("videos", {}).get(row["video_id"], {}).get("threads_status", "pending") for row in seed]
    reply_counts = manifest.get("replies", {}).get("status_counts")
    if "failed" in video_states or (reply_counts or {}).get("failed"):
        return "failed"
    if stop_reason or any(state not in FINAL_VIDEO_STATUSES for state in video_states):
        return "partial"
    if reply_counts is None or reply_counts.get("pending"):
        return "partial"
    return "complete"


def run_collection(args: argparse.Namespace, api_key: str | None = None) -> tuple[Path, dict[str, Any]]:
    seed = load_seed(args.seed, set(args.videos) if args.videos else None)
    if not seed:
        raise SystemExit("no seed videos selected")
    api_key = api_key or load_api_key()
    run_dir = args.resume if args.resume else new_run_dir(args.out_root)
    manifest = load_manifest(run_dir) or {
        "created_at_utc": utc_now_iso(), "platform": "youtube", "stage": "collect", "run_id": run_dir.name,
        "script": "src.platforms.youtube.collect", "sessions": [],
    }
    manifest.update({"git_commit": git_commit(), "seed_path": str(args.seed), "seed_sha256": sha256_file(args.seed), "seed_video_count": len(seed)})
    ledger = Ledger(max_units=args.max_units)
    started_at = utc_now_iso()
    stop_reason = ""
    try:
        if args.phase in ("threads", "all"):
            hydrate_seed_videos(run_dir, seed, ledger, api_key)
            collect_threads(run_dir, seed, manifest, ledger, api_key)
        if args.phase in ("replies", "all"):
            collect_replies(run_dir, seed, manifest, ledger, api_key)
    except QuotaExceededError:
        stop_reason = "quota"
    except BudgetExceededError:
        stop_reason = "budget"
    except ApiError as exc:
        stop_reason = f"api_error:{exc.status}:{exc.reason}"
    finally:
        manifest["sessions"].append({
            "started_at_utc": started_at, "ended_at_utc": utc_now_iso(), "phase": args.phase, "usage": ledger.as_dict(),
            "stop_reason": stop_reason, "args": {key: str(value) for key, value in vars(args).items()},
        })
        manifest["stop_reason"] = stop_reason
        manifest["completion_status"] = compute_status(manifest, seed, stop_reason)
        write_manifest(run_dir, manifest)
    return run_dir, manifest


def print_summary(run_dir: Path, manifest: dict[str, Any], args: argparse.Namespace) -> None:
    per_event: dict[str, Counter] = {}
    threads_by_event: Counter = Counter()
    for state in manifest.get("videos", {}).values():
        per_event.setdefault(state["event_id"], Counter())[state["threads_status"]] += 1
        threads_by_event[state["event_id"]] += state["threads_written"]
    for event_id in sorted(per_event):
        print(f"{event_id}: videos {dict(per_event[event_id])} threads_written={threads_by_event[event_id]}")
    print(f"replies: {manifest.get('replies', {})}")
    print(f"usage this session: {manifest['sessions'][-1]['usage']}")
    print(f"completion_status: {manifest['completion_status']} stop_reason: {manifest['stop_reason'] or '-'}")
    print(f"run folder: {run_dir}")
    if manifest["stop_reason"] in ("quota", "budget"):
        print(f"resume: python -m src.platforms.youtube.collect --seed {args.seed} --phase {args.phase} --resume {run_dir}")


def run_self_test() -> None:
    threads = [
        {"id": "a", "snippet": {"videoId": "v", "totalReplyCount": 3}, "replies": {"comments": [{}, {}, {}]}},
        {"id": "b", "snippet": {"videoId": "v", "totalReplyCount": 9}, "replies": {"comments": [{}]}},
        {"id": "c", "snippet": {"videoId": "v", "totalReplyCount": 40}},
        {"id": "d", "snippet": {"videoId": "other", "totalReplyCount": 99}},
    ]
    assert threads_needing_replies(threads, {"v"}) == [("c", "v", 40), ("b", "v", 9)]
    assert seed_comment_count({"comment_count": "120.0"}) == 120
    print("self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=Path, default=SEED_PATH, help="Seed CSV with video_id and event_id columns")
    parser.add_argument("--phase", choices=["threads", "replies", "all"], default="all")
    parser.add_argument("--videos", nargs="+", help="Only these video IDs (smoke runs)")
    parser.add_argument("--out-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--resume", type=Path, help="Continue an existing collection run folder")
    parser.add_argument("--max-units", type=int, help="Stop after this many requests (smoke runs)")
    parser.add_argument("--plan-only", action="store_true", help="Print the unit estimate and exit without using quota")
    parser.add_argument("--self-test", action="store_true", help="Run offline checks and exit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.plan_only:
        seed = load_seed(args.seed, set(args.videos) if args.videos else None)
        print(json.dumps(estimate_units(seed, args.resume), indent=2))
        return 0
    run_dir, manifest = run_collection(args)
    print_summary(run_dir, manifest, args)
    return 0 if manifest["completion_status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
