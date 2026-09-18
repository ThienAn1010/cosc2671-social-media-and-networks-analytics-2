# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

"""Collection QA gates G1-G8 for one YouTube collection run (task Y6).

Each gate is a function returning plain data so the notebook can reuse it; main() writes docs/youtube/collection_qa.md
and a 30-comment spot-check sheet (G7) for manual comparison with the YouTube UI.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.platforms.youtube.storage import REPO_ROOT, load_manifest, read_jsonl, utc_now_iso

QA_DOC_PATH = REPO_ROOT / "docs" / "youtube" / "collection_qa.md"
# Comment text stays out of git, so the spot-check sheet lives in the ignored interim layer.
SPOT_CHECK_PATH = REPO_ROOT / "data" / "interim" / "youtube" / "qa" / "spot_check_g7.csv"
COVERAGE_FLAG = 0.70
REPLY_GAP_FLAG = 0.20
SPOT_CHECK_SIZE = 30
RANDOM_STATE = 42
KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{35}")
HANDLE_RE = re.compile(r"(?<![\w@])@[\w-]+(?:\.[\w-]+)*")


def load_run(run_dir: Path) -> dict[str, Any]:
    return {
        "manifest": load_manifest(run_dir),
        "videos": [row["data"] for row in read_jsonl(run_dir / "videos.jsonl")],
        "threads": [row["data"] for row in read_jsonl(run_dir / "comment_threads.jsonl")],
        "replies": list(read_jsonl(run_dir / "replies.jsonl")),
    }


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


# Replies come from two places: embedded in threads and fetched with comments.list; count each ID once.
def unique_replies(run: dict[str, Any]) -> dict[str, dict[str, str]]:
    replies: dict[str, dict[str, str]] = {}
    for thread in run["threads"]:
        for reply in thread.get("replies", {}).get("comments", []):
            replies[reply["id"]] = {"video_id": thread["snippet"]["videoId"], "parent_id": reply["snippet"]["parentId"],
                                    "published_at": reply["snippet"]["publishedAt"], "text": reply["snippet"].get("textOriginal", "")}
    for wrapper in run["replies"]:
        reply = wrapper["data"]
        replies[reply["id"]] = {"video_id": wrapper["collection"]["video_id"], "parent_id": reply["snippet"]["parentId"],
                                "published_at": reply["snippet"]["publishedAt"], "text": reply["snippet"].get("textOriginal", "")}
    return replies


def g1_manifest_status(run: dict[str, Any]) -> dict[str, Any]:
    manifest = run["manifest"]
    videos = manifest.get("videos", {})
    return {
        "completion_status": manifest.get("completion_status"),
        "stop_reason": manifest.get("stop_reason") or "",
        "video_statuses": dict(Counter(state["threads_status"] for state in videos.values())),
        "failed_videos": sorted(video_id for video_id, state in videos.items() if state["threads_status"] == "failed"),
        "reply_thread_statuses": manifest.get("replies", {}).get("status_counts", {}),
        "sessions": [{"started_at_utc": s["started_at_utc"], "units": s["usage"]["units"], "stop_reason": s["stop_reason"]} for s in manifest.get("sessions", [])],
    }


# G2: collected comments (threads + unique replies) over the commentCount the API reported at collection time.
def g2_coverage(run: dict[str, Any]) -> pd.DataFrame:
    threads_per_video = Counter(thread["snippet"]["videoId"] for thread in run["threads"])
    replies_per_video = Counter(reply["video_id"] for reply in unique_replies(run).values())
    event_by_video = {video_id: state["event_id"] for video_id, state in run["manifest"].get("videos", {}).items()}
    rows = []
    for video in run["videos"]:
        video_id = video["id"]
        api_count = int(video.get("statistics", {}).get("commentCount") or 0)
        collected = threads_per_video[video_id] + replies_per_video[video_id]
        rows.append({
            "video_id": video_id, "event_id": event_by_video.get(video_id, ""), "api_comment_count": api_count,
            "threads": threads_per_video[video_id], "replies": replies_per_video[video_id], "collected": collected,
            "coverage": round(collected / api_count, 3) if api_count else None,
        })
    frame = pd.DataFrame(rows)
    frame["flag_low_coverage"] = frame["coverage"].fillna(1.0) < COVERAGE_FLAG
    return frame.sort_values(["coverage", "video_id"]).reset_index(drop=True)


def g3_duplicates(run: dict[str, Any]) -> dict[str, int]:
    thread_ids = [thread["id"] for thread in run["threads"]]
    fetched_ids = [wrapper["data"]["id"] for wrapper in run["replies"]]
    inline_ids = {reply["id"] for thread in run["threads"] for reply in thread.get("replies", {}).get("comments", [])}
    return {
        "thread_rows": len(thread_ids), "thread_duplicate_rows": len(thread_ids) - len(set(thread_ids)),
        "fetched_reply_rows": len(fetched_ids), "fetched_reply_duplicate_rows": len(fetched_ids) - len(set(fetched_ids)),
        "inline_replies_also_fetched": len(inline_ids & set(fetched_ids)),
    }


def g4_reply_parents(run: dict[str, Any]) -> dict[str, Any]:
    thread_ids = {thread["id"] for thread in run["threads"]}
    replies = unique_replies(run)
    missing = sum(1 for reply in replies.values() if reply["parent_id"] not in thread_ids)
    return {"replies": len(replies), "missing_parent": missing, "share_with_parent": round(1 - missing / len(replies), 4) if replies else 1.0}


def g5_timestamps(run: dict[str, Any]) -> dict[str, Any]:
    published = {video["id"]: _time(video["snippet"]["publishedAt"]) for video in run["videos"]}
    comments = [(thread["snippet"]["videoId"], thread["snippet"]["topLevelComment"]["snippet"]["publishedAt"]) for thread in run["threads"]]
    comments += [(reply["video_id"], reply["published_at"]) for reply in unique_replies(run).values()]
    failures, before_video, times = 0, 0, []
    for video_id, value in comments:
        try:
            moment = _time(value)
        except (TypeError, ValueError):
            failures += 1
            continue
        times.append(moment)
        before_video += video_id in published and moment < published[video_id]
    return {
        "comments": len(comments), "parse_failures": failures, "before_video_publish": int(before_video),
        "earliest_utc": min(times).strftime("%Y-%m-%dT%H:%M:%SZ") if times else "",
        "latest_utc": max(times).strftime("%Y-%m-%dT%H:%M:%SZ") if times else "",
    }


# G6: per thread, replies collected vs totalReplyCount; deleted or held replies make small gaps normal.
def g6_reply_reconciliation(run: dict[str, Any]) -> dict[str, Any]:
    collected = Counter(reply["parent_id"] for reply in unique_replies(run).values())
    expected_total, collected_total, gaps = 0, 0, 0
    threads_with_replies = 0
    for thread in run["threads"]:
        total = int(thread["snippet"].get("totalReplyCount", 0))
        if not total:
            continue
        threads_with_replies += 1
        got = collected[thread["id"]]
        expected_total += total
        collected_total += got
        gaps += abs(got - total) / total > REPLY_GAP_FLAG
    return {
        "threads_with_replies": threads_with_replies, "expected_replies": expected_total, "collected_replies": collected_total,
        "ratio": round(collected_total / expected_total, 4) if expected_total else 1.0, "threads_gap_over_20pct": int(gaps),
    }


# G7: a reproducible random sample for Kien to compare with the YouTube UI; handles are masked in the preview.
def g7_spot_check_sample(run: dict[str, Any], size: int = SPOT_CHECK_SIZE) -> pd.DataFrame:
    rows = [{"video_id": t["snippet"]["videoId"], "comment_id": t["id"], "kind": "comment",
             "published_at": t["snippet"]["topLevelComment"]["snippet"]["publishedAt"],
             "text": t["snippet"]["topLevelComment"]["snippet"].get("textOriginal", "")} for t in run["threads"]]
    rows += [{"video_id": r["video_id"], "comment_id": comment_id, "kind": "reply", "published_at": r["published_at"], "text": r["text"]}
             for comment_id, r in unique_replies(run).items()]
    frame = pd.DataFrame(rows)
    sample = frame.sample(n=min(size, len(frame)), random_state=RANDOM_STATE).reset_index(drop=True)
    sample["text_preview"] = sample["text"].map(lambda text: HANDLE_RE.sub("@user", text)[:80])
    sample["url"] = "https://www.youtube.com/watch?v=" + sample["video_id"] + "&lc=" + sample["comment_id"]
    for column in ("ui_text_matches", "ui_time_matches", "checked_by"):
        sample[column] = ""
    return sample.drop(columns=["text"])


def g8_secret_scan(paths: list[Path]) -> list[str]:
    hits = []
    for root in paths:
        files = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
        for path in files:
            with path.open(encoding="utf-8", errors="ignore") as handle:
                if any(KEY_PATTERN.search(line) for line in handle):
                    hits.append(str(path))
    return hits


def run_gates(run_dir: Path, scan_paths: list[Path]) -> dict[str, Any]:
    run = load_run(run_dir)
    return {
        "run_id": run_dir.name, "g1": g1_manifest_status(run), "g2": g2_coverage(run), "g3": g3_duplicates(run),
        "g4": g4_reply_parents(run), "g5": g5_timestamps(run), "g6": g6_reply_reconciliation(run),
        "g7": g7_spot_check_sample(run), "g8": g8_secret_scan(scan_paths),
    }


# Minimal Markdown table writer, so the report needs no extra dependency (pandas.to_markdown requires tabulate).
def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    divider = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *body])


def _verdict(passed: bool, flag_word: str = "FLAG") -> str:
    return "PASS" if passed else flag_word


def render_markdown(results: dict[str, Any], spot_check_path: Path) -> str:
    g1, coverage, g3, g4, g5, g6, g8 = (results[key] for key in ("g1", "g2", "g3", "g4", "g5", "g6", "g8"))
    low = coverage[coverage["flag_low_coverage"]]
    by_event = coverage.groupby("event_id").agg(videos=("video_id", "count"), api_comments=("api_comment_count", "sum"),
                                                collected=("collected", "sum"), min_coverage=("coverage", "min"))
    by_event["coverage"] = (by_event["collected"] / by_event["api_comments"]).round(3)
    rows = [
        ("G1", "Manifest complete, no failed videos", _verdict(g1["completion_status"] == "complete" and not g1["failed_videos"]),
         f"completion={g1['completion_status']}, videos={g1['video_statuses']}, reply threads={g1['reply_thread_statuses']}"),
        ("G2", f"Coverage per video ≥ {COVERAGE_FLAG}", _verdict(low.empty),
         f"min={coverage['coverage'].min()}, median={coverage['coverage'].median()}, max={coverage['coverage'].max()}, flagged={len(low)}"),
        ("G3", "No duplicate comment IDs", _verdict(g3["thread_duplicate_rows"] == 0 and g3["fetched_reply_duplicate_rows"] == 0), str(g3)),
        ("G4", "Every reply has its top-level comment", _verdict(g4["missing_parent"] == 0), str(g4)),
        ("G5", "Timestamps parse; not before video publish", _verdict(g5["parse_failures"] == 0 and g5["before_video_publish"] == 0), str(g5)),
        ("G6", f"Replies vs totalReplyCount (gap > {int(REPLY_GAP_FLAG * 100)}% flagged)", _verdict(g6["threads_gap_over_20pct"] == 0), str(g6)),
        ("G7", "Manual spot-check of 30 comments in the YouTube UI", "PENDING (manual)", f"sheet: `{spot_check_path.relative_to(REPO_ROOT) if spot_check_path.is_relative_to(REPO_ROOT) else spot_check_path}`"),
        ("G8", "No API key pattern in raw, processed or docs", _verdict(not g8), f"hits={g8}"),
    ]
    lines = [
        "# YouTube collection QA", "",
        f"- Run: `{results['run_id']}`",
        f"- Generated: {utc_now_iso()} by `python -m src.platforms.youtube.qa`",
        f"- Sessions: {g1['sessions']}", "",
        "| Gate | Check | Result | Evidence |", "|---|---|---|---|",
        *[f"| {gate} | {check} | {verdict} | {evidence.replace('|', '/')} |" for gate, check, verdict, evidence in rows],
        "", "## Coverage by event", "", markdown_table(by_event.reset_index()), "",
        "## Videos below coverage threshold", "",
        markdown_table(low[["video_id", "event_id", "api_comment_count", "collected", "coverage"]]) if not low.empty else "None.",
        "", "## Notes", "",
        "- Coverage can exceed 1.0 slightly when comments arrive between the videos.list refresh and the thread crawl.",
        "- Replies counted once across embedded thread replies and comments.list results.",
        "- G5: a video's publishedAt is when it became public. Creators who give members or patrons early access receive comments "
        "before that time, so a small count on non-live creator videos is expected and is not a collection error.",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collect-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=QA_DOC_PATH)
    parser.add_argument("--spot-check-out", type=Path, default=SPOT_CHECK_PATH)
    parser.add_argument("--scan", type=Path, nargs="+", help="Paths to scan for API keys (default: run folder, processed youtube, docs/youtube)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scan = args.scan or [args.collect_run, REPO_ROOT / "data" / "processed" / "youtube", REPO_ROOT / "docs" / "youtube"]
    results = run_gates(args.collect_run, scan)
    args.spot_check_out.parent.mkdir(parents=True, exist_ok=True)
    results["g7"].to_csv(args.spot_check_out, index=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(results, args.spot_check_out), encoding="utf-8")
    print(args.out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
