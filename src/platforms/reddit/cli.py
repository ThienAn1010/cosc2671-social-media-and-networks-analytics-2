#!/usr/bin/env python3
# Student name: Nguyen Hoang Thien An
# Student ID: s3825455
# Assignment 1 - Social Media and Networks Analytics

"""Collect the Reddit event-window corpus for the Age-Gate Paradox study."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

try:
    from src.platforms.reddit.collector import (
        AGE_ASSURANCE_QUERIES,
        API_BASE,
        COMMENT_TREE_LIMIT,
        STUDY_NAME,
        TARGET_SUBREDDITS,
        build_plan_summary,
        event_window_records,
        run_collection,
        run_self_test,
    )
except ModuleNotFoundError:
    try:
        from platforms.reddit.collector import (
            AGE_ASSURANCE_QUERIES,
            API_BASE,
            COMMENT_TREE_LIMIT,
            STUDY_NAME,
            TARGET_SUBREDDITS,
            build_plan_summary,
            event_window_records,
            run_collection,
            run_self_test,
        )
    except ModuleNotFoundError:
        from collector import (
            AGE_ASSURANCE_QUERIES,
            API_BASE,
            COMMENT_TREE_LIMIT,
            STUDY_NAME,
            TARGET_SUBREDDITS,
            build_plan_summary,
            event_window_records,
            run_collection,
            run_self_test,
        )


def print_plan() -> None:
    planned_tasks, task_count = build_plan_summary()
    print(f"study: {STUDY_NAME}")
    print(f"api: {API_BASE}")
    print(f"target subreddits: {', '.join(f'r/{name}' for name in TARGET_SUBREDDITS)}")
    print(f"queries: {', '.join(AGE_ASSURANCE_QUERIES)}")
    print("event windows:")
    for event in event_window_records():
        print(
            f"  {event['event_id']} ({event['jurisdiction']}): "
            f"{event['start_utc']} <= created_utc < {event['end_utc']}"
        )
    print(f"static post-search task count: {task_count}")
    for task in planned_tasks:
        print(
            f"post search: {task.event_id} {task.window_start_utc}..{task.window_end_utc} "
            f"r/{task.subreddit} query={task.query!r}"
        )
    print("dynamic comment-tree task: one complete available tree per unique matched post within each event window")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default="data/raw", help="Output parent folder or resume folder")
    parser.add_argument("--limit", type=int, default=100, help="Post-search page limit, 1-100")
    parser.add_argument("--delay", type=float, default=1.5, help="Sleep seconds after each successful request")
    parser.add_argument("--max-retries", type=int, default=6, help="Retries for rate limits, 5xx and transient failures")
    parser.add_argument("--min-window-seconds", type=int, default=2, help="Smallest post-search split window")
    parser.add_argument("--resume", action="store_true", help="Resume the compatible collection in --out-root")
    parser.add_argument("--overwrite", action="store_true", help="Write directly into --out-root")
    parser.add_argument("--plan-only", action="store_true", help="Print every static task without calling the API")
    parser.add_argument("--max-requests", type=int, default=None, help="Stop after N API requests in this run")
    parser.add_argument(
        "--comment-tree-limit",
        type=int,
        default=COMMENT_TREE_LIMIT,
        help=f"Tree limit and traversal breadth/depth, 1-{COMMENT_TREE_LIMIT}",
    )
    parser.add_argument("--self-test", action="store_true", help="Run offline collector checks")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0
    if not 1 <= args.limit <= 100:
        raise SystemExit("--limit must be between 1 and 100")
    if args.delay < 0:
        raise SystemExit("--delay must be non-negative")
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be non-negative")
    if args.min_window_seconds < 2:
        raise SystemExit("--min-window-seconds must be at least 2")
    if not 1 <= args.comment_tree_limit <= COMMENT_TREE_LIMIT:
        raise SystemExit(f"--comment-tree-limit must be between 1 and {COMMENT_TREE_LIMIT}")
    if args.max_requests is not None and args.max_requests < 1:
        raise SystemExit("--max-requests must be positive")
    if args.resume and args.overwrite:
        raise SystemExit("use --resume or --overwrite, not both")
    if args.plan_only:
        print_plan()
        return 0

    stats, _, out_dir, completion_status, exit_code = run_collection(args)
    print(json.dumps(asdict(stats), indent=2, sort_keys=True))
    print(f"completion_status: {completion_status}")
    print(f"output: {out_dir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
