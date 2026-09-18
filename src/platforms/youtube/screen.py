# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

"""Screening helpers for task Y3.

suggest: fill include_suggested / video_type_suggested / reason_suggested using the configured screening rules.
freeze:  validate the human-screened sheet and write config/youtube/seed_videos.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.platforms.youtube.storage import REPO_ROOT, utc_now_iso

LEXICONS_PATH = REPO_ROOT / "config" / "youtube" / "lexicons.json"
SEED_PATH = REPO_ROOT / "config" / "youtube" / "seed_videos.csv"
MAX_PER_EVENT = 50
MIN_PER_EVENT = 20
# A few videos of each main channel type per event, so news vs commentary vs tech audiences can be compared.
MIN_PER_TYPE = 5
MAIN_TYPES = ("news", "commentary", "tech_explainer")
VIDEO_TYPES = {"news", "commentary", "tech_explainer", "official", "other"}
SEED_COLUMNS = [
    "video_id", "event_id", "video_type", "channel_id", "title", "published_at", "view_count_at_screening",
    "comment_count_at_screening", "inclusion_reason", "screened_by", "screened_at_utc",
]


# Channel-name heuristic; official is checked first because e.g. "ABC News" should not win over "Parliament".
def suggest_video_type(channel_title: str, patterns: dict[str, list[str]]) -> str:
    channel = channel_title.lower()
    for video_type in ("official", "news", "tech_explainer"):
        if any(pattern in channel for pattern in patterns[video_type]):
            return video_type
    return "commentary"


def suggest(candidates: pd.DataFrame, patterns: dict[str, list[str]]) -> pd.DataFrame:
    frame = candidates.copy()
    flags = frame["auto_flags"].fillna("")
    frame["video_type_suggested"] = [suggest_video_type(str(name), patterns) for name in frame["channel_title"].fillna("")]
    frame["include_suggested"] = "n"
    frame["reason_suggested"] = "auto_flags: " + flags

    for _, group in frame[flags == ""].groupby("event_id"):
        ranked = group.sort_values("view_count", ascending=False)
        chosen: list[int] = []
        for video_type in MAIN_TYPES:
            chosen += ranked.index[ranked["video_type_suggested"] == video_type][:MIN_PER_TYPE].tolist()
        for index in ranked.index:
            if len(chosen) >= MAX_PER_EVENT:
                break
            if index not in chosen:
                chosen.append(index)
        chosen = chosen[:MAX_PER_EVENT]
        frame.loc[group.index, "reason_suggested"] = f"no auto-flags but beyond top {MAX_PER_EVENT} by view_count"
        frame.loc[chosen, "include_suggested"] = "y"
        frame.loc[chosen, "reason_suggested"] = "no auto-flags; top view_count in event window (type quota applied)"
    return frame


def accept_suggestions(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    # A column left completely blank in the CSV is read as float NaN, which cannot hold "y"/"n" strings.
    for column in ("include", "video_type", "include_suggested", "video_type_suggested"):
        frame[column] = frame[column].fillna("").astype(str)
    blank_include = frame["include"].fillna("").astype(str).str.strip() == ""
    blank_type = frame["video_type"].fillna("").astype(str).str.strip() == ""
    frame.loc[blank_include, "include"] = frame.loc[blank_include, "include_suggested"]
    frame.loc[blank_type, "video_type"] = frame.loc[blank_type, "video_type_suggested"]
    return frame


# Returns (seed, blocking problems, warnings). Blank include counts as "not included".
def freeze(screened: pd.DataFrame, screened_by: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    include = screened["include"].fillna("").astype(str).str.strip().str.lower()
    problems, warnings = [], []
    invalid = sorted(set(include) - {"y", "n", ""})
    if invalid:
        problems.append(f"include has values other than y/n/blank: {invalid}")

    chosen = screened[include == "y"].copy()
    chosen["video_type"] = chosen["video_type"].fillna("").astype(str).str.strip()
    bad_type = chosen.loc[~chosen["video_type"].isin(VIDEO_TYPES), "video_id"].tolist()
    if bad_type:
        problems.append(f"included rows without a valid video_type: {bad_type[:10]}")
    duplicated = chosen.loc[chosen["video_id"].duplicated(), "video_id"].tolist()
    if duplicated:
        problems.append(f"duplicate video_id among included rows: {duplicated[:10]}")
    for event_id, count in chosen.groupby("event_id").size().items():
        if not MIN_PER_EVENT <= count <= MAX_PER_EVENT:
            warnings.append(f"{event_id}: {count} included (target {MIN_PER_EVENT}-{MAX_PER_EVENT})")

    note = chosen["reviewer_note"].fillna("").astype(str).str.strip()
    seed = pd.DataFrame({
        "video_id": chosen["video_id"],
        "event_id": chosen["event_id"],
        "video_type": chosen["video_type"],
        "channel_id": chosen["channel_id"],
        "title": chosen["title"],
        "published_at": chosen["published_at"],
        "view_count_at_screening": chosen["view_count"],
        "comment_count_at_screening": chosen["comment_count"],
        "inclusion_reason": note.where(note != "", chosen["reason_suggested"].fillna("")),
        "screened_by": screened_by,
        "screened_at_utc": utc_now_iso(),
    }, columns=SEED_COLUMNS)
    return seed.sort_values(["event_id", "view_count_at_screening"], ascending=[True, False]), problems, warnings


def print_suggestion_table(frame: pd.DataFrame) -> None:
    chosen = frame[frame["include_suggested"] == "y"]
    table = chosen.groupby(["event_id", "video_type_suggested"]).agg(videos=("video_id", "count"), comment_count=("comment_count", "sum"))
    print(table.to_string())
    totals = chosen.groupby("event_id").agg(videos=("video_id", "count"), comment_count=("comment_count", "sum"))
    print(totals.to_string())
    # Upper bound for threads: comment_count includes replies, so real thread pages are fewer.
    print(f"estimated commentThreads units (upper bound): {int((chosen['comment_count'].fillna(0) // 100 + 1).sum())}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    suggest_cmd = sub.add_parser("suggest")
    suggest_cmd.add_argument("--candidates", type=Path, required=True)
    suggest_cmd.add_argument("--lexicons", type=Path, default=LEXICONS_PATH)
    freeze_cmd = sub.add_parser("freeze")
    freeze_cmd.add_argument("--screened", type=Path, required=True)
    freeze_cmd.add_argument("--screened-by", required=True)
    freeze_cmd.add_argument("--out", type=Path, default=SEED_PATH)
    freeze_cmd.add_argument("--accept-suggestions", action="store_true", help="Fill blank include/video_type from the suggestions")
    freeze_cmd.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.command == "suggest":
        patterns = json.loads(args.lexicons.read_text(encoding="utf-8"))["channel_type_patterns"]
        frame = suggest(pd.read_csv(args.candidates, dtype={"video_id": str}, keep_default_na=False, na_values=[""]), patterns)
        frame.to_csv(args.candidates, index=False)
        print_suggestion_table(frame)
        print(f"suggestions written to {args.candidates}")
        return 0

    screened = pd.read_csv(args.screened, dtype={"video_id": str}, keep_default_na=False, na_values=[""])
    if args.accept_suggestions:
        screened = accept_suggestions(screened)
    seed, problems, warnings = freeze(screened, args.screened_by)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if problems:
        for problem in problems:
            print(f"BLOCKING: {problem}")
        return 1
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite to replace the frozen seed list")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seed.to_csv(args.out, index=False)
    print(seed.groupby(["event_id", "video_type"]).size().to_string())
    print(f"seed videos: {len(seed)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
