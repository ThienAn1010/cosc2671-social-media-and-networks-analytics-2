# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

"""Export the ≤10 MB YouTube submission sample (task Y9).

Whole threads are sampled (top-level comment + all its replies) so the sample keeps real reply structure,
stratified by video event × video type with a fixed seed. Interactions and author-video rows are cut to match.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.platforms.youtube.edges import build_author_video
from src.platforms.youtube.storage import REPO_ROOT, utc_now_iso

PROCESSED_ROOT = REPO_ROOT / "data" / "processed" / "youtube"
SAMPLE_ROOT = REPO_ROOT / "data" / "sample" / "youtube"
THREADS_PER_STRATUM = 150
RANDOM_STATE = 42
MAX_SAMPLE_BYTES = 10 * 1024 * 1024
OUTPUT_FILES = ("documents_sample.csv", "interactions_sample.csv", "author_video_sample.csv", "README.md")


def sample_threads(documents: pd.DataFrame, threads_per_stratum: int = THREADS_PER_STRATUM) -> pd.DataFrame:
    usable = documents[documents["exclusion_status"] != "sensitive"]
    top_level = usable[usable["thing"] == "comment"]
    # Shuffle once with a fixed seed, then take the first N per stratum; groupby.apply would drop the stratum columns in pandas 3.
    chosen = (
        top_level.sample(frac=1, random_state=RANDOM_STATE)
        .groupby(["yt_video_event_id", "yt_video_type"])
        .head(threads_per_stratum)
    )
    thread_ids = set(chosen["doc_id"])
    replies = usable[(usable["thing"] == "reply") & usable["parent_doc_id"].isin(thread_ids)]
    videos = usable[(usable["thing"] == "video") & usable["doc_id"].isin(chosen["root_doc_id"])]
    return pd.concat([videos, chosen, replies]).sort_values(["root_doc_id", "created_utc", "doc_id"]).reset_index(drop=True)


def sample_tables(documents: pd.DataFrame, interactions: pd.DataFrame, threads_per_stratum: int = THREADS_PER_STRATUM) -> dict[str, pd.DataFrame]:
    sampled = sample_threads(documents, threads_per_stratum)
    doc_ids = set(sampled["doc_id"])
    edges = interactions[interactions["source_doc_id"].isin(doc_ids) & interactions["target_doc_id"].isin(doc_ids)]
    return {"documents": sampled, "interactions": edges.reset_index(drop=True), "author_video": build_author_video(sampled)}


def readme_text(tables: dict[str, pd.DataFrame], threads_per_stratum: int) -> str:
    docs = tables["documents"]
    strata = docs[docs["thing"] == "comment"].groupby(["yt_video_event_id", "yt_video_type"]).size()
    strata_lines = "\n".join(f"| {event} | {video_type} | {count} |" for (event, video_type), count in strata.items())
    return f"""# YouTube data sample

Generated {utc_now_iso()} by `python -m src.platforms.youtube.sample` (seed {RANDOM_STATE}).

## What is in here
| File | Rows | Content |
|---|---|---|
| `documents_sample.csv` | {len(docs)} | videos, top-level comments and replies ({dict(docs['thing'].value_counts())}) |
| `interactions_sample.csv` | {len(tables['interactions'])} | directed reply edges between sampled comments (replier -> author replied to) |
| `author_video_sample.csv` | {len(tables['author_video'])} | commenter <-> video counts within the sample |

Column definitions: `docs/youtube/data_dictionary.md`. Cleaning rules: `docs/youtube/cleaning_decisions.md`.

## How it was sampled
Up to {threads_per_stratum} top-level comments per video event × video type, drawn at random with a fixed seed, each kept with
**all** of its replies so reply trees are intact. Videos that own a sampled thread are included. Rows flagged `sensitive` are excluded.
This sample shows structure for the network and NLP components; it is not representative of volumes per event.

| Event | Video type | Threads |
|---|---|---|
{strata_lines}

## Why the full dataset is not submitted
- Size: raw collection ≈145 MB and processed tables ≈32 MB, above the 10 MB limit.
- Platform terms: YouTube API Developer Policies limit storing API data to 30 days, so raw data is kept locally and deleted by 2026-10-13.
- Privacy: users appear only as salted hashes; handles, names, emails and phone numbers are masked.
"""


def write_sample(tables: dict[str, pd.DataFrame], out_root: Path, threads_per_stratum: int) -> int:
    out_root.mkdir(parents=True, exist_ok=True)
    tables["documents"].to_csv(out_root / "documents_sample.csv", index=False)
    tables["interactions"].to_csv(out_root / "interactions_sample.csv", index=False)
    tables["author_video"].to_csv(out_root / "author_video_sample.csv", index=False)
    (out_root / "README.md").write_text(readme_text(tables, threads_per_stratum), encoding="utf-8")
    total = sum((out_root / name).stat().st_size for name in OUTPUT_FILES)
    if total > MAX_SAMPLE_BYTES:
        raise ValueError(f"sample is {total / 1e6:.1f} MB, above the 10 MB submission limit; lower --threads-per-stratum")
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--documents", type=Path, default=PROCESSED_ROOT / "documents.parquet")
    parser.add_argument("--interactions", type=Path, default=PROCESSED_ROOT / "interactions.parquet")
    parser.add_argument("--out-root", type=Path, default=SAMPLE_ROOT)
    parser.add_argument("--threads-per-stratum", type=int, default=THREADS_PER_STRATUM)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tables = sample_tables(pd.read_parquet(args.documents), pd.read_parquet(args.interactions), args.threads_per_stratum)
    total = write_sample(tables, args.out_root, args.threads_per_stratum)
    for name, frame in tables.items():
        print(f"{name}: {len(frame)} rows")
    print(f"sample size: {total / 1e6:.2f} MB -> {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
