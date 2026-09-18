"""Draw extra YouTube comments to code so each H4 video has enough human-labelled audience.

Targets the E2 and E3 news and commentary videos in config/youtube/seed_videos.csv. For each
video it counts the eligible adjudicated comments already coded (all validation splits) and
draws enough new comments to reach --target, allowing for the observed eligibility rate.

Draw rules, matching src/analysis/validation.py where they overlap:
- comments and replies on the video, strict-English eligible rows only (exclusion_status
  "eligible"), with a pseudonymous author and non-empty masked text;
- never a comment that is already coded, or whose normalized text matches a coded one;
- at most one comment per author per video, and no author already coded on that video;
- a fixed seed, so the draw can be reproduced.

Text is masked with src.shared.masking.mask_text, like the validation packets. The output holds
comment text, so it stays under data/ (ignored by Git).

Usage (from the SMFR root):
    python -m src.analysis.h4_comment_topup --build --target 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.artifacts import REPO_ROOT
from src.analysis.h4_human_sample import SPLITS, load_comment_labels
from src.analysis.networks import canonical_youtube_video_id
from src.shared.masking import mask_text

DOCUMENTS = REPO_ROOT / "data" / "processed" / "youtube" / "documents.parquet"
SEED_VIDEOS = REPO_ROOT / "config" / "youtube" / "seed_videos.csv"
VALIDATION_ROOT = REPO_ROOT / "data" / "analysis" / "validation"
OUTPUT = REPO_ROOT / "data" / "analysis" / "annotation" / "youtube_comment_topup_sample.csv"
ELIGIBLE_RATE = 0.67  # share of coded YouTube comments that ended up eligible across all splits
SEED = 20260917


def _digest(text: str) -> str:
    normalized = " ".join(str(text).split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build(target: int) -> dict[str, object]:
    seed = pd.read_csv(SEED_VIDEOS, usecols=["video_id", "event_id", "video_type"])
    seed = seed[seed["event_id"].isin(["E2", "E3"]) & seed["video_type"].isin(["news", "commentary"])].copy()
    seed["video_id"] = seed["video_id"].map(canonical_youtube_video_id)

    eligible = load_comment_labels(SPLITS, topup=None)
    have = eligible.groupby("video_id").size()
    seed["have"] = seed["video_id"].map(have).fillna(0).astype(int)
    seed["deficit"] = (target - seed["have"]).clip(lower=0)
    seed["draw"] = seed["deficit"].map(lambda value: math.ceil(value / ELIGIBLE_RATE) if value else 0)

    coded = pd.concat(
        [pd.read_csv(VALIDATION_ROOT / f"coordinator_labels_{split}.csv", usecols=["platform", "doc_id", "author_key", "cluster_id", "text_digest"], keep_default_na=False)
         for split in SPLITS],
        ignore_index=True,
    )
    coded = coded[coded["platform"].eq("youtube")]
    coded_docs = set(coded["doc_id"])
    coded_authors = set(zip(coded["cluster_id"], coded["author_key"]))

    columns = ["doc_id", "thing", "root_doc_id", "parent_doc_id", "author_hash", "yt_video_id", "yt_video_event_id", "exclusion_status", "text_clean"]
    documents = pd.read_parquet(DOCUMENTS, columns=columns)
    lookup = documents.set_index("doc_id")["text_clean"].to_dict()
    pool = documents[
        documents["thing"].ne("video")
        & documents["exclusion_status"].eq("eligible")
        & documents["author_hash"].fillna("").ne("")
        & ~documents["doc_id"].isin(coded_docs)
    ].copy()
    pool["video_id"] = pool["yt_video_id"].map(canonical_youtube_video_id)
    pool = pool[pool["video_id"].isin(set(seed.loc[seed["draw"] > 0, "video_id"]))]
    pool["text_for_annotation"] = pool["text_clean"].map(mask_text)
    pool = pool[pool["text_for_annotation"].str.strip().ne("")]
    pool["text_digest"] = pool["text_for_annotation"].map(_digest)
    pool = pool[~pool["text_digest"].isin(set(coded["text_digest"]))]
    pool["cluster_id"] = pool["root_doc_id"].astype(str)
    pool = pool[[(cluster, author) not in coded_authors for cluster, author in zip(pool["cluster_id"], pool["author_hash"])]]
    pool = pool.drop_duplicates("text_digest")

    rng = np.random.default_rng(SEED)
    pool["random_key"] = rng.random(len(pool))
    pool = pool.sort_values("random_key").drop_duplicates(["video_id", "author_hash"])
    wanted = seed.set_index("video_id")["draw"]
    pool["rank"] = pool.groupby("video_id").cumcount()
    sample = pool[pool["rank"] < pool["video_id"].map(wanted)].copy()

    sample["parent_context"] = sample["parent_doc_id"].map(lookup).fillna("").map(mask_text)
    sample["root_context"] = sample["root_doc_id"].map(lookup).fillna("").map(mask_text)
    sample = sample.sort_values("random_key").reset_index(drop=True)
    sample["annotation_id"] = [f"yt-topup-{index + 1:04d}" for index in range(len(sample))]
    sample["platform"] = "youtube"
    sample["split"] = "h4_topup"
    sample["author_key"] = sample["author_hash"]
    sample["case_window_id"] = sample["yt_video_event_id"]
    out = sample[["annotation_id", "platform", "split", "doc_id", "author_key", "cluster_id", "case_window_id", "video_id",
                  "text_digest", "random_key", "text_for_annotation", "parent_context", "root_context"]]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False, lineterminator="\n")

    short = sample.groupby("video_id").size().reindex(seed["video_id"]).fillna(0).astype(int)
    shortfall = int((seed.set_index("video_id")["draw"] - short).clip(lower=0).sum())
    return {
        "output": str(OUTPUT.relative_to(REPO_ROOT)),
        "target_per_video": target,
        "videos": int(len(seed)),
        "videos_needing_more": int((seed["draw"] > 0).sum()),
        "eligible_already_coded": int(seed["have"].sum()),
        "comments_drawn": int(len(out)),
        "videos_short_of_pool": int(((seed.set_index("video_id")["draw"] - short) > 0).sum()),
        "draw_shortfall": shortfall,
        "replies": int(sample["thing"].eq("reply").sum()),
        "by_event": out["case_window_id"].value_counts().to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", action="store_true", required=True)
    parser.add_argument("--target", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(build(args.target), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
