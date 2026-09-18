#!/usr/bin/env python3
# Student name: Nguyen Hoang Thien An
# Student ID: s3825455
# Assignment 1 - Social Media and Networks Analytics

"""Create manual validation sample from processed analysis corpus."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_PROCESSED_ROOT = Path("data/processed")
DEFAULT_OUTPUT = Path("data/validation/manual_validation_sample.csv")
# Fixed seed so the same records are drawn on every rerun.
RANDOM_STATE = 42
# Phrases that often signal sarcasm, deliberately targeted as hard cases for the lexicon.
SARCASM_HINT_RE = r"(?:^|\W)(?:yeah right|sure jan|as if|totally|great job|love that for us|what a joke|lol sure|obviously)(?:\W|$)"


# Match one frame in the pipe-separated multi-label representation.
def frame_mask(frame: pd.DataFrame, label: str) -> pd.Series:
    values = frame["relevance_frames"] if "relevance_frames" in frame else frame["relevance_level"]
    pattern = rf"(?:^|\|){re.escape(label)}(?:\||$)"
    return values.astype(str).str.contains(pattern, regex=True, na=False)


# Draw a quota from one stratum, skipping records another stratum already took.
def take_without_replacement(
    frame: pd.DataFrame,
    mask: pd.Series,
    count: int,
    selected_keys: set[str],
    row_keys: pd.Series,
    label: str,
) -> pd.DataFrame:
    subset = frame.loc[mask].copy()
    subset = subset.loc[~row_keys.loc[subset.index].isin(selected_keys)]
    if subset.empty:
        return subset
    sample_count = min(count, len(subset))
    sampled = subset.sample(n=sample_count, random_state=RANDOM_STATE)
    sampled["sample_bucket"] = label
    selected_keys.update(row_keys.loc[sampled.index].tolist())
    return sampled


# Define every sampling stratum: study frames, corpora, edge cases and VADER extremes.
def build_sample_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    valid_mask = frame["schema_valid"].eq(True)
    analyzable_mask = (
        valid_mask
        & (frame["text_topic"].astype(str).str.strip() != "")
        & frame["is_url_only"].eq(False)
        & frame["is_no_substantive_text"].eq(False)
    )
    # Sample unique text only, so one viral copy-paste cannot dominate the labels.
    unique_mask = frame["unique_text_only"].eq(True)
    relevance_base = (
        analyzable_mask
        & frame["human_only"].eq(True)
        & frame["age_assurance_related"].eq(True)
        & unique_mask
    )
    sentiment_base = (
        valid_mask
        & frame["sentiment_analysis_eligible"].eq(True)
        & unique_mask
        & frame["is_url_only"].eq(False)
        & frame["is_no_substantive_text"].eq(False)
    )
    # Deliberately oversample hard cases: bots, spam, duplicates and uncertain language.
    special_case_base = valid_mask & (
        frame["is_likely_bot_author"].eq(True)
        | frame["is_likely_spam"].eq(True)
        | frame["is_exact_duplicate_text"].eq(True)
        | frame["is_near_duplicate_text"].eq(True)
        | frame["language_review_required"].eq(True)
    )
    return {
        "age_policy": relevance_base & frame_mask(frame, "age_policy"),
        "child_safety": relevance_base & frame_mask(frame, "child_safety"),
        "privacy_surveillance": relevance_base & frame_mask(frame, "privacy_surveillance"),
        "governance": relevance_base & frame_mask(frame, "governance"),
        "circumvention_autonomy": relevance_base & frame_mask(frame, "circumvention_autonomy"),
        "unrelated_discussion": analyzable_mask
        & frame["human_only"].eq(True)
        & unique_mask
        & frame["age_assurance_related"].eq(False),
        "cross_community": relevance_base,
        "low_information": relevance_base & frame["is_low_information"].eq(True),
        "bot_spam_duplicate_language_uncertain": special_case_base,
        "sentiment_edge_negative": sentiment_base & (frame["sentiment_compound"] <= -0.6),
        "sentiment_edge_positive": sentiment_base & (frame["sentiment_compound"] >= 0.6),
        "sentiment_edge_neutral": sentiment_base & (frame["sentiment_compound"] > -0.05) & (frame["sentiment_compound"] < 0.05),
        "sentiment_edge_sarcasm": sentiment_base
        & frame["text_sentiment"].astype(str).str.contains(SARCASM_HINT_RE, case=False, regex=True, na=False),
    }


# Build the stratified sample and write it with blank columns for a human to fill in.
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    analysis_path = args.processed_root / "analysis_corpus.parquet"
    if not analysis_path.exists():
        raise SystemExit(f"missing {analysis_path}")

    frame = pd.read_parquet(analysis_path)
    row_keys = frame["thing"].astype(str).str.cat(frame["record_id"].astype(str), sep="|")
    selected_keys: set[str] = set()
    masks = build_sample_masks(frame)
    # Quotas per stratum; the edge buckets target where VADER is most likely to be wrong.
    samples = [
        take_without_replacement(frame, masks["age_policy"], 40, selected_keys, row_keys, "age_policy"),
        take_without_replacement(frame, masks["child_safety"], 30, selected_keys, row_keys, "child_safety"),
        take_without_replacement(frame, masks["privacy_surveillance"], 40, selected_keys, row_keys, "privacy_surveillance"),
        take_without_replacement(frame, masks["governance"], 25, selected_keys, row_keys, "governance"),
        take_without_replacement(frame, masks["circumvention_autonomy"], 25, selected_keys, row_keys, "circumvention_autonomy"),
        take_without_replacement(frame, masks["unrelated_discussion"], 30, selected_keys, row_keys, "unrelated_discussion"),
        take_without_replacement(frame, masks["cross_community"], 20, selected_keys, row_keys, "cross_community"),
        take_without_replacement(frame, masks["low_information"], 20, selected_keys, row_keys, "low_information"),
        take_without_replacement(frame, masks["bot_spam_duplicate_language_uncertain"], 20, selected_keys, row_keys, "bot_spam_duplicate_language_uncertain"),
        take_without_replacement(frame, masks["sentiment_edge_negative"], 8, selected_keys, row_keys, "sentiment_edge_negative"),
        take_without_replacement(frame, masks["sentiment_edge_positive"], 8, selected_keys, row_keys, "sentiment_edge_positive"),
        take_without_replacement(frame, masks["sentiment_edge_neutral"], 7, selected_keys, row_keys, "sentiment_edge_neutral"),
        take_without_replacement(frame, masks["sentiment_edge_sarcasm"], 7, selected_keys, row_keys, "sentiment_edge_sarcasm"),
    ]
    sample = pd.concat([subset for subset in samples if not subset.empty], ignore_index=True)
    # Blank columns only; labelling is done by a human, never generated by this script.
    sample["manual_relevant"] = ""
    sample["manual_sentiment"] = ""
    sample["manual_frame"] = ""
    sample["manual_notes"] = ""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out, index=False)
    print(f"rows: {len(sample)}")
    print(f"output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
