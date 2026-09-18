#!/usr/bin/env python3
# Student name: Nguyen Hoang Thien An
# Student ID: s3825455
# Assignment 1 - Social Media and Networks Analytics

"""Prepare Arctic Shift Reddit JSONL for analysis.

Creates auditable Parquet and CSV outputs while retaining parsed rows with
explicit exclusion status instead of silent dropping.
"""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

try:
    from src.prepare_corpus_records import (
        add_corpus_flags,
        apply_language_overrides,
        build_prepare_manifest,
        finalize_analysis_masks,
        load_language_overrides,
        read_jsonl,
        require_complete_collection_manifest,
        research_window_from_manifest,
        unwrap_comment,
        unwrap_post,
        write_manifest,
        write_submission_sample,
        write_outputs,
    )
    from src.prepare_corpus_text import (
        build_language_detector,
        build_stopwords,
        build_vader_analyzer,
        ensure_nltk_resources,
        enrich_rows,
        load_spacy_model,
    )
except ModuleNotFoundError:
    from prepare_corpus_records import (
        add_corpus_flags,
        apply_language_overrides,
        build_prepare_manifest,
        finalize_analysis_masks,
        load_language_overrides,
        read_jsonl,
        require_complete_collection_manifest,
        research_window_from_manifest,
        unwrap_comment,
        unwrap_post,
        write_manifest,
        write_submission_sample,
        write_outputs,
    )
    from prepare_corpus_text import (
        build_language_detector,
        build_stopwords,
        build_vader_analyzer,
        ensure_nltk_resources,
        enrich_rows,
        load_spacy_model,
    )

# Each run writes three output stems, each as both Parquet and CSV.
DEFAULT_OUT_ROOT = Path("data/processed")
OUTPUT_STEMS = ("posts_clean", "comments_clean", "analysis_corpus")


# Refuse to clobber an existing processed folder unless --overwrite was passed.
def ensure_output_safe(out_root: Path, overwrite: bool, submission_sample: Path | None = None) -> None:
    targets = [out_root / f"{stem}.{ext}" for stem in OUTPUT_STEMS for ext in ("parquet", "csv")]
    targets.append(out_root / "prepare_manifest.json")
    if submission_sample is not None:
        targets.append(submission_sample)
    existing = [str(path) for path in targets if path.exists()]
    if existing and not overwrite:
        raise SystemExit("processed outputs already exist. choose new --out-root or pass --overwrite: " + ", ".join(existing))


# Full preprocessing run: read raw JSONL, clean, flag, validate and write outputs.
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing processed outputs in --out-root")
    parser.add_argument("--skip-topics", action="store_true", help="Skip spaCy topic tokenization; sentiment and metadata still run")
    parser.add_argument("--allow-partial", action="store_true", help="Allow preprocessing when collection manifest is missing or incomplete")
    parser.add_argument(
        "--submission-sample",
        type=Path,
        help="Optional redacted CSV for submission; raw text and Reddit IDs are removed",
    )
    parser.add_argument("--submission-sample-size", type=int, default=500)
    parser.add_argument(
        "--language-overrides",
        type=Path,
        help="Optional CSV with thing, record_id and manual_language_override columns",
    )
    args = parser.parse_args()

    posts_file = args.raw_root / "posts_raw.jsonl"
    comments_file = args.raw_root / "comments_raw.jsonl"
    # Fail fast on missing inputs rather than producing a half-empty corpus.
    if not posts_file.exists():
        raise SystemExit(f"missing {posts_file}")
    if not comments_file.exists():
        raise SystemExit(f"missing {comments_file}")
    if args.submission_sample_size < 1:
        raise SystemExit("--submission-sample-size must be positive")
    ensure_output_safe(args.out_root, args.overwrite, args.submission_sample)

    # Gate the run on a complete collection; a partial one yields plausible but wrong totals.
    collection_manifest = require_complete_collection_manifest(args.raw_root, allow_partial=args.allow_partial)
    research_start_epoch, research_end_epoch = research_window_from_manifest(collection_manifest)
    nltk_resources = ensure_nltk_resources()
    # Load the models once and reuse them across the collected corpus.
    nlp = None if args.skip_topics else load_spacy_model()
    vader_analyzer = build_vader_analyzer()
    detector = build_language_detector()
    stopword_set = set() if args.skip_topics else build_stopwords()
    author_hash_salt = secrets.token_bytes(32)

    # Parse both files into the fixed schema, then enrich each with tokens, language and VADER.
    posts_rows, posts_parsed = read_jsonl(
        posts_file,
        unwrap_post,
        research_start_epoch=research_start_epoch,
        research_end_epoch=research_end_epoch,
        author_hash_salt=author_hash_salt,
    )
    comments_rows, comments_parsed = read_jsonl(
        comments_file,
        unwrap_comment,
        research_start_epoch=research_start_epoch,
        research_end_epoch=research_end_epoch,
        author_hash_salt=author_hash_salt,
    )
    enrich_rows(posts_rows, nlp, detector, stopword_set, vader_analyzer=vader_analyzer)
    enrich_rows(comments_rows, nlp, detector, stopword_set, vader_analyzer=vader_analyzer)

    # Duplicate and bot flags need the combined corpus, so posts and comments are merged first.
    analysis_rows = add_corpus_flags(posts_rows + comments_rows)
    override_count = apply_language_overrides(analysis_rows, load_language_overrides(args.language_overrides))
    finalize_analysis_masks(analysis_rows)

    # Split back apart so the per-type outputs carry the corpus-wide flags.
    posts_rows = [row for row in analysis_rows if row["thing"] == "post"]
    comments_rows = [row for row in analysis_rows if row["thing"] == "comment"]

    _, _, posts_frame = write_outputs(posts_rows, "posts_clean", args.out_root)
    _, _, comments_frame = write_outputs(comments_rows, "comments_clean", args.out_root)
    _, _, analysis_frame = write_outputs(analysis_rows, "analysis_corpus", args.out_root)
    if args.submission_sample is not None:
        write_submission_sample(analysis_frame, args.submission_sample, args.submission_sample_size)

    # Manifest records hashes, versions and counts so the run can be audited later.
    manifest = build_prepare_manifest(
        raw_root=args.raw_root,
        posts_frame=posts_frame,
        comments_frame=comments_frame,
        analysis_frame=analysis_frame,
        parsed_counts={"posts_raw": posts_parsed, "comments_raw": comments_parsed},
        collection_manifest=collection_manifest,
        nltk_resources=nltk_resources,
        spacy_model=nlp,
        stopword_set=stopword_set,
        submission_sample=args.submission_sample,
        submission_sample_size=args.submission_sample_size if args.submission_sample is not None else None,
    )
    manifest["language_overrides"] = {"path": str(args.language_overrides or ""), "applied_rows": override_count}
    manifest_path = write_manifest(args.out_root, manifest)
    print(json.dumps(manifest["summaries"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
