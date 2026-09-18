"""Frozen source inventory and analytical-population contracts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.analysis.artifacts import ANALYSIS_ROOT, REPO_ROOT, read_json, relative_path, source_record, utc_now_iso, write_json
from src.analysis.scope import REDDIT_EVENT_BY_CASE, check_scope

PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
POPULATION_ARTIFACT = ANALYSIS_ROOT / "populations" / "population_manifest.json"
REDDIT_RELEVANCE_AUDIT = ANALYSIS_ROOT / "annotation" / "reddit_thread_relevance.csv"

POPULATION_IDS = (
    "R_AU_LEGISLATION",
    "R_AU_IMPLEMENTATION",
    "R_REPLY_CORE",
    "B_REPLY_CORE",
    "R_BROKER_CORE",
    "Y_VIDEO_E2E3",
)

POPULATION_RULES: dict[str, dict[str, Any]] = {
    "R_AU_LEGISLATION": {
        "platform": "reddit",
        "unit": "document",
        "status_without_audit": "human_audit_required",
        "definition": "Substantive English human documents in audited-relevant Reddit threads in AU_LEGISLATION.",
        "rule": "case_window=AU_LEGISLATION AND thread_relevant=true AND human_only=true AND English/inclusive-language=true AND substantive=true",
    },
    "R_AU_IMPLEMENTATION": {
        "platform": "reddit",
        "unit": "document",
        "status_without_audit": "human_audit_required",
        "definition": "Substantive English human documents in audited-relevant Reddit threads in AU_IMPLEMENTATION.",
        "rule": "case_window=AU_IMPLEMENTATION AND thread_relevant=true AND human_only=true AND English/inclusive-language=true AND substantive=true",
    },
    "R_REPLY_CORE": {
        "platform": "reddit",
        "unit": "observed_reply_edge",
        "status_without_audit": "human_audit_required",
        "definition": "Valid author-to-author reply edges in audited-relevant Reddit threads across proposal case windows.",
        "rule": "case_window in core AND accepted_thread=true AND valid_parent=true AND distinct_nonempty_author_hashes=true",
    },
    "B_REPLY_CORE": {
        "platform": "bluesky",
        "unit": "observed_reply_edge",
        "status_without_audit": "candidate_network_ready",
        "definition": "Observed replies descending from phrase-exact relevant roots in E2/E3 core windows, excluding labelled spam/repeater endpoints.",
        "rule": "root_phrase_exact=true AND event in {E2,E3} AND observed_parent=true AND spam/repeater=false",
    },
    "R_BROKER_CORE": {
        "platform": "reddit",
        "unit": "author",
        "status_without_audit": "human_audit_required",
        "definition": "R_REPLY_CORE authors with at least five classifiable documents in an analysed connected component.",
        "rule": "R_REPLY_CORE AND classifiable_document_count>=5 AND analysed_component=true",
    },
    "Y_VIDEO_E2E3": {
        "platform": "youtube",
        "unit": "selected_video",
        "status_without_audit": "metadata_audit_required",
        "definition": "Frozen selected E2/E3 videos; the inferential subset requires adjudicated metadata and at least 20 strict-eligible comments.",
        "rule": "frozen_seed_event in {E2,E3} AND metadata_double_coded=true; inferential_subset=strict_eligible_comments>=20",
    },
}


def _parquet_source(path: Path, role: str) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    return source_record(path, role, rows=parquet.metadata.num_rows, columns=parquet.schema.names)


def _csv_source(path: Path, role: str) -> dict[str, Any]:
    header = pd.read_csv(path, nrows=0)
    rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
    return source_record(path, role, rows=max(rows, 0), columns=list(header.columns))


def _json_source(path: Path, role: str) -> dict[str, Any]:
    return source_record(path, role)


def _jsonl_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _jsonl_source(path: Path, role: str) -> dict[str, Any]:
    return source_record(path, role, rows=_jsonl_rows(path))


def frozen_sources() -> list[dict[str, Any]]:
    paths: list[tuple[Path, str, str]] = [
        (PROCESSED_ROOT / "reddit_age_gate" / "analysis_corpus.parquet", "processed_primary", "parquet"),
        (PROCESSED_ROOT / "reddit_age_gate" / "prepare_manifest.json", "processed_manifest", "json"),
        (REPO_ROOT / "data" / "reddit_age_gate_full" / "arctic_shift_20260915_111812" / "collection_manifest.json", "frozen_collection_manifest", "json"),
        (REPO_ROOT / "data" / "reddit_age_gate_full" / "arctic_shift_20260915_111812" / "posts_raw.jsonl", "frozen_raw_restricted", "jsonl"),
        (REPO_ROOT / "data" / "reddit_age_gate_full" / "arctic_shift_20260915_111812" / "comments_raw.jsonl", "frozen_raw_restricted", "jsonl"),
        (PROCESSED_ROOT / "youtube" / "documents.parquet", "processed_primary", "parquet"),
        (PROCESSED_ROOT / "youtube" / "interactions.parquet", "processed_network", "parquet"),
        (PROCESSED_ROOT / "youtube" / "author_video.parquet", "processed_network", "parquet"),
        (PROCESSED_ROOT / "youtube" / "prepare_manifest.json", "processed_manifest", "json"),
        (PROCESSED_ROOT / "youtube" / "edges_manifest.json", "processed_manifest", "json"),
        (REPO_ROOT / "config" / "youtube" / "seed_videos.csv", "frozen_selection_registry", "csv"),
        (PROCESSED_ROOT / "bluesky" / "posts.parquet", "processed_primary", "parquet"),
        (PROCESSED_ROOT / "bluesky" / "post_query.parquet", "processed_query_pairs", "parquet"),
        (PROCESSED_ROOT / "bluesky" / "thread_edges.parquet", "processed_network", "parquet"),
        (PROCESSED_ROOT / "bluesky" / "reply_edges.parquet", "processed_network", "parquet"),
        (PROCESSED_ROOT / "bluesky" / "prepare_manifest.json", "processed_manifest", "json"),
        (PROCESSED_ROOT / "google_trends" / "google_trends_daily.csv", "processed_context_approval_gated", "csv"),
        (PROCESSED_ROOT / "google_trends" / "google_trends_scaling_factors.csv", "processed_context_approval_gated", "csv"),
    ]
    records = []
    for path, role, kind in paths:
        if kind == "parquet":
            records.append(_parquet_source(path, role))
        elif kind == "csv":
            records.append(_csv_source(path, role))
        elif kind == "jsonl":
            records.append(_jsonl_source(path, role))
        else:
            records.append(_json_source(path, role))
    youtube_raw = REPO_ROOT / "data" / "raw" / "youtube" / "youtube_20260913_160448"
    for path in sorted(youtube_raw.glob("*")):
        if path.is_file() and path.name != ".DS_Store":
            records.append(_jsonl_source(path, "frozen_raw_restricted") if path.suffix == ".jsonl" else _json_source(path, "frozen_raw_restricted"))
    for path in sorted((REPO_ROOT / "data" / "raw" / "google_trends").rglob("*.csv")):
        records.append(_csv_source(path, "frozen_raw_restricted"))
    return records


def _reddit_relevance_audit(candidate_threads: set[str]) -> dict[str, Any]:
    if not REDDIT_RELEVANCE_AUDIT.exists():
        return {"status": "missing", "audited_thread_count": 0, "accepted_thread_ids": None}
    audit = pd.read_csv(REDDIT_RELEVANCE_AUDIT, keep_default_na=False)
    required = {"thread_id", "coder_a_relevant", "coder_b_relevant", "adjudicated_relevant"}
    if audit.empty or required - set(audit.columns) or audit["thread_id"].astype(str).str.strip().eq("").any() or audit["thread_id"].duplicated().any():
        raise ValueError("invalid Reddit relevance audit")
    for column in sorted(required - {"thread_id"}):
        values = audit[column].astype(str).str.strip().str.casefold()
        if not values.isin({"yes", "no", "true", "false", "1", "0"}).all():
            raise ValueError(f"invalid Reddit relevance audit values in {column}")
        audit[column] = values.isin({"yes", "true", "1"})
    audited_threads = set(audit["thread_id"].astype(str))
    if audited_threads != candidate_threads:
        raise ValueError("Reddit relevance audit does not cover every frozen core thread")
    return {
        "status": "valid",
        "audited_thread_count": len(audited_threads),
        "accepted_thread_count": int(audit["adjudicated_relevant"].sum()),
        "accepted_thread_ids": set(audit.loc[audit["adjudicated_relevant"], "thread_id"].astype(str)),
    }


def _reddit_candidate_counts() -> dict[str, int]:
    path = PROCESSED_ROOT / "reddit_age_gate" / "analysis_corpus.parquet"
    columns = [
        "record_id", "event_id", "thing", "human_only", "is_english", "is_language_uncertain", "schema_valid",
        "text_topic", "is_url_only", "is_no_substantive_text", "author_hash", "thread_id", "parent_record_id",
        "exclusion_status",
    ]
    frame = pd.read_parquet(path, columns=columns)
    core = frame["event_id"].isin(REDDIT_EVENT_BY_CASE.values()) & frame["thread_id"].fillna("").astype(str).ne("")
    audit = _reddit_relevance_audit(set(frame.loc[core, "thread_id"].astype(str)))
    eligible_threads = audit["accepted_thread_ids"]
    eligible = core if eligible_threads is None else frame["thread_id"].astype(str).isin(eligible_threads)
    substantive = (
        eligible
        & frame["schema_valid"].fillna(False).astype(bool)
        & frame["human_only"].fillna(False).astype(bool)
        & frame["is_english"].fillna(False).astype(bool)
        & frame["text_topic"].fillna("").astype(str).str.strip().ne("")
        & ~frame["is_url_only"].fillna(False).astype(bool)
        & ~frame["is_no_substantive_text"].fillna(False).astype(bool)
    )
    case_masks = {case: frame["event_id"].eq(event) & substantive for case, event in REDDIT_EVENT_BY_CASE.items()}
    broker_author_counts = frame.loc[substantive & frame["author_hash"].fillna("").astype(str).ne("")].groupby("author_hash").size()
    comments = frame[
        substantive
        & frame["thing"].eq("comment")
        & frame["parent_record_id"].fillna("").astype(str).ne("")
        & frame["author_hash"].fillna("").astype(str).ne("")
    ].copy()
    parents = frame.loc[eligible, ["record_id", "author_hash", "thread_id", "event_id", "schema_valid", "human_only", "is_english", "text_topic", "is_url_only", "is_no_substantive_text"]].copy()
    parent_substantive = (
        parents["schema_valid"].fillna(False).astype(bool)
        & parents["human_only"].fillna(False).astype(bool)
        & parents["is_english"].fillna(False).astype(bool)
        & parents["text_topic"].fillna("").astype(str).str.strip().ne("")
        & ~parents["is_url_only"].fillna(False).astype(bool)
        & ~parents["is_no_substantive_text"].fillna(False).astype(bool)
        & parents["author_hash"].fillna("").astype(str).ne("")
    )
    parents = parents.assign(parent_substantive=parent_substantive).rename(
        columns={"record_id": "target_record_id", "author_hash": "target_author_hash", "thread_id": "parent_thread_id", "event_id": "parent_event_id"}
    )
    reply = comments.rename(columns={"record_id": "source_record_id", "author_hash": "source_author_hash"}).merge(
        parents[["target_record_id", "target_author_hash", "parent_thread_id", "parent_event_id", "parent_substantive"]],
        left_on="parent_record_id",
        right_on="target_record_id",
        how="inner",
    )
    reply = reply[
        reply["parent_substantive"]
        & reply["source_author_hash"].ne(reply["target_author_hash"])
        & reply["event_id"].eq(reply["parent_event_id"])
        & reply["thread_id"].eq(reply["parent_thread_id"])
    ].copy()
    connected_authors = set(reply["source_author_hash"].astype(str)) | set(reply["target_author_hash"].astype(str))
    broker_author_counts = broker_author_counts[broker_author_counts.index.astype(str).isin(connected_authors)]
    counts = {
        "R_AU_LEGISLATION": int(case_masks["AU_LEGISLATION"].sum()),
        "R_AU_IMPLEMENTATION": int(case_masks["AU_IMPLEMENTATION"].sum()),
        "R_REPLY_CORE": int(len(reply)),
        "R_BROKER_CORE": int((broker_author_counts >= 5).sum()),
    }
    return counts


def _bluesky_candidate_count() -> int:
    root = PROCESSED_ROOT / "bluesky"
    posts = pd.read_parquet(root / "posts.parquet", columns=["doc_id", "event_window", "is_reply", "in_search", "is_english", "is_meme", "is_repeat_burst", "author_hash", "root_doc_id"])
    authors = pd.read_parquet(root / "authors.parquet", columns=["author_hash", "account_class"])
    posts = posts.merge(authors, on="author_hash", how="left")
    strict = posts["is_english"].fillna(False).astype(bool) & ~posts["is_meme"].fillna(False).astype(bool) & ~posts["is_repeat_burst"].fillna(False).astype(bool) & ~posts["account_class"].isin({"labelled_spam", "repeater"}) & posts["author_hash"].fillna("").ne("")
    exact = pd.read_parquet(root / "post_query.parquet", columns=["doc_id", "phrase_exact"])
    roots = set(posts.loc[posts["doc_id"].isin(set(exact.loc[exact["phrase_exact"].fillna(False).astype(bool), "doc_id"])) & posts["event_window"].isin(["E2", "E3"]) & posts["in_search"].fillna(False).astype(bool) & ~posts["is_reply"].fillna(False).astype(bool) & strict, "doc_id"])
    raw_edges = pd.read_parquet(root / "thread_edges.parquet", columns=["source_doc_id", "target_doc_id"])
    children: dict[str, list[str]] = defaultdict(list)
    for row in raw_edges.itertuples(index=False):
        children[str(row.target_doc_id)].append(str(row.source_doc_id))
    frontier, seen, selected = set(roots), set(roots), set()
    while frontier:
        next_frontier = set()
        for parent in frontier:
            for child in children.get(parent, []):
                selected.add(child)
                if child not in seen:
                    seen.add(child)
                    next_frontier.add(child)
        frontier = next_frontier
    posts = posts.assign(strict=strict.to_numpy())
    endpoints = posts.set_index("doc_id")
    count = 0
    for row in raw_edges.loc[raw_edges["source_doc_id"].isin(selected)].itertuples(index=False):
        if row.source_doc_id not in endpoints.index or row.target_doc_id not in endpoints.index:
            continue
        source, target = endpoints.loc[row.source_doc_id], endpoints.loc[row.target_doc_id]
        if source["event_window"] not in {"E2", "E3"} or target["event_window"] not in {"E2", "E3"}:
            continue
        if not bool(source["strict"]) or not bool(target["strict"]):
            continue
        if str(source["author_hash"]) == str(target["author_hash"]):
            continue
        count += 1
    return count


def _youtube_candidate_count() -> int:
    docs = pd.read_parquet(PROCESSED_ROOT / "youtube" / "documents.parquet", columns=["doc_id", "thing", "root_doc_id", "yt_video_event_id", "exclusion_status"])
    comments = docs[docs["thing"].ne("video")]
    strict = comments[comments["exclusion_status"].eq("eligible")].groupby("root_doc_id").size()
    videos = docs[docs["thing"].eq("video") & docs["yt_video_event_id"].isin(["E2", "E3"])]
    return int(videos["doc_id"].isin(strict[strict.ge(20)].index).sum())


def _youtube_population_details() -> dict[str, int]:
    seed = pd.read_csv(REPO_ROOT / "config" / "youtube" / "seed_videos.csv", usecols=["video_id", "event_id"])
    selected = int(seed["event_id"].isin(["E2", "E3"]).sum())
    eligible = _youtube_candidate_count()
    return {"frozen_selected_video_count": selected, "videos_excluded_for_fewer_than_20_strict_comments": selected - eligible}


def build_population_manifest() -> dict[str, Any]:
    scope = check_scope()
    reddit_counts = _reddit_candidate_counts()
    reddit_path = PROCESSED_ROOT / "reddit_age_gate" / "analysis_corpus.parquet"
    reddit_threads = pd.read_parquet(reddit_path, columns=["event_id", "thread_id"])
    reddit_core = reddit_threads[reddit_threads["event_id"].isin(REDDIT_EVENT_BY_CASE.values()) & reddit_threads["thread_id"].fillna("").astype(str).ne("")]
    reddit_audit = _reddit_relevance_audit(set(reddit_core["thread_id"].astype(str)))
    population_counts = {
        **reddit_counts,
        "B_REPLY_CORE": _bluesky_candidate_count(),
        "Y_VIDEO_E2E3": _youtube_candidate_count(),
    }
    youtube_details = _youtube_population_details()
    registry = []
    for population_id in POPULATION_IDS:
        rule = dict(POPULATION_RULES[population_id])
        rule.update({"population_id": population_id, "candidate_count": population_counts[population_id], "ready_for_confirmatory": False})
        if population_id == "Y_VIDEO_E2E3":
            rule.update(youtube_details)
        registry.append(rule)
    return {
        "schema_version": "populations.v1",
        "population_version": "age-gate-populations-2026-09-15",
        "created_at_utc": utc_now_iso(),
        "command": "python -m src.analysis.run populations --check",
        "scope_artifact": relative_path(ANALYSIS_ROOT / "scope" / "scope_manifest.json"),
        "scope_version": scope["scope_version"],
        "frozen_sources": frozen_sources(),
        "population_registry": registry,
        "relevance_audit": {
            "reddit_thread_audit_path": relative_path(REDDIT_RELEVANCE_AUDIT),
            "status": "missing_human_audit" if reddit_audit["status"] == "missing" else reddit_audit["status"],
            "audited_thread_count": reddit_audit.get("audited_thread_count", 0),
            "accepted_thread_count": reddit_audit.get("accepted_thread_count", 0),
            "keyword_and_legacy_masks_are_sensitivity_only": True,
        },
        "human_label_boundary": {
            "required_for": ["T5", "T6", "T7", "confirmatory_frame_stance_sentiment_claims"],
            "gold_labels_present": False,
            "label_files": [
                "data/analysis/validation/labels_development.csv",
                "data/analysis/validation/labels_evaluation.csv",
                "data/analysis/validation/labels_reserve.csv",
            ],
            "instruction": "Do not use existing model predictions as gold labels.",
        },
        "denominator_policy": {
            "candidate_counts_are_not_headline_denominators": True,
            "reddit_count_basis": "audited_relevant_threads" if reddit_audit["status"] == "valid" else "pre_audit_candidate",
            "missing_human_audit_state": "inconclusive_or_fallback",
            "no_cross_platform_identity_join": True,
        },
    }


def write_population_manifest(path: Path = POPULATION_ARTIFACT) -> dict[str, Any]:
    payload = build_population_manifest()
    write_json(path, payload)
    return {"status": "built", "artifact": relative_path(path), "populations": len(payload["population_registry"])}


def check_populations(path: Path = POPULATION_ARTIFACT) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"population manifest is missing: {path}; run the explicit populations build first")
    expected = build_population_manifest()
    actual = read_json(path)
    comparable_expected = {key: value for key, value in expected.items() if key != "created_at_utc"}
    comparable_actual = {key: value for key, value in actual.items() if key != "created_at_utc"}
    if comparable_actual != comparable_expected:
        raise ValueError(f"population manifest is stale or inconsistent at {path}")
    return {"status": "valid", "artifact": relative_path(path), "populations": len(actual["population_registry"])}
