"""Materialise frozen-corpus measurement outputs after the reserve gate."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.artifacts import ANALYSIS_ROOT, REPO_ROOT, read_json, relative_path, sha256_file, utc_now_iso, write_json
from src.analysis.measurement import (
    FRAME_LABELS,
    NLI_MODEL_ID,
    ROBERTA_MODEL_ID,
    RESERVE_ASSESSMENT,
    SENTIMENT_SELECTION,
    STANCE_FRAME_SELECTION,
    _frame_labels,
    _frame_pipeline,
    _packet,
    _platform_frames,
    _selected_frame_thresholds,
    _stance_pipeline,
    _stored_selection,
    _transformer_predictions,
    sentiment_from_compound,
    vader_scores,
    check_reserve_assessment,
)
from src.analysis.networks import (
    canonical_youtube_video_id,
    frozen_youtube_video_ids,
    validate_reddit_relevance_audit,
    validate_youtube_metadata_audit,
    youtube_video_event_assignments,
)
from src.shared.case_windows import REDDIT_CASE_BY_EVENT
from src.shared.masking import mask_text

PREDICTIONS_ROOT = ANALYSIS_ROOT / "predictions"
PREDICTIONS_ARTIFACT = PREDICTIONS_ROOT / "full_corpus_predictions.parquet"
PREDICTIONS_MANIFEST = PREDICTIONS_ROOT / "predictions_manifest.json"
PREDICTIONS_SCHEMA_VERSION = "measurement-predictions.v2"
POPULATION_CONTRACT_VERSION = "frozen-populations-v2-audited-reply-membership"
PREDICTION_BATCH_ROWS = 2_000


def _digest_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _base_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        [
            "platform",
            "doc_id",
            "population_id",
            "case_window_id",
            "event_id",
            "author_key",
            "cluster_id",
            "container_id",
            "day",
            "text_digest",
            "text_for_annotation",
        ]
    ].drop_duplicates(["platform", "doc_id"], keep="first")


def _reddit_documents() -> pd.DataFrame:
    path = REPO_ROOT / "data" / "processed" / "reddit_age_gate" / "analysis_corpus.parquet"
    columns = [
        "record_id", "thing", "parent_record_id", "event_id", "author_hash", "thread_id", "schema_valid", "human_only",
        "is_english", "is_language_uncertain", "text_topic", "is_url_only", "is_no_substantive_text", "date", "subreddit",
    ]
    raw = pd.read_parquet(path, columns=columns)
    raw["case_window_id"] = raw["event_id"].map(REDDIT_CASE_BY_EVENT).fillna("none")
    text = raw["text_topic"].fillna("").astype(str)
    candidate_threads = set(raw.loc[raw["case_window_id"].ne("none") & raw["thread_id"].fillna("").astype(str).ne(""), "thread_id"].astype(str))
    relevance = validate_reddit_relevance_audit()
    if relevance["status"] != "valid" or relevance.get("audited_thread_ids") != candidate_threads:
        raise ValueError("Reddit relevance audit must cover every frozen core thread before prediction")
    usable = (
        raw["case_window_id"].ne("none")
        & raw["schema_valid"].fillna(False).astype(bool)
        & raw["human_only"].fillna(False).astype(bool)
        & raw["author_hash"].fillna("").astype(str).ne("")
        & text.str.strip().ne("")
        & ~raw["is_url_only"].fillna(False).astype(bool)
        & ~raw["is_no_substantive_text"].fillna(False).astype(bool)
        & (raw["is_english"].fillna(False).astype(bool) | raw["is_language_uncertain"].fillna(False).astype(bool))
        & raw["thread_id"].astype(str).isin(relevance["accepted_thread_ids"])
    )
    raw = raw.loc[usable].copy()
    parents = raw[["record_id", "thing", "parent_record_id", "event_id", "author_hash", "thread_id"]].copy()
    parents = parents.rename(columns={
        "record_id": "parent_record_id_join",
        "thing": "parent_thing",
        "event_id": "parent_event_id",
        "author_hash": "parent_author_hash",
        "thread_id": "parent_thread_id",
    })
    replies = raw[raw["thing"].eq("comment") & raw["parent_record_id"].fillna("").astype(str).ne("")].merge(
        parents,
        left_on="parent_record_id",
        right_on="parent_record_id_join",
        how="inner",
    )
    reply_ids = set(replies.loc[
        replies["event_id"].eq(replies["parent_event_id"])
        & replies["thread_id"].eq(replies["parent_thread_id"])
        & replies["author_hash"].ne(replies["parent_author_hash"]),
        "record_id",
    ].astype(str))
    raw = raw[(raw["case_window_id"].ne("UK_ENFORCEMENT_CLUSTER")) | raw["record_id"].astype(str).isin(reply_ids)].copy()
    raw["text_for_annotation"] = text.loc[raw.index].map(mask_text)
    case_to_population = {"AU_LEGISLATION": "R_AU_LEGISLATION", "AU_IMPLEMENTATION": "R_AU_IMPLEMENTATION"}
    return pd.DataFrame(
        {
            "platform": "reddit",
            "doc_id": raw["record_id"].astype(str),
            "population_id": raw["case_window_id"].map(case_to_population).fillna("R_REPLY_CORE"),
            "case_window_id": raw["case_window_id"].astype(str),
            "event_id": raw["event_id"].astype(str),
            "author_key": raw["author_hash"].astype(str),
            "cluster_id": raw["thread_id"].astype(str),
            "container_id": raw["subreddit"].fillna("").astype(str),
            "day": pd.to_datetime(raw["date"], errors="coerce").dt.date.astype("string"),
            "text_digest": raw["text_for_annotation"].map(_digest_text),
            "text_for_annotation": raw["text_for_annotation"],
        }
    )


def _bluesky_documents() -> pd.DataFrame:
    path = REPO_ROOT / "data" / "processed" / "bluesky" / "posts.parquet"
    columns = [
        "doc_id", "root_doc_id", "event_window", "day", "author_hash", "is_english", "lang_detect_prob", "n_chars",
        "is_meme", "is_repeat_burst", "is_reply", "in_search", "text_raw",
    ]
    raw = pd.read_parquet(path, columns=columns)
    authors = pd.read_parquet(REPO_ROOT / "data" / "processed" / "bluesky" / "authors.parquet", columns=["author_hash", "account_class"])
    raw = raw.merge(authors, on="author_hash", how="left", validate="many_to_one")
    reply_ids = _bluesky_reply_document_ids(raw)
    text = raw["text_raw"].fillna("").astype(str)
    usable = (
        raw["event_window"].isin(["E2", "E3"])
        & raw["doc_id"].astype(str).isin(reply_ids)
        & raw["author_hash"].fillna("").astype(str).ne("")
        & text.str.strip().ne("")
        & raw["is_english"].fillna(False).astype(bool)
        & ~raw["is_meme"].fillna(False).astype(bool)
        & ~raw["is_repeat_burst"].fillna(False).astype(bool)
        & ~raw["account_class"].isin({"labelled_spam", "repeater"})
    )

    raw = raw.loc[usable].copy()
    raw["text_for_annotation"] = text.loc[raw.index].map(mask_text)
    return pd.DataFrame(
        {
            "platform": "bluesky",
            "doc_id": raw["doc_id"].astype(str),
            "population_id": "B_REPLY_CORE",
            "case_window_id": raw["event_window"].astype(str),
            "event_id": raw["event_window"].astype(str),
            "author_key": raw["author_hash"].astype(str),
            "cluster_id": raw["root_doc_id"].fillna(raw["doc_id"]).astype(str),
            "container_id": "",
            "day": raw["day"].astype("string"),
            "text_digest": raw["text_for_annotation"].map(_digest_text),
            "text_for_annotation": raw["text_for_annotation"],
        }
    )


def _bluesky_reply_document_ids(posts: pd.DataFrame) -> set[str]:
    """Return strict observed descendants of phrase-exact E2/E3 roots."""

    root = REPO_ROOT / "data" / "processed" / "bluesky"
    exact = pd.read_parquet(root / "post_query.parquet", columns=["doc_id", "phrase_exact"])
    exact_ids = set(exact.loc[exact["phrase_exact"].fillna(False).astype(bool), "doc_id"].astype(str))
    eligible = (
        posts["event_window"].isin(["E2", "E3"])
        & posts["is_english"].fillna(False).astype(bool)
        & ~posts["is_meme"].fillna(False).astype(bool)
        & ~posts["is_repeat_burst"].fillna(False).astype(bool)
        & ~posts["account_class"].isin({"labelled_spam", "repeater"})
        & posts["author_hash"].fillna("").astype(str).ne("")
    )
    roots = set(posts.loc[
        eligible
        & posts["in_search"].fillna(False).astype(bool)
        & ~posts["is_reply"].fillna(False).astype(bool)
        & posts["doc_id"].astype(str).isin(exact_ids),
        "doc_id",
    ].astype(str))
    edges = pd.read_parquet(root / "thread_edges.parquet", columns=["source_doc_id", "target_doc_id"])
    children: dict[str, list[str]] = defaultdict(list)
    for row in edges.itertuples(index=False):
        children[str(row.target_doc_id)].append(str(row.source_doc_id))
    frontier, seen, descendants = set(roots), set(roots), set()
    while frontier:
        next_frontier = set()
        for parent in frontier:
            for child in children.get(parent, []):
                descendants.add(child)
                if child not in seen:
                    seen.add(child)
                    next_frontier.add(child)
        frontier = next_frontier
    by_id = posts.assign(_eligible=eligible.to_numpy()).set_index(posts["doc_id"].astype(str))
    accepted: set[str] = set()
    for row in edges.itertuples(index=False):
        source_id, target_id = str(row.source_doc_id), str(row.target_doc_id)
        if source_id not in descendants or source_id not in by_id.index or target_id not in by_id.index:
            continue
        source, target = by_id.loc[source_id], by_id.loc[target_id]
        if bool(source["_eligible"]) and bool(source["is_reply"]) and bool(target["_eligible"]) and str(source["author_hash"]) != str(target["author_hash"]):
            accepted.add(source_id)
    return accepted


def _youtube_documents() -> pd.DataFrame:
    path = REPO_ROOT / "data" / "processed" / "youtube" / "documents.parquet"
    columns = [
        "doc_id", "thing", "root_doc_id", "parent_doc_id", "container_id", "author_hash", "yt_video_event_id",
        "created_date_utc", "exclusion_status", "is_english", "is_language_uncertain", "text_clean",
    ]
    raw = pd.read_parquet(path, columns=columns)
    assignments = youtube_video_event_assignments(raw[["doc_id", "thing", "root_doc_id", "yt_video_event_id"]])
    raw = raw.drop(columns=["yt_video_event_id"]).merge(assignments[["doc_id", "video_event"]], on="doc_id", how="left", validate="one_to_one")
    text = raw["text_clean"].fillna("").astype(str)
    videos = raw[raw["thing"].eq("video") & raw["video_event"].isin(["E2", "E3"])].copy()
    strict_comments = raw[
        raw["thing"].ne("video")
        & raw["exclusion_status"].eq("eligible")
        & raw["is_english"].fillna(False).astype(bool)
        & raw["root_doc_id"].notna()
    ]
    strict_counts = strict_comments.groupby("root_doc_id").size()
    seed_ids = {canonical_youtube_video_id(value) for value in frozen_youtube_video_ids()}
    candidate_ids = set(videos.loc[
        videos["doc_id"].astype(str).isin(seed_ids)
        & videos["doc_id"].isin(strict_counts[strict_counts.ge(20)].index),
        "doc_id",
    ].astype(str))
    metadata = validate_youtube_metadata_audit()
    if metadata["status"] != "valid" or metadata.get("audited_video_ids") != candidate_ids:
        raise ValueError("YouTube metadata audit must cover every frozen E2/E3 video with at least 20 strict comments")
    selected_ids = set(metadata["accepted_video_ids"]) & candidate_ids
    usable = (
        raw["thing"].ne("video")
        & raw["video_event"].isin(["E2", "E3"])
        & raw["exclusion_status"].eq("eligible")
        & raw["is_english"].fillna(False).astype(bool)
        & raw["author_hash"].fillna("").astype(str).ne("")
        & text.str.strip().ne("")
        & raw["root_doc_id"].astype(str).isin(selected_ids)
    )
    raw = raw.loc[usable].copy()
    raw["text_for_annotation"] = text.loc[raw.index].map(mask_text)
    return pd.DataFrame(
        {
            "platform": "youtube",
            "doc_id": raw["doc_id"].astype(str),
            "population_id": "Y_VIDEO_E2E3",
            "case_window_id": raw["video_event"].astype(str),
            "event_id": raw["video_event"].astype(str),
            "author_key": raw["author_hash"].astype(str),
            "cluster_id": raw["root_doc_id"].astype(str),
            "container_id": raw["container_id"].fillna("").astype(str),
            "day": pd.to_datetime(raw["created_date_utc"], errors="coerce").dt.date.astype("string"),
            "text_digest": raw["text_for_annotation"].map(_digest_text),
            "text_for_annotation": raw["text_for_annotation"],
        }
    )


def load_frozen_documents() -> pd.DataFrame:
    return _base_columns(pd.concat([_reddit_documents(), _bluesky_documents(), _youtube_documents()], ignore_index=True))


def _source_paths() -> dict[str, Path]:
    return {
        "reddit": REPO_ROOT / "data" / "processed" / "reddit_age_gate" / "analysis_corpus.parquet",
        "bluesky": REPO_ROOT / "data" / "processed" / "bluesky" / "posts.parquet",
        "bluesky_authors": REPO_ROOT / "data" / "processed" / "bluesky" / "authors.parquet",
        "bluesky_queries": REPO_ROOT / "data" / "processed" / "bluesky" / "post_query.parquet",
        "bluesky_thread_edges": REPO_ROOT / "data" / "processed" / "bluesky" / "thread_edges.parquet",
        "youtube": REPO_ROOT / "data" / "processed" / "youtube" / "documents.parquet",
        "reddit_relevance_audit": REPO_ROOT / "data" / "analysis" / "annotation" / "reddit_thread_relevance.csv",
        "youtube_metadata_audit": REPO_ROOT / "data" / "analysis" / "annotation" / "youtube_video_metadata.csv",
        "youtube_seed_videos": REPO_ROOT / "config" / "youtube" / "seed_videos.csv",
    }


def _reserve_cell_status(task: str, platform: str, selections: dict[str, dict[str, Any]], reserve: dict[str, Any]) -> tuple[str, str]:
    selection = selections[task]
    if selection.get("status") != "selected":
        return "fallback_human_sample", "development_selection_is_predeclared_fallback"
    cell = reserve.get("metrics", {}).get(task, {}).get("by_platform", {}).get(platform, {})
    overall = cell.get("gates", {}).get("passed") is True
    unseen = cell.get("unseen_author", {}).get("gates", {}).get("passed") is True
    if overall and unseen:
        return "validated_automated", "reserve_overall_and_unseen_author_gates_passed"
    return "fallback_human_sample", "reserve_validity_gate_failed"


def _validated_tasks(selections: dict[str, dict[str, Any]], reserve: dict[str, Any], platform: str) -> dict[str, tuple[str, str]]:
    return {task: _reserve_cell_status(task, platform, selections, reserve) for task in ("sentiment", "stance", "frames")}


def _apply_sentiment_predictions(group: pd.DataFrame, candidate: dict[str, Any], split: str, platform: str) -> np.ndarray:
    if candidate.get("name") == ROBERTA_MODEL_ID:
        parts = []
        for start in range(0, len(group), PREDICTION_BATCH_ROWS):
            parts.append(_transformer_predictions("sentiment", group.iloc[start : start + PREDICTION_BATCH_ROWS], f"{split}_batch_{start // PREDICTION_BATCH_ROWS}", platform)[0])
        return np.concatenate(parts) if parts else np.asarray([], dtype=str)
    scores = vader_scores(group["text_for_annotation"])
    selected = candidate["selected"]
    return sentiment_from_compound(scores, selected["negative_threshold"], selected["positive_threshold"])


def _apply_stance_predictions(group: pd.DataFrame, development: pd.DataFrame, candidate: dict[str, Any], split: str, platform: str) -> np.ndarray:
    if candidate.get("name") == NLI_MODEL_ID:
        parts = []
        for start in range(0, len(group), PREDICTION_BATCH_ROWS):
            parts.append(_transformer_predictions("stance", group.iloc[start : start + PREDICTION_BATCH_ROWS], f"{split}_batch_{start // PREDICTION_BATCH_ROWS}", platform)[0])
        return np.concatenate(parts) if parts else np.asarray([], dtype=str)
    model = _stance_pipeline().fit(development["text_for_annotation"], development["adjudicated_stance"].astype(str))
    return model.predict(group["text_for_annotation"])


def _apply_frame_predictions(group: pd.DataFrame, development: pd.DataFrame, candidate: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    labelled = development.loc[development["adjudicated_frame_labels"].notna()].copy()
    labels = np.asarray([[label in _frame_labels(value) for label in FRAME_LABELS] for value in labelled["adjudicated_frame_labels"]], dtype=int)
    model, _ = _frame_pipeline()
    model.fit(labelled["text_for_annotation"], labels)
    probabilities = np.asarray(model.predict_proba(group["text_for_annotation"]), dtype=float)
    thresholds = _selected_frame_thresholds(candidate)
    return probabilities, (probabilities >= thresholds).astype(int)


def build_predictions() -> dict[str, Any]:
    reserve_status = check_reserve_assessment()
    reserve = read_json(RESERVE_ASSESSMENT)
    selections = {task: _stored_selection(task) for task in ("sentiment", "stance", "frames")}
    documents = load_frozen_documents()
    output = documents.drop(columns=["text_for_annotation"]).copy()
    output["prediction_preprocessing_version"] = "masked-text-v2"
    for task in ("sentiment", "stance", "frames"):
        output[f"{task}_status"] = ""
        output[f"{task}_fallback_reason"] = ""
    output["sentiment_prediction"] = pd.NA
    output["stance_prediction"] = pd.NA
    for label in FRAME_LABELS:
        output[f"frame_probability_{label}"] = np.nan
        output[f"frame_decision_{label}"] = pd.NA
    output["claim_scope"] = ""
    development = _packet("development")
    task_status_by_platform: dict[str, dict[str, str]] = {}
    for platform in sorted(documents["platform"].astype(str).unique()):
        group = documents[documents["platform"].astype(str).eq(platform)].reset_index(drop=True)
        task_status = _validated_tasks(selections, reserve, platform)
        task_status_by_platform[platform] = {task: status for task, (status, _) in task_status.items()}
        indexes = output.index[output["platform"].astype(str).eq(platform)]
        for task, (status, reason) in task_status.items():
            output.loc[indexes, f"{task}_status"] = status
            output.loc[indexes, f"{task}_fallback_reason"] = "" if status == "validated_automated" else reason
        output.loc[indexes, "claim_scope"] = "validated_automated" if all(status == "validated_automated" for status, _ in task_status.values()) else "fallback_human_sample"
        if task_status["sentiment"][0] == "validated_automated":
            candidate = selections["sentiment"]["candidate_result"]["by_platform"][platform]
            output.loc[indexes, "sentiment_prediction"] = _apply_sentiment_predictions(group, candidate, "full_corpus", platform)
        if task_status["stance"][0] == "validated_automated":
            dev_group = development[development["platform"].astype(str).eq(platform)]
            candidate = selections["stance"]["candidate_result"]["by_platform"][platform]
            output.loc[indexes, "stance_prediction"] = _apply_stance_predictions(group, dev_group, candidate, "full_corpus", platform)
        if task_status["frames"][0] == "validated_automated":
            dev_group = development[development["platform"].astype(str).eq(platform)]
            candidate = selections["frames"]["candidate_result"]["by_platform"][platform]
            probabilities, decisions = _apply_frame_predictions(group, dev_group, candidate)
            for index, label in enumerate(FRAME_LABELS):
                output.loc[indexes, f"frame_probability_{label}"] = probabilities[:, index]
                output.loc[indexes, f"frame_decision_{label}"] = decisions[:, index]
    PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    output.to_parquet(PREDICTIONS_ARTIFACT, compression="zstd", index=False)
    source_paths = _source_paths()
    manifest = {
        "schema_version": PREDICTIONS_SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "source_checksums": {name: sha256_file(path) for name, path in source_paths.items()},
        "selection_checksums": {
            "sentiment": sha256_file(SENTIMENT_SELECTION),
            "stance_frames": sha256_file(STANCE_FRAME_SELECTION),
        },
        "population_contract_version": POPULATION_CONTRACT_VERSION,
        "population_contract": {
            "reddit": "audited relevant core threads; UK_ENFORCEMENT_CLUSTER rows are valid observed cross-author replies",
            "bluesky": "strict observed reply descendants of in-search phrase-exact E2/E3 roots",
            "youtube": "strict eligible E2/E3 comments on audited seed videos with at least 20 strict comments",
        },
        "reserve_assessment": {"path": relative_path(RESERVE_ASSESSMENT), "sha256": sha256_file(RESERVE_ASSESSMENT), "status": reserve_status["status"]},
        "preprocessing_version": "masked-text-v2",
        "batch_rows": PREDICTION_BATCH_ROWS,
        "rows": int(len(output)),
        "by_platform": {platform: int(len(group)) for platform, group in _platform_frames(documents).items()},
        "claim_scope_counts": {str(key): int(value) for key, value in output["claim_scope"].value_counts().items()},
        "task_status_counts": {
            task: {str(key): int(value) for key, value in output[f"{task}_status"].value_counts().items()}
            for task in ("sentiment", "stance", "frames")
        },
        "task_status_by_platform": task_status_by_platform,
        "artifact": {"path": relative_path(PREDICTIONS_ARTIFACT), "sha256": sha256_file(PREDICTIONS_ARTIFACT)},
        "interpretation": "Only reserve-passed cells may emit validated automated labels; current failed cells are explicit human-sample fallbacks.",
    }
    write_json(PREDICTIONS_MANIFEST, manifest)
    return {"status": "valid", "artifact": relative_path(PREDICTIONS_MANIFEST), "rows": manifest["rows"]}


def check_predictions() -> dict[str, Any]:
    reserve_status = check_reserve_assessment()
    if not PREDICTIONS_MANIFEST.exists() or not PREDICTIONS_ARTIFACT.exists():
        raise FileNotFoundError(f"full-corpus prediction artifacts are missing: {PREDICTIONS_MANIFEST}; run the explicit predictions build first")
    manifest = read_json(PREDICTIONS_MANIFEST)
    source_paths = _source_paths()
    if manifest.get("schema_version") != PREDICTIONS_SCHEMA_VERSION or manifest.get("source_checksums") != {name: sha256_file(path) for name, path in source_paths.items()}:
        raise ValueError("full-corpus prediction source checksum changed")
    expected_reserve = {"path": relative_path(RESERVE_ASSESSMENT), "sha256": sha256_file(RESERVE_ASSESSMENT), "status": reserve_status["status"]}
    if manifest.get("reserve_assessment") != expected_reserve:
        raise ValueError("full-corpus predictions were built from a different reserve assessment")
    if manifest.get("population_contract_version") != POPULATION_CONTRACT_VERSION:
        raise ValueError("full-corpus prediction population contract is stale")
    expected_selection = {"sentiment": sha256_file(SENTIMENT_SELECTION), "stance_frames": sha256_file(STANCE_FRAME_SELECTION)}
    if manifest.get("selection_checksums") != expected_selection:
        raise ValueError("full-corpus prediction selection changed")
    if manifest.get("artifact", {}).get("sha256") != sha256_file(PREDICTIONS_ARTIFACT):
        raise ValueError("full-corpus prediction artifact changed")
    if not manifest.get("task_status_by_platform"):
        raise ValueError("full-corpus prediction task status by platform is missing")
    frame = pd.read_parquet(PREDICTIONS_ARTIFACT)
    if len(frame) != manifest.get("rows") or frame.duplicated(["platform", "doc_id"]).any():
        raise ValueError("full-corpus prediction row count or uniqueness changed")
    expected = load_frozen_documents().sort_values(["platform", "doc_id"]).reset_index(drop=True)
    population_columns = [column for column in expected.columns if column != "text_for_annotation"]
    actual = frame[population_columns].sort_values(["platform", "doc_id"]).reset_index(drop=True)
    expected = expected[population_columns]
    if not actual.fillna("").astype(str).equals(expected.fillna("").astype(str)):
        raise ValueError("full-corpus prediction rows no longer match the frozen population loaders")
    return {"status": "valid", "artifact": relative_path(PREDICTIONS_MANIFEST), "rows": int(len(frame))}
