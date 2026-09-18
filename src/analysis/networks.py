"""Build the frozen, explicitly named graph artifacts used by later analysis."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.artifacts import (
    ANALYSIS_ROOT,
    REPO_ROOT,
    read_json,
    relative_path,
    sha256_file,
    utc_now_iso,
    write_json,
)
from src.analysis.codebook import FRAME_DEFINITIONS
from src.analysis.populations import POPULATION_ARTIFACT, check_populations
from src.analysis.scope import SCOPE_ARTIFACT_PATH
from src.shared.case_windows import REDDIT_CASE_BY_EVENT, REDDIT_EVENT_BY_CASE

NETWORK_ROOT = ANALYSIS_ROOT / "networks"
NETWORK_MANIFEST = NETWORK_ROOT / "network_manifest.json"
NETWORK_SCHEMA_VERSION = "networks.v2"
ANNOTATION_ROOT = ANALYSIS_ROOT / "annotation"
REDDIT_RELEVANCE_AUDIT = ANNOTATION_ROOT / "reddit_thread_relevance.csv"
YOUTUBE_METADATA_AUDIT = ANNOTATION_ROOT / "youtube_video_metadata.csv"
YOUTUBE_SEED_VIDEOS = REPO_ROOT / "config" / "youtube" / "seed_videos.csv"
CORE_CASES = REDDIT_EVENT_BY_CASE
CORE_EVENTS = {"E2", "E3"}
EXCLUDED_BLUESKY_ACCOUNT_CLASSES = {"labelled_spam", "repeater"}
REDDIT_AUDIT_COLUMNS = ("thread_id", "coder_a_relevant", "coder_b_relevant", "adjudicated_relevant")
YOUTUBE_AUDIT_COLUMNS = (
    "video_id", "coder_a_source_type", "coder_b_source_type", "adjudicated_source_type",
    "coder_a_frame", "coder_b_frame", "adjudicated_frame", "coder_a_stance", "coder_b_stance", "adjudicated_stance",
)
FRAME_LABELS = {frame["label"] for frame in FRAME_DEFINITIONS} | {"none", "none_unclear"}
STANCE_LABELS = {"support", "oppose", "mixed_conditional", "neutral_descriptive", "unclear_ambiguous"}
YOUTUBE_SOURCE_TYPES = {"news", "commentary", "tech_explainer"}
EDGE_COLUMNS = [
    "source_id",
    "target_id",
    "scope",
    "day",
    "first_day",
    "last_day",
    "weight",
    "source_doc_id",
    "target_doc_id",
    "source_author_hash",
    "target_author_hash",
    "container_id",
    "event_id",
    "edge_resolution",
    "depth",
]


def _valid_pipe_labels(value: str, allowed: set[str]) -> bool:
    labels = [label.strip() for label in str(value).split("|") if label.strip()]
    return bool(labels) and all(label in allowed for label in labels)


def _normalise_yes_no(values: pd.Series, field: str) -> pd.Series:
    normalised = values.astype(str).str.strip().str.casefold()
    allowed = {"yes", "no", "true", "false", "1", "0"}
    if not normalised.isin(allowed).all():
        raise ValueError(f"{field} must contain only yes/no values")
    return normalised.isin({"yes", "true", "1"})


def validate_reddit_relevance_audit(path: Path = REDDIT_RELEVANCE_AUDIT) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": relative_path(path), "accepted_thread_count": 0, "accepted_thread_ids": set()}
    frame = pd.read_csv(path, keep_default_na=False)
    missing = sorted(set(REDDIT_AUDIT_COLUMNS) - set(frame.columns))
    if missing or frame.empty or frame["thread_id"].astype(str).str.strip().eq("").any() or frame["thread_id"].duplicated().any():
        raise ValueError(f"invalid Reddit relevance audit schema: missing={missing}")
    for column in REDDIT_AUDIT_COLUMNS[1:]:
        frame[column] = _normalise_yes_no(frame[column], column)
    audited = set(frame["thread_id"].astype(str))
    accepted = set(frame.loc[frame["adjudicated_relevant"], "thread_id"].astype(str))
    return {"status": "valid", "path": relative_path(path), "rows": int(len(frame)), "audited_thread_count": len(audited), "audited_thread_ids": audited, "accepted_thread_count": len(accepted), "accepted_thread_ids": accepted}


def validate_youtube_metadata_audit(path: Path = YOUTUBE_METADATA_AUDIT) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": relative_path(path), "accepted_video_count": 0, "accepted_video_ids": set(), "audited_video_ids": set(), "source_types": {}}
    frame = pd.read_csv(path, keep_default_na=False)
    missing = sorted(set(YOUTUBE_AUDIT_COLUMNS) - set(frame.columns))
    if missing or frame.empty or frame["video_id"].astype(str).str.strip().eq("").any() or frame["video_id"].duplicated().any():
        raise ValueError(f"invalid YouTube metadata audit schema: missing={missing}")
    source_types = frame["adjudicated_source_type"].astype(str).str.strip().str.casefold()
    if source_types.eq("").any() or ~source_types.isin(YOUTUBE_SOURCE_TYPES).all():
        raise ValueError(f"YouTube metadata audit has an unknown source type; expected one of {sorted(YOUTUBE_SOURCE_TYPES)}")
    for column in ("coder_a_source_type", "coder_b_source_type"):
        values = frame[column].astype(str).str.strip().str.casefold()
        if values.eq("").any() or ~values.isin(YOUTUBE_SOURCE_TYPES).all():
            raise ValueError(f"{column} contains an unknown YouTube source type")
    for column in ("coder_a_frame", "coder_b_frame", "adjudicated_frame"):
        values = frame[column].astype(str).str.strip()
        if values.eq("").any() or ~values.map(lambda value: _valid_pipe_labels(value, FRAME_LABELS)).all():
            raise ValueError(f"{column} contains an unknown or empty frame label")
    for column in ("coder_a_stance", "coder_b_stance", "adjudicated_stance"):
        values = frame[column].astype(str).str.strip()
        if values.eq("").any() or ~values.isin(STANCE_LABELS).all():
            raise ValueError(f"{column} contains an unknown or empty stance label")
    audited_video_ids = {canonical_youtube_video_id(value) for value in frame["video_id"].astype(str)}
    accepted = audited_video_ids
    return {
        "status": "valid",
        "path": relative_path(path),
        "rows": int(len(frame)),
        "accepted_video_count": len(accepted),
        "accepted_video_ids": accepted,
        "audited_video_ids": audited_video_ids,
        "source_types": dict(zip(frame["video_id"].astype(str), source_types)),
        "coder_agreement": {
            field: float((frame[f"coder_a_{field}"].astype(str).str.strip().str.casefold() == frame[f"coder_b_{field}"].astype(str).str.strip().str.casefold()).mean())
            for field in ("source_type", "frame", "stance")
        },
    }


def frozen_youtube_video_ids(events: set[str] | None = None) -> set[str]:
    seed = pd.read_csv(YOUTUBE_SEED_VIDEOS, usecols=["video_id", "event_id"])
    if events is not None:
        seed = seed[seed["event_id"].isin(events)]
    return {canonical_youtube_video_id(value) for value in seed["video_id"].astype(str)}


def canonical_youtube_video_id(value: object) -> str:
    text = str(value)
    return text if text.startswith("yt:video:") else f"yt:video:{text}"


def scoped_edge_frames(edges: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if edges.empty or "scope" not in edges:
        return []
    return [(str(scope), edges.loc[edges["scope"].astype(str).eq(str(scope))].copy().reset_index(drop=True)) for scope in sorted(edges["scope"].dropna().astype(str).unique())]


def youtube_video_event_assignments(documents: pd.DataFrame) -> pd.DataFrame:
    assignments = documents[["doc_id", "thing", "root_doc_id", "yt_video_event_id"]].copy()
    video_events = assignments.loc[assignments["thing"].eq("video"), ["doc_id", "yt_video_event_id"]].rename(columns={"doc_id": "root_doc_id", "yt_video_event_id": "video_event"})
    assignments["video_event"] = assignments["yt_video_event_id"]
    comments = assignments["thing"].ne("video")
    assignments.loc[comments, "video_event"] = assignments.loc[comments, "root_doc_id"].map(video_events.set_index("root_doc_id")["video_event"])
    return assignments[["doc_id", "root_doc_id", "thing", "video_event"]]


def _strict_reddit(frame: pd.DataFrame) -> pd.Series:
    text = frame["text_topic"].fillna("").astype(str).str.strip().ne("")
    return (
        frame["schema_valid"].fillna(False).astype(bool)
        & frame["human_only"].fillna(False).astype(bool)
        & frame["is_english"].fillna(False).astype(bool)
        & text
        & ~frame["is_url_only"].fillna(False).astype(bool)
        & ~frame["is_no_substantive_text"].fillna(False).astype(bool)
        & frame["author_hash"].fillna("").astype(str).ne("")
    )


def _strict_bluesky(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["is_english"].fillna(False).astype(bool)
        & ~frame["is_meme"].fillna(False).astype(bool)
        & ~frame["is_repeat_burst"].fillna(False).astype(bool)
        & ~frame["account_class"].isin(EXCLUDED_BLUESKY_ACCOUNT_CLASSES)
        & frame["author_hash"].fillna("").astype(str).ne("")
    )


def _strict_youtube(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["thing"].ne("video")
        & frame["exclusion_status"].eq("eligible")
        & frame["is_english"].fillna(False).astype(bool)
        & frame["author_hash"].fillna("").astype(str).ne("")
    )


def _edge_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in EDGE_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    if frame.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    frame = frame[EDGE_COLUMNS].copy()
    for column in ("source_id", "target_id", "scope", "edge_resolution"):
        frame[column] = frame[column].fillna("").astype(str)
    return frame.sort_values(
        ["scope", "source_id", "target_id", "day", "source_doc_id", "target_doc_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _actor_edges(
    edges: pd.DataFrame,
    *,
    source_column: str = "source_author_hash",
    target_column: str = "target_author_hash",
    resolution: str,
) -> pd.DataFrame:
    if edges.empty:
        return _edge_frame([])
    grouped = (
        edges.groupby([source_column, target_column, "scope"], sort=True, dropna=False)
        .agg(weight=("source_doc_id", "size"), first_day=("day", "min"), last_day=("day", "max"))
        .reset_index()
    )
    return _edge_frame(
        [
            {
                "source_id": row[source_column],
                "target_id": row[target_column],
                "scope": row["scope"],
                "day": row["first_day"],
                "first_day": row["first_day"],
                "last_day": row["last_day"],
                "weight": int(row["weight"]),
                "source_author_hash": row[source_column],
                "target_author_hash": row[target_column],
                "event_id": row["scope"],
                "edge_resolution": resolution,
            }
            for _, row in grouped.iterrows()
        ]
    )


def _reverse_actor_edges(frame: pd.DataFrame) -> pd.DataFrame:
    reversed_frame = frame.copy()
    reversed_frame[["source_id", "target_id"]] = frame[["target_id", "source_id"]].to_numpy()
    reversed_frame[["source_author_hash", "target_author_hash"]] = frame[
        ["target_author_hash", "source_author_hash"]
    ].to_numpy()
    return reversed_frame.sort_values(
        ["scope", "source_id", "target_id"], kind="mergesort"
    ).reset_index(drop=True)


def _message_edges(edges: pd.DataFrame, *, resolution: str) -> pd.DataFrame:
    if edges.empty:
        return _edge_frame([])
    return _edge_frame(
        [
            {
                "source_id": row["target_doc_id"],
                "target_id": row["source_doc_id"],
                "scope": row["scope"],
                "day": row["day"],
                "weight": 1,
                "source_doc_id": row["target_doc_id"],
                "target_doc_id": row["source_doc_id"],
                "source_author_hash": row["target_author_hash"],
                "target_author_hash": row["source_author_hash"],
                "container_id": row.get("container_id", pd.NA),
                "event_id": row.get("event_id", row["scope"]),
                "edge_resolution": resolution,
                "depth": row.get("depth", pd.NA),
            }
            for _, row in edges.iterrows()
        ]
    )


def _reddit_graphs() -> dict[str, dict[str, Any]]:
    path = REPO_ROOT / "data" / "processed" / "reddit_age_gate" / "analysis_corpus.parquet"
    columns = [
        "record_id",
        "thing",
        "event_id",
        "date",
        "subreddit",
        "author_hash",
        "thread_id",
        "parent_record_id",
        "human_only",
        "schema_valid",
        "is_english",
        "text_topic",
        "is_url_only",
        "is_no_substantive_text",
    ]
    docs = pd.read_parquet(path, columns=columns)
    docs["scope"] = docs["event_id"].map(REDDIT_CASE_BY_EVENT).fillna("none")
    docs["strict"] = _strict_reddit(docs)
    comments = docs[
        docs["thing"].eq("comment")
        & docs["scope"].isin(CORE_CASES)
        & docs["strict"]
        & docs["parent_record_id"].fillna("").astype(str).ne("")
    ].copy()
    parents = docs[
        ["record_id", "author_hash", "thread_id", "scope", "strict"]
    ].rename(
        columns={
            "record_id": "target_doc_id",
            "author_hash": "target_author_hash",
            "thread_id": "parent_thread_id",
            "scope": "parent_scope",
            "strict": "parent_strict",
        }
    )
    reply = comments.rename(
        columns={"record_id": "source_doc_id", "author_hash": "source_author_hash"}
    ).merge(parents, left_on="parent_record_id", right_on="target_doc_id", how="inner")
    reply = reply[
        reply["parent_strict"]
        & reply["source_author_hash"].ne(reply["target_author_hash"])
        & reply["scope"].eq(reply["parent_scope"])
        & reply["thread_id"].eq(reply["parent_thread_id"])
    ].copy()
    relevance_audit = validate_reddit_relevance_audit()
    if relevance_audit["status"] == "valid":
        candidate_threads = set(docs.loc[docs["scope"].isin(CORE_CASES), "thread_id"].dropna().astype(str))
        audited_threads = set(relevance_audit["audited_thread_ids"])
        if audited_threads != candidate_threads:
            raise ValueError("Reddit relevance audit does not cover every frozen core thread")
        accepted_threads = relevance_audit["accepted_thread_ids"]
        reply = reply[reply["thread_id"].astype(str).isin(accepted_threads)].copy()
    reply["day"] = reply["date"].astype(str)
    reply["container_id"] = reply["thread_id"]
    reply["event_id"] = reply["scope"]
    reply["weight"] = 1
    reply["edge_resolution"] = "observed_parent"
    base = reply[
        [
            "source_doc_id",
            "target_doc_id",
            "source_author_hash",
            "target_author_hash",
            "scope",
            "day",
            "container_id",
            "event_id",
            "edge_resolution",
        ]
    ].copy()
    attention = _actor_edges(base, resolution="observed_parent")
    message = _message_edges(base, resolution="observed_parent")
    activity = docs[docs["scope"].isin(CORE_CASES) & docs["strict"] & docs["subreddit"].notna()].copy()
    if relevance_audit["status"] == "valid":
        activity = activity[activity["thread_id"].astype(str).isin(relevance_audit["accepted_thread_ids"])].copy()
    activity["subreddit"] = activity["subreddit"].astype(str).str.strip()
    activity = activity[activity["subreddit"].ne("")]
    grouped = (
        activity.groupby(["author_hash", "subreddit", "scope"], sort=True)
        .size()
        .rename("weight")
        .reset_index()
    )
    bipartite = _edge_frame(
        [
            {
                "source_id": row["author_hash"],
                "target_id": f"subreddit:{row['subreddit']}",
                "scope": row["scope"],
                "day": row["scope"],
                "weight": int(row["weight"]),
                "source_author_hash": row["author_hash"],
                "container_id": row["subreddit"],
                "event_id": row["scope"],
                "edge_resolution": "accepted_core_activity_candidate",
            }
            for _, row in grouped.iterrows()
        ]
    )
    candidate_edges = int(
        (
            docs["thing"].eq("comment")
            & docs["scope"].isin(CORE_CASES)
            & docs["parent_record_id"].fillna("").astype(str).ne("")
        ).sum()
    )
    exclusions = {
        "candidate_comment_rows": candidate_edges,
        "strict_source_parent_edges": int(len(reply)),
        "dropped_missing_or_invalid_parent": candidate_edges - int(len(reply)),
        "human_relevance_audit": relevance_audit["status"],
        "audited_relevant_threads": relevance_audit.get("accepted_thread_count", 0),
    }
    graph_status = "audited_ready" if relevance_audit["status"] == "valid" else "candidate_requires_reddit_thread_relevance_audit"
    return {
        "R_ACTOR_ATTENTION": {
            "frame": attention,
            "definition": "replier author -> replied-to author; weight = valid observed reply count",
            "status": graph_status,
            "exclusions": exclusions,
        },
        "R_ACTOR_CASCADE": {
            "frame": _reverse_actor_edges(attention),
            "definition": "replied-to author -> replier author; structural conversational-flow proxy",
            "status": graph_status,
            "exclusions": exclusions,
        },
        "R_MESSAGE_TREE": {
            "frame": message,
            "definition": "observed parent document -> child document; child date assigns the edge",
            "status": graph_status,
            "exclusions": exclusions,
        },
        "R_AUTHOR_SUBREDDIT": {
            "frame": bipartite,
            "definition": "author <-> subreddit; weight = accepted candidate activity",
            "status": graph_status,
            "exclusions": {"candidate_documents": int(len(activity)), "human_relevance_audit": relevance_audit["status"], "audited_relevant_threads": relevance_audit.get("accepted_thread_count", 0)},
        },
    }


def descendant_edge_indexes(edges: pd.DataFrame, roots: set[str]) -> set[int]:
    """Return observed edge rows reachable from the frozen root set."""

    children: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for index, row in edges[["source_doc_id", "target_doc_id"]].iterrows():
        children[str(row["target_doc_id"])].append((int(index), str(row["source_doc_id"])))
    frontier = set(roots)
    seen = set(roots)
    selected: set[int] = set()
    while frontier:
        next_frontier: set[str] = set()
        for parent in sorted(frontier):
            for index, child in children.get(parent, []):
                selected.add(index)
                if child not in seen:
                    seen.add(child)
                    next_frontier.add(child)
        frontier = next_frontier
    return selected


def _bluesky_graphs() -> dict[str, dict[str, Any]]:
    root = REPO_ROOT / "data" / "processed" / "bluesky"
    posts = pd.read_parquet(
        root / "posts.parquet",
        columns=[
            "doc_id",
            "event_window",
            "is_reply",
            "in_search",
            "is_english",
            "is_meme",
            "is_repeat_burst",
            "author_hash",
            "root_doc_id",
        ],
    )
    authors = pd.read_parquet(root / "authors.parquet", columns=["author_hash", "account_class"])
    posts = posts.merge(authors, on="author_hash", how="left")
    posts["strict"] = _strict_bluesky(posts)
    exact_docs = pd.read_parquet(
        root / "post_query.parquet", columns=["doc_id", "phrase_exact"]
    )
    exact_docs = set(exact_docs.loc[exact_docs["phrase_exact"].fillna(False).astype(bool), "doc_id"])
    root_rows = posts[
        posts["doc_id"].isin(exact_docs)
        & posts["event_window"].isin(CORE_EVENTS)
        & posts["in_search"].fillna(False).astype(bool)
        & ~posts["is_reply"].fillna(False).astype(bool)
        & posts["strict"]
    ]
    roots = set(root_rows["doc_id"])
    raw_edges = pd.read_parquet(
        root / "thread_edges.parquet",
        columns=["reply_day", "source_doc_id", "target_doc_id", "depth"],
    )
    selected = descendant_edge_indexes(raw_edges, roots)
    edges = raw_edges.loc[sorted(selected)].copy()
    source = posts[
        ["doc_id", "event_window", "author_hash", "root_doc_id", "strict"]
    ].rename(
        columns={
            "doc_id": "source_doc_id",
            "event_window": "source_event_window",
            "author_hash": "source_author_hash",
            "root_doc_id": "source_root_doc_id",
            "strict": "source_strict",
        }
    )
    target = posts[
        ["doc_id", "event_window", "author_hash", "root_doc_id", "strict"]
    ].rename(
        columns={
            "doc_id": "target_doc_id",
            "event_window": "target_event_window",
            "author_hash": "target_author_hash",
            "root_doc_id": "target_root_doc_id",
            "strict": "target_strict",
        }
    )
    edges = edges.merge(source, on="source_doc_id", how="inner").merge(target, on="target_doc_id", how="inner")
    edges = edges[
        edges["source_event_window"].isin(CORE_EVENTS)
        & edges["target_event_window"].isin(CORE_EVENTS)
        & edges["source_strict"]
        & edges["target_strict"]
        & edges["source_author_hash"].ne(edges["target_author_hash"])
    ].copy()
    edges["scope"] = edges["source_event_window"]
    edges["day"] = edges["reply_day"].astype(str)
    edges["container_id"] = edges["source_root_doc_id"]
    edges["event_id"] = edges["scope"]
    edges["weight"] = 1
    edges["edge_resolution"] = "observed_parent_descendant"
    edges = edges.rename(columns={"source_doc_id": "source_child_doc_id", "target_doc_id": "target_parent_doc_id"})
    base = edges[
        [
            "source_child_doc_id",
            "target_parent_doc_id",
            "source_author_hash",
            "target_author_hash",
            "scope",
            "day",
            "container_id",
            "event_id",
            "edge_resolution",
            "depth",
        ]
    ].rename(
        columns={"source_child_doc_id": "source_doc_id", "target_parent_doc_id": "target_doc_id"}
    )
    attention = _actor_edges(base, resolution="observed_parent_descendant")
    message = _message_edges(base, resolution="observed_parent_descendant")
    exclusions = {
        "phrase_exact_search_roots": int(len(roots)),
        "observed_descendant_edges": int(len(selected)),
        "strict_valid_edges": int(len(base)),
        "dropped_endpoint_or_self_loop": int(len(selected) - len(base)),
        "non_exact_search_hits_excluded": int(
            posts[posts["event_window"].isin(CORE_EVENTS) & posts["in_search"].fillna(False).astype(bool)]["doc_id"].nunique()
            - len(roots)
        ),
    }
    return {
        "B_ACTOR_ATTENTION": {
            "frame": attention,
            "definition": "replier author -> replied-to author among observed descendants of phrase-exact roots",
            "status": "candidate_network_ready",
            "exclusions": exclusions,
        },
        "B_ACTOR_CASCADE": {
            "frame": _reverse_actor_edges(attention),
            "definition": "replied-to author -> replier author; structural conversational-flow proxy",
            "status": "candidate_network_ready",
            "exclusions": exclusions,
        },
        "B_MESSAGE_TREE": {
            "frame": message,
            "definition": "observed parent document -> descendant document; processed day assigns the edge",
            "status": "candidate_network_ready",
            "exclusions": exclusions,
        },
    }


def _youtube_graphs() -> dict[str, dict[str, Any]]:
    root = REPO_ROOT / "data" / "processed" / "youtube"
    columns = [
        "doc_id",
        "thing",
        "root_doc_id",
        "author_hash",
        "event_window",
        "exclusion_status",
        "is_english",
        "yt_video_event_id",
        "yt_video_type",
        "container_id",
        "created_utc",
    ]
    docs = pd.read_parquet(root / "documents.parquet", columns=columns)
    docs["strict"] = _strict_youtube(docs)
    assignments = youtube_video_event_assignments(docs)
    docs = docs.drop(columns=["yt_video_event_id"]).merge(assignments[["doc_id", "video_event"]], on="doc_id", how="left", validate="one_to_one")
    metadata_audit = validate_youtube_metadata_audit()
    videos = docs[docs["thing"].eq("video") & docs["video_event"].isin(CORE_EVENTS)].copy()
    strict_comments = docs[docs["strict"] & docs["root_doc_id"].notna()].copy()
    video_counts = strict_comments.groupby("root_doc_id").size()
    selected_videos = videos[
        videos["doc_id"].isin(video_counts[video_counts.ge(20)].index)
    ].copy()
    if metadata_audit["status"] == "valid":
        if metadata_audit["audited_video_ids"] != frozen_youtube_video_ids(CORE_EVENTS):
            raise ValueError("YouTube metadata audit must cover every frozen E2/E3 seed video before selection filters are applied")
        selected_videos = selected_videos[selected_videos["doc_id"].isin(metadata_audit["accepted_video_ids"])].copy()
    selected_ids = set(selected_videos["doc_id"])
    audience = strict_comments[strict_comments["root_doc_id"].isin(selected_ids) & strict_comments["video_event"].isin(CORE_EVENTS)].copy()
    grouped = (
        audience.groupby(
            ["author_hash", "root_doc_id", "video_event", "yt_video_type", "container_id"],
            sort=True,
        )
        .size()
        .rename("weight")
        .reset_index()
    )
    bipartite = _edge_frame(
        [
            {
                "source_id": row["author_hash"],
                "target_id": f"video:{row['root_doc_id']}",
                "scope": row["video_event"],
                "day": row["video_event"],
                "weight": int(row["weight"]),
                "source_author_hash": row["author_hash"],
                "container_id": row["container_id"],
                "event_id": row["video_event"],
                "edge_resolution": "strict_eligible_comment",
            }
            for _, row in grouped.iterrows()
        ]
    )
    interactions = pd.read_parquet(
        root / "interactions.parquet",
        columns=[
            "source_author_hash",
            "target_author_hash",
            "source_doc_id",
            "target_doc_id",
            "root_doc_id",
            "container_id",
            "created_utc",
            "event_window",
            "event_id_nearest",
            "is_self_loop",
            "target_resolution",
            "source_exclusion_status",
        ],
    )
    interactions = interactions[
        interactions["source_exclusion_status"].eq("eligible")
        & interactions["source_author_hash"].fillna("").ne("")
        & interactions["target_author_hash"].fillna("").ne("")
        & ~interactions["is_self_loop"].fillna(False).astype(bool)
    ].copy()
    video_roots = docs[["doc_id", "video_event"]].rename(columns={"doc_id": "interaction_video_id"})
    interactions = interactions.merge(video_roots, left_on="root_doc_id", right_on="interaction_video_id", how="inner", validate="many_to_one")
    interactions = interactions[interactions["video_event"].isin(CORE_EVENTS) & interactions["root_doc_id"].isin(selected_ids)].copy()
    def interaction_graph(resolution: str) -> pd.DataFrame:
        selected = interactions[interactions["target_resolution"].eq(resolution)].copy()
        selected["scope"] = selected["video_event"]
        selected["day"] = selected["created_utc"].astype(str)
        selected["event_id"] = selected["video_event"]
        selected["weight"] = 1
        selected["edge_resolution"] = resolution
        return selected.rename(columns={"source_doc_id": "source_doc_id", "target_doc_id": "target_doc_id"})[
            [
                "source_doc_id",
                "target_doc_id",
                "source_author_hash",
                "target_author_hash",
                "scope",
                "day",
                "container_id",
                "event_id",
                "edge_resolution",
            ]
        ]
    handle_base = interaction_graph("handle_match")
    fallback_base = interaction_graph("thread_root")
    handle_attention = _actor_edges(handle_base, resolution="handle_match")
    fallback_attention = _actor_edges(fallback_base, resolution="thread_root")
    handle_message = _message_edges(handle_base, resolution="handle_match")
    fallback_message = _message_edges(fallback_base, resolution="thread_root")
    exclusions = {
        "core_videos_before_metadata_audit": int(len(videos)),
        "frozen_seed_videos": int(len(frozen_youtube_video_ids(CORE_EVENTS) & set(docs.loc[docs["thing"].eq("video"), "doc_id"].astype(str)))),
        "metadata_audit_packets_required": int(len(frozen_youtube_video_ids(CORE_EVENTS))),
        "videos_with_at_least_20_strict_comments": int(len(selected_videos)),
        "strict_audience_comments": int(len(audience)),
        "strict_interactions_after_endpoint_filter": int(len(interactions)),
        "handle_match_interactions": int(len(handle_base)),
        "thread_root_fallback_interactions": int(len(fallback_base)),
        "metadata_audit": metadata_audit["status"],
        "audited_videos": metadata_audit.get("accepted_video_count", 0),
    }
    common_status = "audited_ready" if metadata_audit["status"] == "valid" else "candidate_requires_youtube_video_metadata_audit"
    return {
        "Y_COMMENTER_VIDEO": {
            "frame": bipartite,
            "definition": "strict-eligible commenter <-> selected E2/E3 video; weight = comment count",
            "status": common_status,
            "exclusions": exclusions,
        },
        "Y_ACTOR_ATTENTION_HANDLE": {
            "frame": handle_attention,
            "definition": "strict source replier -> handle-matched replied-to author; secondary graph",
            "status": common_status,
            "exclusions": exclusions,
        },
        "Y_ACTOR_CASCADE_HANDLE": {
            "frame": _reverse_actor_edges(handle_attention),
            "definition": "handle-matched replied-to author -> replier; secondary structural proxy",
            "status": common_status,
            "exclusions": exclusions,
        },
        "Y_MESSAGE_TREE_HANDLE": {
            "frame": handle_message,
            "definition": "observed handle-matched parent document -> reply document",
            "status": common_status,
            "exclusions": exclusions,
        },
        "Y_ACTOR_ATTENTION_ROOT_FALLBACK": {
            "frame": fallback_attention,
            "definition": "thread-root fallback actor graph; structural sensitivity only",
            "status": common_status,
            "exclusions": exclusions,
        },
        "Y_MESSAGE_TREE_ROOT_FALLBACK": {
            "frame": fallback_message,
            "definition": "thread-root fallback message edges; structural sensitivity only",
            "status": common_status,
            "exclusions": exclusions,
        },
    }


def _input_checksums() -> dict[str, str | None]:
    paths = {
        "reddit_corpus": REPO_ROOT / "data" / "processed" / "reddit_age_gate" / "analysis_corpus.parquet",
        "bluesky_posts": REPO_ROOT / "data" / "processed" / "bluesky" / "posts.parquet",
        "bluesky_queries": REPO_ROOT / "data" / "processed" / "bluesky" / "post_query.parquet",
        "bluesky_authors": REPO_ROOT / "data" / "processed" / "bluesky" / "authors.parquet",
        "bluesky_thread_edges": REPO_ROOT / "data" / "processed" / "bluesky" / "thread_edges.parquet",
        "youtube_documents": REPO_ROOT / "data" / "processed" / "youtube" / "documents.parquet",
        "youtube_interactions": REPO_ROOT / "data" / "processed" / "youtube" / "interactions.parquet",
        "reddit_relevance_audit": REDDIT_RELEVANCE_AUDIT,
        "youtube_metadata_audit": YOUTUBE_METADATA_AUDIT,
        "population_manifest": POPULATION_ARTIFACT,
    }
    return {name: sha256_file(path) if path.exists() else None for name, path in paths.items()}


def _graph_record(name: str, spec: dict[str, Any], path: Path) -> dict[str, Any]:
    frame = spec["frame"]
    forbidden = [
        column
        for column in frame.columns
        if any(term in column.casefold() for term in ("text", "handle", "display", "name"))
    ]
    if forbidden:
        raise ValueError(f"restricted columns in network artifact {name}: {forbidden}")
    frame.to_parquet(path, compression="zstd", index=False)
    nodes = set(frame["source_id"]) | set(frame["target_id"]) if not frame.empty else set()
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "nodes": int(len(nodes)),
        "edges": int(len(frame)),
        "weighted_edge_total": int(frame["weight"].fillna(0).sum()) if not frame.empty else 0,
        "definition": spec["definition"],
        "status": spec["status"],
        "exclusions": spec["exclusions"],
    }


def build_network_artifacts() -> dict[str, Any]:
    check_populations()
    scope = read_json(SCOPE_ARTIFACT_PATH)
    population = read_json(POPULATION_ARTIFACT)
    graphs: dict[str, dict[str, Any]] = {}
    for name, spec in {
        **_reddit_graphs(),
        **_bluesky_graphs(),
        **_youtube_graphs(),
    }.items():
        graphs[name] = spec
    NETWORK_ROOT.mkdir(parents=True, exist_ok=True)
    records = {
        name: _graph_record(name, spec, NETWORK_ROOT / f"{name.lower()}.parquet")
        for name, spec in sorted(graphs.items())
    }
    population_counts = {row["population_id"]: int(row["candidate_count"]) for row in population["population_registry"]}
    reconciliation = {
        "R_REPLY_CORE": {"graph_id": "R_MESSAGE_TREE", "population_count": population_counts["R_REPLY_CORE"], "graph_edge_count": records["R_MESSAGE_TREE"]["edges"]},
        "B_REPLY_CORE": {"graph_id": "B_MESSAGE_TREE", "population_count": population_counts["B_REPLY_CORE"], "graph_edge_count": records["B_MESSAGE_TREE"]["edges"]},
        "Y_VIDEO_E2E3": {"graph_id": "Y_COMMENTER_VIDEO", "population_count": population_counts["Y_VIDEO_E2E3"], "graph_selected_video_count": records["Y_COMMENTER_VIDEO"]["exclusions"]["videos_with_at_least_20_strict_comments"]},
    }
    if any(item["population_count"] != item.get("graph_edge_count", item.get("graph_selected_video_count")) for item in reconciliation.values()):
        raise ValueError(f"population and graph registry counts do not reconcile: {reconciliation}")
    manifest = {
        "schema_version": NETWORK_SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "command": "python -m src.analysis.run networks build --check",
        "scope_version": scope["scope_version"],
        "population_version": population["population_version"],
        "source_checksums": _input_checksums(),
        "population_reconciliation": reconciliation,
        "graph_registry": records,
        "global_notes": [
            "Actor attention direction is replier -> replied-to author; it measures observed received attention, not exposure or persuasion.",
            "Actor cascade direction reverses observed reply edges as a structural conversational-flow proxy.",
            "Reddit graphs remain candidate populations until the frozen thread relevance audit is completed.",
            "YouTube handle-match graphs are secondary; thread-root fallback graphs are separate sensitivities.",
            "YouTube graph scope is inherited from the assigned root video event, never the comment calendar window or nearest event.",
            "When present, relevance and metadata audits are schema-validated and applied to graph membership.",
            "Bluesky temporal assignment uses processed day, never client-supplied created_at.",
            "No cross-platform identity join is performed.",
        ],
    }
    write_json(NETWORK_MANIFEST, manifest)
    return manifest


def check_networks() -> dict[str, Any]:
    check_populations()
    if not NETWORK_MANIFEST.exists():
        raise FileNotFoundError(f"network manifest is missing: {NETWORK_MANIFEST}; run the explicit networks build first")
    manifest = read_json(NETWORK_MANIFEST)
    if manifest.get("schema_version") != NETWORK_SCHEMA_VERSION:
        raise ValueError("network manifest schema is stale; run the explicit networks build")
    if manifest.get("source_checksums") != _input_checksums():
        raise ValueError("network source checksum changed")
    population = read_json(POPULATION_ARTIFACT)
    expected_counts = {row["population_id"]: int(row["candidate_count"]) for row in population["population_registry"]}
    reconciliation = manifest.get("population_reconciliation")
    if not reconciliation:
        raise ValueError("network population reconciliation is missing; run the explicit networks build")
    for population_id, record in reconciliation.items():
        graph_count = record.get("graph_edge_count", record.get("graph_selected_video_count"))
        if record.get("population_count") != expected_counts.get(population_id) or record.get("population_count") != graph_count:
            raise ValueError(f"network population reconciliation changed: {population_id}")
    for name, record in manifest.get("graph_registry", {}).items():
        path = REPO_ROOT / record["path"]
        if not path.exists() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"network artifact changed: {name}")
        frame = pd.read_parquet(path, columns=["source_id", "target_id", "weight"])
        if len(frame) != record["rows"]:
            raise ValueError(f"network row count changed: {name}")
    return {"status": "valid", "artifact": relative_path(NETWORK_MANIFEST), "graphs": len(manifest.get("graph_registry", {}))}
