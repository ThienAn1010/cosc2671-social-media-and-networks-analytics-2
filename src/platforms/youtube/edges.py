# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

"""Build network-ready tables from one YouTube collection run (task Y8).

Output schemas are documented in ``docs/youtube/data_dictionary.md``.

interactions.parquet: one directed reply edge per reply (replier -> the person replied to).
author_video.parquet: bipartite commenter <-> video counts, for audience-overlap projections.
Only pseudonymous author hashes are written; display handles are used in memory to resolve reply targets.
This module builds tables and a readiness summary only; community detection and centrality belong to Section 5.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from src.shared.ids import author_hash, make_doc_id
from src.platforms.youtube.prepare import load_salt
from src.platforms.youtube.storage import REPO_ROOT, read_jsonl, sha256_file, utc_now_iso

PROCESSED_ROOT = REPO_ROOT / "data" / "processed" / "youtube"
# YouTube prefixes replies-to-replies with "@handle", often after a zero-width space.
ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿]")
LEADING_HANDLE_RE = re.compile(r"^\s*@([\w-]+(?:\.[\w-]+)*)")
INTERACTION_COLUMNS = [
    "edge_id", "platform", "edge_type", "source_author_hash", "target_author_hash", "source_doc_id", "target_doc_id",
    "root_doc_id", "container_id", "created_utc", "event_id_nearest", "event_window", "is_self_loop", "target_resolution",
    "yt_video_event_id", "yt_video_type", "source_exclusion_status",
]


def normalize_handle(display_name: str) -> str:
    return display_name.strip().lstrip("@").lower()


# One entry per comment in memory: ID, author channel, display handle, time and text (text is only used to read the @handle).
def _entry(comment: dict[str, Any], thing: str) -> dict[str, str]:
    snippet = comment["snippet"]
    return {
        "doc_id": make_doc_id("youtube", thing, comment["id"]),
        "author_id": snippet.get("authorChannelId", {}).get("value", ""),
        "handle": normalize_handle(snippet.get("authorDisplayName", "")),
        "published_at": snippet["publishedAt"],
        "text": snippet.get("textOriginal") or snippet.get("textDisplay", ""),
    }


def group_threads(threads: list[dict[str, Any]], reply_wrappers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for thread in threads:
        replies = {reply["id"]: _entry(reply, "reply") for reply in thread.get("replies", {}).get("comments", [])}
        grouped[thread["id"]] = {"video_id": thread["snippet"]["videoId"], "top": _entry(thread["snippet"]["topLevelComment"], "comment"), "replies": replies}
    for wrapper in reply_wrappers:
        reply = wrapper["data"]
        # Same preference as prepare.py: the comments.list copy replaces the embedded one.
        grouped[reply["snippet"]["parentId"]]["replies"][reply["id"]] = _entry(reply, "reply")
    return grouped


# Walk a thread in time order; a leading @handle points at the most recent earlier commenter with that handle,
# otherwise the reply is addressed to the thread's top-level comment.
def resolve_thread(top: dict[str, str], replies: list[dict[str, str]]) -> list[dict[str, Any]]:
    last_by_handle = {top["handle"]: top}
    edges = []
    for reply in sorted(replies, key=lambda item: (item["published_at"], item["doc_id"])):
        match = LEADING_HANDLE_RE.match(ZERO_WIDTH_RE.sub("", reply["text"]))
        target = last_by_handle.get(match.group(1).lower()) if match else None
        edges.append({
            "source": reply,
            "target": target or top,
            "target_resolution": "handle_match" if target else "thread_root",
        })
        last_by_handle[reply["handle"]] = reply
    return edges


def build_interactions(
    threads: list[dict[str, Any]], reply_wrappers: list[dict[str, Any]], videos: dict[str, dict[str, Any]],
    documents: pd.DataFrame, salt: str,
) -> pd.DataFrame:
    docs = documents.set_index("doc_id")
    rows = []
    for thread in group_threads(threads, reply_wrappers).values():
        channel_id = videos[thread["video_id"]]["snippet"]["channelId"]
        for edge in resolve_thread(thread["top"], list(thread["replies"].values())):
            source, target = edge["source"], edge["target"]
            source_doc = docs.loc[source["doc_id"]]
            rows.append({
                "edge_id": f"{source['doc_id']}->{target['doc_id']}",
                "platform": "youtube",
                "edge_type": "reply",
                "source_author_hash": author_hash(salt, "youtube", source["author_id"]),
                "target_author_hash": author_hash(salt, "youtube", target["author_id"]),
                "source_doc_id": source["doc_id"],
                "target_doc_id": target["doc_id"],
                "root_doc_id": make_doc_id("youtube", "video", thread["video_id"]),
                "container_id": f"yt:channel:{channel_id}",
                "created_utc": source_doc["created_utc"],
                "event_id_nearest": source_doc["event_id_nearest"],
                "event_window": source_doc["event_window"],
                "is_self_loop": source["author_id"] == target["author_id"],
                "target_resolution": edge["target_resolution"],
                "yt_video_event_id": source_doc["yt_video_event_id"],
                "yt_video_type": source_doc["yt_video_type"],
                "source_exclusion_status": source_doc["exclusion_status"],
            })
    return pd.DataFrame(rows, columns=INTERACTION_COLUMNS)


# Bipartite commenter <-> video counts; excluded rows are counted separately so Section 5 can choose.
def build_author_video(documents: pd.DataFrame) -> pd.DataFrame:
    comments = documents[(documents["thing"] != "video") & (documents["author_hash"] != "")]
    grouped = comments.groupby(["author_hash", "root_doc_id"])
    frame = grouped.agg(
        n_comments=("doc_id", "count"),
        n_eligible=("exclusion_status", lambda status: int((status == "eligible").sum())),
        first_created_utc=("created_utc", "min"),
        yt_video_event_id=("yt_video_event_id", "first"),
        yt_video_type=("yt_video_type", "first"),
        container_id=("container_id", "first"),
    ).reset_index()
    return frame.rename(columns={"root_doc_id": "video_doc_id"})


# Readiness only: size and shape of the directed reply graph per video event, not community or centrality results.
def readiness_summary(interactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event_id, group in interactions.groupby("yt_video_event_id"):
        graph = nx.DiGraph()
        graph.add_edges_from(zip(group["source_author_hash"], group["target_author_hash"]))
        graph.remove_edges_from(nx.selfloop_edges(graph))
        components = list(nx.weakly_connected_components(graph))
        rows.append({
            "yt_video_event_id": event_id,
            "reply_edges": len(group),
            "self_loops": int(group["is_self_loop"].sum()),
            "nodes": graph.number_of_nodes(),
            "directed_pairs": graph.number_of_edges(),
            "reciprocity": round(nx.reciprocity(graph), 4) if graph.number_of_edges() else 0.0,
            "weak_components": len(components),
            "largest_component_share": round(max(map(len, components)) / graph.number_of_nodes(), 4) if components else 0.0,
            "handle_match_share": round(float((group["target_resolution"] == "handle_match").mean()), 4),
        })
    return pd.DataFrame(rows)


def validate_tables(interactions: pd.DataFrame, author_video: pd.DataFrame, documents: pd.DataFrame) -> None:
    problems = []
    reply_docs = set(documents.loc[documents["thing"] == "reply", "doc_id"])
    if not interactions["source_doc_id"].is_unique or set(interactions["source_doc_id"]) != reply_docs:
        problems.append("interactions must contain exactly one edge per reply document")
    forbidden = [column for column in (*interactions.columns, *author_video.columns) if any(word in column.lower() for word in ("text", "handle", "display", "name"))]
    if forbidden:
        problems.append(f"identifying or text columns present: {forbidden}")
    if problems:
        raise ValueError("network tables failed validation: " + "; ".join(problems))


def run_edges(args: argparse.Namespace, salt: str | None = None) -> dict[str, Any]:
    outputs = [args.out_root / "interactions.parquet", args.out_root / "author_video.parquet"]
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise SystemExit("network tables already exist; pass --overwrite to replace them")
    documents = pd.read_parquet(args.documents)
    videos = {row["data"]["id"]: row["data"] for row in read_jsonl(args.collect_run / "videos.jsonl")}
    threads = [row["data"] for row in read_jsonl(args.collect_run / "comment_threads.jsonl")]
    reply_wrappers = list(read_jsonl(args.collect_run / "replies.jsonl"))

    interactions = build_interactions(threads, reply_wrappers, videos, documents, salt or load_salt())
    author_video = build_author_video(documents)
    validate_tables(interactions, author_video, documents)
    summary = readiness_summary(interactions)

    args.out_root.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(outputs[0], index=False)
    author_video.to_parquet(outputs[1], index=False)
    manifest = {
        "created_at_utc": utc_now_iso(),
        "platform": "youtube",
        "stage": "edges",
        "script": "src.platforms.youtube.edges",
        "input": {"collect_run": args.collect_run.name, "documents_sha256": sha256_file(args.documents)},
        "counts": {
            "reply_edges": len(interactions),
            "self_loops": int(interactions["is_self_loop"].sum()),
            "target_resolution": {key: int(value) for key, value in interactions["target_resolution"].value_counts().items()},
            "author_video_rows": len(author_video),
            "authors": int(author_video["author_hash"].nunique()),
            "authors_on_2plus_videos": int((author_video.groupby("author_hash").size() >= 2).sum()),
        },
        "readiness_summary": summary.to_dict(orient="records"),
        "notes": [
            "Direction: source = replier, target = author replied to (handle_match) or the thread's top-level author (thread_root).",
            "Edges are not aggregated or filtered; weights, self-loop removal and exclusion filters are Section 5 decisions.",
            "Mid-text @mentions are not turned into edges; only the reply structure is used.",
        ],
        "output_sha256": {path.name: sha256_file(path) for path in outputs},
    }
    (args.out_root / "edges_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collect-run", type=Path, required=True)
    parser.add_argument("--documents", type=Path, default=PROCESSED_ROOT / "documents.parquet")
    parser.add_argument("--out-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    manifest = run_edges(build_parser().parse_args())
    print(json.dumps(manifest["counts"], indent=2))
    print(pd.DataFrame(manifest["readiness_summary"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
