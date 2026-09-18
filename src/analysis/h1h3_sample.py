"""Draw the small human-coded samples for exploratory H1 to H3 runs.

The planned H1 to H3 tests read frames from model labels, and no model passed validation. These
samples are small enough for two people to code by hand, so the tests can run as exploratory
analyses whose limited power the report states. Every draw is deterministic.

Samples:
- H1-B: Bluesky authors in the largest reply-graph communities of E2 and E3, up to 2 posts each.
- H1-R: Reddit authors in the largest reply-graph communities of each case window, 2 documents each.
- H3: Reddit brokers (top-decile betweenness) with three activity-matched comparison authors
  (below-median betweenness) each, 4 documents per author.
- H2: Reddit documents in the two Australian windows, stratified by subreddit, one per author;
  this is a fixed-quota selected sample and does not support population or stage-share estimates.
- Reddit threads: every thread the Reddit samples touch, for the thread relevance check.

Coders see only an item id and masked text, never the platform ids, roles or sets behind it; the
register holding those stays with the coordinator. Documents already in the validation packets
are left out, because they have labels.

Usage: python -m src.analysis.h1h3_sample --build
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.analysis.artifacts import ANALYSIS_ROOT, REPO_ROOT, relative_path, utc_now_iso, write_json
from src.analysis.validation import _base_frame

SEED = 20260917
OUTPUT_ROOT = ANALYSIS_ROOT / "annotation" / "h1h3"
ROLES = ANALYSIS_ROOT / "networks" / "structure" / "roles.parquet"
VALIDATION_ROOT = ANALYSIS_ROOT / "validation"
REDDIT_CORPUS = REPO_ROOT / "data" / "processed" / "reddit_age_gate" / "analysis_corpus.parquet"

CODERS = ("kien", "duy")
H1B = {"communities": 8, "authors": 10, "docs": 2}
H1R = {"communities": 4, "authors": 8, "docs": 2}
H3 = {"brokers": 6, "matches": 3, "docs": 4, "min_docs": 5}
H2_STRATA = {"australia": 110, "technology": 40}
H2_THREAD_CAP = 2
H2_ESTIMAND = (
    "unweighted exploratory fixed-quota selected-sample comparison within the sampled Reddit documents; "
    "not a population or stage-share estimate"
)
ACTIVITY_BINS = [(5, 7), (8, 14), (15, 10**9)]
DOC_FIELDS = ["relevance", "language", "target_policy", "frame_labels", "notes"]
THREAD_FIELDS = ["thread_relevant", "notes"]
TEXT_COLUMNS = ["text_for_annotation", "parent_context", "root_context"]


def _key(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{SEED}|{value}".encode("utf-8"), digest_size=8).digest(), "big")


def _shuffled(frame: pd.DataFrame, column: str, salt: str) -> pd.DataFrame:
    order = frame[column].astype(str).map(lambda value: _key(f"{salt}|{value}"))
    return frame.assign(_order=order).sort_values(["_order", column]).drop(columns="_order")


def _labelled_doc_ids() -> set[str]:
    ids: set[str] = set()
    for split in ("development", "evaluation", "reserve"):
        register = pd.read_csv(VALIDATION_ROOT / f"coordinator_register_{split}.csv", usecols=["doc_id"])
        ids |= set(register["doc_id"].astype(str))
    return ids


def _candidates(platform: str, labelled: set[str]) -> pd.DataFrame:
    frame = _base_frame(platform)
    frame = frame[~frame["doc_id"].isin(labelled)]
    # One copy of any repeated text, so coders never see the same words twice.
    return _shuffled(frame, "doc_id", f"{platform}|dedupe").drop_duplicates("duplicate_key")


def _take_docs(docs: pd.DataFrame, author: str, count: int, salt: str, used: set[str]) -> pd.DataFrame:
    pool = docs[docs["author_key"].eq(author) & ~docs["doc_id"].isin(used)]
    return _shuffled(pool, "doc_id", f"{salt}|{author}").head(count)


def draw_community_sample(
    roles: pd.DataFrame, docs: pd.DataFrame, graph: str, spec: dict, task: str, min_docs: int, used: set[str]
) -> pd.DataFrame:
    """Authors from the largest communities of each scope, with their documents in that scope."""

    rows = []
    for scope, nodes in roles[roles["graph_id"].eq(graph)].groupby("scope"):
        scope_docs = docs[docs["case_window_id"].eq(scope)]
        activity = scope_docs.groupby("author_key").size()
        sizes = nodes.groupby("community_id").size().sort_values(ascending=False)
        for community in sizes.index[: spec["communities"]]:
            members = nodes[nodes["community_id"].eq(community)]
            members = members[members["node_id"].map(activity).fillna(0).ge(min_docs)]
            picked = 0
            for author in _shuffled(members, "node_id", f"{task}|{scope}|{community}")["node_id"]:
                chosen = _take_docs(scope_docs, author, spec["docs"], task, used)
                if len(chosen) < min_docs:
                    continue
                used |= set(chosen["doc_id"])
                rows.append(chosen.assign(task=task, scope=scope, community_id=community, author_slot=picked, community_size=len(nodes)))
                picked += 1
                if picked == spec["authors"]:
                    break
    return pd.concat(rows, ignore_index=True)


def draw_broker_sample(roles: pd.DataFrame, docs: pd.DataFrame, used: set[str], used_authors: set[str]) -> pd.DataFrame:
    """Top-decile-betweenness brokers, each with three below-median authors of similar activity."""

    rows = []
    for scope, nodes in roles[roles["graph_id"].eq("R_ACTOR_ATTENTION")].groupby("scope"):
        scope_docs = docs[docs["case_window_id"].eq(scope)]
        activity = scope_docs.groupby("author_key").size()
        nodes = nodes.assign(activity=nodes["node_id"].map(activity).fillna(0).astype(int))
        nodes = nodes[nodes["activity"].ge(H3["min_docs"]) & ~nodes["node_id"].isin(used_authors)]
        top_cut, median = nodes["betweenness"].quantile(0.9), nodes["betweenness"].median()
        nodes = nodes.assign(activity_bin=nodes["activity"].map(lambda n: next(i for i, (lo, hi) in enumerate(ACTIVITY_BINS) if lo <= n <= hi)))
        brokers = _shuffled(nodes[nodes["betweenness"].ge(top_cut)], "node_id", f"H3|{scope}|broker")
        comparisons = _shuffled(nodes[nodes["betweenness"].le(median)], "node_id", f"H3|{scope}|comparison")
        taken: set[str] = set()
        sets = 0
        for broker in brokers.itertuples(index=False):
            pool = comparisons[comparisons["activity_bin"].eq(broker.activity_bin) & ~comparisons["node_id"].isin(taken)]
            if len(pool) < H3["matches"]:
                continue
            match_set = f"{scope}:m{sets:02d}"
            members = [(broker.node_id, "broker", broker.betweenness, broker.activity)] + [
                (row.node_id, "comparison", row.betweenness, row.activity) for row in pool.head(H3["matches"]).itertuples(index=False)
            ]
            for author, role, betweenness, count in members:
                chosen = _take_docs(scope_docs, author, H3["docs"], "H3", used)
                used |= set(chosen["doc_id"])
                taken.add(author)
                rows.append(chosen.assign(task="H3", scope=scope, match_set=match_set, role=role, betweenness=betweenness, author_activity=count))
            sets += 1
            if sets == H3["brokers"]:
                break
    return pd.concat(rows, ignore_index=True)


def _reddit_subreddits() -> pd.Series:
    corpus = pd.read_parquet(REDDIT_CORPUS, columns=["record_id", "thing", "thread_id", "subreddit"])
    threads = corpus.dropna(subset=["thread_id"]).drop_duplicates("thread_id")
    return pd.Series(threads["subreddit"].to_numpy(), index="reddit:thread:" + threads["thread_id"].astype(str))


def draw_au_sample(docs: pd.DataFrame, used: set[str], subreddits: pd.Series) -> pd.DataFrame:
    """Australian-window documents by subreddit stratum: one per author, at most two per thread."""

    rows = []
    docs = docs.assign(subreddit=docs["cluster_id"].map(subreddits))
    for scope in ("AU_LEGISLATION", "AU_IMPLEMENTATION"):
        for subreddit, quota in H2_STRATA.items():
            pool = docs[docs["case_window_id"].eq(scope) & docs["subreddit"].eq(subreddit) & ~docs["doc_id"].isin(used)]
            pool = _shuffled(pool, "doc_id", f"H2|{scope}|{subreddit}").drop_duplicates("author_key")
            pool = pool[pool.groupby("cluster_id").cumcount().lt(H2_THREAD_CAP)]
            chosen = pool.head(quota)
            used |= set(chosen["doc_id"])
            rows.append(chosen.assign(task="H2", scope=scope, stratum_subreddit=subreddit))
    return pd.concat(rows, ignore_index=True)


def _assign_bluesky(sample: pd.DataFrame) -> pd.DataFrame:
    """Alternate authors between the two human coders inside each community."""

    return sample.assign(assigned_to=[CODERS[int(slot) % 2] for slot in sample["author_slot"]])


def _packet(sample: pd.DataFrame, prefix: str, salt: str) -> pd.DataFrame:
    """Give every document an opaque item id and a shuffled order within its set."""

    sample = sample.drop_duplicates("doc_id").copy()
    sample = _shuffled(sample, "doc_id", salt).reset_index(drop=True)
    sample["item_id"] = [f"{prefix}{index + 1:04d}" for index in range(len(sample))]
    return sample


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _blank(frame: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    return frame.assign(**{field: "" for field in fields})


def build(output_root: Path = OUTPUT_ROOT) -> dict:
    labelled = _labelled_doc_ids()
    roles = pd.read_parquet(ROLES, columns=["graph_id", "scope", "node_id", "community_id", "betweenness"])
    bluesky = _candidates("bluesky", labelled)
    reddit = _candidates("reddit", labelled)

    h1b = _assign_bluesky(draw_community_sample(roles, bluesky, "B_ACTOR_ATTENTION", H1B, "H1-B", 1, set()))
    h1b = _packet(h1b, "B", "H1-B|order")

    used: set[str] = set()
    h1r = draw_community_sample(roles, reddit, "R_ACTOR_ATTENTION", H1R, "H1-R", H1R["docs"], used)
    h3 = draw_broker_sample(roles, reddit, used, set(h1r["author_key"]))
    h2 = draw_au_sample(reddit, used, _reddit_subreddits())
    sets = {"H2": ("R1", h2), "H3": ("R2", h3), "H1-R": ("R3", h1r)}
    reddit_packets = {task: _packet(frame, prefix, f"{task}|order") for task, (prefix, frame) in sets.items()}
    reddit_docs = pd.concat([reddit_packets[task] for task in ("H2", "H3", "H1-R")], ignore_index=True)
    reddit_docs["set"] = reddit_docs["item_id"].str[:2]

    threads = reddit_docs.drop_duplicates("cluster_id")[["cluster_id", "root_context", "case_window_id"]]
    threads = _shuffled(threads, "cluster_id", "threads|order").reset_index(drop=True)
    threads["item_id"] = [f"T{index + 1:04d}" for index in range(len(threads))]
    threads = threads.rename(columns={"root_context": "thread_text"})
    reddit_docs["thread_item_id"] = reddit_docs["cluster_id"].map(threads.set_index("cluster_id")["item_id"])

    register_columns = [
        "item_id", "task", "platform", "doc_id", "author_key", "cluster_id", "case_window_id", "scope", "community_id",
        "community_size", "author_slot", "match_set", "role", "betweenness", "author_activity", "stratum_subreddit",
        "assigned_to", "thread_item_id", "stratum",
    ]
    register = pd.concat([h1b, reddit_docs], ignore_index=True).reindex(columns=register_columns)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(register, output_root / "coordinator" / "h1h3_register.csv")
    _write_csv(threads[["item_id", "cluster_id", "case_window_id"]], output_root / "coordinator" / "reddit_thread_register.csv")

    doc_columns = ["item_id", *TEXT_COLUMNS]
    for coder in CODERS:
        mine = h1b[h1b["assigned_to"].eq(coder)]
        _write_csv(_blank(mine[doc_columns], DOC_FIELDS), output_root / coder / f"bluesky_h1b_{coder}.csv")
        _write_csv(_blank(reddit_docs[["item_id", "set", *TEXT_COLUMNS]], DOC_FIELDS), output_root / coder / f"reddit_docs_{coder}.csv")
        _write_csv(_blank(threads[["item_id", "thread_text"]], THREAD_FIELDS), output_root / coder / f"reddit_threads_{coder}.csv")
    _write_csv(_blank(h1b[doc_columns], DOC_FIELDS), output_root / "claude" / "bluesky_h1b_claude.csv")

    summary = {
        "created_at_utc": utc_now_iso(),
        "command": "python -m src.analysis.h1h3_sample --build",
        "seed": SEED,
        "design": {
            "H1-B": H1B, "H1-R": H1R, "H3": H3,
            "H2_strata_per_window": H2_STRATA, "H2_thread_cap": H2_THREAD_CAP,
            "H2_weighting": "unweighted", "H2_estimand": H2_ESTIMAND,
        },
        "excluded_already_labelled_docs": len(labelled),
        "counts": {
            "bluesky_h1b_items": len(h1b),
            "bluesky_h1b_by_coder": h1b["assigned_to"].value_counts().to_dict(),
            "bluesky_h1b_authors": int(h1b["author_key"].nunique()),
            "bluesky_h1b_by_scope": h1b["scope"].value_counts().to_dict(),
            "reddit_doc_items": len(reddit_docs),
            "reddit_items_by_set": reddit_docs["set"].value_counts().sort_index().to_dict(),
            "h1r_authors": int(reddit_packets["H1-R"]["author_key"].nunique()),
            "h3_match_sets": int(reddit_packets["H3"]["match_set"].nunique()),
            "h3_authors_by_role": reddit_packets["H3"].drop_duplicates("author_key")["role"].value_counts().to_dict(),
            "h2_by_window_and_subreddit": {
                f"{scope}|{subreddit}": int(count)
                for (scope, subreddit), count in reddit_packets["H2"].groupby(["scope", "stratum_subreddit"]).size().items()
            },
            "reddit_thread_items": len(threads),
        },
        "outputs": {
            "register": relative_path(output_root / "coordinator" / "h1h3_register.csv"),
            "coder_folders": [relative_path(output_root / coder) for coder in (*CODERS, "claude")],
        },
    }
    write_json(output_root / "coordinator" / "h1h3_sample_manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true", help="draw the samples and write the coder packets")
    args = parser.parse_args()
    if not args.build:
        parser.error("nothing to do; pass --build")
    print(json.dumps(build()["counts"], indent=2))


if __name__ == "__main__":
    main()
