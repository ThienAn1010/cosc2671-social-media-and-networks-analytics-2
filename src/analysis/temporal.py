"""Observed message-tree and temporal-community summaries."""

from __future__ import annotations

import itertools
import random
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.analysis.artifacts import ANALYSIS_ROOT, REPO_ROOT, read_json, relative_path, sha256_file, utc_now_iso, write_json
from src.analysis.networks import NETWORK_MANIFEST, NETWORK_ROOT, check_networks, validate_reddit_relevance_audit
from src.shared.case_windows import REDDIT_CASE_BY_EVENT

TEMPORAL_ROOT = ANALYSIS_ROOT / "networks" / "temporal"
TEMPORAL_MANIFEST = TEMPORAL_ROOT / "temporal_manifest.json"
MESSAGE_GRAPHS = {
    "reddit": ("R_MESSAGE_TREE", "candidate_requires_reddit_thread_relevance_audit"),
    "bluesky": ("B_MESSAGE_TREE", "candidate_network_ready"),
}
COMMUNITY_MATCH_THRESHOLD = 0.20
TEMPORAL_REPLICATES = 9_999
TRANSITION_COLUMNS = [
    "source_frame", "target_frame", "transitions", "row_share", "share_of_all_transitions",
    "cluster_bootstrap_lower_95", "cluster_bootstrap_median", "cluster_bootstrap_upper_95",
    "shuffled_label_null_lower_95", "shuffled_label_null_median", "shuffled_label_null_upper_95",
    "status",
]
UPTAKE_COLUMNS = [
    "scope", "community_id", "frame", "first_sustained_date", "bootstrap_lower_95",
    "bootstrap_median", "bootstrap_upper_95", "null_detected", "null_replicates", "status",
]


def _structural_virality(graph: nx.Graph) -> float | None:
    if graph.number_of_nodes() < 2:
        return 0.0
    values: list[int] = []
    rng = random.Random(20260915 + graph.number_of_nodes())
    for component in nx.connected_components(graph):
        nodes = sorted(component)
        if len(nodes) < 2:
            continue
        pairs = list(itertools.combinations(nodes, 2))
        if len(pairs) > 500:
            pairs = rng.sample(pairs, 500)
        for source, target in pairs:
            try:
                values.append(nx.shortest_path_length(graph, source, target))
            except nx.NetworkXNoPath:
                continue
    return float(sum(values) / len(values)) if values else None


def _max_depth(graph: nx.DiGraph) -> int:
    roots = [node for node, degree in graph.in_degree() if degree == 0]
    if not roots:
        return 0
    depths = {}
    for root in roots:
        for node, distance in nx.single_source_shortest_path_length(graph, root).items():
            depths[node] = max(depths.get(node, 0), distance)
    return max(depths.values(), default=0)


CASCADE_COLUMNS = [
    "platform", "scope", "container_id", "documents", "edges", "root_documents", "max_depth",
    "max_breadth", "duration_days", "structural_virality", "observed_parent_edges", "validity_status",
]


def _cascade_summary(platform: str, edges: pd.DataFrame, validity_status: str, roots: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    roots = roots if roots is not None else pd.DataFrame(columns=["scope", "container_id", "root_id", "day"])
    edges = edges.copy()
    for frame in (edges, roots):
        for column in ("scope", "container_id"):
            if column not in frame:
                frame[column] = ""
            frame[column] = frame[column].fillna("").astype(str)
    if "root_id" not in roots:
        roots["root_id"] = ""
    keys = set(map(tuple, edges[["scope", "container_id"]].drop_duplicates().to_numpy())) | set(map(tuple, roots[["scope", "container_id"]].drop_duplicates().to_numpy()))
    for scope, container in sorted(keys):
        group = edges[edges["scope"].eq(scope) & edges["container_id"].eq(container)].copy()
        root_group = roots[roots["scope"].eq(scope) & roots["container_id"].eq(container)].copy()
        graph = nx.DiGraph()
        if not group.empty:
            graph.add_edges_from((str(row.source_id), str(row.target_id)) for row in group.itertuples())
        if not root_group.empty:
            graph.add_nodes_from(root_group["root_id"].astype(str))
        undirected = graph.to_undirected()
        day_values = [group["day"]] if "day" in group else []
        if "day" in root_group:
            day_values.append(root_group["day"])
        days = pd.to_datetime(pd.concat(day_values, ignore_index=True), errors="coerce").dropna() if day_values else pd.Series(dtype="datetime64[ns]")
        duration = int((days.max() - days.min()).days) if not days.empty else None
        levels: dict[int, int] = defaultdict(int)
        for root in (node for node, degree in graph.in_degree() if degree == 0):
            for _, distance in nx.single_source_shortest_path_length(graph, root).items():
                levels[distance] += 1
        rows.append(
            {
                "platform": platform,
                "scope": scope,
                "container_id": container,
                "documents": int(undirected.number_of_nodes()),
                "edges": int(graph.number_of_edges()),
                "root_documents": int(sum(1 for _, degree in graph.in_degree() if degree == 0)),
                "max_depth": _max_depth(graph),
                "max_breadth": max(levels.values(), default=0),
                "duration_days": duration,
                "structural_virality": _structural_virality(undirected),
                "observed_parent_edges": bool(not group.empty),
                "validity_status": validity_status,
            }
        )
    return pd.DataFrame(rows, columns=CASCADE_COLUMNS)


def _root_records(platform: str) -> pd.DataFrame:
    """Load frozen root documents so isolated/root-only cascades remain visible."""

    columns = ["scope", "container_id", "root_id", "day"]
    if platform == "reddit":
        path = REPO_ROOT / "data" / "processed" / "reddit_age_gate" / "analysis_corpus.parquet"
        docs = pd.read_parquet(path, columns=["record_id", "thing", "event_id", "date", "thread_id", "schema_valid", "human_only", "is_english", "text_topic", "is_url_only", "is_no_substantive_text"])
        docs["scope"] = docs["event_id"].map(REDDIT_CASE_BY_EVENT)
        strict = (
            docs["schema_valid"].fillna(False).astype(bool)
            & docs["human_only"].fillna(False).astype(bool)
            & docs["is_english"].fillna(False).astype(bool)
            & docs["text_topic"].fillna("").astype(str).str.strip().ne("")
            & ~docs["is_url_only"].fillna(False).astype(bool)
            & ~docs["is_no_substantive_text"].fillna(False).astype(bool)
        )
        roots = docs[docs["thing"].eq("post") & docs["scope"].notna() & docs["thread_id"].fillna("").astype(str).ne("") & strict].copy()
        audit = validate_reddit_relevance_audit()
        if audit["status"] == "valid":
            roots = roots[roots["thread_id"].astype(str).isin(audit["accepted_thread_ids"])].copy()
        return roots.rename(columns={"thread_id": "container_id", "record_id": "root_id", "date": "day"})[columns]
    if platform == "bluesky":
        root = REPO_ROOT / "data" / "processed" / "bluesky"
        posts = pd.read_parquet(root / "posts.parquet", columns=["doc_id", "event_window", "day", "is_reply", "in_search", "is_english", "is_meme", "is_repeat_burst", "author_hash"])
        exact = pd.read_parquet(root / "post_query.parquet", columns=["doc_id", "phrase_exact"])
        exact_ids = set(exact.loc[exact["phrase_exact"].fillna(False).astype(bool), "doc_id"])
        roots = posts[
            posts["doc_id"].isin(exact_ids)
            & posts["event_window"].isin(["E2", "E3"])
            & posts["in_search"].fillna(False).astype(bool)
            & ~posts["is_reply"].fillna(False).astype(bool)
            & posts["is_english"].fillna(False).astype(bool)
            & ~posts["is_meme"].fillna(False).astype(bool)
            & ~posts["is_repeat_burst"].fillna(False).astype(bool)
            & posts["author_hash"].fillna("").astype(str).ne("")
        ].copy()
        return pd.DataFrame({"scope": roots["event_window"], "container_id": roots["doc_id"], "root_id": roots["doc_id"], "day": roots["day"]})[columns]
    raise ValueError(f"unsupported temporal platform: {platform}")


def _transition_rows(
    edges: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    source_column: str = "source_doc_id",
    target_column: str = "target_doc_id",
    label_column: str = "frame_labels",
    label_id_column: str = "doc_id",
) -> pd.DataFrame:
    required_edges = {source_column, target_column}
    required_labels = {label_id_column, label_column}
    if not required_edges <= set(edges.columns) or not required_labels <= set(labels.columns):
        raise ValueError("frame transitions need edge endpoints and document labels")
    lookup = {str(key): value for key, value in labels.set_index(label_id_column)[label_column].to_dict().items()}
    rows = []
    for edge in edges.itertuples(index=False):
        source_value = str(lookup.get(str(getattr(edge, source_column)), "")).strip()
        target_value = str(lookup.get(str(getattr(edge, target_column)), "")).strip()
        if not source_value or source_value.casefold() == "nan" or not target_value or target_value.casefold() == "nan":
            continue
        scope = str(getattr(edge, "scope", "all"))
        cluster = str(getattr(edge, "container_id", getattr(edge, "scope", "all")))
        rows.extend(
            {"source_frame": source, "target_frame": target, "_cluster": cluster, "_scope": scope}
            for source in source_value.split("|")
            for target in target_value.split("|")
            if source.strip() and target.strip()
        )
    return pd.DataFrame(rows, columns=["source_frame", "target_frame", "_cluster", "_scope"])


def _conditional_transition_share(counts: pd.Series, source_frame: str, target_frame: str) -> float:
    row_total = float(counts.groupby(level=0).sum().get(source_frame, 0.0))
    return float(counts.get((source_frame, target_frame), 0.0) / row_total) if row_total else 0.0


def _authoritative_scoped_labels(edges: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join document labels to scope from frozen observed edges."""

    required = {"scope", "source_doc_id", "target_doc_id"}
    if not required <= set(edges.columns):
        raise ValueError("scoped temporal nulls require authoritative edge scopes")
    endpoints: dict[str, set[str]] = defaultdict(set)
    for row in edges.itertuples(index=False):
        scope = str(row.scope).strip()
        if not scope or scope.casefold() == "nan":
            raise ValueError("scoped temporal nulls require non-empty edge scopes")
        endpoints[str(row.source_doc_id)].add(scope)
        endpoints[str(row.target_doc_id)].add(scope)
    if any(len(scopes) != 1 for scopes in endpoints.values()):
        raise ValueError("a labelled document belongs to multiple temporal scopes")
    if not {"doc_id", "frame_labels"} <= set(labels.columns):
        raise ValueError("temporal labels need document IDs and frame labels")
    scoped = labels.copy()
    scoped["_doc_key"] = scoped["doc_id"].astype(str)
    scoped = scoped[scoped["_doc_key"].isin(endpoints)].copy()
    scoped["scope"] = scoped["_doc_key"].map(lambda key: next(iter(endpoints[key])))
    if "scope" in labels.columns:
        declared = labels.loc[scoped.index, "scope"].astype(str).str.strip()
        if (declared.ne("") & declared.ne("nan") & declared.ne(scoped["scope"])).any():
            raise ValueError("temporal label scope conflicts with authoritative frozen scope")
    return scoped.drop(columns="_doc_key")


def _shuffle_labels_within_scope(labels: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if "scope" not in labels or labels["scope"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("label null requires a non-empty authoritative scope")
    shuffled = labels.copy()
    for indexes in shuffled.groupby("scope", sort=True).groups.values():
        shuffled.loc[indexes, "frame_labels"] = rng.permutation(shuffled.loc[indexes, "frame_labels"].to_numpy())
    return shuffled


def frame_transition_matrix(
    edges: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    source_column: str = "source_doc_id",
    target_column: str = "target_doc_id",
    label_column: str = "frame_labels",
    label_id_column: str = "doc_id",
) -> pd.DataFrame:
    """Count observed parent-to-child frame transitions from validated labels."""

    rows = _transition_rows(
        edges,
        labels,
        source_column=source_column,
        target_column=target_column,
        label_column=label_column,
        label_id_column=label_id_column,
    )
    if rows.empty:
        return pd.DataFrame(columns=["source_frame", "target_frame", "transitions", "row_share", "share_of_all_transitions"])
    result = rows.value_counts(["source_frame", "target_frame"]).rename("transitions").reset_index()
    result["row_share"] = result["transitions"] / result.groupby("source_frame")["transitions"].transform("sum")
    result["share_of_all_transitions"] = result["transitions"] / result["transitions"].sum()
    return result[["source_frame", "target_frame", "transitions", "row_share", "share_of_all_transitions"]]


def transition_diagnostics(
    edges: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    replicates: int = TEMPORAL_REPLICATES,
    seed: int = 20260915,
) -> pd.DataFrame:
    """Return observed transitions with clustered uncertainty and a label null."""

    rows = _transition_rows(edges, labels)
    if rows.empty:
        return pd.DataFrame(columns=TRANSITION_COLUMNS)
    if replicates < 1:
        raise ValueError("transition diagnostic replicates must be positive")
    keys = sorted(set(zip(rows["source_frame"], rows["target_frame"])))
    clusters = list(rows[["_scope", "_cluster"]].drop_duplicates().itertuples(index=False, name=None))
    by_cluster = {(scope, cluster): rows[rows["_scope"].eq(scope) & rows["_cluster"].eq(cluster)] for scope, cluster in clusters}
    rng = np.random.default_rng(seed)
    bootstrap = {key: [] for key in keys}
    for _ in range(replicates):
        sampled = rng.choice(len(clusters), size=len(clusters), replace=True)
        sample = pd.concat([by_cluster[clusters[index]] for index in sampled], ignore_index=True)
        counts = sample.value_counts(["source_frame", "target_frame"])
        for key in keys:
            bootstrap[key].append(_conditional_transition_share(counts, *key))

    null_values = {key: [] for key in keys}
    for _ in range(replicates):
        shuffled = _shuffle_labels_within_scope(_authoritative_scoped_labels(edges, labels), rng)
        null_rows = _transition_rows(edges, shuffled)
        counts = null_rows.value_counts(["source_frame", "target_frame"]) if not null_rows.empty else pd.Series(dtype=float)
        for key in keys:
            null_values[key].append(_conditional_transition_share(counts, *key))

    observed = rows.value_counts(["source_frame", "target_frame"]).rename("transitions").reset_index()
    observed["row_share"] = observed["transitions"] / observed.groupby("source_frame")["transitions"].transform("sum")
    observed["share_of_all_transitions"] = observed["transitions"] / observed["transitions"].sum()
    output = []
    for row in observed.itertuples(index=False):
        key = (row.source_frame, row.target_frame)
        output.append(
            {
                "source_frame": row.source_frame,
                "target_frame": row.target_frame,
                "transitions": int(row.transitions),
                "row_share": float(row.row_share),
                "share_of_all_transitions": float(row.share_of_all_transitions),
                "cluster_bootstrap_lower_95": float(np.quantile(bootstrap[key], 0.025)),
                "cluster_bootstrap_median": float(np.quantile(bootstrap[key], 0.5)),
                "cluster_bootstrap_upper_95": float(np.quantile(bootstrap[key], 0.975)),
                "shuffled_label_null_lower_95": float(np.quantile(null_values[key], 0.025)),
                "shuffled_label_null_median": float(np.quantile(null_values[key], 0.5)),
                "shuffled_label_null_upper_95": float(np.quantile(null_values[key], 0.975)),
                "status": "available",
            }
        )
    return pd.DataFrame(output, columns=TRANSITION_COLUMNS)


def first_sustained_uptake(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    community_column: str = "community_id",
    frame_column: str = "frame_labels",
    scope_column: str = "scope",
    window_days: int = 7,
    consecutive_windows: int = 2,
    minimum_documents: int = 3,
) -> pd.DataFrame:
    """Find the first frame/community window meeting a sustained count rule."""

    required = {date_column, community_column, frame_column}
    if not required <= set(frame.columns):
        raise ValueError(f"sustained uptake input missing columns: {sorted(required - set(frame.columns))}")
    values = frame.copy()
    values[date_column] = pd.to_datetime(values[date_column], errors="raise")
    values["_frame"] = values[frame_column].astype(str).str.split("|")
    values = values.explode("_frame")
    values = values[values["_frame"].astype(str).str.strip().ne("")]
    scope_column = scope_column if scope_column in values else None
    group_columns = ([scope_column] if scope_column else []) + [community_column, "_frame", date_column]
    counts = values.groupby(group_columns).size().rename("documents").reset_index()
    rows = []
    uptake_group_columns = ([scope_column] if scope_column else []) + [community_column, "_frame"]
    for keys, group in counts.groupby(uptake_group_columns, sort=True):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        offset = 1 if scope_column else 0
        scope = keys[0] if scope_column else None
        community, frame_name = keys[offset], keys[offset + 1]
        dates = sorted(group[date_column].unique())
        for start in dates:
            windows = []
            for offset in range(consecutive_windows):
                lower = pd.Timestamp(start) + pd.Timedelta(days=offset * window_days)
                upper = lower + pd.Timedelta(days=window_days)
                count = int(group[group[date_column].between(lower, upper, inclusive="left")]["documents"].sum())
                windows.append(count)
            if all(count >= minimum_documents for count in windows):
                row = {"community_id": community, "frame": frame_name, "first_sustained_date": pd.Timestamp(start).date().isoformat(), "window_days": window_days, "consecutive_windows": consecutive_windows, "minimum_documents": minimum_documents}
                if scope_column:
                    row[scope_column] = scope
                rows.append(row)
                break
    columns = (["scope"] if scope_column else []) + ["community_id", "frame", "first_sustained_date", "window_days", "consecutive_windows", "minimum_documents"]
    return pd.DataFrame(rows, columns=columns)


def uptake_diagnostics(
    frame: pd.DataFrame,
    *,
    replicates: int = TEMPORAL_REPLICATES,
    seed: int = 20260915,
    cluster_column: str = "cluster_id",
) -> pd.DataFrame:
    """Return sustained uptake dates with clustered bootstrap and shuffled-label nulls."""

    if "scope" not in frame or frame["scope"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("uptake diagnostics require authoritative non-empty case/event scope")
    observed = first_sustained_uptake(frame, scope_column="scope")
    if observed.empty:
        return pd.DataFrame(columns=UPTAKE_COLUMNS)
    if replicates < 1:
        raise ValueError("uptake diagnostic replicates must be positive")
    cluster_column = cluster_column if cluster_column in frame else "community_id"
    values = frame.copy()
    values["_bootstrap_cluster"] = values["scope"].astype(str) + "|" + values[cluster_column].astype(str)
    clusters = np.asarray(sorted(values["_bootstrap_cluster"].unique()))
    by_cluster = {cluster: values[values["_bootstrap_cluster"].eq(cluster)] for cluster in clusters}
    rng = np.random.default_rng(seed)
    bootstrap: dict[tuple[str, str], list[int]] = defaultdict(list)
    null_values: dict[tuple[str, str], list[int]] = defaultdict(list)
    for _ in range(replicates):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([by_cluster[cluster] for cluster in sampled], ignore_index=True)
        for row in first_sustained_uptake(sample, scope_column="scope").itertuples(index=False):
            bootstrap[(str(row.scope), str(row.community_id), str(row.frame))].append(pd.Timestamp(row.first_sustained_date).toordinal())
        shuffled = _shuffle_labels_within_scope(values.drop(columns="_bootstrap_cluster"), rng)
        for row in first_sustained_uptake(shuffled, scope_column="scope").itertuples(index=False):
            null_values[(str(row.scope), str(row.community_id), str(row.frame))].append(pd.Timestamp(row.first_sustained_date).toordinal())
    output = []
    for row in observed.itertuples(index=False):
        key = (str(row.scope), str(row.community_id), str(row.frame))
        values = bootstrap.get(key, [])
        output.append(
            {
                "scope": row.scope,
                "community_id": row.community_id,
                "frame": row.frame,
                "first_sustained_date": row.first_sustained_date,
                "bootstrap_lower_95": pd.Timestamp.fromordinal(int(np.quantile(values, 0.025))).date().isoformat() if values else None,
                "bootstrap_median": pd.Timestamp.fromordinal(int(np.quantile(values, 0.5))).date().isoformat() if values else None,
                "bootstrap_upper_95": pd.Timestamp.fromordinal(int(np.quantile(values, 0.975))).date().isoformat() if values else None,
                "null_detected": int(len(null_values.get(key, []))),
                "null_replicates": int(replicates),
                "status": "available",
            }
        )
    return pd.DataFrame(output, columns=UPTAKE_COLUMNS)


def _community_lifecycle(previous: dict[str, set[str]], current: dict[str, set[str]], threshold: float = COMMUNITY_MATCH_THRESHOLD) -> dict[str, int]:
    overlaps = {
        (old_label, new_label): len(old_members & new_members) / len(old_members | new_members)
        for old_label, old_members in previous.items()
        for new_label, new_members in current.items()
        if old_members | new_members
    }
    births = sum(not any(overlaps.get((old, new), 0.0) >= threshold for old in previous) for new in current)
    deaths = sum(not any(overlaps.get((old, new), 0.0) >= threshold for new in current) for old in previous)
    splits = sum(sum(overlaps.get((old, new), 0.0) >= threshold for new in current) > 1 for old in previous)
    merges = sum(sum(overlaps.get((old, new), 0.0) >= threshold for old in previous) > 1 for new in current)
    return {"births": int(births), "deaths": int(deaths), "splits": int(splits), "merges": int(merges)}


def _temporal_communities(platform: str, edges: pd.DataFrame, validity_status: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if edges.empty:
        return pd.DataFrame(), pd.DataFrame()
    edges = edges.copy()
    edges["period"] = pd.to_datetime(edges["day"], errors="coerce").dt.to_period("M").astype(str)
    edges = edges[edges["period"].ne("NaT") & edges["source_author_hash"].fillna("").ne("") & edges["target_author_hash"].fillna("").ne("")]
    summaries, partition_rows = [], []
    previous: dict[str, dict[str, set[str]]] = {}
    for (scope, period), group in edges.groupby(["scope", "period"], sort=True):
        graph = nx.Graph()
        for row in group.itertuples():
            source, target = str(row.source_author_hash), str(row.target_author_hash)
            if source != target:
                graph.add_edge(source, target, weight=graph.get_edge_data(source, target, {}).get("weight", 0) + 1)
        if graph.number_of_nodes():
            communities = nx.community.louvain_communities(graph, weight="weight", seed=20260915)
        else:
            communities = []
        current = {}
        for index, community in enumerate(sorted(communities, key=lambda values: min(values) if values else "")):
            label = f"c{index:04d}"
            for node in community:
                current[node] = label
                partition_rows.append({"platform": platform, "scope": scope, "period": period, "node_id": node, "community_id": label})
        current_sets = {label: {node for node, value in current.items() if value == label} for label in set(current.values())}
        previous_sets = previous.get(scope, {})
        lifecycle = _community_lifecycle(previous_sets, current_sets)
        overlaps = [max((len(members & old) / len(members | old) for old in previous_sets.values()), default=0.0) for members in current_sets.values()]
        summaries.append(
            {
                "platform": platform,
                "scope": scope,
                "period": period,
                "nodes": int(graph.number_of_nodes()),
                "edges": int(graph.number_of_edges()),
                "communities": int(len(communities)),
                "modularity": float(nx.community.modularity(graph, communities, weight="weight")) if communities and graph.number_of_edges() else 0.0,
                "mean_best_previous_overlap": float(pd.Series(overlaps).mean()) if overlaps else None,
                **lifecycle,
                "community_match_threshold": COMMUNITY_MATCH_THRESHOLD,
                "method": "louvain_temporal_fallback",
                "validity_status": validity_status,
            }
        )
        previous[scope] = current_sets
    return pd.DataFrame(summaries), pd.DataFrame(partition_rows)


def _temporal_label_frame() -> pd.DataFrame | None:
    """Load adjudicated validation labels only when the full frame field is present."""

    try:
        from src.analysis.validation import load_label_packet

        labels = load_label_packet("evaluation")
    except (FileNotFoundError, ValueError):
        return None
    required = {"annotation_id", "doc_id", "platform", "adjudicated_frame_labels"}
    if not required <= set(labels.columns):
        return None
    labels = labels[labels["adjudicated_frame_labels"].astype(str).str.strip().ne("")].copy()
    return labels[["annotation_id", "doc_id", "platform", "adjudicated_frame_labels"]] if len(labels) else None


def _uptake_population_frame(
    platform: str,
    labels: pd.DataFrame,
    partitions: pd.DataFrame,
) -> tuple[pd.DataFrame | None, str]:
    """Join evaluation labels to dates, clusters and fresh temporal communities."""

    if platform == "reddit":
        source = pd.read_parquet(
            REPO_ROOT / "data" / "processed" / "reddit_age_gate" / "analysis_corpus.parquet",
            columns=["record_id", "event_id", "date", "author_hash", "thread_id"],
        ).rename(columns={"record_id": "doc_id", "thread_id": "cluster_id"})
        source["scope"] = source["event_id"].map(REDDIT_CASE_BY_EVENT)
    elif platform == "bluesky":
        source = pd.read_parquet(
            REPO_ROOT / "data" / "processed" / "bluesky" / "posts.parquet",
            columns=["doc_id", "event_window", "day", "author_hash", "root_doc_id"],
        ).rename(columns={"event_window": "scope", "day": "date", "root_doc_id": "cluster_id"})
        source = source[source["scope"].isin(["E2", "E3"])].copy()
    else:
        return None, f"unsupported uptake platform: {platform}"
    source["doc_id"] = source["doc_id"].astype(str)
    source["author_hash"] = source["author_hash"].fillna("").astype(str)
    source["cluster_id"] = source["cluster_id"].fillna("").astype(str)
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source = source[source["scope"].notna() & source["author_hash"].ne("") & source["cluster_id"].ne("") & source["date"].notna()]
    labelled = labels[labels["platform"].eq(platform)][["doc_id", "adjudicated_frame_labels"]].copy()
    labelled["doc_id"] = labelled["doc_id"].astype(str)
    joined = source.merge(labelled, on="doc_id", how="inner", validate="one_to_one")
    if joined.empty:
        return None, "no labelled documents joined to authoritative dates and clusters"
    joined["period"] = joined["date"].dt.to_period("M").astype(str)
    required = {"platform", "scope", "period", "node_id", "community_id"}
    if not required <= set(partitions.columns):
        return None, "authoritative temporal community partitions are unavailable"
    partitions = partitions[partitions["platform"].eq(platform)].copy()
    partitions["node_id"] = partitions["node_id"].astype(str)
    joined = joined.merge(
        partitions,
        left_on=["scope", "period", "author_hash"],
        right_on=["scope", "period", "node_id"],
        how="inner",
        validate="many_to_one",
    )
    if joined.empty:
        return None, "labelled documents have no authoritative temporal community assignment"
    return joined[["scope", "date", "community_id", "cluster_id", "adjudicated_frame_labels"]].rename(columns={"adjudicated_frame_labels": "frame_labels"}), ""


def _blocked_transition_rows(platform: str, reason: str) -> pd.DataFrame:
    return pd.DataFrame([{**{"platform": platform}, **{column: None for column in TRANSITION_COLUMNS[:-1]}, "status": "human-labels-required", "reason": reason}])


def _blocked_uptake_rows(platform: str, reason: str) -> pd.DataFrame:
    return pd.DataFrame([{**{"platform": platform}, **{column: None for column in UPTAKE_COLUMNS[:-1]}, "status": "human-labels-required", "reason": reason}])


def _network_checksums() -> dict[str, str]:
    return {"network_manifest": sha256_file(NETWORK_MANIFEST)}


def build_temporal_artifacts() -> dict[str, Any]:
    check_networks()
    TEMPORAL_ROOT.mkdir(parents=True, exist_ok=True)
    cascade_frames, community_frames, partition_frames = [], [], []
    transition_frames, uptake_frames = [], []
    labels = _temporal_label_frame()
    uptake_statuses = []
    for platform, (graph_name, status) in MESSAGE_GRAPHS.items():
        edges = pd.read_parquet(NETWORK_ROOT / f"{graph_name.lower()}.parquet")
        cascade_frames.append(_cascade_summary(platform, edges, status, _root_records(platform)))
        communities, partitions = _temporal_communities(platform, edges, status)
        community_frames.append(communities)
        partition_frames.append(partitions)
        if labels is None:
            transition_frames.append(_blocked_transition_rows(platform, "adjudicated evaluation frame labels are unavailable"))
            uptake_frames.append(_blocked_uptake_rows(platform, "population-level frame labels and temporal community assignments are unavailable"))
            uptake_statuses.append("human-labels-required")
        else:
            platform_labels = labels[labels["platform"].eq(platform)][["doc_id", "adjudicated_frame_labels"]].rename(columns={"adjudicated_frame_labels": "frame_labels"})
            transitions = transition_diagnostics(edges, platform_labels)
            if transitions.empty:
                transition_frames.append(_blocked_transition_rows(platform, "no observed labelled parent-child transitions"))
            else:
                transitions.insert(0, "platform", platform)
                transition_frames.append(transitions)
            uptake_frame, uptake_reason = _uptake_population_frame(platform, labels, partitions)
            if uptake_frame is None:
                uptake_frames.append(_blocked_uptake_rows(platform, uptake_reason))
                uptake_statuses.append("human-labels-required")
            else:
                try:
                    uptake = uptake_diagnostics(uptake_frame)
                except ValueError as error:
                    uptake = pd.DataFrame()
                    uptake_reason = str(error)
                if uptake.empty:
                    uptake_frames.append(_blocked_uptake_rows(platform, uptake_reason or "no sustained labelled uptake result"))
                    uptake_statuses.append("human-labels-required")
                else:
                    uptake.insert(0, "platform", platform)
                    uptake_frames.append(uptake)
                    uptake_statuses.append("available")
    outputs = {
        "cascade_summary": TEMPORAL_ROOT / "cascade_summary.csv",
        "temporal_communities": TEMPORAL_ROOT / "temporal_communities.csv",
        "temporal_partitions": TEMPORAL_ROOT / "temporal_partitions.parquet",
        "frame_transition_diagnostics": TEMPORAL_ROOT / "frame_transition_diagnostics.csv",
        "uptake_diagnostics": TEMPORAL_ROOT / "uptake_diagnostics.csv",
    }
    pd.concat(cascade_frames, ignore_index=True).sort_values(["platform", "scope", "container_id"]).to_csv(outputs["cascade_summary"], index=False, lineterminator="\n")
    pd.concat(community_frames, ignore_index=True).sort_values(["platform", "scope", "period"]).to_csv(outputs["temporal_communities"], index=False, lineterminator="\n")
    pd.concat(partition_frames, ignore_index=True).sort_values(["platform", "scope", "period", "node_id"]).to_parquet(outputs["temporal_partitions"], compression="zstd", index=False)
    pd.concat(transition_frames, ignore_index=True).sort_values(["platform", "status", "source_frame", "target_frame"], na_position="last").to_csv(outputs["frame_transition_diagnostics"], index=False, lineterminator="\n")
    pd.concat(uptake_frames, ignore_index=True).sort_values(["platform", "status", "scope", "community_id", "frame"], na_position="last").to_csv(outputs["uptake_diagnostics"], index=False, lineterminator="\n")
    transition_status = "available" if labels is not None and any(frame["status"].eq("available").any() for frame in transition_frames) else "human-labels-required"
    manifest = {
        "schema_version": "network-temporal.v2",
        "created_at_utc": utc_now_iso(),
        "command": "python -m src.analysis.run networks temporal --check",
        "network_checksums": _network_checksums(),
        "message_graphs": {platform: graph for platform, (graph, _) in MESSAGE_GRAPHS.items()},
        "implemented_methods": {
            "root_document_registry": {"status": "computed", "function": "_root_records"},
            "frame_transitions": {"function": "transition_diagnostics", "status": transition_status, "replicates": TEMPORAL_REPLICATES, "null": "within-scope shuffled labels"},
            "sustained_uptake": {"function": "uptake_diagnostics", "status": "available" if "available" in uptake_statuses else "human-labels-required", "replicates": TEMPORAL_REPLICATES, "null": "within-scope shuffled labels"},
            "community_lifecycle": {"function": "_community_lifecycle", "status": "computed"},
        },
        "outputs": {
            name: {"path": relative_path(path), "sha256": sha256_file(path), "rows": int(len(pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)))}
            for name, path in outputs.items()
        },
        "limitations": [
            "Only observed parent-child trees are analysed; YouTube heuristic reply resolution is excluded.",
            "Reddit results remain candidate until thread relevance is human-audited.",
            "Temporal communities use a descriptive Louvain temporal fallback; static structure slices use the pinned Leiden method.",
            "Transition uncertainty and shuffled-label diagnostics are emitted only for labelled observed edges; uptake remains gated until population-level labels are joined to temporal communities.",
        ],
    }
    write_json(TEMPORAL_MANIFEST, manifest)
    return manifest


def check_temporal() -> dict[str, Any]:
    check_networks()
    if not TEMPORAL_MANIFEST.exists():
        raise FileNotFoundError(f"temporal manifest is missing: {TEMPORAL_MANIFEST}; run the explicit temporal build first")
    manifest = read_json(TEMPORAL_MANIFEST)
    if manifest.get("schema_version") != "network-temporal.v2":
        raise ValueError("temporal manifest schema is stale; run the explicit temporal build")
    if manifest.get("network_checksums") != _network_checksums():
        raise ValueError("temporal network input changed")
    for record in manifest.get("outputs", {}).values():
        path = REPO_ROOT / record["path"]
        if not path.exists() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"temporal artifact changed: {record['path']}")
    return {"status": "valid", "artifact": relative_path(TEMPORAL_MANIFEST), "outputs": len(manifest.get("outputs", {}))}
