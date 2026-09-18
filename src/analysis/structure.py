"""Topology-only communities, centrality and structural-role diagnostics."""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

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
from src.analysis.measurement import STANCE_FRAME_SELECTION, _frame_evaluation_mask
from src.analysis.networks import NETWORK_MANIFEST, NETWORK_ROOT, check_networks
from src.analysis.validation import load_label_packet

STRUCTURE_ROOT = ANALYSIS_ROOT / "networks" / "structure"
STRUCTURE_MANIFEST = STRUCTURE_ROOT / "structure_manifest.json"
RESOLUTIONS = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
SEEDS = tuple(range(20))
NULL_REPS = 10
BOOTSTRAP_REPS = 5
TOP_K = 100
BETWEENNESS_EXACT_NODE_LIMIT = 5000
BETWEENNESS_STABILITY_SEEDS = (1729, 1730)
BETWEENNESS_STABILITY_THRESHOLD = 0.80
STRUCTURAL_GRAPHS = (
    "R_ACTOR_ATTENTION",
    "B_ACTOR_ATTENTION",
    "Y_ACTOR_ATTENTION_HANDLE",
    "R_AUTHOR_SUBREDDIT",
    "Y_COMMENTER_VIDEO",
)
CASCADE_GRAPHS = {
    "R_ACTOR_ATTENTION": "R_ACTOR_CASCADE",
    "B_ACTOR_ATTENTION": "B_ACTOR_CASCADE",
    "Y_ACTOR_ATTENTION_HANDLE": "Y_ACTOR_CASCADE_HANDLE",
}
FRAME_LABELS = [frame["label"] for frame in FRAME_DEFINITIONS]
FRAME_PAGERANK_COLUMNS = ["graph_id", "scope", "node_id", "frame", "pagerank", "alpha"]


def _load_edges(name: str, scope: str | None = None) -> pd.DataFrame:
    path = NETWORK_ROOT / f"{name.lower()}.parquet"
    edges = pd.read_parquet(path)
    if scope is not None and "scope" in edges:
        edges = edges[edges["scope"].astype(str).eq(scope)].copy()
    return edges


def scoped_structure_frames(edges: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if edges.empty or "scope" not in edges:
        return []
    return [(str(scope), edges.loc[edges["scope"].astype(str).eq(str(scope))].copy().reset_index(drop=True)) for scope in sorted(edges["scope"].dropna().astype(str).unique())]


def _add_edges(graph: nx.Graph, frame: pd.DataFrame, directed: bool) -> nx.Graph:
    for row in frame.itertuples(index=False):
        source, target = str(row.source_id), str(row.target_id)
        if not source or not target or source == target:
            continue
        weight = float(row.weight) if pd.notna(row.weight) else 1.0
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += weight
        else:
            graph.add_edge(source, target, weight=weight)
    return graph


def _directed_graph(frame: pd.DataFrame) -> nx.DiGraph:
    return _add_edges(nx.DiGraph(), frame, True)  # type: ignore[return-value]


def _undirected_graph(frame: pd.DataFrame) -> nx.Graph:
    return _add_edges(nx.Graph(), frame, False)


def _leiden_partition(graph: nx.Graph, resolution: float, seed: int) -> dict[str, int] | None:
    try:
        import igraph as ig
        import leidenalg

        nodes = sorted(graph)
        index = {node: position for position, node in enumerate(nodes)}
        igraph = ig.Graph(
            n=len(nodes),
            edges=[(index[source], index[target]) for source, target in graph.edges()],
            directed=False,
        )
        weights = [float(graph[source][target].get("weight", 1.0)) for source, target in graph.edges()]
        partition = leidenalg.find_partition(
            igraph,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights or None,
            resolution_parameter=resolution,
            seed=seed,
        )
        return {node: int(partition.membership[position]) for position, node in enumerate(nodes)}
    except Exception:
        return None


def _louvain_partition(graph: nx.Graph, resolution: float, seed: int) -> dict[str, int]:
    communities = nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=seed)
    result: dict[str, int] = {}
    for community_id, community in enumerate(sorted(communities, key=lambda values: min(values) if values else "")):
        result.update({node: community_id for node in community})
    return result


def _community_metrics(graph: nx.Graph, partition: dict[str, int]) -> dict[str, Any]:
    groups = {}
    for node, label in partition.items():
        groups.setdefault(label, set()).add(node)
    labels = list(groups.values())
    sizes = sorted((len(group) for group in labels), reverse=True)
    return {
        "communities": len(labels),
        "largest_community_share": (sizes[0] / graph.number_of_nodes()) if sizes and graph.number_of_nodes() else 0.0,
        "modularity": float(nx.community.modularity(graph, labels, weight="weight")) if labels and graph.number_of_edges() else 0.0,
        "community_size_min": min(sizes) if sizes else 0,
        "community_size_median": float(pd.Series(sizes).median()) if sizes else 0.0,
        "community_size_max": max(sizes) if sizes else 0,
    }


def _partition_stability(partitions: list[dict[str, int]]) -> tuple[float | None, float | None]:
    if len(partitions) < 2:
        return None, None
    nodes = sorted(partitions[0])
    aris, nmis = [], []
    for left, right in itertools.combinations(partitions, 2):
        aris.append(adjusted_rand_score([left[node] for node in nodes], [right[node] for node in nodes]))
        nmis.append(normalized_mutual_info_score([left[node] for node in nodes], [right[node] for node in nodes]))
    return float(pd.Series(aris).median()), float(pd.Series(nmis).median())


def _canonical_partition(partition: dict[str, int]) -> dict[str, str]:
    groups: dict[int, list[str]] = {}
    for node, label in partition.items():
        groups.setdefault(label, []).append(node)
    ordered = sorted(groups.items(), key=lambda item: min(item[1]) if item[1] else "")
    labels = {old: f"c{position:04d}" for position, (old, _) in enumerate(ordered)}
    return {node: labels[label] for node, label in partition.items()}


def _choose_resolution(candidates: pd.DataFrame, tie_tolerance: float = 0.01) -> float:
    if candidates.empty:
        raise ValueError("at least one resolution candidate is required")
    best_modularity = float(candidates["mean_modularity"].max())
    tied = candidates[candidates["mean_modularity"] >= best_modularity - tie_tolerance]
    return float(tied["resolution"].min())


def _select_partition(graph: nx.Graph) -> tuple[dict[str, str], dict[str, Any], pd.DataFrame]:
    method = "leiden"
    probe = _leiden_partition(graph, RESOLUTIONS[0], SEEDS[0])
    if probe is None:
        method = "louvain_fallback"
    resolutions = RESOLUTIONS if method == "leiden" else (1.00,)
    rows: list[dict[str, Any]] = []
    runs: dict[float, list[tuple[int, dict[str, int], dict[str, Any]]]] = {}
    for resolution in resolutions:
        resolution_runs = []
        for seed in SEEDS:
            partition = (
                _leiden_partition(graph, resolution, seed)
                if method == "leiden"
                else _louvain_partition(graph, resolution, seed)
            )
            if partition is None:
                method = "louvain_fallback"
                partition = _louvain_partition(graph, 1.00, seed)
            metrics = _community_metrics(graph, partition)
            resolution_runs.append((seed, partition, metrics))
        runs[resolution] = resolution_runs
        ari, nmi = _partition_stability([run[1] for run in resolution_runs])
        mean_modularity = float(pd.Series([run[2]["modularity"] for run in resolution_runs]).mean())
        rows.append(
            {
                "resolution": resolution,
                "method": method,
                "seed_count": len(resolution_runs),
                "median_pairwise_ari": ari,
                "median_pairwise_nmi": nmi,
                "mean_modularity": mean_modularity,
                "eligible": bool(
                    method == "leiden"
                    and (ari or 0.0) >= 0.80
                    and max(run[2]["communities"] for run in resolution_runs) >= 3
                    and max(run[2]["largest_community_share"] for run in resolution_runs) <= 0.80
                ),
            }
        )
    stability = pd.DataFrame(rows)
    eligible = stability[stability["eligible"]]
    selection_candidates = eligible if not eligible.empty else stability
    if eligible.empty:
        selected_resolution = _choose_resolution(selection_candidates)
        selection_status = "inconclusive_leiden_unavailable" if method != "leiden" else "inconclusive_no_stable_resolution"
    else:
        selected_resolution = _choose_resolution(selection_candidates)
        selection_status = "primary_leiden_stable"
    selected_runs = runs[selected_resolution]
    best_seed, best_partition, best_metrics = sorted(
        selected_runs, key=lambda run: (-run[2]["modularity"], run[0])
    )[0]
    selected_stability = stability.loc[stability["resolution"].eq(selected_resolution)].iloc[0]
    if method == "leiden":
        cross_check_partition = _louvain_partition(graph, selected_resolution, best_seed)
        cross_check_metrics = _community_metrics(graph, cross_check_partition)
        cross_check = {
            "method": "louvain",
            "status": "computed",
            "resolution": selected_resolution,
            "seed": best_seed,
            "communities": cross_check_metrics["communities"],
            "modularity": cross_check_metrics["modularity"],
            "modularity_delta": float(best_metrics["modularity"] - cross_check_metrics["modularity"]),
        }
    else:
        cross_check = {"method": "leiden", "status": "unavailable"}
    selection = {
        "method": method,
        "resolution": selected_resolution,
        "seed": int(best_seed),
        "status": selection_status,
        "median_pairwise_ari": selected_stability["median_pairwise_ari"],
        "median_pairwise_nmi": selected_stability["median_pairwise_nmi"],
        "cross_check": cross_check,
        **best_metrics,
    }
    return _canonical_partition(best_partition), selection, stability


def _degree_preserving_null_modularity(graph: nx.Graph, partition: dict[str, str]) -> dict[str, Any]:
    if graph.number_of_edges() < 2:
        return {"replicates": 0, "median_modularity": None, "observed_minus_null": None, "status": "too_few_edges"}
    groups: dict[str, set[str]] = {}
    for node, label in partition.items():
        groups.setdefault(label, set()).add(node)
    observed = float(nx.community.modularity(graph, list(groups.values()), weight="weight"))
    values = []
    for seed in range(NULL_REPS):
        null = graph.copy()
        try:
            swaps = min(2000, max(10, 5 * null.number_of_edges()))
            nx.double_edge_swap(null, nswap=swaps, max_tries=max(100, swaps * 10), seed=seed)
            weights = sorted((float(data.get("weight", 1.0)) for _, _, data in graph.edges(data=True)), reverse=True)
            random.Random(seed).shuffle(weights)
            for (source, target), weight in zip(null.edges(), weights):
                null[source][target]["weight"] = weight
            values.append(float(nx.community.modularity(null, list(groups.values()), weight="weight")))
        except (nx.NetworkXAlgorithmError, nx.NetworkXError):
            continue
    median = float(pd.Series(values).median()) if values else None
    return {
        "replicates": len(values),
        "median_modularity": median,
        "observed_minus_null": observed - median if median is not None else None,
        "status": "degree_preserving_untyped_null",
    }


def _rank_correlation(left: dict[str, float], right: dict[str, float]) -> float | None:
    nodes = sorted(set(left) & set(right))
    if len(nodes) < 2:
        return None
    left_ranks = pd.Series({node: left[node] for node in nodes}).rank()
    right_ranks = pd.Series({node: right[node] for node in nodes}).rank()
    if left_ranks.nunique() < 2 or right_ranks.nunique() < 2:
        return None
    return float(left_ranks.corr(right_ranks))


def _top_jaccard(left: dict[str, float], right: dict[str, float], k: int = TOP_K) -> float | None:
    if not left or not right:
        return None
    left_top = set(sorted(left, key=lambda node: (-left[node], node))[:k])
    right_top = set(sorted(right, key=lambda node: (-right[node], node))[:k])
    union = left_top | right_top
    return len(left_top & right_top) / len(union) if union else 1.0


def _betweenness_scores(graph: nx.Graph) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute unweighted betweenness and record the top-decile stability check."""

    node_count = graph.number_of_nodes()
    if not node_count:
        return {}, {
            "method": "none",
            "pivots": 0,
            "top_decile_jaccard": None,
            "stability_status": "unavailable",
            "weighting": "unweighted_shortest_paths",
        }
    nodes = sorted(str(node) for node in graph)
    index = {node: position for position, node in enumerate(nodes)}
    try:
        import igraph as ig

        igraph = ig.Graph(
            n=len(nodes),
            edges=[(index[str(source)], index[str(target)]) for source, target in graph.edges()],
            directed=False,
        )
        values = igraph.betweenness(directed=False, weights=None)
        normalizer = 2 / ((node_count - 1) * (node_count - 2)) if node_count > 2 else 1.0
        scores = {node: float(value * normalizer) for node, value in zip(nodes, values)}
        return scores, {
            "method": "igraph_exact",
            "pivots": node_count,
            "top_decile_jaccard": 1.0,
            "stability_status": "stable",
            "weighting": "unweighted_shortest_paths",
        }
    except (ImportError, OSError, RuntimeError, ValueError):
        pivots = None if node_count <= BETWEENNESS_EXACT_NODE_LIMIT else max(2000, node_count // 10)
        scores = nx.betweenness_centrality(graph, k=pivots, normalized=True, weight=None, seed=BETWEENNESS_STABILITY_SEEDS[0])
        comparison = nx.betweenness_centrality(graph, k=pivots, normalized=True, weight=None, seed=BETWEENNESS_STABILITY_SEEDS[1])
        top_decile_jaccard = _top_jaccard(scores, comparison, k=max(1, math.ceil(node_count * 0.10)))
        return scores, {
            "method": "networkx_exact_fallback" if pivots is None else "networkx_sampled_fallback",
            "pivots": node_count if pivots is None else pivots,
            "top_decile_jaccard": top_decile_jaccard,
            "stability_status": "stable" if (top_decile_jaccard or 0.0) >= BETWEENNESS_STABILITY_THRESHOLD else "unstable",
            "weighting": "unweighted_shortest_paths",
        }


def personalized_frame_pagerank(
    graph: nx.Graph,
    profiles: pd.DataFrame,
    frame_columns: list[str],
    *,
    node_column: str = "node_id",
    alpha: float = 0.85,
) -> pd.DataFrame:
    """Compute one personalized PageRank vector per validated frame profile."""

    required = {node_column, *frame_columns}
    if not required <= set(profiles.columns):
        raise ValueError(f"frame profiles missing columns: {sorted(required - set(profiles.columns))}")
    rows = []
    for frame_column in frame_columns:
        values = pd.to_numeric(profiles.set_index(node_column)[frame_column], errors="raise").clip(lower=0)
        personalization = {node: float(values.get(node, 0.0)) for node in graph}
        total = sum(personalization.values())
        if not total:
            continue
        ranks = nx.pagerank(graph, alpha=alpha, weight="weight", personalization={node: value / total for node, value in personalization.items()}, max_iter=300)
        rows.extend({"node_id": node, "frame": frame_column, "pagerank": float(score), "alpha": alpha} for node, score in ranks.items())
    return pd.DataFrame(rows, columns=["node_id", "frame", "pagerank", "alpha"])


def _validated_frame_profiles() -> tuple[pd.DataFrame, str]:
    """Return adjudicated author profiles only after the frozen frame model gate."""

    try:
        labels = load_label_packet("development")
        selection = read_json(STANCE_FRAME_SELECTION)
        frame_selection = selection.get("results", {}).get("frames", {})
    except (FileNotFoundError, KeyError, ValueError):
        return pd.DataFrame(), "human-labels-required"
    required = {"platform", "case_window_id", "author_key", "adjudicated_frame_labels"}
    if not required <= set(labels.columns):
        return pd.DataFrame(), "human-labels-required"
    labels = labels.loc[_frame_evaluation_mask(labels)].copy()
    if labels.empty:
        return pd.DataFrame(), "human-labels-required"
    if frame_selection.get("status") != "selected":
        return pd.DataFrame(), "measurement-selection-required"
    rows = []
    for row in labels.itertuples(index=False):
        active = set(str(row.adjudicated_frame_labels).split("|"))
        rows.append({
            "platform": str(row.platform),
            "scope": str(row.case_window_id),
            "node_id": str(row.author_key),
            **{label: int(label in active) for label in FRAME_LABELS},
        })
    profiles = pd.DataFrame(rows)
    profiles = profiles.groupby(["platform", "scope", "node_id"], sort=True)[FRAME_LABELS].mean().reset_index()
    return profiles, "ready"


def _frame_personalized_rankings() -> tuple[pd.DataFrame, str]:
    profiles, status = _validated_frame_profiles()
    if status != "ready":
        return pd.DataFrame(columns=FRAME_PAGERANK_COLUMNS), status
    rows = []
    graph_platform = {"R": "reddit", "B": "bluesky", "Y": "youtube"}
    for graph_name in STRUCTURAL_GRAPHS:
        platform = graph_platform[graph_name[0]]
        edges = _load_edges(graph_name)
        for scope, scoped_edges in scoped_structure_frames(edges):
            graph = _undirected_graph(scoped_edges)
            profile_slice = profiles[profiles["platform"].eq(platform) & profiles["scope"].eq(scope)].drop(columns=["platform", "scope"])
            if profile_slice.empty or graph.number_of_nodes() == 0:
                continue
            ranked = personalized_frame_pagerank(graph, profile_slice, FRAME_LABELS)
            if ranked.empty:
                continue
            ranked.insert(0, "scope", scope)
            ranked.insert(0, "graph_id", graph_name)
            rows.append(ranked)
    return (pd.concat(rows, ignore_index=True)[FRAME_PAGERANK_COLUMNS] if rows else pd.DataFrame(columns=FRAME_PAGERANK_COLUMNS), "ready")


def bipartite_rankings(edges: pd.DataFrame) -> pd.DataFrame:
    """Return HITS and a normalized bipartite rank for author-to-target edges."""

    if edges.empty:
        return pd.DataFrame(columns=["node_id", "node_type", "hits_hub", "hits_authority", "birank"])
    graph = _directed_graph(edges)
    try:
        hubs, authorities = nx.hits(graph, max_iter=500, normalized=True)
    except nx.NetworkXException:
        hubs, authorities = ({node: 0.0 for node in graph}, {node: 0.0 for node in graph})
    source_nodes = set(edges["source_id"].astype(str))
    target_nodes = set(edges["target_id"].astype(str))
    author_scores = {node: 1.0 for node in source_nodes}
    target_scores = {node: 1.0 for node in target_nodes}
    for _ in range(100):
        next_authors = {node: sum(float(data.get("weight", 1.0)) * target_scores.get(target, 0.0) for _, target, data in graph.out_edges(node, data=True)) for node in source_nodes}
        next_targets = {node: sum(float(data.get("weight", 1.0)) * author_scores.get(source, 0.0) for source, _, data in graph.in_edges(node, data=True)) for node in target_nodes}
        author_total, target_total = sum(next_authors.values()), sum(next_targets.values())
        if author_total:
            author_scores = {node: value / author_total for node, value in next_authors.items()}
        if target_total:
            target_scores = {node: value / target_total for node, value in next_targets.items()}
    rows = []
    for node in sorted(graph):
        rows.append({"node_id": node, "node_type": _node_type(node), "hits_hub": float(hubs.get(node, 0.0)), "hits_authority": float(authorities.get(node, 0.0)), "birank": float(author_scores.get(node, target_scores.get(node, 0.0)))})
    return pd.DataFrame(rows)


def _participation_and_zscore(graph: nx.Graph, partition: dict[str, str]) -> tuple[dict[str, float], dict[str, float]]:
    strengths = dict(graph.degree(weight="weight"))
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    within_strengths: dict[str, float] = defaultdict(float)
    for source, target, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        source_group, target_group = partition.get(source), partition.get(target)
        if source_group is not None:
            totals[source][target_group] += weight
            if source_group == target_group:
                within_strengths[source] += weight
        if target_group is not None:
            totals[target][source_group] += weight
            if source_group == target_group:
                within_strengths[target] += weight
    participation = {}
    for node, strength in strengths.items():
        if not strength:
            participation[node] = 0.0
        else:
            participation[node] = 1.0 - sum((value / strength) ** 2 for value in totals[node].values())
    by_group: dict[str, list[float]] = defaultdict(list)
    for node, group in partition.items():
        by_group[group].append(within_strengths.get(node, 0.0))
    zscore = {}
    for node, group in partition.items():
        values = pd.Series(by_group[group], dtype=float)
        scale = float(values.std(ddof=0))
        zscore[node] = (within_strengths.get(node, 0.0) - float(values.mean())) / scale if scale else 0.0
    return participation, zscore


def _node_type(node: str) -> str:
    if node.startswith("video:"):
        return "video"
    if node.startswith("subreddit:"):
        return "subreddit"
    return "author"


def _role_frame(
    graph_name: str,
    edges: pd.DataFrame,
    partition: dict[str, str],
    scope: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    undirected = _undirected_graph(edges)
    directed = _directed_graph(edges) if "ACTOR" in graph_name else None
    cascade_name = CASCADE_GRAPHS.get(graph_name)
    cascade = _directed_graph(_load_edges(cascade_name, scope)) if cascade_name else undirected
    pageranks: dict[str, dict[str, float]] = {}
    for damping in (0.75, 0.85, 0.95):
        pageranks[f"weighted_{str(damping).replace('.', '')}"] = nx.pagerank(
            directed or undirected, alpha=damping, weight="weight", max_iter=300
        )
    binary_graph = (directed or undirected).copy()
    for _, _, data in binary_graph.edges(data=True):
        data["weight"] = 1.0
    pageranks["binary_085"] = nx.pagerank(binary_graph, alpha=0.85, weight="weight", max_iter=300)
    base_pr = pageranks["weighted_085"]
    betweenness, betweenness_metadata = _betweenness_scores(undirected)
    participation, zscore = _participation_and_zscore(undirected, partition)
    core = nx.core_number(undirected) if undirected.number_of_edges() else {node: 0 for node in undirected}
    try:
        vote_nodes = nx.voterank(cascade, number_of_nodes=min(TOP_K, cascade.number_of_nodes())) if cascade.number_of_nodes() else []
        voterank_status = "cascade_oriented"
    except Exception:
        vote_nodes = nx.voterank(undirected, number_of_nodes=min(TOP_K, undirected.number_of_nodes())) if undirected.number_of_nodes() else []
        voterank_status = "undirected_fallback"
    vote_rank = {node: rank + 1 for rank, node in enumerate(vote_nodes)}
    rows = []
    in_strength = dict(directed.in_degree(weight="weight")) if directed else dict(undirected.degree(weight="weight"))
    out_strength = dict(directed.out_degree(weight="weight")) if directed else dict(undirected.degree(weight="weight"))
    for node in sorted(undirected):
        rows.append(
            {
                "graph_id": graph_name,
                "scope": scope or "",
                "node_id": node,
                "node_type": _node_type(node),
                "community_id": partition.get(node, ""),
                "in_strength": float(in_strength.get(node, 0.0)),
                "out_strength": float(out_strength.get(node, 0.0)),
                "pagerank_weighted_075": pageranks["weighted_075"].get(node, 0.0),
                "pagerank_weighted_085": pageranks["weighted_085"].get(node, 0.0),
                "pagerank_weighted_095": pageranks["weighted_095"].get(node, 0.0),
                "pagerank_binary_085": pageranks["binary_085"].get(node, 0.0),
                "betweenness": betweenness.get(node, 0.0),
                "betweenness_method": betweenness_metadata["method"],
                "betweenness_pivots": betweenness_metadata["pivots"],
                "betweenness_top_decile_jaccard": betweenness_metadata["top_decile_jaccard"],
                "betweenness_stability_status": betweenness_metadata["stability_status"],
                "betweenness_weighting": betweenness_metadata["weighting"],
                "participation_coefficient": participation.get(node, 0.0),
                "within_module_z": zscore.get(node, 0.0),
                "k_core": int(core.get(node, 0)),
                "voterank_rank": vote_rank.get(node),
            }
        )
    correlations = []
    for label in ("weighted_075", "weighted_095", "binary_085"):
        correlations.append(
            {
                "graph_id": graph_name,
                "scope": scope or "",
                "comparison": f"weighted_085_vs_{label}",
                "rank_correlation": _rank_correlation(base_pr, pageranks[label]),
                "top_k_jaccard": _top_jaccard(base_pr, pageranks[label]),
                "replicates": 1,
                "status": "damping_or_binary_sensitivity",
            }
        )
    bootstrap_rng = random.Random(20260915)
    for replicate in range(BOOTSTRAP_REPS):
        if edges.empty:
            continue
        sample = edges.iloc[[bootstrap_rng.randrange(len(edges)) for _ in range(len(edges))]].copy()
        sample = (
            sample.groupby(["source_id", "target_id", "scope"], sort=True, as_index=False)["weight"]
            .sum()
            .assign(source_doc_id="", target_doc_id="")
        )
        bootstrap_graph = _directed_graph(sample) if directed else _undirected_graph(sample)
        boot_pr = nx.pagerank(bootstrap_graph, alpha=0.85, weight="weight", max_iter=300)
        correlations.append(
            {
                "graph_id": graph_name,
                "scope": scope or "",
                "comparison": "weighted_085_vs_edge_bootstrap",
                "rank_correlation": _rank_correlation(base_pr, boot_pr),
                "top_k_jaccard": _top_jaccard(base_pr, boot_pr),
                "replicate": replicate,
                "replicates": BOOTSTRAP_REPS,
                "status": "edge_bootstrap",
            }
        )
    if base_pr:
        hubs = max(1, math.ceil(0.01 * len(base_pr)))
        removed = set(sorted(base_pr, key=lambda node: (-base_pr[node], node))[:hubs])
        hub_graph = (directed or undirected).copy()
        hub_graph.remove_nodes_from(removed)
        hub_pr = nx.pagerank(hub_graph, alpha=0.85, weight="weight", max_iter=300) if hub_graph else {}
        correlations.append(
            {
                "graph_id": graph_name,
                "scope": scope or "",
                "comparison": "weighted_085_after_top_1pct_hub_removal",
                "rank_correlation": _rank_correlation(base_pr, hub_pr),
                "top_k_jaccard": _top_jaccard(base_pr, hub_pr),
                "removed_nodes": len(removed),
                "replicates": 1,
                "status": "hub_removal",
            }
        )
    for row in correlations:
        row.setdefault("replicate", 0)
        row.setdefault("removed_nodes", 0)
    for row in rows:
        row["voterank_status"] = voterank_status
    return pd.DataFrame(rows), correlations, betweenness_metadata


def _summary_row(
    name: str,
    scope: str,
    graph: nx.DiGraph | nx.Graph,
    undirected: nx.Graph,
    selection: dict[str, Any],
    null: dict[str, Any],
    betweenness: dict[str, Any],
) -> dict[str, Any]:
    directed = graph.is_directed()
    components = list(nx.weakly_connected_components(graph)) if directed else list(nx.connected_components(graph))
    strengths = [float(value) for _, value in undirected.degree(weight="weight")]
    weighted_edge_total = float(sum(float(data.get("weight", 1.0)) for _, _, data in graph.edges(data=True)))
    return {
        "graph_id": name,
        "scope": scope,
        "directed": directed,
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "weighted_edge_total": weighted_edge_total,
        "density": float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0,
        "components": int(len(components)),
        "largest_component_share": (max(map(len, components)) / graph.number_of_nodes()) if components else 0.0,
        "mean_weighted_degree": float(pd.Series(strengths).mean()) if strengths else 0.0,
        "median_weighted_degree": float(pd.Series(strengths).median()) if strengths else 0.0,
        "reciprocity": float(nx.reciprocity(graph)) if directed and graph.number_of_edges() else None,
        "community_method": selection["method"],
        "community_resolution": selection["resolution"],
        "community_status": selection["status"],
        "communities": selection["communities"],
        "community_modularity": selection["modularity"],
        "community_largest_share": selection["largest_community_share"],
        "community_median_ari": selection.get("median_pairwise_ari"),
        "community_median_nmi": selection.get("median_pairwise_nmi"),
        "community_cross_check_method": selection["cross_check"]["method"],
        "community_cross_check_status": selection["cross_check"]["status"],
        "community_cross_check_modularity": selection["cross_check"].get("modularity"),
        "community_cross_check_modularity_delta": selection["cross_check"].get("modularity_delta"),
        "null_replicates": null["replicates"],
        "null_median_modularity": null["median_modularity"],
        "observed_minus_null_modularity": null["observed_minus_null"],
        "betweenness_method": betweenness["method"],
        "betweenness_pivots": betweenness["pivots"],
        "betweenness_top_decile_jaccard": betweenness["top_decile_jaccard"],
        "betweenness_stability_status": betweenness["stability_status"],
        "betweenness_weighting": betweenness["weighting"],
    }


def build_structure_artifacts() -> dict[str, Any]:
    check_networks()
    network_manifest = read_json(NETWORK_MANIFEST)
    STRUCTURE_ROOT.mkdir(parents=True, exist_ok=True)
    summaries, stability_rows, role_frames, rank_rows, partition_rows, bipartite_rank_frames = [], [], [], [], [], []
    structure_ids = []
    for name in STRUCTURAL_GRAPHS:
        edges = _load_edges(name)
        for scope, scoped_edges in scoped_structure_frames(edges):
            structure_ids.append(f"{name}:{scope}")
            graph = _directed_graph(scoped_edges) if "ACTOR" in name else _undirected_graph(scoped_edges)
            undirected = graph.to_undirected() if graph.is_directed() else graph
            partition, selection, stability = _select_partition(undirected)
            null = _degree_preserving_null_modularity(undirected, partition)
            stability.insert(0, "scope", scope)
            stability.insert(0, "graph_id", name)
            stability_rows.extend(stability.to_dict(orient="records"))
            partition_rows.extend(
                {"graph_id": name, "scope": scope, "node_id": node, "community_id": community, "node_type": _node_type(node)}
                for node, community in partition.items()
            )
            roles, correlations, betweenness = _role_frame(name, scoped_edges, partition, scope)
            summaries.append(_summary_row(name, scope, graph, undirected, selection, null, betweenness))
            role_frames.append(roles)
            rank_rows.extend(correlations)
            if name in {"R_AUTHOR_SUBREDDIT", "Y_COMMENTER_VIDEO"}:
                ranks = bipartite_rankings(scoped_edges)
                ranks.insert(0, "scope", scope)
                ranks.insert(0, "graph_id", name)
                bipartite_rank_frames.append(ranks)
    frame_rankings, frame_rank_status = _frame_personalized_rankings()
    outputs = {
        "structure_summary": STRUCTURE_ROOT / "structure_summary.csv",
        "community_stability": STRUCTURE_ROOT / "community_stability.csv",
        "partitions": STRUCTURE_ROOT / "partitions.parquet",
        "roles": STRUCTURE_ROOT / "roles.parquet",
        "pagerank_stability": STRUCTURE_ROOT / "pagerank_stability.csv",
        "bipartite_rankings": STRUCTURE_ROOT / "bipartite_rankings.csv",
        "frame_personalized_pagerank": STRUCTURE_ROOT / "frame_personalized_pagerank.csv",
    }
    pd.DataFrame(summaries).sort_values(["graph_id", "scope"]).to_csv(outputs["structure_summary"], index=False, lineterminator="\n")
    pd.DataFrame(stability_rows).sort_values(["graph_id", "scope", "resolution"]).to_csv(outputs["community_stability"], index=False, lineterminator="\n")
    pd.DataFrame(partition_rows).sort_values(["graph_id", "scope", "node_id"]).to_parquet(outputs["partitions"], compression="zstd", index=False)
    pd.concat(role_frames, ignore_index=True).sort_values(["graph_id", "scope", "node_id"]).to_parquet(outputs["roles"], compression="zstd", index=False)
    pd.DataFrame(rank_rows).sort_values(["graph_id", "scope", "comparison", "replicate"]).to_csv(outputs["pagerank_stability"], index=False, lineterminator="\n")
    pd.concat(bipartite_rank_frames, ignore_index=True).sort_values(["graph_id", "scope", "node_id"]).to_csv(outputs["bipartite_rankings"], index=False, lineterminator="\n")
    frame_rankings.sort_values(["graph_id", "scope", "node_id", "frame"]).to_csv(outputs["frame_personalized_pagerank"], index=False, lineterminator="\n")
    manifest = {
        "schema_version": "network-structure.v2",
        "created_at_utc": utc_now_iso(),
        "command": "python -m src.analysis.run networks structure --check",
        "network_manifest_sha256": sha256_file(NETWORK_MANIFEST),
        "network_source_checksums": network_manifest["source_checksums"],
        "resolution_grid": list(RESOLUTIONS),
        "resolution_tie_tolerance": 0.01,
        "fixed_seeds": list(SEEDS),
        "null_replicates": NULL_REPS,
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "betweenness": {
            "exact_node_limit": BETWEENNESS_EXACT_NODE_LIMIT,
            "stability_seeds": list(BETWEENNESS_STABILITY_SEEDS),
            "stability_threshold": BETWEENNESS_STABILITY_THRESHOLD,
            "weighting": "unweighted_shortest_paths",
        },
        "frame_personalized_pagerank": {"status": frame_rank_status, "function": "personalized_frame_pagerank", "profile_source": "adjudicated_development_frame_labels"},
        "bipartite_rankings": {"status": "computed", "methods": ["HITS", "BiRank"]},
        "community_cross_check": {"method": "louvain", "status": "computed_when_leiden_is_available"},
        "graphs": structure_ids,
        "outputs": {
            name: {"path": relative_path(path), "sha256": sha256_file(path), "rows": int(len(pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)))}
            for name, path in outputs.items()
        },
        "limitations": [
            "Leiden is the primary method when its pinned runtime is available; Louvain remains a declared cross-check rather than a substitute.",
            "Betweenness uses exact unweighted igraph scores when available; the NetworkX fallback uses up to one tenth of nodes with a minimum of 2,000 pivots and a two-seed top-decile stability check.",
            "Degree-preserving nulls are untyped for bipartite graphs and are diagnostics, not PRIMARY-6 nulls.",
            "No frame labels or confirmatory outcomes are joined at this stage.",
        ],
    }
    write_json(STRUCTURE_MANIFEST, manifest)
    return manifest


def check_structure() -> dict[str, Any]:
    check_networks()
    if not STRUCTURE_MANIFEST.exists():
        raise FileNotFoundError(f"structure manifest is missing: {STRUCTURE_MANIFEST}; run the explicit structure build first")
    manifest = read_json(STRUCTURE_MANIFEST)
    if manifest.get("schema_version") != "network-structure.v2":
        raise ValueError("structure manifest schema is stale; run the explicit structure build")
    if manifest.get("network_manifest_sha256") != sha256_file(NETWORK_MANIFEST):
        raise ValueError("network structure input manifest changed")
    summary_path = STRUCTURE_ROOT / "structure_summary.csv"
    summary = pd.read_csv(summary_path)
    if "community_cross_check_status" not in summary or manifest.get("resolution_tie_tolerance") != 0.01:
        raise ValueError("network structure safeguards are missing; run the explicit structure build")
    roles_path = STRUCTURE_ROOT / "roles.parquet"
    roles = pd.read_parquet(roles_path)
    required_role_columns = {
        "betweenness_method",
        "betweenness_pivots",
        "betweenness_top_decile_jaccard",
        "betweenness_stability_status",
        "betweenness_weighting",
    }
    if not required_role_columns <= set(roles.columns):
        raise ValueError("betweenness diagnostics are missing; run the explicit structure build")
    leiden_rows = summary[summary["community_method"].eq("leiden")]
    if not leiden_rows.empty and not leiden_rows["community_cross_check_status"].eq("computed").all():
        raise ValueError("Leiden structure rows are missing the Louvain cross-check")
    for record in manifest.get("outputs", {}).values():
        path = REPO_ROOT / record["path"]
        if not path.exists() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"network structure artifact changed: {record['path']}")
    return {"status": "valid", "artifact": relative_path(STRUCTURE_MANIFEST), "graphs": len(manifest.get("graphs", []))}
