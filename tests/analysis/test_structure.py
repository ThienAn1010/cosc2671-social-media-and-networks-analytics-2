from __future__ import annotations

import networkx as nx
import pandas as pd

import src.analysis.structure as structure
from src.analysis.structure import _betweenness_scores, _canonical_partition, _choose_resolution, _degree_preserving_null_modularity, _participation_and_zscore, _partition_stability, _top_jaccard, bipartite_rankings, personalized_frame_pagerank


def test_partition_stability_is_one_for_identical_partitions():
    partition = {"a": 0, "b": 0, "c": 1}
    assert _partition_stability([partition, partition]) == (1.0, 1.0)


def test_canonical_partition_is_ordered_by_smallest_node():
    assert _canonical_partition({"b": 4, "a": 4, "c": 2}) == {"a": "c0000", "b": "c0000", "c": "c0001"}


def test_top_jaccard_is_zero_for_disjoint_sets():
    left = {node: float(node) for node in range(3)}
    right = {node: float(2 - node) for node in range(3)}
    assert _top_jaccard(left, right, k=1) == 0.0


def test_resolution_tie_prefers_lower_resolution_within_tolerance():
    candidates = pd.DataFrame({"resolution": [0.50, 0.75, 1.00], "mean_modularity": [0.40, 0.409, 0.415]})

    assert _choose_resolution(candidates) == 0.75


def test_within_module_zscore_ignores_external_edge_strength():
    graph = nx.Graph()
    graph.add_weighted_edges_from([("a", "b", 1), ("a", "x", 100), ("x", "y", 1)])

    _, zscore = _participation_and_zscore(graph, {"a": "m", "b": "m", "x": "n", "y": "n"})

    assert zscore == {"a": 0.0, "b": 0.0, "x": 0.0, "y": 0.0}


def test_weighted_null_restores_edge_weight_multiset_after_rewiring(monkeypatch):
    graph = nx.cycle_graph(8)
    nx.set_edge_attributes(graph, 5.0, "weight")
    partition = {node: "left" if node < 4 else "right" for node in graph}

    def remove_weights(null, **kwargs):
        for _, _, data in null.edges(data=True):
            data.clear()

    monkeypatch.setattr(structure, "NULL_REPS", 1)
    monkeypatch.setattr(structure.nx, "double_edge_swap", remove_weights)

    result = _degree_preserving_null_modularity(graph, partition)

    assert result["replicates"] == 1
    assert result["observed_minus_null"] == 0.0


def test_betweenness_uses_exact_scores_and_records_stability():
    scores, metadata = _betweenness_scores(nx.path_graph(["a", "b", "c", "d"]))

    assert metadata["method"] == "igraph_exact"
    assert metadata["pivots"] == 4
    assert metadata["top_decile_jaccard"] == 1.0
    assert metadata["stability_status"] == "stable"
    assert scores["b"] > scores["a"]


def test_structure_scopes_are_processed_as_separate_graphs():
    edges = pd.DataFrame({"scope": ["AU_LEGISLATION", "AU_IMPLEMENTATION"], "source_id": ["a", "b"], "target_id": ["b", "c"]})

    slices = structure.scoped_structure_frames(edges)

    assert {scope: len(frame) for scope, frame in slices} == {"AU_IMPLEMENTATION": 1, "AU_LEGISLATION": 1}


def test_personalized_pagerank_returns_one_vector_per_frame():
    graph = nx.path_graph(["a", "b"])
    profiles = pd.DataFrame({"node_id": ["a", "b"], "privacy": [1, 0], "safety": [0, 1]})

    result = personalized_frame_pagerank(graph, profiles, ["privacy", "safety"])

    assert set(result["frame"]) == {"privacy", "safety"}
    assert result.groupby("frame").size().to_dict() == {"privacy": 2, "safety": 2}


def test_bipartite_rankings_emit_hits_and_birank():
    edges = pd.DataFrame({"source_id": ["a", "b"], "target_id": ["video:v1", "video:v1"], "weight": [2, 1]})

    result = bipartite_rankings(edges)

    assert {"hits_hub", "hits_authority", "birank"} <= set(result.columns)
    assert result.loc[result["node_id"].eq("video:v1"), "node_type"].item() == "video"
