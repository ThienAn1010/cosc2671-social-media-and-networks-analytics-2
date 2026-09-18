from __future__ import annotations

import pandas as pd
import numpy as np

import src.analysis.temporal as temporal
from src.analysis.temporal import _cascade_summary, _shuffle_labels_within_scope, _structural_virality, _community_lifecycle, first_sustained_uptake, frame_transition_matrix, transition_diagnostics, uptake_diagnostics


def test_cascade_summary_uses_parent_to_child_orientation():
    edges = pd.DataFrame(
        [
            {
                "source_id": "root",
                "target_id": "child",
                "scope": "E2",
                "container_id": "thread",
                "day": "2025-06-15",
            },
            {
                "source_id": "child",
                "target_id": "grandchild",
                "scope": "E2",
                "container_id": "thread",
                "day": "2025-06-16",
            },
        ]
    )
    summary = _cascade_summary("test", edges, "candidate")
    assert summary.loc[0, "max_depth"] == 2
    assert summary.loc[0, "duration_days"] == 1


def test_cascade_summary_keeps_root_only_documents_and_root_time():
    roots = pd.DataFrame({"scope": ["E2"], "container_id": ["thread"], "root_id": ["root"], "day": ["2025-06-14"]})
    summary = _cascade_summary("test", pd.DataFrame(), "candidate", roots)
    assert summary.loc[0, "documents"] == 1
    assert summary.loc[0, "root_documents"] == 1
    assert summary.loc[0, "max_depth"] == 0
    assert summary.loc[0, "duration_days"] == 0


def test_structural_virality_is_positive_for_a_path():
    import networkx as nx

    graph = nx.path_graph(3)
    assert _structural_virality(graph) == 4 / 3


def test_frame_transitions_follow_observed_parent_child_edges():
    edges = pd.DataFrame({"source_doc_id": ["parent"], "target_doc_id": ["child"]})
    labels = pd.DataFrame({"doc_id": ["parent", "child"], "frame_labels": ["privacy_surveillance", "child_safety|governance_platform_responsibility"]})

    result = frame_transition_matrix(edges, labels)

    assert result["transitions"].sum() == 2
    assert set(result["target_frame"]) == {"child_safety", "governance_platform_responsibility"}


def test_frame_transition_row_share_is_conditional_on_source_frame():
    edges = pd.DataFrame(
        {
            "source_doc_id": ["p1", "p2", "q1", "r1"],
            "target_doc_id": ["c1", "c2", "q2", "r2"],
        }
    )
    labels = pd.DataFrame(
        {
            "doc_id": ["p1", "p2", "q1", "r1", "c1", "c2", "q2", "r2"],
            "frame_labels": ["privacy_surveillance", "privacy_surveillance", "child_safety", "governance_platform_responsibility", "child_safety", "privacy_surveillance", "child_safety", "governance_platform_responsibility"],
        }
    )

    result = temporal.frame_transition_matrix(edges, labels)

    privacy_to_child = result[(result["source_frame"] == "privacy_surveillance") & (result["target_frame"] == "child_safety")].iloc[0]
    assert privacy_to_child["row_share"] == 0.5
    assert privacy_to_child["share_of_all_transitions"] == 0.25
    assert "share_of_all_transitions" in result


def test_sustained_uptake_requires_consecutive_windows():
    dates = pd.to_datetime(["2025-01-01"] * 3 + ["2025-01-08"] * 3)
    frame = pd.DataFrame({"date": dates, "community_id": ["c1"] * 6, "frame_labels": ["privacy_surveillance"] * 6})

    result = first_sustained_uptake(frame, window_days=7, consecutive_windows=2, minimum_documents=3)

    assert result.loc[0, "first_sustained_date"] == "2025-01-01"


def test_transition_diagnostics_persist_cluster_interval_and_label_null():
    edges = pd.DataFrame(
        {
            "source_doc_id": ["p1", "p2"],
            "target_doc_id": ["c1", "c2"],
            "container_id": ["thread-1", "thread-2"],
            "scope": ["E2", "E3"],
        }
    )
    labels = pd.DataFrame(
        {
            "doc_id": ["p1", "p2", "c1", "c2"],
            "frame_labels": ["privacy_surveillance", "privacy_surveillance", "child_safety", "child_safety"],
        }
    )

    result = transition_diagnostics(edges, labels, replicates=20)

    assert result.loc[0, "status"] == "available"
    assert result.loc[0, "cluster_bootstrap_lower_95"] <= result.loc[0, "cluster_bootstrap_upper_95"]
    assert "shuffled_label_null_median" in result


def test_uptake_diagnostics_persist_cluster_interval_and_shuffled_null():
    dates = pd.to_datetime(["2025-01-01"] * 3 + ["2025-01-08"] * 3)
    frame = pd.DataFrame(
        {
            "date": dates,
            "scope": ["E2"] * 6,
            "community_id": ["c1"] * 6,
            "cluster_id": ["t1", "t1", "t2", "t2", "t3", "t3"],
            "frame_labels": ["privacy_surveillance"] * 6,
        }
    )

    result = uptake_diagnostics(frame, replicates=20)

    assert result.loc[0, "status"] == "available"
    assert result.loc[0, "bootstrap_median"] == "2025-01-01"
    assert result.loc[0, "null_replicates"] == 20


def test_temporal_null_shuffles_labels_only_within_authoritative_scope():
    labels = pd.DataFrame(
        {
            "doc_id": ["e2-a", "e2-b", "e3-a", "e3-b"],
            "scope": ["E2", "E2", "E3", "E3"],
            "frame_labels": ["privacy_surveillance", "child_safety", "privacy_surveillance", "governance_platform_responsibility"],
        }
    )

    shuffled = _shuffle_labels_within_scope(labels, np.random.default_rng(20260915))

    for scope in ("E2", "E3"):
        assert sorted(shuffled.loc[shuffled["scope"].eq(scope), "frame_labels"]) == sorted(labels.loc[labels["scope"].eq(scope), "frame_labels"])


def test_uptake_population_frame_uses_fresh_partitions_without_reading_stale_artifact(monkeypatch):
    calls = []

    def fake_read_parquet(path, columns=None):
        calls.append(str(path))
        if str(path).endswith("posts.parquet"):
            return pd.DataFrame(
                {
                    "doc_id": ["d1"],
                    "event_window": ["E2"],
                    "day": ["2025-01-01"],
                    "author_hash": ["author-1"],
                    "root_doc_id": ["thread-1"],
                }
            )
        raise AssertionError(f"unexpected parquet read: {path}")

    monkeypatch.setattr(temporal.pd, "read_parquet", fake_read_parquet)
    labels = pd.DataFrame(
        {
            "platform": ["bluesky"],
            "doc_id": ["d1"],
            "adjudicated_frame_labels": ["privacy_surveillance"],
        }
    )
    fresh_partitions = pd.DataFrame(
        {
            "platform": ["bluesky"],
            "scope": ["E2"],
            "period": ["2025-01"],
            "node_id": ["author-1"],
            "community_id": ["c0000"],
        }
    )

    result, reason = temporal._uptake_population_frame("bluesky", labels, fresh_partitions)

    assert reason == ""
    assert result.loc[0, "community_id"] == "c0000"
    assert not any(path.endswith("temporal_partitions.parquet") for path in calls)


def test_community_lifecycle_counts_split_merge_and_death():
    lifecycle = _community_lifecycle({"old-a": {"a", "b"}, "old-b": {"c"}}, {"new-a": {"a"}, "new-b": {"b", "c"}}, threshold=0.2)

    assert lifecycle == {"births": 0, "deaths": 0, "splits": 1, "merges": 1}
