from collections import Counter

import pandas as pd
import pytest

import src.analysis.fallback_analysis as fallback


def test_topic_review_preserves_substantive_and_non_substantive_components():
    classifications = Counter(
        interpretation_class
        for platform_topics in fallback.TOPIC_NAMES.values()
        for _, _, interpretation_class in platform_topics.values()
    )

    assert classifications == {
        "substantive": 7,
        "generic_or_discourse_style": 8,
        "artifact_or_contamination": 3,
    }


def test_h2_restricts_both_stages_to_the_adjudicated_au_policy():
    rows = []
    for index in range(6):
        rows.append(
            {
                "_eligible": True,
                "platform": "reddit",
                "adjudicated_target_policy": "AU_SOCIAL_MINIMUM_AGE",
                "case_window_id": "AU_LEGISLATION",
                "cluster_id": f"legislation-{index}",
                "inclusion_probability": 1.0,
                "_weight": 1.0,
                "adjudicated_frame_labels": "",
            }
        )
    rows.extend(
        [
            {
                "_eligible": True,
                "platform": "reddit",
                "adjudicated_target_policy": "AU_SOCIAL_MINIMUM_AGE",
                "case_window_id": "AU_IMPLEMENTATION",
                "cluster_id": "implementation-au",
                "inclusion_probability": 1.0,
                "_weight": 1.0,
                "adjudicated_frame_labels": "",
            },
            {
                "_eligible": True,
                "platform": "reddit",
                "adjudicated_target_policy": "UK_OSA",
                "case_window_id": "AU_IMPLEMENTATION",
                "cluster_id": "implementation-uk",
                "inclusion_probability": 1.0,
                "_weight": 1.0,
                "adjudicated_frame_labels": "",
            },
        ]
    )

    _, details = fallback._h2_contrasts(pd.DataFrame(rows))

    assert details["status"] == "unavailable"
    assert details["cells"]["privacy_surveillance"]["earlier_n"] == 6
    assert details["cells"]["privacy_surveillance"]["later_n"] == 1
    assert "AU_SOCIAL_MINIMUM_AGE" in details["cells"]["privacy_surveillance"]["reason"]


def test_fallback_freshness_rejects_config_or_code_drift(monkeypatch):
    original_replicates = fallback.BOOTSTRAP_REPLICATES
    manifest = {
        "analysis_config": fallback._analysis_config(),
        "analysis_code_sha256": fallback._analysis_code_sha256(),
        "analysis_code_checksums": fallback._analysis_code_checksums(),
    }

    monkeypatch.setattr(fallback, "BOOTSTRAP_REPLICATES", fallback.BOOTSTRAP_REPLICATES + 1)
    with pytest.raises(AssertionError, match="configuration"):
        fallback._validate_analysis_freshness(manifest)

    monkeypatch.setattr(fallback, "BOOTSTRAP_REPLICATES", original_replicates)
    monkeypatch.setattr(fallback, "_analysis_code_sha256", lambda: "changed")
    with pytest.raises(AssertionError, match="analysis code"):
        fallback._validate_analysis_freshness(manifest)

    monkeypatch.setattr(fallback, "_analysis_code_sha256", lambda: manifest["analysis_code_sha256"])
    monkeypatch.setattr(fallback, "_analysis_code_checksums", lambda: {"measurement.py": "changed"})
    with pytest.raises(AssertionError, match="dependency code"):
        fallback._validate_analysis_freshness(manifest)


def test_f3_network_summary_excludes_youtube_graph_rows():
    summary = fallback._network_summaries(
        {
            "structure": pd.DataFrame(
                [
                    {
                        "community_status": "primary_leiden_stable",
                        "betweenness_stability_status": "stable",
                        "graph_id": "B_ACTOR_ATTENTION",
                        "scope": "E2",
                        "observed_minus_null_modularity": 0.8,
                        "nodes": 10,
                        "edges": 12,
                        "directed": True,
                        "weighted_edge_total": 15.0,
                        "communities": 3,
                        "community_modularity": 0.7,
                        "null_median_modularity": 0.1,
                        "community_median_ari": 0.9,
                        "betweenness_method": "exact",
                        "betweenness_top_decile_jaccard": 0.8,
                    }
                ]
            ),
            "roles": pd.DataFrame(
                [{"graph_id": "B_ACTOR_ATTENTION", "scope": "E2", "node_type": "author", "participation_coefficient": 0.2}]
            ),
            "pagerank": pd.DataFrame(
                [
                    {"graph_id": "B_ACTOR_ATTENTION", "scope": "E2", "status": "edge_bootstrap", "rank_correlation": 0.995, "top_k_jaccard": 0.75},
                    {"graph_id": "Y_ACTOR_ATTENTION_HANDLE", "scope": "E2", "status": "edge_bootstrap", "rank_correlation": 0.55, "top_k_jaccard": 0.25},
                    {"graph_id": "Y_COMMENTER_VIDEO", "scope": "E2", "status": "edge_bootstrap", "rank_correlation": 0.60, "top_k_jaccard": 0.30},
                ]
            ),
            "cascade": pd.DataFrame(
                [
                    {
                        "observed_parent_edges": True,
                        "validity_status": "candidate_network_ready",
                        "container_id": "E2",
                        "max_depth": 1,
                        "max_breadth": 1,
                        "structural_virality": 1.0,
                        "platform": "bluesky",
                        "scope": "E2",
                    }
                ]
            ),
            "statuses": {
                "B_ACTOR_ATTENTION": "candidate_network_ready",
                "Y_ACTOR_ATTENTION_HANDLE": "audited_ready",
                "Y_COMMENTER_VIDEO": "audited_ready",
            },
        }
    )

    graphs = summary["report_eligible_pagerank_edge_bootstrap"]["graphs"]

    assert graphs == ["B_ACTOR_ATTENTION"]
    assert not any(graph.startswith("Y_") for graph in graphs)
