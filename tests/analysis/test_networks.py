from __future__ import annotations

import pandas as pd
import pytest

import src.analysis.networks as networks
from src.analysis.networks import _actor_edges, _reverse_actor_edges, descendant_edge_indexes


def test_descendant_edges_follow_observed_parent_links():
    edges = pd.DataFrame(
        [
            {"source_doc_id": "child", "target_doc_id": "root"},
            {"source_doc_id": "grandchild", "target_doc_id": "child"},
            {"source_doc_id": "unrelated", "target_doc_id": "other"},
        ]
    )
    assert descendant_edge_indexes(edges, {"root"}) == {0, 1}


def test_actor_edges_aggregate_and_reverse_direction():
    edges = pd.DataFrame(
        [
            {
                "source_doc_id": "d1",
                "target_doc_id": "p1",
                "source_author_hash": "a",
                "target_author_hash": "b",
                "scope": "E2",
                "day": "2025-06-15",
            },
            {
                "source_doc_id": "d2",
                "target_doc_id": "p2",
                "source_author_hash": "a",
                "target_author_hash": "b",
                "scope": "E2",
                "day": "2025-06-16",
            },
        ]
    )
    attention = _actor_edges(edges, resolution="observed_parent")
    assert attention.loc[0, "source_id"] == "a"
    assert attention.loc[0, "target_id"] == "b"
    assert attention.loc[0, "weight"] == 2
    cascade = _reverse_actor_edges(attention)
    assert cascade.loc[0, "source_id"] == "b"
    assert cascade.loc[0, "target_id"] == "a"


def test_relevance_audit_requires_double_coding_and_returns_accepted_threads(tmp_path):
    path = tmp_path / "reddit_thread_relevance.csv"
    pd.DataFrame(
        {
            "thread_id": ["t1", "t2"],
            "coder_a_relevant": ["yes", "no"],
            "coder_b_relevant": ["yes", "no"],
            "adjudicated_relevant": ["yes", "no"],
        }
    ).to_csv(path, index=False)

    audit = networks.validate_reddit_relevance_audit(path)

    assert audit["status"] == "valid"
    assert audit["accepted_thread_count"] == 1


def test_empty_relevance_audit_is_not_valid(tmp_path):
    path = tmp_path / "reddit_thread_relevance.csv"
    pd.DataFrame(columns=networks.REDDIT_AUDIT_COLUMNS).to_csv(path, index=False)

    try:
        networks.validate_reddit_relevance_audit(path)
    except ValueError:
        pass
    else:
        raise AssertionError("empty audit must remain blocked")


def test_youtube_metadata_audit_validates_frame_and_stance_fields(tmp_path):
    path = tmp_path / "youtube_video_metadata.csv"
    pd.DataFrame(
        {
            "video_id": ["v1"],
            "coder_a_source_type": ["news"],
            "coder_b_source_type": ["news"],
            "adjudicated_source_type": ["news"],
            "coder_a_frame": ["privacy_surveillance"],
            "coder_b_frame": ["privacy_surveillance"],
            "adjudicated_frame": ["privacy_surveillance"],
            "coder_a_stance": ["support"],
            "coder_b_stance": ["support"],
            "adjudicated_stance": ["support"],
        }
    ).to_csv(path, index=False)

    audit = networks.validate_youtube_metadata_audit(path)
    assert audit["status"] == "valid"
    assert audit["accepted_video_ids"] == {"yt:video:v1"}
    assert audit["coder_agreement"]["source_type"] == 1.0


def test_youtube_metadata_audit_allows_screened_tech_explainer_type(tmp_path):
    path = tmp_path / "youtube_video_metadata.csv"
    row = {
        "video_id": "v1",
        "coder_a_source_type": "tech_explainer",
        "coder_b_source_type": "tech_explainer",
        "adjudicated_source_type": "tech_explainer",
        "coder_a_frame": "privacy_surveillance",
        "coder_b_frame": "privacy_surveillance",
        "adjudicated_frame": "privacy_surveillance",
        "coder_a_stance": "support",
        "coder_b_stance": "support",
        "adjudicated_stance": "support",
    }
    pd.DataFrame([row]).to_csv(path, index=False)
    assert networks.validate_youtube_metadata_audit(path)["status"] == "valid"


def test_youtube_scope_uses_video_assignment_not_comment_calendar():
    docs = pd.DataFrame(
        {
            "doc_id": ["v1", "c1"],
            "thing": ["video", "comment"],
            "root_doc_id": [None, "v1"],
            "yt_video_event_id": ["E2", None],
            "event_window": ["E2", "E4"],
        }
    )

    assigned = networks.youtube_video_event_assignments(docs)

    assert assigned.loc[assigned["doc_id"].eq("c1"), "video_event"].item() == "E2"


def test_scope_slices_never_mix_events():
    edges = pd.DataFrame({"scope": ["E2", "E3"], "source_id": ["a", "b"], "target_id": ["b", "c"]})

    slices = networks.scoped_edge_frames(edges)

    assert {scope: len(frame) for scope, frame in slices} == {"E2": 1, "E3": 1}


def test_network_check_revalidates_population_manifest_fingerprint(tmp_path, monkeypatch):
    manifest_path = tmp_path / "network_manifest.json"
    manifest_path.write_text(
        '{"schema_version": "networks.v2", "source_checksums": {"population_manifest": "old"}}\n',
        encoding="utf-8",
    )
    called = []
    monkeypatch.setattr(networks, "NETWORK_MANIFEST", manifest_path)
    monkeypatch.setattr(networks, "check_populations", lambda: called.append(True))
    monkeypatch.setattr(networks, "_input_checksums", lambda: {"population_manifest": "new"})

    with pytest.raises(ValueError, match="network source checksum changed"):
        networks.check_networks()

    assert called == [True]
