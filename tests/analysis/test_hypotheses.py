from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.hypotheses import (
    author_balanced_difference,
    holm_adjust,
    js_divergence,
    normalized_frame_entropy,
    frozen_design_power_simulation,
    power_mde_diagnostic,
)
import src.analysis.hypotheses as hypotheses


def test_js_divergence_is_zero_for_equal_profiles():
    assert js_divergence([0.5, 0.5], [0.5, 0.5]) == 0.0


def test_author_balanced_difference_is_not_document_weighted():
    frame = pd.DataFrame(
        {
            "author": ["a", "a", "b", "c"],
            "case": ["later", "later", "later", "earlier"],
            "label": [1, 0, 1, 0],
        }
    )
    assert author_balanced_difference(frame, "author", "case", "label", "later", "earlier") == 0.75


def test_entropy_is_normalized_to_one_for_uniform_profile():
    assert np.isclose(normalized_frame_entropy(np.ones(5)), 1.0)


def test_holm_adjustment_preserves_monotonicity():
    adjusted = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.8})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert adjusted["a"] == 0.03


def test_frozen_design_power_simulation_is_deterministic_and_flags_underpower():
    design = dict(
        cluster_sizes=[2, 2],
        topology={"nodes": 2, "edges": 1, "components": 2},
        source_type_counts={"single_source": 4},
        minimum_effect=0.05,
        replicates=200,
    )

    first = frozen_design_power_simulation(**design)
    second = frozen_design_power_simulation(**design)

    assert first == second
    assert first["mde"] is not None
    assert first["status"] == "inconclusive_underpowered"
    assert first["target_power"] == 0.80


def test_endpoint_power_simulation_uses_named_endpoint_designs():
    design = dict(
        cluster_sizes=[4, 5, 6, 7, 8, 9, 10, 11],
        topology={"nodes": 8, "edges": 12, "components": 2},
        source_type_counts={"news": 4, "commentary": 4},
        minimum_effect=0.05,
        replicates=100,
    )

    results = [frozen_design_power_simulation(**design, endpoint_id=endpoint) for endpoint in hypotheses.PRIMARY_ENDPOINTS]

    assert all(result["method"] == "endpoint_specific_frozen_design_monte_carlo" for result in results)
    assert [result["endpoint_id"] for result in results] == list(hypotheses.PRIMARY_ENDPOINTS)
    assert results[0]["design"] != results[2]["design"]
    assert "case-stratified profile shuffles" in results[0]["estimator_contract"]
    assert "matched controls" in results[4]["estimator_contract"]
    assert "event-adjusted news coefficient" in results[-1]["estimator_contract"]


def test_h1_power_simulation_uses_crossed_communities_and_empirical_null():
    design = dict(
        cluster_sizes=[4, 5, 6, 7, 8, 9, 10, 11],
        topology={"nodes": 8, "edges": 12, "components": 2},
        source_type_counts={"news": 4, "commentary": 4},
        replicates=100,
    )

    weak = frozen_design_power_simulation(**design, endpoint_id="H1-R", minimum_effect=0.05)
    strong = frozen_design_power_simulation(**design, endpoint_id="H1-R", minimum_effect=10.0)

    assert weak["critical_value_method"] == "empirical_case_degree_activity_permutation_null"
    assert weak["power_at_minimum_effect"] > 0.0
    assert strong["power_at_minimum_effect"] >= 0.80


def test_h1_author_vector_permutation_returns_declared_primary_statistics():
    frame = pd.DataFrame(
        [
            {"case": "E2", "author": "a1", "community_id": "c1", "degree": 1, "activity": 3, "documents": 3, "f1": 1, "f2": 0},
            {"case": "E2", "author": "a2", "community_id": "c1", "degree": 1, "activity": 3, "documents": 3, "f1": 1, "f2": 0},
            {"case": "E2", "author": "b1", "community_id": "c2", "degree": 1, "activity": 3, "documents": 3, "f1": 0, "f2": 1},
            {"case": "E2", "author": "b2", "community_id": "c2", "degree": 1, "activity": 3, "documents": 3, "f1": 0, "f2": 1},
        ]
    )

    result = hypotheses.author_vector_permutation_test(frame, ["f1", "f2"], permutations=40)

    assert result["permutations"] == 40
    assert {"observed", "null_median", "excess", "p_value"} <= set(result)
    assert all(key.startswith("E2|") for key in result["stratum_counts"])


def test_sparse_strata_terminates_when_cases_cannot_merge():
    frame = pd.DataFrame({"case": ["E2", "E3"], "degree": [1, 2], "activity": [1, 2]})

    result = hypotheses._sparse_strata(frame, "degree", "activity", "case")

    assert len(result) == len(frame)
    assert result.tolist() == ["E2|merged|merged", "E3|merged|merged"]


def test_bluesky_frame_documents_derives_language_uncertainty_from_frozen_schema(monkeypatch):
    calls = []

    def fake_read_parquet(path, columns=None):
        calls.append((str(path), columns))
        if str(path).endswith("posts.parquet"):
            return pd.DataFrame(
                {
                    "doc_id": ["d1", "d2"],
                    "root_doc_id": ["d1", "d1"],
                    "event_window": ["E2", "E2"],
                    "is_english": [True, True],
                    "lang_detect_prob": [0.95, 0.70],
                    "n_chars": [40, 40],
                    "is_meme": [False, False],
                    "is_repeat_burst": [False, False],
                    "author_hash": ["a1", "a2"],
                    "text_raw": ["first post", "second post"],
                }
            )
        if str(path).endswith("authors.parquet"):
            return pd.DataFrame({"author_hash": ["a1", "a2"], "account_class": ["ordinary", "ordinary"]})
        return pd.DataFrame({"source_doc_id": ["d1"], "target_doc_id": ["d2"], "scope": ["E2"]})

    monkeypatch.setattr(hypotheses.pd, "read_parquet", fake_read_parquet)

    result = hypotheses._frame_documents("bluesky")

    assert result["language_population"].tolist() == ["strict_english", "inclusive_uncertain_bound"]
    assert "is_language_uncertain" not in calls[0][1]
    assert {"lang_detect_prob", "n_chars"} <= set(calls[0][1])


def test_hypothesis_structure_gate_requires_stable_partition(tmp_path, monkeypatch):
    structure_root = tmp_path / "networks" / "structure"
    structure_root.mkdir(parents=True)
    (structure_root / "structure_summary.csv").write_text(
        "graph_id,community_status\nR_ACTOR_ATTENTION,primary_leiden_stable\n",
        encoding="utf-8",
    )
    pd.DataFrame({"graph_id": ["R_ACTOR_ATTENTION"], "node_type": ["author"], "betweenness_stability_status": ["stable"]}).to_parquet(structure_root / "roles.parquet")
    structure_manifest = tmp_path / "structure_manifest.json"
    structure_manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(hypotheses, "STRUCTURE_MANIFEST", structure_manifest)
    monkeypatch.setattr(hypotheses, "ANALYSIS_ROOT", tmp_path)
    monkeypatch.setattr(hypotheses, "check_structure", lambda: {"status": "valid"})

    assert hypotheses._stable_structure_available({"R_ACTOR_ATTENTION"}) is True
    pd.DataFrame({"graph_id": ["R_ACTOR_ATTENTION"], "node_type": ["author"], "betweenness_stability_status": ["unstable"]}).to_parquet(structure_root / "roles.parquet")
    assert hypotheses._stable_structure_available({"R_ACTOR_ATTENTION"}) is False
    (structure_root / "structure_summary.csv").write_text(
        "graph_id,community_status\nR_ACTOR_ATTENTION,inconclusive_no_stable_resolution\n",
        encoding="utf-8",
    )
    assert hypotheses._stable_structure_available({"R_ACTOR_ATTENTION"}) is False


def test_blocked_hypothesis_build_refreshes_primary6(tmp_path, monkeypatch):
    monkeypatch.setattr(hypotheses, "HYPOTHESIS_ROOT", tmp_path)
    monkeypatch.setattr(hypotheses, "PRIMARY6_ARTIFACT", tmp_path / "primary6.json")
    monkeypatch.setattr(hypotheses, "_artifact", lambda *args, **kwargs: {"status": "blocked", "missing_prerequisites": ["labels"]})

    result = hypotheses.build_hypothesis_artifact("H1", platform="reddit")

    assert result["status"] == "blocked"
    assert (tmp_path / "h1_reddit.json").exists()
    assert (tmp_path / "primary6.json").exists()


def test_h2_thread_cluster_bootstrap_returns_interval():
    frame = pd.DataFrame(
        {
            "case": ["earlier", "earlier", "later", "later"],
            "author": ["a", "b", "a", "c"],
            "thread": ["t1", "t2", "t3", "t4"],
            "outcome": [0, 0, 1, 1],
        }
    )

    result = hypotheses.thread_cluster_bootstrap_difference(frame, "outcome", "later", "earlier", replicates=40)

    assert result["replicates"] == 40
    assert result["lower_95"] <= result["upper_95"]


def test_h2_bootstrap_preserves_repeated_cluster_occurrences(monkeypatch):
    frame = pd.DataFrame(
        {
            "case": ["later"] * 4 + ["earlier"] * 2,
            "author": ["a", "b", "a", "c", "d", "e"],
            "thread": ["t1", "t1", "t2", "t3", "e1", "e2"],
            "outcome": [1, 0, 0, 0, 0, 0],
        }
    )

    class FixedDraws:
        def __init__(self):
            self.draws = [np.array(["t1", "t1", "t2"]), np.array(["e1", "e2"])]

        def choice(self, clusters, size, replace):
            return self.draws.pop(0)

    monkeypatch.setattr(hypotheses.np.random, "default_rng", lambda seed: FixedDraws())

    result = hypotheses.thread_cluster_bootstrap_difference(frame, "outcome", "later", "earlier", replicates=1)

    assert result["lower_95"] == pytest.approx(0.4)
    assert result["p_value"] == pytest.approx(1.0)


def test_h4_video_alignment_uses_channel_clustered_resampling():
    rows = []
    for event in ("E2", "E3"):
        for source, channel, metadata, audience in (
            ("news", f"{event}-n1", [1, 0], [1, 0]),
            ("news", f"{event}-n2", [1, 0], [0, 1]),
            ("commentary", f"{event}-c1", [0, 1], [0, 1]),
            ("commentary", f"{event}-c2", [0, 1], [1, 0]),
        ):
            rows.append(
                {
                    "video_id": f"{channel}-video",
                    "event": event,
                    "channel_id": channel,
                    "source_type": source,
                    "m1": metadata[0],
                    "m2": metadata[1],
                    "a1": audience[0],
                    "a2": audience[1],
                }
            )

    result = hypotheses.video_alignment_contrast(pd.DataFrame(rows), ["m1", "m2"], ["a1", "a2"], permutations=40)

    assert result["permutations"] == 40
    assert result["lower_95"] <= result["upper_95"]


def test_primary6_holm_decision_requires_all_named_endpoints():
    result = hypotheses.primary6_decision(
        {endpoint: (0.0001 if endpoint == "H4" else 0.5) for endpoint in hypotheses.PRIMARY_ENDPOINTS},
        {endpoint: 0.10 for endpoint in hypotheses.PRIMARY_ENDPOINTS},
    )

    assert set(result["adjusted_p_values"]) == set(hypotheses.PRIMARY_ENDPOINTS)
    assert result["decisions"]["H4"] is True


def test_confirmatory_result_requires_power_and_estimate_status():
    result = {"p_value": 0.01, "status": "inconclusive_underpowered", "power_status": "inconclusive_underpowered"}

    assert hypotheses._is_confirmatory_result(result) is False
    result.update(status="estimated", power_status="passed")
    assert hypotheses._is_confirmatory_result(result) is True


def test_two_sided_power_includes_both_rejection_tails():
    result = power_mde_diagnostic(100, 1.0, 0.5, alternative="two-sided")

    assert 0 < result["power_at_minimum_effect"] < 1


def test_predicted_frame_documents_apply_frozen_platform_thresholds(monkeypatch):
    monkeypatch.setattr(hypotheses, "_frame_documents", lambda platform: pd.DataFrame({"text": ["one"]}))

    class Model:
        frame_thresholds = np.asarray([0.30, 0.40, 0.50, 0.60, 0.70])

        def predict_proba(self, texts):
            return np.asarray([[0.35, 0.45, 0.55, 0.65, 0.75]])

    result = hypotheses._predicted_frame_documents("youtube", {"youtube": Model()})

    assert result[hypotheses.FRAME_LABELS].iloc[0].tolist() == [1, 1, 1, 1, 1]


def test_profile_estimators_reject_nonfinite_inputs():
    with pytest.raises(ValueError):
        js_divergence([1.0, np.nan], [0.5, 0.5])


def test_weighted_human_frame_estimate_keeps_blank_rows_as_negative_examples():
    frame = pd.DataFrame(
        {
            "platform": ["reddit", "reddit", "reddit", "reddit", "bluesky"],
            "case_window_id": ["AU_LEGISLATION", "AU_LEGISLATION", "AU_IMPLEMENTATION", "AU_IMPLEMENTATION", "E2"],
            "author_key": ["a", "b", "a", "d", "c"],
            "cluster_id": ["t1", "t2", "t3", "t4", "b1"],
            "adjudicated_relevance": ["relevant", "relevant", "relevant", "relevant", "relevant"],
            "adjudicated_language": ["english", "english", "english", "english", "english"],
            "adjudicated_frame_labels": ["privacy_surveillance", "", "privacy_surveillance", "", "child_safety"],
            "inclusion_probability": [0.5, 1.0, 1.0, 1.0, 1.0],
        }
    )

    result = hypotheses.weighted_human_sample_frame_estimate(frame, platform="reddit")

    legislation = next(row for row in result["rows"] if row["case_window_id"] == "AU_LEGISLATION")
    assert result["status"] == "available"
    assert result["interval_method"] == "inverse-inclusion-probability weighted multiway cluster bootstrap"
    assert legislation["rows"] == 2
    assert legislation["cluster_dimensions"] == ["cluster_id"]
    assert legislation["frame_prevalence"]["privacy_surveillance"]["estimate"] == pytest.approx(2 / 3)
    assert legislation["effective_sample_size"] == pytest.approx(9 / 5)


def test_human_fallback_is_available_from_adjudicated_reserve_sample(tmp_path, monkeypatch):
    assessment = tmp_path / "reserve_assessment.json"
    assessment.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(hypotheses, "RESERVE_ASSESSMENT", assessment)
    monkeypatch.setattr(hypotheses, "check_reserve_assessment", lambda: {"status": "fallback-human-sample"})
    frame = pd.DataFrame(
        {
            "platform": ["reddit", "reddit", "reddit", "reddit"],
            "case_window_id": ["AU_LEGISLATION", "AU_LEGISLATION", "AU_IMPLEMENTATION", "AU_IMPLEMENTATION"],
            "author_key": ["a", "b", "a", "b"],
            "cluster_id": ["t1", "t2", "t3", "t4"],
            "adjudicated_relevance": ["relevant", "relevant", "relevant", "relevant"],
            "adjudicated_language": ["english", "english", "english", "english"],
            "annotation_status": ["adjudicated", "adjudicated", "adjudicated", "adjudicated"],
            "adjudicated_frame_labels": ["", "", "privacy_surveillance", "privacy_surveillance"],
            "inclusion_probability": [1.0, 1.0, 1.0, 1.0],
        }
    )
    monkeypatch.setattr(hypotheses, "load_label_packet", lambda split: frame)

    result = hypotheses._human_sample_fallback("H2", "reddit")

    assert result["status"] == "available"
    assert result["source_split"] == "reserve"
    assert result["case_contrasts"]["privacy_surveillance"]["estimate"] == 1.0


def test_gate_can_return_narrow_human_fallback_without_automated_predictions(tmp_path, monkeypatch):
    assessment = tmp_path / "reserve_assessment.json"
    assessment.write_text(json.dumps({"metrics": {"frames": {"by_platform": {"reddit": {"gates": {"passed": False}, "unseen_author": {"gates": {"passed": False}}}}}}}), encoding="utf-8")
    monkeypatch.setattr(hypotheses, "RESERVE_ASSESSMENT", assessment)
    monkeypatch.setattr(hypotheses, "check_reserve_assessment", lambda: {"status": "fallback-human-sample"})
    frame = pd.DataFrame(
        {
            "platform": ["reddit", "reddit", "reddit", "reddit"],
            "case_window_id": ["AU_LEGISLATION", "AU_LEGISLATION", "AU_IMPLEMENTATION", "AU_IMPLEMENTATION"],
            "author_key": ["a", "b", "a", "b"],
            "cluster_id": ["t1", "t2", "t3", "t4"],
            "adjudicated_relevance": ["relevant", "relevant", "relevant", "relevant"],
            "adjudicated_language": ["english", "english", "english", "english"],
            "annotation_status": ["adjudicated", "adjudicated", "adjudicated", "adjudicated"],
            "adjudicated_frame_labels": ["", "", "privacy_surveillance", "privacy_surveillance"],
            "inclusion_probability": [1.0, 1.0, 1.0, 1.0],
        }
    )
    monkeypatch.setattr(hypotheses, "load_label_packet", lambda split: frame)
    monkeypatch.setattr(hypotheses, "validate_reddit_relevance_audit", lambda: {"status": "valid"})
    monkeypatch.setattr(hypotheses, "_graph_status", lambda name: "audited_ready")
    monkeypatch.setattr(hypotheses, "_stable_structure_available", lambda names: True)

    assert hypotheses._gate_status("H1", "reddit") == ("fallback-human-sample", [])


def test_invalid_reserve_integrity_cannot_feed_human_fallback(tmp_path, monkeypatch):
    assessment = tmp_path / "reserve_assessment.json"
    assessment.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(hypotheses, "RESERVE_ASSESSMENT", assessment)
    monkeypatch.setattr(hypotheses, "check_reserve_assessment", lambda: (_ for _ in ()).throw(ValueError("invalid reserve")))

    def unexpected_load(split):
        raise AssertionError(f"must not load {split} labels")

    monkeypatch.setattr(hypotheses, "load_label_packet", unexpected_load)

    result = hypotheses._human_sample_fallback("H2", "reddit")

    assert result["status"] == "human-labels-required"
    assert "invalid reserve" in result["reason"]


def test_gate_does_not_enter_fallback_after_invalid_reserve(tmp_path, monkeypatch):
    assessment = tmp_path / "reserve_assessment.json"
    assessment.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(hypotheses, "RESERVE_ASSESSMENT", assessment)
    monkeypatch.setattr(hypotheses, "check_reserve_assessment", lambda: (_ for _ in ()).throw(ValueError("invalid reserve")))
    monkeypatch.setattr(hypotheses, "_human_sample_fallback", lambda *args: (_ for _ in ()).throw(AssertionError("fallback must not run")))
    monkeypatch.setattr(hypotheses, "validate_reddit_relevance_audit", lambda: {"status": "valid"})
    monkeypatch.setattr(hypotheses, "_graph_status", lambda name: "audited_ready")
    monkeypatch.setattr(hypotheses, "_stable_structure_available", lambda names: True)

    status, missing = hypotheses._gate_status("H1", "reddit")

    assert status == "blocked"
    assert "measurement_validity" in missing
