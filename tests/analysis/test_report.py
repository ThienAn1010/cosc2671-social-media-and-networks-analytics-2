import json

import pytest
import pandas as pd

import src.analysis.report as report
from src.analysis.report import SOURCE_ARTIFACTS, _actions, _primary6_finding_card, _report_status


def test_report_freshness_tracks_confirmatory_hypotheses():
    assert {f"hypothesis_{name}" for name in ("h1", "h2", "h3", "h4")} <= set(SOURCE_ARTIFACTS)
    assert "primary6" in SOURCE_ARTIFACTS
    assert set(report.HYPOTHESIS_CONTRACTS) == {
        "hypothesis_h1", "hypothesis_h1_reddit", "hypothesis_h1_bluesky",
        "hypothesis_h2", "hypothesis_h2_privacy", "hypothesis_h2_circumvention",
        "hypothesis_h3", "hypothesis_h4",
    }
    assert {"descriptive_coverage", "structure_summary", "temporal_cascade_summary", "topic_prevalence"} <= set(SOURCE_ARTIFACTS)


def test_report_status_and_actions_follow_current_gate_state():
    statuses = {"H1": "blocked", "H2": "blocked", "H3": "blocked", "H4": "blocked"}
    assert _report_status(statuses) == "descriptive_bundle_ready_confirmatory_blocked"
    actions = _actions("descriptive_bundle_ready_confirmatory_blocked", statuses)
    assert len(actions) == 2
    assert "H1=blocked" in actions.iloc[0]["success_indicator"]


def test_confirmatory_actions_are_selected_from_passing_endpoint_and_robustness_artifacts(tmp_path, monkeypatch):
    primary_path = tmp_path / "primary6.json"
    robustness_manifest = tmp_path / "robustness_manifest.json"
    variant_results = tmp_path / "variant_results.csv"
    endpoints = ["H1-R", "H1-B", "H2-P", "H2-C", "H3", "H4"]
    primary_path.write_text(
        json.dumps(
            {
                "decision": {
                    "status": "complete",
                    "all_endpoints_pass": True,
                    "adjusted_p_values": {endpoint: 0.001 for endpoint in endpoints},
                    "effects": {endpoint: 0.10 + index / 100 for index, endpoint in enumerate(endpoints)},
                    "decisions": {endpoint: True for endpoint in endpoints},
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"hypothesis_id": "H1", "variant_id": "primary", "status": "available", "decision": True, "decision_stable": True},
            {"hypothesis_id": "H2", "variant_id": "privacy_primary", "status": "available", "decision": True, "decision_stable": True},
            {"hypothesis_id": "H2", "variant_id": "circumvention_primary", "status": "available", "decision": True, "decision_stable": True},
            {"hypothesis_id": "H3", "variant_id": "primary", "status": "available", "decision": True, "decision_stable": True},
            {"hypothesis_id": "H4", "variant_id": "primary", "status": "available", "decision": True, "decision_stable": True},
        ]
    ).to_csv(variant_results, index=False)
    robustness_manifest.write_text(
        json.dumps({"status": "complete", "outputs": {"variant_results": {"path": str(variant_results)}}}),
        encoding="utf-8",
    )
    monkeypatch.setitem(SOURCE_ARTIFACTS, "primary6", primary_path)
    monkeypatch.setitem(SOURCE_ARTIFACTS, "robustness", robustness_manifest)

    actions = _actions("confirmatory_bundle_ready", {name: "complete" for name in ("H1", "H2", "H3", "H4")})
    candidates = actions[actions["priority_candidate"]]

    assert 1 <= len(candidates) <= 2
    assert {"actor", "change_first", "mechanism", "why_this_first", "success_indicator", "failure_indicator", "guardrail"} <= set(candidates.columns)


def test_complete_mixed_null_endpoints_are_reportable(tmp_path, monkeypatch):
    primary_path = tmp_path / "primary6.json"
    robustness_manifest = tmp_path / "robustness_manifest.json"
    variant_results = tmp_path / "variant_results.csv"
    endpoints = ["H1-R", "H1-B", "H2-P", "H2-C", "H3", "H4"]
    primary_path.write_text(
        json.dumps(
            {
                "decision": {
                    "status": "complete",
                    "all_endpoints_pass": False,
                    "missing_endpoints": [],
                    "adjusted_p_values": {endpoint: 0.5 for endpoint in endpoints},
                    "effects": {endpoint: 0.0 for endpoint in endpoints},
                    "decisions": {endpoint: False for endpoint in endpoints},
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"hypothesis_id": "H1", "variant_id": "primary", "status": "available", "decision": False, "decision_stable": True},
            {"hypothesis_id": "H2", "variant_id": "privacy_primary", "status": "available", "decision": False, "decision_stable": True},
            {"hypothesis_id": "H2", "variant_id": "circumvention_primary", "status": "available", "decision": False, "decision_stable": True},
            {"hypothesis_id": "H3", "variant_id": "primary", "status": "available", "decision": False, "decision_stable": True},
            {"hypothesis_id": "H4", "variant_id": "primary", "status": "available", "decision": False, "decision_stable": True},
        ]
    ).to_csv(variant_results, index=False)
    robustness_manifest.write_text(json.dumps({"status": "complete", "outputs": {"variant_results": {"path": str(variant_results)}}}), encoding="utf-8")
    monkeypatch.setitem(SOURCE_ARTIFACTS, "primary6", primary_path)
    monkeypatch.setitem(SOURCE_ARTIFACTS, "robustness", robustness_manifest)

    assert _report_status({name: "complete" for name in ("H1", "H2", "H3", "H4")}) == "confirmatory_bundle_ready"


def test_report_build_keeps_tracked_manual_report_untouched(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    report_root = repo / "data" / "analysis" / "report"
    monkeypatch.setattr(report, "REPO_ROOT", repo)
    monkeypatch.setattr(report, "REPORT_ROOT", report_root)
    monkeypatch.setattr(report, "REPORT_MANIFEST", report_root / "report_manifest.json")
    monkeypatch.setattr(report, "check_topics", lambda: None)
    monkeypatch.setattr(report, "check_robustness", lambda: None)
    monkeypatch.setattr(report, "_check_hypothesis_contracts", lambda: None)
    monkeypatch.setattr(report, "_check_human_sample_artifacts", lambda: None)
    monkeypatch.setattr(report, "_coverage", lambda: pd.DataFrame([{"platform": "reddit", "scope": "E2", "rows": 1, "strict_english_rows": 1, "inclusive_english_rows": 1}]))
    monkeypatch.setattr(report, "_structure", lambda: pd.DataFrame([{"graph_id": "R_ACTOR_ATTENTION", "scope": "E2", "nodes": 1, "edges": 1, "community_method": "leiden", "community_status": "stable"}]))
    monkeypatch.setattr(report, "_topics", lambda: pd.DataFrame([{"platform": "reddit", "topic_count": 6, "mean_top_term_jaccard": 1.0}]))
    monkeypatch.setattr(report, "_hypothesis_statuses", lambda: {name: "blocked" for name in ("H1", "H2", "H3", "H4")})
    monkeypatch.setattr(report, "_report_status", lambda statuses: "descriptive_bundle_ready_confirmatory_blocked")
    monkeypatch.setattr(report, "_measurement_summary", lambda: [])
    monkeypatch.setattr(report, "_build_figures", lambda *args: {})
    monkeypatch.setattr(report, "_report_markdown", lambda *args: "generated\n")

    report.build_report_artifacts()

    assert not (repo / "reports" / "age_gate_paradox_report.md").exists()
    assert "tracked_report" not in report.read_json(report_root / "report_manifest.json").get("outputs", {})


def test_primary6_finding_card_describes_missing_effects(monkeypatch, tmp_path):
    monkeypatch.setitem(SOURCE_ARTIFACTS, "primary6", tmp_path / "missing-primary6.json")
    assert _primary6_finding_card()["effect_interval"] == "no endpoint effect or adjusted p-value is available"


def test_report_rejects_stale_variant_hypothesis_contract(monkeypatch):
    def reject_stale_variant(hypothesis_id, platform=None, outcome=None):
        if hypothesis_id == "H1" and platform == "reddit":
            raise ValueError("stale h1_reddit.json")
        source_name = next(
            name
            for name, contract in report.HYPOTHESIS_CONTRACTS.items()
            if contract == (hypothesis_id, platform, outcome)
        )
        return {"artifact": str(report.SOURCE_ARTIFACTS[source_name].relative_to(report.REPO_ROOT))}

    monkeypatch.setattr(report, "check_hypothesis", reject_stale_variant)
    with pytest.raises(ValueError, match="stale h1_reddit.json"):
        report._check_hypothesis_contracts()
