"""Assemble the traceable frozen-data report bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.artifacts import ANALYSIS_ROOT, REPO_ROOT, read_json, relative_path, sha256_file, utc_now_iso, write_json
from src.analysis.h1h3_human_sample import validate_h1h3_result
from src.analysis.h4_human_sample import validate_h4_result
from src.analysis.hypotheses import check_hypothesis
from src.analysis.measurement import (
    EVALUATION_REPORT,
    MODEL_RECEIPTS,
    RESERVE_ACTIVATION,
    RESERVE_ASSESSMENT,
    SENTIMENT_SELECTION,
    STANCE_FRAME_SELECTION,
    VALIDATION_ROOT,
)
from src.analysis.robustness import PRIMARY_VARIANT_IDS, check_robustness
from src.analysis.topics import check_topics

REPORT_ROOT = ANALYSIS_ROOT / "report"
REPORT_MANIFEST = REPORT_ROOT / "report_manifest.json"
REPORT_SCHEMA_VERSION = "report.v3"
FIGURE_ROOT = REPORT_ROOT / "figures"
SOURCE_ARTIFACTS = {
    "scope": ANALYSIS_ROOT / "scope" / "scope_manifest.json",
    "populations": ANALYSIS_ROOT / "populations" / "population_manifest.json",
    "codebook": ANALYSIS_ROOT / "codebook" / "codebook_v2.json",
    "validation": VALIDATION_ROOT / "validation_manifest.json",
    "descriptive": ANALYSIS_ROOT / "descriptive" / "descriptive_manifest.json",
    "descriptive_coverage": ANALYSIS_ROOT / "descriptive" / "coverage.csv",
    "descriptive_concentration": ANALYSIS_ROOT / "descriptive" / "concentration.csv",
    "descriptive_timeline": ANALYSIS_ROOT / "descriptive" / "timeline.csv",
    "networks": ANALYSIS_ROOT / "networks" / "network_manifest.json",
    "structure": ANALYSIS_ROOT / "networks" / "structure" / "structure_manifest.json",
    "structure_summary": ANALYSIS_ROOT / "networks" / "structure" / "structure_summary.csv",
    "structure_pagerank_stability": ANALYSIS_ROOT / "networks" / "structure" / "pagerank_stability.csv",
    "temporal": ANALYSIS_ROOT / "networks" / "temporal" / "temporal_manifest.json",
    "temporal_cascade_summary": ANALYSIS_ROOT / "networks" / "temporal" / "cascade_summary.csv",
    "topics": ANALYSIS_ROOT / "topics" / "topics_manifest.json",
    "topic_prevalence": ANALYSIS_ROOT / "topics" / "topic_prevalence.csv",
    "topic_terms": ANALYSIS_ROOT / "topics" / "topic_terms.csv",
    "topic_stability": ANALYSIS_ROOT / "topics" / "topic_stability.csv",
    "topic_review_packets": ANALYSIS_ROOT / "topics" / "topic_review_packets.csv",
    "topic_naming_ledger": ANALYSIS_ROOT / "topics" / "topic_naming_ledger.csv",
    "topic_novelty": ANALYSIS_ROOT / "topics" / "topic_novelty.csv",
    "robustness": ANALYSIS_ROOT / "robustness" / "robustness_manifest.json",
    "robustness_variant_results": ANALYSIS_ROOT / "robustness" / "variant_results.csv",
    "evaluation": EVALUATION_REPORT,
    "selection_sentiment": SENTIMENT_SELECTION,
    "selection_stance_frames": STANCE_FRAME_SELECTION,
    "reserve_activation": RESERVE_ACTIVATION,
    "reserve_assessment": RESERVE_ASSESSMENT,
    "predictions": ANALYSIS_ROOT / "predictions" / "predictions_manifest.json",
    "extended_timing": ANALYSIS_ROOT / "extended" / "timing.json",
    "extended_prediction": ANALYSIS_ROOT / "extended" / "prediction.json",
    **{
        f"hypothesis_{name}": ANALYSIS_ROOT / "hypotheses" / f"{name}.json"
        for name in ("h1", "h2", "h3", "h4")
    },
    "hypothesis_h1_reddit": ANALYSIS_ROOT / "hypotheses" / "h1_reddit.json",
    "hypothesis_h1_bluesky": ANALYSIS_ROOT / "hypotheses" / "h1_bluesky.json",
    "hypothesis_h2_privacy": ANALYSIS_ROOT / "hypotheses" / "h2_privacy_surveillance.json",
    "hypothesis_h2_circumvention": ANALYSIS_ROOT / "hypotheses" / "h2_circumvention_censorship_autonomy.json",
    "primary6": ANALYSIS_ROOT / "hypotheses" / "primary6.json",
    "h4_human": ANALYSIS_ROOT / "h4_human" / "h4_result.json",
    "h1h3_human": ANALYSIS_ROOT / "h1h3_human" / "h1h3_result.json",
}
HYPOTHESIS_CONTRACTS = {
    "hypothesis_h1": ("H1", None, None),
    "hypothesis_h1_reddit": ("H1", "reddit", None),
    "hypothesis_h1_bluesky": ("H1", "bluesky", None),
    "hypothesis_h2": ("H2", None, None),
    "hypothesis_h2_privacy": ("H2", None, "privacy_surveillance"),
    "hypothesis_h2_circumvention": ("H2", None, "circumvention_censorship_autonomy"),
    "hypothesis_h3": ("H3", None, None),
    "hypothesis_h4": ("H4", None, None),
}
TRANSITIVE_SOURCE_FILES = {
    "authority_plan": REPO_ROOT / "verify_me_not_analysis_plan_revised.md",
    "authority_proposal": REPO_ROOT / "docs" / "age_gate_paradox_project_proposal.md",
    "assessment_spec": REPO_ROOT / "ASSESSMENT_SPEC.md",
    "data_handling": REPO_ROOT / "docs" / "data_handling.md",
    "events": REPO_ROOT / "config" / "events.csv",
    "case_windows": REPO_ROOT / "config" / "case_windows.csv",
    "youtube_seed_videos": REPO_ROOT / "config" / "youtube" / "seed_videos.csv",
    "reddit_corpus": REPO_ROOT / "data" / "processed" / "reddit_age_gate" / "analysis_corpus.parquet",
    "bluesky_posts": REPO_ROOT / "data" / "processed" / "bluesky" / "posts.parquet",
    "bluesky_authors": REPO_ROOT / "data" / "processed" / "bluesky" / "authors.parquet",
    "bluesky_queries": REPO_ROOT / "data" / "processed" / "bluesky" / "post_query.parquet",
    "bluesky_thread_edges": REPO_ROOT / "data" / "processed" / "bluesky" / "thread_edges.parquet",
    "bluesky_frames": REPO_ROOT / "data" / "processed" / "bluesky" / "frames.parquet",
    "youtube_documents": REPO_ROOT / "data" / "processed" / "youtube" / "documents.parquet",
    "youtube_interactions": REPO_ROOT / "data" / "processed" / "youtube" / "interactions.parquet",
    "reddit_relevance_audit": REPO_ROOT / "data" / "analysis" / "annotation" / "reddit_thread_relevance.csv",
    "youtube_metadata_audit": REPO_ROOT / "data" / "analysis" / "annotation" / "youtube_video_metadata.csv",
    "model_receipts": MODEL_RECEIPTS,
    "lecturer_approval": REPO_ROOT / "data" / "analysis" / "scope" / "lecturer_approval.json",
}


def _coverage() -> pd.DataFrame:
    return pd.read_csv(ANALYSIS_ROOT / "descriptive" / "coverage.csv")


def _structure() -> pd.DataFrame:
    return pd.read_csv(ANALYSIS_ROOT / "networks" / "structure" / "structure_summary.csv")


def _topics() -> pd.DataFrame:
    return pd.read_csv(ANALYSIS_ROOT / "topics" / "topic_stability.csv")


def _measurement_summary() -> list[str]:
    sentiment = read_json(SOURCE_ARTIFACTS["selection_sentiment"])
    stance_frames = read_json(SOURCE_ARTIFACTS["selection_stance_frames"])
    reserve = read_json(SOURCE_ARTIFACTS["reserve_assessment"])
    return [
        f"- Sentiment: {', '.join(f'{platform}={pipeline}' for platform, pipeline in sorted(sentiment.get('primary_pipeline', {}).items()))}; mixed_ambiguous uses the declared weighted human-sample fallback.",
        f"- Stance: {', '.join(f'{platform}={pipeline}' for platform, pipeline in sorted(stance_frames.get('results', {}).get('stance', {}).get('primary_pipeline', {}).items()))}; status={stance_frames.get('results', {}).get('stance', {}).get('status', 'unknown')} because the exact pinned NLI receipt is unavailable.",
        f"- Frames: {', '.join(f'{platform}={pipeline}' for platform, pipeline in sorted(stance_frames.get('results', {}).get('frames', {}).get('primary_pipeline', {}).items()))}; development thresholds are frozen per platform.",
        f"- Reserve: status={reserve.get('status', 'unknown')}; activation reason={reserve.get('activation_reason', 'unknown')}; one-time opening={reserve.get('evaluation_access', 'unknown')}; current reserve task cells remain fallback-gated.",
    ]


def _hypothesis_statuses() -> dict[str, str]:
    return {
        name.upper(): read_json(ANALYSIS_ROOT / "hypotheses" / f"{name}.json")["status"]
        for name in ("h1", "h2", "h3", "h4")
    }


def _report_status(hypothesis_status: dict[str, str]) -> str:
    primary = read_json(SOURCE_ARTIFACTS["primary6"]) if SOURCE_ARTIFACTS["primary6"].exists() else {}
    robustness = read_json(SOURCE_ARTIFACTS["robustness"]) if SOURCE_ARTIFACTS["robustness"].exists() else {}
    decision = primary.get("decision", {})
    confirmatory_ready = (
        set(hypothesis_status) == {"H1", "H2", "H3", "H4"}
        and all(status == "complete" for status in hypothesis_status.values())
        and decision.get("status") == "complete"
        and robustness.get("status") == "complete"
        and all(_robustness_primary_stability().get(name) is True for name in ("H1", "H2", "H3", "H4"))
    )
    return "confirmatory_bundle_ready" if confirmatory_ready else "descriptive_bundle_ready_confirmatory_blocked"


ACTION_COLUMNS = [
    "priority",
    "priority_candidate",
    "actor",
    "change_first",
    "replication_basis",
    "mechanism",
    "why_this_first",
    "success_indicator",
    "failure_indicator",
    "guardrail",
    "eligibility_status",
]


PRIMARY_ENDPOINT_HYPOTHESIS = {
    "H1-R": "H1",
    "H1-B": "H1",
    "H2-P": "H2",
    "H2-C": "H2",
    "H3": "H3",
    "H4": "H4",
}
POLICY_ACTIONS = {
    "H1-R": {
        "priority": "community-aware assurance communication",
        "actor": "platform safety teams and public communicators",
        "change_first": "Test community-specific explanations and appeal routes for age-assurance changes.",
        "replication_basis": "H1-R and H1-B are the predeclared cross-platform community-frame endpoints.",
        "mechanism": "Clear, locally relevant explanations can reduce avoidable ambiguity without treating observed community structure as exposure or persuasion.",
        "success_indicator": "predeclared comprehension and appeal measures improve without widening privacy or access disparities",
        "failure_indicator": "comprehension or appeal outcomes do not improve, or subgroup disparities widen",
        "guardrail": "do not infer identity, geography, exposure or persuasion from network structure",
    },
    "H2-P": {
        "priority": "privacy-preserving age assurance",
        "actor": "platform implementers and regulators",
        "change_first": "Prioritise data minimisation, non-retention and independently tested appeal paths.",
        "replication_basis": "H2-P is a predeclared Reddit implementation-versus-legislation endpoint; its mechanism is platform-specific.",
        "mechanism": "Reducing identifier retention and making appeals auditable directly addresses the measured privacy/surveillance frame risk.",
        "success_indicator": "retention and appeal-failure measures fall while verified access and accessibility targets hold",
        "failure_indicator": "privacy complaints or appeal failures rise, or legitimate access completion falls",
        "guardrail": "retain no raw identifiers in analysis and require independent child-safety and accessibility review",
    },
    "H2-C": {
        "priority": "circumvention and autonomy safeguards",
        "actor": "platform implementers and regulators",
        "change_first": "Run adversarial circumvention and free-expression testing before broadening enforcement.",
        "replication_basis": "H2-C is a predeclared Reddit implementation-versus-legislation endpoint; its mechanism is platform-specific.",
        "mechanism": "Testing bypass routes and overreach before rollout exposes efficacy and autonomy failures while they remain reversible.",
        "success_indicator": "documented bypass and false-positive rates fall without suppressing legitimate speech or support access",
        "failure_indicator": "bypass, false-positive or appeal-failure rates increase after the change",
        "guardrail": "publish no operational bypass details and maintain an independent appeal and rights review",
    },
    "H3": {
        "priority": "broker-accessible policy explanations",
        "actor": "platform safety teams and regulators",
        "change_first": "Make evidence, limits and appeal procedures available through the observed high-brokerage discussion contexts.",
        "replication_basis": "H3 is a predeclared Reddit broker-versus-matched-author endpoint, not a claim about individual influence.",
        "mechanism": "Accessible documentation in structurally connected contexts can improve scrutiny without ranking people as influential.",
        "success_indicator": "independent comprehension and appeal-quality measures improve in predeclared contexts",
        "failure_indicator": "no improvement or increased concentration of attention around a small set of accounts",
        "guardrail": "use aggregate structural measures only; do not target, profile or contact individual authors",
    },
    "H4": {
        "priority": "video-source alignment review",
        "actor": "platform policy and trust teams",
        "change_first": "Review whether news and commentary videos present age-assurance metadata consistently before changing recommendation or enforcement rules.",
        "replication_basis": "H4 is bounded to the purposively selected E2/E3 videos and requires the declared channel/video robustness checks.",
        "mechanism": "Consistent metadata and audience-facing explanations reduce the risk that source format changes interpretation of the same policy information.",
        "success_indicator": "independent alignment and comprehension checks improve across channels without reducing legitimate reach",
        "failure_indicator": "alignment remains unstable across leave-one-video/channel checks or access disparities increase",
        "guardrail": "do not generalise beyond selected videos and do not infer audience beliefs from comments",
    },
}
POLICY_ACTIONS["H1-B"] = POLICY_ACTIONS["H1-R"]


def _robustness_primary_stability() -> dict[str, bool]:
    path = SOURCE_ARTIFACTS["robustness"]
    if not path.exists():
        return {}
    try:
        manifest = read_json(path)
        if manifest.get("status") != "complete":
            return {}
        result_path = REPO_ROOT / manifest["outputs"]["variant_results"]["path"]
        results = pd.read_csv(result_path)
    except (KeyError, OSError, ValueError):
        return {}
    required = {"hypothesis_id", "variant_id", "status", "decision", "decision_stable"}
    if not required <= set(results.columns):
        return {}
    stable: dict[str, bool] = {}
    for hypothesis_id, variant_ids in PRIMARY_VARIANT_IDS.items():
        primary = results[(results["hypothesis_id"].eq(hypothesis_id)) & (results["variant_id"].isin(variant_ids))]
        if set(primary["variant_id"]) != set(variant_ids):
            continue
        stable[hypothesis_id] = bool(
            primary["status"].astype(str).str.casefold().eq("available").all()
            and primary["decision_stable"].astype(str).str.casefold().eq("true").all()
        )
    return stable


def _policy_action_rows() -> list[dict[str, Any]]:
    primary_path = SOURCE_ARTIFACTS["primary6"]
    if not primary_path.exists():
        return []
    try:
        decision = read_json(primary_path)["decision"]
        if decision.get("status") != "complete" or decision.get("missing_endpoints"):
            return []
        adjusted = decision["adjusted_p_values"]
        effects = decision["effects"]
        decisions = decision["decisions"]
    except (KeyError, OSError, TypeError, ValueError):
        return []
    stable = _robustness_primary_stability()
    eligible = []
    for endpoint in PRIMARY_ENDPOINT_HYPOTHESIS:
        effect, p_value = effects.get(endpoint), adjusted.get(endpoint)
        if decisions.get(endpoint) is not True or PRIMARY_ENDPOINT_HYPOTHESIS[endpoint] not in stable or not stable[PRIMARY_ENDPOINT_HYPOTHESIS[endpoint]]:
            continue
        if effect is None or p_value is None or endpoint not in POLICY_ACTIONS:
            continue
        try:
            eligible.append((endpoint, abs(float(effect))))
        except (TypeError, ValueError):
            continue
    rows = []
    seen_priorities: set[str] = set()
    for endpoint, _ in sorted(eligible, key=lambda item: (-item[1], item[0])):
        template = POLICY_ACTIONS[endpoint]
        if template["priority"] in seen_priorities:
            continue
        seen_priorities.add(template["priority"])
        rows.append(
            {
                **template,
                "priority_candidate": True,
                "why_this_first": f"{endpoint} passed the stored Holm-adjusted test (p={float(adjusted[endpoint]):.4g}) with effect={float(effects[endpoint]):.4f}; its primary robustness decision is stable.",
                "eligibility_status": "policy-priority candidate; confirmatory endpoint and decisive robustness passed",
            }
        )
        if len(rows) == 2:
            break
    return rows


def _actions(report_status: str, hypothesis_status: dict[str, str]) -> pd.DataFrame:
    rows = []
    if report_status == "confirmatory_bundle_ready":
        rows.extend(_policy_action_rows())
    if report_status != "confirmatory_bundle_ready":
        statuses = ", ".join(f"{name}={status}" for name, status in sorted(hypothesis_status.items()))
        rows.append(
            {
                "priority": "not_selected",
                "priority_candidate": False,
                "actor": "research team",
                "change_first": "Resolve the current measurement, population, and endpoint gates before making a confirmatory claim.",
                "replication_basis": "No policy priority is eligible while a required endpoint or robustness gate is blocked.",
                "mechanism": "The evidence prerequisites protect the analysis from unsupported policy interpretation.",
                "why_this_first": "Current endpoint status is incomplete, so selecting a policy issue would overstate the evidence.",
                "success_indicator": f"all six endpoint and robustness gates are current and pass; current hypothesis status: {statuses}",
                "failure_indicator": "any endpoint or prerequisite remains blocked, incomplete, stale, or fails its declared gate",
                "guardrail": "retain masked text, no identity/geography inference, and no model predictions as gold",
                "eligibility_status": "evidence-prerequisite, not a policy priority",
            }
        )
    timing = _extended_stage_status(SOURCE_ARTIFACTS["extended_timing"])
    prediction = _extended_stage_status(SOURCE_ARTIFACTS["extended_prediction"])
    if not (timing["assessed_output_written"] and prediction["assessed_output_written"]):
        rows.append(
            {
                "priority": "not_selected",
                "priority_candidate": False,
                "actor": "research team and lecturer",
                "change_first": f"Record current approval before activating timing/forecast extensions (timing={timing['status']}; prediction={prediction['status']}).",
                "replication_basis": "Extended stages are approval-gated and separate from PRIMARY-6.",
                "mechanism": "Stage-specific approval and leakage checks prevent exploratory extensions from becoming unsupported claims.",
                "why_this_first": "The current timing/prediction artifacts do not contain an assessed output.",
                "success_indicator": "a dated receipt and clean stage-specific diagnostics are present",
                "failure_indicator": "approval, assessed output, or leakage/placebo checks are absent",
                "guardrail": "do not convert association into causal policy effectiveness",
                "eligibility_status": "evidence-prerequisite, not a policy priority",
            }
        )
    return pd.DataFrame(rows, columns=ACTION_COLUMNS)


def _sources() -> dict[str, str | None]:
    return {name: sha256_file(path) if path.exists() else None for name, path in SOURCE_ARTIFACTS.items()}


def _transitive_sources() -> dict[str, str | None]:
    return {name: sha256_file(path) if path.exists() else None for name, path in TRANSITIVE_SOURCE_FILES.items()}


def _check_human_sample_artifacts() -> None:
    validate_h4_result(SOURCE_ARTIFACTS["h4_human"])
    validate_h1h3_result(SOURCE_ARTIFACTS["h1h3_human"])


def _check_hypothesis_contracts() -> None:
    for source_name, (hypothesis_id, platform, outcome) in HYPOTHESIS_CONTRACTS.items():
        expected_path = SOURCE_ARTIFACTS[source_name].resolve()
        checked = check_hypothesis(hypothesis_id, platform, outcome)
        checked_path = (REPO_ROOT / checked["artifact"]).resolve()
        if checked_path != expected_path:
            raise ValueError(f"hypothesis contract resolved to the wrong artifact: {source_name}")


def _primary6_finding_card() -> dict[str, Any]:
    path = SOURCE_ARTIFACTS["primary6"]
    intervals: dict[str, dict[str, float]] = {}
    if not path.exists():
        available: dict[str, float] = {}
        effects: dict[str, float] = {}
        missing = ["H1-R", "H1-B", "H2-P", "H2-C", "H3", "H4"]
    else:
        decision = read_json(path).get("decision", {})
        available = {str(endpoint): float(value) for endpoint, value in decision.get("adjusted_p_values", {}).items() if value is not None}
        effects = {str(endpoint): float(value) for endpoint, value in decision.get("effects", {}).items() if value is not None}
        intervals = decision.get("intervals", {})
        missing = [str(endpoint) for endpoint in decision.get("missing_endpoints", [])]
    detail_parts = [
        f"{endpoint}: effect={effects[endpoint]:.4f}, interval=[{intervals[endpoint]['lower_95']:.4f}, {intervals[endpoint]['upper_95']:.4f}], Holm-adjusted p={available[endpoint]:.4f}"
        for endpoint in sorted(set(available) & set(effects))
        if endpoint in intervals and {"lower_95", "upper_95"} <= set(intervals[endpoint])
    ]
    detail_parts.extend(
        f"{endpoint}: effect={effects[endpoint]:.4f}, interval=not-stored, Holm-adjusted p={available[endpoint]:.4f}"
        for endpoint in sorted((set(available) & set(effects)) - set(intervals))
    )
    details = "; ".join(detail_parts) or "no endpoint effect or adjusted p-value is available"
    if not missing:
        status = "confirmatory estimates available"
        claim = "PRIMARY-6 contains complete endpoint results; non-passing endpoints are reported as tested null or small-effect results, not missing data."
        conclusion = "Interpret passing and non-passing effects only with their stored intervals, adjusted tests and declared robustness outputs."
    else:
        status = "process status; not a finding"
        claim = f"PRIMARY-6 has {6 - len(missing)}/6 endpoint result(s); missing endpoints remain gated: {', '.join(missing)}."
        conclusion = "Do not report a complete confirmatory family or policy priority until every endpoint passes its prerequisites."
    return {
        "finding_id": "F4",
        "claim": claim,
        "evidence": "stored hypothesis result tables, PRIMARY-6 correction artifact, reserve assessment and full-corpus prediction manifest",
        "evidential_status": status,
        "population": "six named PRIMARY-6 endpoints",
        "sample_or_graph_size": f"{6 - len(missing)}/6 endpoint result(s) available",
        "label_model_version": "age-gate-codebook-v2; status follows the stored measurement and audit gates",
        "effect_interval": details,
        "corrected_test": "Holm-adjusted PRIMARY-6 p-values from primary6.json",
        "primary_visual": "data/analysis/report/number_audit.csv",
        "source_data_path": "data/analysis/hypotheses/primary6.json",
        "decisive_robustness": "only stored endpoint results are rendered; blocked endpoints are not treated as nulls",
        "alternative_explanation": "missing or inconclusive gates represent unresolved measurement/design state, not evidence of no effect",
        "strongest_alternative_explanation": "missing or inconclusive gates represent unresolved measurement/design state, not evidence of no effect",
        "bounded_conclusion": conclusion,
        "decision_implication": "Complete labels, audits and model receipts before inferential policy claims.",
        "decision_relevance": "Complete labels, audits and model receipts before inferential policy claims.",
    }


def _extended_stage_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "assessed_output_written": False}
    payload = read_json(path)
    return {
        "status": payload.get("status", "unknown"),
        "assessed_output_written": bool(payload.get("assessed_output_written")),
        "result_present": payload.get("result") is not None or payload.get("metrics") is not None,
    }


def _extended_finding_card() -> dict[str, Any]:
    timing = _extended_stage_status(SOURCE_ARTIFACTS["extended_timing"])
    prediction = _extended_stage_status(SOURCE_ARTIFACTS["extended_prediction"])
    statuses = f"timing={timing['status']}; prediction={prediction['status']}"
    assessed = timing["assessed_output_written"] or prediction["assessed_output_written"]
    return {
        "finding_id": "F5",
        "claim": "Extended timing and forecasting status is read from the current approval-bound artifacts.",
        "evidence": "approval-bound timing and prediction artifacts",
        "evidential_status": "assessed output present" if assessed else "process status; not a finding",
        "population": "frozen Google Trends context and proposed extended outcomes",
        "sample_or_graph_size": statuses,
        "label_model_version": "approval-gated; status follows the stored approval binding",
        "effect_interval": "stored result intervals are rendered from the stage artifacts" if assessed else f"no assessed interval; {statuses}",
        "corrected_test": "stage-specific diagnostics only; no PRIMARY-6 correction",
        "primary_visual": "data/analysis/extended/timing.json",
        "source_data_path": "data/analysis/extended/prediction.json",
        "decisive_robustness": "placebo/leakage and rolling-origin diagnostics are stage-gated",
        "alternative_explanation": "an unapproved or unexecuted stage is not evidence of temporal non-association",
        "strongest_alternative_explanation": "an unapproved or unexecuted stage is not evidence of temporal non-association",
        "bounded_conclusion": "No extended association or forecast claim is made unless the current stage artifact contains an approved assessed output.",
        "decision_implication": "Review the current approval binding and stage diagnostics before using an extended result.",
        "decision_relevance": "Review the current approval binding and stage diagnostics before using an extended result.",
    }


def _build_figures(coverage: pd.DataFrame, structure: pd.DataFrame, topics: pd.DataFrame) -> dict[str, Path]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {
        "coverage": FIGURE_ROOT / "coverage_denominators.png",
        "network": FIGURE_ROOT / "network_sizes.png",
        "topics": FIGURE_ROOT / "topic_seed_stability.png",
        "concentration": FIGURE_ROOT / "concentration_top10.png",
        "cascade_depth": FIGURE_ROOT / "cascade_depth.png",
        "centrality_stability": FIGURE_ROOT / "centrality_stability.png",
    }
    coverage_plot = coverage.copy()
    coverage_plot["label"] = coverage_plot["platform"] + ":" + coverage_plot["scope"]
    coverage_plot = coverage_plot.sort_values("inclusive_english_rows")
    ax = coverage_plot.set_index("label")[["strict_english_rows", "inclusive_english_rows"]].plot.barh(figsize=(10, 7), color=["#355C7D", "#F67280"])
    ax.set_xlabel("documents/posts/comments; source sample denominator")
    ax.set_ylabel("platform and frozen scope")
    ax.set_title("Strict and inclusive language denominators")
    ax.legend(["strict English", "inclusive uncertain-language bound"], loc="lower right")
    ax.figure.tight_layout()
    ax.figure.savefig(paths["coverage"], dpi=160)
    plt.close(ax.figure)

    structure_plot = structure.copy()
    structure_plot["label"] = structure_plot["graph_id"] + ":" + structure_plot["scope"]
    ax = structure_plot.set_index("label")[["nodes", "edges"]].plot.bar(figsize=(10, 6), color=["#355C7D", "#99B898"])
    ax.set_ylabel("count")
    ax.set_title("Topology-only graph sizes")
    ax.tick_params(axis="x", rotation=35)
    ax.figure.tight_layout()
    ax.figure.savefig(paths["network"], dpi=160)
    plt.close(ax.figure)

    stability = topics.groupby("platform", as_index=True)["mean_top_term_jaccard"].mean().sort_values()
    ax = stability.plot.barh(figsize=(8, 4), color="#2A9D8F")
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean matched top-term Jaccard")
    ax.set_title("NMF seed stability; topic naming remains human-gated")
    ax.figure.tight_layout()
    ax.figure.savefig(paths["topics"], dpi=160)
    plt.close(ax.figure)

    concentration = pd.read_csv(ANALYSIS_ROOT / "descriptive" / "concentration.csv")
    concentration = concentration[concentration["entity"].isin(["author", "thread_id", "root_doc_id"])].copy()
    concentration["label"] = concentration["platform"] + ":" + concentration["entity"]
    ax = concentration.sort_values("top_10_share").set_index("label")["top_10_share"].plot.barh(figsize=(9, 6), color="#E76F51")
    ax.set_xlabel("share held by the top ten entities")
    ax.set_title("Observed concentration by platform and entity type")
    ax.figure.tight_layout()
    ax.figure.savefig(paths["concentration"], dpi=160)
    plt.close(ax.figure)

    cascade = pd.read_csv(ANALYSIS_ROOT / "networks" / "temporal" / "cascade_summary.csv")
    observed = cascade[cascade["observed_parent_edges"].fillna(False).astype(bool)].copy()
    cascade_summary = observed.groupby("platform", as_index=True).agg(
        median_max_depth=("max_depth", "median"),
        median_structural_virality=("structural_virality", "median"),
    )
    ax = cascade_summary.plot.bar(figsize=(8, 5), color=["#457B9D", "#A8DADC"])
    ax.set_ylabel("proxy units")
    ax.set_title("Observed cascade depth and structural virality")
    ax.tick_params(axis="x", rotation=0)
    ax.figure.tight_layout()
    ax.figure.savefig(paths["cascade_depth"], dpi=160)
    plt.close(ax.figure)

    pagerank = pd.read_csv(ANALYSIS_ROOT / "networks" / "structure" / "pagerank_stability.csv")
    pagerank["top_k_jaccard"] = pd.to_numeric(pagerank["top_k_jaccard"], errors="coerce")
    stability = pagerank.groupby("graph_id", as_index=True)["top_k_jaccard"].mean().sort_values()
    ax = stability.plot.barh(figsize=(9, 6), color="#6A994E")
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean top-k Jaccard across declared sensitivity checks")
    ax.set_title("Centrality ranking stability")
    ax.figure.tight_layout()
    ax.figure.savefig(paths["centrality_stability"], dpi=160)
    plt.close(ax.figure)
    return paths


def build_report_artifacts() -> dict[str, Any]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    _check_human_sample_artifacts()
    check_topics()
    check_robustness()
    _check_hypothesis_contracts()
    coverage, structure, topics = _coverage(), _structure(), _topics()
    hypothesis_status = _hypothesis_statuses()
    report_status = _report_status(hypothesis_status)
    measurement_summary = _measurement_summary()
    figures = _build_figures(coverage, structure, topics)
    largest_scope = coverage.sort_values("rows", ascending=False).iloc[0]
    actor_slices = structure[structure["graph_id"].astype(str).str.contains("ACTOR")]
    largest_actor = actor_slices.sort_values("nodes", ascending=False).iloc[0]
    topic_stability = topics.groupby("platform")["mean_top_term_jaccard"].mean()
    finding_cards = pd.DataFrame(
        [
            {
                "finding_id": "F1",
                "claim": f"The frozen material contains {int(coverage['rows'].sum()):,} scope-level rows; the largest scope is {largest_scope['platform']}:{largest_scope['scope']} with {int(largest_scope['rows']):,} rows, so it is not a public-opinion prevalence estimate.",
                "evidence": "descriptive coverage, exclusions and timeline artifacts",
                "evidential_status": "supported descriptive",
                "population": "frozen processed platform-event sources",
                "sample_or_graph_size": f"{int(coverage['rows'].sum()):,} scope rows across {coverage['platform'].nunique()} platforms",
                "label_model_version": "label-independent; age-gate-codebook-v2 boundary",
                "effect_interval": "descriptive denominators only; no prevalence interval",
                "corrected_test": "not applicable",
                "primary_visual": "data/analysis/report/figures/coverage_denominators.png",
                "source_data_path": "data/analysis/descriptive/coverage.csv",
                "decisive_robustness": "strict/inclusive language bounds are reported as separate populations",
                "alternative_explanation": "collection and language filters may shape observed volume",
                "strongest_alternative_explanation": "collection and language filters may shape observed volume",
                "bounded_conclusion": "Report denominators, unsupported days and language bounds with every comparison.",
                "decision_implication": "No platform-wide prevalence or user-geography claim.",
                "decision_relevance": "No platform-wide prevalence or user-geography claim.",
            },
            {
                "finding_id": "F2",
                "claim": f"Observed topology varies by frozen slice: {largest_actor['graph_id']}:{largest_actor['scope']} has {int(largest_actor['nodes']):,} actor nodes and {int(largest_actor['edges']):,} directed edges; attention, brokerage and conversation flow remain distinct measures.",
                "evidence": "graph registry, topology-only roles, null diagnostics and cascade summaries",
                "evidential_status": f"exploratory; static community method(s): {', '.join(sorted(structure['community_method'].astype(str).unique()))}",
                "population": "scoped observed interaction graphs",
                "sample_or_graph_size": f"{len(structure)} graph-scope slices; sizes in structure_summary.csv",
                "label_model_version": "topology-only; v2 frame integration human-gated",
                "effect_interval": "centrality/null diagnostics; no causal effect interval",
                "corrected_test": "degree-preserving null diagnostics; no PRIMARY-6 endpoint",
                "primary_visual": "data/analysis/report/figures/network_sizes.png",
                "source_data_path": "data/analysis/networks/structure/structure_summary.csv",
                "decisive_robustness": "scope-separated graph views and cascade sensitivities",
                "alternative_explanation": "fallback communities and incomplete observed replies can alter structure",
                "strongest_alternative_explanation": "fallback communities and incomplete observed replies can alter structure",
                "bounded_conclusion": "Describe observed interaction and structural proxies, never exposure or persuasion.",
                "decision_implication": "No actor-level policy ranking is justified from centrality alone.",
                "decision_relevance": "No actor-level policy ranking is justified from centrality alone.",
            },
            {
                "finding_id": "F3",
                "claim": f"NMF selected {', '.join(f'{platform}={int(count)}' for platform, count in sorted(topics.groupby('platform')['topic_count'].max().items()))} topics; mean matched top-term stability is {topic_stability.min():.3f}-{topic_stability.max():.3f}, while names and theory mapping remain open.",
                "evidence": "topic model-selection diagnostics, stability table, masked review packets and naming ledger",
                "evidential_status": "exploratory",
                "population": "platform-specific masked topic documents",
                "sample_or_graph_size": f"{topics['platform'].nunique()} platform topic spaces; selected counts in topics_manifest.json",
                "label_model_version": "NMF topic selection; v2 frame labels not substituted",
                "effect_interval": "stability and support diagnostics; no inferential effect",
                "corrected_test": "not applicable; exploratory model selection",
                "primary_visual": "data/analysis/report/figures/topic_seed_stability.png",
                "source_data_path": "data/analysis/topics/topic_review_packets.csv",
                "decisive_robustness": "6/8/10 component grid across three fixed seeds",
                "alternative_explanation": "topic terms may reflect platform vocabulary rather than stable concerns",
                "strongest_alternative_explanation": "topic terms may reflect platform vocabulary rather than stable concerns",
                "bounded_conclusion": "Use topics to propose questions, not to replace v2 human frame labels.",
                "decision_implication": "Human topic naming and representative-document review are required.",
                "decision_relevance": "Human topic naming and representative-document review are required.",
            },
            _primary6_finding_card(),
            _extended_finding_card(),
        ]
    )
    evidence = finding_cards.rename(columns={"evidence": "evidence_artifact", "evidential_status": "status"})
    actions = _actions(report_status, hypothesis_status)
    number_audit = []
    for platform, group in coverage.groupby("platform", sort=True):
        number_audit.append({"metric": f"{platform}_scope_rows", "value": int(group["rows"].sum()), "source": "data/analysis/descriptive/coverage.csv", "field": "rows", "status": "audited_artifact"})
        number_audit.append({"metric": f"{platform}_strict_english_rows", "value": int(group["strict_english_rows"].sum()), "source": "data/analysis/descriptive/coverage.csv", "field": "strict_english_rows", "status": "audited_artifact"})
    for row in structure.itertuples(index=False):
        number_audit.append({"metric": f"{row.graph_id}_{row.scope}_nodes", "value": int(row.nodes), "source": "data/analysis/networks/structure/structure_summary.csv", "field": "nodes", "status": str(row.community_status)})
        number_audit.append({"metric": f"{row.graph_id}_{row.scope}_edges", "value": int(row.edges), "source": "data/analysis/networks/structure/structure_summary.csv", "field": "edges", "status": str(row.community_status)})
    for platform, value in topics.groupby("platform")["mean_top_term_jaccard"].mean().items():
        number_audit.append({"metric": f"{platform}_mean_topic_stability", "value": round(float(value), 6), "source": "data/analysis/topics/topic_stability.csv", "field": "mean_top_term_jaccard", "status": "exploratory"})
    number_audit.extend({"metric": f"{name}_status", "value": status, "source": f"data/analysis/hypotheses/{name.lower()}.json", "field": "status", "status": status} for name, status in hypothesis_status.items())
    outputs = {
        "finding_cards": REPORT_ROOT / "finding_cards.csv",
        "evidence_matrix": REPORT_ROOT / "evidence_matrix.csv",
        "action_matrix": REPORT_ROOT / "action_matrix.csv",
        "number_audit": REPORT_ROOT / "number_audit.csv",
    }
    finding_cards.to_csv(outputs["finding_cards"], index=False, lineterminator="\n")
    evidence.to_csv(outputs["evidence_matrix"], index=False, lineterminator="\n")
    actions.to_csv(outputs["action_matrix"], index=False, lineterminator="\n")
    pd.DataFrame(number_audit).to_csv(outputs["number_audit"], index=False, lineterminator="\n")
    report_markdown = _report_markdown(finding_cards, actions, coverage, structure, topics, figures, hypothesis_status, measurement_summary, report_status)
    markdown_path = REPORT_ROOT / "age_gate_paradox_report.md"
    markdown_path.write_text(report_markdown, encoding="utf-8")
    outputs["report_markdown"] = markdown_path
    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "command": "python -m src.analysis.run report --check",
        "sources": _sources(),
        "transitive_sources": _transitive_sources(),
        "finding_count": int(len(finding_cards)),
        "action_count": int(len(actions)),
        "policy_priorities_selected": int(actions["priority_candidate"].sum()),
        "hypothesis_status": hypothesis_status,
        "outputs": {
            name: {"path": relative_path(path), "sha256": sha256_file(path), "rows": int(len(pd.read_csv(path))) if path.suffix == ".csv" else None}
            for name, path in {**outputs, **{f"figure_{name}": path for name, path in figures.items()}}.items()
        },
        "status": report_status,
        "limitations": [f"Report status is {report_status}; policy prioritisation is outside this evidence bundle.", "Figures are derived from versioned tables and contain no raw text or identifiers."],
    }
    write_json(REPORT_MANIFEST, manifest)
    return manifest


def _report_markdown(
    finding_cards: pd.DataFrame,
    actions: pd.DataFrame,
    coverage: pd.DataFrame,
    structure: pd.DataFrame,
    topics: pd.DataFrame,
    figures: dict[str, Path],
    hypothesis_status: dict[str, str],
    measurement_summary: list[str],
    report_status: str,
) -> str:
    topic_manifest = read_json(SOURCE_ARTIFACTS["topics"]) if SOURCE_ARTIFACTS["topics"].exists() else {}
    topic_status = topic_manifest.get("status", "unknown")
    topic_gate_text = {
        "exploratory_named": "human topic names are recorded",
        "exploratory_review_required": "human topic naming and representative review remain pending",
    }.get(topic_status, f"topic gate status={topic_status}")
    action_text = (
        "No policy priority is selected."
        if not int(actions["priority_candidate"].sum())
        else f"{int(actions['priority_candidate'].sum())} policy priority candidate(s) are recorded."
    )
    lines = [
        "# Age-Gate Paradox: frozen-data analysis report bundle",
        "",
        "This report is generated from the frozen processed sources. It distinguishes observed descriptive evidence, exploratory structure, blocked validity gates, and approval-gated extensions. It does not publish raw text, usernames, identifiers, inferred geography, or cross-platform identities.",
        "",
        "## Evidence and gate cards",
        "",
    ]
    for row in finding_cards.itertuples(index=False):
        lines.extend(
            [
                f"### {row.finding_id}: {row.claim}",
                "",
                f"- Status: {row.evidential_status}",
                f"- Population/size: {row.population}; {row.sample_or_graph_size}",
                f"- Label/model: {row.label_model_version}",
                f"- Effect/interval: {row.effect_interval}",
                f"- Corrected test: {row.corrected_test}",
                f"- Evidence: {row.evidence}",
                f"- Primary visual/source: {row.primary_visual}; {row.source_data_path}",
                f"- Decisive robustness: {row.decisive_robustness}",
                f"- Alternative explanation: {row.strongest_alternative_explanation}",
                f"- Bounded conclusion: {row.bounded_conclusion}",
                f"- Decision relevance: {row.decision_relevance}",
                "",
            ]
        )
    confirmatory_card = finding_cards.loc[finding_cards["finding_id"].eq("F4")].iloc[0]
    confirmatory_text = (
        "Stored PRIMARY-6 endpoint results are rendered above; no causal or policy claim is implied."
        if confirmatory_card["evidential_status"] == "confirmatory estimates available"
        else "No complete PRIMARY-6 effect, interval, p-value, causal claim, or policy priority is reported from this snapshot. The blocked state is not a null result."
    )
    lines.extend(["## Measurement selection and reserve", "", *measurement_summary, "", "## Frozen denominators", "", f"Coverage rows: {len(coverage)}; total frozen scope rows: {int(coverage['rows'].sum()):,}.", "", "![Strict and inclusive language denominators](figures/coverage_denominators.png)", "", "## Topology-only structure", "", f"Analysed graph slices: {len(structure)}; all community statuses: {', '.join(sorted(structure['community_status'].unique()))}.", "", "![Topology-only graph sizes](figures/network_sizes.png)", "", "![Observed concentration](figures/concentration_top10.png)", "", "![Observed cascade depth](figures/cascade_depth.png)", "", "![Centrality ranking stability](figures/centrality_stability.png)", "", "## Topic discovery", "", f"NMF platforms: {topics['platform'].nunique()}; selected topic counts and masked representative packets are recorded; {topic_gate_text}.", "", "![NMF seed stability](figures/topic_seed_stability.png)", "", "## Confirmatory and extended gates", "", f"Report status: {report_status}.", "", "Hypothesis statuses: " + ", ".join(f"{name}={status}" for name, status in hypothesis_status.items()) + ".", "", confirmatory_text, "", "## Actions", "", action_text + f" {len(actions)} evidence-prerequisite action(s) are recorded from the current gates.", "", "See the generated `finding_cards.csv`, `evidence_matrix.csv`, `action_matrix.csv`, and `number_audit.csv` for traceability.", ""])
    return "\n".join(lines)


def check_report() -> dict[str, Any]:
    if not REPORT_MANIFEST.exists():
        raise FileNotFoundError(f"report manifest is missing: {REPORT_MANIFEST}; run the explicit report build first")
    manifest = read_json(REPORT_MANIFEST)
    if manifest.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("report manifest schema is stale")
    if manifest.get("sources") != _sources():
        raise ValueError("report source artifact changed")
    if manifest.get("transitive_sources") != _transitive_sources():
        raise ValueError("transitive report source changed")
    _check_human_sample_artifacts()
    check_topics()
    check_robustness()
    _check_hypothesis_contracts()
    if manifest.get("status") != _report_status(_hypothesis_statuses()):
        raise ValueError("report status is stale")
    for record in manifest.get("outputs", {}).values():
        path = REPO_ROOT / record["path"]
        if not path.exists() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"report artifact changed: {record['path']}")
    return {"status": manifest["status"], "artifact": relative_path(REPORT_MANIFEST), "findings": manifest["finding_count"]}
