"""Lean, report-ready analysis for the frozen Age-Gate Paradox evidence.

This module deliberately lives outside the blocked measurement-cycle and
PRIMARY-6 lifecycle. It consumes the adjudicated evaluation sample plus
already-built topology/topic artifacts and writes only to
``data/analysis/fallback``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.analysis.artifacts import REPO_ROOT
from src.analysis.measurement import reliability_report
from src.analysis.networks import check_networks
from src.analysis.structure import check_structure
from src.analysis.topics import check_topics
from src.analysis.temporal import check_temporal


FALLBACK_ROOT = REPO_ROOT / "data" / "analysis" / "fallback"
FIGURES_ROOT = FALLBACK_ROOT / "figures"
EVALUATION_LABELS = REPO_ROOT / "data" / "analysis" / "validation" / "coordinator_labels_evaluation.csv"
FALLBACK_SCHEMA_VERSION = "age-gate-fallback.v3"
ANALYSIS_CODE_VERSION = "fallback-analysis.v4"
VALIDATION_SPLITS = ("development", "evaluation", "reserve")
SPLIT_RECORDS = {"development": 600, "evaluation": 600, "reserve": 300}
RELIABILITY_ROLES = {
    "development": "development_reliability",
    "evaluation": "evaluation_agreement",
    "reserve": "reserve_sensitivity",
}
H1_MIN_FRAME_SUPPORTED_ROWS = 5
TOPIC_NAMING_STATUS = "team-authorized_agent-assisted_review"
TOPIC_INTERPRETATION_CLASSES = {
    "substantive",
    "generic_or_discourse_style",
    "artifact_or_contamination",
}
SOURCE_PATHS = {
    "development_labels": REPO_ROOT / "data" / "analysis" / "validation" / "coordinator_labels_development.csv",
    "evaluation_labels": EVALUATION_LABELS,
    "reserve_labels": REPO_ROOT / "data" / "analysis" / "validation" / "coordinator_labels_reserve.csv",
    "coverage": REPO_ROOT / "data" / "analysis" / "descriptive" / "coverage.csv",
    "network_structure": REPO_ROOT / "data" / "analysis" / "networks" / "structure" / "structure_summary.csv",
    "network_pagerank": REPO_ROOT / "data" / "analysis" / "networks" / "structure" / "pagerank_stability.csv",
    "network_roles": REPO_ROOT / "data" / "analysis" / "networks" / "structure" / "roles.parquet",
    "network_partitions": REPO_ROOT / "data" / "analysis" / "networks" / "structure" / "partitions.parquet",
    "network_cascade": REPO_ROOT / "data" / "analysis" / "networks" / "temporal" / "cascade_summary.csv",
    "network_manifest": REPO_ROOT / "data" / "analysis" / "networks" / "network_manifest.json",
    "youtube_documents": REPO_ROOT / "data" / "processed" / "youtube" / "documents.parquet",
    "youtube_seed_videos": REPO_ROOT / "config" / "youtube" / "seed_videos.csv",
    "topic_terms": REPO_ROOT / "data" / "analysis" / "topics" / "topic_terms.csv",
    "topic_novelty": REPO_ROOT / "data" / "analysis" / "topics" / "topic_novelty.csv",
    "topic_prevalence": REPO_ROOT / "data" / "analysis" / "topics" / "topic_prevalence.csv",
    "topic_stability": REPO_ROOT / "data" / "analysis" / "topics" / "topic_stability.csv",
    "topic_review_packets": REPO_ROOT / "data" / "analysis" / "topics" / "topic_review_packets.csv",
    "topic_manifest": REPO_ROOT / "data" / "analysis" / "topics" / "topics_manifest.json",
    "bertopic_sensitivity": REPO_ROOT / "data" / "analysis" / "topics" / "bertopic_sensitivity.csv",
}
ANALYSIS_CODE_PATHS = {
    "fallback_analysis": Path(__file__).resolve(),
    "measurement": REPO_ROOT / "src" / "analysis" / "measurement.py",
    "codebook": REPO_ROOT / "src" / "analysis" / "codebook.py",
    "topics": REPO_ROOT / "src" / "analysis" / "topics.py",
}
SEED = 20260915
BOOTSTRAP_REPLICATES = 9_999
PRIMARY_RECORDS = 600
VALID_TARGETS = {"UK_OSA", "AU_SOCIAL_MINIMUM_AGE", "OTHER_EXTENDED_EVENT"}
FRAME_LABELS = [
    "policy_assurance",
    "child_safety",
    "privacy_surveillance",
    "governance_platform_responsibility",
    "circumvention_censorship_autonomy",
]
STANCE_LABELS = ["support", "oppose", "mixed_conditional", "neutral_descriptive", "unclear_ambiguous"]
SENTIMENT_LABELS = ["positive", "negative", "neutral", "mixed_ambiguous"]
REQUIRED_ADJUDICATED_FIELDS = (
    "adjudicated_relevance",
    "adjudicated_language",
    "adjudicated_target_policy",
    "adjudicated_stance",
    "adjudicated_sentiment",
    "adjudicated_bypass_techniques",
)
ENDPOINTS = ["H1-R", "H1-B", "H2-P", "H2-C", "H3", "H4"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _source_registry() -> dict[str, str]:
    return {name: _relative(path) for name, path in SOURCE_PATHS.items()}


def _source_checksums() -> dict[str, str]:
    missing = [name for name, path in SOURCE_PATHS.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"fallback source files are missing: {missing}")
    return {_relative(path): _sha256(path) for path in SOURCE_PATHS.values()}


def _analysis_code_path() -> Path:
    return Path(__file__).resolve()


def _analysis_code_sha256() -> str:
    return _sha256(_analysis_code_path())


def _analysis_code_checksums() -> dict[str, str]:
    return {_relative(path): _sha256(path) for path in ANALYSIS_CODE_PATHS.values()}


def _analysis_config() -> dict[str, Any]:
    return {
        "schema_version": FALLBACK_SCHEMA_VERSION,
        "analysis_code_version": ANALYSIS_CODE_VERSION,
        "bootstrap_seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_records": PRIMARY_RECORDS,
        "valid_targets": sorted(VALID_TARGETS),
        "frame_labels": FRAME_LABELS,
        "h1_min_frame_supported_rows": H1_MIN_FRAME_SUPPORTED_ROWS,
    }


def _validate_analysis_freshness(manifest: dict[str, Any]) -> None:
    if manifest.get("analysis_config") != _analysis_config():
        raise AssertionError("fallback analysis configuration changed; rebuild required")
    if manifest.get("analysis_code_sha256") != _analysis_code_sha256():
        raise AssertionError("fallback analysis code changed; rebuild required")
    if manifest.get("analysis_code_checksums") != _analysis_code_checksums():
        raise AssertionError("fallback analysis dependency code changed; rebuild required")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stable_seed(*parts: object) -> int:
    value = "|".join(map(str, parts)).encode("utf-8")
    return SEED + int(hashlib.sha256(value).hexdigest()[:8], 16) % 100_000


def _frame_set(value: object) -> set[str]:
    text = "" if value is None else str(value).strip()
    if not text:
        return set()
    values = {part.strip() for part in text.split("|") if part.strip()}
    unknown = values - set(FRAME_LABELS)
    if unknown:
        raise ValueError(f"unknown adjudicated frame labels: {sorted(unknown)}")
    return values


def _validate_label_values(frame: pd.DataFrame) -> None:
    allowed = {
        "adjudicated_relevance": {"relevant", "adjacent_contextual", "irrelevant"},
        "adjudicated_language": {"english", "mixed", "non_english", "too_short_ambiguous"},
        "adjudicated_target_policy": VALID_TARGETS | {"multiple", "unclear"},
        "adjudicated_stance": set(STANCE_LABELS),
        "adjudicated_sentiment": set(SENTIMENT_LABELS),
    }
    for column, values in allowed.items():
        observed = set(frame[column].astype(str).str.strip())
        invalid = observed - values
        if invalid:
            raise ValueError(f"{column} contains invalid values: {sorted(invalid)}")
    frame["adjudicated_frame_labels"].map(_frame_set)


def _load_label_split(split: str) -> pd.DataFrame:
    if split not in VALIDATION_SPLITS:
        raise ValueError(f"unknown validation split: {split}")
    frame = pd.read_csv(SOURCE_PATHS[f"{split}_labels"], keep_default_na=False)
    expected_rows = SPLIT_RECORDS[split]
    if len(frame) != expected_rows:
        raise AssertionError(f"{split} validation sample must contain exactly {expected_rows} rows, found {len(frame)}")
    if frame["annotation_id"].duplicated().any() or frame["annotation_id"].isna().any():
        raise AssertionError(f"{split} annotation IDs must be present exactly once")
    if not frame["split"].astype(str).eq(split).all():
        raise AssertionError(f"validation labels must be the {split} split")
    required_coder_fields = {
        "coder_a_relevance",
        "coder_b_relevance",
        "coder_a_language",
        "coder_b_language",
        "coder_a_target_policy",
        "coder_b_target_policy",
        "coder_a_stance",
        "coder_b_stance",
        "coder_a_sentiment",
        "coder_b_sentiment",
        "coder_a_frame_labels",
        "coder_b_frame_labels",
    }
    missing_coder_fields = sorted(required_coder_fields - set(frame.columns))
    if missing_coder_fields:
        raise ValueError(f"{split} labels missing coder columns: {missing_coder_fields}")
    return frame


def _load_evaluation() -> pd.DataFrame:
    frame = _load_label_split("evaluation")
    required = {
        "platform",
        "case_window_id",
        "author_key",
        "cluster_id",
        "inclusion_probability",
        "coder_a_relevance",
        "coder_b_relevance",
        "adjudicated_relevance",
        "adjudicated_language",
        "adjudicated_target_policy",
        "adjudicated_stance",
        "adjudicated_sentiment",
        "adjudicated_frame_labels",
        "adjudicated_bypass_techniques",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"evaluation labels missing columns: {missing}")
    _validate_label_values(frame)

    present = frame[list(REQUIRED_ADJUDICATED_FIELDS)].astype(str).apply(lambda column: column.str.strip().ne(""))
    frame["_complete"] = present.all(axis=1)
    frame["_relevant"] = frame["adjudicated_relevance"].eq("relevant")
    frame["_english"] = frame["adjudicated_language"].eq("english")
    frame["_valid_target"] = frame["adjudicated_target_policy"].isin(VALID_TARGETS)
    frame["_author_valid"] = frame["author_key"].astype(str).str.strip().ne("")
    frame["_cluster_valid"] = frame["cluster_id"].astype(str).str.strip().ne("")
    frame["_eligible"] = frame[["_complete", "_relevant", "_english", "_valid_target", "_author_valid", "_cluster_valid"]].all(axis=1)

    probabilities = pd.to_numeric(frame["inclusion_probability"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities <= 0):
        raise AssertionError("all used inclusion probabilities must be finite and positive")
    frame["_weight"] = 1.0 / probabilities
    frame["_frame_count"] = frame["adjudicated_frame_labels"].map(lambda value: len(_frame_set(value)))
    return frame


def _flow_rows(frame: pd.DataFrame) -> pd.DataFrame:
    steps = [
        ("primary_evaluation_records", pd.Series(True, index=frame.index), "Frozen coordinator evaluation input."),
        ("required_adjudicated_fields_complete", frame["_complete"], "All substantive fields populated; blank frame labels remain valid all-negative labels."),
        ("relevant_policy_discourse", frame["_relevant"], "Codebook relevance is relevant."),
        ("strict_english", frame["_english"], "Primary estimates exclude mixed, non-English and too-short/ambiguous language."),
        ("single_valid_target_policy", frame["_valid_target"], "Target is one of the three valid single-policy/event labels."),
        ("author_identifier_present", frame["_author_valid"], "Required for author-level joins and platform estimates."),
        ("cluster_identifier_present", frame["_cluster_valid"], "Required for clustered uncertainty."),
    ]
    running = pd.Series(True, index=frame.index)
    rows: list[dict[str, Any]] = []
    for step, condition, note in steps:
        before = int(running.sum())
        running &= condition
        after = int(running.sum())
        rows.append(
            {
                "analysis_type": "eligibility_flow",
                "platform": "all",
                "case_window_id": "ALL_EVALUATION",
                "measure": step,
                "category": "all",
                "estimate": float(after / len(frame)),
                "lower_95": np.nan,
                "upper_95": np.nan,
                "unweighted_n": after,
                "effective_sample_size": np.nan,
                "cluster_count": np.nan,
                "weighted_total": np.nan,
                "status": "available",
                "weighting": "unweighted_flow_count",
                "notes": f"excluded_at_step={before - after}; {note}",
            }
        )

    first_failure = pd.Series("eligible", index=frame.index, dtype="string")
    prior = pd.Series(True, index=frame.index)
    for name, condition, _ in steps[1:]:
        failed = prior & ~condition
        first_failure.loc[failed] = name
        prior &= condition
    for category, count in first_failure.value_counts().sort_index().items():
        rows.append(
            {
                "analysis_type": "eligibility_exclusion",
                "platform": "all",
                "case_window_id": "ALL_EVALUATION",
                "measure": "first_exclusion_reason",
                "category": str(category),
                "estimate": np.nan,
                "lower_95": np.nan,
                "upper_95": np.nan,
                "unweighted_n": int(count),
                "effective_sample_size": np.nan,
                "cluster_count": np.nan,
                "weighted_total": np.nan,
                "status": "available",
                "weighting": "unweighted_flow_count",
                "notes": "Mutually exclusive first failing eligibility rule.",
            }
        )
    relevant = frame[frame["_complete"] & frame["_relevant"]]
    ambiguous_language = int(relevant["adjudicated_language"].isin(["mixed", "too_short_ambiguous"]).sum())
    non_english = int(relevant["adjudicated_language"].eq("non_english").sum())
    rows.append(
        {
            "analysis_type": "eligibility_exclusion",
            "platform": "all",
            "case_window_id": "ALL_EVALUATION",
            "measure": "language_exclusion_breakdown",
            "category": "ambiguous_language",
            "estimate": np.nan,
            "lower_95": np.nan,
            "upper_95": np.nan,
            "unweighted_n": ambiguous_language,
            "effective_sample_size": np.nan,
            "cluster_count": np.nan,
            "weighted_total": np.nan,
            "status": "available",
            "weighting": "unweighted_flow_count",
            "notes": f"Among complete relevant rows; non_english={non_english}.",
        }
    )
    return pd.DataFrame(rows)


def _frame_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray([[label in _frame_set(value) for label in FRAME_LABELS] for value in frame["adjudicated_frame_labels"]], dtype=float)


def _category_matrix(frame: pd.DataFrame, field: str, categories: list[str]) -> np.ndarray:
    return np.asarray([[str(value) == category for category in categories] for value in frame[field]], dtype=float)


def _bootstrap_proportions(
    frame: pd.DataFrame,
    outcomes: np.ndarray,
    rng: np.random.Generator,
    cluster_column: str = "cluster_id",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frame.empty:
        raise ValueError("empty analytical cell")
    probabilities = frame["inclusion_probability"].to_numpy(dtype=float)
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities <= 0):
        raise ValueError("non-positive or non-finite inclusion probability")
    clusters = frame[cluster_column].astype(str).to_numpy()
    unique_clusters = np.asarray(sorted(set(clusters)))
    if len(unique_clusters) < 2:
        raise ValueError(f"cluster bootstrap requires at least two {cluster_column} clusters")
    weights = 1.0 / probabilities
    outcomes = np.asarray(outcomes, dtype=float)
    if outcomes.ndim == 1:
        outcomes = outcomes[:, None]
    cluster_weights = np.asarray([weights[clusters == cluster].sum() for cluster in unique_clusters])
    cluster_outcomes = np.asarray([(weights[clusters == cluster, None] * outcomes[clusters == cluster]).sum(axis=0) for cluster in unique_clusters])
    draws = rng.integers(len(unique_clusters), size=(BOOTSTRAP_REPLICATES, len(unique_clusters)))
    multiplicities = np.zeros((BOOTSTRAP_REPLICATES, len(unique_clusters)), dtype=np.int32)
    np.add.at(multiplicities, (np.arange(BOOTSTRAP_REPLICATES)[:, None], draws), 1)
    denominators = multiplicities @ cluster_weights
    numerators = multiplicities @ cluster_outcomes
    values = numerators / denominators[:, None]
    return values, np.quantile(values, 0.025, axis=0), np.quantile(values, 0.975, axis=0)


def _distribution_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = frame[frame["_eligible"]].copy()
    if eligible.empty:
        raise ValueError("no eligible primary evaluation records")
    rows: list[dict[str, Any]] = []
    cell_summary: dict[str, Any] = {}
    groupings = [
        ("platform", ["platform"]),
        ("platform_case", ["platform", "case_window_id"]),
    ]
    for grouping_name, columns in groupings:
        for key, group in eligible.groupby(columns, sort=True, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            platform = str(key[0])
            case = "ALL_CASES" if grouping_name == "platform" else str(key[1])
            cluster_column = "cluster_id"
            rng = np.random.default_rng(_stable_seed(grouping_name, platform, case))
            outcomes = np.column_stack([_frame_matrix(group), _category_matrix(group, "adjudicated_stance", STANCE_LABELS), _category_matrix(group, "adjudicated_sentiment", SENTIMENT_LABELS)])
            boot, lower, upper = _bootstrap_proportions(group, outcomes, rng, cluster_column)
            weights = group["_weight"].to_numpy(dtype=float)
            denominator = float(weights.sum())
            effective_n = float(denominator**2 / np.square(weights).sum())
            labels = [*FRAME_LABELS, *STANCE_LABELS, *SENTIMENT_LABELS]
            measure_names = ["frame_prevalence"] * len(FRAME_LABELS) + ["stance_distribution"] * len(STANCE_LABELS) + ["sentiment_distribution"] * len(SENTIMENT_LABELS)
            for index, (label, measure) in enumerate(zip(labels, measure_names)):
                rows.append(
                    {
                        "analysis_type": "distribution",
                        "platform": platform,
                        "case_window_id": case,
                        "measure": measure,
                        "category": label,
                        "estimate": float((weights * outcomes[:, index]).sum() / denominator),
                        "lower_95": float(max(0.0, lower[index])),
                        "upper_95": float(min(1.0, upper[index])),
                        "unweighted_n": int(len(group)),
                        "effective_sample_size": effective_n,
                        "cluster_count": int(group[cluster_column].astype(str).nunique()),
                        "weighted_total": denominator,
                        "status": "available",
                        "weighting": "Hajek_inverse_inclusion_probability",
                        "notes": "Blank adjudicated frame labels are included as all-negative frame vectors.",
                    }
                )
            cell_summary[f"{platform}|{case}"] = {
                "platform": platform,
                "case_window_id": case,
                "rows": int(len(group)),
                "authors": int(group["author_key"].nunique()),
                "clusters": int(group[cluster_column].astype(str).nunique()),
                "effective_sample_size": effective_n,
            }

    denominator_rows = []
    for (platform, case), group in eligible.groupby(["platform", "case_window_id"], sort=True):
        weights = group["_weight"].to_numpy(dtype=float)
        denominator_rows.append(
            {
                "analysis_type": "denominator",
                "platform": str(platform),
                "case_window_id": str(case),
                "measure": "eligible_documents",
                "category": "all",
                "estimate": np.nan,
                "lower_95": np.nan,
                "upper_95": np.nan,
                "unweighted_n": int(len(group)),
                "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
                "cluster_count": int(group["cluster_id"].astype(str).nunique()),
                "weighted_total": float(weights.sum()),
                "status": "available",
                "weighting": "Hajek_inverse_inclusion_probability",
                "notes": "Strict English, relevant, single valid target, non-empty author and cluster identifiers.",
            }
        )
    return pd.concat([pd.DataFrame(rows), pd.DataFrame(denominator_rows)], ignore_index=True), cell_summary


def _h2_contrasts(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = frame[
        frame["_eligible"]
        & frame["platform"].eq("reddit")
        & frame["adjudicated_target_policy"].eq("AU_SOCIAL_MINIMUM_AGE")
    ].copy()
    earlier = eligible[eligible["case_window_id"].eq("AU_LEGISLATION")]
    later = eligible[eligible["case_window_id"].eq("AU_IMPLEMENTATION")]
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    earlier_clusters = int(earlier["cluster_id"].nunique())
    later_clusters = int(later["cluster_id"].nunique())
    if earlier.empty or later.empty or earlier_clusters < 2 or later_clusters < 2:
        reason = (
            "Target-restricted H2 is unavailable: AU_SOCIAL_MINIMUM_AGE requires "
            f"at least two thread clusters per stage; legislation n={len(earlier)} "
            f"({earlier_clusters} clusters), implementation n={len(later)} ({later_clusters} clusters)."
        )
        for label in ["privacy_surveillance", "circumvention_censorship_autonomy"]:
            rows.append(
                {
                    "analysis_type": "h2_contrast",
                    "platform": "reddit",
                    "case_window_id": "AU_IMPLEMENTATION_MINUS_AU_LEGISLATION",
                    "measure": "risk_difference",
                    "category": label,
                    "estimate": np.nan,
                    "lower_95": np.nan,
                    "upper_95": np.nan,
                    "unweighted_n": int(len(earlier) + len(later)),
                    "effective_sample_size": np.nan,
                    "cluster_count": earlier_clusters + later_clusters,
                    "weighted_total": float(earlier["_weight"].sum() + later["_weight"].sum()),
                    "status": "unavailable",
                    "weighting": "not_estimated_target_restricted_Hajek_thread_cluster_bootstrap",
                    "notes": reason,
                    "earlier_estimate": np.nan,
                    "later_estimate": np.nan,
                    "earlier_n": int(len(earlier)),
                    "later_n": int(len(later)),
                }
            )
            details[label] = {
                "status": "unavailable",
                "reason": reason,
                "target_policy": "AU_SOCIAL_MINIMUM_AGE",
                "earlier_n": int(len(earlier)),
                "later_n": int(len(later)),
                "earlier_clusters": earlier_clusters,
                "later_clusters": later_clusters,
            }
        return pd.DataFrame(rows), {"status": "unavailable", "cells": details}

    for offset, label in enumerate(["privacy_surveillance", "circumvention_censorship_autonomy"]):
        left_outcomes = _frame_matrix(earlier)[:, FRAME_LABELS.index(label)][:, None]
        right_outcomes = _frame_matrix(later)[:, FRAME_LABELS.index(label)][:, None]
        rng = np.random.default_rng(SEED + 100 + offset)
        left_boot, _, _ = _bootstrap_proportions(earlier, left_outcomes, rng)
        right_boot, _, _ = _bootstrap_proportions(later, right_outcomes, rng)
        left_weight = earlier["_weight"].to_numpy(dtype=float)
        right_weight = later["_weight"].to_numpy(dtype=float)
        left_estimate = float((left_weight * left_outcomes[:, 0]).sum() / left_weight.sum())
        right_estimate = float((right_weight * right_outcomes[:, 0]).sum() / right_weight.sum())
        difference = right_estimate - left_estimate
        boot_difference = right_boot[:, 0] - left_boot[:, 0]
        lower, upper = float(np.quantile(boot_difference, 0.025)), float(np.quantile(boot_difference, 0.975))
        status = "descriptive_partial" if upper < 0 or lower > 0 else "inconclusive"
        rows.append(
            {
                "analysis_type": "h2_contrast",
                "platform": "reddit",
                "case_window_id": "AU_IMPLEMENTATION_MINUS_AU_LEGISLATION",
                "measure": "risk_difference",
                "category": label,
                "estimate": difference,
                "lower_95": max(-1.0, lower),
                "upper_95": min(1.0, upper),
                "unweighted_n": int(len(earlier) + len(later)),
                "effective_sample_size": float(np.nan),
                "cluster_count": int(earlier["cluster_id"].nunique() + later["cluster_id"].nunique()),
                "weighted_total": float(left_weight.sum() + right_weight.sum()),
                "status": status,
                "weighting": "Hajek_inverse_inclusion_probability_thread_cluster_bootstrap",
                "notes": f"earlier={left_estimate:.6f}; later={right_estimate:.6f}; replicates={BOOTSTRAP_REPLICATES}; not PRIMARY-6 confirmation.",
                "earlier_estimate": left_estimate,
                "later_estimate": right_estimate,
                "earlier_n": int(len(earlier)),
                "later_n": int(len(later)),
            }
        )
        details[label] = {
            "status": "available",
            "estimate": difference,
            "lower_95": max(-1.0, lower),
            "upper_95": min(1.0, upper),
            "earlier": left_estimate,
            "later": right_estimate,
            "earlier_n": int(len(earlier)),
            "later_n": int(len(later)),
            "earlier_effective_sample_size": float(left_weight.sum() ** 2 / np.square(left_weight).sum()),
            "later_effective_sample_size": float(right_weight.sum() ** 2 / np.square(right_weight).sum()),
        }
    return pd.DataFrame(rows), {"status": "available", "cells": details}


def _jsd(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.sum() <= 0 or right.sum() <= 0:
        raise ValueError("Jensen-Shannon divergence requires positive frame mass")
    left = left / left.sum()
    right = right / right.sum()
    midpoint = (left + right) / 2
    return float(0.5 * np.sum(np.where(left > 0, left * np.log2(left / midpoint), 0)) + 0.5 * np.sum(np.where(right > 0, right * np.log2(right / midpoint), 0)))


def _community_evidence(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = frame[frame["_eligible"]].copy()
    partitions = pd.read_parquet(SOURCE_PATHS["network_partitions"], columns=["graph_id", "scope", "node_id", "node_type", "community_id"])
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for platform, graph_id in (("reddit", "R_ACTOR_ATTENTION"), ("bluesky", "B_ACTOR_ATTENTION")):
        sample = eligible[eligible["platform"].eq(platform)].copy()
        topology = partitions[(partitions["graph_id"].eq(graph_id)) & partitions["node_type"].eq("author")]
        joined = sample.merge(topology, left_on=["author_key", "case_window_id"], right_on=["node_id", "scope"], how="inner")
        platform_details = {"eligible_rows": int(len(sample)), "joined_rows": int(len(joined)), "scopes": {}}
        for scope, group in joined.groupby("case_window_id", sort=True):
            supported = group[group["_frame_count"].gt(0)].copy()
            scope_details = {
                "eligible_rows": int(len(sample[sample["case_window_id"].eq(scope)])),
                "joined_rows": int(len(group)),
                "frame_supported_rows": int(len(supported)),
                "coverage": float(len(group) / max(1, len(sample[sample["case_window_id"].eq(scope)]))),
                "adequate_communities": 0,
                "mean_jsd": None,
                "zero_frame_communities": 0,
                "minimum_frame_supported_rows": H1_MIN_FRAME_SUPPORTED_ROWS,
                "communities_below_frame_support": 0,
            }
            if not group.empty and not supported.empty:
                weights = group["_weight"].to_numpy(dtype=float)
                overall = (weights[:, None] * _frame_matrix(group)).sum(axis=0)
                divergences = []
                divergence_weights = []
                for community, community_group in group.groupby("community_id", sort=True):
                    if len(community_group) < 5:
                        continue
                    frame_supported_rows = int(community_group["_frame_count"].gt(0).sum())
                    if frame_supported_rows < H1_MIN_FRAME_SUPPORTED_ROWS:
                        scope_details["communities_below_frame_support"] += 1
                        continue
                    community_weights = community_group["_weight"].to_numpy(dtype=float)
                    community_profile = (community_weights[:, None] * _frame_matrix(community_group)).sum(axis=0)
                    if community_profile.sum() <= 0:
                        scope_details["zero_frame_communities"] += 1
                        continue
                    divergences.append(_jsd(community_profile, overall))
                    divergence_weights.append(float(community_weights.sum()))
                if divergences:
                    scope_details["adequate_communities"] = len(divergences)
                    scope_details["mean_jsd"] = float(np.average(divergences, weights=divergence_weights))
                    rows.append(
                        {
                            "analysis_type": "community_frame",
                            "platform": platform,
                            "case_window_id": str(scope),
                            "measure": "mean_community_jsd",
                            "category": "adequate_communities",
                            "estimate": scope_details["mean_jsd"],
                            "lower_95": np.nan,
                            "upper_95": np.nan,
                            "unweighted_n": int(len(group)),
                            "effective_sample_size": np.nan,
                            "cluster_count": int(scope_details["adequate_communities"]),
                            "weighted_total": float(sum(divergence_weights)),
                            "status": "descriptive_partial",
                            "weighting": "Hajek_inverse_inclusion_probability_sample_join",
                            "notes": f"joined_rows={len(group)}; frame_supported_rows={len(supported)}; all valid joined rows remain in the population; adequate communities require >=5 joined rows and >=5 frame-supported rows; zero-frame communities have undefined JSD; no permutation null.",
                        }
                    )
            platform_details["scopes"][str(scope)] = scope_details
        details[platform] = platform_details
    return pd.DataFrame(rows), details


def _h3_evidence(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = frame[(frame["_eligible"]) & frame["platform"].eq("reddit")].copy()
    roles = pd.read_parquet(
        SOURCE_PATHS["network_roles"],
        columns=["graph_id", "scope", "node_id", "node_type", "betweenness", "participation_coefficient", "within_module_z", "pagerank_weighted_085"],
    )
    roles = roles[(roles["graph_id"].eq("R_ACTOR_ATTENTION")) & roles["node_type"].eq("author")]
    roles["betweenness"] = pd.to_numeric(roles["betweenness"], errors="coerce")
    thresholds = roles.groupby("scope")["betweenness"].quantile(0.90).rename("broker_threshold")
    roles = roles.join(thresholds, on="scope")
    joined = eligible.merge(roles, left_on=["author_key", "case_window_id"], right_on=["node_id", "scope"], how="inner")
    if joined.empty:
        return pd.DataFrame(), {"status": "unavailable", "reason": "no labelled Reddit authors joined to frozen roles"}
    joined["broker_top_decile"] = joined["betweenness"].ge(joined["broker_threshold"])
    repeated = int((joined.groupby("author_key").size() >= 5).sum())
    broker = joined[joined["broker_top_decile"]]
    other = joined[~joined["broker_top_decile"]]
    adjacent = None
    if not broker.empty and not other.empty:
        adjacent = float(broker["_frame_count"].mean() - other["_frame_count"].mean())
    rows = pd.DataFrame(
        [
            {
                "analysis_type": "h3_adjacent",
                "platform": "reddit",
                "case_window_id": "ALL_CORE_CASES",
                "measure": "brokerage_join_coverage",
                "category": "joined_labelled_rows",
                "estimate": float(len(joined)),
                "lower_95": np.nan,
                "upper_95": np.nan,
                "unweighted_n": int(len(joined)),
                "effective_sample_size": np.nan,
                "cluster_count": int(joined["case_window_id"].nunique()),
                "weighted_total": np.nan,
                "status": "descriptive_partial",
                "weighting": "frozen_structural_roles_join",
                "notes": f"unique_authors={joined['author_key'].nunique()}; authors_with_>=5_docs={repeated}; top_decile_rows={len(broker)}; broker threshold comes from the complete case network; adjacent_frame_count_difference={adjacent}.",
            }
        ]
    )
    return rows, {
        "status": "unavailable",
        "reason": "the planned per-author frame-diversity estimand requires repeated labelled documents; the evaluation sample has no author with >=5 joined documents",
        "joined_rows": int(len(joined)),
        "unique_authors": int(joined["author_key"].nunique()),
        "authors_with_at_least_5_documents": repeated,
        "broker_top_decile_rows": int(len(broker)),
        "non_broker_rows": int(len(other)),
        "adjacent_frame_count_difference": adjacent,
        "broker_thresholds": {str(scope): float(value) for scope, value in thresholds.dropna().items()},
    }


def _h4_evidence(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = frame[(frame["_eligible"]) & frame["platform"].eq("youtube")].copy()
    if eligible.empty:
        return pd.DataFrame(), {"status": "unavailable", "reason": "no eligible YouTube evaluation rows"}
    documents = pd.read_parquet(SOURCE_PATHS["youtube_documents"], columns=["doc_id", "thing", "yt_video_id"])
    videos = documents[documents["thing"].eq("video")][["doc_id", "yt_video_id"]].drop_duplicates()
    seed = pd.read_csv(SOURCE_PATHS["youtube_seed_videos"], keep_default_na=False)[["video_id", "video_type", "event_id"]]
    video_map = videos.merge(seed, left_on="yt_video_id", right_on="video_id", how="left")
    joined = eligible.merge(video_map[["doc_id", "video_type", "event_id"]], left_on="cluster_id", right_on="doc_id", how="left")
    joined = joined[joined["video_type"].astype(str).str.strip().ne("")].copy()
    rows: list[dict[str, Any]] = []
    profiles: dict[str, dict[str, Any]] = {}
    for source_type, group in joined.groupby("video_type", sort=True):
        weights = group["_weight"].to_numpy(dtype=float)
        outcomes = _frame_matrix(group)
        rng = np.random.default_rng(_stable_seed("h4", source_type))
        _, lower, upper = _bootstrap_proportions(group, outcomes, rng, "cluster_id")
        estimate = (weights[:, None] * outcomes).sum(axis=0) / weights.sum()
        for index, label in enumerate(FRAME_LABELS):
            rows.append(
                {
                    "analysis_type": "h4_audience_profile",
                    "platform": "youtube",
                    "case_window_id": "E2_E3",
                    "measure": "audience_frame_prevalence_by_source_type",
                    "category": f"{source_type}:{label}",
                    "estimate": float(estimate[index]),
                    "lower_95": float(max(0.0, lower[index])),
                    "upper_95": float(min(1.0, upper[index])),
                    "unweighted_n": int(len(group)),
                    "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
                    "cluster_count": int(group["cluster_id"].nunique()),
                    "weighted_total": float(weights.sum()),
                    "status": "descriptive_partial",
                    "weighting": "Hajek_inverse_inclusion_probability_video_cluster_bootstrap",
                    "notes": "Audience profile only; no adjudicated video-metadata frame labels were available, so alignment is not estimated.",
                }
            )
            profiles[f"{source_type}:{label}"] = {
                "estimate": float(estimate[index]),
                "lower_95": float(max(0.0, lower[index])),
                "upper_95": float(min(1.0, upper[index])),
                "unweighted_n": int(len(group)),
                "cluster_count": int(group["cluster_id"].nunique()),
                "videos": int(group["cluster_id"].nunique()),
            }
    details = {
        "status": "unavailable",
        "reason": "the existing YouTube metadata audit is absent and no adjudicated video-metadata frame labels are available",
        "joined_rows": int(len(joined)),
        "videos": int(joined["cluster_id"].nunique()),
        "source_types": {str(key): {"rows": int(len(group)), "videos": int(group["cluster_id"].nunique())} for key, group in joined.groupby("video_type")},
        "profiles": profiles,
    }
    return pd.DataFrame(rows), details


TOPIC_NAMES = {
    "bluesky": {
        0: ("Broad age-verification conversation", "Age-verification vocabulary and masked examples are broad or brief; no narrower theme is supported.", "generic_or_discourse_style"),
        1: ("Australian policy link-only posts", "URL-heavy terms and link-only review examples prevent a defensible content interpretation despite Australian-policy tokens.", "artifact_or_contamination"),
        2: ("Australian under-16 social-media ban", "Social-media, Australian, under-16 and children terms consistently identify the central Australian policy debate.", "substantive"),
        3: ("UK Online Safety Act and repeal debate", "Online Safety Act, UK, repeal and petition terms consistently identify enforcement and repeal contestation.", "substantive"),
        4: ("Templated GB News/Ofcom petition spillover", "Repeated GB News, Trump and Ofcom petition text is off-topic templated spillover rather than age-assurance discourse.", "artifact_or_contamination"),
        5: ("UK parliamentary petition mobilisation", "Petition, Parliament, signing and UK terms consistently identify mobilisation around a policy request.", "substantive"),
    },
    "reddit": {
        0: ("General long-form policy discussion", "Function words dominate a broad long-form component; the review examples do not support a narrower policy theme.", "generic_or_discourse_style"),
        1: ("Second-person address and interpersonal reactions", "Second-person terms and short review examples capture conversational address or interpersonal reaction rather than a coherent policy theme.", "generic_or_discourse_style"),
        2: ("Social-media-ban posts with removed-content artifact", "Social-media-ban titles are visible, but the representative post bodies are removed, preventing substantive interpretation.", "artifact_or_contamination"),
        3: ("Short demonstrative and evaluative reactions", "This/is constructions and brief evaluative examples describe a response style, not a stable substantive frame.", "generic_or_discourse_style"),
        4: ("Third-person actor and institutional references", "Third-person pronouns and scattered government terms form a weak actor-reference component without a coherent issue theme.", "generic_or_discourse_style"),
        5: ("Generic evaluative and future-modal language", "Future-modal terms coexist with generic evaluations; masked examples do not support an implementation-effects interpretation.", "generic_or_discourse_style"),
    },
    "youtube": {
        0: ("Broad long-form policy arguments", "Function words dominate long policy arguments spanning several mechanisms; no narrower theme is supported.", "generic_or_discourse_style"),
        1: ("Social-media restrictions and parenting debate", "Social media, children, parents, bans and prescriptive terms consistently identify debate about restrictions and parenting.", "substantive"),
        2: ("Direct-address argumentative replies", "Second-person and conditional terms capture direct argumentative response style rather than one policy issue.", "generic_or_discourse_style"),
        3: ("Parental responsibility versus government protection", "Children, parents, protection and responsibility terms consistently contrast parental and governmental roles.", "substantive"),
        4: ("Child protection versus government control", "Government, protection, censorship and control language consistently contests the stated child-safety rationale.", "substantive"),
        5: ("Age-verification ID and privacy objections", "Age, ID and internet terms plus masked examples consistently identify objections to identity-based verification and privacy intrusion.", "substantive"),
    },
}


def _topic_candidates() -> pd.DataFrame:
    terms = pd.read_csv(SOURCE_PATHS["topic_terms"], keep_default_na=False)
    novelty = pd.read_csv(SOURCE_PATHS["topic_novelty"], keep_default_na=False)
    prevalence = pd.read_csv(SOURCE_PATHS["topic_prevalence"], keep_default_na=False)
    stability = pd.read_csv(SOURCE_PATHS["topic_stability"], keep_default_na=False)
    review = pd.read_csv(SOURCE_PATHS["topic_review_packets"], keep_default_na=False)
    rows: list[dict[str, Any]] = []
    for (platform, topic_id), group in terms.groupby(["platform", "topic_id"], sort=True):
        topic_id = int(topic_id)
        platform = str(platform)
        selected = prevalence[(prevalence["platform"].eq(platform)) & (prevalence["topic_id"].eq(topic_id))]
        topic_stability = stability[(stability["platform"].eq(platform)) & (stability["topic_count"].eq(6))]["mean_top_term_jaccard"].mean()
        novelty_row = novelty[(novelty["platform"].eq(platform)) & (novelty["topic_id"].eq(topic_id))].iloc[0]
        candidate_name, rationale, interpretation_class = TOPIC_NAMES[platform][topic_id]
        if interpretation_class not in TOPIC_INTERPRETATION_CLASSES:
            raise ValueError(f"unknown topic interpretation class: {interpretation_class}")
        shares = pd.to_numeric(selected["share"], errors="coerce")
        scope_documents = pd.to_numeric(selected["documents"], errors="coerce")
        rows.append(
            {
                "platform": platform,
                "topic_id": topic_id,
                "topic_count": 6,
                "candidate_name": candidate_name,
                "naming_rationale": rationale,
                "interpretation_class": interpretation_class,
                "top_terms": "|".join(group.sort_values("rank")["term"].astype(str).head(15)),
                "mean_seed_stability": float(topic_stability),
                "mean_share": float(shares.mean()),
                "scope_share_min": float(shares.min()),
                "scope_share_max": float(shares.max()),
                "scope_share_sd": float(shares.std(ddof=1)) if len(shares) > 1 else 0.0,
                "max_share": float(shares.max()),
                "documents": int(scope_documents.sum()),
                "scope_count": int(selected["scope"].nunique()),
                "min_scope_documents": int(scope_documents.min()),
                "max_scope_documents": int(scope_documents.max()),
                "novelty_status": str(novelty_row["status"]),
                "masked_review_examples": int(review[(review["platform"].eq(platform)) & (review["topic_id"].eq(topic_id))].shape[0]),
                "status": TOPIC_NAMING_STATUS,
                "notes": "Names and interpretation classes use top terms plus three masked review examples per component and were authorized for report use by the project team; this was agent-assisted qualitative review, not independent human frame coding. Scope share range/SD is descriptive spread rather than a confidence interval; minimum support is the smallest observed scope document count; raw review text is not exported.",
            }
        )
    result = pd.DataFrame(rows).sort_values(["platform", "topic_id"]).reset_index(drop=True)
    if len(result) != 18:
        raise AssertionError(f"expected 18 topic candidates, found {len(result)}")
    return result


def _network_inputs() -> dict[str, pd.DataFrame]:
    structure = pd.read_csv(SOURCE_PATHS["network_structure"])
    pagerank = pd.read_csv(SOURCE_PATHS["network_pagerank"])
    roles = pd.read_parquet(SOURCE_PATHS["network_roles"], columns=["graph_id", "scope", "node_type", "participation_coefficient"])
    cascade = pd.read_csv(SOURCE_PATHS["network_cascade"])
    manifest = json.loads(SOURCE_PATHS["network_manifest"].read_text(encoding="utf-8"))
    statuses = {
        str(graph_id): str(record.get("status"))
        for graph_id, record in manifest.get("graph_registry", {}).items()
    }
    return {"structure": structure, "pagerank": pagerank, "roles": roles, "cascade": cascade, "statuses": statuses}


def _network_summaries(networks: dict[str, pd.DataFrame]) -> dict[str, Any]:
    structure = networks["structure"].copy()
    stable = structure[
        structure["community_status"].astype(str).eq("primary_leiden_stable")
        & structure["betweenness_stability_status"].astype(str).eq("stable")
    ].copy()
    eligible_graphs = {
        graph_id
        for graph_id, status in networks_statuses(networks).items()
        if status in {"candidate_network_ready", "audited_ready"}
    }
    stable = stable[stable["graph_id"].isin(eligible_graphs)].copy()
    if stable.empty:
        raise AssertionError("no stable report-eligible topology/network rows available")
    best = stable.sort_values("observed_minus_null_modularity", ascending=False).iloc[0]
    roles = networks["roles"]
    role_summary = []
    for scope, group in roles[(roles["graph_id"].eq("B_ACTOR_ATTENTION")) & roles["node_type"].eq("author")].groupby("scope", sort=True):
        structure_row = stable[stable["scope"].eq(scope)].iloc[0]
        participation = pd.to_numeric(group["participation_coefficient"], errors="coerce").fillna(0.0)
        role_summary.append(
            {
                "graph_id": "B_ACTOR_ATTENTION",
                "scope": str(scope),
                "nodes": int(structure_row["nodes"]),
                "edges": int(structure_row["edges"]),
                "directed": bool(structure_row["directed"]),
                "weighted_edge_total": float(structure_row["weighted_edge_total"]),
                "betweenness_method": str(structure_row["betweenness_method"]),
                "betweenness_top_decile_jaccard": float(structure_row["betweenness_top_decile_jaccard"]),
                "participation_nonzero_share": float((participation > 0).mean()),
                "participation_median": float(participation.median()),
                "participation_p90": float(participation.quantile(0.90)),
            }
        )
    pagerank = networks["pagerank"]
    edge = pagerank[
        pagerank["status"].eq("edge_bootstrap")
        & pagerank["graph_id"].eq("B_ACTOR_ATTENTION")
    ]
    if edge.empty:
        raise AssertionError("PageRank edge-bootstrap stability artifact is empty")
    report_edge = edge[edge["graph_id"].isin(eligible_graphs)]
    cascade = networks["cascade"]
    observed = cascade[cascade["observed_parent_edges"] & cascade["validity_status"].eq("candidate_network_ready")]
    cascade_summary = (
        observed.groupby(["platform", "scope"], as_index=False)
        .agg(
            cascades=("container_id", "size"),
            median_depth=("max_depth", "median"),
            median_breadth=("max_breadth", "median"),
            p90_breadth=("max_breadth", lambda values: values.quantile(0.90)),
            median_structural_virality=("structural_virality", "median"),
        )
    )
    return {
        "stable_structure_rows": int(len(stable)),
        "eligible_structure": stable[["graph_id", "scope", "nodes", "edges", "directed", "weighted_edge_total"]].to_dict(orient="records"),
        "network_definition": {
            "graph_id": "B_ACTOR_ATTENTION",
            "direction": "replier_to_replied_to_author",
            "weighting": "observed reply multiplicity",
            "filters": "phrase-exact roots and observed descendants; valid endpoints; self-loops and non-exact hits excluded",
        },
        "role_summary": role_summary,
        "strongest_structure": {
            "graph_id": str(best["graph_id"]),
            "scope": str(best["scope"]),
            "nodes": int(best["nodes"]),
            "edges": int(best["edges"]),
            "directed": bool(best["directed"]),
            "weighted_edge_total": float(best["weighted_edge_total"]),
            "communities": int(best["communities"]),
            "modularity": float(best["community_modularity"]),
            "null_median_modularity": float(best["null_median_modularity"]),
            "observed_minus_null_modularity": float(best["observed_minus_null_modularity"]),
            "median_ari": float(best["community_median_ari"]),
            "betweenness_method": str(best["betweenness_method"]),
        },
        "report_eligible_pagerank_edge_bootstrap": {
            "graphs": sorted(report_edge["graph_id"].unique()),
            "rank_correlation_min": float(report_edge["rank_correlation"].min()),
            "rank_correlation_max": float(report_edge["rank_correlation"].max()),
            "top_100_jaccard_min": float(report_edge["top_k_jaccard"].min()),
            "top_100_jaccard_max": float(report_edge["top_k_jaccard"].max()),
            "scopes": int(report_edge["scope"].nunique()),
        },
        "graph_statuses": networks_statuses(networks),
        "cascade_summary": cascade_summary.to_dict(orient="records"),
    }


def networks_statuses(networks: dict[str, pd.DataFrame]) -> dict[str, str]:
    return dict(networks.get("statuses", {}))


def _hypothesis_evidence(
    h2: dict[str, Any],
    h1: dict[str, Any],
    h3: dict[str, Any],
    h4: dict[str, Any],
) -> pd.DataFrame:
    questions = {
        "H1-R": "Do Reddit interaction communities contain more frame concentration than expected under a structure-preserving frame permutation baseline?",
        "H1-B": "Do Bluesky interaction communities contain more frame concentration than expected under a structure-preserving frame permutation baseline?",
        "H2-P": "Within the Australian case, does privacy/surveillance discourse occupy a larger share around implementation than legislation?",
        "H2-C": "Within the Australian case, does circumvention/autonomy discourse occupy a larger share around implementation than legislation?",
        "H3": "Do high-betweenness Reddit participants engage with a wider range of frames than activity-matched low-betweenness participants?",
        "H4": "Do YouTube comment sections partly reproduce the dominant frame signalled by their video's title and description?",
    }
    rows: list[dict[str, Any]] = []
    for endpoint in ("H1-R", "H1-B"):
        platform = "reddit" if endpoint == "H1-R" else "bluesky"
        platform_details = h1.get(platform, {})
        scopes = platform_details.get("scopes", {})
        adequate = sum(int(value.get("adequate_communities", 0)) for value in scopes.values())
        joined = sum(int(value.get("joined_rows", 0)) for value in scopes.values())
        estimates = [value["mean_jsd"] for value in scopes.values() if value.get("mean_jsd") is not None]
        adjacent_estimate = float(np.average(estimates)) if estimates else None
        rows.append(
            {
                "endpoint": endpoint,
                "original_question": questions[endpoint],
                "evidence_available": f"joined_labelled_rows={joined}; adequate_communities={adequate}; adjacent_descriptive_mean_jsd={adjacent_estimate}",
                "estimate": np.nan,
                "interval_lower_95": np.nan,
                "interval_upper_95": np.nan,
                "interval": "not estimated",
                "status": "unavailable",
                "exact_limitation": "Only sparse primary evaluation labels joined to topology-only communities; no full-corpus human frame labels and no 9,999 author-vector permutation null were run.",
                "report_safe_wording": "The pre-specified H1 test is unavailable because the author-vector permutation null was not run. Any joined-sample JSD is separately labelled descriptive evidence, not an inconclusive H1 result.",
            }
        )
    for endpoint, label in (("H2-P", "privacy_surveillance"), ("H2-C", "circumvention_censorship_autonomy")):
        result = h2["cells"][label]
        if result.get("status") == "unavailable":
            rows.append(
                {
                    "endpoint": endpoint,
                    "original_question": questions[endpoint],
                    "evidence_available": f"target_policy={result['target_policy']}; legislation_n={result['earlier_n']}; implementation_n={result['later_n']}; legislation_clusters={result['earlier_clusters']}; implementation_clusters={result['later_clusters']}",
                    "estimate": np.nan,
                    "interval_lower_95": np.nan,
                    "interval_upper_95": np.nan,
                    "interval": "not estimated",
                    "status": "unavailable",
                    "exact_limitation": result["reason"],
                    "report_safe_wording": "After restricting the comparison to AU_SOCIAL_MINIMUM_AGE records, the Australian stage contrast is unavailable because the implementation cell has too few labelled records and clusters; this is an evidence gap, not a substantive null effect.",
                }
            )
            continue
        rows.append(
            {
                "endpoint": endpoint,
                "original_question": questions[endpoint],
                "evidence_available": f"primary_evaluation_rows={result['earlier_n'] + result['later_n']}; weighted_Hajek_risk_difference; thread_cluster_bootstrap_replicates={BOOTSTRAP_REPLICATES}",
                "estimate": result["estimate"],
                "interval_lower_95": result["lower_95"],
                "interval_upper_95": result["upper_95"],
                "interval": f"[{result['lower_95']:.3f}, {result['upper_95']:.3f}]",
                "status": "inconclusive" if result["lower_95"] <= 0 <= result["upper_95"] else "descriptive_partial",
                "exact_limitation": "The evaluation fallback is not formal PRIMARY-6 confirmation; eligible AU cells are small and effective sample sizes are low, so the cluster-bootstrap intervals are wide.",
                "report_safe_wording": f"The fallback estimate is {result['estimate']:+.3f} with a 95% cluster-bootstrap interval of [{result['lower_95']:.3f}, {result['upper_95']:.3f}]; this is inconclusive descriptive evidence, not confirmation.",
            }
        )
    rows.append(
        {
            "endpoint": "H3",
            "original_question": questions["H3"],
            "evidence_available": f"joined_labelled_rows={h3.get('joined_rows', 0)}; unique_authors={h3.get('unique_authors', 0)}; broker_adjacent_frame_count_difference={h3.get('adjacent_frame_count_difference')}",
            "estimate": np.nan,
            "interval_lower_95": np.nan,
            "interval_upper_95": np.nan,
            "interval": "not estimated",
            "status": "unavailable",
            "exact_limitation": h3.get("reason", "The planned repeated-document author estimand is unavailable."),
            "report_safe_wording": "Frozen roles can be joined to a small labelled subset, but the planned repeated-document broker/frame-diversity estimand is unavailable; any broker comparison is H3-adjacent descriptive evidence only.",
        }
    )
    rows.append(
        {
            "endpoint": "H4",
            "original_question": questions["H4"],
            "evidence_available": f"selected_video_source_type_audience_profiles={h4.get('joined_rows', 0)} labelled comments; source_types={','.join(sorted(h4.get('source_types', {})))}",
            "estimate": np.nan,
            "interval_lower_95": np.nan,
            "interval_upper_95": np.nan,
            "interval": "not estimated",
            "status": "unavailable",
            "exact_limitation": h4.get("reason", "Adjudicated video-metadata frame labels are unavailable."),
            "report_safe_wording": "Selected-video audience frame profiles differ by source type descriptively, but metadata–audience alignment cannot be estimated without adjudicated video-metadata frame labels.",
        }
    )
    result = pd.DataFrame(rows)
    if set(result["endpoint"]) != set(ENDPOINTS) or len(result) != len(ENDPOINTS):
        raise AssertionError("hypothesis evidence must contain exactly one row per required endpoint")
    return result


def _finding_cards(
    estimates: pd.DataFrame,
    network: dict[str, Any],
    topics: pd.DataFrame,
    h4: dict[str, Any],
) -> pd.DataFrame:
    strongest = network["strongest_structure"]
    rank = network["report_eligible_pagerank_edge_bootstrap"]
    scope_details = "; ".join(
        f"{row['scope']}: {int(row['nodes']):,} nodes/{int(row['edges']):,} directed weighted reply edges (weight={float(row['weighted_edge_total']):,.0f})"
        for row in network["eligible_structure"]
    )
    role_details = "; ".join(
        f"{row['scope']}: betweenness top-decile Jaccard={row['betweenness_top_decile_jaccard']:.3f}, nonzero cross-community participation={row['participation_nonzero_share']:.3f}"
        for row in network["role_summary"]
    )
    topic_stability = topics.groupby("platform")["mean_seed_stability"].first().to_dict()
    generic_share = topics[topics["topic_id"].eq(0)].groupby("platform")["max_share"].first().to_dict()
    topic_class_counts = topics["interpretation_class"].value_counts().to_dict()
    review_examples_min = int(topics["masked_review_examples"].min())
    review_examples_max = int(topics["masked_review_examples"].max())
    frame_rows = estimates[
        estimates["analysis_type"].eq("distribution")
        & estimates["case_window_id"].eq("ALL_CASES")
        & estimates["measure"].eq("frame_prevalence")
    ].copy()

    def frame_profile(platform: str, labels: list[str]) -> str:
        selected = frame_rows[frame_rows["platform"].eq(platform)].set_index("category")
        values = ", ".join(
            f"{label}={selected.loc[label, 'estimate']:.3f} [{selected.loc[label, 'lower_95']:.3f}, {selected.loc[label, 'upper_95']:.3f}]"
            for label in labels
        )
        first = selected.iloc[0]
        return f"{platform}: {values} (n={int(first['unweighted_n'])}, ESS={first['effective_sample_size']:.1f})"

    frame_numbers = "; ".join(
        [
            frame_profile("bluesky", ["policy_assurance", "circumvention_censorship_autonomy"]),
            frame_profile("reddit", ["policy_assurance", "circumvention_censorship_autonomy"]),
            frame_profile("youtube", ["policy_assurance", "child_safety", "circumvention_censorship_autonomy"]),
        ]
    )
    bertopic = pd.read_csv(SOURCE_PATHS["bertopic_sensitivity"], keep_default_na=False)
    bertopic_runs = (
        bertopic.groupby(["platform", "seed"], as_index=False)
        .agg(components=("topic_id", "nunique"), median_documents=("documents", "median"), mean_seed_stability=("mean_seed_stability", "first"))
    )
    bertopic_summary = "; ".join(
        f"{platform}={int(group['components'].min())}–{int(group['components'].max())} components/run (median n={group['median_documents'].median():.1f}, mean seed stability={group['mean_seed_stability'].iloc[0]:.3f})"
        for platform, group in bertopic_runs.groupby("platform", sort=True)
    )

    def h4_profile(source_type: str, label: str) -> str:
        profile = h4.get("profiles", {}).get(f"{source_type}:{label}")
        if profile is None:
            return f"{source_type} {label}=not available"
        return f"{source_type} {label}={profile['estimate']:.3f} [{profile['lower_95']:.3f}, {profile['upper_95']:.3f}] (n={profile['unweighted_n']}, {profile['videos']} videos)"

    h4_numbers = "; ".join(
        h4_profile(source_type, label)
        for source_type in sorted(h4.get("source_types", {}))
        for label in ["privacy_surveillance", "child_safety", "circumvention_censorship_autonomy"]
    )
    cards = [
        {
            "finding_id": "F1",
            "claim": "Within-platform human-coded estimates show policy assurance as the largest point estimate in the Bluesky and Reddit samples, while the YouTube sample is distributed across policy assurance, child safety and circumvention/autonomy.",
            "exact_supporting_numbers": frame_numbers,
            "evidence_type": "Non-exclusive adjudicated human frame labels with normalized Hajek inverse-inclusion weighting and platform-appropriate cluster-bootstrap intervals.",
            "figure_or_table_source": "data/analysis/fallback/human_sample_estimates.csv;data/analysis/fallback/figures/weighted_frame_prevalence.png",
            "interpretation": "The observed sample supports different within-platform frame profiles; because frames are non-exclusive and no direct platform contrast was tested, it does not rank platforms or establish between-platform differences.",
            "limitation": "Frame exact-set agreement is below 0.80 in every split, and evaluation/reserve include per-frame agreement or kappa failures against the predeclared thresholds. Reddit's effective sample size is only 7.3 and several intervals are wide. Estimates rely on adjudicated evaluation labels and are not platform-wide prevalence.",
            "stakeholder_implication": "Use the observed profiles to pre-register platform-specific message-pilot hypotheses, then measure comprehension and rights-risk outcomes rather than assuming the platforms differ.",
            "measurable_indicator_of_success": "Each pilot reports its platform-specific denominator, comprehension, unresolved safety/privacy questions and circumvention confusion, with contrasts and stopping thresholds declared before results are opened.",
            "reconsideration_trigger": "If direct pilot contrasts do not reproduce the descriptive profile or any variant raises rights-risk complaints, withdraw the tailoring claim and use a common safeguarded explanation.",
            "evidential_status": "descriptive_human_sample",
        },
        {
            "finding_id": "F2",
            "claim": "The strongest frozen topology slice shows community structure well above a degree-preserving null.",
            "exact_supporting_numbers": f"{strongest['graph_id']} / {strongest['scope']}: {strongest['nodes']:,} nodes, {strongest['edges']:,} directed weighted reply edges (weight={strongest['weighted_edge_total']:,.0f}), {strongest['communities']} Leiden communities, modularity={strongest['modularity']:.3f} versus null median={strongest['null_median_modularity']:.3f}, excess={strongest['observed_minus_null_modularity']:.3f}, median ARI={strongest['median_ari']:.3f}; {role_details}; filter=phrase-exact roots and observed descendants with valid endpoints, non-exact hits/self-loops excluded.",
            "evidence_type": "Frozen topology-only network structure with replier-to-replied-to direction, weighted reply multiplicity, degree-preserving weighted null and 20-seed Leiden stability.",
            "figure_or_table_source": "data/analysis/networks/structure/structure_summary.csv;data/analysis/fallback/figures/community_null_comparison.png",
            "interpretation": "Observed reply structure is strongly modular relative to the null in this platform-event slice, so arguments are not arranged as one homogeneous conversation.",
            "limitation": "This is a bounded candidate-network slice, not a complete platform graph. Community membership is structural position, not ideology, residence, persuasion or causal influence; human frame coverage is sparse.",
            "stakeholder_implication": "Platform trust-and-safety and public-interest communications teams should first test separate age-assurance explanations for structurally distinct communities, with privacy and child-safety safeguards visible in each version.",
            "measurable_indicator_of_success": "A pilot reports comprehension, unresolved privacy/safety questions, and response coverage separately for every adequately observed community, with no community below the pre-registered coverage threshold.",
            "reconsideration_trigger": "If one community shows materially higher privacy complaints, exclusion, or misunderstanding than the aggregate, withdraw the one-size-fits-all message and commission a rights review.",
            "evidential_status": "directly_observed_topology",
        },
        {
            "finding_id": "F3",
            "claim": "Bluesky received-attention rankings are stable under frozen edge-bootstrap perturbations.",
            "exact_supporting_numbers": f"{scope_details}; across {rank['scopes']} Bluesky actor-attention scopes, edge-bootstrap rank correlation ranges {rank['rank_correlation_min']:.3f}–{rank['rank_correlation_max']:.3f}, while top-100 Jaccard ranges {rank['top_100_jaccard_min']:.3f}–{rank['top_100_jaccard_max']:.3f}; direction=replier-to-replied-to, filter=phrase-exact roots/observed descendants with valid endpoints and self-loop exclusion.",
            "evidence_type": "PageRank stability on directed replier-to-replied-to-author graphs, using phrase-exact roots, observed descendants, strict valid edges, endpoint/self-loop filtering and five edge-bootstrap replicates per scope.",
            "figure_or_table_source": "data/analysis/networks/structure/pagerank_stability.csv;data/analysis/fallback/figures/pagerank_stability.png",
            "interpretation": "The received-attention ordering is reproducible across edge perturbations, but it remains a structural attention measure rather than persuasion or influence.",
            "limitation": "The Bluesky graph is a bounded phrase-exact descendant sample; historical search completeness and platform affordances constrain generalisation.",
            "stakeholder_implication": "Platform research and communications leads should first prohibit individual targeting or enforcement decisions based on these rankings and use only aggregate, scope-limited structural summaries.",
            "measurable_indicator_of_success": "An independent review finds that every public centrality claim states edge direction, weight, scope, and stability, and that no individual-level intervention cites PageRank as influence.",
            "reconsideration_trigger": "If rankings become unstable under the recorded perturbation or are used to target named accounts, remove the claim from decision workflows and run a privacy/harms review.",
            "evidential_status": "directly_observed_topology",
        },
        {
            "finding_id": "F4",
            "claim": "Seed stability did not guarantee semantic validity: only seven of 18 NMF components were substantively interpretable.",
            "exact_supporting_numbers": f"Interpretation review: substantive={topic_class_counts.get('substantive', 0)}/18, generic/discourse-style={topic_class_counts.get('generic_or_discourse_style', 0)}/18, artifact/contamination={topic_class_counts.get('artifact_or_contamination', 0)}/18; masked examples per component={review_examples_min}–{review_examples_max}. Mean six-topic seed stability is Bluesky={topic_stability.get('bluesky', np.nan):.3f}, Reddit={topic_stability.get('reddit', np.nan):.3f}, YouTube={topic_stability.get('youtube', np.nan):.3f}; generic topic 0 reaches {min(generic_share.values()):.3f}–{max(generic_share.values()):.3f} of documents by platform/scope. BERTopic sensitivity is highly fragmented: {bertopic_summary}.",
            "evidence_type": "Exploratory NMF with three seeds and masked-document interpretation, plus an independently generated BERTopic fragmentation sensitivity.",
            "figure_or_table_source": "data/analysis/fallback/topic_name_candidates.csv;data/analysis/topics/topic_prevalence.csv;data/analysis/topics/bertopic_sensitivity.csv;data/analysis/topics/topics_manifest.json;data/analysis/fallback/figures/topic_prevalence_stability.png",
            "interpretation": "The factorisation is reproducible, but 11 components primarily encode background language, interaction style or data artifacts. Only the seven substantive components should support thematic interpretation.",
            "limitation": "Topic classification used three masked examples per component and agent-assisted qualitative judgement. BERTopic produced hundreds of small components per run with only moderate cross-seed stability, so it is fragmentation sensitivity rather than one-to-one confirmation; these topics are not independently human-coded frames.",
            "stakeholder_implication": "Policy and communications analysts should use only the seven substantively interpretable components for exploratory question design and exclude generic, stylistic and artifact components from policy claims.",
            "measurable_indicator_of_success": "Every topic claim in the report maps to a substantive component and cites its terms, masked-example review, prevalence support and exploratory status; none of the 11 excluded components is narrated as a policy theme.",
            "reconsideration_trigger": "If an independent analyst cannot reproduce a substantive interpretation from the masked examples and terms, downgrade that component to generic or artifact and remove the associated topic claim.",
            "evidential_status": "exploratory_partial",
        },
        {
            "finding_id": "F5",
            "claim": "Selected YouTube audiences show source-type-specific frame profiles, but the original metadata–audience alignment estimand is unavailable.",
            "exact_supporting_numbers": h4_numbers,
            "evidence_type": "Primary evaluation human labels joined to selected-video source metadata; weighted audience profiles only.",
            "figure_or_table_source": "data/analysis/fallback/human_sample_estimates.csv",
            "interpretation": "Audience frame mixtures vary by selected source type, which supports source-aware description but not a claim that audiences reproduce video metadata framing.",
            "limitation": "No adjudicated video-metadata frame labels exist; the tech-explainer cell is especially small and the videos are purposively selected.",
            "stakeholder_implication": "Platform education and policy teams should first separate child-safety, privacy, and circumvention explanations by video source type, without claiming that audience frames reproduce video metadata.",
            "measurable_indicator_of_success": "A source-type pilot tracks comprehension, privacy complaints, circumvention confusion, and child-safety response coverage separately for each source type and reports the denominator and uncertainty.",
            "reconsideration_trigger": "If a source type produces a material rise in privacy complaints, exclusion, or circumvention misunderstanding, stop that message variant and refer it for rights and safety review.",
            "evidential_status": "descriptive_partial",
        },
    ]
    result = pd.DataFrame(cards)
    if len(result) != 5:
        raise AssertionError("fallback finding bundle must contain five cards")
    return result


def _write_figures(
    flow: pd.DataFrame,
    estimates: pd.DataFrame,
    networks: dict[str, pd.DataFrame],
    topics: pd.DataFrame,
) -> list[Path]:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    (FIGURES_ROOT / "h2_forest_plot.png").unlink(missing_ok=True)
    paths: list[Path] = []

    flow_view = flow[flow["analysis_type"].eq("eligibility_flow")].copy()
    flow_view["label"] = flow_view["measure"].map(
        {
            "primary_evaluation_records": "Primary evaluation records",
            "required_adjudicated_fields_complete": "Required adjudicated fields complete",
            "relevant_policy_discourse": "Relevant policy discourse",
            "strict_english": "Strict English",
            "single_valid_target_policy": "Single valid target policy",
            "author_identifier_present": "Author identifier present",
            "cluster_identifier_present": "Cluster identifier present",
        }
    ).fillna(flow_view["measure"].str.replace("_", " ").str.title())
    primary_count = int(flow_view.loc[flow_view["measure"].eq("primary_evaluation_records"), "unweighted_n"].iloc[0])
    eligible_count = int(flow_view["unweighted_n"].iloc[-1])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(flow_view["label"].iloc[::-1], flow_view["unweighted_n"].iloc[::-1], color="#3b82f6")
    ax.set_xlabel("Rows remaining")
    ax.set_title(f"Primary evaluation sample: eligibility narrows from {primary_count:,} to {eligible_count:,}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = FIGURES_ROOT / "sample_eligibility_flow.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    frame = estimates[(estimates["analysis_type"].eq("distribution")) & (estimates["case_window_id"].eq("ALL_CASES")) & estimates["measure"].eq("frame_prevalence")].copy()
    order = {label: index for index, label in enumerate(FRAME_LABELS)}
    frame["order"] = frame["category"].map(order)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    positions = np.arange(len(FRAME_LABELS))
    for index, (platform, group) in enumerate(frame.groupby("platform", sort=True)):
        group = group.set_index("category").loc[FRAME_LABELS].reset_index()
        x = positions + (index - 1) * 0.24
        ax.errorbar(x, group["estimate"], yerr=[group["estimate"] - group["lower_95"], group["upper_95"] - group["estimate"]], fmt="o", capsize=3, label=platform)
    ax.set_xticks(positions, [label.replace("_", "\n") for label in FRAME_LABELS], fontsize=8)
    ax.set_ylabel("Weighted prevalence")
    ax.set_ylim(0, 1)
    ax.set_title("Human-coded non-exclusive frame prevalence by platform (95% cluster intervals)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURES_ROOT / "weighted_frame_prevalence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    structure = networks["structure"].copy()
    stable = structure[
        structure["community_status"].astype(str).eq("primary_leiden_stable")
        & structure["betweenness_stability_status"].astype(str).eq("stable")
        & structure["graph_id"].eq("B_ACTOR_ATTENTION")
    ].copy()
    stable["label"] = stable["graph_id"].map({"R_ACTOR_ATTENTION": "Reddit reply", "B_ACTOR_ATTENTION": "Bluesky reply", "Y_ACTOR_ATTENTION_HANDLE": "YouTube reply", "Y_COMMENTER_VIDEO": "YouTube audience", "R_AUTHOR_SUBREDDIT": "Reddit subreddit"}) + "\n" + stable["scope"]
    stable = stable.sort_values("observed_minus_null_modularity")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(stable["label"], stable["observed_minus_null_modularity"], color="#0f766e")
    ax.axvline(0, color="#444", linewidth=1)
    ax.set_xlabel("Observed modularity minus degree-preserving null median")
    ax.set_title("Stable topology slices retain modular structure beyond the null")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = FIGURES_ROOT / "community_null_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    edge = networks["pagerank"][
        networks["pagerank"]["status"].eq("edge_bootstrap")
        & networks["pagerank"]["graph_id"].eq("B_ACTOR_ATTENTION")
    ].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    for scope, group in edge.groupby("scope", sort=True):
        ax.scatter(group["rank_correlation"], group["top_k_jaccard"], s=60, alpha=0.8, label=f"Bluesky reply {scope}")
    ax.set_xlabel("PageRank rank correlation (zoomed)")
    ax.set_ylabel("Top-100 Jaccard")
    ax.set_xlim(max(0.0, float(edge["rank_correlation"].min()) - 0.002), 1.0005)
    ax.set_ylim(max(0.0, float(edge["top_k_jaccard"].min()) - 0.04), min(1.0, float(edge["top_k_jaccard"].max()) + 0.04))
    ax.axvline(1.0, color="#6b7280", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_title("Received-attention rankings: global ordering vs top-100 overlap")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = FIGURES_ROOT / "pagerank_stability.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    topic_colors = {
        "substantive": "#7c3aed",
        "generic_or_discourse_style": "#6b7280",
        "artifact_or_contamination": "#d97706",
    }
    fig, axes = plt.subplots(1, 3, figsize=(19, 9), sharey=False)
    for ax, (platform, group) in zip(axes, topics.groupby("platform", sort=True)):
        group = group.sort_values("mean_share", ascending=True)
        labels = [f"T{int(topic)} {name}" for topic, name in zip(group["topic_id"], group["candidate_name"])]
        positions = np.arange(len(group))
        colors = group["interpretation_class"].map(topic_colors)
        ax.barh(labels, group["mean_share"], color=colors, alpha=0.86)
        ax.errorbar(
            group["mean_share"],
            positions,
            xerr=[group["mean_share"] - group["scope_share_min"], group["scope_share_max"] - group["mean_share"]],
            fmt="none",
            ecolor="#374151",
            capsize=3,
            linewidth=1,
        )
        for position, maximum_share, minimum_documents in zip(positions, group["scope_share_max"], group["min_scope_documents"]):
            ax.text(float(maximum_share) + 0.012, position, f"min n={int(minimum_documents):,}", va="center", fontsize=7)
        ax.set_title(f"{platform}\nseed stability={group['mean_seed_stability'].iloc[0]:.3f}")
        ax.set_xlabel("Mean scope share")
        ax.set_xlim(0, max(0.85, float(group["scope_share_max"].max()) + 0.12))
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_ylabel("Reviewed NMF component")
    legend = [
        Patch(facecolor=topic_colors["substantive"], label="Substantive"),
        Patch(facecolor=topic_colors["generic_or_discourse_style"], label="Generic or discourse style"),
        Patch(facecolor=topic_colors["artifact_or_contamination"], label="Artifact or contamination"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Only 7 of 18 seed-stable NMF components are substantively interpretable", y=1.02)
    fig.tight_layout()
    path = FIGURES_ROOT / "topic_prevalence_stability.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    if len(paths) > 6:
        raise AssertionError("fallback bundle must contain no more than six figures")
    return paths


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _assert_report_safe(path: Path) -> None:
    forbidden = ("text", "title", "body", "context", "author_key", "cluster_id", "node_id", "doc_id", "video_id", "channel_id", "url", "uri", "hash", "record_id", "source_id", "target_id")
    if path.suffix == ".csv":
        columns = list(pd.read_csv(path, nrows=0).columns)
        unsafe = [column for column in columns if any(token in column.casefold() for token in forbidden)]
        if unsafe:
            raise AssertionError(f"report-facing output contains restricted identifier/text columns: {path}: {unsafe}")


def _output_records(paths: list[Path]) -> dict[str, dict[str, Any]]:
    records = {}
    for path in paths:
        records[_relative(path)] = {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        if path.suffix == ".csv":
            records[_relative(path)]["rows"] = int(pd.read_csv(path).shape[0])
    return records


def _validate_upstream_network_artifacts() -> dict[str, dict[str, Any]]:
    """Require every upstream graph layer to be current before using fallback evidence."""

    return {
        "network": check_networks(),
        "structure": check_structure(),
        "temporal": check_temporal(),
    }


def build() -> dict[str, Any]:
    upstream_validators = _validate_upstream_network_artifacts()
    FALLBACK_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    topic_validation = check_topics()
    labels = _load_evaluation()
    flow = _flow_rows(labels)
    reliability_frames: list[pd.DataFrame] = []
    reliability_manifest: dict[str, Any] = {}
    reliability_splits = {split: _load_label_split(split) for split in VALIDATION_SPLITS}
    for split, split_frame in reliability_splits.items():
        split_payload: dict[str, Any] = {"split_role": RELIABILITY_ROLES[split], "platforms": {}}
        for platform in ["all", "reddit", "youtube", "bluesky"]:
            subset = split_frame if platform == "all" else split_frame[split_frame["platform"].eq(platform)]
            platform_payload: dict[str, Any] = {}
            for field in ["relevance", "language", "target_policy", "stance", "sentiment", "frame_labels"]:
                report = reliability_report(subset, field, multilabel=field == "frame_labels")
                platform_payload[field] = report
                if report.get("status") != "available":
                    continue
                base = {
                    "analysis_type": "reliability",
                    "platform": platform,
                    "split": split,
                    "case_window_id": f"ALL_{split.upper()}",
                    "measure": field,
                    "category": "all",
                    "lower_95": np.nan,
                    "upper_95": np.nan,
                    "unweighted_n": int(report["usable_rows"]),
                    "effective_sample_size": np.nan,
                    "cluster_count": np.nan,
                    "weighted_total": np.nan,
                    "status": "available",
                    "weighting": "unweighted_coder_agreement",
                    "notes": f"Split role={RELIABILITY_ROLES[split]}; agreement is before adjudication; blank frame labels are treated as no supported frame.",
                }
                reliability_frames.append(pd.DataFrame([{**base, "category": "all", "statistic": "percent_agreement", "estimate": report.get("raw_agreement", np.nan)}, {**base, "category": "all", "statistic": "cohen_kappa", "estimate": report.get("cohen_kappa", np.nan)}]))
                if field == "frame_labels":
                    a_sets = subset["coder_a_frame_labels"].map(_frame_set)
                    b_sets = subset["coder_b_frame_labels"].map(_frame_set)
                    jaccard = [1.0 if not left and not right else len(left & right) / len(left | right) for left, right in zip(a_sets, b_sets)]
                    reliability_frames.append(pd.DataFrame([{**base, "category": "all", "statistic": "mean_multilabel_jaccard", "estimate": float(np.mean(jaccard))}]))
                    for item in report.get("per_frame_agreement", []):
                        reliability_frames.append(pd.DataFrame([{**base, "category": item["label"], "statistic": "percent_agreement", "estimate": item["raw_agreement"]}, {**base, "category": item["label"], "statistic": "cohen_kappa", "estimate": item["cohen_kappa"]}]))
            split_payload["platforms"][platform] = platform_payload
        reliability_manifest[split] = split_payload

    distributions, denominator_cells = _distribution_rows(labels)
    h2_frame, h2_details = _h2_contrasts(labels)
    community_rows, community_details = _community_evidence(labels)
    h3_rows, h3_details = _h3_evidence(labels)
    h4_rows, h4_details = _h4_evidence(labels)
    estimates = pd.concat([flow, *reliability_frames, distributions, h2_frame, community_rows, h3_rows, h4_rows], ignore_index=True, sort=False)
    estimates = estimates.sort_values(["analysis_type", "platform", "case_window_id", "measure", "category"], na_position="last").reset_index(drop=True)

    networks = _network_inputs()
    network_details = _network_summaries(networks)
    topics = _topic_candidates()
    h1_details = community_details
    hypothesis = _hypothesis_evidence(h2_details, h1_details, h3_details, h4_details)
    figures = _write_figures(flow, estimates, networks, topics)
    cards = _finding_cards(estimates, network_details, topics, h4_details)

    human_path = FALLBACK_ROOT / "human_sample_estimates.csv"
    hypothesis_path = FALLBACK_ROOT / "hypothesis_evidence.csv"
    cards_path = FALLBACK_ROOT / "finding_cards.csv"
    topics_path = FALLBACK_ROOT / "topic_name_candidates.csv"
    _write_csv(human_path, estimates)
    _write_csv(hypothesis_path, hypothesis)
    _write_csv(cards_path, cards)
    _write_csv(topics_path, topics)

    for path in [human_path, hypothesis_path, cards_path, topics_path]:
        _assert_report_safe(path)
    expected_source_paths = set()
    for source_cell in cards["figure_or_table_source"].astype(str):
        expected_source_paths.update(part.strip() for part in source_cell.split(";") if part.strip())
    missing_sources = [path for path in sorted(expected_source_paths) if not (REPO_ROOT / path).exists()]
    if missing_sources:
        raise AssertionError(f"finding card references missing output: {missing_sources}")
    h2_intervals = h2_frame["lower_95"].notna() & h2_frame["upper_95"].notna()
    if not h2_frame.loc[h2_intervals, "lower_95"].between(-1, 1).all() or not h2_frame.loc[h2_intervals, "upper_95"].between(-1, 1).all():
        raise AssertionError("H2 confidence intervals must remain within [-1, 1]")
    bounded = estimates[estimates["measure"].isin(["frame_prevalence", "stance_distribution", "sentiment_distribution", "audience_frame_prevalence_by_source_type"])]
    bounded_intervals = bounded["lower_95"].notna() & bounded["upper_95"].notna()
    if not bounded.loc[bounded_intervals, "lower_95"].between(0, 1).all() or not bounded.loc[bounded_intervals, "upper_95"].between(0, 1).all():
        raise AssertionError("prevalence confidence intervals must remain within [0, 1]")

    output_paths = [human_path, hypothesis_path, cards_path, topics_path, *figures]
    manifest_path = FALLBACK_ROOT / "analysis_manifest.json"
    manifest = {
        "schema_version": FALLBACK_SCHEMA_VERSION,
        "analysis_config": _analysis_config(),
        "analysis_code_version": ANALYSIS_CODE_VERSION,
        "analysis_code_path": _relative(_analysis_code_path()),
        "analysis_code_sha256": _analysis_code_sha256(),
        "analysis_code_checksums": _analysis_code_checksums(),
        "created_at_utc": _now(),
        "command": ".venv/bin/python -m src.analysis.fallback_analysis --build",
        "primary_input": _relative(SOURCE_PATHS["evaluation_labels"]),
        "source_registry": _source_registry(),
        "primary_evaluation_records": PRIMARY_RECORDS,
        "eligible_records": int(labels["_eligible"].sum()),
        "eligible_by_platform": labels[labels["_eligible"]].groupby("platform").size().astype(int).to_dict(),
        "eligible_by_case": labels[labels["_eligible"]].groupby("case_window_id").size().astype(int).to_dict(),
        "estimand": "strict-English relevant single-valid-target human evaluation records with normalized Hajek inverse-inclusion weighting",
        "blank_frame_policy": "blank adjudicated_frame_labels are valid all-negative frame vectors when other required fields are complete",
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": SEED, "clusters": {"reddit": "cluster_id/thread", "youtube": "cluster_id/video", "bluesky": "cluster_id/root_thread"}},
        "reliability_before_adjudication": reliability_manifest,
        "h2": h2_details,
        "h1": h1_details,
        "h3": h3_details,
        "h4": h4_details,
        "network": network_details,
        "topic_summary": {
            "topics": int(len(topics)),
            "status": TOPIC_NAMING_STATUS,
            "interpretation_class_counts": topics["interpretation_class"].value_counts().astype(int).to_dict(),
        },
        "topic_validation": topic_validation,
        "upstream_validators": upstream_validators,
        "hypothesis_statuses": hypothesis.set_index("endpoint")["status"].to_dict(),
        "finding_ids": cards["finding_id"].tolist(),
        "excluded_actions": [
            "no scraping or relabelling",
            "no measurement_cycle.json",
            "no model selection, held-out evaluation, reserve activation, full-corpus automated classification or PRIMARY-6 execution",
            "no overwrite of data/analysis/report/age_gate_paradox_report.md",
        ],
        "report_safety": {"raw_text_exported": False, "direct_identifiers_exported": False, "topic_review_text_exported": False},
        "source_checksums": _source_checksums(),
        "outputs": _output_records(output_paths),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["outputs"][_relative(manifest_path)] = {"path": _relative(manifest_path), "sha256": _sha256(manifest_path), "bytes": manifest_path.stat().st_size}
    return manifest


def check() -> dict[str, Any]:
    upstream_validators = _validate_upstream_network_artifacts()
    check_topics()
    manifest_path = FALLBACK_ROOT / "analysis_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"fallback manifest is missing: {manifest_path}; run --build first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("upstream_validators") != upstream_validators:
        raise AssertionError("fallback upstream validator results changed; rebuild required")
    if manifest.get("primary_evaluation_records") != PRIMARY_RECORDS:
        raise AssertionError("fallback manifest primary sample count is not 600")
    _validate_analysis_freshness(manifest)
    if manifest.get("source_registry") != _source_registry():
        raise AssertionError("fallback source registry changed; rebuild required")
    current_source_checksums = _source_checksums()
    if manifest.get("source_checksums") != current_source_checksums:
        raise AssertionError("fallback source checksums changed; rebuild required")
    labels = _load_evaluation()
    if int(labels["_eligible"].sum()) != int(manifest["eligible_records"]):
        raise AssertionError("fallback eligible denominator changed")
    for path_text, record in manifest.get("outputs", {}).items():
        path = REPO_ROOT / path_text
        if not path.exists():
            raise FileNotFoundError(f"fallback output is missing: {path}")
        if path_text != _relative(manifest_path) and _sha256(path) != record.get("sha256"):
            raise AssertionError(f"fallback output checksum changed: {path_text}")
        if path.suffix == ".csv":
            _assert_report_safe(path)
    hypothesis = pd.read_csv(FALLBACK_ROOT / "hypothesis_evidence.csv", keep_default_na=False)
    if set(hypothesis["endpoint"]) != set(ENDPOINTS) or len(hypothesis) != len(ENDPOINTS):
        raise AssertionError("fallback hypothesis evidence endpoint contract failed")
    h1_rows = hypothesis[hypothesis["endpoint"].isin(["H1-R", "H1-B"])]
    if not h1_rows["status"].eq("unavailable").all():
        raise AssertionError("confirmatory H1 must remain unavailable without its permutation test")
    cards = pd.read_csv(FALLBACK_ROOT / "finding_cards.csv", keep_default_na=False)
    if len(cards) != 5:
        raise AssertionError("fallback finding card count changed")
    topics = pd.read_csv(FALLBACK_ROOT / "topic_name_candidates.csv", keep_default_na=False)
    if len(topics) != 18 or not topics["status"].eq(TOPIC_NAMING_STATUS).all():
        raise AssertionError("fallback topic naming contract failed")
    if set(topics["interpretation_class"]) != TOPIC_INTERPRETATION_CLASSES:
        raise AssertionError("fallback topic interpretation classes changed")
    figure_paths = list(FIGURES_ROOT.glob("*.png"))
    if len(figure_paths) > 6:
        raise AssertionError("fallback figure count exceeds six")
    return {
        "status": "valid",
        "primary_evaluation_records": PRIMARY_RECORDS,
        "eligible_records": int(labels["_eligible"].sum()),
        "outputs": len(manifest.get("outputs", {})),
        "figures": len(figure_paths),
        "hypothesis_statuses": manifest.get("hypothesis_statuses", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true", help="generate the frozen-data fallback evidence bundle")
    group.add_argument("--check", action="store_true", help="validate the existing fallback bundle without writing")
    args = parser.parse_args()
    result = build() if args.build else check()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
