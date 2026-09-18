"""Frozen PRIMARY-6 estimands and fail-closed confirmatory hypothesis gates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.analysis.artifacts import ANALYSIS_ROOT, REPO_ROOT, read_json, relative_path, sha256_file, utc_now_iso, write_json
from src.analysis.codebook import FRAME_DEFINITIONS
from src.analysis.measurement import (
    EVALUATION_REPORT,
    RESERVE_ASSESSMENT,
    RESERVE_ACTIVATION,
    VALIDATION_ROOT,
    SENTIMENT_SELECTION,
    STANCE_FRAME_SELECTION,
    _frame_evaluation_mask,
    _frame_pipeline,
    _selected_frame_thresholds,
    _validate_frame_threshold_values,
    check_reserve_assessment,
    check_selection,
    _stored_selection,
)
from src.analysis.networks import NETWORK_MANIFEST, NETWORK_ROOT, YOUTUBE_METADATA_AUDIT, canonical_youtube_video_id, frozen_youtube_video_ids, validate_reddit_relevance_audit, validate_youtube_metadata_audit, youtube_video_event_assignments
from src.analysis.predictions import PREDICTIONS_MANIFEST
from src.analysis.populations import POPULATION_ARTIFACT
from src.analysis.structure import STRUCTURE_MANIFEST, check_structure
from src.analysis.validation import load_label_packet
from src.shared.case_windows import REDDIT_CASE_BY_EVENT
from src.shared.masking import mask_text

HYPOTHESIS_ROOT = ANALYSIS_ROOT / "hypotheses"
HYPOTHESIS_MANIFEST = HYPOTHESIS_ROOT / "hypothesis_manifest.json"
PERMUTATIONS = 9_999
POWER_SIMULATION_SEED = 20260915
POWER_SIMULATION_REPLICATES = 2_000
HYPOTHESIS_SCHEMA_VERSION = "hypothesis.v4"
PRIMARY_FAMILY = "PRIMARY-6"
PRIMARY_ENDPOINTS = ("H1-R", "H1-B", "H2-P", "H2-C", "H3", "H4")
PRIMARY_EFFECT_THRESHOLDS = {"H1-R": 0.05, "H1-B": 0.05, "H2-P": 0.05, "H2-C": 0.05, "H3": 0.10, "H4": 0.05}
FRAME_LABELS = [frame["label"] for frame in FRAME_DEFINITIONS]

HYPOTHESIS_SPECS: dict[str, dict[str, Any]] = {
    "H1": {
        "endpoints": ["H1-R", "H1-B"],
        "population_ids": ["R_REPLY_CORE", "B_REPLY_CORE"],
        "estimand": "activity-weighted mean community-to-platform Jensen-Shannon divergence minus median author-vector permutation null",
        "null": "shuffle complete author frame vectors within case/event x degree x activity bins; fixed topology and partition",
        "threshold": "excess divergence >= 0.05 and Holm-adjusted one-sided p < 0.05",
        "status_requirements": ["human_evaluation_labels", "reddit_thread_relevance_audit", "topology_partition"],
    },
    "H2": {
        "endpoints": ["H2-P", "H2-C"],
        "population_ids": ["R_AU_LEGISLATION", "R_AU_IMPLEMENTATION"],
        "estimand": "author-balanced implementation-minus-legislation risk difference for the named v2 frame",
        "null": "one-sided null-boundary test from 9,999 thread-cluster bootstrap draws with author-balanced weights; no event relabelling",
        "threshold": "Holm-adjusted one-sided p < 0.05, lower 95% interval > 0 and difference >= 0.05",
        "status_requirements": ["human_evaluation_labels", "reddit_thread_relevance_audit"],
    },
    "H3": {
        "endpoints": ["H3"],
        "population_ids": ["R_BROKER_CORE"],
        "estimand": "mean normalized five-frame entropy difference between matched top-decile-betweenness brokers and below-median comparison authors",
        "null": "9,999 broker-label permutations within frozen 1:3 matched sets; bootstrap matched sets for uncertainty",
        "threshold": "Holm-adjusted one-sided p < 0.05, lower 95% interval > 0 and difference >= 0.10",
        "status_requirements": ["human_evaluation_labels", "reddit_thread_relevance_audit", "topology_partition"],
    },
    "H4": {
        "endpoints": ["H4"],
        "population_ids": ["Y_VIDEO_E2E3"],
        "estimand": "event-adjusted news-minus-commentary difference in 1 - Jensen-Shannon metadata/audience frame alignment",
        "null": "9,999 channel-level source-label permutations with channel-cluster bootstrap uncertainty",
        "threshold": "Holm-adjusted two-sided p < 0.05, interval excludes 0 and absolute difference >= 0.05",
        "status_requirements": ["human_evaluation_labels", "youtube_video_metadata_audit", "topology_edges"],
    },
}


def js_divergence(left: np.ndarray | list[float], right: np.ndarray | list[float]) -> float:
    """Return Jensen-Shannon divergence in bits for two probability vectors."""

    p = np.asarray(left, dtype=float)
    q = np.asarray(right, dtype=float)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("probability vectors must have the same one-dimensional shape")
    if np.any(p < 0) or np.any(q < 0) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("probability vectors must be finite and non-negative")
    p = p / p.sum() if p.sum() else np.zeros_like(p)
    q = q / q.sum() if q.sum() else np.zeros_like(q)
    midpoint = (p + q) / 2
    def terms(values: np.ndarray) -> np.ndarray:
        result = np.zeros_like(values)
        positive = values > 0
        result[positive] = values[positive] * np.log2(values[positive] / midpoint[positive])
        return result
    return float((terms(p).sum() + terms(q).sum()) / 2)


def activity_weighted_segregation(
    community_profiles: dict[str, np.ndarray], platform_profile: np.ndarray, activities: dict[str, float]
) -> float:
    if not community_profiles:
        raise ValueError("at least one community profile is required")
    total_activity = sum(float(value) for value in activities.values())
    if total_activity <= 0:
        raise ValueError("community activity must have a positive total")
    return float(
        sum(
            float(activities.get(community, 0.0)) / total_activity * js_divergence(profile, platform_profile)
            for community, profile in community_profiles.items()
        )
    )


def author_balanced_difference(frame: pd.DataFrame, author_column: str, case_column: str, label_column: str, later: str, earlier: str) -> float:
    usable = frame[author_column].fillna("").astype(str).ne("") & frame[case_column].isin([later, earlier])
    values = frame.loc[usable].copy()
    if values.empty:
        raise ValueError("no usable documents for author-balanced difference")
    per_author = values.groupby([case_column, author_column], sort=True)[label_column].mean().reset_index()
    means = per_author.groupby(case_column)[label_column].mean()
    if later not in means or earlier not in means:
        raise ValueError("both case windows need at least one author")
    return float(means[later] - means[earlier])


def normalized_frame_entropy(profile: np.ndarray) -> float:
    values = np.asarray(profile, dtype=float)
    if values.ndim != 1 or len(values) < 2 or np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("profile must be a finite non-negative one-dimensional vector")
    total = values.sum()
    if not total:
        return 0.0
    probabilities = values / total
    entropy = -float(np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0).sum())
    return entropy / math.log(len(values))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, retaining endpoint names."""

    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for position, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - position) * float(value)))
        adjusted[name] = running
    return adjusted


def _normalised_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("frame profiles must be a finite non-negative matrix")
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("each frame profile must have positive mass")
    return values / totals


def _sparse_strata(
    frame: pd.DataFrame,
    degree_column: str,
    activity_column: str,
    case_column: str,
    minimum_size: int = 20,
) -> pd.Series:
    degree = pd.to_numeric(frame[degree_column], errors="raise").rank(method="first")
    activity = pd.to_numeric(frame[activity_column], errors="raise").rank(method="first")
    bins = max(1, min(4, len(frame)))
    degree_bin = ((degree - 1) * bins // max(len(frame), 1)).astype(int)
    activity_bin = ((activity - 1) * bins // max(len(frame), 1)).astype(int)
    cases = frame[case_column].astype(str)
    result = pd.Series([f"{case}|{left}|{right}" for case, left, right in zip(cases, degree_bin, activity_bin)], index=frame.index)
    counts = result.value_counts()
    terminal_buckets: set[str] = set()
    while len(counts) > 1:
        sparse_candidates = [str(value) for value in counts.index if int(counts[value]) < minimum_size and str(value) not in terminal_buckets]
        if not sparse_candidates:
            break
        sparse = min(sparse_candidates, key=lambda value: (int(counts[value]), value))
        sparse_case, sparse_degree, sparse_activity = sparse.split("|")
        choices = [str(value) for value in counts.index if str(value) != sparse and str(value).split("|")[0] == sparse_case]
        if not choices:
            merged = f"{sparse_case}|merged|merged"
            result.loc[result.eq(sparse)] = merged
            terminal_buckets.add(merged)
            counts = result.value_counts()
            continue
        def distance(value: str) -> tuple[int, int, str]:
            _, target_degree, target_activity = value.split("|")
            if target_degree == "merged":
                return (-1, 0, value)
            return (0, abs(int(target_degree) - int(sparse_degree)) + abs(int(target_activity) - int(sparse_activity)), value)

        target = min(choices, key=distance)
        result.loc[result.eq(sparse)] = target
        counts = result.value_counts()
    return result


def _segregation_statistic(
    frame: pd.DataFrame,
    profiles: np.ndarray,
    community_column: str,
    case_column: str,
    activity_column: str,
) -> float:
    values = frame.reset_index(drop=True)
    profiles = _normalised_rows(profiles)
    activities = pd.to_numeric(values[activity_column], errors="raise").to_numpy(dtype=float)
    if np.any(activities <= 0):
        raise ValueError("activity must be positive")
    case_values = values[case_column].astype(str).to_numpy()
    case_statistics = []
    case_weights = []
    for case in sorted(set(case_values)):
        indexes = np.flatnonzero(case_values == case)
        case_activity = activities[indexes]
        platform_profile = np.average(profiles[indexes], axis=0, weights=case_activity)
        community_values = values.iloc[indexes][community_column].astype(str).to_numpy()
        weighted_divergence = 0.0
        for community in sorted(set(community_values)):
            community_indexes = indexes[community_values == community]
            community_activity = activities[community_indexes]
            community_profile = np.average(profiles[community_indexes], axis=0, weights=community_activity)
            weighted_divergence += community_activity.sum() / case_activity.sum() * js_divergence(community_profile, platform_profile)
        case_statistics.append(weighted_divergence)
        case_weights.append(case_activity.sum())
    return float(np.average(case_statistics, weights=case_weights) if case_statistics else 0.0)


def author_vector_permutation_test(
    frame: pd.DataFrame,
    profile_columns: list[str],
    *,
    community_column: str = "community_id",
    case_column: str = "case",
    degree_column: str = "degree",
    activity_column: str = "activity",
    document_count_column: str = "documents",
    min_documents: int = 3,
    permutations: int = PERMUTATIONS,
    seed: int = 20260915,
) -> dict[str, Any]:
    required = [community_column, case_column, degree_column, activity_column, document_count_column, *profile_columns]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"H1 input missing columns: {missing}")
    usable = frame[pd.to_numeric(frame[document_count_column], errors="coerce").ge(min_documents)].copy().reset_index(drop=True)
    if usable.empty:
        raise ValueError("H1 has no authors meeting the minimum document rule")
    profiles = usable[profile_columns].to_numpy(dtype=float)
    strata = _sparse_strata(usable, degree_column, activity_column, case_column)
    observed = _segregation_statistic(usable, profiles, community_column, case_column, activity_column)
    rng = np.random.default_rng(seed)
    null_values = np.empty(permutations, dtype=float)
    for replicate in range(permutations):
        shuffled = profiles.copy()
        for stratum in sorted(strata.unique()):
            indexes = np.flatnonzero(strata.to_numpy() == stratum)
            shuffled[indexes] = profiles[rng.permutation(indexes)]
        null_values[replicate] = _segregation_statistic(usable, shuffled, community_column, case_column, activity_column)
    null_median = float(np.median(null_values))
    return {
        "observed": observed,
        "null_median": null_median,
        "excess": observed - null_median,
        "p_value": float((1 + np.count_nonzero(null_values >= observed)) / (permutations + 1)),
        "permutations": int(permutations),
        "seed": seed,
        "eligible_authors": int(len(usable)),
        "excluded_authors": int(len(frame) - len(usable)),
        "stratum_counts": {str(key): int(value) for key, value in strata.value_counts().sort_index().items()},
        "profile_columns": profile_columns,
    }


def _author_balanced_prevalence(frame: pd.DataFrame, author_column: str, label_column: str) -> float:
    usable = frame[author_column].fillna("").astype(str).ne("")
    if not usable.any():
        raise ValueError("no authors available for author-balanced prevalence")
    return float(frame.loc[usable].groupby(author_column, sort=True)[label_column].mean().mean())


def _document_weighted_prevalence(frame: pd.DataFrame, label_column: str) -> float:
    values = pd.to_numeric(frame[label_column], errors="raise")
    if values.empty:
        raise ValueError("no documents available for document-weighted prevalence")
    return float(values.mean())


def _resample_cluster_occurrences(
    frame: pd.DataFrame, author_column: str, cluster_column: str, rng: np.random.Generator
) -> pd.DataFrame:
    clusters = np.asarray(sorted(frame[cluster_column].astype(str).unique()))
    if not len(clusters):
        raise ValueError("H2 case has no clusters")
    sampled = rng.choice(clusters, size=len(clusters), replace=True)
    occurrences = []
    for occurrence, cluster in enumerate(sampled):
        rows = frame[frame[cluster_column].astype(str).eq(cluster)].copy()
        authors = rows[author_column].fillna("").astype(str)
        rows["_bootstrap_author"] = authors.where(authors.eq(""), authors + f"|{occurrence}")
        occurrences.append(rows)
    return pd.concat(occurrences, ignore_index=True)


def thread_cluster_bootstrap_difference(
    frame: pd.DataFrame,
    outcome_column: str,
    later: str,
    earlier: str,
    *,
    author_column: str = "author",
    case_column: str = "case",
    cluster_column: str = "thread",
    weighting: str = "author_balanced",
    replicates: int = PERMUTATIONS,
    seed: int = 20260915,
) -> dict[str, Any]:
    usable = frame[frame[case_column].isin([later, earlier])].copy()
    if usable.empty or outcome_column not in usable or cluster_column not in usable:
        raise ValueError("H2 requires both case and thread-clustered outcomes")
    if weighting not in {"author_balanced", "document_weighted", "unweighted"}:
        raise ValueError("H2 weighting must be author_balanced, document_weighted, or unweighted")
    prevalence = _author_balanced_prevalence if weighting == "author_balanced" else _document_weighted_prevalence
    observed = (prevalence(usable[usable[case_column].eq(later)], author_column, outcome_column) if weighting == "author_balanced" else prevalence(usable[usable[case_column].eq(later)], outcome_column)) - (prevalence(usable[usable[case_column].eq(earlier)], author_column, outcome_column) if weighting == "author_balanced" else prevalence(usable[usable[case_column].eq(earlier)], outcome_column))
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates, dtype=float)
    for index in range(replicates):
        samples = []
        for case in (later, earlier):
            case_frame = usable[usable[case_column].eq(case)]
            try:
                samples.append(_resample_cluster_occurrences(case_frame, author_column, cluster_column, rng))
            except ValueError as error:
                raise ValueError(f"H2 case has no clusters: {case}") from error
        later_sample, earlier_sample = samples
        bootstrap_author = "_bootstrap_author" if weighting == "author_balanced" else author_column
        boot[index] = (prevalence(later_sample, bootstrap_author, outcome_column) if weighting == "author_balanced" else prevalence(later_sample, outcome_column)) - (prevalence(earlier_sample, bootstrap_author, outcome_column) if weighting == "author_balanced" else prevalence(earlier_sample, outcome_column))
    null_boundary = 0.0
    # The confidence distribution is centred on the observed statistic.  The
    # basic-bootstrap tail at 2*observed is the one-sided null-boundary test;
    # using boot <= 0 would incorrectly reuse the confidence distribution as a
    # null distribution.
    null_threshold = 2 * observed - null_boundary
    return {
        "observed": observed,
        "lower_95": float(np.quantile(boot, 0.025)),
        "median": float(np.quantile(boot, 0.5)),
        "upper_95": float(np.quantile(boot, 0.975)),
        "p_value": float((1 + np.count_nonzero(boot >= null_threshold)) / (replicates + 1)),
        "p_value_method": "one_sided_basic_cluster_bootstrap_null_boundary",
        "null_boundary": null_boundary,
        "null_threshold": null_threshold,
        "replicates": int(replicates),
        "seed": seed,
        "author_column": author_column,
        "cluster_column": cluster_column,
        "weighting": weighting,
    }


def _standardised_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    pooled = math.sqrt((float(np.var(left)) + float(np.var(right))) / 2)
    return abs(float(np.mean(left) - np.mean(right)) / pooled) if pooled else 0.0


def match_broker_sets(
    frame: pd.DataFrame,
    *,
    case_column: str = "case",
    author_column: str = "author",
    betweenness_column: str = "betweenness",
    document_count_column: str = "documents",
    degree_column: str = "degree",
    subreddit_column: str = "subreddit_breadth",
    outcome_column: str = "entropy",
    broker_quantile: float = 0.90,
    comparison_quantile: float = 0.50,
    matching: str = "1_to_3_mahalanobis_calipers",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {case_column, author_column, betweenness_column, document_count_column, degree_column, subreddit_column, outcome_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"H3 input missing columns: {missing}")
    if not 0 < broker_quantile < 1 or not 0 < comparison_quantile < 1 or matching != "1_to_3_mahalanobis_calipers":
        raise ValueError("H3 matching specification is unsupported")
    matched_rows: list[pd.Series] = []
    unmatched_brokers = 0
    unmatched_controls = 0
    set_index = 0
    for case, group in frame.groupby(case_column, sort=True):
        group = group.drop_duplicates(author_column).copy()
        if len(group) < 4:
            unmatched_brokers += int(len(group))
            continue
        broker_rank = group[betweenness_column].rank(method="first", pct=True)
        broker_mask = broker_rank > broker_quantile
        control_mask = group[betweenness_column] < group[betweenness_column].quantile(comparison_quantile)
        covariates = pd.DataFrame({
            "log_documents": np.log1p(pd.to_numeric(group[document_count_column], errors="raise")),
            "log_degree": np.log1p(pd.to_numeric(group[degree_column], errors="raise")),
            "subreddit_breadth": pd.to_numeric(group[subreddit_column], errors="raise"),
        }, index=group.index)
        scale = covariates.std(ddof=0).replace(0, 1.0)
        standardised = (covariates - covariates.mean()) / scale
        covariance = np.cov(standardised.to_numpy().T) if len(group) > 1 else np.eye(3)
        inverse = np.linalg.pinv(np.atleast_2d(covariance))
        unused = set(group.index[control_mask])
        for broker_index in group.index[broker_mask]:
            candidates = []
            for control_index in sorted(unused):
                if abs(standardised.loc[broker_index, "log_documents"] - standardised.loc[control_index, "log_documents"]) > 0.20:
                    continue
                if abs(standardised.loc[broker_index, "log_degree"] - standardised.loc[control_index, "log_degree"]) > 0.20:
                    continue
                if abs(covariates.loc[broker_index, "subreddit_breadth"] - covariates.loc[control_index, "subreddit_breadth"]) > 1:
                    continue
                delta = (standardised.loc[broker_index] - standardised.loc[control_index]).to_numpy(dtype=float)
                candidates.append((float(delta @ inverse @ delta), control_index))
            if len(candidates) < 3:
                unmatched_brokers += 1
                continue
            selected = [control for _, control in sorted(candidates)[:3]]
            unused.difference_update(selected)
            set_id = f"match-{set_index:04d}"
            set_index += 1
            broker = group.loc[broker_index].copy()
            broker["match_set"] = set_id
            broker["broker_label"] = 1
            matched_rows.append(broker)
            for control_index in selected:
                control = group.loc[control_index].copy()
                control["match_set"] = set_id
                control["broker_label"] = 0
                matched_rows.append(control)
        unmatched_controls += len(unused)
    if not matched_rows:
        return pd.DataFrame(columns=[*frame.columns, "match_set", "broker_label"]), {"matched_sets": 0, "unmatched_brokers": unmatched_brokers, "unmatched_controls": unmatched_controls, "balance": {}, "balance_passed": False, "broker_quantile": broker_quantile, "comparison_quantile": comparison_quantile, "matching": matching}
    matched = pd.DataFrame(matched_rows).reset_index(drop=True)
    balance = {}
    for column in (document_count_column, degree_column, subreddit_column):
        transformed = np.log1p(pd.to_numeric(matched[column], errors="raise")) if column != subreddit_column else pd.to_numeric(matched[column], errors="raise")
        balance[column] = _standardised_mean_difference(transformed[matched["broker_label"].eq(1)], transformed[matched["broker_label"].eq(0)])
    return matched, {"matched_sets": int(matched["match_set"].nunique()), "unmatched_brokers": unmatched_brokers, "unmatched_controls": unmatched_controls, "balance": balance, "balance_passed": all(value < 0.10 for value in balance.values()), "broker_quantile": broker_quantile, "comparison_quantile": comparison_quantile, "matching": matching}


def broker_permutation_test(
    matched: pd.DataFrame,
    *,
    outcome_column: str = "entropy",
    set_column: str = "match_set",
    label_column: str = "broker_label",
    permutations: int = PERMUTATIONS,
    seed: int = 20260915,
) -> dict[str, Any]:
    if matched.empty or not {outcome_column, set_column, label_column} <= set(matched.columns):
        raise ValueError("H3 requires matched sets and an entropy outcome")
    sets = [group for _, group in matched.groupby(set_column, sort=True) if len(group) == 4]
    if not sets:
        raise ValueError("H3 has no complete 1:3 matched sets")
    observed_values = [float(group.loc[group[label_column].eq(1), outcome_column].mean() - group.loc[group[label_column].eq(0), outcome_column].mean()) for group in sets]
    observed = float(np.mean(observed_values))
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        differences = []
        for group in sets:
            chosen = int(rng.integers(len(group)))
            values = pd.to_numeric(group[outcome_column], errors="raise").to_numpy(dtype=float)
            differences.append(float(values[chosen] - np.delete(values, chosen).mean()))
        null[index] = float(np.mean(differences))
    boot = np.empty(permutations, dtype=float)
    for index in range(permutations):
        chosen_sets = rng.integers(len(sets), size=len(sets))
        boot[index] = float(np.mean([observed_values[chosen] for chosen in chosen_sets]))
    return {"observed": observed, "lower_95": float(np.quantile(boot, 0.025)), "median": float(np.median(boot)), "upper_95": float(np.quantile(boot, 0.975)), "p_value": float((1 + np.count_nonzero(null >= observed)) / (permutations + 1)), "permutations": int(permutations), "bootstrap_replicates": int(permutations), "matched_sets": len(sets), "seed": seed}


def _event_adjusted_difference(frame: pd.DataFrame, source_column: str, alignment_column: str, event_column: str) -> float:
    values = frame.copy()
    values[source_column] = values[source_column].astype(str).str.casefold()
    values[event_column] = values[event_column].astype(str)
    source = values[source_column]
    events = sorted(values[event_column].unique())
    if len(events) < 2 or any(set(source[values[event_column].eq(event)].unique()) != {"news", "commentary"} for event in events):
        raise ValueError("H4 needs news and commentary videos in both event cells")
    design = np.column_stack([np.ones(len(values)), source.eq("news").astype(float), *[(values[event_column].eq(event)).astype(float) for event in events[1:]]])
    return float(np.linalg.lstsq(design, pd.to_numeric(values[alignment_column], errors="raise"), rcond=None)[0][1])


def video_alignment_contrast(
    frame: pd.DataFrame,
    metadata_profile_columns: list[str],
    audience_profile_columns: list[str],
    *,
    video_column: str = "video_id",
    event_column: str = "event",
    channel_column: str = "channel_id",
    source_column: str = "source_type",
    permutations: int = PERMUTATIONS,
    seed: int = 20260915,
) -> dict[str, Any]:
    required = {video_column, event_column, channel_column, source_column, *metadata_profile_columns, *audience_profile_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"H4 input missing columns: {missing}")
    values = frame.copy().reset_index(drop=True)
    if values[video_column].astype(str).duplicated().any():
        raise ValueError("H4 requires one metadata/audience profile row per video")
    metadata = _normalised_rows(values[metadata_profile_columns].to_numpy(dtype=float))
    audience = _normalised_rows(values[audience_profile_columns].to_numpy(dtype=float))
    values["alignment"] = [1.0 - js_divergence(left, right) for left, right in zip(metadata, audience)]
    observed = _event_adjusted_difference(values, source_column, "alignment", event_column)
    channel_values = values.assign(_channel=values[channel_column].astype(str), _event=values[event_column].astype(str), _source=values[source_column].astype(str).str.casefold())
    channel_source = channel_values.groupby("_channel", sort=True)["_source"].nunique()
    if channel_source.gt(1).any():
        raise ValueError("H4 source type must be constant within a channel")
    channel_table = channel_values.groupby("_channel", sort=True).agg(
        source=("_source", "first"),
        event_signature=("_event", lambda values: tuple(sorted(set(values)))),
    )
    if not set(channel_table["source"]) <= {"news", "commentary"}:
        raise ValueError("H4 source type must be news or commentary for the inferential subset")
    channels_by_signature = {
        signature: list(group.index)
        for signature, group in channel_table.groupby("event_signature", sort=True)
    }
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        permuted = values.copy()
        lookup: dict[str, str] = {}
        for channels in channels_by_signature.values():
            labels = channel_table.loc[channels, "source"].to_numpy()
            shuffled = rng.permutation(labels)
            lookup.update(dict(zip(channels, shuffled)))
        permuted[source_column] = permuted[channel_column].astype(str).map(lookup)
        null[index] = _event_adjusted_difference(permuted, source_column, "alignment", event_column)
    boot = np.empty(permutations, dtype=float)
    channel_rows = {channel: group.drop(columns=["_channel", "_event", "_source"]) for channel, group in channel_values.groupby("_channel", sort=True)}
    channels_by_source = {
        source_type: np.asarray(sorted(channel_table.index[channel_table["source"].eq(source_type)]))
        for source_type in ("news", "commentary")
    }
    for index in range(permutations):
        for attempt in range(1000):
            sampled_channels = np.concatenate([
                rng.choice(channels, size=len(channels), replace=True)
                for channels in channels_by_source.values()
            ])
            candidate = pd.concat([channel_rows[channel] for channel in sampled_channels], ignore_index=True)
            try:
                boot[index] = _event_adjusted_difference(candidate, source_column, "alignment", event_column)
                break
            except ValueError:
                if attempt == 999:
                    raise ValueError("H4 channel bootstrap could not preserve both source types in every event")
    return {
        "observed": observed,
        "lower_95": float(np.quantile(boot, 0.025)),
        "median": float(np.median(boot)),
        "upper_95": float(np.quantile(boot, 0.975)),
        "p_value": float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (permutations + 1)),
        "p_value_method": "two_sided_whole_channel_permutation",
        "bootstrap_method": "whole_channel_source_stratified_bootstrap",
        "permutations": int(permutations),
        "bootstrap_replicates": int(permutations),
        "video_count": int(len(values)),
        "channel_count": int(values[channel_column].nunique()),
        "alignment_column": "alignment",
    }


def power_mde_diagnostic(
    effective_sample_size: float,
    outcome_sd: float,
    minimum_effect: float,
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
    alternative: str = "one-sided",
) -> dict[str, Any]:
    if effective_sample_size <= 0 or outcome_sd <= 0 or not 0 < alpha < 1 or not 0 < target_power < 1 or alternative not in {"one-sided", "two-sided"}:
        raise ValueError("power inputs must be positive and probabilities must be in (0, 1)")
    critical = norm.ppf(1 - alpha if alternative == "one-sided" else 1 - alpha / 2)
    standard_error = outcome_sd / math.sqrt(effective_sample_size)
    mde = (critical + norm.ppf(target_power)) * standard_error
    noncentrality = minimum_effect / standard_error
    power = 1 - norm.cdf(critical - noncentrality)
    if alternative == "two-sided":
        power += norm.cdf(-critical - noncentrality)
    return {"effective_sample_size": effective_sample_size, "outcome_sd": outcome_sd, "standard_error": standard_error, "minimum_effect": minimum_effect, "mde": float(mde), "power_at_minimum_effect": float(power), "target_power": target_power, "alternative": alternative}


def _h1_power_segregation(profiles: np.ndarray, communities: np.ndarray, cases: np.ndarray, activities: np.ndarray) -> float:
    profiles = _normalised_rows(profiles)
    values = []
    weights = []
    for case in np.unique(cases):
        indexes = np.flatnonzero(cases == case)
        case_activity = activities[indexes]
        platform_profile = np.average(profiles[indexes], axis=0, weights=case_activity)
        divergence = 0.0
        for community in np.unique(communities[indexes]):
            community_indexes = indexes[communities[indexes] == community]
            community_activity = activities[community_indexes]
            divergence += community_activity.sum() / case_activity.sum() * js_divergence(
                np.average(profiles[community_indexes], axis=0, weights=community_activity), platform_profile
            )
        values.append(divergence)
        weights.append(case_activity.sum())
    return float(np.average(values, weights=weights))


def _permutation_z(observed: float, null_values: np.ndarray) -> float:
    scale = float(np.std(null_values, ddof=1)) if len(null_values) > 1 else 0.0
    if scale <= 1e-12:
        return 0.0 if abs(observed) <= 1e-12 else math.copysign(1e6, observed)
    return float(observed / scale)


def _h1_power_statistic(
    sizes: np.ndarray,
    topology: dict[str, float],
    counts: dict[str, int],
    effect: float,
    rng: np.random.Generator,
) -> float:
    return _h1_power_draw(sizes, topology, counts, effect, rng)[0]


def _h1_power_draw(
    sizes: np.ndarray,
    topology: dict[str, float],
    counts: dict[str, int],
    effect: float,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Keep the simulation bounded while retaining the frozen activity weights
    # and the complete H1 permutation statistic.
    count = max(4, min(len(sizes), int(topology["nodes"]), 256))
    activity = np.resize(sizes, count).astype(float)
    case_count = max(2, min(len(counts), count // 2, 3))
    cases = np.arange(count) % case_count
    communities = (np.arange(count) // case_count) % 2
    profiles = rng.dirichlet(np.ones(len(FRAME_LABELS)), size=count)
    profiles[communities == 1, 0] += effect
    profiles /= profiles.sum(axis=1, keepdims=True)
    degree_mean = max(1.0, 2.0 * float(topology["edges"]) / max(float(topology["nodes"]), 1.0))
    degrees = np.maximum(1, rng.poisson(degree_mean, count))
    strata = _sparse_strata(
        pd.DataFrame({"case": cases, "degree": degrees, "activity": activity}),
        "degree",
        "activity",
        "case",
    ).to_numpy()
    return _h1_power_segregation(profiles, communities, cases, activity), profiles, communities, cases, activity, strata


def _h1_power_null_statistic(
    sizes: np.ndarray,
    topology: dict[str, float],
    counts: dict[str, int],
    rng: np.random.Generator,
) -> float:
    _, profiles, communities, cases, activity, strata = _h1_power_draw(sizes, topology, counts, 0.0, rng)
    shuffled = profiles.copy()
    for stratum in np.unique(strata):
        indexes = np.flatnonzero(strata == stratum)
        shuffled[indexes] = profiles[rng.permutation(indexes)]
    return _h1_power_segregation(shuffled, communities, cases, activity)


def _case_cluster_sizes(
    sizes: np.ndarray,
    counts: dict[str, int],
    details: dict[str, Any] | None,
    case_name: str,
    fallback_index: int,
) -> np.ndarray:
    by_case = (details or {}).get("case_cluster_sizes", {})
    if case_name in by_case and by_case[case_name]:
        values = np.asarray(by_case[case_name], dtype=int)
    else:
        ordered = list(counts.values())
        count = max(2, min(len(sizes), int(ordered[min(fallback_index, len(ordered) - 1)])))
        values = np.resize(sizes if fallback_index % 2 == 0 else sizes[::-1], count).astype(int)
    return np.maximum(values[:256], 1)


def _named_case(counts: dict[str, int], details: dict[str, Any] | None, tokens: tuple[str, ...], fallback_index: int) -> str:
    names = list((details or {}).get("case_cluster_sizes", {}))
    for name in names + list(counts):
        if any(token in str(name).casefold() for token in tokens):
            return str(name)
    return names[fallback_index] if fallback_index < len(names) else (list(counts)[fallback_index] if counts else str(fallback_index))


def _h2_case_values(cluster_sizes: np.ndarray, probability: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    author_counts = np.maximum(1, np.minimum(cluster_sizes, np.rint(np.sqrt(cluster_sizes)).astype(int)))
    draws = [rng.binomial(1, probability, int(count)) for count in author_counts]
    cluster_means = np.asarray([values.mean() for values in draws])
    authors = np.concatenate(draws)
    return authors, cluster_means


def _h2_power_statistic(
    sizes: np.ndarray,
    counts: dict[str, int],
    details: dict[str, Any] | None,
    effect: float,
    rng: np.random.Generator,
) -> float:
    earlier_name = _named_case(counts, details, ("legislation", "earlier"), 0)
    later_name = _named_case(counts, details, ("implementation", "later"), 1)
    earlier_sizes = _case_cluster_sizes(sizes, counts, details, earlier_name, 0)
    later_sizes = _case_cluster_sizes(sizes, counts, details, later_name, 1)
    earlier, earlier_clusters = _h2_case_values(earlier_sizes, 0.30, rng)
    later, later_clusters = _h2_case_values(later_sizes, min(0.95, 0.30 + effect), rng)
    observed = float(later.mean() - earlier.mean())
    earlier_se = float(np.std(earlier_clusters, ddof=1) / math.sqrt(len(earlier_clusters))) if len(earlier_clusters) > 1 else 0.0
    later_se = float(np.std(later_clusters, ddof=1) / math.sqrt(len(later_clusters))) if len(later_clusters) > 1 else 0.0
    fallback_se = math.sqrt(max(0.30 * 0.70 / len(earlier), 1e-12) + max((0.30 + effect) * (0.70 - effect) / len(later), 1e-12))
    return observed / max(math.sqrt(earlier_se**2 + later_se**2), fallback_se, 1e-6)


def _entropy_boost(profile: np.ndarray, effect: float) -> np.ndarray:
    target = min(1.0, normalized_frame_entropy(profile) + effect)
    uniform = np.ones(len(profile)) / len(profile)
    low, high = 0.0, 1.0
    for _ in range(20):
        midpoint = (low + high) / 2
        candidate = (1 - midpoint) * profile + midpoint * uniform
        if normalized_frame_entropy(candidate) < target:
            low = midpoint
        else:
            high = midpoint
    return (1 - high) * profile + high * uniform


def _h3_power_statistic(sizes: np.ndarray, topology: dict[str, float], effect: float, rng: np.random.Generator) -> float:
    set_count = max(2, min(len(sizes) // 4, int(topology["nodes"]) // 4 or 2, 128))
    observed_values = []
    sets = []
    for _ in range(set_count):
        controls = rng.dirichlet(np.ones(len(FRAME_LABELS)) * 2, size=3)
        broker = _entropy_boost(rng.dirichlet(np.ones(len(FRAME_LABELS)) * 2), effect)
        values = np.asarray([normalized_frame_entropy(broker), *[normalized_frame_entropy(row) for row in controls]])
        observed_values.append(float(values[0] - values[1:].mean()))
        sets.append(values)
    observed = float(np.mean(observed_values))
    null = np.asarray(
        [
            float(
                np.mean(
                    [
                        values[chosen] - np.delete(values, chosen).mean()
                        for values in sets
                        for chosen in [int(rng.integers(4))]
                    ]
                )
            )
            for _ in range(3)
        ]
    )
    return _permutation_z(observed, null)


def _fallback_channel_specs(counts: dict[str, int]) -> list[dict[str, Any]]:
    news = max(2, int(counts.get("news", list(counts.values())[0] if counts else 2)))
    commentary = max(2, int(counts.get("commentary", list(counts.values())[-1] if counts else news)))
    return [
        {"source": source, "events": {"E2": 1, "E3": 1}}
        for source, count in (("news", news), ("commentary", commentary))
        for _ in range(min(count, 128))
    ]


def _h4_power_statistic(counts: dict[str, int], details: dict[str, Any] | None, effect: float, rng: np.random.Generator) -> float:
    specs = (details or {}).get("channel_specs") or _fallback_channel_specs(counts)
    specs = [spec for spec in specs if spec.get("source") in {"news", "commentary"}]
    events = sorted({event for spec in specs for event in spec.get("events", {})})
    if len(events) < 2 or any(not {"news", "commentary"} <= {spec["source"] for spec in specs if event in spec.get("events", {})} for event in events):
        specs = _fallback_channel_specs(counts)
        events = ["E2", "E3"]
    rows = []
    for index, spec in enumerate(specs):
        source = str(spec["source"])
        for event, number in spec.get("events", {}).items():
            for _ in range(max(1, min(int(number), 8))):
                event_offset = 0.03 if str(event) == events[-1] else -0.03
                source_offset = effect if source == "news" else 0.0
                rows.append({"channel": f"channel-{index}", "event": str(event), "source": source, "alignment": np.clip(0.5 + event_offset + source_offset + rng.normal(0, 0.08), 0, 1)})
    values = pd.DataFrame(rows)
    observed = _event_adjusted_difference(values, "source", "alignment", "event")
    channel_table = values.groupby("channel", sort=True).agg(source=("source", "first"), signature=("event", lambda value: tuple(sorted(set(value)))))
    null_values = []
    for _ in range(3):
        permuted = values.copy()
        lookup = {}
        for signature, group in channel_table.groupby("signature", sort=True):
            labels = rng.permutation(group["source"].to_numpy())
            lookup.update(dict(zip(group.index, labels)))
        permuted["source"] = permuted["channel"].map(lookup)
        null_values.append(_event_adjusted_difference(permuted, "source", "alignment", "event"))
    return _permutation_z(observed, np.asarray(null_values))


def _endpoint_power_statistic(
    endpoint_id: str,
    cluster_sizes: np.ndarray,
    topology: dict[str, float],
    source_type_counts: dict[str, int],
    effect: float,
    rng: np.random.Generator,
    design_details: dict[str, Any] | None = None,
) -> float:
    """Draw one statistic using the endpoint's declared estimator and units."""

    sizes = np.maximum(cluster_sizes.astype(int), 1)

    if endpoint_id in {"H1-R", "H1-B"}:
        return _h1_power_statistic(sizes, topology, source_type_counts, effect, rng)

    if endpoint_id in {"H2-P", "H2-C"}:
        return _h2_power_statistic(sizes, source_type_counts, design_details, effect, rng)

    if endpoint_id == "H3":
        return _h3_power_statistic(sizes, topology, effect, rng)

    if endpoint_id == "H4":
        return _h4_power_statistic(source_type_counts, design_details, effect, rng)

    raise ValueError(f"unknown endpoint-specific power design: {endpoint_id}")


def _endpoint_specific_power_simulation(
    endpoint_id: str,
    sizes: np.ndarray,
    topology_values: dict[str, float],
    counts: dict[str, int],
    minimum_effect: float,
    *,
    alpha: float,
    target_power: float,
    alternative: str,
    seed: int,
    replicates: int,
    design_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    critical = norm.ppf(1 - alpha if alternative == "one-sided" else 1 - alpha / 2)
    critical_method = "normal_approximation"
    if endpoint_id in {"H1-R", "H1-B"}:
        null_rng = np.random.default_rng(seed)
        null_statistics = np.asarray(
            [_h1_power_null_statistic(sizes, topology_values, counts, null_rng) for _ in range(replicates)],
            dtype=float,
        )
        critical = float(
            np.quantile(
                null_statistics if alternative == "one-sided" else np.abs(null_statistics),
                1 - alpha,
            )
        )
        critical_method = "empirical_case_degree_activity_permutation_null"

    def simulated_power(effect: float) -> float:
        rng = np.random.default_rng(seed + 1)
        statistics = np.asarray(
            [_endpoint_power_statistic(endpoint_id, sizes, topology_values, counts, effect, rng, design_details) for _ in range(replicates)],
            dtype=float,
        )
        rejected = statistics > critical if alternative == "one-sided" else np.abs(statistics) > critical
        return float(np.mean(rejected))

    declared_power = simulated_power(minimum_effect)
    high = max(float(minimum_effect), 0.05)
    while simulated_power(high) < target_power and high < 1_000:
        high *= 2
    low = 0.0
    if simulated_power(high) >= target_power:
        for _ in range(16):
            midpoint = (low + high) / 2
            if simulated_power(midpoint) >= target_power:
                high = midpoint
            else:
                low = midpoint
        mde = high
    else:
        mde = None
    return {
        "method": "endpoint_specific_frozen_design_monte_carlo",
        "endpoint_id": endpoint_id,
        "design": {
            "H1-R": "author_profile_community_contrast",
            "H1-B": "author_profile_community_contrast",
            "H2-P": "thread_cluster_binary_risk_difference",
            "H2-C": "thread_cluster_binary_risk_difference",
            "H3": "one_to_three_matched_entropy_contrast",
            "H4": "channel_clustered_video_alignment_contrast",
        }[endpoint_id],
        "seed": int(seed),
        "replicates": int(replicates),
        "critical_value": float(critical),
        "critical_value_method": critical_method,
        "cluster_sizes": sizes.astype(int).tolist(),
        "topology": {key: int(value) if float(value).is_integer() else float(value) for key, value in topology_values.items()},
        "source_type_counts": counts,
        "simulation_unit_cap": 256,
        "estimator_contract": {
            "H1-R": "author_vector_permutation_test: activity-weighted JS segregation with case-stratified profile shuffles; empirical permutation-null critical value",
            "H1-B": "author_vector_permutation_test: activity-weighted JS segregation with case-stratified profile shuffles; empirical permutation-null critical value",
            "H2-P": "thread_cluster_bootstrap_difference: author-balanced binary risk difference over legislation/implementation thread clusters",
            "H2-C": "thread_cluster_bootstrap_difference: author-balanced binary risk difference over legislation/implementation thread clusters",
            "H3": "broker_permutation_test: one broker versus three matched controls per set on normalized five-frame entropy",
            "H4": "video_alignment_contrast: event-adjusted news coefficient with whole-channel source permutations",
        }[endpoint_id],
        "minimum_effect": float(minimum_effect),
        "mde": float(mde) if mde is not None else None,
        "power_at_minimum_effect": declared_power,
        "target_power": target_power,
        "alternative": alternative,
        "status": "passed" if declared_power >= target_power else "inconclusive_underpowered",
    }


def frozen_design_power_simulation(
    cluster_sizes: list[int] | np.ndarray,
    topology: dict[str, int | float],
    source_type_counts: dict[str, int],
    minimum_effect: float,
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
    alternative: str = "one-sided",
    endpoint_id: str | None = None,
    seed: int = POWER_SIMULATION_SEED,
    replicates: int = POWER_SIMULATION_REPLICATES,
    design_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate power from frozen cluster, topology and source-type design inputs."""

    sizes = np.asarray(cluster_sizes, dtype=float)
    if sizes.ndim != 1 or not len(sizes) or not np.isfinite(sizes).all() or np.any(sizes < 1) or not np.equal(sizes, sizes.astype(int)).all():
        raise ValueError("power simulation needs positive integer cluster sizes")
    if minimum_effect <= 0 or not 0 < alpha < 1 or not 0 < target_power < 1 or alternative not in {"one-sided", "two-sided"} or replicates < 100:
        raise ValueError("power simulation inputs are invalid")
    required_topology = {"nodes", "edges", "components"}
    if not required_topology <= set(topology):
        raise ValueError("power simulation needs nodes, edges and components")
    topology_values = {key: float(topology[key]) for key in required_topology}
    if any(not np.isfinite(value) or value < 0 for value in topology_values.values()) or topology_values["nodes"] < 1:
        raise ValueError("power topology values must be finite and non-negative")
    counts = {str(key): int(value) for key, value in source_type_counts.items()}
    if not counts or any(value < 1 for value in counts.values()):
        raise ValueError("power simulation needs positive source-type counts")
    if endpoint_id is not None:
        if endpoint_id not in PRIMARY_ENDPOINTS:
            raise ValueError(f"unknown endpoint-specific power design: {endpoint_id}")
        return _endpoint_specific_power_simulation(
            endpoint_id,
            sizes,
            topology_values,
            counts,
            minimum_effect,
            alpha=alpha,
            target_power=target_power,
            alternative=alternative,
            seed=seed,
            replicates=replicates,
            design_details=design_details,
        )
    total_sources = sum(counts.values())
    proportions = np.asarray(list(counts.values()), dtype=float) / total_sources
    source_balance = float(1 / np.sum(proportions**2) / len(proportions))
    connected_units = max(topology_values["nodes"] - topology_values["components"], 1.0)
    topology_factor = min(1.0, connected_units / topology_values["nodes"]) * min(1.0, topology_values["edges"] / topology_values["nodes"])
    topology_factor = max(topology_factor, 1 / max(topology_values["nodes"], 1.0))
    kish_clusters = float(sizes.sum() ** 2 / np.square(sizes).sum())
    cluster_design_effect = 1 + 0.05 * (float(np.square(sizes).sum() / sizes.sum()) - 1)
    effective_sample_size = max(kish_clusters * source_balance * topology_factor / cluster_design_effect, 1e-9)
    standard_error = 1 / math.sqrt(effective_sample_size)
    critical = norm.ppf(1 - alpha if alternative == "one-sided" else 1 - alpha / 2)
    rng = np.random.default_rng(seed)
    standard_normal = rng.standard_normal(replicates)

    def simulated_power(effect: float) -> float:
        statistic = effect / standard_error + standard_normal
        rejected = statistic > critical if alternative == "one-sided" else np.abs(statistic) > critical
        return float(np.mean(rejected))

    declared_power = simulated_power(float(minimum_effect))
    high = max(float(minimum_effect), standard_error)
    while simulated_power(high) < target_power and high < 1_000:
        high *= 2
    low = 0.0
    if simulated_power(high) >= target_power:
        for _ in range(40):
            midpoint = (low + high) / 2
            if simulated_power(midpoint) >= target_power:
                high = midpoint
            else:
                low = midpoint
        mde = high
    else:
        mde = None
    analytic = power_mde_diagnostic(effective_sample_size, 1.0, minimum_effect, alpha=alpha, target_power=target_power, alternative=alternative)
    return {
        "method": "frozen_design_monte_carlo",
        "seed": int(seed),
        "replicates": int(replicates),
        "cluster_sizes": sizes.astype(int).tolist(),
        "topology": {key: int(value) if float(value).is_integer() else float(value) for key, value in topology_values.items()},
        "source_type_counts": counts,
        "source_balance_factor": source_balance,
        "topology_factor": topology_factor,
        "cluster_design_effect": cluster_design_effect,
        "effective_sample_size": effective_sample_size,
        "standard_error": standard_error,
        "minimum_effect": float(minimum_effect),
        "mde": float(mde) if mde is not None else None,
        "power_at_minimum_effect": declared_power,
        "target_power": target_power,
        "alternative": alternative,
        "status": "passed" if declared_power >= target_power else "inconclusive_underpowered",
        "analytic_supplement": analytic,
    }


def power_mde_simulation(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility name for the frozen-design power gate."""

    return frozen_design_power_simulation(*args, **kwargs)


def primary6_decision(p_values: dict[str, float | None], effects: dict[str, float | None]) -> dict[str, Any]:
    missing = [endpoint for endpoint in PRIMARY_ENDPOINTS if p_values.get(endpoint) is None or effects.get(endpoint) is None]
    available = {endpoint: float(p_values[endpoint]) for endpoint in PRIMARY_ENDPOINTS if endpoint not in missing}
    adjusted = holm_adjust(available)
    decisions = {endpoint: False for endpoint in PRIMARY_ENDPOINTS}
    for endpoint in available:
        effect = float(effects[endpoint])
        threshold = PRIMARY_EFFECT_THRESHOLDS[endpoint]
        meaningful = abs(effect) >= threshold if endpoint == "H4" else effect >= threshold
        decisions[endpoint] = bool(adjusted[endpoint] < 0.05 and meaningful)
    return {"family": PRIMARY_FAMILY, "endpoints": list(PRIMARY_ENDPOINTS), "adjusted_p_values": adjusted, "effects": {endpoint: effects.get(endpoint) for endpoint in PRIMARY_ENDPOINTS}, "decisions": decisions, "missing_endpoints": missing, "status": "complete" if not missing else "incomplete_missing_endpoint", "all_endpoints_pass": bool(not missing and all(decisions.values()))}


def _is_confirmatory_result(result: Any) -> bool:
    return isinstance(result, dict) and result.get("p_value") is not None and result.get("status") == "estimated" and result.get("power_status") == "passed"


PRIMARY6_ARTIFACT = HYPOTHESIS_ROOT / "primary6.json"


def _labels_present() -> bool:
    try:
        frame = load_label_packet("evaluation")
    except FileNotFoundError:
        return False
    fields = ("sentiment", "stance")
    return bool(
        all(f"adjudicated_{field}" in frame and frame[f"adjudicated_{field}"].astype(str).str.strip().ne("").all() for field in fields)
        and _frame_evaluation_mask(frame).any()
    )


def weighted_human_sample_frame_estimate(
    frame: pd.DataFrame,
    *,
    platform: str | None = None,
    platform_column: str = "platform",
    case_column: str = "case_window_id",
    author_column: str = "author_key",
    cluster_column: str = "cluster_id",
    relevance_column: str = "adjudicated_relevance",
    language_column: str = "adjudicated_language",
    frame_column: str = "adjudicated_frame_labels",
    inclusion_probability_column: str = "inclusion_probability",
    bootstrap_replicates: int = 9_999,
    seed: int = POWER_SIMULATION_SEED,
) -> dict[str, Any]:
    """Estimate eligible frame prevalence with cluster-bootstrap uncertainty."""

    required = {
        platform_column,
        case_column,
        author_column,
        cluster_column,
        relevance_column,
        language_column,
        frame_column,
        inclusion_probability_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        return {"status": "human-labels-required", "missing_columns": missing}
    if bootstrap_replicates < 2:
        return {"status": "human-labels-required", "reason": "cluster bootstrap requires at least two replicates"}
    values = frame.copy()
    if platform not in {None, "both"}:
        values = values[values[platform_column].astype(str).eq(platform)].copy()
    values = values.loc[values[frame_column].notna()].copy()
    if values.empty:
        return {"status": "human-labels-required", "reason": "no coded frame rows for the requested platform"}
    relevance = values[relevance_column].fillna("").astype(str).str.strip().str.casefold()
    language = values[language_column].fillna("").astype(str).str.strip().str.casefold()
    eligible = (
        relevance.eq("relevant")
        & language.isin({"english", "mixed"})
        & values[author_column].fillna("").astype(str).str.strip().ne("")
        & values[cluster_column].fillna("").astype(str).str.strip().ne("")
    )
    eligibility = {
        "relevance": "relevant",
        "language": ["english", "mixed"],
        "input_rows": int(len(values)),
        "eligible_rows": int(eligible.sum()),
        "excluded_rows": int((~eligible).sum()),
    }
    values = values.loc[eligible].reset_index(drop=True)
    if values.empty:
        return {"status": "human-labels-required", "reason": "no rows meet reserve relevance/language eligibility", "eligibility": eligibility}
    probabilities = pd.to_numeric(values[inclusion_probability_column], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities <= 0):
        return {"status": "human-labels-required", "reason": "human-sample inclusion probabilities are invalid"}
    weights = 1.0 / probabilities
    labels = _frame_label_matrix(values[frame_column])
    rng = np.random.default_rng(seed)

    def cluster_interval(group_positions: np.ndarray, cluster_values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        clusters = np.unique(cluster_values[group_positions])
        if len(clusters) < 2:
            return None
        cluster_weights = np.asarray([weights[group_positions[cluster_values[group_positions] == cluster]].sum() for cluster in clusters])
        cluster_labels = np.asarray(
            [
                (weights[group_positions[cluster_values[group_positions] == cluster], None] * labels[group_positions[cluster_values[group_positions] == cluster]]).sum(axis=0)
                for cluster in clusters
            ]
        )
        draws = rng.integers(len(clusters), size=(bootstrap_replicates, len(clusters)))
        multiplicities = np.zeros((bootstrap_replicates, len(clusters)), dtype=np.int32)
        np.add.at(multiplicities, (np.arange(bootstrap_replicates)[:, None], draws), 1)
        sampled_weights = multiplicities @ cluster_weights
        sampled_labels = multiplicities @ cluster_labels
        prevalence = sampled_labels / sampled_weights[:, None]
        return np.quantile(prevalence, 0.025, axis=0), np.quantile(prevalence, 0.975, axis=0)

    rows: list[dict[str, Any]] = []
    for (platform_value, case_value), indexes in values.groupby([platform_column, case_column], sort=True, dropna=False).groups.items():
        positions = np.asarray(list(indexes), dtype=int)
        group_weights = weights[positions]
        total_weight = float(group_weights.sum())
        effective_n = float(total_weight**2 / np.square(group_weights).sum())
        prevalence = (group_weights[:, None] * labels[positions]).sum(axis=0) / total_weight
        dimensions = [cluster_column]
        if str(platform_value).casefold() == "bluesky":
            dimensions = [author_column, cluster_column]
        intervals = [cluster_interval(positions, values[column].fillna("").astype(str).to_numpy()) for column in dimensions]
        if any(interval is None for interval in intervals):
            return {
                "status": "human-labels-required",
                "reason": f"{platform_value}/{case_value} has fewer than two clusters for clustered uncertainty",
                "eligibility": eligibility,
            }
        lower = np.min([interval[0] for interval in intervals if interval is not None], axis=0)
        upper = np.max([interval[1] for interval in intervals if interval is not None], axis=0)
        rows.append(
            {
                "platform": str(platform_value),
                "case_window_id": str(case_value),
                "rows": int(len(positions)),
                "authors": int(values.iloc[positions][author_column].astype(str).nunique()),
                "cluster_dimensions": dimensions,
                "cluster_count": {column: int(values.iloc[positions][column].astype(str).nunique()) for column in dimensions},
                "weighted_rows": total_weight,
                "effective_sample_size": effective_n,
                "frame_prevalence": {
                    label: {
                        "estimate": float(prevalence[index]),
                        "lower_95": float(max(0.0, lower[index])),
                        "upper_95": float(min(1.0, upper[index])),
                    }
                    for index, label in enumerate(FRAME_LABELS)
                },
            }
        )
    return {
        "status": "available",
        "estimator": "inverse_inclusion_probability_weighted_human_sample_frame_prevalence",
        "rows": rows,
        "claim_scope": "descriptive coded-sample prevalence; not a confirmatory endpoint estimate",
        "blank_frame_labels": "included as valid all-negative rows",
        "eligibility": eligibility,
        "bootstrap_replicates": int(bootstrap_replicates),
        "interval_method": "inverse-inclusion-probability weighted multiway cluster bootstrap",
        "interval_note": "95% cluster-bootstrap intervals use Reddit thread, YouTube video, and Bluesky author/root resampling; Bluesky reports the conservative envelope across author and root clusters",
    }


def _human_sample_fallback(hypothesis_id: str, platform: str | None) -> dict[str, Any]:
    if not RESERVE_ASSESSMENT.exists():
        return {"status": "not_available", "reason": "reserve assessment has not been opened"}
    try:
        check_reserve_assessment()
        frame = load_label_packet("reserve")
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        return {"status": "human-labels-required", "reason": str(error)}
    if "annotation_status" not in frame or not frame["annotation_status"].astype(str).eq("adjudicated").all():
        return {"status": "human-labels-required", "reason": "reserve frame labels are not fully adjudicated"}
    fallback = weighted_human_sample_frame_estimate(frame, platform=platform)
    fallback.update({"hypothesis_id": hypothesis_id, "source_split": "reserve"})
    if hypothesis_id == "H2" and fallback.get("status") == "available":
        contrasts = {}
        grouped = {row["case_window_id"]: row for row in fallback["rows"] if row["platform"] == "reddit"}
        earlier, later = grouped.get("AU_LEGISLATION"), grouped.get("AU_IMPLEMENTATION")
        if earlier and later:
            contrasts = {
                label: {
                    "estimate": later["frame_prevalence"][label]["estimate"] - earlier["frame_prevalence"][label]["estimate"],
                    "lower_95": later["frame_prevalence"][label]["lower_95"] - earlier["frame_prevalence"][label]["upper_95"],
                    "upper_95": later["frame_prevalence"][label]["upper_95"] - earlier["frame_prevalence"][label]["lower_95"],
                    "earlier": earlier["frame_prevalence"][label]["estimate"],
                    "later": later["frame_prevalence"][label]["estimate"],
                }
                for label in FRAME_LABELS
            }
        fallback["case_contrasts"] = contrasts
    return fallback


def _graph_status(name: str) -> str | None:
    if not NETWORK_MANIFEST.exists():
        return None
    return read_json(NETWORK_MANIFEST).get("graph_registry", {}).get(name, {}).get("status")


def _stable_structure_available(graph_names: set[str]) -> bool:
    if not STRUCTURE_MANIFEST.exists():
        return False
    try:
        check_structure()
        summary = pd.read_csv(ANALYSIS_ROOT / "networks" / "structure" / "structure_summary.csv")
    except (FileNotFoundError, ValueError):
        return False
    required = summary[summary["graph_id"].isin(graph_names)]
    if required.empty or not required["community_status"].eq("primary_leiden_stable").all():
        return False
    try:
        roles = pd.read_parquet(ANALYSIS_ROOT / "networks" / "structure" / "roles.parquet")
    except (FileNotFoundError, OSError, ValueError):
        return False
    if not {"graph_id", "node_type", "betweenness_stability_status"} <= set(roles.columns):
        return False
    required_roles = roles[roles["graph_id"].isin(graph_names) & roles["node_type"].eq("author")]
    return bool(not required_roles.empty and required_roles["betweenness_stability_status"].eq("stable").all())


def _gate_status(hypothesis_id: str, platform: str | None = None) -> tuple[str, list[str]]:
    missing: list[str] = []
    required_platforms = {
        "reddit": hypothesis_id in {"H1", "H2", "H3"} and platform in {None, "both", "reddit"},
        "bluesky": hypothesis_id == "H1" and platform in {None, "both", "bluesky"},
        "youtube": hypothesis_id == "H4",
    }
    required_platforms = {name for name, required in required_platforms.items() if required}
    reserve_ready = False
    human_fallback = {"status": "not_available"}
    if RESERVE_ASSESSMENT.exists():
        reserve_check_error = None
        try:
            check_reserve_assessment()
            reserve = read_json(RESERVE_ASSESSMENT)
            reserve_ready = all(
                reserve.get("metrics", {}).get("frames", {}).get("by_platform", {}).get(name, {}).get("gates", {}).get("passed") is True
                and reserve.get("metrics", {}).get("frames", {}).get("by_platform", {}).get(name, {}).get("unseen_author", {}).get("gates", {}).get("passed") is True
                for name in required_platforms
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            reserve_ready = False
            reserve_check_error = str(error)
        if reserve_check_error is None:
            human_fallback = _human_sample_fallback(hypothesis_id, platform)
        else:
            human_fallback = {"status": "human-labels-required", "reason": f"reserve integrity check failed: {reserve_check_error}"}
    if not reserve_ready and human_fallback.get("status") != "available":
        missing.append("measurement_validity")
    prediction_ready = False
    if reserve_ready and PREDICTIONS_MANIFEST.exists():
        try:
            from src.analysis.predictions import check_predictions

            check_predictions()
            prediction_manifest = read_json(PREDICTIONS_MANIFEST)
            prediction_ready = all(
                prediction_manifest.get("task_status_by_platform", {}).get(name, {}).get("frames") == "validated_automated"
                for name in required_platforms
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            prediction_ready = False
    if reserve_ready and not prediction_ready:
        missing.append("selected_measurement_predictions")
    needs_reddit_audit = hypothesis_id in {"H2", "H3"} or (hypothesis_id == "H1" and platform in {None, "both", "reddit"})
    if needs_reddit_audit:
        if validate_reddit_relevance_audit()["status"] != "valid" or _graph_status("R_ACTOR_ATTENTION") != "audited_ready":
            missing.append("reddit_thread_relevance_audit")
    if hypothesis_id == "H4":
        if validate_youtube_metadata_audit()["status"] != "valid" or _graph_status("Y_COMMENTER_VIDEO") != "audited_ready":
            missing.append("youtube_video_metadata_audit")
    if hypothesis_id in {"H1", "H3"}:
        required_graphs = {"R_ACTOR_ATTENTION"}
        if hypothesis_id == "H1" and platform in {None, "both", "bluesky"}:
            required_graphs.add("B_ACTOR_ATTENTION")
        if not _stable_structure_available(required_graphs):
            missing.append("stable_topology_partition")
    if hypothesis_id == "H4" and not NETWORK_MANIFEST.exists():
        missing.append("topology_edges")
    if missing:
        return "blocked", sorted(set(missing))
    return ("ready" if reserve_ready else "fallback-human-sample"), []


def _input_checksums() -> dict[str, str | None]:
    paths = {
        "population_manifest": POPULATION_ARTIFACT,
        "network_manifest": NETWORK_MANIFEST,
        "structure_manifest": STRUCTURE_MANIFEST,
        "evaluation_report": EVALUATION_REPORT,
        "reserve_activation": RESERVE_ACTIVATION,
        "reserve_assessment": RESERVE_ASSESSMENT,
        "predictions_manifest": PREDICTIONS_MANIFEST,
        "evaluation_labels": VALIDATION_ROOT / "coordinator_labels_evaluation.csv",
        "evaluation_coder_a": VALIDATION_ROOT / "coder_a_labels_evaluation.csv",
        "evaluation_coder_b": VALIDATION_ROOT / "coder_b_labels_evaluation.csv",
        "selection_sentiment": SENTIMENT_SELECTION,
        "selection_stance_frames": STANCE_FRAME_SELECTION,
        "reddit_relevance_audit": REPO_ROOT / "data" / "analysis" / "annotation" / "reddit_thread_relevance.csv",
        "youtube_metadata_audit": REPO_ROOT / "data" / "analysis" / "annotation" / "youtube_video_metadata.csv",
    }
    return {name: sha256_file(path) if path.exists() else None for name, path in paths.items()}


def _frame_label_matrix(values: pd.Series) -> np.ndarray:
    return np.asarray(
        [[label in set(str(value).split("|")) for label in FRAME_LABELS] for value in values],
        dtype=int,
    )


def _primary_frame_models() -> dict[str, Any]:
    development = load_label_packet("development")
    if development.empty or not development["platform"].notna().all():
        raise ValueError("development frame labels are unavailable")
    check_selection("frames")
    selection = _stored_selection("frames")
    if selection.get("status") != "selected":
        raise ValueError("the primary frame pipeline is not frozen")
    results = selection.get("candidate_result", {}).get("by_platform", {})
    models: dict[str, Any] = {}
    for platform, group in development.groupby("platform", sort=True):
        selected = results.get(str(platform), {})
        if selected.get("name") != "tfidf_ovr_logistic_regression":
            raise ValueError(f"unexpected frame pipeline for {platform}")
        group = group.loc[_frame_evaluation_mask(group)].copy()
        if group.empty:
            raise ValueError(f"development frame labels are unavailable for {platform}")
        labels = _frame_label_matrix(group["adjudicated_frame_labels"])
        model, _ = _frame_pipeline()
        model.fit(group["text_for_annotation"].astype(str), labels)
        model.frame_thresholds = _selected_frame_thresholds(selected)
        models[str(platform)] = model
    return models


def _frame_documents(platform: str) -> pd.DataFrame:
    """Load one platform's frozen primary frame population with masked text only."""

    if platform == "reddit":
        raw = pd.read_parquet(
            REPO_ROOT / "data" / "processed" / "reddit_age_gate" / "analysis_corpus.parquet",
            columns=[
                "record_id", "thing", "event_id", "author_hash", "thread_id", "subreddit",
                "schema_valid", "human_only", "is_english", "is_language_uncertain", "text_topic", "is_url_only", "is_no_substantive_text", "relevance_terms",
            ],
        )
        raw["case"] = raw["event_id"].map(REDDIT_CASE_BY_EVENT)
        accepted = validate_reddit_relevance_audit()
        if accepted["status"] != "valid":
            raise ValueError("Reddit relevance audit is required before frame inference")
        raw = raw[
            raw["case"].notna()
            & raw["thread_id"].fillna("").astype(str).ne("")
            & raw["author_hash"].fillna("").astype(str).ne("")
            & raw["schema_valid"].fillna(False).astype(bool)
            & raw["human_only"].fillna(False).astype(bool)
            & raw["is_english"].fillna(False).astype(bool)
            & ~raw["is_url_only"].fillna(False).astype(bool)
            & ~raw["is_no_substantive_text"].fillna(False).astype(bool)
            & raw["text_topic"].fillna("").astype(str).str.strip().ne("")
            & raw["thread_id"].astype(str).isin(accepted["accepted_thread_ids"])
        ].copy()
        keyword_threads = set(raw.loc[raw["relevance_terms"].fillna("").astype(str).str.strip().ne(""), "thread_id"].astype(str))
        return pd.DataFrame(
            {
                "doc_id": raw["record_id"].astype(str),
                "author": raw["author_hash"].astype(str),
                "case": raw["case"].astype(str),
                "thread": raw["thread_id"].astype(str),
                "subreddit": raw["subreddit"].fillna("").astype(str),
                "language_population": np.where(raw["is_language_uncertain"].fillna(False).astype(bool), "inclusive_uncertain_bound", "strict_english"),
                "thread_population": "audited_relevant",
                "keyword_screen": raw["thread_id"].astype(str).isin(keyword_threads).map({True: "keyword_screen", False: "not_keyword_screen"}),
                "text": raw["text_topic"].fillna("").astype(str).map(mask_text),
            }
        ).reset_index(drop=True)

    if platform == "bluesky":
        root = REPO_ROOT / "data" / "processed" / "bluesky"
        posts = pd.read_parquet(
            root / "posts.parquet",
            columns=["doc_id", "root_doc_id", "event_window", "is_english", "lang_detect_prob", "n_chars", "is_meme", "is_repeat_burst", "author_hash", "text_raw"],
        )
        authors = pd.read_parquet(root / "authors.parquet", columns=["author_hash", "account_class"])
        posts = posts.merge(authors, on="author_hash", how="left", validate="many_to_one")
        edges = pd.read_parquet(NETWORK_ROOT / "b_message_tree.parquet", columns=["source_doc_id", "target_doc_id", "scope"])
        endpoint_ids = set(edges["source_doc_id"].astype(str)) | set(edges["target_doc_id"].astype(str))
        posts = posts[
            posts["doc_id"].astype(str).isin(endpoint_ids)
            & posts["event_window"].isin(["E2", "E3"])
            & posts["is_english"].fillna(False).astype(bool)
            & ~posts["is_meme"].fillna(False).astype(bool)
            & ~posts["is_repeat_burst"].fillna(False).astype(bool)
            & ~posts["account_class"].isin({"labelled_spam", "repeater"})
            & posts["author_hash"].fillna("").astype(str).ne("")
            & posts["text_raw"].fillna("").astype(str).str.strip().ne("")
        ].copy()
        language_uncertain = posts["lang_detect_prob"].fillna(0).lt(0.8) | posts["n_chars"].fillna(0).lt(24)
        return pd.DataFrame(
            {
                "doc_id": posts["doc_id"].astype(str),
                "author": posts["author_hash"].astype(str),
                "case": posts["event_window"].astype(str),
                "thread": posts["root_doc_id"].fillna(posts["doc_id"]).astype(str),
                "subreddit": "",
                "language_population": np.where(language_uncertain, "inclusive_uncertain_bound", "strict_english"),
                "text": posts["text_raw"].fillna("").astype(str).map(mask_text),
            }
        ).reset_index(drop=True)

    if platform == "youtube":
        root = REPO_ROOT / "data" / "processed" / "youtube"
        docs = pd.read_parquet(
            root / "documents.parquet",
            columns=["doc_id", "thing", "root_doc_id", "container_id", "author_hash", "yt_video_event_id", "exclusion_status", "is_english", "text_clean"],
        )
        assignments = youtube_video_event_assignments(docs)
        docs = docs.drop(columns=["yt_video_event_id"]).merge(assignments[["doc_id", "video_event"]], on="doc_id", how="left", validate="one_to_one")
        videos = docs[docs["thing"].eq("video") & docs["video_event"].isin(["E2", "E3"])]
        strict_comments = docs[
            docs["thing"].ne("video")
            & docs["exclusion_status"].eq("eligible")
            & docs["is_english"].fillna(False).astype(bool)
            & docs["author_hash"].fillna("").astype(str).ne("")
            & docs["root_doc_id"].notna()
            & docs["text_clean"].fillna("").astype(str).str.strip().ne("")
        ]
        counts = strict_comments.groupby("root_doc_id").size()
        seed_ids = {canonical_youtube_video_id(value) for value in frozen_youtube_video_ids()}
        candidate_ids = set(videos.loc[
            videos["doc_id"].astype(str).isin(seed_ids)
            & videos["doc_id"].isin(counts[counts.ge(20)].index),
            "doc_id",
        ].astype(str))
        audit = validate_youtube_metadata_audit()
        if audit["status"] != "valid":
            raise ValueError("YouTube metadata audit is required before frame inference")
        audited_ids = set(audit["audited_video_ids"])
        if audited_ids != candidate_ids:
            raise ValueError("YouTube metadata audit does not cover the frozen selected-video population")
        selected_ids = set(audit["accepted_video_ids"])
        comments = strict_comments[strict_comments["root_doc_id"].astype(str).isin(selected_ids)].copy()
        return pd.DataFrame(
            {
                "doc_id": comments["doc_id"].astype(str),
                "author": comments["author_hash"].astype(str),
                "case": comments["video_event"].astype(str),
                "thread": comments["root_doc_id"].astype(str),
                "subreddit": comments["container_id"].fillna("").astype(str),
                "video_id": comments["root_doc_id"].astype(str),
                "channel_id": comments["container_id"].fillna("").astype(str),
                "text": comments["text_clean"].fillna("").astype(str).map(mask_text),
            }
        ).reset_index(drop=True)

    raise ValueError(f"unsupported frame platform: {platform}")


def _predicted_frame_documents(platform: str, models: dict[str, Any]) -> pd.DataFrame:
    documents = _frame_documents(platform)
    model = models[platform]
    probabilities = np.asarray(model.predict_proba(documents["text"]), dtype=float)
    if not hasattr(model, "frame_thresholds"):
        raise ValueError("primary frame model has no frozen thresholds")
    thresholds = _validate_frame_threshold_values(model.frame_thresholds)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(FRAME_LABELS) or thresholds.shape != (len(FRAME_LABELS),):
        raise ValueError("primary frame model probabilities or thresholds have the wrong shape")
    predicted = (probabilities >= thresholds).astype(int)
    for index, label in enumerate(FRAME_LABELS):
        documents[label] = predicted[:, index]
    return documents


def _author_profiles(platform: str, models: dict[str, Any], language_population: str = "human_calibrated") -> pd.DataFrame:
    documents = _predicted_frame_documents(platform, models)
    if language_population != "human_calibrated":
        if "language_population" not in documents:
            raise ValueError("author profiles lack the requested language population")
        documents = documents[documents["language_population"].eq(language_population)].copy()
    if documents.empty:
        raise ValueError("requested language population has no author documents")
    profiles = documents.groupby(["case", "author"], sort=True)[FRAME_LABELS].mean().reset_index()
    counts = documents.groupby(["case", "author"], sort=True).size().rename("documents").reset_index()
    breadth = documents.groupby(["case", "author"], sort=True)["subreddit"].nunique().rename("subreddit_breadth").reset_index()
    profiles = profiles.merge(counts, on=["case", "author"], validate="one_to_one").merge(breadth, on=["case", "author"], validate="one_to_one")
    graph_name = "R_ACTOR_ATTENTION" if platform == "reddit" else "B_ACTOR_ATTENTION"
    roles = pd.read_parquet(NETWORK_ROOT / "structure" / "roles.parquet")
    roles = roles[
        roles["graph_id"].eq(graph_name)
        & roles["node_type"].eq("author")
    ][["scope", "node_id", "community_id", "in_strength", "out_strength", "betweenness"]].copy()
    roles = roles.rename(columns={"scope": "case", "node_id": "author"})
    roles["author"] = roles["author"].astype(str)
    roles["case"] = roles["case"].astype(str)
    roles = roles.drop_duplicates(["case", "author"])
    profiles = profiles.merge(roles, on=["case", "author"], how="inner", validate="one_to_one")
    profiles["degree"] = pd.to_numeric(profiles["in_strength"], errors="raise") + pd.to_numeric(profiles["out_strength"], errors="raise")
    profiles["activity"] = profiles["documents"].astype(float)
    if language_population != "human_calibrated":
        profiles["language_population"] = language_population
    return profiles


def _power_for_units(endpoint: str, values: np.ndarray, effective_sample_size: float) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    spread = float(np.std(values, ddof=1)) if len(values) > 1 else 1.0
    analytic = power_mde_diagnostic(
        max(float(effective_sample_size), 1.0),
        max(spread, 1e-6),
        PRIMARY_EFFECT_THRESHOLDS[endpoint],
        alternative="two-sided" if endpoint == "H4" else "one-sided",
    )
    design = _frozen_power_design(endpoint)
    if design is None:
        return {"status": "frozen_design_inputs_unavailable", "target_power": 0.80, "minimum_effect": PRIMARY_EFFECT_THRESHOLDS[endpoint], "analytic_supplement": analytic}
    result = frozen_design_power_simulation(
        design["cluster_sizes"],
        design["topology"],
        design["source_type_counts"],
        PRIMARY_EFFECT_THRESHOLDS[endpoint],
        endpoint_id=endpoint,
        alternative="two-sided" if endpoint == "H4" else "one-sided",
        design_details=design,
    )
    result["analytic_observed_outcome_supplement"] = analytic
    return result


def _frozen_power_design(endpoint: str) -> dict[str, Any] | None:
    """Read the fixed design inputs used by the power gate; never use observed outcomes."""

    graph_names = {
        "H1-R": "R_ACTOR_ATTENTION",
        "H1-B": "B_ACTOR_ATTENTION",
        "H2-P": "R_ACTOR_ATTENTION",
        "H2-C": "R_ACTOR_ATTENTION",
        "H3": "R_ACTOR_ATTENTION",
        "H4": "Y_ACTOR_ATTENTION_HANDLE",
    }
    cluster_paths = {
        "H1-R": NETWORK_ROOT / "r_message_tree.parquet",
        "H1-B": NETWORK_ROOT / "b_message_tree.parquet",
        "H2-P": NETWORK_ROOT / "r_message_tree.parquet",
        "H2-C": NETWORK_ROOT / "r_message_tree.parquet",
        "H3": NETWORK_ROOT / "r_message_tree.parquet",
    }
    try:
        summary = pd.read_csv(ANALYSIS_ROOT / "networks" / "structure" / "structure_summary.csv")
        topology_rows = summary[summary["graph_id"].eq(graph_names[endpoint])]
        if topology_rows.empty:
            return None
        topology = {column: int(topology_rows[column].sum()) for column in ("nodes", "edges", "components")}
        if endpoint == "H4":
            videos = pd.read_csv(REPO_ROOT / "config" / "youtube" / "seed_videos.csv")
            videos = videos[videos["event_id"].isin(["E2", "E3"]) & videos["video_type"].isin(["news", "commentary"])]
            cluster_sizes = videos.groupby("channel_id").size().tolist()
            source_type_counts = videos["video_type"].astype(str).value_counts().to_dict()
            channel_specs = []
            for channel_id, group in videos.groupby("channel_id", sort=True):
                source_types = group["video_type"].astype(str).str.casefold().unique().tolist()
                if len(source_types) != 1:
                    continue
                channel_specs.append(
                    {
                        "channel_id": str(channel_id),
                        "source": source_types[0],
                        "events": group["event_id"].astype(str).value_counts().to_dict(),
                    }
                )
            design_details = {"channel_specs": channel_specs}
        else:
            graph = pd.read_parquet(cluster_paths[endpoint], columns=["scope", "container_id", "edge_resolution"])
            cluster_sizes = graph["container_id"].fillna("").astype(str).loc[lambda values: values.ne("")].value_counts().tolist()
            source_type_counts = graph["scope"].astype(str).value_counts().to_dict()
            design_details = {
                "case_cluster_sizes": {
                    str(scope): group["container_id"].fillna("").astype(str).loc[lambda values: values.ne("")].value_counts().tolist()
                    for scope, group in graph.groupby("scope", sort=True)
                }
            }
        if not cluster_sizes or not source_type_counts:
            return None
        return {"cluster_sizes": cluster_sizes, "topology": topology, "source_type_counts": source_type_counts, **design_details}
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return None


def _apply_power_gate(result: dict[str, Any], power: dict[str, Any]) -> dict[str, Any]:
    result["power_mde"] = power
    result["power_status"] = power.get("status", "inconclusive_underpowered")
    result["status"] = "estimated" if result["power_status"] == "passed" else "inconclusive_underpowered"
    return result


def _youtube_alignment_frame(models: dict[str, Any]) -> pd.DataFrame:
    comments = _predicted_frame_documents("youtube", models)
    per_author = comments.groupby(["video_id", "author"], sort=True)[FRAME_LABELS].mean().reset_index()
    audience = per_author.groupby("video_id", sort=True)[FRAME_LABELS].mean().reset_index()
    comment_weighted_audience = comments.groupby("video_id", sort=True)[FRAME_LABELS].mean().reset_index()
    comment_weighted_audience = comment_weighted_audience.rename(columns={label: f"audience_comment_{label}" for label in FRAME_LABELS})
    audit = pd.read_csv(YOUTUBE_METADATA_AUDIT, keep_default_na=False)
    video_info = comments.groupby("video_id", sort=True).agg(event=("case", "first"), channel_id=("channel_id", "first")).reset_index()
    video_info = video_info.merge(audience, on="video_id", how="inner", validate="one_to_one").merge(comment_weighted_audience, on="video_id", how="inner", validate="one_to_one")
    rows = []
    for row in audit.itertuples(index=False):
        info = video_info[video_info["video_id"].eq(canonical_youtube_video_id(row.video_id))]
        if info.empty or str(row.adjudicated_source_type).casefold() not in {"news", "commentary"}:
            continue
        metadata = {label: int(label in str(row.adjudicated_frame).split("|")) for label in FRAME_LABELS}
        audience_values = info.iloc[0][FRAME_LABELS].to_dict()
        if sum(metadata.values()) == 0 or sum(float(audience_values[label]) for label in FRAME_LABELS) == 0:
            continue
        rows.append(
            {
                "video_id": canonical_youtube_video_id(row.video_id),
                "event": str(info.iloc[0]["event"]),
                "channel_id": str(info.iloc[0]["channel_id"]),
                "source_type": str(row.adjudicated_source_type).casefold(),
                **{f"metadata_{label}": value for label, value in metadata.items()},
                **{f"audience_{label}": float(audience_values[label]) for label in FRAME_LABELS},
                **{f"audience_comment_{label}": float(info.iloc[0][f"audience_comment_{label}"]) for label in FRAME_LABELS},
            }
        )
    return pd.DataFrame(rows)


def _execute_hypothesis(hypothesis_id: str, endpoint_ids: list[str]) -> dict[str, Any]:
    models = _primary_frame_models()
    results: dict[str, Any] = {}
    power: dict[str, Any] = {}
    if hypothesis_id == "H1":
        for endpoint in endpoint_ids:
            platform = "reddit" if endpoint == "H1-R" else "bluesky"
            profiles = _author_profiles(platform, models)
            result = author_vector_permutation_test(
                profiles,
                FRAME_LABELS,
                community_column="community_id",
                case_column="case",
                degree_column="degree",
                activity_column="activity",
                document_count_column="documents",
                permutations=PERMUTATIONS,
            )
            result["population"] = "R_REPLY_CORE" if platform == "reddit" else "B_REPLY_CORE"
            _apply_power_gate(result, _power_for_units(endpoint, profiles[FRAME_LABELS].to_numpy().mean(axis=1), result["eligible_authors"]))
            results[endpoint] = result
            power[endpoint] = result["power_mde"]
    elif hypothesis_id == "H2":
        documents = _predicted_frame_documents("reddit", models)
        for endpoint in endpoint_ids:
            label = "privacy_surveillance" if endpoint == "H2-P" else "circumvention_censorship_autonomy"
            result = thread_cluster_bootstrap_difference(
                documents.assign(outcome=documents[label].astype(float)),
                "outcome",
                "AU_IMPLEMENTATION",
                "AU_LEGISLATION",
                author_column="author",
                case_column="case",
                cluster_column="thread",
                replicates=PERMUTATIONS,
            )
            result["population"] = "R_AU_IMPLEMENTATION_vs_R_AU_LEGISLATION"
            result["outcome"] = label
            _apply_power_gate(result, _power_for_units(endpoint, documents[label].to_numpy(), float(documents.groupby("case")["thread"].nunique().min())))
            results[endpoint] = result
            power[endpoint] = result["power_mde"]
    elif hypothesis_id == "H3":
        profiles = _author_profiles("reddit", models)
        profiles["entropy"] = profiles[FRAME_LABELS].apply(lambda row: normalized_frame_entropy(row.to_numpy(dtype=float)), axis=1)
        profiles = profiles[profiles["documents"].ge(5)].copy()
        matched, matching = match_broker_sets(profiles, author_column="author", case_column="case", outcome_column="entropy")
        if matching["matched_sets"] == 0:
            results["H3"] = {"status": "inconclusive_no_complete_matched_sets", "matching": matching}
        elif not matching.get("balance_passed", False):
            results["H3"] = {"status": "inconclusive_balance_failed", "matching": matching}
        else:
            result = broker_permutation_test(matched, outcome_column="entropy")
            result.update({"population": "R_BROKER_CORE", "matching": matching})
            _apply_power_gate(result, _power_for_units("H3", matched["entropy"].to_numpy(), result["matched_sets"]))
            results["H3"] = result
            power["H3"] = result["power_mde"]
    elif hypothesis_id == "H4":
        values = _youtube_alignment_frame(models)
        if values.empty:
            results["H4"] = {"status": "inconclusive_no_video_profiles", "population": "Y_VIDEO_E2E3"}
        else:
            result = video_alignment_contrast(
                values,
                [f"metadata_{label}" for label in FRAME_LABELS],
                [f"audience_{label}" for label in FRAME_LABELS],
                video_column="video_id",
                event_column="event",
                channel_column="channel_id",
                source_column="source_type",
                permutations=PERMUTATIONS,
            )
            result["population"] = "Y_VIDEO_E2E3"
            _apply_power_gate(result, _power_for_units("H4", values[[f"metadata_{label}" for label in FRAME_LABELS]].to_numpy().mean(axis=1), result["channel_count"]))
            results["H4"] = result
            power["H4"] = result["power_mde"]
    else:
        raise ValueError(f"unknown hypothesis: {hypothesis_id}")
    return {"status": "estimated" if all(result.get("status") == "estimated" for result in results.values()) else "inconclusive", "results": results, "power_mde": power}


def _request_variant(hypothesis_id: str, platform: str | None, outcome: str | None) -> tuple[dict[str, str], str, list[str]]:
    requested_platform = platform or ({"H1": "both", "H2": "reddit", "H3": "reddit", "H4": "youtube"}[hypothesis_id])
    if hypothesis_id == "H1" and requested_platform not in {"reddit", "bluesky", "both"}:
        raise ValueError("H1 platform must be reddit, bluesky, or both")
    if hypothesis_id in {"H2", "H3"} and requested_platform != "reddit":
        raise ValueError(f"{hypothesis_id} is Reddit-only")
    if hypothesis_id == "H4" and requested_platform != "youtube":
        raise ValueError("H4 is YouTube-only")
    outcome_aliases = {"privacy": "privacy_surveillance", "privacy_surveillance": "privacy_surveillance", "circumvention_autonomy": "circumvention_censorship_autonomy", "circumvention_censorship_autonomy": "circumvention_censorship_autonomy"}
    requested_outcome = outcome_aliases.get(outcome or "both", outcome or "both")
    if hypothesis_id == "H2" and requested_outcome not in {"privacy_surveillance", "circumvention_censorship_autonomy", "both"}:
        raise ValueError("H2 outcome must be privacy or circumvention_autonomy")
    if hypothesis_id != "H2" and outcome is not None:
        raise ValueError(f"--outcome is only supported for H2, got {hypothesis_id}")
    endpoint_ids = list(HYPOTHESIS_SPECS[hypothesis_id]["endpoints"])
    if hypothesis_id == "H1" and requested_platform != "both":
        endpoint_ids = ["H1-R" if requested_platform == "reddit" else "H1-B"]
    if hypothesis_id == "H2" and requested_outcome != "both":
        endpoint_ids = ["H2-P" if requested_outcome == "privacy_surveillance" else "H2-C"]
    return {"platform": requested_platform, "outcome": requested_outcome}, requested_platform if hypothesis_id != "H2" else (requested_outcome if requested_outcome != "both" else "reddit"), endpoint_ids


ESTIMATOR_CONTRACTS = {
    "H1": {"primary": "author_vector_permutation_test", "secondary": ["dominant_frame_assortativity", "edge_profile_similarity", "community_entropy"]},
    "H2": {"primary": "thread_cluster_bootstrap_difference", "secondary": ["strict_language", "inclusive_language", "document_weighted"]},
    "H3": {"primary": "match_broker_sets + broker_permutation_test", "secondary": ["alternative_broker_thresholds", "hub_sensitivity"]},
    "H4": {"primary": "video_alignment_contrast", "secondary": ["comment_weighted", "leave_one_video_out", "leave_one_channel_out"]},
}


def _artifact(hypothesis_id: str, platform: str | None = None, outcome: str | None = None) -> dict[str, Any]:
    spec = HYPOTHESIS_SPECS[hypothesis_id]
    request, variant, endpoint_ids = _request_variant(hypothesis_id, platform, outcome)
    status, missing = _gate_status(hypothesis_id, request["platform"])
    human_fallback = _human_sample_fallback(hypothesis_id, request["platform"])
    power_endpoints = {}
    for endpoint in endpoint_ids:
        design = _frozen_power_design(endpoint)
        power_endpoints[endpoint] = (
            frozen_design_power_simulation(
                design["cluster_sizes"],
                design["topology"],
                design["source_type_counts"],
                PRIMARY_EFFECT_THRESHOLDS[endpoint],
                alternative="two-sided" if endpoint == "H4" else "one-sided",
                endpoint_id=endpoint,
                design_details=design,
            )
            if design
            else {
                "status": "frozen_design_inputs_unavailable",
                "target_power": 0.80,
                "minimum_effect": PRIMARY_EFFECT_THRESHOLDS[endpoint],
            }
        )
    power_statuses = {value["status"] for value in power_endpoints.values()}
    payload: dict[str, Any] = {
        "schema_version": HYPOTHESIS_SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "hypothesis_id": hypothesis_id,
        "family": PRIMARY_FAMILY,
        "endpoints": spec["endpoints"],
        "requested_endpoint_ids": endpoint_ids,
        "requested_variant": request,
        "variant_id": variant,
        "population_ids": spec["population_ids"],
        "estimand": spec["estimand"],
        "null": spec["null"],
        "permutations_or_bootstrap_replicates": PERMUTATIONS,
        "decision_threshold": spec["threshold"],
        "status": status,
        "missing_prerequisites": missing,
        "source_checksums": _input_checksums(),
        "estimator_contract": ESTIMATOR_CONTRACTS[hypothesis_id],
        "estimators_implemented": True,
        "execution_contract": {
            "status": "blocked_prerequisites" if status != "ready" else "ready_for_execution",
            "required_inputs": ["adjudicated_v2_labels", "selected_measurement_predictions", "declared_network_audit"],
        },
        "power_mde": {
            "status": "available" if power_endpoints and power_statuses == {"passed"} else "inconclusive_underpowered" if "inconclusive_underpowered" in power_statuses else "frozen_design_inputs_unavailable",
            "target_power": 0.80,
            "minimum_effects": {endpoint: PRIMARY_EFFECT_THRESHOLDS[endpoint] for endpoint in endpoint_ids},
            "endpoints": power_endpoints,
        },
        "result": None,
        "human_sample_fallback": human_fallback,
        "interpretation": "Automated confirmatory estimates are gated; the included fallback is a narrower descriptive estimate from the adjudicated reserve sample and is not a null or confirmatory endpoint result.",
    }
    if status == "ready":
        payload["status"] = "ready_for_execution"
    return payload


def _write_primary6_status() -> None:
    p_values: dict[str, float | None] = {}
    effects: dict[str, float | None] = {}
    intervals: dict[str, dict[str, float]] = {}
    variants = {
        "H1-R": ("h1.json", "h1_reddit.json"),
        "H1-B": ("h1.json", "h1_bluesky.json"),
        "H2-P": ("h2.json", "h2_privacy_surveillance.json"),
        "H2-C": ("h2.json", "h2_circumvention_censorship_autonomy.json"),
        "H3": ("h3.json",),
        "H4": ("h4.json",),
    }
    source_paths = {}
    for endpoint, filenames in variants.items():
        endpoint_result = None
        for filename in filenames:
            path = HYPOTHESIS_ROOT / filename
            source_paths[filename] = path
            if not path.exists():
                continue
            result = read_json(path).get("result")
            candidate = result.get(endpoint, result) if isinstance(result, dict) else None
            if _is_confirmatory_result(candidate):
                endpoint_result = candidate
                break
        p_values[endpoint] = endpoint_result.get("p_value") if isinstance(endpoint_result, dict) else None
        effects[endpoint] = endpoint_result.get("excess", endpoint_result.get("observed")) if isinstance(endpoint_result, dict) else None
        if isinstance(endpoint_result, dict):
            bounds = {bound: endpoint_result.get(bound) for bound in ("lower_95", "upper_95") if endpoint_result.get(bound) is not None}
            if bounds:
                intervals[endpoint] = {bound: float(value) for bound, value in bounds.items()}
    decision = primary6_decision(p_values, effects)
    decision["intervals"] = intervals
    payload = {"schema_version": "primary6.v1", "family": PRIMARY_FAMILY, "endpoints": list(PRIMARY_ENDPOINTS), "decision": decision, "source_checksums": {name: sha256_file(path) if path.exists() else None for name, path in source_paths.items()}, "status": "blocked_until_all_six_results_exist" if decision["missing_endpoints"] else "complete"}
    write_json(PRIMARY6_ARTIFACT, payload)


def _hypothesis_path(hypothesis_id: str, platform: str | None, outcome: str | None):
    request, variant, _ = _request_variant(hypothesis_id, platform, outcome)
    suffix = "" if platform is None and outcome is None else f"_{variant}"
    return HYPOTHESIS_ROOT / f"{hypothesis_id.lower()}{suffix}.json", request


def build_hypothesis_artifact(hypothesis_id: str, platform: str | None = None, outcome: str | None = None) -> dict[str, Any]:
    if hypothesis_id not in HYPOTHESIS_SPECS:
        raise ValueError(f"unknown hypothesis: {hypothesis_id}")
    HYPOTHESIS_ROOT.mkdir(parents=True, exist_ok=True)
    path, _ = _hypothesis_path(hypothesis_id, platform, outcome)
    payload = _artifact(hypothesis_id, platform, outcome)
    write_json(path, payload)
    _write_primary6_status()
    return {"status": payload["status"], "artifact": relative_path(path), "missing_prerequisites": payload.get("missing_prerequisites", [])}


def check_hypothesis(hypothesis_id: str, platform: str | None = None, outcome: str | None = None) -> dict[str, Any]:
    if hypothesis_id not in HYPOTHESIS_SPECS:
        raise ValueError(f"unknown hypothesis: {hypothesis_id}")
    path, request = _hypothesis_path(hypothesis_id, platform, outcome)
    if not path.exists():
        raise FileNotFoundError(f"hypothesis artifact is missing: {path}; run the explicit hypothesis build first")
    payload = read_json(path)
    if (
        payload.get("schema_version") != HYPOTHESIS_SCHEMA_VERSION
        or payload.get("source_checksums") != _input_checksums()
        or payload.get("requested_variant") != request
        or payload.get("execution_contract") is None
        or payload.get("estimators_implemented") is not True
        or payload.get("human_sample_fallback") is None
    ):
        raise ValueError(f"hypothesis artifact is stale or incomplete: {path}; run the explicit hypothesis build")
    return {"status": payload["status"], "artifact": relative_path(path), "missing_prerequisites": payload.get("missing_prerequisites", [])}


def execute_hypothesis(hypothesis_id: str, platform: str | None = None, outcome: str | None = None) -> dict[str, Any]:
    check_hypothesis(hypothesis_id, platform, outcome)
    path, _ = _hypothesis_path(hypothesis_id, platform, outcome)
    payload = read_json(path)
    if payload["status"] != "ready_for_execution":
        return {"status": payload["status"], "artifact": relative_path(path), "missing_prerequisites": payload.get("missing_prerequisites", [])}
    if payload["status"] == "ready_for_execution" and payload.get("result") is None:
        try:
            execution = _execute_hypothesis(hypothesis_id, payload["requested_endpoint_ids"])
        except ValueError as error:
            payload["status"] = "inconclusive_execution_gate"
            payload["blocking_reason"] = str(error)
            payload["execution_contract"].update({"status": "inconclusive", "permutations": PERMUTATIONS})
        else:
            payload["result"] = execution["results"]
            payload["power_mde"] = execution["power_mde"]
            payload["execution_contract"].update({"status": execution["status"], "permutations": PERMUTATIONS})
            payload["status"] = "complete" if all(
                _is_confirmatory_result(execution["results"].get(endpoint))
                for endpoint in payload["requested_endpoint_ids"]
            ) else "inconclusive"
        write_json(path, payload)
    _write_primary6_status()
    return {"status": payload["status"], "artifact": relative_path(path), "missing_prerequisites": payload.get("missing_prerequisites", [])}
