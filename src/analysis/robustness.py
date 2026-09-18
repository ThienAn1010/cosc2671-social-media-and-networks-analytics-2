"""Predeclared, estimand-specific robustness specifications."""

from __future__ import annotations

import json
from typing import Any, Callable

import networkx as nx
import pandas as pd

from src.analysis.artifacts import ANALYSIS_ROOT, REPO_ROOT, read_json, relative_path, sha256_file, utc_now_iso, write_json
from src.analysis.hypotheses import (
    FRAME_LABELS,
    HYPOTHESIS_ROOT,
    HYPOTHESIS_SPECS,
    author_vector_permutation_test,
    broker_permutation_test,
    normalized_frame_entropy,
    _author_profiles,
    _primary_frame_models,
    _predicted_frame_documents,
    _youtube_alignment_frame,
    match_broker_sets,
    thread_cluster_bootstrap_difference,
    video_alignment_contrast,
)

ROBUSTNESS_ROOT = ANALYSIS_ROOT / "robustness"
ROBUSTNESS_MANIFEST = ROBUSTNESS_ROOT / "robustness_manifest.json"

SPECIFICATIONS = [
    {"hypothesis_id": "H1", "variant_id": "primary", "language_population": "human_calibrated", "graph_view": "frozen_primary_partition", "weighting": "activity_weighted", "decisive": True},
    {"hypothesis_id": "H1", "variant_id": "strict_language", "language_population": "strict_english", "graph_view": "frozen_primary_partition", "weighting": "activity_weighted", "decisive": True},
    {"hypothesis_id": "H1", "variant_id": "inclusive_language", "language_population": "inclusive_uncertain_bound", "graph_view": "frozen_primary_partition", "weighting": "activity_weighted", "decisive": True},
    {"hypothesis_id": "H1", "variant_id": "louvain_sensitivity", "language_population": "strict_english", "graph_view": "louvain_cross_check", "weighting": "activity_weighted", "decisive": False},
    {"hypothesis_id": "H2", "variant_id": "privacy_primary", "outcome": "privacy_surveillance", "language_population": "human_calibrated", "weighting": "author_balanced", "thread_population": "audited_relevant", "decisive": True},
    {"hypothesis_id": "H2", "variant_id": "circumvention_primary", "outcome": "circumvention_censorship_autonomy", "language_population": "human_calibrated", "weighting": "author_balanced", "thread_population": "audited_relevant", "decisive": True},
    {"hypothesis_id": "H2", "variant_id": "strict_keyword_sensitivity", "outcome": "legacy_revalidated", "language_population": "strict_english", "weighting": "document_weighted", "thread_population": "keyword_screen", "decisive": False},
    {"hypothesis_id": "H3", "variant_id": "primary", "broker_threshold": "top_decile", "comparison_threshold": "below_case_median", "matching": "1_to_3_mahalanobis_calipers", "outcome": "normalized_frame_entropy", "decisive": True},
    {"hypothesis_id": "H3", "variant_id": "top_quintile_sensitivity", "broker_threshold": "top_quintile", "comparison_threshold": "below_case_median", "matching": "1_to_3_mahalanobis_calipers", "outcome": "normalized_frame_entropy", "decisive": False},
    {"hypothesis_id": "H3", "variant_id": "top_five_percent_sensitivity", "broker_threshold": "top_five_percent", "comparison_threshold": "below_case_median", "matching": "1_to_3_mahalanobis_calipers", "outcome": "normalized_frame_entropy", "decisive": False},
    {"hypothesis_id": "H4", "variant_id": "primary", "audience_weighting": "contributor_balanced", "reply_resolution": "handle_match_secondary_not_endpoint", "video_scope": "selected_E2_E3", "decisive": True},
    {"hypothesis_id": "H4", "variant_id": "comment_weighted_sensitivity", "audience_weighting": "comment_weighted", "reply_resolution": "handle_match_secondary_not_endpoint", "video_scope": "selected_E2_E3", "decisive": False},
    {"hypothesis_id": "H4", "variant_id": "leave_one_video_out", "audience_weighting": "contributor_balanced", "reply_resolution": "handle_match_secondary_not_endpoint", "video_scope": "selected_E2_E3_leave_one_video_out", "decisive": False},
    {"hypothesis_id": "H4", "variant_id": "leave_one_channel_out", "audience_weighting": "contributor_balanced", "reply_resolution": "handle_match_secondary_not_endpoint", "video_scope": "selected_E2_E3_leave_one_channel_out", "decisive": False},
]
PRIMARY_VARIANT_IDS = {
    "H1": ("primary",),
    "H2": ("privacy_primary", "circumvention_primary"),
    "H3": ("primary",),
    "H4": ("primary",),
}
DECISIVE_VARIANT_IDS = {
    hypothesis_id: tuple(item["variant_id"] for item in SPECIFICATIONS if item["hypothesis_id"] == hypothesis_id and item["decisive"])
    for hypothesis_id in PRIMARY_VARIANT_IDS
}


def _specification(hypothesis_id: str, variant_id: str) -> dict[str, Any]:
    for item in SPECIFICATIONS:
        if item["hypothesis_id"] == hypothesis_id and item["variant_id"] == variant_id:
            return dict(item)
    raise ValueError(f"unknown predeclared robustness variant: {hypothesis_id}/{variant_id}")


def _apply_variant_specification(frame: pd.DataFrame, spec: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply only the named filters; missing non-primary alternatives fail closed."""

    values = frame.copy()
    before = len(values)
    applied: dict[str, Any] = {}
    filters: dict[str, set[str]] = {}
    language = spec.get("language_population")
    if language and language != "human_calibrated":
        column = next((name for name in ("language_population", "language_status", "language") if name in values), None)
        if column is None:
            raise ValueError(f"{spec['hypothesis_id']}/{spec['variant_id']} requires {language} input")
        allowed = {
            "strict_english": {"strict_english", "strict", "english"},
            "inclusive_uncertain_bound": {"inclusive_uncertain_bound", "inclusive", "uncertain"},
        }.get(language)
        if allowed is None:
            raise ValueError(f"unsupported language population: {language}")
        filters[column] = allowed
    thread_population = spec.get("thread_population")
    if thread_population:
        preferred = ("keyword_screen", "thread_population", "thread_status") if thread_population == "keyword_screen" else ("thread_population", "thread_status")
        column = next((name for name in preferred if name in values), None)
        if column is None:
            if thread_population != "audited_relevant":
                raise ValueError(f"{spec['hypothesis_id']}/{spec['variant_id']} requires {thread_population} input")
            applied["thread_population"] = "authoritative_pre_filtered_input"
        else:
            filters[column] = {thread_population}
    graph_view = spec.get("graph_view")
    if graph_view == "louvain_cross_check" and "graph_view" not in values:
        raise ValueError("louvain sensitivity requires the frozen graph-view assignment")
    if "graph_view" in values and graph_view:
        filters["graph_view"] = {graph_view}
    for column, allowed in filters.items():
        values = values[values[column].astype(str).isin(allowed)].copy()
        applied[column] = sorted(allowed)
    video_scope = spec.get("video_scope", "")
    if video_scope.startswith("selected_E2_E3"):
        column = next((name for name in ("event", "event_id", "video_event") if name in values), None)
        if column is None:
            raise ValueError("H4 video scope requires an authoritative event column")
        values = values[values[column].astype(str).isin({"E2", "E3"})].copy()
        applied[column] = ["E2", "E3"]
    if video_scope.endswith("leave_one_channel_out") or video_scope.endswith("leave_one_video_out"):
        unit_column = "channel_id" if video_scope.endswith("leave_one_channel_out") else "video_id"
        if unit_column not in values:
            raise ValueError(f"leave-one-out requires {unit_column}")
        units = sorted(values[unit_column].astype(str).unique())
        if not units:
            raise ValueError(f"leave-one-out has no {unit_column}")
        applied["leave_out_unit"] = unit_column
        applied["leave_out_units"] = units
    if values.empty:
        raise ValueError(f"{spec['hypothesis_id']}/{spec['variant_id']} has no rows after declared filters")
    return values.reset_index(drop=True), {"rows_before": int(before), "rows_after": int(len(values)), "filters": applied}
def _run_h1(frame: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    columns = spec.get("profile_columns") or FRAME_LABELS
    return author_vector_permutation_test(frame, columns, permutations=int(spec.get("permutations", 9_999)))


def _run_h2(frame: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    outcome = spec.get("outcome_column") or (spec.get("outcome") if spec.get("outcome") in frame.columns else "outcome")
    return thread_cluster_bootstrap_difference(
        frame,
        outcome,
        spec.get("later", "AU_IMPLEMENTATION"),
        spec.get("earlier", "AU_LEGISLATION"),
        author_column=spec.get("author_column", "author"),
        case_column=spec.get("case_column", "case"),
        cluster_column=spec.get("cluster_column", "thread"),
        weighting=spec.get("weighting", "author_balanced"),
        replicates=int(spec.get("replicates", 9_999)),
    )


def _run_h3(frame: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    quantiles = {"top_decile": 0.90, "top_quintile": 0.80, "top_five_percent": 0.95}
    matched, matching = match_broker_sets(
        frame,
        outcome_column=spec.get("outcome_column") or (spec.get("outcome") if spec.get("outcome") in frame.columns else "entropy"),
        broker_quantile=quantiles[spec.get("broker_threshold", "top_decile")],
        comparison_quantile=0.50,
        matching=spec.get("matching", "1_to_3_mahalanobis_calipers"),
    )
    if not matching.get("balance_passed", False):
        return {"status": "inconclusive_balance_failed", "matching": matching}
    result = broker_permutation_test(matched, permutations=int(spec.get("permutations", 9_999)))
    result["matching"] = matching
    return result


def _run_h4(frame: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    metadata = spec.get("metadata_profile_columns") or [f"metadata_{label}" for label in FRAME_LABELS]
    if spec.get("audience_weighting") == "comment_weighted":
        audience = spec.get("audience_profile_columns") or [f"audience_comment_{label}" for label in FRAME_LABELS]
    else:
        audience = spec.get("audience_profile_columns") or [f"audience_{label}" for label in FRAME_LABELS]
    permutations = int(spec.get("permutations", 9_999))
    leave_out_unit = spec.get("video_scope", "")
    if leave_out_unit.endswith("leave_one_channel_out") or leave_out_unit.endswith("leave_one_video_out"):
        unit_column = "channel_id" if leave_out_unit.endswith("leave_one_channel_out") else "video_id"
        units = sorted(frame[unit_column].astype(str).unique())
        results = []
        for excluded in units:
            subset = frame[frame[unit_column].astype(str).ne(excluded)].copy()
            result = video_alignment_contrast(frame=subset, metadata_profile_columns=metadata, audience_profile_columns=audience, permutations=permutations)
            results.append({"excluded_unit": excluded, **result})
        effects = [float(result["observed"]) for result in results]
        lower = [float(result["lower_95"]) for result in results]
        upper = [float(result["upper_95"]) for result in results]
        p_values = [float(result["p_value"]) for result in results]
        decisions = [p < 0.05 and abs(effect) >= 0.05 and (lo > 0 or hi < 0) for p, effect, lo, hi in zip(p_values, effects, lower, upper)]
        return {
            "observed": float(pd.Series(effects).median()),
            "lower_95": min(lower),
            "median": float(pd.Series(effects).median()),
            "upper_95": max(upper),
            "p_value": max(p_values),
            "decision_stable": len(set(decisions)) == 1,
            "leave_out_unit": unit_column,
            "excluded_units": units,
            "leave_out_results": results,
            "leave_out_decisions": decisions,
            "permutations": permutations,
            "status": "available",
        }
    return video_alignment_contrast(frame, metadata, audience, permutations=permutations)


VARIANT_RUNNERS: dict[str, Callable[[pd.DataFrame, dict[str, Any]], dict[str, Any]]] = {
    "H1": _run_h1,
    "H2": _run_h2,
    "H3": _run_h3,
    "H4": _run_h4,
}
RUNNER_NAMES = {
    "H1": "author_vector_permutation_test",
    "H2": "thread_cluster_bootstrap_difference",
    "H3": "match_broker_sets + broker_permutation_test",
    "H4": "video_alignment_contrast",
}


def summarize_variant_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize named robustness results by direction, magnitude and decision."""

    required = {"hypothesis_id", "variant_id", "effect", "p_value"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"robustness results missing columns: {missing}")
    output = results.copy()
    effect = pd.to_numeric(output["effect"], errors="coerce")
    p_value = pd.to_numeric(output["p_value"], errors="coerce")
    output["direction"] = effect.map(lambda value: "positive" if value > 0 else "negative" if value < 0 else "zero" if value == 0 else "unavailable")
    threshold = output["hypothesis_id"].map({"H1": 0.05, "H2": 0.05, "H3": 0.10, "H4": 0.05}).fillna(0.05)
    meaningful = effect.ge(threshold).where(output["hypothesis_id"].isin({"H1", "H2", "H3"}), effect.abs().ge(threshold))
    output["passes_nominal_gate"] = p_value.lt(0.05) & meaningful
    for column in ("lower_95", "upper_95", "decision", "interval_excludes_zero", "decision_stable"):
        if column not in output:
            output[column] = None
    lower = pd.to_numeric(output["lower_95"], errors="coerce")
    upper = pd.to_numeric(output["upper_95"], errors="coerce")
    inferred_interval = (lower > 0) | (upper < 0)
    output["interval_excludes_zero"] = output["interval_excludes_zero"].where(lower.notna() & upper.notna(), inferred_interval)
    return output


def execute_named_variant(hypothesis_id: str, variant_id: str, frame: pd.DataFrame, **options: Any) -> dict[str, Any]:
    """Dispatch one predeclared variant through its estimand-specific estimator."""

    if hypothesis_id not in VARIANT_RUNNERS:
        raise ValueError(f"unknown robustness hypothesis: {hypothesis_id}")
    specification = _specification(hypothesis_id, variant_id)
    specification.update(options)
    prepared, input_contract = _apply_variant_specification(frame, specification)
    result = VARIANT_RUNNERS[hypothesis_id](prepared, specification)
    result.update({"hypothesis_id": hypothesis_id, "variant_id": variant_id, "variant_spec": specification, "input_contract": input_contract})
    return result


def _input_checksums() -> dict[str, str | None]:
    paths = {
        "population_manifest": ANALYSIS_ROOT / "populations" / "population_manifest.json",
        "network_manifest": ANALYSIS_ROOT / "networks" / "network_manifest.json",
        "structure_manifest": ANALYSIS_ROOT / "networks" / "structure" / "structure_manifest.json",
        "topic_manifest": ANALYSIS_ROOT / "topics" / "topics_manifest.json",
        "timing_spec": ANALYSIS_ROOT / "extended" / "timing.json",
        "prediction_spec": ANALYSIS_ROOT / "extended" / "prediction.json",
        "h1": HYPOTHESIS_ROOT / "h1.json",
        "h1_reddit": HYPOTHESIS_ROOT / "h1_reddit.json",
        "h1_bluesky": HYPOTHESIS_ROOT / "h1_bluesky.json",
        "h2": HYPOTHESIS_ROOT / "h2.json",
        "h2_privacy": HYPOTHESIS_ROOT / "h2_privacy_surveillance.json",
        "h2_circumvention": HYPOTHESIS_ROOT / "h2_circumvention_censorship_autonomy.json",
        "h3": HYPOTHESIS_ROOT / "h3.json",
        "h4": HYPOTHESIS_ROOT / "h4.json",
        "primary6": HYPOTHESIS_ROOT / "primary6.json",
    }
    return {name: sha256_file(path) if path.exists() else None for name, path in paths.items()}


def _louvain_profiles(platform: str, models: dict[str, Any], language_population: str) -> pd.DataFrame:
    profiles = _author_profiles(platform, models, language_population)
    graph = pd.read_parquet(
        REPO_ROOT / "data" / "analysis" / "networks" / ("r_actor_attention.parquet" if platform == "reddit" else "b_actor_attention.parquet"),
        columns=["scope", "source_author_hash", "target_author_hash", "weight"],
    )
    memberships: dict[tuple[str, str], str] = {}
    for scope, group in graph.groupby("scope", sort=True):
        network = nx.Graph()
        for row in group.itertuples(index=False):
            source, target = str(row.source_author_hash), str(row.target_author_hash)
            if source and target and source != target:
                network.add_edge(source, target, weight=float(row.weight))
        communities = nx.community.louvain_communities(network, weight="weight", seed=20260915) if network.number_of_nodes() else []
        for index, members in enumerate(sorted(communities, key=lambda values: min(values) if values else "")):
            for author in members:
                memberships[(str(scope), str(author))] = f"louvain-{index:04d}"
    profiles["community_id"] = [memberships.get((str(row.case), str(row.author))) for row in profiles.itertuples(index=False)]
    profiles = profiles[profiles["community_id"].notna()].copy()
    if profiles.empty:
        raise ValueError("louvain cross-check has no labelled author memberships")
    profiles["graph_view"] = "louvain_cross_check"
    return profiles


def _default_variant_frame(spec: dict[str, Any]) -> pd.DataFrame:
    models = _primary_frame_models()
    if spec["hypothesis_id"] == "H1":
        profiles = []
        for platform in ("reddit", "bluesky"):
            language = spec.get("language_population", "human_calibrated")
            if spec.get("graph_view") == "louvain_cross_check":
                profiles.append(_louvain_profiles(platform, models, language))
            else:
                values = _author_profiles(platform, models, language)
                values["graph_view"] = "frozen_primary_partition"
                profiles.append(values)
        return pd.concat(profiles, ignore_index=True)
    if spec["hypothesis_id"] == "H2":
        return _predicted_frame_documents("reddit", models)
    if spec["hypothesis_id"] == "H3":
        values = _author_profiles("reddit", models)
        values["entropy"] = values[FRAME_LABELS].apply(lambda row: normalized_frame_entropy(row.to_numpy(dtype=float)), axis=1)
        return values[values["documents"].ge(5)].copy()
    if spec["hypothesis_id"] == "H4":
        return _youtube_alignment_frame(models)
    raise ValueError(f"unknown robustness hypothesis: {spec['hypothesis_id']}")


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _result_row(spec: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    effect = _finite_number(result.get("excess", result.get("observed")))
    lower = _finite_number(result.get("lower_95"))
    upper = _finite_number(result.get("upper_95"))
    p_value = _finite_number(result.get("p_value"))
    interval_excludes_zero = (lower > 0 or upper < 0) if lower is not None and upper is not None else None
    threshold = {"H1": 0.05, "H2": 0.05, "H3": 0.10, "H4": 0.05}[spec["hypothesis_id"]]
    if effect is None or p_value is None:
        decision = None
    elif spec["hypothesis_id"] == "H4":
        decision = bool(p_value < 0.05 and abs(effect) >= threshold and interval_excludes_zero is True)
    else:
        decision = bool(p_value < 0.05 and effect >= threshold and (interval_excludes_zero in {True, None}))
    return {
        "hypothesis_id": spec["hypothesis_id"],
        "variant_id": spec["variant_id"],
        "runner": RUNNER_NAMES[spec["hypothesis_id"]],
        "effect": effect,
        "lower_95": lower,
        "upper_95": upper,
        "p_value": p_value,
        "decision": decision,
        "interval_excludes_zero": interval_excludes_zero,
        "decision_stable": result.get("decision_stable"),
        "leave_out_unit": result.get("leave_out_unit"),
        "excluded_units": json.dumps(result.get("excluded_units", []), sort_keys=True),
        "leave_out_results": json.dumps(result.get("leave_out_results", []), sort_keys=True),
        "status": "available" if effect is not None and p_value is not None else result.get("status", "unavailable"),
        "variant_spec": json.dumps(spec, sort_keys=True),
        "input_contract": json.dumps(result.get("input_contract", {}), sort_keys=True),
    }


def build_robustness_artifacts(variant_frames: dict[str, pd.DataFrame] | None = None) -> dict[str, Any]:
    ROBUSTNESS_ROOT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(SPECIFICATIONS)
    hypothesis_status = {}
    hypothesis_checksums = {}
    for hypothesis_id in HYPOTHESIS_SPECS:
        path = HYPOTHESIS_ROOT / f"{hypothesis_id.lower()}.json"
        hypothesis_status[hypothesis_id] = read_json(path).get("status") if path.exists() else "not_checked"
        hypothesis_checksums[hypothesis_id] = sha256_file(path) if path.exists() else None
    frame["required_endpoint"] = frame.apply(
        lambda row: "H1-R,H1-B" if row["hypothesis_id"] == "H1" else "H2-P" if row["hypothesis_id"] == "H2" and row.get("outcome") == "privacy_surveillance" else "H2-C" if row["hypothesis_id"] == "H2" and row.get("outcome") == "circumvention_censorship_autonomy" else row["hypothesis_id"],
        axis=1,
    )
    frame["runner"] = frame["hypothesis_id"].map(RUNNER_NAMES)
    frame["result_status"] = frame["hypothesis_id"].map(lambda value: "ready_for_named_variant_runner" if hypothesis_status[value] in {"ready_for_execution", "complete"} else "blocked_primary_prerequisite")
    frame["fishing_control"] = "predeclared_named_variant"
    output = ROBUSTNESS_ROOT / "specification_matrix.csv"
    frame.to_csv(output, index=False, lineterminator="\n")
    result_rows = []
    for spec in SPECIFICATIONS:
        key = f"{spec['hypothesis_id']}/{spec['variant_id']}"
        frame_input = (variant_frames or {}).get(key) if variant_frames is not None else None
        if frame_input is None and variant_frames is not None:
            frame_input = (variant_frames or {}).get(spec["variant_id"])
        prerequisite_ready = hypothesis_status[spec["hypothesis_id"]] in {"ready_for_execution", "complete"}
        if not prerequisite_ready:
            result_rows.append({**_result_row(spec, {"status": "blocked_primary_prerequisite"}), "status": "blocked_primary_prerequisite"})
            continue
        try:
            frame_input = frame_input if frame_input is not None else _default_variant_frame(spec)
            result = execute_named_variant(spec["hypothesis_id"], spec["variant_id"], frame_input)
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            result_rows.append({**_result_row(spec, {"status": "variant_input_required"}), "status": "variant_input_required", "blocking_reason": str(error)})
        else:
            result_rows.append(_result_row(spec, result))
    result_frame = pd.DataFrame(result_rows)
    for hypothesis_id, group in result_frame.groupby("hypothesis_id", sort=False):
        required_variants = set(DECISIVE_VARIANT_IDS[hypothesis_id])
        decisions = group.loc[group["variant_id"].isin(required_variants) & group["decision"].notna(), ["variant_id", "decision"]]
        if set(decisions["variant_id"]) != required_variants:
            continue
        stable = True if hypothesis_id == "H2" else bool(decisions["decision"].astype(bool).nunique() == 1)
        result_frame.loc[group.index[group["variant_id"].isin(required_variants)], "decision_stable"] = stable
    result_output = ROBUSTNESS_ROOT / "variant_results.csv"
    result_frame.to_csv(result_output, index=False, lineterminator="\n")
    executed_count = int(result_frame["status"].eq("available").sum())
    manifest = {
        "schema_version": "robustness.v3",
        "created_at_utc": utc_now_iso(),
        "command": "python -m src.analysis.run robustness --check",
        "specification_count": int(len(frame)),
        "decisive_specification_count": int(frame["decisive"].sum()),
        "no_cartesian_product": True,
        "hypothesis_status": hypothesis_status,
        "hypothesis_checksums": hypothesis_checksums,
        "input_checksums": _input_checksums(),
        "runner_registry": {hypothesis_id: {"name": name, "callable": VARIANT_RUNNERS[hypothesis_id].__name__, "status": "implemented"} for hypothesis_id, name in RUNNER_NAMES.items()},
        "outputs": {
            "specification_matrix": {"path": relative_path(output), "sha256": sha256_file(output), "rows": int(len(frame))},
            "variant_results": {"path": relative_path(result_output), "sha256": sha256_file(result_output), "rows": int(len(result_frame))},
        },
        "executed_variant_count": executed_count,
        "status": "complete" if executed_count == len(SPECIFICATIONS) else "partially_executed" if executed_count else "blocked_until_prerequisites_are_available",
        "limitations": [
            "Only the predeclared named variants are dispatched; H4 leave-one-out variants enumerate every video or channel unit.",
            "Unavailable prerequisites remain numerically empty and are not treated as null evidence.",
        ],
    }
    write_json(ROBUSTNESS_MANIFEST, manifest)
    return manifest


def check_robustness() -> dict[str, Any]:
    if not ROBUSTNESS_MANIFEST.exists():
        raise FileNotFoundError(f"robustness manifest is missing: {ROBUSTNESS_MANIFEST}; run the explicit robustness build first")
    manifest = read_json(ROBUSTNESS_MANIFEST)
    if manifest.get("schema_version") != "robustness.v3":
        raise ValueError("robustness manifest schema is stale; run the explicit robustness build")
    if manifest.get("input_checksums") != _input_checksums():
        raise ValueError("robustness inputs changed; run the explicit robustness build")
    for record in manifest.get("outputs", {}).values():
        output = REPO_ROOT / record["path"]
        if not output.exists() or sha256_file(output) != record["sha256"]:
            raise ValueError(f"robustness artifact changed: {record['path']}")
    result_path = REPO_ROOT / manifest["outputs"]["variant_results"]["path"]
    results = pd.read_csv(result_path)
    required = {"hypothesis_id", "variant_id", "effect", "lower_95", "upper_95", "p_value", "decision", "interval_excludes_zero", "decision_stable", "status"}
    if len(results) != len(SPECIFICATIONS) or not required <= set(results.columns):
        raise ValueError("robustness variant results are incomplete")
    if set(zip(results["hypothesis_id"], results["variant_id"])) != {(item["hypothesis_id"], item["variant_id"]) for item in SPECIFICATIONS}:
        raise ValueError("robustness variant result registry is incomplete")
    blocked = results[~results["status"].eq("available")]
    if blocked[["effect", "lower_95", "upper_95", "p_value"]].notna().any().any():
        raise ValueError("blocked robustness variants must not contain numeric results")
    if set(manifest.get("runner_registry", {})) != set(RUNNER_NAMES):
        raise ValueError("robustness runner registry is incomplete")
    return {"status": manifest["status"], "artifact": relative_path(ROBUSTNESS_MANIFEST), "specifications": manifest["specification_count"]}
