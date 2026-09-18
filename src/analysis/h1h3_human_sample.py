"""Exploratory H1 to H3 on the small human-coded samples.

The planned H1 to H3 tests (src/analysis/hypotheses.py) read frames from validated model labels,
and no model passed validation. These runs use the settled human labels from the samples drawn
by src/analysis/h1h3_sample.py instead, with the pipeline's own statistics. Declared changes
from the plan, all reported with the results:

- H1: an author's profile averages their usable coded documents (at most two were sampled, so
  the planned minimum of three documents becomes one); degree and activity for the permutation
  strata come from the reply graph (in plus out strength, and replies written plus one); only
  communities with at least MIN_H1_AUTHORS usable authors count.
- H2: only the Reddit set R1 is used, stratified by subreddit; estimates are unweighted within
  the sample, with an r/australia-only sensitivity run.
- H3: entropy comes from each author's usable coded documents (at most four were sampled,
  against the planned five); only complete matched sets with one broker and three comparison
  authors are estimable, and the contrast is broker minus the comparison mean.

A document is usable when it's relevant, English and has at least one frame; Reddit documents
also need a thread the coders accepted. H2 also needs the Australian policy as target.

Usage:
    python -m src.analysis.h1h3_human_sample --build --h4-result data/analysis/h4_human/h4_result.json
    python -m src.analysis.h1h3_human_sample --build --labels-dir <dir> --output-dir <dir>   # trial runs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.artifacts import ANALYSIS_ROOT, REPO_ROOT, read_json, relative_path, sha256_file, write_json
from src.analysis.h4_human_sample import validate_h4_result
from src.analysis.h1h3_sample import H2_ESTIMAND, OUTPUT_ROOT as SAMPLE_ROOT
from src.analysis.hypotheses import (
    PERMUTATIONS,
    author_vector_permutation_test,
    holm_adjust,
    normalized_frame_entropy,
    thread_cluster_bootstrap_difference,
)

FRAME_LABELS = [
    "policy_assurance",
    "child_safety",
    "privacy_surveillance",
    "governance_platform_responsibility",
    "circumvention_censorship_autonomy",
]
SETTLED_ROOT = SAMPLE_ROOT / "settled"
OUTPUT_ROOT = ANALYSIS_ROOT / "h1h3_human"
REGISTER = SAMPLE_ROOT / "coordinator" / "h1h3_register.csv"
ROLES = ANALYSIS_ROOT / "networks" / "structure" / "roles.parquet"
SEED = 20260917
MIN_H1_AUTHORS = 3
H1_EXCESS = 0.05
H2_DIFFERENCE = 0.05
H3_DIFFERENCE = 0.10
H1H3_SCHEMA_VERSION = "h1h3-human-sample.v3"
H1H3_RESULT = OUTPUT_ROOT / "h1h3_result.json"
H1H3_OUTPUT_MANIFEST = OUTPUT_ROOT / "h1h3_manifest.json"
H4_RESULT = ANALYSIS_ROOT / "h4_human" / "h4_result.json"
SAMPLE_MANIFEST = SAMPLE_ROOT / "coordinator" / "h1h3_sample_manifest.json"
ANALYSIS_CODE_PATHS = {
    "h1h3_human_sample": Path(__file__),
    "h1h3_sample": Path(__file__).with_name("h1h3_sample.py"),
    "h4_human_sample": Path(__file__).with_name("h4_human_sample.py"),
    "hypotheses": Path(__file__).with_name("hypotheses.py"),
}


def _stored_path(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    return relative_path(resolved) if resolved.is_relative_to(REPO_ROOT) else str(resolved)


def _resolve_stored_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _analysis_code_checksums() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in ANALYSIS_CODE_PATHS.items()}


def _analysis_code_sha256(checksums: dict[str, str] | None = None) -> str:
    values = checksums or _analysis_code_checksums()
    encoded = "\n".join(f"{name}:{values[name]}" for name in sorted(values)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _analysis_config(labels_dir: Path, permutations: int, h4_result_path: Path | None) -> dict[str, Any]:
    return {
        "seed": SEED,
        "permutations": int(permutations),
        "min_h1_authors": MIN_H1_AUTHORS,
        "h1_excess": H1_EXCESS,
        "h2_difference": H2_DIFFERENCE,
        "h3_difference": H3_DIFFERENCE,
        "frame_labels": FRAME_LABELS,
        "h2_estimand": H2_ESTIMAND,
        "labels_dir": _stored_path(labels_dir),
        "h4_result_path": _stored_path(h4_result_path),
    }


def _analysis_config_sha256(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _output_record(path: Path) -> dict[str, Any]:
    return {"path": _stored_path(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _assert_finite(value: Any, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"H1-H4 result is stale or incomplete: {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{location}[{index}]")


def load_settled(labels_dir: Path) -> dict[str, pd.DataFrame]:
    """Settled labels per packet: item ids plus label columns, with frames as 0/1 columns."""

    frames = {}
    for packet in ("bluesky_h1b", "reddit_docs"):
        labels = pd.read_csv(labels_dir / f"{packet}.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
        for label in FRAME_LABELS:
            labels[label] = labels["frame_labels"].str.split("|").map(lambda values, label=label: int(label in values))
        labels["usable"] = labels["relevance"].eq("relevant") & labels["language"].eq("english") & labels[FRAME_LABELS].sum(axis=1).gt(0)
        frames[packet] = labels
    threads = pd.read_csv(labels_dir / "reddit_threads.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    frames["reddit_threads"] = threads.assign(accepted=threads["thread_relevant"].str.casefold().eq("yes"))
    return frames


def _settled_register(register: pd.DataFrame, labels: pd.DataFrame, packet: str) -> pd.DataFrame:
    expected = set(register["item_id"])
    actual = set(labels["item_id"])
    if expected != actual:
        raise ValueError(f"{packet} settlement coverage mismatch: missing {len(expected - actual)}, extra {len(actual - expected)}")
    return register.merge(labels, on="item_id", how="inner", validate="one_to_one")


def _documents(settled: dict[str, pd.DataFrame]) -> pd.DataFrame:
    register = pd.read_csv(REGISTER, dtype=str, keep_default_na=False)
    bluesky = _settled_register(register[register["platform"].eq("bluesky")], settled["bluesky_h1b"], "bluesky_h1b")
    reddit_register = register[register["platform"].eq("reddit")]
    reddit = _settled_register(reddit_register, settled["reddit_docs"], "reddit_docs")
    expected_threads = set(reddit_register["thread_item_id"])
    actual_threads = set(settled["reddit_threads"]["item_id"])
    if expected_threads != actual_threads:
        raise ValueError(f"reddit_threads settlement coverage mismatch: missing {len(expected_threads - actual_threads)}, extra {len(actual_threads - expected_threads)}")
    accepted = settled["reddit_threads"].set_index("item_id")["accepted"]
    reddit["thread_accepted"] = reddit["thread_item_id"].map(accepted).fillna(False).astype(bool)
    reddit["usable"] = reddit["usable"] & reddit["thread_accepted"]
    bluesky["thread_accepted"] = True
    return pd.concat([bluesky, reddit], ignore_index=True)


def _author_profiles(docs: pd.DataFrame) -> pd.DataFrame:
    usable = docs[docs["usable"]]
    profiles = usable.groupby(["scope", "community_id", "author_key"], sort=True)[FRAME_LABELS].mean()
    profiles["documents"] = usable.groupby(["scope", "community_id", "author_key"], sort=True).size()
    return profiles.reset_index()


def run_h1(docs: pd.DataFrame, graph: str, task: str, permutations: int) -> dict[str, Any]:
    sample = docs[docs["task"].eq(task)]
    profiles = _author_profiles(sample)
    roles = pd.read_parquet(ROLES, columns=["graph_id", "scope", "node_id", "in_strength", "out_strength"])
    roles = roles[roles["graph_id"].eq(graph)].rename(columns={"node_id": "author_key"})
    profiles = profiles.merge(roles, on=["scope", "author_key"], how="left", validate="one_to_one")
    profiles["degree"] = profiles["in_strength"].fillna(0) + profiles["out_strength"].fillna(0)
    profiles["activity"] = profiles["out_strength"].fillna(0) + 1
    sizes = profiles.groupby(["scope", "community_id"])["author_key"].transform("size")
    kept = profiles[sizes.ge(MIN_H1_AUTHORS)].copy()
    kept["community"] = kept["scope"] + ":" + kept["community_id"]
    summary = {
        "sampled_authors": int(sample["author_key"].nunique()),
        "usable_authors": int(len(profiles)),
        "authors_in_kept_communities": int(len(kept)),
        "communities_kept": int(kept["community"].nunique()),
        "usable_documents": int(sample["usable"].sum()),
        "sampled_documents": int(len(sample)),
    }
    if kept["community"].nunique() < 2:
        return {**summary, "status": "not_estimable", "reason": f"fewer than two communities with {MIN_H1_AUTHORS} or more usable authors"}
    result = author_vector_permutation_test(
        kept, FRAME_LABELS, community_column="community", case_column="scope", degree_column="degree",
        activity_column="activity", document_count_column="documents", min_documents=1,
        permutations=permutations, seed=SEED,
    )
    result.pop("profile_columns", None)
    meets = result["excess"] >= H1_EXCESS and result["p_value"] < 0.05
    return {**summary, "status": "estimated", **result, "meets_nominal_effect_and_p": bool(meets)}


def run_h2(docs: pd.DataFrame, permutations: int) -> dict[str, Any]:
    sample = docs[docs["task"].eq("H2")]
    usable = sample[sample["usable"] & sample["target_policy"].eq("AU_SOCIAL_MINIMUM_AGE")].rename(
        columns={"author_key": "author", "scope": "case", "cluster_id": "thread"})
    counts = usable.groupby("case").agg(documents=("item_id", "size"), threads=("thread", "nunique")).to_dict(orient="index")
    summary = {
        "estimand": H2_ESTIMAND,
        "weighting": "unweighted",
        "sampled_documents": int(len(sample)),
        "usable_documents": int(len(usable)),
        "by_window": counts,
    }
    results: dict[str, Any] = {}
    for endpoint, label in (("H2-P", "privacy_surveillance"), ("H2-C", "circumvention_censorship_autonomy")):
        if any(counts.get(case, {}).get("threads", 0) < 5 for case in ("AU_LEGISLATION", "AU_IMPLEMENTATION")):
            results[endpoint] = {"status": "not_estimable", "reason": "a window has fewer than five threads with usable documents"}
            continue
        runs = {}
        for name, frame in (("all_strata", usable), ("australia_only", usable[usable["stratum_subreddit"].eq("australia")])):
            runs[name] = thread_cluster_bootstrap_difference(
                frame.assign(outcome=frame[label].astype(float)), "outcome", "AU_IMPLEMENTATION", "AU_LEGISLATION",
                author_column="author", case_column="case", cluster_column="thread", weighting="unweighted",
                replicates=permutations, seed=SEED)
            runs[name]["selected_sample_prevalence"] = frame.groupby("case")[label].mean().round(4).to_dict()
        main = runs["all_strata"]
        results[endpoint] = {
            "status": "estimated", "outcome": label, "estimand": H2_ESTIMAND, "weighting": "unweighted",
            **main, "sensitivity_australia_only": runs["australia_only"],
            "meets_nominal_effect_and_p": bool(main["observed"] >= H2_DIFFERENCE and main["lower_95"] > 0 and main["p_value"] < 0.05),
        }
    return {**summary, "endpoints": results}


def _set_contrasts(authors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for match_set, group in authors.groupby("match_set", sort=True):
        broker = group[group["role"].eq("broker")]
        comparison = group[group["role"].eq("comparison")]
        if broker.empty or comparison.empty:
            continue
        rows.append({"match_set": match_set, "broker": float(broker["entropy"].iloc[0]),
                     "comparison_mean": float(comparison["entropy"].mean()), "comparisons": int(len(comparison))})
    return pd.DataFrame(rows, columns=["match_set", "broker", "comparison_mean", "comparisons"])


def run_h3(docs: pd.DataFrame, permutations: int, min_documents: int = 2) -> dict[str, Any]:
    sample = docs[docs["task"].eq("H3")]
    usable = sample[sample["usable"]]
    authors = usable.groupby(["match_set", "role", "author_key"], sort=True)[FRAME_LABELS].sum()
    authors["documents"] = usable.groupby(["match_set", "role", "author_key"], sort=True).size()
    authors = authors.reset_index()
    authors = authors[authors["documents"].ge(min_documents)].copy()
    with np.errstate(divide="ignore", invalid="ignore"):  # the pipeline's entropy logs zero shares before masking them
        authors["entropy"] = [normalized_frame_entropy(row) for row in authors[FRAME_LABELS].to_numpy(dtype=float)]
    usable_contrasts = _set_contrasts(authors)
    contrasts = usable_contrasts[usable_contrasts["comparisons"].eq(3)].copy()
    summary = {
        "min_usable_documents": min_documents,
        "sampled_sets": int(sample["match_set"].nunique()),
        "usable_authors_by_role": authors["role"].value_counts().to_dict(),
        "usable_sets": int(len(usable_contrasts)),
        "complete_sets": int(len(contrasts)),
    }
    if len(contrasts) < 5:
        return {**summary, "status": "not_estimable", "reason": "fewer than five complete 1:3 matched sets with usable authors"}
    # Within each set, the permutation null swaps which usable member is called the broker.
    members = {match_set: group["entropy"].to_numpy(dtype=float) for match_set, group in authors.groupby("match_set") if match_set in set(contrasts["match_set"])}
    differences = (contrasts["broker"] - contrasts["comparison_mean"]).to_numpy()
    observed = float(differences.mean())
    rng = np.random.default_rng(SEED)
    null = np.empty(permutations)
    boot = np.empty(permutations)
    for index in range(permutations):
        draws = []
        for values in members.values():
            chosen = int(rng.integers(len(values)))
            draws.append(values[chosen] - np.delete(values, chosen).mean())
        null[index] = float(np.mean(draws))
        boot[index] = float(differences[rng.integers(len(differences), size=len(differences))].mean())
    result = {
        "observed": observed, "lower_95": float(np.quantile(boot, 0.025)), "upper_95": float(np.quantile(boot, 0.975)),
        "p_value": float((1 + np.count_nonzero(null >= observed)) / (permutations + 1)), "permutations": permutations,
        "mean_entropy": {"broker": round(float(contrasts["broker"].mean()), 4), "comparison": round(float(contrasts["comparison_mean"].mean()), 4)},
    }
    return {**summary, "status": "estimated", **result,
            "meets_nominal_effect_and_p": bool(observed >= H3_DIFFERENCE and result["lower_95"] > 0 and result["p_value"] < 0.05)}


def _input_checksums(labels_dir: Path, h4_result_path: Path | None) -> dict[str, str | None]:
    paths: dict[str, Path | None] = {
        "register": REGISTER,
        "sample_manifest": SAMPLE_MANIFEST,
        "roles": ROLES,
        "bluesky_labels": labels_dir / "bluesky_h1b.csv",
        "reddit_labels": labels_dir / "reddit_docs.csv",
        "reddit_threads": labels_dir / "reddit_threads.csv",
        "h4_result": h4_result_path,
    }
    return {name: sha256_file(path) if path is not None and path.exists() else None for name, path in paths.items()}


def validate_h1h3_result(path: Path = H1H3_RESULT) -> dict[str, Any]:
    """Reject H1-H4 outputs whose code, configuration, inputs, or result file changed."""

    result_path = Path(path).resolve()
    try:
        payload = read_json(result_path)
        config = payload["analysis_config"]
        labels_value = payload["labels_dir"]
        h4_value = payload.get("h4_result_path")
        input_checksums = payload["input_checksums"]
        code_checksums = payload["analysis_code_checksums"]
        manifest_value = payload["output_manifest"]
    except (KeyError, TypeError, json.JSONDecodeError, OSError) as error:
        raise ValueError(f"H1-H4 result is stale or incomplete: {result_path}") from error
    if (payload.get("schema_version") != H1H3_SCHEMA_VERSION or payload.get("status") != "exploratory_small_sample"
            or not isinstance(config, dict) or not isinstance(labels_value, str)
            or not isinstance(input_checksums, dict) or not isinstance(code_checksums, dict)
            or not isinstance(manifest_value, str) or not isinstance(payload.get("analysis_code_sha256"), str)
            or not isinstance(payload.get("analysis_config_sha256"), str)
            or not isinstance(payload.get("H2"), dict)):
        raise ValueError(f"H1-H4 result is stale or incomplete: {result_path}")
    if payload["H2"].get("estimand") != H2_ESTIMAND or payload["H2"].get("weighting") != "unweighted":
        raise ValueError(f"H1-H4 result is stale or incomplete: {result_path}")
    for endpoint in payload["H2"].get("endpoints", {}).values():
        if endpoint.get("status") == "estimated" and (endpoint.get("estimand") != H2_ESTIMAND or endpoint.get("weighting") != "unweighted"):
            raise ValueError(f"H1-H4 result is stale or incomplete: {result_path}")
    labels_dir = _resolve_stored_path(labels_value)
    h4_path = None if h4_value in (None, "") else _resolve_stored_path(str(h4_value))
    expected_config = _analysis_config(labels_dir, int(config.get("permutations", -1)), h4_path)
    if config != expected_config or payload["analysis_config_sha256"] != _analysis_config_sha256(config):
        raise ValueError(f"H1-H4 result is stale or incomplete: {result_path}")
    expected_code_checksums = _analysis_code_checksums()
    if code_checksums != expected_code_checksums or payload["analysis_code_sha256"] != _analysis_code_sha256(code_checksums):
        raise ValueError(f"H1-H4 result is stale or incomplete: {result_path}")
    expected_inputs = _input_checksums(labels_dir, h4_path)
    if any(value is None for name, value in expected_inputs.items() if name != "h4_result") or input_checksums != expected_inputs:
        raise ValueError(f"H1-H4 result is stale or incomplete: {result_path}")
    if h4_path is not None:
        validate_h4_result(h4_path)
    manifest_path = _resolve_stored_path(manifest_value)
    try:
        manifest = read_json(manifest_path)
        record = manifest["result"]
        if (manifest.get("schema_version") != "h1h3-human-sample-manifest.v1"
                or record.get("path") != _stored_path(result_path)
                or record.get("bytes") != result_path.stat().st_size
                or record.get("sha256") != sha256_file(result_path)):
            raise ValueError
    except (KeyError, TypeError, OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"H1-H4 result is stale or incomplete: {result_path}") from error
    if not all(key in payload for key in ("H1", "H2", "H3", "H3_sensitivity_min_one_document", "multiple_testing")):
        raise ValueError(f"H1-H4 result is stale or incomplete: {result_path}")
    _assert_finite(payload, "result")
    return payload


def build(labels_dir: Path = SETTLED_ROOT, output_dir: Path = OUTPUT_ROOT, permutations: int = PERMUTATIONS,
          h4_result_path: Path | None = H4_RESULT) -> dict[str, Any]:
    h4_payload = None
    if h4_result_path is not None:
        h4_path = Path(h4_result_path)
        if not h4_path.exists():
            raise FileNotFoundError(f"H4 result is missing: {h4_path}")
        h4_payload = validate_h4_result(h4_path)
    settled = load_settled(labels_dir)
    docs = _documents(settled)
    h1 = {"H1-B": run_h1(docs, "B_ACTOR_ATTENTION", "H1-B", permutations), "H1-R": run_h1(docs, "R_ACTOR_ATTENTION", "H1-R", permutations)}
    h2 = run_h2(docs, permutations)
    h3 = run_h3(docs, permutations)
    h3_sensitivity = run_h3(docs, permutations, min_documents=1)
    p_values = {name: result["p_value"] for name, result in h1.items() if result["status"] == "estimated"}
    p_values.update({name: result["p_value"] for name, result in h2["endpoints"].items() if result["status"] == "estimated"})
    if h3["status"] == "estimated":
        p_values["H3"] = h3["p_value"]
    if h4_payload is not None:
        p_values["H4"] = float(h4_payload["result"]["p_value"])
    adjusted = holm_adjust(p_values) if p_values else {}
    holm_decisions = {}
    for endpoint, result in [("H1-B", h1["H1-B"]), ("H1-R", h1["H1-R"]),
                             ("H2-P", h2["endpoints"]["H2-P"]), ("H2-C", h2["endpoints"]["H2-C"]),
                             ("H3", h3)]:
        if result.get("status") != "estimated":
            continue
        result["holm_adjusted_p_value"] = adjusted[endpoint]
        result["meets_planned_effect_and_p"] = bool(result["meets_nominal_effect_and_p"] and adjusted[endpoint] < 0.05)
        holm_decisions[endpoint] = result["meets_planned_effect_and_p"]
    if h4_payload is not None:
        holm_decisions["H4"] = bool(h4_payload["meets_nominal_rule"] and adjusted["H4"] < 0.05)
    summary = {
        "schema_version": H1H3_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "labels_dir": str(labels_dir.resolve().relative_to(REPO_ROOT)) if labels_dir.resolve().is_relative_to(REPO_ROOT) else str(labels_dir),
        "h4_result_path": str(Path(h4_result_path).resolve().relative_to(REPO_ROOT)) if h4_result_path is not None and Path(h4_result_path).resolve().is_relative_to(REPO_ROOT) else str(h4_result_path) if h4_result_path is not None else None,
        "input_checksums": _input_checksums(labels_dir, h4_result_path),
        "analysis_config": _analysis_config(labels_dir, permutations, h4_result_path),
        "analysis_config_sha256": _analysis_config_sha256(_analysis_config(labels_dir, permutations, h4_result_path)),
        "analysis_code_checksums": _analysis_code_checksums(),
        "analysis_code_sha256": _analysis_code_sha256(),
        "output_manifest": _stored_path(output_dir / "h1h3_manifest.json"),
        "status": "exploratory_small_sample",
        "thread_check": {"threads": int(len(settled["reddit_threads"])), "accepted": int(settled["reddit_threads"]["accepted"].sum())},
        "usable_share_by_task": docs.groupby("task")["usable"].mean().round(3).to_dict(),
        "H1": h1,
        "H2": h2,
        "H3": h3,
        "H3_sensitivity_min_one_document": h3_sensitivity,
        "holm_adjusted_p": adjusted,
        "holm_meets_planned_rule": holm_decisions,
        "multiple_testing": {"family": "PRIMARY-6", "adjustment": "holm", "alpha": 0.05, "estimable_endpoints": sorted(p_values)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "h1h3_result.json"
    write_json(result_path, summary)
    write_json(output_dir / "h1h3_manifest.json", {
        "schema_version": "h1h3-human-sample-manifest.v1",
        "result": _output_record(result_path),
    })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--labels-dir", type=Path, default=SETTLED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--h4-result", type=Path, default=H4_RESULT, help="validated H4 artifact to include in the PRIMARY-6 adjustment")
    parser.add_argument("--result", type=Path, default=H1H3_RESULT, help="H1-H4 result to validate with --check")
    args = parser.parse_args()
    if args.check:
        validate_h1h3_result(args.result)
        print(json.dumps({"status": "valid", "path": str(args.result)}))
        return 0
    print(json.dumps(build(args.labels_dir, args.output_dir, args.permutations, args.h4_result), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
