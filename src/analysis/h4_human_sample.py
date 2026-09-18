"""H4 on human labels: do comment frames track video framing differently for news and commentary?

The planned H4 (src/analysis/hypotheses.py) reads audience frames from validated automated
labels, and no frame model passed validation. This module runs the same estimand on human
labels instead, which is a declared change from the planned design:

- video side: adjudicated video metadata labels in data/analysis/annotation/youtube_video_metadata.csv;
- audience side: adjudicated frame labels of coded YouTube comments (all three validation splits
  by default, plus any top-up file), author-balanced within each video;
- statistic: the pipeline's own video_alignment_contrast (event-adjusted news minus commentary
  difference in 1 - Jensen-Shannon alignment, whole-channel permutations and bootstrap).

Usage (from the SMFR root):
    python -m src.analysis.h4_human_sample --build
    python -m src.analysis.h4_human_sample --build --metadata <csv> --min-comments 2
    python -m src.analysis.h4_human_sample --build --sensitivity a=<coder A csv> b=<coder B csv>

Outputs go to data/analysis/h4_human/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.artifacts import REPO_ROOT, read_json, relative_path, sha256_file, write_json
from src.analysis.hypotheses import PERMUTATIONS, video_alignment_contrast
from src.analysis.networks import YOUTUBE_METADATA_AUDIT, canonical_youtube_video_id

FRAME_LABELS = [
    "policy_assurance",
    "child_safety",
    "privacy_surveillance",
    "governance_platform_responsibility",
    "circumvention_censorship_autonomy",
]
VALID_TARGETS = {"UK_OSA", "AU_SOCIAL_MINIMUM_AGE", "OTHER_EXTENDED_EVENT"}
VALIDATION_ROOT = REPO_ROOT / "data" / "analysis" / "validation"
TOPUP_LABELS = REPO_ROOT / "data" / "analysis" / "annotation" / "youtube_comment_topup_labels.csv"
SEED_VIDEOS = REPO_ROOT / "config" / "youtube" / "seed_videos.csv"
OUTPUT_ROOT = REPO_ROOT / "data" / "analysis" / "h4_human"
SPLITS = ("development", "evaluation", "reserve")
SEED = 20260917
H4_SCHEMA_VERSION = "h4-human-sample.v3"
H4_RESULT = OUTPUT_ROOT / "h4_result.json"
H4_TABLE = OUTPUT_ROOT / "h4_video_table.csv"
ANALYSIS_CODE_PATHS = {
    "h4_human_sample": Path(__file__),
    "hypotheses": Path(__file__).with_name("hypotheses.py"),
    "networks": Path(__file__).with_name("networks.py"),
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


def _analysis_config(metadata_path: Path, min_comments: int, permutations: int,
                     splits: tuple[str, ...], topup: Path | None) -> dict[str, Any]:
    return {
        "metadata_path": _stored_path(metadata_path),
        "topup_labels_path": _stored_path(topup),
        "label_splits": list(splits),
        "min_comments_per_video": int(min_comments),
        "permutations": int(permutations),
        "seed": SEED,
        "frame_labels": FRAME_LABELS,
        "valid_targets": sorted(VALID_TARGETS),
    }


def _analysis_config_sha256(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _output_record(path: Path, rows: int, columns: list[str]) -> dict[str, Any]:
    return {
        "path": _stored_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": int(rows),
        "columns": columns,
    }


def _input_checksums(metadata_path: Path, splits: tuple[str, ...], topup: Path | None) -> dict[str, str | None]:
    paths: dict[str, Path | None] = {
        "metadata": Path(metadata_path),
        "seed_videos": SEED_VIDEOS,
        "topup_labels": topup,
    }
    paths.update({f"labels_{split}": VALIDATION_ROOT / f"coordinator_labels_{split}.csv" for split in splits})
    return {name: sha256_file(path) if path is not None and path.exists() else None for name, path in paths.items()}


def validate_h4_result(path: Path) -> dict[str, Any]:
    """Reject H4 outputs whose declared design or source files no longer match."""

    payload = read_json(path)
    metadata_value = payload.get("metadata_path")
    splits = payload.get("label_splits")
    topup_value = payload.get("topup_labels_path")
    result = payload.get("result")
    config = payload.get("analysis_config")
    output_checksums = payload.get("output_checksums")
    if (payload.get("schema_version") != H4_SCHEMA_VERSION or payload.get("status") != "estimated"
            or not isinstance(metadata_value, str) or not isinstance(splits, list)
            or not all(isinstance(split, str) for split in splits) or not isinstance(result, dict)
            or "topup_labels_path" not in payload or not isinstance(payload.get("input_checksums"), dict)
            or not isinstance(payload.get("meets_nominal_rule"), bool) or not isinstance(config, dict)
            or not isinstance(payload.get("analysis_config_sha256"), str)
            or not isinstance(payload.get("analysis_code_checksums"), dict)
            or not isinstance(payload.get("analysis_code_sha256"), str)
            or not isinstance(output_checksums, dict)):
        raise ValueError(f"H4 result is stale or incomplete: {path}")
    if any(not isinstance(result.get(key), (int, float)) or isinstance(result.get(key), bool)
           or not math.isfinite(float(result[key])) for key in ("p_value", "observed", "lower_95", "upper_95")):
        raise ValueError(f"H4 result is stale or incomplete: {path}")
    metadata_path = _resolve_stored_path(metadata_value)
    topup = None if topup_value in (None, "") else _resolve_stored_path(str(topup_value))
    expected_config = _analysis_config(metadata_path, int(config.get("min_comments_per_video", -1)),
                                       int(config.get("permutations", -1)), tuple(splits), topup)
    expected_code_checksums = _analysis_code_checksums()
    required = [metadata_path, SEED_VIDEOS, *(VALIDATION_ROOT / f"coordinator_labels_{split}.csv" for split in splits)]
    if topup is not None:
        required.append(topup)
    if any(not required_path.exists() for required_path in required):
        raise ValueError(f"H4 result is stale or incomplete: {path}")
    if (config != expected_config or payload["analysis_config_sha256"] != _analysis_config_sha256(config)
            or payload["analysis_code_checksums"] != expected_code_checksums
            or payload["analysis_code_sha256"] != _analysis_code_sha256(expected_code_checksums)
            or payload["input_checksums"] != _input_checksums(metadata_path, tuple(splits), topup)):
        raise ValueError(f"H4 result is stale or incomplete: {path}")
    table_record = output_checksums.get("video_table")
    if not isinstance(table_record, dict) or not isinstance(table_record.get("path"), str):
        raise ValueError(f"H4 result is stale or incomplete: {path}")
    table_path = _resolve_stored_path(table_record["path"])
    if (not table_path.exists() or table_record.get("bytes") != table_path.stat().st_size
            or table_record.get("sha256") != sha256_file(table_path)):
        raise ValueError(f"H4 result is stale or incomplete: {path}")
    table = pd.read_csv(table_path, keep_default_na=False)
    if table_record.get("rows") != len(table) or table_record.get("columns") != list(table.columns):
        raise ValueError(f"H4 result is stale or incomplete: {path}")
    return payload


def _frames(value: object) -> set[str]:
    labels = {part.strip() for part in str(value or "").split("|") if part.strip()}
    unknown = labels - set(FRAME_LABELS) - {"none", "none_unclear"}
    if unknown:
        raise ValueError(f"unknown frame labels: {sorted(unknown)}")
    return labels & set(FRAME_LABELS)


def load_comment_labels(splits: tuple[str, ...] = SPLITS, topup: Path | None = TOPUP_LABELS) -> pd.DataFrame:
    """Eligible adjudicated YouTube comments, one row per comment, with 0/1 frame columns."""

    columns = ["platform", "split", "doc_id", "author_key", "cluster_id", "case_window_id",
               "adjudicated_relevance", "adjudicated_language", "adjudicated_target_policy",
               "adjudicated_frame_labels"]
    parts = [pd.read_csv(VALIDATION_ROOT / f"coordinator_labels_{split}.csv", usecols=columns, keep_default_na=False)
             for split in splits]
    if topup is not None and topup.exists():
        extra = pd.read_csv(topup, keep_default_na=False)
        missing = sorted(set(columns) - set(extra.columns))
        if missing:
            raise ValueError(f"top-up labels are missing columns: {missing}")
        parts.append(extra[columns])
    labels = pd.concat(parts, ignore_index=True)
    labels = labels[labels["platform"].eq("youtube")]
    if labels["doc_id"].duplicated().any():
        raise ValueError("a YouTube comment is labelled more than once across the label files")
    eligible = labels[
        labels["adjudicated_relevance"].eq("relevant")
        & labels["adjudicated_language"].eq("english")
        & labels["adjudicated_target_policy"].isin(VALID_TARGETS)
        & labels["author_key"].astype(str).str.strip().ne("")
    ].copy()
    eligible["video_id"] = eligible["cluster_id"].astype(str).str.removeprefix("yt:video:").map(canonical_youtube_video_id)
    for label in FRAME_LABELS:
        eligible[label] = eligible["adjudicated_frame_labels"].map(lambda value, label=label: int(label in _frames(value)))
    return eligible


def audience_profiles(comments: pd.DataFrame) -> pd.DataFrame:
    """Author-balanced frame shares per video: average within author, then across authors."""

    per_author = comments.groupby(["video_id", "author_key"], sort=True)[FRAME_LABELS].mean().reset_index()
    profiles = per_author.groupby("video_id", sort=True)[FRAME_LABELS].mean()
    profiles["comments"] = comments.groupby("video_id").size()
    profiles["authors"] = per_author.groupby("video_id").size()
    return profiles.reset_index()


def load_metadata(path: Path) -> pd.DataFrame:
    audit = pd.read_csv(path, keep_default_na=False)
    required = {"video_id", "adjudicated_source_type", "adjudicated_frame"}
    missing = sorted(required - set(audit.columns))
    if missing:
        raise ValueError(f"video metadata file is missing columns: {missing}")
    audit = audit.assign(video_id=audit["video_id"].map(canonical_youtube_video_id))
    table = pd.DataFrame({"video_id": audit["video_id"],
                          "source_type": audit["adjudicated_source_type"].str.strip().str.casefold()})
    for label in FRAME_LABELS:
        table[f"metadata_{label}"] = audit["adjudicated_frame"].map(lambda value, label=label: int(label in _frames(value)))
    return table


def build_table(metadata_path: Path, min_comments: int, comments: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    seed = pd.read_csv(SEED_VIDEOS, usecols=["video_id", "event_id", "channel_id"])
    seed["video_id"] = seed["video_id"].map(canonical_youtube_video_id)
    table = (load_metadata(metadata_path)
             .merge(seed, on="video_id", how="inner", validate="one_to_one")
             .merge(audience_profiles(comments), on="video_id", how="left"))
    table = table.rename(columns={"event_id": "event", **{label: f"audience_{label}" for label in FRAME_LABELS}})
    table[["comments", "authors"]] = table[["comments", "authors"]].fillna(0).astype(int)
    notes: dict[str, Any] = {"videos_coded": int(len(table))}
    table = table[table["event"].isin(["E2", "E3"]) & table["source_type"].isin(["news", "commentary"])]
    notes["news_or_commentary_videos"] = int(len(table))
    metadata_columns = [f"metadata_{label}" for label in FRAME_LABELS]
    audience_columns = [f"audience_{label}" for label in FRAME_LABELS]
    usable = (table[metadata_columns].sum(axis=1) > 0) & (table["comments"] >= min_comments) & (table[audience_columns].fillna(0).sum(axis=1) > 0)
    notes["dropped_no_video_frame"] = int((table[metadata_columns].sum(axis=1) == 0).sum())
    notes["dropped_too_few_comments"] = int((table["comments"] < min_comments).sum())
    notes["dropped_comments_without_frames"] = int(((table["comments"] >= min_comments) & (table[audience_columns].fillna(0).sum(axis=1) == 0)).sum())
    table = table[usable].copy()
    # The permutation test needs one source type per channel. Exclude mixed channels rather
    # than changing an adjudicated video label to make the estimator fit.
    mixed = table.groupby("channel_id")["source_type"].nunique()
    mixed_channels = set(mixed[mixed.gt(1)].index)
    mixed_rows = table[table["channel_id"].isin(mixed_channels)]
    notes["dropped_mixed_source_channels"] = sorted(str(channel) for channel in mixed_channels)
    notes["dropped_mixed_source_videos"] = mixed_rows["video_id"].tolist()
    table = table[~table["channel_id"].isin(mixed_channels)].copy()
    notes["videos_analysed"] = int(len(table))
    notes["cells"] = {f"{event}/{source}": int(n) for (event, source), n in table.groupby(["event", "source_type"]).size().items()}
    return table.reset_index(drop=True), notes


def leave_one_channel_out(table: pd.DataFrame, metadata_columns: list[str], audience_columns: list[str]) -> dict[str, Any]:
    from src.analysis.hypotheses import _event_adjusted_difference, _normalised_rows, js_divergence

    values = table.copy()
    metadata = _normalised_rows(values[metadata_columns].to_numpy(dtype=float))
    audience = _normalised_rows(values[audience_columns].to_numpy(dtype=float))
    values["alignment"] = [1.0 - js_divergence(left, right) for left, right in zip(metadata, audience)]
    estimates = {}
    for channel in sorted(values["channel_id"].unique()):
        try:
            estimates[channel] = _event_adjusted_difference(values[values["channel_id"].ne(channel)], "source_type", "alignment", "event")
        except ValueError:
            continue
    series = pd.Series(estimates, dtype=float)
    return {"channels_dropped": int(len(series)), "min": float(series.min()), "max": float(series.max()),
            "sign_changes": int((series.apply(lambda value: value > 0) != (series.median() > 0)).sum())}


def build(metadata_path: Path, min_comments: int, permutations: int, splits: tuple[str, ...], topup: Path | None = TOPUP_LABELS) -> dict[str, Any]:
    comments = load_comment_labels(splits, topup)
    table, notes = build_table(metadata_path, min_comments, comments)
    metadata_columns = [f"metadata_{label}" for label in FRAME_LABELS]
    audience_columns = [f"audience_{label}" for label in FRAME_LABELS]
    result = video_alignment_contrast(table, metadata_columns, audience_columns, permutations=permutations, seed=SEED)
    table_with_alignment = table.copy()
    from src.analysis.hypotheses import _normalised_rows, js_divergence
    table_with_alignment["alignment"] = [
        1.0 - js_divergence(left, right)
        for left, right in zip(_normalised_rows(table[metadata_columns].to_numpy(dtype=float)),
                               _normalised_rows(table[audience_columns].to_numpy(dtype=float)))
    ]
    means = table_with_alignment.groupby(["event", "source_type"])["alignment"].mean().round(4)
    summary = {
        "schema_version": H4_SCHEMA_VERSION,
        "status": "estimated",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "design_change": "Audience frames come from adjudicated human comment labels, not validated automated labels; video frames from the adjudicated metadata audit.",
        "metadata_path": _stored_path(Path(metadata_path)),
        "topup_labels_path": _stored_path(topup),
        "label_splits": list(splits),
        "input_checksums": _input_checksums(Path(metadata_path), splits, topup),
        "analysis_config": _analysis_config(Path(metadata_path), min_comments, permutations, splits, topup),
        "analysis_config_sha256": _analysis_config_sha256(_analysis_config(Path(metadata_path), min_comments, permutations, splits, topup)),
        "analysis_code_checksums": _analysis_code_checksums(),
        "analysis_code_sha256": _analysis_code_sha256(),
        "eligible_comments": int(len(comments)),
        "min_comments_per_video": min_comments,
        "selection": notes,
        "mean_alignment_by_cell": {f"{event}/{source}": value for (event, source), value in means.items()},
        "result": result,
        "leave_one_channel_out": leave_one_channel_out(table, metadata_columns, audience_columns),
        "threshold": "nominal rule: two-sided p < 0.05, interval excludes 0 and |difference| >= 0.05; Holm adjustment is applied in the aggregate H1-H4 report",
        "meets_nominal_rule": bool(result["p_value"] < 0.05 and (result["lower_95"] > 0 or result["upper_95"] < 0) and abs(result["observed"]) >= 0.05),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    table_with_alignment.to_csv(H4_TABLE, index=False, lineterminator="\n")
    summary["output_checksums"] = {"video_table": _output_record(H4_TABLE, len(table_with_alignment), list(table_with_alignment.columns))}
    write_json(H4_RESULT, summary)
    return summary


def _single_coder_metadata(coder_csv: Path, target: Path) -> Path:
    """Write one coder's video labels in the adjudicated-audit shape, for a sensitivity run."""

    codes = pd.read_csv(coder_csv, keep_default_na=False)
    frame = pd.DataFrame({"video_id": codes["video_id"], "adjudicated_source_type": codes["source_type"],
                          "adjudicated_frame": codes["frame"]})
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target


def sensitivity(permutations: int, splits: tuple[str, ...], coder_files: dict[str, Path]) -> dict[str, Any]:
    """Rerun the contrast under stricter comment minimums and with each coder's own video labels."""

    import numpy as np
    from src.analysis.hypotheses import _normalised_rows, js_divergence

    comments = load_comment_labels(splits)
    metadata_columns = [f"metadata_{label}" for label in FRAME_LABELS]
    audience_columns = [f"audience_{label}" for label in FRAME_LABELS]
    variants = {f"min_comments_{n}": (YOUTUBE_METADATA_AUDIT, n) for n in (3, 5)}
    for name, path in coder_files.items():
        variants[f"video_labels_{name}_only"] = (_single_coder_metadata(path, OUTPUT_ROOT / "sensitivity_inputs" / f"{name}.csv"), 1)
    runs = {}
    for name, (path, min_comments) in variants.items():
        table, notes = build_table(path, min_comments, comments)
        result = video_alignment_contrast(table, metadata_columns, audience_columns, permutations=permutations, seed=SEED)
        runs[name] = {"videos": notes["videos_analysed"], "cells": notes["cells"],
                      **{key: result[key] for key in ("observed", "lower_95", "upper_95", "p_value")}}
    # Comment counts differ by cell, and alignment from few comments is noisy. Check whether the
    # news-commentary gap survives holding the number of labelled comments fixed.
    table, _ = build_table(YOUTUBE_METADATA_AUDIT, 1, comments)
    table["alignment"] = [1.0 - js_divergence(left, right) for left, right in zip(
        _normalised_rows(table[metadata_columns].to_numpy(dtype=float)),
        _normalised_rows(table[audience_columns].to_numpy(dtype=float)))]
    design = np.column_stack([np.ones(len(table)), table["source_type"].eq("news").astype(float),
                              table["event"].eq("E3").astype(float), np.log(table["comments"].astype(float))])
    coefficients, *_ = np.linalg.lstsq(design, table["alignment"].to_numpy(dtype=float), rcond=None)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs": runs,
        "comments_per_video_by_source": table.groupby("source_type")["comments"].describe()[["count", "mean", "50%"]].round(2).to_dict(orient="index"),
        "ols_news_coefficient_controlling_event_and_log_comments": round(float(coefficients[1]), 4),
        "ols_log_comments_coefficient": round(float(coefficients[3]), 4),
    }
    (OUTPUT_ROOT / "h4_sensitivity.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--sensitivity", nargs="*", metavar="NAME=CODER_CSV",
                        help="also rerun with stricter comment minimums and each named coder's own video labels")
    parser.add_argument("--metadata", type=Path, default=YOUTUBE_METADATA_AUDIT)
    parser.add_argument("--topup-labels", type=Path, default=TOPUP_LABELS)
    parser.add_argument("--min-comments", type=int, default=1)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--result", type=Path, default=H4_RESULT, help="H4 result to validate with --check")
    args = parser.parse_args()
    if args.check:
        validate_h4_result(args.result)
        print(json.dumps({"status": "valid", "path": str(args.result)}))
        return 0
    splits = tuple(args.splits.split(","))
    summary = build(args.metadata, args.min_comments, args.permutations, splits, args.topup_labels)
    print(json.dumps({key: summary[key] for key in ("selection", "mean_alignment_by_cell", "result", "leave_one_channel_out", "meets_nominal_rule")}, indent=2))
    if args.sensitivity is not None:
        coder_files = dict(item.split("=", 1) for item in args.sensitivity)
        print(json.dumps(sensitivity(args.permutations, splits, {name: Path(path) for name, path in coder_files.items()}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
