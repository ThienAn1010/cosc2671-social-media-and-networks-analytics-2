"""Deterministic, model-blind human-validation sampling and packet generation."""

from __future__ import annotations

import hashlib
import re
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.artifacts import ANALYSIS_ROOT, REPO_ROOT, read_json, relative_path, sha256_file, utc_now_iso, write_json
from src.analysis.codebook import ANNOTATION_SCHEMA, CODEBOOK_VERSION, check_codebook
from src.shared.case_windows import REDDIT_CASE_BY_EVENT
from src.shared.ids import stable_analysis_doc_id
from src.shared.masking import mask_text

BASE_VALIDATION_ROOT = ANALYSIS_ROOT / "validation"
BASE_MODEL_ROOT = ANALYSIS_ROOT / "models"
MEASUREMENT_CYCLE_STATE = ANALYSIS_ROOT / "measurement_cycle.json"
SAMPLING_SEED = 20260915
SAMPLING_VERSION = "validation-sampling-v1.4-unified-reddit-context"


def _cycle_state() -> dict[str, Any]:
    if not MEASUREMENT_CYCLE_STATE.exists():
        return {}
    try:
        value = read_json(MEASUREMENT_CYCLE_STATE)
    except (OSError, ValueError):
        return {}
    return value if value.get("status") == "active" else {}


def _cycle_path(value: object, fallback: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        return fallback
    path = Path(value)
    path = path if path.is_absolute() else REPO_ROOT / path
    try:
        path.resolve().relative_to(ANALYSIS_ROOT.resolve())
    except ValueError:
        return fallback
    return path


def active_validation_root() -> Path:
    return _cycle_path(_cycle_state().get("validation_root"), BASE_VALIDATION_ROOT)


def active_model_root() -> Path:
    return _cycle_path(_cycle_state().get("model_root"), BASE_MODEL_ROOT)


VALIDATION_ROOT = active_validation_root()
VALIDATION_MANIFEST = VALIDATION_ROOT / "validation_manifest.json"
SPLIT_TARGETS = {"development": 200, "evaluation": 200, "reserve": 100}
STRATUM_QUOTAS = {"development": {"L_LANGUAGE": 30, "S_SHORT_CONTEXT": 30, "R_RARE_DISAGREE": 60, "G_GENERAL": 80}, "evaluation": {"L_LANGUAGE": 30, "S_SHORT_CONTEXT": 30, "R_RARE_DISAGREE": 60, "G_GENERAL": 80}, "reserve": {"L_LANGUAGE": 15, "S_SHORT_CONTEXT": 15, "R_RARE_DISAGREE": 30, "G_GENERAL": 40}}
SPLIT_ORDER = ("development", "evaluation", "reserve")

LABEL_COLUMNS = [
    f"{coder}_{field}"
    for coder in ("coder_a", "coder_b")
    for field in ("relevance", "language", "target_policy", "stance", "sentiment", "frame_labels", "bypass_techniques", "notes")
] + [f"adjudicated_{field}" for field in ("relevance", "language", "target_policy", "stance", "sentiment", "frame_labels", "bypass_techniques", "notes")]
CODER_LABEL_FIELDS = ("relevance", "language", "target_policy", "stance", "sentiment", "frame_labels", "bypass_techniques", "notes")
BLIND_PACKET_COLUMNS = ("annotation_id", "text_for_annotation", "parent_context", "root_context")
CODER_PACKET_COLUMNS = BLIND_PACKET_COLUMNS + ("label_version", "annotation_status", *CODER_LABEL_FIELDS)
REGISTER_COLUMNS = (
    "annotation_id", "platform", "split", "stratum", "doc_id", "author_key", "cluster_id", "case_window_id",
    "text_digest", "text_owner_split", "random_key", "p_author_cap", "p_context_cap", "p_quota",
    "inclusion_probability", "duplicate_key", "sampling_seed", "selection_key", "label_version",
)
_CANDIDATE_FRAME_PATTERNS = {
    "policy_assurance": re.compile(r"\b(age verification|age assurance|age check|online safety|minimum age|age limit)\b", re.I),
    "child_safety": re.compile(r"\b(child|children|kids|minor|groom|csam|harmful content|protect)\w*\b", re.I),
    "privacy_surveillance": re.compile(r"\b(privacy|surveillance|biometric|facial recognition|personal data|id upload|passport|tracking|retention)\b", re.I),
    "governance_platform_responsibility": re.compile(r"\b(government|regulator|platform|enforcement|accountability|transparen|appeal|audit)\w*\b", re.I),
    "circumvention_censorship_autonomy": re.compile(r"\b(vpn|proxy|tor|bypass|circumvent|censor|free speech|autonomy|overreach)\w*\b", re.I),
}


def _key(value: str, seed: int = SAMPLING_SEED) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}|{value}".encode("utf-8"), digest_size=8).digest(), "big")


def _candidate_frame_signature(text: str) -> str:
    return "|".join(name for name, pattern in _CANDIDATE_FRAME_PATTERNS.items() if pattern.search(str(text)))


def _duplicate_key(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", str(text).casefold())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def assign_context_splits(candidates: pd.DataFrame, seed: int = SAMPLING_SEED) -> pd.DataFrame:
    """Assign complete context containers to one split using stable hash buckets."""

    frame = candidates.copy()
    clusters = sorted(frame["cluster_id"].astype(str).unique(), key=lambda value: (_key(f"cluster|{value}", seed), value))
    split_by_cluster = {
        cluster: SPLIT_ORDER[_key(f"split|{cluster}", seed) % len(SPLIT_ORDER)] for cluster in clusters
    }
    frame["split"] = frame["cluster_id"].astype(str).map(split_by_cluster)
    return frame


def _base_frame(platform: str) -> pd.DataFrame:
    if platform == "reddit":
        path = REPO_ROOT / "data" / "processed" / "reddit_age_gate" / "analysis_corpus.parquet"
        columns = ["record_id", "thing", "event_id", "author_hash", "thread_id", "parent_record_id", "schema_valid", "human_only", "is_english", "is_language_uncertain", "text_topic", "text_sentiment", "is_url_only", "is_no_substantive_text", "sentiment_compound"]
        raw = pd.read_parquet(path, columns=columns)
        raw["_annotation_text"] = raw["text_sentiment"].fillna(raw["text_topic"]).fillna("").astype(str)
        text_lookup = raw.set_index("record_id")["_annotation_text"].to_dict()
        root_text_lookup = raw[raw["thing"].eq("post")].set_index("record_id")["_annotation_text"].to_dict()
        raw = raw[raw["schema_valid"].fillna(False).astype(bool) & raw["human_only"].fillna(False).astype(bool)]
        raw["doc_id"] = raw.apply(lambda row: stable_analysis_doc_id("reddit", str(row["thing"]), str(row["record_id"])), axis=1)
        raw["text_for_annotation"] = raw["_annotation_text"].map(mask_text)
        raw["parent_context"] = raw["parent_record_id"].map(text_lookup).fillna("").map(mask_text)
        raw["root_context"] = raw["thread_id"].map(root_text_lookup).fillna("").map(mask_text)
        raw["case_window_id"] = raw["event_id"].map(REDDIT_CASE_BY_EVENT).fillna("none")
        raw["cluster_id"] = "reddit:thread:" + raw["thread_id"].fillna("").astype(str)
        raw["language_uncertain"] = raw["is_language_uncertain"].fillna(False).astype(bool)
        compound = pd.to_numeric(raw["sentiment_compound"], errors="coerce")
        raw["candidate_disagreement"] = compound.abs().between(0.05, 0.20, inclusive="both")
        raw["eligible"] = raw["case_window_id"].ne("none") & raw["author_hash"].fillna("").ne("") & raw["text_for_annotation"].str.strip().ne("") & ~raw["is_url_only"].fillna(False).astype(bool) & ~raw["is_no_substantive_text"].fillna(False).astype(bool) & (raw["is_english"].fillna(False).astype(bool) | raw["language_uncertain"])
    elif platform == "youtube":
        path = REPO_ROOT / "data" / "processed" / "youtube" / "documents.parquet"
        columns = ["doc_id", "thing", "root_doc_id", "parent_doc_id", "author_hash", "yt_video_event_id", "exclusion_status", "is_english", "is_language_uncertain", "text_clean", "yt_video_type"]
        raw = pd.read_parquet(path, columns=columns)
        lookup = raw.set_index("doc_id")["text_clean"].to_dict()
        raw = raw[raw["thing"].ne("video")]
        raw["text_for_annotation"] = raw["text_clean"].map(mask_text)
        raw["parent_context"] = raw["parent_doc_id"].map(lookup).fillna("").map(mask_text)
        raw["root_context"] = raw["root_doc_id"].map(lookup).fillna("").map(mask_text)
        raw["case_window_id"] = raw["yt_video_event_id"].where(raw["yt_video_event_id"].isin(["E2", "E3"]), "none")
        raw["cluster_id"] = raw["root_doc_id"].astype(str)
        raw["language_uncertain"] = raw["is_language_uncertain"].fillna(False).astype(bool)
        raw["candidate_disagreement"] = False
        raw["eligible"] = raw["case_window_id"].ne("none") & raw["author_hash"].fillna("").ne("") & raw["text_for_annotation"].str.strip().ne("") & raw["exclusion_status"].isin(["eligible", "language_uncertain"])
    elif platform == "bluesky":
        path = REPO_ROOT / "data" / "processed" / "bluesky" / "posts.parquet"
        columns = ["doc_id", "author_hash", "parent_doc_id", "root_doc_id", "event_window", "is_english", "lang_detect_prob", "n_chars", "text_raw", "is_reply", "is_meme", "is_repeat_burst"]
        raw = pd.read_parquet(path, columns=columns)
        raw["text_for_annotation"] = raw["text_raw"].map(mask_text)
        lookup = raw.set_index("doc_id")["text_raw"].map(mask_text).to_dict()
        raw["parent_context"] = raw["parent_doc_id"].map(lookup).fillna("")
        raw["root_context"] = raw["root_doc_id"].map(lookup).fillna("")
        raw["case_window_id"] = raw["event_window"].where(raw["event_window"].isin(["E2", "E3"]), "none")
        raw["cluster_id"] = raw["root_doc_id"].fillna(raw["doc_id"]).replace("", pd.NA).fillna(raw["doc_id"]).astype(str)
        raw["language_uncertain"] = raw["lang_detect_prob"].fillna(0).lt(0.8) | raw["n_chars"].fillna(0).lt(24)
        raw["candidate_disagreement"] = False
        raw["eligible"] = raw["case_window_id"].ne("none") & raw["author_hash"].fillna("").ne("") & raw["text_for_annotation"].str.strip().ne("") & (raw["is_english"].fillna(False).astype(bool) | raw["language_uncertain"])
    else:
        raise ValueError(f"unsupported validation platform: {platform}")

    raw = raw[raw["eligible"]].copy()
    raw["candidate_frame_signature"] = raw["text_for_annotation"].map(_candidate_frame_signature)
    frame_counts = Counter(
        frame_name
        for signature in raw["candidate_frame_signature"]
        for frame_name in signature.split("|")
        if frame_name
    )
    rare_cutoff = max(5, int(len(raw) * 0.10))
    rare_frames = {frame_name for frame_name, count in frame_counts.items() if count <= rare_cutoff}
    raw["candidate_disagreement"] = raw["candidate_disagreement"] | raw["candidate_frame_signature"].map(lambda value: any(frame_name in rare_frames for frame_name in value.split("|") if frame_name))
    raw["author_key"] = raw["author_hash"].fillna("").astype(str)
    raw["text_digest"] = raw["text_for_annotation"].str.replace(r"\s+", " ", regex=True).str.strip().str.casefold().map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    raw["duplicate_key"] = raw["text_for_annotation"].map(_duplicate_key)
    raw["short_context"] = raw["text_for_annotation"].str.len().lt(80) | raw["parent_context"].str.strip().ne("")
    raw["stratum"] = "G_GENERAL"
    raw.loc[raw["language_uncertain"], "stratum"] = "L_LANGUAGE"
    raw.loc[~raw["language_uncertain"] & raw["short_context"], "stratum"] = "S_SHORT_CONTEXT"
    raw.loc[~raw["language_uncertain"] & ~raw["short_context"] & raw["candidate_disagreement"], "stratum"] = "R_RARE_DISAGREE"
    keep = ["doc_id", "author_key", "cluster_id", "case_window_id", "text_for_annotation", "parent_context", "root_context", "stratum", "text_digest", "duplicate_key"]
    return raw[keep].assign(platform=platform)[["platform", *keep]]


def load_candidates() -> pd.DataFrame:
    return pd.concat([_base_frame(platform) for platform in ("reddit", "bluesky", "youtube")], ignore_index=True)


def _stage_cap(frame: pd.DataFrame, group_column: str, limit: int, probability_name: str, stage: str, seed: int = SAMPLING_SEED) -> pd.DataFrame:
    selected = []
    for group_value, group in frame.groupby(group_column, sort=True, dropna=False):
        group = group.sort_values(["random_key", "doc_id"]).copy()
        take = min(limit, len(group))
        rng = random.Random(_key(f"{stage}|{group_column}|{group_value}", seed))
        selected_ids = set(rng.sample(group["doc_id"].tolist(), take))
        chosen = group[group["doc_id"].isin(selected_ids)].copy()
        chosen[probability_name] = take / len(group)
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True) if selected else frame.head(0).copy()


def _case_quotas(pool: pd.DataFrame, quota: int) -> tuple[dict[str, int], list[str]]:
    counts = pool["case_window_id"].fillna("none").astype(str).value_counts().sort_index()
    if counts.empty or quota <= 0:
        return {}, []
    allocations = {case: 0 for case in counts.index}
    floor_cases = [case for case, count in counts.items() if count >= 10]
    floor_notes = []
    if len(floor_cases) * 10 <= quota:
        for case in floor_cases:
            allocations[case] = 10
    elif floor_cases:
        floor_notes.append("case_floor_not_applied_because_quota_is_smaller_than_ten_per_available_case")
    remaining = quota - sum(allocations.values())
    if remaining <= 0:
        return allocations, floor_notes
    capacities = {case: int(counts[case] - allocations[case]) for case in counts.index}
    total_capacity = sum(max(value, 0) for value in capacities.values())
    if not total_capacity:
        return allocations, floor_notes
    raw = {case: remaining * max(capacity, 0) / total_capacity for case, capacity in capacities.items()}
    for case, value in raw.items():
        allocations[case] += min(capacities[case], int(value))
    left = quota - sum(allocations.values())
    order = sorted(raw, key=lambda case: (-(raw[case] - int(raw[case])), case))
    while left:
        changed = False
        for case in order:
            if allocations[case] < counts[case]:
                allocations[case] += 1
                left -= 1
                changed = True
                if not left:
                    break
        if not changed:
            break
    return allocations, floor_notes


def select_split_samples(candidates: pd.DataFrame, split: str, target: int | None = None, seed: int = SAMPLING_SEED) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = target or SPLIT_TARGETS[split]
    quotas = STRATUM_QUOTAS[split]
    selected: list[pd.DataFrame] = []
    used_docs: set[str] = set()
    audit: dict[str, Any] = {
        "requested_by_stratum": dict(quotas),
        "shortfalls": {},
        "transferred_to_general": 0,
        "case_allocations": {},
        "case_floor_notes": {},
        "draw_method": "seeded_uniform_without_replacement_by_group",
        "replacement_rule": "same_platform_split_stratum_case_then_general_transfer",
    }
    split_frame = candidates[candidates["split"].eq(split)].copy()
    if "text_owner_split" in split_frame:
        split_frame = split_frame[split_frame["text_owner_split"].eq(split)].copy()
    if "case_window_id" not in split_frame:
        split_frame["case_window_id"] = "none"
    split_frame["random_key"] = split_frame["doc_id"].map(lambda value: _key(f"row|{value}", seed))
    # Exact/near-duplicate text may only occur in one split. Keep the first stable row.
    split_frame = split_frame.sort_values(["random_key", "doc_id"])
    duplicate_column = "duplicate_key" if "duplicate_key" in split_frame else "text_digest"
    split_frame = split_frame.loc[~split_frame[duplicate_column].duplicated(keep="first")].copy()
    # Apply the two-stage design globally within the platform/split, not once per stratum.
    stage_one = _stage_cap(split_frame, "author_key", 2, "p_author_cap", f"{split}|author", seed)
    stage_two = _stage_cap(stage_one, "cluster_id", 10, "p_context_cap", f"{split}|context", seed)
    for stratum, quota in quotas.items():
        pool = stage_two[stage_two["stratum"].eq(stratum)].copy()
        if pool.empty:
            audit["shortfalls"][stratum] = quota
            continue
        allocations, floor_notes = _case_quotas(pool, quota)
        audit["case_allocations"][stratum] = allocations
        audit["case_floor_notes"][stratum] = floor_notes
        parts = []
        for case, case_quota in allocations.items():
            case_pool = pool[pool["case_window_id"].fillna("none").astype(str).eq(case)].copy()
            if not case_quota or case_pool.empty:
                continue
            take_count = min(case_quota, len(case_pool))
            rng = random.Random(_key(f"{split}|{stratum}|case|{case}", seed))
            selected_ids = set(rng.sample(case_pool["doc_id"].tolist(), take_count))
            take = case_pool[case_pool["doc_id"].isin(selected_ids)].copy()
            take["p_quota"] = take_count / len(case_pool)
            take["inclusion_probability"] = take["p_author_cap"] * take["p_context_cap"] * take["p_quota"]
            parts.append(take)
        take = pd.concat(parts, ignore_index=True) if parts else pool.head(0).copy()
        selected.append(take)
        used_docs.update(take["doc_id"])
        audit["shortfalls"][stratum] = max(quota - len(take), 0)
    selected_count = sum(len(part) for part in selected)
    if selected_count < target:
        remainder = stage_two[~stage_two["doc_id"].isin(used_docs)].copy()
        # Fill any unfillable stratum floors with stable remaining candidates.
        need = target - selected_count
        if need > 0 and not remainder.empty:
            rng = random.Random(_key(f"{split}|transfer|general", seed))
            filler_ids = set(rng.sample(remainder["doc_id"].tolist(), min(need, len(remainder))))
            filler = remainder[remainder["doc_id"].isin(filler_ids)].copy()
            filler["p_quota"] = len(filler) / len(remainder)
            filler["inclusion_probability"] = filler["p_author_cap"] * filler["p_context_cap"] * filler["p_quota"]
            filler["stratum"] = "G_GENERAL"
            selected.append(filler)
            audit["transferred_to_general"] = len(filler)
            audit["transfer_note"] = "quota shortfalls transferred to G_GENERAL using the frozen transfer draw"
    output = pd.concat(selected, ignore_index=True) if selected else split_frame.head(0).copy()
    if len(output) != target:
        raise ValueError(f"{split} validation sample has {len(output)} rows; required {target}")
    return output.sort_values(["stratum", "random_key", "doc_id"]).reset_index(drop=True), audit


def _source_checksums() -> dict[str, str]:
    paths = {
        "reddit": REPO_ROOT / "data" / "processed" / "reddit_age_gate" / "analysis_corpus.parquet",
        "bluesky": REPO_ROOT / "data" / "processed" / "bluesky" / "posts.parquet",
        "youtube": REPO_ROOT / "data" / "processed" / "youtube" / "documents.parquet",
    }
    return {platform: sha256_file(path) for platform, path in paths.items()}


def _blind_packet_frame(frame: pd.DataFrame, annotation_ids: list[str] | None = None) -> pd.DataFrame:
    if annotation_ids is None:
        annotation_ids = [f"annotation-{index:06d}" for index in range(1, len(frame) + 1)]
    if len(annotation_ids) != len(frame):
        raise ValueError("annotation ID count does not match packet rows")
    packet = frame[["text_for_annotation", "parent_context", "root_context"]].copy()
    packet.insert(0, "annotation_id", annotation_ids)
    return packet[list(BLIND_PACKET_COLUMNS)]


def _coder_packet_frame(packet: pd.DataFrame) -> pd.DataFrame:
    output = packet[list(BLIND_PACKET_COLUMNS)].copy()
    output["label_version"] = CODEBOOK_VERSION
    output["annotation_status"] = "unlabeled"
    for field in CODER_LABEL_FIELDS:
        output[field] = ""
    return output[list(CODER_PACKET_COLUMNS)]


def _validate_annotation_values(frame: pd.DataFrame, columns: list[str]) -> None:
    """Reject non-empty coder/adjudicator values outside the frozen codebook."""

    fields = ANNOTATION_SCHEMA["fields"]
    for column in columns:
        field = column.removeprefix("coder_a_").removeprefix("coder_b_").removeprefix("adjudicated_")
        if field not in fields or field == "notes" or column not in frame:
            continue
        allowed = set(fields[field])
        for row_number, raw_value in frame[column].items():
            value = str(raw_value).strip()
            if not value:
                continue
            values = {part.strip() for part in value.split("|")}
            if "" in values:
                raise ValueError(f"{column} contains an empty pipe-separated label at row {row_number}")
            if field == "frame_labels" and not values <= allowed:
                raise ValueError(f"{column} contains unknown frame labels: {sorted(values - allowed)}")
            if field == "bypass_techniques" and (not values <= allowed or ("none_unclear" in values and len(values) > 1)):
                raise ValueError(f"{column} contains invalid bypass technique labels: {sorted(values - allowed) or sorted(values)}")
            if field not in {"frame_labels", "bypass_techniques"} and value not in allowed:
                raise ValueError(f"{column} contains unknown {field} value: {value}")


def _register_frame(frame: pd.DataFrame) -> pd.DataFrame:
    register = frame[list(REGISTER_COLUMNS)].copy()
    return register.sort_values("annotation_id").reset_index(drop=True)


def _write_blinded_packets(frame: pd.DataFrame, split: str, output_root: Path | None = None) -> dict[str, Path]:
    output_root = output_root or VALIDATION_ROOT
    packet = _blind_packet_frame(frame, frame["annotation_id"].tolist())
    paths = {
        "packet": output_root / f"labels_{split}.csv",
        "coder_a": output_root / f"coder_a_labels_{split}.csv",
        "coder_b": output_root / f"coder_b_labels_{split}.csv",
        "register": output_root / f"coordinator_register_{split}.csv",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    packet.to_csv(paths["packet"], index=False, lineterminator="\n")
    _coder_packet_frame(packet).to_csv(paths["coder_a"], index=False, lineterminator="\n")
    _coder_packet_frame(packet).to_csv(paths["coder_b"], index=False, lineterminator="\n")
    _register_frame(frame).to_csv(paths["register"], index=False, lineterminator="\n")
    return paths


def _load_label_packet_with_coders(split: str) -> pd.DataFrame:
    if split not in SPLIT_ORDER:
        raise ValueError(f"unknown validation split: {split}")
    packet_path = VALIDATION_ROOT / f"labels_{split}.csv"
    register_path = VALIDATION_ROOT / f"coordinator_register_{split}.csv"
    if not packet_path.exists() or not register_path.exists():
        raise FileNotFoundError(f"blinded validation artifacts are incomplete for {split}")
    packet = pd.read_csv(packet_path, keep_default_na=False)
    register = pd.read_csv(register_path, keep_default_na=False)
    expected_ids = set(packet["annotation_id"])
    if len(expected_ids) != len(packet) or set(register["annotation_id"]) != expected_ids:
        raise ValueError(f"validation register and packet IDs do not match for {split}")
    frame = register.merge(packet, on="annotation_id", how="left", validate="one_to_one")
    for coder in ("a", "b"):
        values = pd.read_csv(VALIDATION_ROOT / f"coder_{coder}_labels_{split}.csv", keep_default_na=False)
        if set(values.columns) != set(CODER_PACKET_COLUMNS) or values["annotation_id"].duplicated().any() or set(values["annotation_id"]) != expected_ids:
            raise ValueError(f"coder_{coder} packet schema or IDs changed for {split}")
        if not values["label_version"].astype(str).eq(CODEBOOK_VERSION).all():
            raise ValueError(f"coder_{coder} packet label version does not match the frozen codebook for {split}")
        _validate_annotation_values(values, list(CODER_LABEL_FIELDS))
        renamed = values[["annotation_id", *CODER_LABEL_FIELDS]].rename(columns={field: f"coder_{coder}_{field}" for field in CODER_LABEL_FIELDS})
        frame = frame.merge(renamed, on="annotation_id", how="left", validate="one_to_one")
    adjudicated_fields = [f"adjudicated_{field}" for field in CODER_LABEL_FIELDS]
    for column in adjudicated_fields:
        frame[column] = ""
    coordinator = VALIDATION_ROOT / f"coordinator_labels_{split}.csv"
    if coordinator.exists():
        previous = pd.read_csv(coordinator, keep_default_na=False)
        if set(previous.get("annotation_id", [])) != expected_ids:
            raise ValueError(f"existing coordinator labels do not match for {split}")
        previous = previous.set_index("annotation_id")
        for column in adjudicated_fields:
            if column in previous:
                frame[column] = previous.loc[frame["annotation_id"], column].to_numpy()
    _validate_annotation_values(frame, [*adjudicated_fields])
    coder_columns = [f"coder_{coder}_{field}" for coder in ("a", "b") for field in CODER_LABEL_FIELDS]
    coder_present = frame[coder_columns].astype(str).apply(lambda column: column.str.strip().ne(""))
    adjudicated_present = frame[adjudicated_fields].astype(str).apply(lambda column: column.str.strip().ne(""))
    frame["annotation_status"] = "unlabeled"
    frame.loc[coder_present.any(axis=1), "annotation_status"] = "coded_unadjudicated"
    frame.loc[adjudicated_present.any(axis=1), "annotation_status"] = "adjudicated"
    return frame


def load_label_packet(split: str) -> pd.DataFrame:
    """Load current coder inputs and coordinator adjudication with hidden metadata."""

    return _load_label_packet_with_coders(split)


def merge_coder_labels(split: str) -> dict[str, Any]:
    """Merge coder files and refresh only adjudications supported by agreement."""

    if split not in SPLIT_ORDER:
        raise ValueError(f"unknown validation split: {split}")
    merged = _load_label_packet_with_coders(split)
    refreshed = {}
    for field in ("stance", "sentiment"):
        agrees = merged[f"coder_a_{field}"].eq(merged[f"coder_b_{field}"])
        merged.loc[agrees, f"adjudicated_{field}"] = merged.loc[agrees, f"coder_a_{field}"]
        refreshed[field] = int(agrees.sum())

    frame_labels = ANNOTATION_SCHEMA["fields"]["frame_labels"]
    frame_presence_agreements = 0
    for index, row in merged.iterrows():
        adjudicated = {label for label in str(row["adjudicated_frame_labels"]).split("|") if label}
        coder_a = {label for label in str(row["coder_a_frame_labels"]).split("|") if label}
        coder_b = {label for label in str(row["coder_b_frame_labels"]).split("|") if label}
        for label in frame_labels:
            if (label in coder_a) == (label in coder_b):
                frame_presence_agreements += 1
                if label in coder_a:
                    adjudicated.add(label)
                else:
                    adjudicated.discard(label)
        merged.at[index, "adjudicated_frame_labels"] = "|".join(label for label in frame_labels if label in adjudicated)
    refreshed["frame_presence_pairs"] = frame_presence_agreements

    coordinator_path = VALIDATION_ROOT / f"coordinator_labels_{split}.csv"
    merged.to_csv(coordinator_path, index=False, lineterminator="\n")
    return {"status": "merged", "split": split, "artifact": relative_path(coordinator_path), "rows": int(len(merged)), "refreshed_agreements": refreshed}


def _selection_digest(frame: pd.DataFrame) -> str:
    columns = ["platform", "split", "stratum", "doc_id", "text_digest", "cluster_id", "author_key"]
    values = frame[columns].fillna("").astype(str).sort_values(columns).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(values.encode("utf-8")).hexdigest()


def _frame_digest(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    values = frame[list(columns)].fillna("").astype(str).sort_values(list(columns)).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(values.encode("utf-8")).hexdigest()


def _expected_sample(
    seed: int = SAMPLING_SEED,
    excluded_doc_ids: set[str] | None = None,
    excluded_duplicate_keys: set[str] | None = None,
    excluded_clusters: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    check_codebook()
    candidates = load_candidates()
    excluded_doc_ids = excluded_doc_ids or set()
    excluded_duplicate_keys = excluded_duplicate_keys or set()
    excluded_clusters = excluded_clusters or set()
    if excluded_doc_ids or excluded_duplicate_keys or excluded_clusters:
        candidates = candidates[
            ~candidates["doc_id"].astype(str).isin(excluded_doc_ids)
            & ~candidates["duplicate_key"].astype(str).isin(excluded_duplicate_keys)
            & ~candidates["cluster_id"].astype(str).isin(excluded_clusters)
        ].copy()
    candidates = assign_context_splits(candidates, seed=seed)
    duplicate_column = "duplicate_key" if "duplicate_key" in candidates else "text_digest"
    text_owners = candidates.sort_values([duplicate_column, "doc_id"]).drop_duplicates(duplicate_column).set_index(duplicate_column)["split"]
    candidates["text_owner_split"] = candidates[duplicate_column].map(text_owners)
    chosen: list[pd.DataFrame] = []
    split_audits: dict[str, Any] = {}
    for platform in ("reddit", "bluesky", "youtube"):
        platform_candidates = candidates[candidates["platform"].eq(platform)].copy()
        for split in SPLIT_ORDER:
            picked, audit = select_split_samples(platform_candidates, split, seed=seed)
            picked["platform_split"] = f"{platform}:{split}"
            chosen.append(picked)
            split_audits[f"{platform}:{split}"] = audit
    sample = pd.concat(chosen, ignore_index=True)
    sample["sampling_seed"] = seed
    sample["label_version"] = CODEBOOK_VERSION
    sample["selection_key"] = sample["doc_id"].map(lambda value: _key(f"row|{value}", seed))
    sample = sample.sort_values(["platform", "split", "stratum", "selection_key", "doc_id"]).reset_index(drop=True)
    sample["annotation_id"] = [f"annotation-{index:06d}" for index in range(1, len(sample) + 1)]
    return sample, split_audits


def _sampling_metadata(sample: pd.DataFrame, split_audits: dict[str, Any]) -> dict[str, Any]:
    strata = sample.groupby(["platform", "split", "stratum"]).size().to_dict()
    author_overlap: dict[str, dict[str, int]] = {}
    unseen_author_evaluation: dict[str, int] = {}
    empty_parent_context: dict[str, dict[str, int]] = {}
    for platform in ("reddit", "bluesky", "youtube"):
        platform_sample = sample[sample["platform"].eq(platform)]
        by_split = {split: set(platform_sample.loc[platform_sample["split"].eq(split), "author_key"]) for split in SPLIT_ORDER}
        author_overlap[platform] = {
            f"{left}_vs_{right}": len(by_split[left] & by_split[right])
            for left, right in (("development", "evaluation"), ("development", "reserve"), ("evaluation", "reserve"))
        }
        unseen_author_evaluation[platform] = len(by_split["evaluation"] - by_split["development"])
        empty_parent_context[platform] = {
            split: int(
                platform_sample.loc[platform_sample["split"].eq(split), "parent_context"]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq("")
                .sum()
            )
            for split in SPLIT_ORDER
        }
    return {
        "platform_split_counts": {platform: {split: int(sample[(sample["platform"] == platform) & (sample["split"] == split)].shape[0]) for split in SPLIT_ORDER} for platform in ("reddit", "bluesky", "youtube")},
        "stratum_counts": {f"{platform}:{split}:{stratum}": int(count) for (platform, split, stratum), count in strata.items()},
        "selection_id_counts": {f"{platform}:{split}": int(((sample["platform"] == platform) & (sample["split"] == split)).sum()) for platform in ("reddit", "bluesky", "youtube") for split in SPLIT_ORDER},
        "split_audits": split_audits,
        "author_overlap_counts": author_overlap,
        "unseen_author_evaluation_counts": unseen_author_evaluation,
        "empty_parent_context_counts": empty_parent_context,
    }


def _validation_cycle_has_labels(root: Path) -> bool:
    for split in SPLIT_ORDER:
        coordinator = root / f"coordinator_labels_{split}.csv"
        if coordinator.exists():
            return True
        for coder in ("a", "b"):
            path = root / f"coder_{coder}_labels_{split}.csv"
            if not path.exists():
                continue
            values = pd.read_csv(path, keep_default_na=False)
            if any(column in values and values[column].astype(str).str.strip().ne("").any() for column in CODER_LABEL_FIELDS):
                return True
    return False


def _existing_sample_exclusions(root: Path) -> dict[str, set[str]]:
    exclusions = {"doc_ids": set(), "duplicate_keys": set(), "clusters": set()}
    register_paths = set(BASE_VALIDATION_ROOT.rglob("coordinator_register_*.csv")) if BASE_VALIDATION_ROOT.exists() else set()
    register_paths.update(root / f"coordinator_register_{split}.csv" for split in SPLIT_ORDER)
    for path in register_paths:
        if not path.is_file():
            continue
        register = pd.read_csv(path, keep_default_na=False)
        for key, column in (("doc_ids", "doc_id"), ("duplicate_keys", "duplicate_key"), ("clusters", "cluster_id")):
            if column in register:
                exclusions[key].update(register[column].astype(str))
    return exclusions


def _revision_seed(revision: str) -> int:
    return _key(f"measurement-cycle|{revision}", SAMPLING_SEED)


def _write_validation_artifacts(
    output_root: Path,
    sample: pd.DataFrame,
    split_audits: dict[str, Any],
    seed: int,
    revision: str | None = None,
    exclusions: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    packet_paths: dict[str, dict[str, str]] = {}
    for split in SPLIT_ORDER:
        paths = _write_blinded_packets(sample[sample["split"].eq(split)], split, output_root)
        packet_paths[split] = {name: relative_path(path) for name, path in paths.items()}
    metadata = _sampling_metadata(sample, split_audits)
    manifest = {
        "schema_version": "validation-sampling.v1",
        "sampling_version": SAMPLING_VERSION,
        "created_at_utc": utc_now_iso(),
        "command": f"python -m src.analysis.run validation sample --check{f' --revision {revision}' if revision else ''}",
        "sampling_seed": seed,
        "revision": revision,
        "allocation": {"per_platform": {"development": 200, "evaluation": 200, "reserve": 100}, "total": 1500},
        **{key: metadata[key] for key in ("platform_split_counts", "stratum_counts", "selection_id_counts")},
        "sampling": {
            "context_container_never_crosses_split": True,
            "author_cap_per_split": 2,
            "context_cap_per_split": 10,
            "duplicate_text_cross_split_excluded": True,
            "near_duplicate_key": "casefolded_text_with_non_alphanumeric_removed",
            "conditional_probabilities_recorded": ["p_author_cap", "p_context_cap", "p_quota", "inclusion_probability"],
            "stratum_precedence": ["L_LANGUAGE", "S_SHORT_CONTEXT", "R_RARE_DISAGREE", "G_GENERAL"],
            "case_floor_rule": "allocate ten per available case when quota permits; otherwise proportional largest-remainder allocation",
            "replacement_rule": "same platform/split/stratum/case frozen-key candidates, then logged G_GENERAL transfer",
        },
        **{key: metadata[key] for key in ("split_audits", "author_overlap_counts", "unseen_author_evaluation_counts", "empty_parent_context_counts")},
        "source_checksums": _source_checksums(),
        "human_label_boundary": {
            "gold_labels_present": False,
            "independent_double_coding_required": True,
            "model_predictions_in_packet": False,
            "rare_disagreement_rule": "rare frozen-codebook frame signature (<=10% or 5 documents) or VADER compound threshold disagreement in [0.05, 0.20]",
            "coders_must_receive_only": ["coder_a", "coder_b"],
            "coordinator_register_is_not_a_coder_export": True,
        },
        "packet_paths": packet_paths,
        "public_packet_columns": list(BLIND_PACKET_COLUMNS),
        "coder_packet_columns": list(CODER_PACKET_COLUMNS),
        "exclusions": {key: sorted(values) for key, values in (exclusions or {}).items()},
    }
    manifest["selection_digest"] = _selection_digest(sample)
    for split in SPLIT_ORDER:
        path = output_root / f"labels_{split}.csv"
        manifest.setdefault("packet_selection_digest", {})[split] = _selection_digest(sample[sample["split"].eq(split)])
        manifest.setdefault("public_packet_sha256", {})[split] = sha256_file(path)
        manifest.setdefault("register_sha256", {})[split] = sha256_file(output_root / f"coordinator_register_{split}.csv")
    write_json(output_root / "validation_manifest.json", manifest)
    return manifest


def build_validation_revision(revision: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", revision):
        raise ValueError("measurement revision must be a safe non-empty identifier")
    output_root = BASE_VALIDATION_ROOT / "revisions" / revision
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"validation revision already exists: {output_root}")
    previous_root = VALIDATION_ROOT
    exclusions = _existing_sample_exclusions(previous_root)
    seed = _revision_seed(revision)
    sample, split_audits = _expected_sample(
        seed=seed,
        excluded_doc_ids=exclusions["doc_ids"],
        excluded_duplicate_keys=exclusions["duplicate_keys"],
        excluded_clusters=exclusions["clusters"],
    )
    manifest = _write_validation_artifacts(output_root, sample, split_audits, seed, revision, exclusions)
    model_root = BASE_MODEL_ROOT / "revisions" / revision
    write_json(
        MEASUREMENT_CYCLE_STATE,
        {
            "schema_version": "measurement-cycle.v1",
            "status": "active",
            "cycle_id": revision,
            "created_at_utc": utc_now_iso(),
            "validation_root": relative_path(output_root),
            "model_root": relative_path(model_root),
            "previous_validation_root": relative_path(previous_root),
            "previous_model_root": relative_path(BASE_MODEL_ROOT),
            "recovery": "new validation labels are required before development selection and the untouched evaluation opening",
        },
    )
    return {"status": "new_measurement_cycle", "revision": revision, "artifact": relative_path(output_root / "validation_manifest.json"), "total": manifest["allocation"]["total"]}


def build_validation_artifacts(revision: str | None = None) -> dict[str, Any]:
    if revision:
        return build_validation_revision(revision)
    if _validation_cycle_has_labels(VALIDATION_ROOT):
        raise ValueError("labeled validation artifacts are frozen; use validation sample --build --revision <id> for a new cycle")
    sample, split_audits = _expected_sample(seed=SAMPLING_SEED)
    return _write_validation_artifacts(VALIDATION_ROOT, sample, split_audits, SAMPLING_SEED)


def check_validation_samples(revision: str | None = None) -> dict[str, Any]:
    root = VALIDATION_ROOT if not revision else BASE_VALIDATION_ROOT / "revisions" / revision
    manifest_path = VALIDATION_MANIFEST if not revision else root / "validation_manifest.json"
    required = [
        root / f"labels_{split}.csv"
        for split in SPLIT_ORDER
    ] + [
        root / f"coordinator_register_{split}.csv"
        for split in SPLIT_ORDER
    ] + [
        root / f"coder_{coder}_labels_{split}.csv"
        for coder in ("a", "b")
        for split in SPLIT_ORDER
    ]
    if not manifest_path.exists() or any(not path.exists() for path in required):
        raise FileNotFoundError(f"validation artifacts are missing; run the explicit validation sample --build first")
    manifest = read_json(manifest_path)
    if manifest.get("sampling_version") != SAMPLING_VERSION:
        raise ValueError("validation sampling algorithm version changed; rebuild the packets")
    expected = {platform: {split: 200 if split != "reserve" else 100 for split in SPLIT_ORDER} for platform in ("reddit", "bluesky", "youtube")}
    if manifest.get("platform_split_counts") != expected:
        raise ValueError("validation allocation is not the required 600/600/300 split")
    exclusions = manifest.get("exclusions", {})
    expected_sample, expected_audits = _expected_sample(
        seed=int(manifest.get("sampling_seed", SAMPLING_SEED)),
        excluded_doc_ids=set(map(str, exclusions.get("doc_ids", []))),
        excluded_duplicate_keys=set(map(str, exclusions.get("duplicate_keys", []))),
        excluded_clusters=set(map(str, exclusions.get("clusters", []))),
    )
    expected_metadata = _sampling_metadata(expected_sample, expected_audits)
    for key, value in expected_metadata.items():
        if manifest.get(key) != value:
            raise ValueError(f"validation {key} does not match the deterministic sampler")
    if manifest.get("selection_digest") != _selection_digest(expected_sample):
        raise ValueError("validation selection digest does not match the deterministic sampler")
    for split in SPLIT_ORDER:
        path = root / f"labels_{split}.csv"
        if not path.exists():
            raise ValueError(f"validation packet is missing: {split}")
        packet = pd.read_csv(path, keep_default_na=False)
        if set(packet.columns) != set(BLIND_PACKET_COLUMNS):
            raise ValueError(f"blind packet exposes metadata or has changed schema: {split}")
        if packet["annotation_id"].duplicated().any():
            raise ValueError(f"blind packet annotation IDs are not unique: {split}")
        if sha256_file(path) != manifest.get("public_packet_sha256", {}).get(split):
            raise ValueError(f"validation packet changed outside the declared artifact: {split}")
        expected_split = expected_sample[expected_sample["split"].eq(split)].copy()
        if _frame_digest(packet, BLIND_PACKET_COLUMNS) != _frame_digest(_blind_packet_frame(expected_split, expected_split["annotation_id"].tolist()), BLIND_PACKET_COLUMNS):
            raise ValueError(f"validation packet does not match the deterministic sampler: {split}")
        register_path = root / f"coordinator_register_{split}.csv"
        register = pd.read_csv(register_path, keep_default_na=False)
        if set(register.columns) != set(REGISTER_COLUMNS) or register["annotation_id"].duplicated().any():
            raise ValueError(f"coordinator register schema changed: {split}")
        if set(register["annotation_id"]) != set(packet["annotation_id"]):
            raise ValueError(f"coordinator register IDs do not match packet: {split}")
        if _selection_digest(register) != _selection_digest(expected_split):
            raise ValueError(f"validation register does not match the deterministic sampler: {split}")
        if not register["split"].astype(str).eq(split).all() or not register["text_owner_split"].astype(str).eq(split).all():
            raise ValueError(f"validation register split ownership is invalid: {split}")
        if register["doc_id"].duplicated().any() or register["duplicate_key"].duplicated().any():
            raise ValueError(f"validation register contains duplicate documents or text: {split}")
        if register.groupby("author_key").size().max() > 2 or register.groupby("cluster_id").size().max() > 10:
            raise ValueError(f"validation author or context cap is exceeded: {split}")
        seed = int(manifest.get("sampling_seed", SAMPLING_SEED))
        if not pd.to_numeric(register["sampling_seed"], errors="coerce").eq(seed).all() or not register["label_version"].astype(str).eq(CODEBOOK_VERSION).all():
            raise ValueError(f"validation register seed or label version changed: {split}")
        expected_selection_keys = register["doc_id"].map(lambda value: _key(f"row|{value}", seed))
        if not pd.to_numeric(register["selection_key"], errors="coerce").eq(expected_selection_keys).all():
            raise ValueError(f"validation register selection keys changed: {split}")
        probabilities = register[["p_author_cap", "p_context_cap", "p_quota", "inclusion_probability"]].apply(pd.to_numeric, errors="coerce")
        if probabilities.isna().any().any() or (probabilities <= 0).any().any() or (probabilities > 1).any().any():
            raise ValueError(f"validation probabilities are not finite positive values in (0, 1]: {split}")
        expected_probability = probabilities["p_author_cap"] * probabilities["p_context_cap"] * probabilities["p_quota"]
        if not (expected_probability - probabilities["inclusion_probability"]).abs().lt(1e-12).all():
            raise ValueError(f"validation inclusion probabilities do not equal the declared stage product: {split}")
        if _selection_digest(register) != manifest.get("packet_selection_digest", {}).get(split):
            raise ValueError(f"validation selection changed outside the declared artifact: {split}")
        if sha256_file(register_path) != manifest.get("register_sha256", {}).get(split):
            raise ValueError(f"coordinator register changed outside the declared artifact: {split}")
        for coder in ("a", "b"):
            coder_path = root / f"coder_{coder}_labels_{split}.csv"
            coder_packet = pd.read_csv(coder_path, keep_default_na=False)
            if set(coder_packet.columns) != set(CODER_PACKET_COLUMNS) or coder_packet["annotation_id"].duplicated().any():
                raise ValueError(f"coder packet schema changed: coder_{coder}/{split}")
            if set(coder_packet["annotation_id"]) != set(packet["annotation_id"]):
                raise ValueError(f"coder packet IDs do not match packet: coder_{coder}/{split}")
        coordinator_path = root / f"coordinator_labels_{split}.csv"
        if coordinator_path.exists():
            coordinator = pd.read_csv(coordinator_path, keep_default_na=False)
            if not set(LABEL_COLUMNS).issubset(coordinator.columns) or set(coordinator["annotation_id"]) != set(packet["annotation_id"]):
                raise ValueError(f"coordinator labels are inconsistent: {split}")
    if manifest.get("source_checksums") != _source_checksums():
        raise ValueError("validation source checksum changed; frozen inputs must not be replaced")
    if revision and manifest.get("revision") != revision:
        raise ValueError("validation manifest revision does not match the requested cycle")
    return {"status": "valid", "artifact": relative_path(manifest_path), "total": manifest["allocation"]["total"]}
