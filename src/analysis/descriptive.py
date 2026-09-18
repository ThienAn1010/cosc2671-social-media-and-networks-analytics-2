"""Coverage, exclusions, concentration and event-timeline aggregates."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.artifacts import ANALYSIS_ROOT, REPO_ROOT, read_json, relative_path, sha256_file, utc_now_iso, write_json
from src.analysis.codebook import CODEBOOK_VERSION
from src.analysis.populations import POPULATION_ARTIFACT
from src.analysis.scope import SCOPE_ARTIFACT_PATH
from src.shared.case_windows import REDDIT_CASE_BY_EVENT, load_case_windows

DESCRIPTIVE_ROOT = ANALYSIS_ROOT / "descriptive"
DESCRIPTIVE_MANIFEST = DESCRIPTIVE_ROOT / "descriptive_manifest.json"
DESCRIPTIVE_SCHEMA_VERSION = "descriptive.v2"
MIN_DAILY_DOCS = 20
CORE_CASES = {window.case_window_id for window in load_case_windows()}


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna(False).astype(bool)


def _language_masks(frame: pd.DataFrame, status_column: str = "exclusion_status") -> tuple[pd.Series, pd.Series]:
    strict = _bool(frame, "is_english") & frame[status_column].eq("eligible")
    uncertain = _bool(frame, "is_language_uncertain") if "is_language_uncertain" in frame else pd.Series(False, index=frame.index)
    inclusive_status = frame[status_column].isin(["eligible", "language_uncertain"])
    inclusive_status |= frame[status_column].eq("non_english") & uncertain
    inclusive = inclusive_status & (_bool(frame, "is_english") | uncertain)
    return strict, inclusive


def _safe_text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna("").astype(str)


def _author_balanced_frame_share(frame: pd.DataFrame, flag: pd.Series, author_column: str = "author_hash") -> float | None:
    usable = frame[author_column].fillna("").astype(str).ne("")
    if not usable.any():
        return None
    per_author = flag[usable].groupby(frame.loc[usable, author_column]).mean()
    return float(per_author.mean()) if not per_author.empty else None


def _concentration_rows(frame: pd.DataFrame, platform: str, scope: str, container_column: str) -> list[dict[str, Any]]:
    usable = frame[frame["author_hash"].fillna("").astype(str).ne("")].copy()
    if usable.empty:
        return []
    author_counts = usable.groupby("author_hash").size().sort_values(ascending=False)
    container_counts = usable.groupby(container_column).size().sort_values(ascending=False)
    def row(entity: str, counts: pd.Series) -> dict[str, Any]:
        proportions = counts / counts.sum()
        return {
            "platform": platform,
            "scope": scope,
            "entity": entity,
            "documents": int(counts.sum()),
            "unique_entities": int(len(counts)),
            "top_1_share": float(proportions.iloc[0]) if len(proportions) else None,
            "top_10_share": float(proportions.head(10).sum()) if len(proportions) else None,
            "effective_number": float(1 / (proportions.pow(2).sum())) if len(proportions) else None,
        }
    return [row("author", author_counts), row(container_column, container_counts)]


def _coverage_row(platform: str, scope: str, frame: pd.DataFrame, strict: pd.Series, inclusive: pd.Series, container_column: str) -> dict[str, Any]:
    authors = frame.loc[frame["author_hash"].fillna("").astype(str).ne(""), "author_hash"].nunique()
    return {
        "platform": platform,
        "scope": scope,
        "rows": int(len(frame)),
        "strict_english_rows": int(strict.sum()),
        "inclusive_english_rows": int(inclusive.sum()),
        "language_uncertain_rows": int(frame.get("is_language_uncertain", pd.Series(False, index=frame.index)).fillna(False).astype(bool).sum()),
        "unique_authors": int(authors),
        "unique_containers": int(frame[container_column].fillna("").astype(str).replace("", pd.NA).dropna().nunique()),
        "exclusions": {str(key): int(value) for key, value in frame["exclusion_status"].value_counts(dropna=False).items()},
        "human_calibrated_status": "not_available_without_human_language_labels",
    }


def _reddit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = REPO_ROOT / "data" / "processed" / "reddit_age_gate" / "analysis_corpus.parquet"
    columns = ["thing", "event_id", "date", "subreddit", "exclusion_status", "author_hash", "thread_id", "is_english", "is_language_uncertain", "text_topic", "is_url_only", "is_no_substantive_text", "relevance_frames"]
    raw = pd.read_parquet(path, columns=columns)
    raw["scope"] = raw["event_id"].map(REDDIT_CASE_BY_EVENT).fillna("none")
    strict, inclusive = _language_masks(raw)
    substantive = _safe_text_column(raw, "text_topic").str.strip().ne("") & ~_bool(raw, "is_url_only") & ~_bool(raw, "is_no_substantive_text")
    base = raw[substantive].copy()
    base["strict"] = strict[substantive].to_numpy()
    base["inclusive"] = inclusive[substantive].to_numpy()
    coverage = pd.DataFrame([_coverage_row("reddit", scope, group, strict[substantive].loc[group.index], inclusive[substantive].loc[group.index], "thread_id") for scope, group in base.groupby("scope") if scope != "none"])
    concentration = pd.DataFrame([row for scope, group in base[base["scope"].isin(CORE_CASES)].groupby("scope") for row in _concentration_rows(group[group["inclusive"]], "reddit", scope, "thread_id")])
    return raw, coverage, concentration, base


def _bluesky() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = REPO_ROOT / "data" / "processed" / "bluesky" / "posts.parquet"
    columns = ["doc_id", "root_doc_id", "day", "event_window", "event_id_nearest", "author_hash", "is_english", "is_reply", "in_search", "in_thread", "is_meme", "is_repeat_burst", "text_raw"]
    raw = pd.read_parquet(path, columns=columns)
    raw["scope"] = raw["event_window"].where(raw["event_window"].isin(["E1", "E2", "E3", "E4", "E5"]), "none")
    raw["root_doc_id"] = raw["root_doc_id"].fillna(raw["doc_id"])
    english = _bool(raw, "is_english")
    short = raw["text_raw"].fillna("").astype(str).str.len().lt(24)
    quality_flag = _bool(raw, "is_meme") | _bool(raw, "is_repeat_burst")
    raw["exclusion_status"] = "eligible"
    raw.loc[~english, "exclusion_status"] = "non_english"
    raw.loc[english & short, "exclusion_status"] = "language_uncertain"
    raw.loc[english & quality_flag, "exclusion_status"] = "quality_flagged"
    raw["is_language_uncertain"] = english & short
    strict, inclusive = _language_masks(raw)
    base = raw[raw["text_raw"].fillna("").astype(str).str.strip().ne("")].copy()
    base["strict"] = strict[base.index].to_numpy()
    base["inclusive"] = inclusive[base.index].to_numpy()
    coverage = pd.DataFrame([_coverage_row("bluesky", scope, group, strict.loc[group.index], inclusive.loc[group.index], "root_doc_id") for scope, group in base.groupby("scope") if scope != "none"])
    concentration = pd.DataFrame([row for scope, group in base[base["scope"].isin(["E2", "E3"])].groupby("scope") for row in _concentration_rows(group[group["inclusive"]], "bluesky", scope, "root_doc_id")])
    return raw, coverage, concentration, base


def _youtube() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = REPO_ROOT / "data" / "processed" / "youtube" / "documents.parquet"
    columns = ["doc_id", "thing", "yt_video_event_id", "created_date_utc", "exclusion_status", "author_hash", "root_doc_id", "is_english", "is_language_uncertain", "yt_video_type", "text_clean"]
    raw = pd.read_parquet(path, columns=columns)
    raw = raw[raw["thing"].ne("video")].copy()
    raw["scope"] = raw["yt_video_event_id"]
    strict, inclusive = _language_masks(raw)
    base = raw[raw["text_clean"].fillna("").astype(str).str.strip().ne("")].copy()
    base["strict"] = strict[base.index].to_numpy()
    base["inclusive"] = inclusive[base.index].to_numpy()
    coverage = pd.DataFrame([_coverage_row("youtube", scope, group, strict.loc[group.index], inclusive.loc[group.index], "root_doc_id") for scope, group in base.groupby("scope")])
    concentration = pd.DataFrame([row for scope, group in base[base["scope"].isin(["E2", "E3"])].groupby("scope") for row in _concentration_rows(group[group["inclusive"]], "youtube", scope, "root_doc_id")])
    return raw, coverage, concentration, base


def _legacy_frame_sensitivity() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reddit = pd.read_parquet(REPO_ROOT / "data" / "processed" / "reddit_age_gate" / "analysis_corpus.parquet", columns=["event_id", "author_hash", "relevance_frames", "text_topic", "exclusion_status"])
    reddit = reddit[reddit["exclusion_status"].isin(["eligible", "language_uncertain"]) & reddit["text_topic"].fillna("").astype(str).str.strip().ne("")]
    reddit["scope"] = reddit["event_id"].map(REDDIT_CASE_BY_EVENT).fillna("none")
    for scope, group in reddit[reddit["scope"].isin(CORE_CASES)].groupby("scope"):
        for frame_label in ("age_policy", "child_safety", "privacy_surveillance", "governance", "circumvention_autonomy"):
            flag = group["relevance_frames"].fillna("").astype(str).str.split("|").map(lambda values: frame_label in values)
            rows.append({"platform": "reddit", "scope": scope, "signal": frame_label, "document_weighted_share": float(flag.mean()), "author_balanced_share": _author_balanced_frame_share(group, flag), "documents": int(len(group)), "authors": int(group["author_hash"].nunique()), "status": "legacy_sensitivity_only"})
    blue = pd.read_parquet(REPO_ROOT / "data" / "processed" / "bluesky" / "frames.parquet", columns=["doc_id", "author_hash", "privacy", "circumvent", "child_safety", "free_speech", "id_upload"])
    posts = pd.read_parquet(REPO_ROOT / "data" / "processed" / "bluesky" / "posts.parquet", columns=["doc_id", "event_window", "is_english"])
    blue = blue.merge(posts, on="doc_id", how="left")
    for scope, group in blue[blue["event_window"].isin(["E2", "E3"])].groupby("event_window"):
        for old_label in ("privacy", "circumvent", "child_safety", "free_speech", "id_upload"):
            flag = group[old_label].fillna(False).astype(bool)
            rows.append({"platform": "bluesky", "scope": scope, "signal": old_label, "document_weighted_share": float(flag.mean()) if len(group) else None, "author_balanced_share": _author_balanced_frame_share(group, flag), "documents": int(len(group)), "authors": int(group["author_hash"].nunique()), "status": "legacy_sensitivity_only"})
    return pd.DataFrame(rows)


def _timeline(base: pd.DataFrame, platform: str, date_column: str, scope_column: str) -> pd.DataFrame:
    rows = []
    for scope, group in base[base[scope_column].isin(CORE_CASES | {"E2", "E3"})].groupby(scope_column):
        dates = pd.to_datetime(group[date_column], errors="coerce").dt.date.dropna()
        if dates.empty:
            continue
        start, end = dates.min(), dates.max()
        by_day = group.assign(_day=dates).groupby("_day").size()
        active = group[group["inclusive"]].copy()
        active["_day"] = pd.to_datetime(active[date_column], errors="coerce").dt.date
        for offset in range((end - start).days + 1):
            day = start + timedelta(days=offset)
            n = int(by_day.get(day, 0)) if day in by_day.index else None
            eligible = active[active["_day"].eq(day)]
            rows.append({"platform": platform, "scope": scope, "day": day.isoformat(), "eligible_documents": int(len(eligible)) if n is not None else None, "all_documents": n, "active_authors": int(eligible["author_hash"].nunique()) if n is not None else None, "support_status": "supported" if n is not None and len(eligible) >= MIN_DAILY_DOCS else ("thin" if n is not None else "missing"), "minimum_documents": MIN_DAILY_DOCS})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["aggregation_recommendation"] = "daily"
    for key, group in result.groupby(["platform", "scope"]):
        if (group["support_status"].isin(["thin", "missing"])).mean() > 1 / 3:
            result.loc[group.index, "aggregation_recommendation"] = "weekly"
    return result


def _input_checksums() -> dict[str, str]:
    paths = {
        "reddit": REPO_ROOT / "data" / "processed" / "reddit_age_gate" / "analysis_corpus.parquet",
        "bluesky": REPO_ROOT / "data" / "processed" / "bluesky" / "posts.parquet",
        "youtube": REPO_ROOT / "data" / "processed" / "youtube" / "documents.parquet",
        "bluesky_frames": REPO_ROOT / "data" / "processed" / "bluesky" / "frames.parquet",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def build_descriptive_artifacts() -> dict[str, Any]:
    scope = read_json(SCOPE_ARTIFACT_PATH)
    population = read_json(POPULATION_ARTIFACT)
    reddit_raw, reddit_coverage, reddit_concentration, reddit_base = _reddit()
    bluesky_raw, bluesky_coverage, bluesky_concentration, bluesky_base = _bluesky()
    youtube_raw, youtube_coverage, youtube_concentration, youtube_base = _youtube()
    coverage = pd.concat([reddit_coverage, bluesky_coverage, youtube_coverage], ignore_index=True).sort_values(["platform", "scope"]).reset_index(drop=True)
    concentration = pd.concat([reddit_concentration, bluesky_concentration, youtube_concentration], ignore_index=True).sort_values(["platform", "scope", "entity"]).reset_index(drop=True)
    timelines = pd.concat([
        _timeline(reddit_base, "reddit", "date", "scope"),
        _timeline(bluesky_base, "bluesky", "day", "scope"),
        _timeline(youtube_base, "youtube", "created_date_utc", "scope"),
    ], ignore_index=True)
    legacy = _legacy_frame_sensitivity()
    DESCRIPTIVE_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = {"coverage": DESCRIPTIVE_ROOT / "coverage.csv", "concentration": DESCRIPTIVE_ROOT / "concentration.csv", "timeline": DESCRIPTIVE_ROOT / "timeline.csv", "legacy_frame_sensitivity": DESCRIPTIVE_ROOT / "legacy_frame_sensitivity.csv"}
    for name, frame in (("coverage", coverage), ("concentration", concentration), ("timeline", timelines), ("legacy_frame_sensitivity", legacy)):
        frame.to_csv(outputs[name], index=False, lineterminator="\n")
    manifest = {
        "schema_version": DESCRIPTIVE_SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "command": "python -m src.analysis.run descriptive --check",
        "scope_version": scope["scope_version"],
        "population_version": population["population_version"],
        "codebook_version": CODEBOOK_VERSION,
        "minimum_daily_documents": MIN_DAILY_DOCS,
        "language_populations": {"strict": "is_english AND exclusion_status=eligible", "inclusive": "eligible/language_uncertain plus records explicitly marked language-uncertain despite non_english status", "human_calibrated": "unavailable_without_human_labels"},
        "source_checksums": _input_checksums(),
        "outputs": {name: {"path": relative_path(path), "sha256": sha256_file(path), "rows": int(pd.read_csv(path).shape[0])} for name, path in outputs.items()},
        "quality_notes": ["legacy frame fields are sensitivity-only", "missing or thin daily support is not treated as zero", "no user geography is inferred from platform or community context"],
    }
    write_json(DESCRIPTIVE_MANIFEST, manifest)
    return manifest


def check_descriptive() -> dict[str, Any]:
    if not DESCRIPTIVE_MANIFEST.exists():
        raise FileNotFoundError(f"descriptive manifest is missing: {DESCRIPTIVE_MANIFEST}; run the explicit descriptive build first")
    manifest = read_json(DESCRIPTIVE_MANIFEST)
    if manifest.get("schema_version") != DESCRIPTIVE_SCHEMA_VERSION:
        raise ValueError("descriptive manifest schema is stale; run the explicit descriptive build")
    if manifest.get("source_checksums") != _input_checksums():
        raise ValueError("descriptive source checksum changed")
    for record in manifest.get("outputs", {}).values():
        path = REPO_ROOT / record["path"]
        if not path.exists() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"descriptive output changed: {record['path']}")
    return {"status": "valid", "artifact": relative_path(DESCRIPTIVE_MANIFEST), "outputs": len(manifest.get("outputs", {}))}
