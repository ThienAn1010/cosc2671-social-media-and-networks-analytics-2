# Student name: Nguyen Hoang Thien An
# Student ID: s3825455
# Assignment 1 - Social Media and Networks Analytics

"""Reliable Arctic Shift collection for the Reddit Age-Gate Paradox study."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE = "https://arctic-shift.photon-reddit.com"
API_DOCS_URL = "https://github.com/ArthurHeitmann/arctic_shift/blob/master/api/README.md"
STUDY_NAME = "reddit_age_gate_paradox"
STUDY_SCOPE = (
    "English-language Reddit discussions of how child-safety and privacy or "
    "surveillance arguments develop inside threaded community discussions; "
    "subreddit membership represents community participation, not residence."
)
PLATFORM = "reddit"
SUBREDDIT_SCOPE = "community_participation_not_residence"

TARGET_SUBREDDITS = [
    "privacy",
    "technology",
    "parenting",
    "unitedkingdom",
    "europe",
    "australia",
]

AGE_ASSURANCE_QUERIES = [
    "age verification",
    "age assurance",
    "age check",
    "age estimation",
    "age gate",
    "under 16",
    "social media ban",
    "child safety",
    "online safety",
    "digital ID",
    "digital identity",
    "facial age estimation",
    "privacy surveillance",
    "biometric verification",
    "VPN age verification",
]
BROAD_RECALL_QUERIES = [
    "under 16",
    "social media ban",
    "child safety",
    "online safety",
    "digital ID",
    "digital identity",
    "privacy surveillance",
]
RELEVANCE_SCREENING = {
    "mode": "candidate_recall_then_audited_filter",
    "broad_queries": BROAD_RECALL_QUERIES,
    "classifier": "src.prepare_corpus_text.classify_relevance_detail",
    "predicate": "age_assurance_related == true",
    "analysis_masks": ["topic_eligible", "sentiment_eligible"],
}

# Windows are policy-event windows, not one broad date range. Every window is
# searched in every target subreddit so community differences remain comparable.
EVENT_WINDOWS = {
    "australia_legislation": {
        "jurisdiction": "australia",
        "start_utc": "2024-11-10T00:00:00Z",
        "end_utc": "2025-01-11T00:00:00Z",
    },
    "uk_eu_policy_cluster": {
        "jurisdiction": "uk_eu",
        "start_utc": "2025-06-14T00:00:00Z",
        "end_utc": "2025-08-26T00:00:00Z",
    },
    "australia_implementation": {
        "jurisdiction": "australia",
        "start_utc": "2025-11-10T00:00:00Z",
        "end_utc": "2026-01-11T00:00:00Z",
    },
}
START_UTC = EVENT_WINDOWS["australia_legislation"]["start_utc"]
END_UTC = EVENT_WINDOWS["australia_implementation"]["end_utc"]

# These match Arctic Shift's documented selectable fields. Search uses query
# for post title/selftext; comments are obtained only through /comments/tree.
POST_FIELDS = (
    "id,created_utc,subreddit,author,author_fullname,author_flair_text,"
    "distinguished,title,selftext,score,num_comments,url"
)
COMMENT_FIELDS = (
    "id,created_utc,subreddit,author,author_fullname,author_flair_text,"
    "distinguished,body,score,link_id,parent_id"
)
COMMENT_FIELD_NAMES = tuple(COMMENT_FIELDS.split(","))
COMMENT_TREE_LIMIT = 25000


@dataclass
class Stats:
    requests: int = 0
    run_requests: int = 0
    retries: int = 0
    rows_seen: int = 0
    rows_written: int = 0
    posts_written: int = 0
    comments_written: int = 0
    duplicates: int = 0
    invalid_rows: int = 0
    splits: int = 0
    truncated_windows: int = 0
    errors: int = 0
    acquisition_matches: int = 0
    out_of_window_rows: int = 0
    unknown_tree_nodes: int = 0
    collapsed_tree_nodes: int = 0

    @classmethod
    def from_manifest(cls, values: Any) -> "Stats":
        if not isinstance(values, dict):
            return cls()
        fields = tuple(cls.__dataclass_fields__)
        restored: dict[str, int] = {}
        for field_name in fields:
            if field_name == "run_requests":
                restored[field_name] = 0
                continue
            try:
                restored[field_name] = int(values.get(field_name, 0) or 0)
            except (TypeError, ValueError):
                restored[field_name] = 0
        return cls(**restored)


@dataclass(frozen=True)
class SearchTask:
    subreddit: str
    event_id: str
    jurisdiction: str
    window_start_utc: str
    window_end_utc: str
    query: str

    @property
    def start_epoch(self) -> int:
        return parse_utc(self.window_start_utc)

    @property
    def end_epoch(self) -> int:
        return parse_utc(self.window_end_utc)


@dataclass
class TaskResult:
    thing: str
    subreddit: str
    event_id: str
    jurisdiction: str
    window_start_utc: str
    window_end_utc: str
    query: str | None = None
    post_id: str | None = None
    matched_queries: list[str] = field(default_factory=list)
    status: str = "pending"
    rows_written: int = 0
    duplicates: int = 0
    requests_used: int = 0
    splits_used: int = 0
    truncated_windows: int = 0
    rows_filtered_outside_window: int = 0
    unknown_tree_nodes: int = 0
    collapsed_tree_nodes: int = 0
    error: str = ""


@dataclass
class IntervalResult:
    status: str
    reason: str = ""


class RequestBudgetExceeded(RuntimeError):
    pass


class ApiTimeoutError(RuntimeError):
    pass


class MalformedApiResponseError(RuntimeError):
    pass


def is_timeout_message(value: Any) -> bool:
    text = str(value).casefold()
    return "timeout" in text or "timed out" in text


def parse_utc(value: str) -> int:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone offset or Z suffix")
    return int(parsed.timestamp())


def iso_utc(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def event_window_records() -> list[dict[str, str]]:
    return [{"event_id": event_id, **window} for event_id, window in EVENT_WINDOWS.items()]


def make_output_dir(root: Path, overwrite: bool) -> Path:
    if overwrite:
        root.mkdir(parents=True, exist_ok=True)
        return root
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / f"arctic_shift_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def ensure_output_files_safe(paths: list[Path], resume: bool) -> None:
    if resume:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise SystemExit(
            "output files already exist without --resume: "
            + ", ".join(existing)
            + ". choose a new --out-root or use --resume"
        )


def validate_resume_manifest(out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = out_dir / "collection_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"--resume requires a collection manifest at {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read resume manifest at {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise SystemExit(f"resume manifest must be a JSON object: {manifest_path}")

    mismatches: list[str] = []
    if int(manifest.get("manifest_version", 0) or 0) < 5:
        mismatches.append("manifest_version")
    if manifest.get("study_name") != STUDY_NAME:
        mismatches.append("study_name")
    if manifest.get("platform") != PLATFORM:
        mismatches.append("platform")
    if manifest.get("api_base") != API_BASE:
        mismatches.append("api_base")
    if manifest.get("target_subreddits") != TARGET_SUBREDDITS:
        mismatches.append("target_subreddits")
    if manifest.get("age_assurance_queries") != AGE_ASSURANCE_QUERIES:
        mismatches.append("age_assurance_queries")
    if manifest.get("relevance_screening") != RELEVANCE_SCREENING:
        mismatches.append("relevance_screening")
    if manifest.get("event_windows") != event_window_records():
        mismatches.append("event_windows")
    expected_files = {
        "posts": "posts_raw.jsonl",
        "comments": "comments_raw.jsonl",
        "acquisition_routes": "acquisition_routes.jsonl",
    }
    files = manifest.get("files")
    if not isinstance(files, dict):
        mismatches.append("files")
    else:
        for field_name, filename in expected_files.items():
            if files.get(field_name) != filename:
                mismatches.append(f"files.{field_name}")
            elif not (out_dir / filename).is_file():
                mismatches.append(f"missing {filename}")

    previous_args = manifest.get("args")
    if not isinstance(previous_args, dict):
        mismatches.append("args")
    else:
        for field_name in ("limit", "min_window_seconds", "comment_tree_limit"):
            if previous_args.get(field_name) != getattr(args, field_name):
                mismatches.append(f"args.{field_name}")

    if mismatches:
        raise SystemExit(
            "cannot resume incompatible collection manifest; mismatched fields: "
            + ", ".join(mismatches)
            + f". choose a new --out-root: {manifest_path}"
        )
    return manifest


def write_jsonl_lines(path: Path, wrappers: list[dict[str, Any]]) -> None:
    if not wrappers:
        return
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for wrapper in wrappers:
            handle.write(json.dumps(wrapper, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def canonical_id(value: Any, thing: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return ""
    if text.startswith(("t1_", "t3_")):
        return text
    return f"{'t1' if thing == 'comment' else 't3'}_{text}"


def record_key(thing: str, value: Any) -> str:
    return f"{thing}:{canonical_id(value, thing)}"


def load_seen(path: Path, thing: str) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            wrapper = json.loads(line)
            data = wrapper.get("data")
            if isinstance(data, dict) and data.get("id"):
                seen.add(record_key(thing, data["id"]))
    return seen


def match_key(match: dict[str, Any]) -> str:
    fields = {
        "event_id": match.get("event_id"),
        "jurisdiction": match.get("jurisdiction"),
        "window_start_utc": match.get("window_start_utc"),
        "window_end_utc": match.get("window_end_utc"),
        "query": match.get("query"),
    }
    return json.dumps(fields, ensure_ascii=False, sort_keys=True)


def normalize_matches(collection: dict[str, Any]) -> list[dict[str, str]]:
    raw_matches = collection.get("acquisition_matches")
    if not isinstance(raw_matches, list):
        raw_matches = []
    matches: dict[str, dict[str, str]] = {}
    for raw in raw_matches:
        if not isinstance(raw, dict):
            continue
        match = {
            "event_id": str(raw.get("event_id") or ""),
            "jurisdiction": str(raw.get("jurisdiction") or ""),
            "window_start_utc": str(raw.get("window_start_utc") or ""),
            "window_end_utc": str(raw.get("window_end_utc") or ""),
            "query": str(raw.get("query") or ""),
        }
        if all(match.values()):
            matches[match_key(match)] = match
    if not matches:
        fallback = {
            "event_id": str(collection.get("event_id") or ""),
            "jurisdiction": str(collection.get("jurisdiction") or ""),
            "window_start_utc": str(collection.get("window_start_utc") or ""),
            "window_end_utc": str(collection.get("window_end_utc") or ""),
            "query": str(collection.get("query") or ""),
        }
        if all(fallback.values()):
            matches[match_key(fallback)] = fallback
    return [matches[key] for key in sorted(matches)]


def with_matches(collection: dict[str, Any], matches: list[dict[str, str]]) -> dict[str, Any]:
    result = dict(collection)
    result["acquisition_matches"] = matches
    result["matched_queries"] = sorted({match["query"] for match in matches})
    result["acquisition_match_count"] = len(matches)
    if matches:
        first = matches[0]
        result.setdefault("event_id", first["event_id"])
        result.setdefault("jurisdiction", first["jurisdiction"])
        result.setdefault("window_start_utc", first["window_start_utc"])
        result.setdefault("window_end_utc", first["window_end_utc"])
        result.setdefault("query", first["query"])
    return result


def acquisition_route_key(thing: str, row_id: Any, match: dict[str, Any], endpoint: str) -> str:
    return f"{record_key(thing, row_id)}:{endpoint}:{match_key(match)}"


def load_seen_routes(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            route = json.loads(line)
            match = route.get("match")
            if isinstance(match, dict) and route.get("thing") and route.get("record_id"):
                seen.add(acquisition_route_key(str(route["thing"]), route["record_id"], match, str(route.get("endpoint") or "")))
    return seen


def reconcile_persisted_stats(
    stats: Stats,
    posts_seen: set[str],
    comments_seen: set[str],
    routes_seen: set[str],
) -> None:
    persisted = {
        "posts_written": len(posts_seen),
        "comments_written": len(comments_seen),
        "acquisition_matches": len(routes_seen),
    }
    shortfalls = [
        name for name, count in persisted.items() if count < getattr(stats, name)
    ]
    if shortfalls:
        raise SystemExit(
            "resume output files contain fewer records than the manifest: "
            + ", ".join(shortfalls)
        )
    persisted_rows = persisted["posts_written"] + persisted["comments_written"]
    unmanifested_rows = max(0, persisted_rows - stats.rows_written)
    stats.rows_seen = max(persisted_rows, stats.rows_seen + unmanifested_rows)
    stats.rows_written = persisted_rows
    stats.posts_written = persisted["posts_written"]
    stats.comments_written = persisted["comments_written"]
    stats.acquisition_matches = persisted["acquisition_matches"]


def refresh_comment_tree_provenance(
    comments_file: Path,
    acquisition_file: Path,
    targets: list[dict[str, Any]],
    routes_seen: set[str],
    stats: Stats,
) -> None:
    matches_by_post: dict[str, list[tuple[Any, dict[str, str]]]] = {}
    for target in targets:
        post_id = canonical_id(target.get("post_id"), "post")
        if not post_id:
            continue
        for query in sorted(set(target.get("queries") or [])):
            match = {
                "event_id": target["event_id"],
                "jurisdiction": target["jurisdiction"],
                "window_start_utc": target["window_start_utc"],
                "window_end_utc": target["window_end_utc"],
                "query": query,
            }
            matches_by_post.setdefault(post_id, []).append((target.get("subreddit"), match))
    routes: list[dict[str, Any]] = []
    if not comments_file.exists() or not matches_by_post:
        return
    with comments_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            wrapper = json.loads(line)
            data = wrapper.get("data")
            if not isinstance(data, dict):
                continue
            matches = matches_by_post.get(canonical_id(data.get("link_id"), "post"))
            if not matches:
                continue
            comment_id = canonical_id(data.get("id"), "comment")
            if not comment_id:
                continue
            for subreddit, match in matches:
                route_key = acquisition_route_key("comment", comment_id, match, "/api/comments/tree")
                if route_key in routes_seen:
                    continue
                routes_seen.add(route_key)
                routes.append(
                    {
                        "thing": "comment",
                        "record_id": comment_id,
                        "subreddit": subreddit,
                        "endpoint": "/api/comments/tree",
                        "match": match,
                    }
                )
    write_jsonl_lines(acquisition_file, routes)
    stats.acquisition_matches += len(routes)


def ensure_request_budget(stats: Stats, max_requests: int | None) -> None:
    if max_requests is not None and stats.run_requests >= max_requests:
        raise RequestBudgetExceeded("request budget reached")


def request_json(
    url: str,
    delay: float,
    max_retries: int,
    stats: Stats,
    max_requests: int | None = None,
) -> dict[str, Any]:
    for attempt in range(max_retries + 1):
        ensure_request_budget(stats, max_requests)
        if attempt:
            stats.retries += 1
        stats.requests += 1
        stats.run_requests += 1
        try:
            request = Request(url, headers={"User-Agent": "social-media-assessment/0.2"})
            with urlopen(request, timeout=60) as response:
                payload = response.read().decode("utf-8")
            if delay:
                time.sleep(delay)
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise MalformedApiResponseError(f"invalid JSON response for {url}") from exc
            if not isinstance(data, dict):
                raise MalformedApiResponseError(f"API response must be an object for {url}")
            if data.get("error"):
                if is_timeout_message(data["error"]):
                    if attempt == 0 and max_retries > 0:
                        wait = 5.0
                        print(f"API query timeout. sleeping {wait:.1f}s", file=sys.stderr)
                        time.sleep(wait)
                        continue
                    raise ApiTimeoutError(f"API timeout for {url}: {data['error']}")
                raise RuntimeError(str(data["error"]))
            return data
        except HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            finally:
                try:
                    exc.close()
                except Exception:
                    pass
            if exc.code == 422 and is_timeout_message(body):
                # A transient server-load timeout can recover without a split;
                # after one retry, let collect_interval reduce the time window.
                if attempt == 0 and max_retries > 0:
                    wait = 5.0
                    print(f"422 query timeout. sleeping {wait:.1f}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                raise ApiTimeoutError(f"HTTP 422 Timeout for {url}") from exc
            if exc.code == 429 and attempt < max_retries:
                retry_after = exc.headers.get("Retry-After")
                reset_seconds = exc.headers.get("X-RateLimit-Reset")
                reset_at = exc.headers.get("X-RateLimit-Reset-At")
                if retry_after:
                    wait = float(retry_after)
                elif reset_seconds:
                    wait = float(reset_seconds)
                elif reset_at:
                    wait = max(0.0, float(reset_at) - time.time())
                else:
                    wait = min(300, 10 * (attempt + 1))
                print(f"429 rate limit. sleeping {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if 500 <= exc.code < 600 and attempt < max_retries:
                wait = min(120, 5 * (attempt + 1))
                print(f"{exc.code} server error. sleeping {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            stats.errors += 1
            raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
        except RequestBudgetExceeded:
            raise
        except ApiTimeoutError:
            raise
        except URLError as exc:
            if attempt < max_retries:
                wait = min(120, 5 * (attempt + 1))
                print(f"{exc}. sleeping {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if is_timeout_message(exc) or is_timeout_message(getattr(exc, "reason", "")):
                raise ApiTimeoutError(f"API timeout for {url}") from exc
            stats.errors += 1
            raise
        except TimeoutError as exc:
            if attempt < max_retries:
                wait = min(120, 5 * (attempt + 1))
                print(f"{exc}. sleeping {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise ApiTimeoutError(f"API timeout for {url}") from exc
        except RuntimeError as exc:
            if attempt < max_retries:
                wait = min(120, 5 * (attempt + 1))
                print(f"{exc}. sleeping {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            stats.errors += 1
            raise
    raise RuntimeError("unreachable retry state")


def build_url(endpoint: str, params: dict[str, Any]) -> str:
    return f"{API_BASE}{endpoint}?{urlencode({key: value for key, value in params.items() if value is not None})}"


def response_rows(response: dict[str, Any]) -> list[Any]:
    if "data" not in response or not isinstance(response["data"], list):
        raise MalformedApiResponseError("malformed API response: data must be a list")
    return list(response["data"])


def build_collection_window(
    collection_base: dict[str, Any],
    endpoint: str,
    request_params: dict[str, Any],
) -> dict[str, Any]:
    return {
        **collection_base,
        "endpoint": endpoint,
        "api_params": dict(request_params),
    }


def normalize_row_ids(row: dict[str, Any], thing: str) -> dict[str, Any]:
    normalized = dict(row)
    normalized["id"] = canonical_id(normalized.get("id"), thing)
    if thing != "comment":
        return normalized

    normalized["link_id"] = canonical_id(normalized.get("link_id"), "post")
    parent_id = str(normalized.get("parent_id") or "").strip()
    if parent_id and not parent_id.startswith(("t1_", "t3_")):
        parent_id = canonical_id(
            parent_id,
            "post" if parent_id == normalized["link_id"].removeprefix("t3_") else "comment",
        )
    normalized["parent_id"] = parent_id
    if parent_id.startswith("t3_"):
        normalized["parent_thing"] = "post"
    elif parent_id.startswith("t1_"):
        normalized["parent_thing"] = "comment"
    return normalized


def has_nonempty_value(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def write_rows(
    *,
    rows: list[Any],
    thing: str,
    out_file: Path,
    seen: set[str],
    collection: dict[str, Any],
    stats: Stats,
    required_fields: tuple[str, ...] = (),
    acquisition_file: Path | None = None,
    route_seen: set[str] | None = None,
) -> int:
    wrappers: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    invalid_rows = 0
    matches = normalize_matches(collection)
    endpoint = str(collection.get("endpoint") or "")
    route_seen = route_seen if route_seen is not None else set()
    for row in rows:
        stats.rows_seen += 1
        if not isinstance(row, dict) or not has_nonempty_value(row.get("id")) or any(
            not has_nonempty_value(row.get(field)) for field in required_fields
        ):
            stats.invalid_rows += 1
            invalid_rows += 1
            continue
        row = normalize_row_ids(row, thing)
        row_id = row["id"]
        for match in matches:
            route_key = acquisition_route_key(thing, row_id, match, endpoint)
            if acquisition_file is not None and route_key not in route_seen:
                route_seen.add(route_key)
                routes.append(
                    {
                        "thing": thing,
                        "record_id": canonical_id(row_id, thing),
                        "subreddit": collection.get("subreddit"),
                        "endpoint": endpoint,
                        "match": match,
                    }
                )
        key = record_key(thing, row_id)
        if key in seen:
            stats.duplicates += 1
            continue
        seen.add(key)
        row_collection = dict(collection)
        row_collection.setdefault("thing", thing)
        wrappers.append({"collection": with_matches(row_collection, matches), "data": row})
        stats.rows_written += 1
        if thing == "post":
            stats.posts_written += 1
        else:
            stats.comments_written += 1
    write_jsonl_lines(out_file, wrappers)
    if acquisition_file is not None:
        write_jsonl_lines(acquisition_file, routes)
    stats.acquisition_matches += len(routes)
    return invalid_rows


def split_window(start_epoch: int, end_epoch: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    window = end_epoch - start_epoch
    if window <= 2:
        return None
    midpoint = max(start_epoch + 1, start_epoch + (window // 2))
    return (start_epoch, midpoint + 1), (midpoint, end_epoch)


def merge_interval_results(left: IntervalResult, right: IntervalResult) -> IntervalResult:
    if left.status == "failed" or right.status == "failed":
        return IntervalResult("failed", left.reason if left.status == "failed" else right.reason)
    if left.status == "partial" or right.status == "partial":
        return IntervalResult("partial", " | ".join(part for part in (left.reason, right.reason) if part))
    return IntervalResult("complete")


def collect_split_children(
    *,
    split: tuple[tuple[int, int], tuple[int, int]],
    endpoint: str,
    base_params: dict[str, Any],
    task: SearchTask,
    limit: int,
    min_window_seconds: int,
    out_file: Path,
    seen: set[str],
    collection_base: dict[str, Any],
    stats: Stats,
    args: argparse.Namespace,
    acquisition_file: Path | None,
    route_seen: set[str],
) -> IntervalResult:
    stats.splits += 1
    left, right = split
    left_result = collect_interval(
        endpoint=endpoint,
        base_params=base_params,
        task=task,
        start_epoch=left[0],
        end_epoch=left[1],
        limit=limit,
        min_window_seconds=min_window_seconds,
        out_file=out_file,
        seen=seen,
        collection_base=collection_base,
        stats=stats,
        args=args,
        acquisition_file=acquisition_file,
        route_seen=route_seen,
    )
    right_result = collect_interval(
        endpoint=endpoint,
        base_params=base_params,
        task=task,
        start_epoch=right[0],
        end_epoch=right[1],
        limit=limit,
        min_window_seconds=min_window_seconds,
        out_file=out_file,
        seen=seen,
        collection_base=collection_base,
        stats=stats,
        args=args,
        acquisition_file=acquisition_file,
        route_seen=route_seen,
    )
    return merge_interval_results(left_result, right_result)


def collect_interval(
    *,
    endpoint: str,
    base_params: dict[str, Any],
    task: SearchTask,
    start_epoch: int,
    end_epoch: int,
    limit: int,
    min_window_seconds: int,
    out_file: Path,
    seen: set[str],
    collection_base: dict[str, Any],
    stats: Stats,
    args: argparse.Namespace,
    acquisition_file: Path | None = None,
    route_seen: set[str] | None = None,
) -> IntervalResult:
    if start_epoch >= end_epoch:
        return IntervalResult("complete")
    ensure_request_budget(stats, args.max_requests)
    route_seen = route_seen if route_seen is not None else set()
    request_params = {
        **base_params,
        # Arctic Shift accepts epoch dates; the one-second guard plus the
        # client-side [event_start, event_end) filter makes split boundaries safe.
        "after": start_epoch - 1,
        "before": end_epoch,
        "sort": "asc",
        "limit": limit,
    }
    try:
        rows = response_rows(
            request_json(
                build_url(endpoint, request_params),
                args.delay,
                args.max_retries,
                stats,
                max_requests=args.max_requests,
            )
        )
    except ApiTimeoutError:
        split = split_window(start_epoch, end_epoch)
        if split is None or end_epoch - start_epoch <= min_window_seconds:
            stats.truncated_windows += 1
            return IntervalResult("partial", f"timeout in minimum window {iso_utc(start_epoch)}..{iso_utc(end_epoch)}")
        return collect_split_children(
            split=split,
            endpoint=endpoint,
            base_params=base_params,
            task=task,
            limit=limit,
            min_window_seconds=min_window_seconds,
            out_file=out_file,
            seen=seen,
            collection_base=collection_base,
            stats=stats,
            args=args,
            acquisition_file=acquisition_file,
            route_seen=route_seen,
        )

    if rows and len(rows) >= limit:
        split = split_window(start_epoch, end_epoch)
        if split is not None and end_epoch - start_epoch > min_window_seconds:
            return collect_split_children(
                split=split,
                endpoint=endpoint,
                base_params=base_params,
                task=task,
                limit=limit,
                min_window_seconds=min_window_seconds,
                out_file=out_file,
                seen=seen,
                collection_base=collection_base,
                stats=stats,
                args=args,
                acquisition_file=acquisition_file,
                route_seen=route_seen,
            )

    fetched_count = len(rows)
    rows, outside, invalid_timestamps = filter_rows_to_window(rows, task.start_epoch, task.end_epoch)
    stats.out_of_window_rows += outside
    stats.invalid_rows += invalid_timestamps
    invalid_rows = write_rows(
        rows=rows,
        thing="post",
        out_file=out_file,
        seen=seen,
        collection=build_collection_window(collection_base, endpoint, request_params),
        stats=stats,
        acquisition_file=acquisition_file,
        route_seen=route_seen,
    )
    if invalid_rows or invalid_timestamps:
        problems = []
        if invalid_rows:
            problems.append(f"{invalid_rows} API rows missing usable ids")
        if invalid_timestamps:
            problems.append(f"{invalid_timestamps} API rows missing usable timestamps")
        return IntervalResult("partial", "; ".join(problems))
    if fetched_count < limit:
        return IntervalResult("complete")
    stats.truncated_windows += 1
    return IntervalResult("partial", f"window saturated at limit={limit} for {iso_utc(start_epoch)}..{iso_utc(end_epoch)}")


def task_collection_base(task: SearchTask, query: str, matches: list[dict[str, str]] | None = None) -> dict[str, Any]:
    event = EVENT_WINDOWS[task.event_id]
    effective_matches = matches or [
        {
            "event_id": task.event_id,
            "jurisdiction": task.jurisdiction,
            "window_start_utc": task.window_start_utc,
            "window_end_utc": task.window_end_utc,
            "query": query,
        }
    ]
    return {
        "platform": PLATFORM,
        "source_corpus": PLATFORM,
        "subreddit": task.subreddit,
        "subreddit_scope": SUBREDDIT_SCOPE,
        "event_id": task.event_id,
        "jurisdiction": event["jurisdiction"],
        "window_start_utc": event["start_utc"],
        "window_end_utc": event["end_utc"],
        "query": query,
        "acquisition_matches": effective_matches,
    }


def collect_search(
    *,
    task: SearchTask,
    out_file: Path,
    seen: set[str],
    stats: Stats,
    args: argparse.Namespace,
    acquisition_file: Path | None = None,
    route_seen: set[str] | None = None,
) -> TaskResult:
    result = TaskResult(
        thing="post_search",
        subreddit=task.subreddit,
        event_id=task.event_id,
        jurisdiction=task.jurisdiction,
        window_start_utc=task.window_start_utc,
        window_end_utc=task.window_end_utc,
        query=task.query,
    )
    before = asdict(stats)
    params = {"subreddit": task.subreddit, "query": task.query, "fields": POST_FIELDS}
    try:
        interval = collect_interval(
            endpoint="/api/posts/search",
            base_params=params,
            task=task,
            start_epoch=task.start_epoch,
            end_epoch=task.end_epoch,
            limit=args.limit,
            min_window_seconds=args.min_window_seconds,
            out_file=out_file,
            seen=seen,
            collection_base=task_collection_base(task, task.query),
            stats=stats,
            args=args,
            acquisition_file=acquisition_file,
            route_seen=route_seen,
        )
        result.status = interval.status
        result.error = interval.reason
    except RequestBudgetExceeded:
        result.status = "partial"
        result.error = "request budget reached"
    except MalformedApiResponseError as exc:
        result.status = "partial"
        result.error = str(exc)
    except Exception as exc:
        result.status = "failed"
        result.error = str(exc)
    result.rows_written = stats.rows_written - before["rows_written"]
    result.duplicates = stats.duplicates - before["duplicates"]
    result.requests_used = stats.requests - before["requests"]
    result.splits_used = stats.splits - before["splits"]
    result.truncated_windows = stats.truncated_windows - before["truncated_windows"]
    result.rows_filtered_outside_window = stats.out_of_window_rows - before["out_of_window_rows"]
    return result


def flatten_comment_tree(nodes: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    comments: list[dict[str, Any]] = []
    collapsed: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []

    def unknown_node(kind: Any, reason: str) -> None:
        unknown.append({"kind": kind, "reason": reason})

    def walk(items: Any) -> None:
        if not isinstance(items, list):
            unknown_node(None, "expected list of tree nodes")
            return
        for node in items:
            if not isinstance(node, dict):
                unknown_node(None, "tree node is not an object")
                continue
            kind = node.get("kind")
            data = node.get("data")
            if kind == "t1" and isinstance(data, dict):
                comments.append({field_name: data[field_name] for field_name in COMMENT_FIELD_NAMES if field_name in data})
                replies = data.get("replies")
                if isinstance(replies, dict):
                    walk([replies])
                elif isinstance(replies, list):
                    walk(replies)
                elif replies not in (None, ""):
                    unknown_node("t1", "comment replies has an unsupported shape")
            elif kind == "t1":
                unknown_node(kind, "comment node data is not an object")
            elif kind == "more" and isinstance(data, dict):
                collapsed.append(data)
            elif kind == "more":
                unknown_node(kind, "collapsed node data is not an object")
            elif kind == "Listing" and isinstance(data, dict):
                walk(data.get("children"))
            elif kind == "Listing":
                unknown_node(kind, "listing node data is not an object")
            else:
                unknown_node(kind, "unsupported tree node kind")

    walk(nodes)
    return comments, collapsed, unknown


def filter_rows_to_window(
    rows: list[dict[str, Any]], start_epoch: int, end_epoch: int
) -> tuple[list[dict[str, Any]], int, int]:
    included: list[dict[str, Any]] = []
    outside = 0
    invalid = 0
    for row in rows:
        try:
            created = float(row.get("created_utc"))
        except (TypeError, ValueError):
            invalid += 1
            continue
        if not math.isfinite(created):
            invalid += 1
        elif start_epoch <= created < end_epoch:
            included.append(row)
        else:
            outside += 1
    return included, outside, invalid


def filter_comment_rows_to_window(
    rows: list[dict[str, Any]], start_epoch: int, end_epoch: int
) -> tuple[list[dict[str, Any]], int, int]:
    return filter_rows_to_window(rows, start_epoch, end_epoch)


def collect_comment_tree(
    *,
    target: dict[str, Any],
    out_file: Path,
    seen: set[str],
    stats: Stats,
    args: argparse.Namespace,
    acquisition_file: Path | None = None,
    route_seen: set[str] | None = None,
) -> TaskResult:
    queries = sorted(set(target.get("queries") or []))
    task = TaskResult(
        thing="comment_tree",
        subreddit=str(target["subreddit"]),
        event_id=str(target["event_id"]),
        jurisdiction=str(target["jurisdiction"]),
        window_start_utc=str(target["window_start_utc"]),
        window_end_utc=str(target["window_end_utc"]),
        post_id=str(target["post_id"]),
        matched_queries=queries,
    )
    before = asdict(stats)
    endpoint = "/api/comments/tree"
    matches = [
        {
            "event_id": target["event_id"],
            "jurisdiction": target["jurisdiction"],
            "window_start_utc": target["window_start_utc"],
            "window_end_utc": target["window_end_utc"],
            "query": query,
        }
        for query in queries
    ]
    collection_base = task_collection_base(
        SearchTask(
            subreddit=target["subreddit"],
            event_id=target["event_id"],
            jurisdiction=target["jurisdiction"],
            window_start_utc=target["window_start_utc"],
            window_end_utc=target["window_end_utc"],
            query=queries[0] if queries else "",
        ),
        queries[0] if queries else "",
        matches=matches,
    )
    collection_base["post_id"] = target["post_id"]
    collection_base["matched_queries"] = queries
    try:
        ensure_request_budget(stats, args.max_requests)
        params = {
            "link_id": target["post_id"],
            "limit": args.comment_tree_limit,
            "start_breadth": args.comment_tree_limit,
            "start_depth": args.comment_tree_limit,
        }
        response = request_json(
            build_url(endpoint, params),
            args.delay,
            args.max_retries,
            stats,
            max_requests=args.max_requests,
        )
        rows, collapsed, unknown = flatten_comment_tree(response_rows(response))
        rows, outside, invalid_timestamps = filter_comment_rows_to_window(
            rows, parse_utc(target["window_start_utc"]), parse_utc(target["window_end_utc"])
        )
        stats.out_of_window_rows += outside
        stats.invalid_rows += invalid_timestamps
        stats.unknown_tree_nodes += len(unknown)
        stats.collapsed_tree_nodes += len(collapsed)
        stats.errors += len(unknown)
        invalid_rows = write_rows(
            rows=rows,
            thing="comment",
            out_file=out_file,
            seen=seen,
            required_fields=("link_id", "parent_id"),
            collection={
                **collection_base,
                "endpoint": endpoint,
                "api_params": params,
                "collection_scope": "comment_created_utc_in_event_window",
                "thread_snapshot": True,
            },
            stats=stats,
            acquisition_file=acquisition_file,
            route_seen=route_seen,
        )
        issues: list[str] = []
        if invalid_rows:
            issues.append(f"{invalid_rows} tree rows missing usable ids")
        if invalid_timestamps:
            issues.append(f"{invalid_timestamps} tree rows missing usable timestamps")
        if unknown:
            issues.append(f"{len(unknown)} unknown tree node(s)")
        if collapsed:
            stats.truncated_windows += 1
            issues.append(f"comment tree returned {len(collapsed)} collapsed more node(s)")
        if issues:
            task.status = "partial"
            task.error = "; ".join(issues)
        else:
            task.status = "complete"
    except RequestBudgetExceeded:
        task.status = "partial"
        task.error = "request budget reached"
    except ApiTimeoutError as exc:
        task.status = "partial"
        task.error = str(exc)
    except MalformedApiResponseError as exc:
        task.status = "partial"
        task.error = str(exc)
    except Exception as exc:
        task.status = "failed"
        task.error = str(exc)
    task.rows_written = stats.rows_written - before["rows_written"]
    task.duplicates = stats.duplicates - before["duplicates"]
    task.requests_used = stats.requests - before["requests"]
    task.splits_used = stats.splits - before["splits"]
    task.truncated_windows = stats.truncated_windows - before["truncated_windows"]
    task.rows_filtered_outside_window = stats.out_of_window_rows - before["out_of_window_rows"]
    task.unknown_tree_nodes = stats.unknown_tree_nodes - before["unknown_tree_nodes"]
    task.collapsed_tree_nodes = stats.collapsed_tree_nodes - before["collapsed_tree_nodes"]
    return task


def _add_post_target(targets: dict[tuple[str, str, str], dict[str, Any]], post_id: Any, subreddit: Any, match: Any) -> None:
    if not isinstance(match, dict):
        return
    event_id = str(match.get("event_id") or "")
    query = str(match.get("query") or "")
    if event_id not in EVENT_WINDOWS or query not in AGE_ASSURANCE_QUERIES:
        return
    event = EVENT_WINDOWS[event_id]
    subreddit_text = str(subreddit or "")
    if subreddit_text not in TARGET_SUBREDDITS or not post_id:
        return
    normalized_id = canonical_id(post_id, "post")
    key = (event_id, subreddit_text, normalized_id)
    target = targets.setdefault(
        key,
        {
            "post_id": normalized_id,
            "subreddit": subreddit_text,
            "event_id": event_id,
            "jurisdiction": event["jurisdiction"],
            "window_start_utc": event["start_utc"],
            "window_end_utc": event["end_utc"],
            "queries": set(),
        },
    )
    target["queries"].add(query)


def load_matched_post_targets(path: Path, acquisition_path: Path | None = None) -> list[dict[str, Any]]:
    targets: dict[tuple[str, str, str], dict[str, Any]] = {}

    def consume_wrapper(wrapper: dict[str, Any]) -> None:
        collection = wrapper.get("collection")
        data = wrapper.get("data")
        if not isinstance(collection, dict) or not isinstance(data, dict) or not data.get("id"):
            return
        for match in normalize_matches(collection):
            _add_post_target(targets, data["id"], collection.get("subreddit"), match)

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    consume_wrapper(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"invalid JSON in {path} line {line_number}") from exc
    if acquisition_path is not None and acquisition_path.exists():
        with acquisition_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    route = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"invalid JSON in {acquisition_path} line {line_number}") from exc
                if route.get("thing") == "post":
                    _add_post_target(targets, route.get("record_id"), route.get("subreddit"), route.get("match"))
    result = []
    for key in sorted(targets):
        target = dict(targets[key])
        target["queries"] = sorted(target["queries"])
        result.append(target)
    return result


def _load_route_matches(path: Path) -> dict[str, list[dict[str, str]]]:
    routes: dict[str, dict[str, dict[str, str]]] = {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            route = json.loads(line)
            key = record_key(str(route.get("thing") or ""), route.get("record_id"))
            match = route.get("match")
            if key and isinstance(match, dict):
                normalized = normalize_matches({"acquisition_matches": [match]})
                if normalized:
                    routes.setdefault(key, {})[match_key(normalized[0])] = normalized[0]
    return {key: list(values.values()) for key, values in routes.items()}


def consolidate_acquisition_matches(raw_path: Path, acquisition_path: Path) -> None:
    route_matches = _load_route_matches(acquisition_path)
    if not raw_path.exists():
        raw_path.touch()
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=raw_path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            with raw_path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    wrapper = json.loads(line)
                    data = wrapper.get("data")
                    collection = wrapper.get("collection")
                    if isinstance(data, dict) and isinstance(collection, dict) and data.get("id"):
                        thing = str(collection.get("thing") or "")
                        if thing not in {"post", "comment"}:
                            thing = "comment" if "body" in data and "title" not in data else "post"
                        key = record_key(thing, data["id"])
                        matches = normalize_matches(collection)
                        all_matches = {match_key(match): match for match in matches}
                        for match in route_matches.get(key, []):
                            all_matches[match_key(match)] = match
                        wrapper["collection"] = with_matches(
                            collection, [all_matches[key] for key in sorted(all_matches)]
                        )
                    handle.write(json.dumps(wrapper, ensure_ascii=False, sort_keys=True))
                    handle.write("\n")
        temporary.replace(raw_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def compute_completion_status(
    tasks: list[TaskResult | dict[str, Any]], expected_count: int | None = None
) -> str:
    statuses = [task.get("status") if isinstance(task, dict) else task.status for task in tasks]
    if "failed" in statuses:
        return "failed"
    if expected_count is not None and len(tasks) < expected_count:
        return "partial"
    if not tasks or any(status != "complete" for status in statuses):
        return "partial"
    return "complete"


def expected_task_count(static_count: int, matched_post_count: int) -> int:
    return static_count + matched_post_count


def summarize_task_counts(tasks: list[TaskResult | dict[str, Any]], expected_count: int) -> dict[str, int]:
    counts = {
        "expected": expected_count,
        "recorded": len(tasks),
        "complete": 0,
        "partial": 0,
        "failed": 0,
        "other": 0,
    }
    for task in tasks:
        status = task.get("status") if isinstance(task, dict) else task.status
        if status in {"complete", "partial", "failed"}:
            counts[status] += 1
        else:
            counts["other"] += 1
    counts["missing"] = max(expected_count - counts["recorded"], 0)
    counts["incomplete"] = counts["recorded"] - counts["complete"] + counts["missing"]
    return counts


def task_identity(task: dict[str, Any]) -> tuple[Any, ...]:
    return (
        task.get("thing"),
        task.get("subreddit"),
        task.get("event_id"),
        task.get("window_start_utc"),
        task.get("window_end_utc"),
        task.get("query"),
        task.get("post_id"),
    )


def refresh_completed_comment_tree_task(previous_tasks: Any, target: dict[str, Any]) -> None:
    if not isinstance(previous_tasks, list):
        return
    identity = task_identity({"thing": "comment_tree", **target})
    matched_queries = sorted(set(target.get("queries") or []))
    for task in previous_tasks:
        if (
            isinstance(task, dict)
            and task.get("status") == "complete"
            and task_identity(task) == identity
        ):
            task["matched_queries"] = matched_queries
            return


ATTEMPT_FIELDS = (
    "status",
    "error",
    "rows_written",
    "duplicates",
    "requests_used",
    "splits_used",
    "truncated_windows",
    "rows_filtered_outside_window",
    "unknown_tree_nodes",
    "collapsed_tree_nodes",
)


def attempt_snapshot(task: dict[str, Any]) -> dict[str, Any]:
    return {field_name: task.get(field_name, "" if field_name in {"status", "error"} else 0) for field_name in ATTEMPT_FIELDS}


def existing_attempt_history(task: dict[str, Any]) -> list[dict[str, Any]]:
    history = task.get("attempt_history")
    if isinstance(history, list) and history:
        return [item for item in history if isinstance(item, dict)]
    return [{"attempt_count": int(task.get("attempts", 1) or 1), **attempt_snapshot(task)}]


def merge_task_history(previous: Any, current: list[TaskResult]) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    if isinstance(previous, list):
        for task in previous:
            if isinstance(task, dict):
                item = dict(task)
                item["attempts"] = int(item.get("attempts", 1) or 1)
                item["attempt_history"] = existing_attempt_history(item)
                merged[task_identity(item)] = item
    for task in current:
        item = asdict(task)
        key = task_identity(item)
        old = merged.get(key)
        if old is None:
            item["attempts"] = 1
            item["attempt_history"] = [attempt_snapshot(item)]
            merged[key] = item
            continue
        current_attempt = attempt_snapshot(item)
        for field_name in (
            "rows_written",
            "duplicates",
            "requests_used",
            "splits_used",
            "truncated_windows",
            "rows_filtered_outside_window",
            "unknown_tree_nodes",
            "collapsed_tree_nodes",
        ):
            item[field_name] = int(old.get(field_name, 0) or 0) + int(item.get(field_name, 0) or 0)
        item["attempts"] = int(old.get("attempts", 1) or 1) + 1
        item["attempt_history"] = existing_attempt_history(old) + [current_attempt]
        merged[key] = item
    return list(merged.values())


def write_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    stats: Stats,
    tasks: list[TaskResult],
    expected_count: int,
    matched_post_count: int,
    static_task_count: int,
    previous_manifest: dict[str, Any] | None = None,
    run_started_at_utc: str | None = None,
) -> str:
    finished_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    merged_tasks = merge_task_history(previous_manifest.get("tasks") if previous_manifest else None, tasks)
    completion_status = compute_completion_status(merged_tasks, expected_count)
    task_counts = summarize_task_counts(merged_tasks, expected_count)
    manifest = {
        "manifest_version": 5,
        "created_at_utc": finished_at_utc,
        "run_started_at_utc": run_started_at_utc or finished_at_utc,
        "run_finished_at_utc": finished_at_utc,
        "study_name": STUDY_NAME,
        "study_scope": STUDY_SCOPE,
        "platform": PLATFORM,
        "api_base": API_BASE,
        "api_docs_url": API_DOCS_URL,
        "start_utc": START_UTC,
        "end_utc": END_UTC,
        "target_subreddits": TARGET_SUBREDDITS,
        "subreddit_scope": SUBREDDIT_SCOPE,
        "age_assurance_queries": AGE_ASSURANCE_QUERIES,
        "relevance_screening": RELEVANCE_SCREENING,
        "event_windows": event_window_records(),
        "api_contract": {
            "post_search_endpoint": "/api/posts/search",
            "comment_tree_endpoint": "/api/comments/tree",
            "post_search_fields": POST_FIELDS,
            "comment_tree_retained_fields": COMMENT_FIELDS,
            "post_search_arguments": ["subreddit", "query", "after", "before", "sort", "limit", "fields"],
            "post_search_date_scope": "request after is one second before the interval start; retained rows use [event_start, event_end)",
            "comment_tree_arguments": ["link_id", "limit", "start_breadth", "start_depth"],
            "comment_tree_date_scope": "client-side created_utc filter because the tree endpoint has no after/before parameters",
        },
        "args": vars(args),
        "request_budget_scope": "per_run_invocation",
        "static_task_count": static_task_count,
        "matched_post_target_count": matched_post_count,
        "expected_task_count": expected_count,
        "task_counts": task_counts,
        "completion_status": completion_status,
        "stats": asdict(stats),
        "counts": {
            "matched_posts": matched_post_count,
            "posts_written": stats.posts_written,
            "comments_written": stats.comments_written,
            "duplicates": stats.duplicates,
            "acquisition_matches": stats.acquisition_matches,
            "rows_out_of_window": stats.out_of_window_rows,
            "unknown_tree_nodes": stats.unknown_tree_nodes,
            "collapsed_tree_nodes": stats.collapsed_tree_nodes,
        },
        "files": {
            "posts": "posts_raw.jsonl",
            "comments": "comments_raw.jsonl",
            "acquisition_routes": "acquisition_routes.jsonl",
        },
        "tasks": merged_tasks,
        "resume": {
            "attempt": (int((previous_manifest or {}).get("resume", {}).get("attempt", 0) or 0) + 1),
            "previous_manifest_created_at_utc": (previous_manifest or {}).get("created_at_utc", ""),
        },
    }
    (out_dir / "collection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return completion_status


def build_plan_summary(args: argparse.Namespace | None = None) -> tuple[list[SearchTask], int]:
    tasks: list[SearchTask] = []
    for event_id, event in EVENT_WINDOWS.items():
        for subreddit in TARGET_SUBREDDITS:
            for query in AGE_ASSURANCE_QUERIES:
                tasks.append(
                    SearchTask(
                        subreddit=subreddit,
                        event_id=event_id,
                        jurisdiction=event["jurisdiction"],
                        window_start_utc=event["start_utc"],
                        window_end_utc=event["end_utc"],
                        query=query,
                    )
                )
    return tasks, len(tasks)


def execute_planned_tasks(
    *,
    planned_tasks: list[SearchTask],
    tasks: list[TaskResult],
    posts_file: Path,
    comments_file: Path,
    acquisition_file: Path,
    posts_seen: set[str],
    comments_seen: set[str],
    routes_seen: set[str],
    stats: Stats,
    args: argparse.Namespace,
    completed_task_ids: set[tuple[Any, ...]],
) -> int:
    exit_code = 0
    for planned in planned_tasks:
        if task_identity({"thing": "post_search", **asdict(planned)}) in completed_task_ids:
            continue
        try:
            task = collect_search(
                task=planned,
                out_file=posts_file,
                seen=posts_seen,
                stats=stats,
                args=args,
                acquisition_file=acquisition_file,
                route_seen=routes_seen,
            )
        except Exception as exc:
            task = TaskResult(
                thing="post_search",
                subreddit=planned.subreddit,
                event_id=planned.event_id,
                jurisdiction=planned.jurisdiction,
                window_start_utc=planned.window_start_utc,
                window_end_utc=planned.window_end_utc,
                query=planned.query,
                status="failed",
                error=str(exc),
            )
        tasks.append(task)
        if task.error:
            print(f"{task.status}: post search {planned.event_id} r/{planned.subreddit} {planned.query!r}: {task.error}", file=sys.stderr)
        if task.error == "request budget reached":
            return 1

    targets = load_matched_post_targets(posts_file, acquisition_file)
    completed_targets = [
        target
        for target in targets
        if task_identity({"thing": "comment_tree", **target}) in completed_task_ids
    ]
    refresh_comment_tree_provenance(
        comments_file=comments_file,
        acquisition_file=acquisition_file,
        targets=completed_targets,
        routes_seen=routes_seen,
        stats=stats,
    )

    for target in targets:
        if task_identity({"thing": "comment_tree", **target}) in completed_task_ids:
            continue
        task = collect_comment_tree(
            target=target,
            out_file=comments_file,
            seen=comments_seen,
            stats=stats,
            args=args,
            acquisition_file=acquisition_file,
            route_seen=routes_seen,
        )
        tasks.append(task)
        if task.error:
            print(f"{task.status}: comment tree {target['event_id']} {target['post_id']}: {task.error}", file=sys.stderr)
        if task.error == "request budget reached":
            return 1
    return exit_code


def run_collection(args: argparse.Namespace) -> tuple[Stats, list[TaskResult], Path, str, int]:
    run_started_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_root = Path(args.out_root)
    out_dir = out_root if args.resume else make_output_dir(out_root, args.overwrite)
    posts_file = out_dir / "posts_raw.jsonl"
    comments_file = out_dir / "comments_raw.jsonl"
    acquisition_file = out_dir / "acquisition_routes.jsonl"
    previous_manifest = validate_resume_manifest(out_dir, args) if args.resume else None
    if not args.resume:
        out_dir.mkdir(parents=True, exist_ok=True)
    ensure_output_files_safe([posts_file, comments_file, acquisition_file], resume=args.resume)
    if not args.resume:
        posts_file.touch(exist_ok=True)
        comments_file.touch(exist_ok=True)
        acquisition_file.touch(exist_ok=True)
    stats = Stats.from_manifest(previous_manifest.get("stats") if previous_manifest else None)
    posts_seen = load_seen(posts_file, "post") if args.resume else set()
    comments_seen = load_seen(comments_file, "comment") if args.resume else set()
    routes_seen = load_seen_routes(acquisition_file) if args.resume else set()
    if args.resume:
        reconcile_persisted_stats(stats, posts_seen, comments_seen, routes_seen)
    planned_tasks, static_count = build_plan_summary(args)
    previous_tasks = previous_manifest.get("tasks") if previous_manifest else None
    completed_task_ids = {
        task_identity(task)
        for task in previous_tasks or []
        if isinstance(task, dict) and task.get("status") == "complete"
    }
    tasks: list[TaskResult] = []
    exit_code = 0
    completion_status = "partial"
    try:
        exit_code = execute_planned_tasks(
            planned_tasks=planned_tasks,
            tasks=tasks,
            posts_file=posts_file,
            comments_file=comments_file,
            acquisition_file=acquisition_file,
            posts_seen=posts_seen,
            comments_seen=comments_seen,
            routes_seen=routes_seen,
            stats=stats,
            args=args,
            completed_task_ids=completed_task_ids,
        )
    finally:
        consolidate_acquisition_matches(posts_file, acquisition_file)
        consolidate_acquisition_matches(comments_file, acquisition_file)
        matched_post_targets = load_matched_post_targets(posts_file, acquisition_file)
        for target in matched_post_targets:
            refresh_completed_comment_tree_task(previous_tasks, target)
        matched_post_count = len(matched_post_targets)
        expected_count = expected_task_count(static_count, matched_post_count)
        completion_status = write_manifest(
            out_dir,
            args,
            stats,
            tasks,
            expected_count=expected_count,
            matched_post_count=matched_post_count,
            static_task_count=static_count,
            previous_manifest=previous_manifest,
            run_started_at_utc=run_started_at_utc,
        )
    if completion_status != "complete":
        exit_code = 1
    return stats, tasks, out_dir, completion_status, exit_code


def run_self_test() -> None:
    assert iso_utc(parse_utc("2024-11-10T00:00:00Z")) == "2024-11-10T00:00:00Z"
    assert TARGET_SUBREDDITS == ["privacy", "technology", "parenting", "unitedkingdom", "europe", "australia"]
    assert len(build_plan_summary()[0]) == 270
    comments, collapsed, unknown = flatten_comment_tree(
        [{"kind": "t1", "data": {"id": "c", "replies": ""}}]
    )
    assert comments == [{"id": "c"}]
    assert collapsed == []
    assert unknown == []
