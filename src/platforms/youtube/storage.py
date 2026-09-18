# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

"""Run folders, JSONL wrappers and manifests shared by the YouTube discovery and collection CLIs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPO_ROOT / "data" / "raw" / "youtube"
MANIFEST_NAME = "collection_manifest.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# One folder per run, named youtube_<UTC timestamp>, so a new run never overwrites an old one.
def new_run_dir(root: Path = RAW_ROOT) -> Path:
    run_dir = root / f"youtube_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# Raw provenance wrapper shared with the Reddit pipeline: {"collection": ..., "data": untouched API item}.
def wrap(item: dict[str, Any], collection: dict[str, Any]) -> dict[str, Any]:
    return {"collection": collection, "data": item}


# Append-only: raw files are never rewritten (RULES R-C6).
def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return len(rows)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


# IDs already on disk, so a resumed run skips them instead of writing duplicates.
def load_ids(path: Path, get_id: Callable[[dict[str, Any]], str | None] = lambda row: row["data"].get("id")) -> set[str]:
    return {item_id for row in read_jsonl(path) if (item_id := get_id(row))}


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / MANIFEST_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# Write to a temp file then rename, so an interrupted run never leaves a half-written manifest.
def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    path = run_dir / MANIFEST_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()
