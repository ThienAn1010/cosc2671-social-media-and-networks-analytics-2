"""Small, shared helpers for versioned derived analysis artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = REPO_ROOT / "data" / "analysis"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON at {path}")
    return value


def source_record(path: Path, role: str, rows: int | None = None, columns: list[str] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {"path": relative_path(path), "role": role, "exists": False, "rows": rows, "columns": columns or [], "sha256": None}
    return {
        "path": relative_path(path),
        "role": role,
        "exists": True,
        "bytes": path.stat().st_size,
        "rows": rows,
        "columns": columns or [],
        "sha256": sha256_file(path),
    }
