# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

r"""Write a verifiable manifest of everything this workstream produced.

    python -m src.platforms.bluesky.build_manifest

Writes `data/manifest.json`.

Why
---
The team shares code through a repository and data through a drive, which means
the two travel separately and can silently drift apart. Without a manifest there
is no way to answer the questions that actually matter when three people pool
data: is my copy the same as yours, was it built from this version of the code,
and how many rows should I expect after copying 2.5 GB over a sync client that
occasionally drops files.

So the manifest records, for every layer: file count, row count, byte size, the
date range covered, and a checksum. For multi-file layers the checksum is rolled
up -- sha256 over the sorted `name:sha256` lines of the member files -- so one
value verifies all 608 day files without listing 608 hashes.

It also hashes `config/bluesky/config.yaml` and every script in `src/`, because a dataset is
not reproducible if you cannot tell which version of the code produced it, and
this project has no git history to lean on.
"""

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.platforms.bluesky.config import PROJECT_ROOT, load_config

log = logging.getLogger("manifest")
CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            n += chunk.count(b"\n")
    return n


def roll_up(entries: list[tuple[str, str]]) -> str:
    """One hash standing for a whole set of files."""
    h = hashlib.sha256()
    for name, digest in sorted(entries):
        h.update(f"{name}:{digest}\n".encode("utf-8"))
    return h.hexdigest()


def describe_group(paths: list[Path], count_rows: bool = True) -> dict:
    entries, total_bytes, rows = [], 0, 0
    for p in sorted(paths):
        entries.append((p.name, sha256_file(p)))
        total_bytes += p.stat().st_size
        if count_rows:
            rows += count_lines(p)
    out = {
        "files": len(entries),
        "bytes": total_bytes,
        "sha256_rollup": roll_up(entries),
    }
    if count_rows:
        out["rows"] = rows
    if len(entries) == 1:
        out["sha256"] = entries[0][1]
    return out


def day_range(paths: list[Path], prefix: str) -> dict:
    days = sorted(p.name[len(prefix) + 1:-6] for p in paths)
    return {"first_day": days[0], "last_day": days[-1], "days": len(days)} if days else {}


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config()
    root = PROJECT_ROOT

    log.info("Hashing raw layers (this reads ~2.5 GB)...")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": "bluesky",
        "workstream": "Bluesky collection and cleaning",
        "window": {"date_start": str(cfg.date_start), "date_end": str(cfg.date_end)},
        "raw": {},
        "interim": {},
        "code": {},
    }

    posts = sorted(cfg.raw_dir.glob("posts_*.jsonl"))
    threads = sorted(cfg.raw_dir.glob("thread_*.jsonl"))
    manifest["raw"]["posts"] = {**describe_group(posts), **day_range(posts, "posts")}
    manifest["raw"]["threads"] = {**describe_group(threads), **day_range(threads, "thread")}
    for name, path in (("profiles", cfg.profiles_dir / "profiles.jsonl"),
                       ("follows", cfg.follows_dir / "follows.jsonl"),
                       ("baseline", cfg.baseline_path)):
        if path.exists():
            manifest["raw"][name] = describe_group([path])

    log.info("Hashing interim tables...")
    for p in sorted(cfg.interim_dir.glob("*")):
        if p.is_file():
            entry = describe_group([p], count_rows=False)
            if p.suffix == ".parquet":
                import pyarrow.parquet as pq
                entry["rows"] = pq.ParquetFile(p).metadata.num_rows
            elif p.suffix == ".csv":
                entry["rows"] = max(count_lines(p) - 1, 0)
            manifest["interim"][p.name] = entry

    log.info("Hashing code...")
    code = [root / "config/bluesky/config.yaml"] + sorted(root.glob("src/**/*.py"))
    manifest["code"] = {
        str(p.relative_to(root)).replace("\\", "/"): sha256_file(p)
        for p in code if p.is_file()
    }

    out = root / "data" / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log.info("-" * 60)
    for layer, info in manifest["raw"].items():
        log.info("  raw/%-10s %4s file(s)  %10s rows  %7.1f MB",
                 layer, info["files"], f"{info.get('rows', 0):,}", info["bytes"] / 1e6)
    for name, info in manifest["interim"].items():
        log.info("  interim/%-26s %10s rows  %7.1f MB",
                 name, f"{info.get('rows', 0):,}", info["bytes"] / 1e6)
    log.info("  %s code file(s) hashed", len(manifest["code"]))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
