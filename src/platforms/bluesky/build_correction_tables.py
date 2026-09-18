# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

r"""Build the small correction tables the 2026-09-14 data-quality audit calls for.

    python -m src.platforms.bluesky.build_correction_tables

Writes two CSVs into `data/interim/`. Read-only with respect to `data/raw/`.

Why these are tables rather than a paragraph in a document
----------------------------------------------------------
Both encode a distinction that is invisible in the raw payload and that any
downstream analysis will otherwise get wrong by default:

* `follows_seed_status.csv` — a seed that returned no follow edges is NOT
  automatically an isolate. Cross-checking the follow layer against the profile
  layer separates 230 accounts that genuinely follow nobody (their own profile
  reports `follows_count = 0`, and they are mostly news bots and bridged RSS
  accounts) from 9 whose account had gone by the time the follow pass ran. Only
  the second group is missing data. A further 804 seeds hit the 2,000-follow
  cap, so their out-degree is censored rather than measured.

  `out_degree` is left BLANK where it is not a measurement. That is the whole
  point of the file: reading a censored or unavailable seed as `0` would put
  813 false zeros into any influence or reciprocity statistic.

* `baseline_saturated_cells.csv` — the 4 (term, day) cells where `hits_total`
  hit the API's 10,000 ceiling and therefore understates the true denominator.
  Mask these before normalising any volume series by the baseline basket.

Both are regenerable from `data/raw/`, which is why they live in `data/interim/`
and are git-ignored. This script is what regenerates them.
"""

import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from src.platforms.bluesky.config import load_config

log = logging.getLogger("corrections")

CAP_SENTINEL = 10_000  # baseline hits_total ceiling


def load_profile_facts(profiles_path: Path) -> tuple[dict, dict, set]:
    """Return (did -> handle, did -> follows_count, {dids tombstoned})."""
    handle, follows_count, gone = {}, {}, set()
    with profiles_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            profile = record.get("profile") or {}
            did = profile.get("did")
            if not did:
                continue
            handle[did] = profile.get("handle")
            follows_count[did] = profile.get("follows_count")
            if (record.get("_meta") or {}).get("gone"):
                gone.add(did)
    return handle, follows_count, gone


def classify_seed(meta: dict, profile_follows: int | None, is_tombstoned: bool) -> tuple[str, object]:
    """Return (status, out_degree) for one seed. Blank out_degree means NA."""
    collected = meta.get("n_collected") or 0
    reported = meta.get("follows_count_reported")

    if meta.get("truncated"):
        # Censored at max_follows_per_user: we know out-degree is AT LEAST this.
        return "capped", ""
    if collected == 0:
        claimed = profile_follows if profile_follows is not None else reported
        if is_tombstoned or (claimed or 0) > 0:
            # The account says it follows people but served none -- it was gone
            # by the time this pass ran. Missing data, not an isolate.
            return "unavailable", ""
        # Profile independently agrees the account follows nobody.
        return "genuine_zero", 0
    return "complete", collected


def build_seed_status(cfg, handle, profile_follows, gone) -> Path:
    rows = []
    counts = Counter()
    with (cfg.follows_dir / "follows.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            meta = json.loads(line)["_meta"]
            did = meta["seed_did"]
            status, out_degree = classify_seed(
                meta, profile_follows.get(did), did in gone)
            counts[status] += 1
            rows.append({
                "seed_did": did,
                "seed_handle": meta.get("seed_handle") or handle.get(did, ""),
                "n_collected": meta.get("n_collected") or 0,
                "follows_count_reported": meta.get("follows_count_reported")
                if meta.get("follows_count_reported") is not None else "",
                "profile_follows_count": profile_follows.get(did)
                if profile_follows.get(did) is not None else "",
                "truncated": int(bool(meta.get("truncated"))),
                "status": status,
                "out_degree": out_degree,
            })

    out = cfg.interim_dir / "follows_seed_status.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    usable = sum(1 for r in rows if r["out_degree"] != "")
    log.info("%s: %s seeds -> %s", out.name, f"{len(rows):,}", dict(counts))
    log.info("  out-degree is a real measurement for %s of %s seeds; "
             "NA for the other %s", usable, len(rows), len(rows) - usable)
    return out


def build_saturated_cells(cfg) -> Path:
    rows = []
    with cfg.baseline_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("saturated") or (record.get("hits_total") or 0) >= CAP_SENTINEL:
                rows.append({"term": record["term"], "day": record["day"],
                             "hits_total": record["hits_total"]})

    out = cfg.interim_dir / "baseline_saturated_cells.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["term", "day", "hits_total"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["term"], r["day"])))
    log.info("%s: %s saturated cell(s) -- mask before using the denominator",
             out.name, len(rows))
    for r in sorted(rows, key=lambda r: (r["term"], r["day"])):
        log.info("    %-10s %s", r["term"], r["day"])
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config()
    cfg.ensure_dirs()

    handle, profile_follows, gone = load_profile_facts(cfg.profiles_dir / "profiles.jsonl")
    log.info("Loaded %s profiles (%s tombstoned).", f"{len(handle):,}", len(gone))

    build_seed_status(cfg, handle, profile_follows, gone)
    build_saturated_cells(cfg)
    log.info("Done. Both tables are derived from data/raw/ and safe to regenerate.")


if __name__ == "__main__":
    main()
