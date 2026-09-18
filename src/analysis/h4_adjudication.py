"""Compare two coders' YouTube video labels and write the adjudicated metadata audit.

Coder files are CSVs with columns n, video_id, source_type, frame, stance (frame is
pipe-separated, or "none"), as exported by the Video Framing Bench page.

Usage (from the SMFR root):
    # 1. agreement report and the list of videos to settle
    python -m src.analysis.h4_adjudication compare --coder-a <a.csv> --coder-b <b.csv>
    # 2. after settling, write data/analysis/annotation/youtube_video_metadata.csv
    python -m src.analysis.h4_adjudication write --coder-a <a.csv> --coder-b <b.csv> --decisions <decisions.json>

decisions.json maps video_id to the settled fields, e.g.
    {"abc123": {"source_type": "news", "frame": "policy_assurance|child_safety", "stance": "oppose"}}
Only disagreeing fields need a decision; agreed fields are copied through.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.analysis.artifacts import REPO_ROOT
from src.analysis.networks import YOUTUBE_AUDIT_COLUMNS, YOUTUBE_METADATA_AUDIT, validate_youtube_metadata_audit

FRAME_LABELS = [
    "policy_assurance",
    "child_safety",
    "privacy_surveillance",
    "governance_platform_responsibility",
    "circumvention_censorship_autonomy",
]
FIELDS = ("source_type", "frame", "stance")
REPORT = REPO_ROOT / "data" / "analysis" / "annotation" / "h4_agreement.json"


def _canonical_frames(value: object) -> str:
    labels = {part.strip() for part in str(value or "").split("|") if part.strip()}
    if not labels or labels <= {"none", "none_unclear"}:
        return "none"
    unknown = labels - set(FRAME_LABELS)
    if unknown:
        raise ValueError(f"unknown frame labels: {sorted(unknown)}")
    return "|".join(label for label in FRAME_LABELS if label in labels)


def load_coder(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    missing = sorted({"video_id", *FIELDS} - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame = frame[["video_id", *FIELDS]].copy()
    if frame["video_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate video ids")
    incomplete = frame[(frame["source_type"].str.strip() == "") | (frame["stance"].str.strip() == "") | (frame["frame"].str.strip() == "")]
    if not incomplete.empty:
        raise ValueError(f"{path} has {len(incomplete)} incomplete rows: {incomplete['video_id'].tolist()[:10]}")
    frame["frame"] = frame["frame"].map(_canonical_frames)
    return frame


def paired(coder_a: Path, coder_b: Path) -> pd.DataFrame:
    left, right = load_coder(coder_a), load_coder(coder_b)
    merged = left.merge(right, on="video_id", suffixes=("_a", "_b"), how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("the two coder files cover different videos")
    return merged.drop(columns="_merge")


def agreement(pairs: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {"videos": int(len(pairs))}
    for field in ("source_type", "stance"):
        report[field] = {
            "percent_agreement": round(float((pairs[f"{field}_a"] == pairs[f"{field}_b"]).mean()), 4),
            "cohen_kappa": round(float(cohen_kappa_score(pairs[f"{field}_a"], pairs[f"{field}_b"])), 4),
        }
    per_frame = {}
    for label in [*FRAME_LABELS, "none"]:
        a = pairs["frame_a"].str.split("|").map(lambda values, label=label: int(label in values))
        b = pairs["frame_b"].str.split("|").map(lambda values, label=label: int(label in values))
        per_frame[label] = {
            "percent_agreement": round(float((a == b).mean()), 4),
            "cohen_kappa": round(float(cohen_kappa_score(a, b)), 4) if a.nunique() > 1 or b.nunique() > 1 else None,
            "coder_a_positive": int(a.sum()),
            "coder_b_positive": int(b.sum()),
        }
    report["frame"] = {
        "exact_set_agreement": round(float((pairs["frame_a"] == pairs["frame_b"]).mean()), 4),
        "per_frame": per_frame,
    }
    disagreements = []
    for row in pairs.itertuples(index=False):
        fields = [field for field in FIELDS if getattr(row, f"{field}_a") != getattr(row, f"{field}_b")]
        if fields:
            disagreements.append({
                "video_id": row.video_id,
                "fields": fields,
                **{f"{field}_a": getattr(row, f"{field}_a") for field in FIELDS},
                **{f"{field}_b": getattr(row, f"{field}_b") for field in FIELDS},
            })
    report["videos_to_settle"] = len(disagreements)
    report["fields_to_settle"] = int(sum(len(item["fields"]) for item in disagreements))
    report["disagreements"] = disagreements
    return report


def write_audit(pairs: pd.DataFrame, decisions: dict[str, dict[str, str]], path: Path = YOUTUBE_METADATA_AUDIT) -> dict[str, Any]:
    rows = []
    unresolved = []
    for row in pairs.itertuples(index=False):
        record = {"video_id": row.video_id}
        for field in FIELDS:
            short = "frame" if field == "frame" else field
            a, b = getattr(row, f"{field}_a"), getattr(row, f"{field}_b")
            if a == b:
                settled = a
            else:
                settled = decisions.get(row.video_id, {}).get(field, "")
                if field == "frame" and settled:
                    settled = _canonical_frames(settled)
                if not settled:
                    unresolved.append(f"{row.video_id}:{field}")
            record[f"coder_a_{short}"] = a
            record[f"coder_b_{short}"] = b
            record[f"adjudicated_{short}"] = settled
        rows.append(record)
    if unresolved:
        raise ValueError(f"{len(unresolved)} disagreements have no decision: {unresolved[:10]}")
    audit = pd.DataFrame(rows)[list(YOUTUBE_AUDIT_COLUMNS)]
    path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(path, index=False, lineterminator="\n")
    check = validate_youtube_metadata_audit(path)
    shown = path.relative_to(REPO_ROOT) if path.resolve().is_relative_to(REPO_ROOT) else path
    return {"path": str(shown), "rows": check["rows"], "status": check["status"], "coder_agreement": check["coder_agreement"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("compare", "write"))
    parser.add_argument("--coder-a", type=Path, required=True)
    parser.add_argument("--coder-b", type=Path, required=True)
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args()
    pairs = paired(args.coder_a, args.coder_b)
    if args.command == "compare":
        report = agreement(pairs)
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary = {key: value for key, value in report.items() if key != "disagreements"}
        print(json.dumps(summary, indent=2))
        return 0
    if args.decisions is None:
        parser.error("write needs --decisions")
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    print(json.dumps(write_audit(pairs, decisions), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
