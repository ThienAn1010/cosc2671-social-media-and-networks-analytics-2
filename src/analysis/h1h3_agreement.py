"""Compare two coders' H1 to H3 label files and list the items to settle.

Reads only item ids and label columns, so returned Reddit label files can be checked without
opening any Reddit text. Accepts the CSVs the coding pages export and the raw packet CSVs
coders filled in by hand (their text columns are skipped).

Usage:
    python -m src.analysis.h1h3_agreement --packet bluesky_h1b --coder-a <a.csv> --coder-b <b.csv>
    python -m src.analysis.h1h3_agreement --packet reddit_docs --coder-a <kien.csv> --coder-b <duy.csv>
    python -m src.analysis.h1h3_agreement --packet reddit_threads --coder-a <kien.csv> --coder-b <duy.csv>
    python -m src.analysis.h1h3_agreement --packet youtube_topup --coder-a <hung.csv> --coder-b <claude.csv>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.analysis.artifacts import write_json
from src.analysis.h1h3_sample import OUTPUT_ROOT

FRAMES = [
    "policy_assurance", "child_safety", "privacy_surveillance",
    "governance_platform_responsibility", "circumvention_censorship_autonomy",
]
ALLOWED = {
    "relevance": {"relevant", "adjacent_contextual", "irrelevant"},
    "language": {"english", "mixed", "non_english", "too_short_ambiguous"},
    "target_policy": {"UK_OSA", "AU_SOCIAL_MINIMUM_AGE", "OTHER_EXTENDED_EVENT", "multiple", "unclear"},
    "thread_relevant": {"yes", "no"},
}
PACKET_FIELDS = {
    "bluesky_h1b": ["relevance", "language", "target_policy", "frame_labels"],
    "reddit_docs": ["relevance", "language", "target_policy", "frame_labels"],
    "reddit_threads": ["thread_relevant"],
    "youtube_topup": ["relevance", "language", "target_policy", "frame_labels"],
}


def load_labels(path: Path, fields: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    wanted = [column for column in ["item_id", *fields, "notes"] if column in header]
    missing = set(["item_id", *fields]) - set(wanted)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    frame = pd.read_csv(path, usecols=wanted, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for field in fields:
        frame[field] = frame[field].str.strip()
        if field == "thread_relevant":
            frame[field] = frame[field].str.casefold().replace({"y": "yes", "n": "no", "true": "yes", "false": "no"})
        if field in ALLOWED:
            bad = frame[frame[field].ne("") & ~frame[field].isin(ALLOWED[field])]
            if len(bad):
                raise ValueError(f"{path.name}: {len(bad)} rows have an unknown {field}, first {bad['item_id'].iloc[0]}")
        if field == "frame_labels":
            parts = frame[field].str.replace(" ", "", regex=False).str.split("|")
            unknown = {part for values in parts for part in values if part and part not in FRAMES}
            if unknown:
                raise ValueError(f"{path.name}: unknown frame labels {sorted(unknown)}")
            frame[field] = parts.map(lambda values: "|".join(label for label in FRAMES if label in values))
    return frame


def _kappa(left: pd.Series, right: pd.Series) -> float | None:
    if left.nunique() < 2 and right.nunique() < 2:
        return None
    return round(float(cohen_kappa_score(left, right)), 3)


def compare(packet: str, coder_a: Path, coder_b: Path) -> dict:
    fields = PACKET_FIELDS[packet]
    a, b = load_labels(coder_a, fields), load_labels(coder_b, fields)
    pairs = a.merge(b, on="item_id", suffixes=("_a", "_b"), how="outer", indicator=True, validate="one_to_one")
    if not pairs["_merge"].eq("both").all():
        raise ValueError(f"{packet} coder files do not cover the same item ids")
    pairs = pairs.drop(columns="_merge")
    first = fields[0]
    coded = pairs[pairs[f"{first}_a"].ne("") & pairs[f"{first}_b"].ne("")]
    summary = {"packet": packet, "items_a": len(a), "items_b": len(b), "items_both_coded": len(coded), "fields": {}}
    disagreements = set()
    for field in fields:
        left, right = coded[f"{field}_a"], coded[f"{field}_b"]
        differ = left.ne(right)
        disagreements |= set(coded.loc[differ, "item_id"])
        summary["fields"][field] = {"percent_agreement": round(float((~differ).mean()), 3) if len(coded) else None, "kappa": _kappa(left, right) if field != "frame_labels" else None}
    if "frame_labels" in fields:
        per_frame = {}
        for label in FRAMES:
            left = coded["frame_labels_a"].str.split("|").map(lambda values: label in values)
            right = coded["frame_labels_b"].str.split("|").map(lambda values: label in values)
            per_frame[label] = {"kappa": _kappa(left, right), "a_share": round(float(left.mean()), 3), "b_share": round(float(right.mean()), 3)}
        summary["fields"]["frame_labels"]["per_frame"] = per_frame
    summary["items_to_settle"] = len(disagreements)
    summary["settle_ids"] = sorted(disagreements)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packet", choices=sorted(PACKET_FIELDS), required=True)
    parser.add_argument("--coder-a", type=Path, required=True)
    parser.add_argument("--coder-b", type=Path, required=True)
    args = parser.parse_args()
    summary = compare(args.packet, args.coder_a, args.coder_b)
    write_json(OUTPUT_ROOT / "coordinator" / f"{args.packet}_agreement.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "settle_ids"}, indent=2))


if __name__ == "__main__":
    main()
