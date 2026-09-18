"""Settle two coders' labels: build a settling page, then write the settled label file.

Only differences that can change a test go on the page:
- relevance when either coder said relevant, and language when either said English (only those
  values make an item usable);
- target policy for Reddit set R1 (H2) and the YouTube top-up (H4 eligibility), when either coder
  called the item relevant;
- frames when either coder called the item relevant and English.
Other differences don't enter any analysis, so the settled file keeps coder A's value there.
For the YouTube top-up, a target difference goes on the page only when one coder named a policy
and the other didn't, and frames only when either coder's labels make the comment count for H4.

Packets and their settled outputs (under data/analysis/annotation/h1h3/settled/ unless noted):
- bluesky_h1b    -> bluesky_h1b.csv
- reddit_docs    -> reddit_docs.csv       (page must stay offline: Reddit text)
- reddit_threads -> reddit_threads.csv    (page must stay offline: Reddit text)
- youtube_topup  -> data/analysis/annotation/youtube_comment_topup_labels.csv, in the shape
                    src/analysis/h4_human_sample.py reads

Usage:
    python -m src.analysis.h1h3_settle page --packet P --coder-a A.csv --coder-b B.csv --out page.html [--hosted]
    python -m src.analysis.h1h3_settle write --packet P --coder-a A.csv --coder-b B.csv --decisions D.csv|D.json [--out settled.csv]
    python -m src.analysis.h1h3_settle prefer --packet P --coder-a A.csv --coder-b B.csv --prefer b --out D.csv
        (writes a decisions file that takes one coder's value on every difference that matters)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.analysis.coding_bench import TEMPLATE as BENCH_TEMPLATE, _script_json, _write
from src.analysis.h1h3_agreement import PACKET_FIELDS, load_labels
from src.analysis.h1h3_sample import OUTPUT_ROOT as SAMPLE_ROOT
from src.analysis.h4_comment_topup import OUTPUT as TOPUP_SAMPLE

TEMPLATE = Path(__file__).with_name("settle_bench_template.html")
SETTLED_ROOT = SAMPLE_ROOT / "settled"
TOPUP_LABELS = TOPUP_SAMPLE.with_name("youtube_comment_topup_labels.csv")
H4_TARGETS = {"UK_OSA", "AU_SOCIAL_MINIMUM_AGE", "OTHER_EXTENDED_EVENT"}
TEXT_SOURCES = {
    "bluesky_h1b": (SAMPLE_ROOT / "claude" / "bluesky_h1b_claude.csv", "text_for_annotation"),
    "reddit_docs": (SAMPLE_ROOT / "kien" / "reddit_docs_kien.csv", "text_for_annotation"),
    "reddit_threads": (SAMPLE_ROOT / "kien" / "reddit_threads_kien.csv", "thread_text"),
    "youtube_topup": (TOPUP_SAMPLE, "text_for_annotation"),
}
PAGE_TEXT = {
    "bluesky_h1b": ("Settle Bluesky Labels", "Bluesky H1-B", "Thread opening", "Where the two coders disagree on a Bluesky post, pick the label the post supports."),
    "reddit_docs": ("Settle Reddit Labels", "Reddit documents", "Thread opening", "Where Kiên and Duy disagree on a Reddit item, pick the label the text supports. Keep this page on your computer."),
    "reddit_threads": ("Settle Reddit Threads", "Reddit threads", "Thread opening", "Where Kiên and Duy disagree on whether a thread is about age checks, decide. Keep this page on your computer."),
    "youtube_topup": ("Settle Comment Labels", "YouTube comments", "Video title and description", "Where the team's labels and Claude's differ on a comment, pick the label the comment supports."),
}
RULES = {
    "youtube_topup": "Most frame differences come from how many frames to tick. Tick only what the comment itself argues, and use the video to name the policy when the comment clearly refers to it.",
}


def _labels(packet: str, path: Path) -> pd.DataFrame:
    fields = PACKET_FIELDS[packet]
    frame = load_labels(path, fields)
    if frame["item_id"].duplicated().any():
        raise ValueError(f"{path.name} lists an item twice")
    return frame[["item_id", *fields]]


def _pairs(packet: str, coder_a: Path, coder_b: Path) -> pd.DataFrame:
    a, b = _labels(packet, coder_a), _labels(packet, coder_b)
    pairs = a.merge(b, on="item_id", suffixes=("_a", "_b"), how="outer", indicator=True, validate="one_to_one")
    if not pairs["_merge"].eq("both").all():
        raise ValueError(f"{packet} coder files do not cover the same item ids")
    pairs = pairs.drop(columns="_merge")
    if packet == "reddit_docs":
        sets = pd.read_csv(TEXT_SOURCES[packet][0], usecols=["item_id", "set"], encoding="utf-8-sig")
        pairs = pairs.merge(sets, on="item_id", how="left")
    return pairs


def fields_to_settle(packet: str, row: pd.Series) -> list[str]:
    fields = PACKET_FIELDS[packet]
    differs = [field for field in fields if row[f"{field}_a"] != row[f"{field}_b"]]
    if packet == "reddit_threads":
        return differs
    relevant = "relevant" in (row["relevance_a"], row["relevance_b"])
    readable = any(row[f"relevance_{side}"] == "relevant" and row[f"language_{side}"] == "english" for side in "ab")
    keep = []
    # Only "relevant" and "english" make an item usable, so other splits (adjacent vs irrelevant,
    # mixed vs non-English) change nothing.
    if "relevance" in differs and relevant:
        keep.append("relevance")
    if "language" in differs and relevant and "english" in (row["language_a"], row["language_b"]):
        keep.append("language")
    if packet == "youtube_topup":
        counts = [row[f"relevance_{side}"] == "relevant" and row[f"language_{side}"] == "english" and row[f"target_policy_{side}"] in H4_TARGETS for side in "ab"]
        if "target_policy" in differs and relevant:
            named = [row[f"target_policy_{side}"] in H4_TARGETS for side in "ab"]
            if named[0] != named[1]:
                keep.append("target_policy")
        if "frame_labels" in differs and any(counts):
            keep.append("frame_labels")
        return keep
    if "target_policy" in differs and relevant and packet == "reddit_docs" and row.get("set") == "R1":
        keep.append("target_policy")
    if "frame_labels" in differs and readable:
        keep.append("frame_labels")
    return keep


def _style() -> str:
    bench = BENCH_TEMPLATE.read_text(encoding="utf-8")
    return re.search(r"<style>.*?</style>", bench, re.S).group(0)


def build_page(packet: str, coder_a: Path, coder_b: Path, out: Path, hosted: bool, names: tuple[str, str]) -> dict:
    pairs = _pairs(packet, coder_a, coder_b)
    source, text_column = TEXT_SOURCES[packet]
    texts = pd.read_csv(source, keep_default_na=False, encoding="utf-8-sig")
    if packet == "youtube_topup":
        texts = texts.rename(columns={"annotation_id": "item_id"})
    texts = texts.set_index("item_id")
    items = []
    for _, row in pairs.iterrows():
        fields = fields_to_settle(packet, row)
        if not fields:
            continue
        text = texts.loc[row["item_id"]]
        item = {"id": row["item_id"], "t": text[text_column], "d": fields,
                "a": {field: row[f"{field}_a"] for field in PACKET_FIELDS[packet]},
                "b": {field: row[f"{field}_b"] for field in PACKET_FIELDS[packet]}}
        if "parent_context" in text:
            item["p"], item["r"] = text["parent_context"], text["root_context"]
        if packet == "reddit_docs":
            item["s"] = row["set"]
        items.append(item)
    title, short, root_label, subtitle = PAGE_TEXT[packet]
    decisions = sum(len(item["d"]) for item in items)
    config = {
        "title": title, "short": short, "subtitle": subtitle, "rootLabel": root_label, "store": packet,
        "kind": "threads" if packet == "reddit_threads" else "docs", "nameA": names[0], "nameB": names[1],
        "rule": RULES.get(packet, ""), "estimate": f"~{max(1, round(decisions * 25 / 60))} min",
        "returnNote": "Tell Claude when you've finished." if hosted else "Send the downloaded decisions CSV to Claude; it holds item ids and labels only.",
    }
    page = TEMPLATE.read_text(encoding="utf-8").replace("__STYLE__", _style())
    page = page.replace("__TITLE__", title).replace("__CONFIG__", _script_json(config)).replace("__ITEMS__", _script_json(items))
    if not hosted:
        head = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        page = head + page.replace("</style>\n\n<div", "</style>\n</head>\n<body>\n<div", 1) + "\n</body>\n</html>\n"
    _write(out, page)
    return {"items": len(items), "decisions": decisions, "page": str(out)}


def _decisions(path: Path) -> dict[str, dict[str, str]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("decisions", payload)
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    result: dict[str, dict[str, str]] = {}
    for row in table.itertuples(index=False):
        result.setdefault(row.item_id, {})[row.field] = row.value
    return result


def write_settled(packet: str, coder_a: Path, coder_b: Path, decisions_path: Path, out_path: Path | None = None) -> dict:
    pairs = _pairs(packet, coder_a, coder_b)
    decisions = _decisions(decisions_path)
    fields = PACKET_FIELDS[packet]
    rows, missing, overridden = [], [], 0
    for _, row in pairs.iterrows():
        settled = {"item_id": row["item_id"]}
        needed = fields_to_settle(packet, row)
        for field in fields:
            if field in needed:
                if field not in decisions.get(row["item_id"], {}):
                    missing.append(f"{row['item_id']}:{field}")
                    continue
                settled[field] = decisions[row["item_id"]][field]
                overridden += 1
            else:
                settled[field] = row[f"{field}_a"]
        rows.append(settled)
    if missing:
        raise ValueError(f"{len(missing)} decisions are missing, first: {missing[:5]}")
    settled = pd.DataFrame(rows)
    if packet == "youtube_topup":
        sample = pd.read_csv(TOPUP_SAMPLE, keep_default_na=False)
        out = sample[["annotation_id", "platform", "split", "doc_id", "author_key", "cluster_id", "case_window_id"]].merge(
            settled.rename(columns={"item_id": "annotation_id", **{field: f"adjudicated_{field}" for field in fields}}),
            on="annotation_id", how="inner", validate="one_to_one")
        path = TOPUP_LABELS
    else:
        out = settled
        path = out_path or SETTLED_ROOT / f"{packet}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return {"packet": packet, "rows": len(out), "decisions_applied": overridden, "path": str(path)}


def prefer_decisions(packet: str, coder_a: Path, coder_b: Path, side: str, out: Path) -> dict:
    """Decisions that settle every difference that matters in favour of one coder."""

    pairs = _pairs(packet, coder_a, coder_b)
    rows = [(row["item_id"], field, row[f"{field}_{side}"]) for _, row in pairs.iterrows() for field in fields_to_settle(packet, row)]
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["item_id", "field", "value"]).to_csv(out, index=False)
    return {"packet": packet, "decisions": len(rows), "items": len({row[0] for row in rows}), "preferred": side, "path": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["page", "write", "prefer"])
    parser.add_argument("--packet", choices=sorted(TEXT_SOURCES), required=True)
    parser.add_argument("--coder-a", type=Path, required=True)
    parser.add_argument("--coder-b", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--hosted", action="store_true", help="write a page body for publishing instead of an offline file")
    parser.add_argument("--names", nargs=2, default=["Coder A", "Coder B"], metavar=("NAME_A", "NAME_B"))
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--prefer", choices=["a", "b"], help="prefer: whose value settles each difference")
    args = parser.parse_args()
    if args.action == "page":
        if args.out is None:
            parser.error("page needs --out")
        if args.hosted and args.packet.startswith("reddit"):
            parser.error("Reddit settling pages contain Reddit text and must stay offline")
        print(json.dumps(build_page(args.packet, args.coder_a, args.coder_b, args.out, args.hosted, tuple(args.names)), indent=2))
    elif args.action == "prefer":
        if args.out is None or args.prefer is None:
            parser.error("prefer needs --prefer and --out")
        print(json.dumps(prefer_decisions(args.packet, args.coder_a, args.coder_b, args.prefer, args.out), indent=2))
    else:
        if args.decisions is None:
            parser.error("write needs --decisions")
        print(json.dumps(write_settled(args.packet, args.coder_a, args.coder_b, args.decisions, args.out), indent=2))


if __name__ == "__main__":
    main()
