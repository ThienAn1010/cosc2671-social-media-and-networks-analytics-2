"""Build self-contained coding pages from the H1 to H3 coder packets.

Each page embeds its items and saves labels in the browser as the coder works. The page exports
item ids and labels only, never text, so a returned label file is safe to pass around. Reddit
pages must stay offline files: Reddit text can't be published until the teaching team approves it
in writing. The Bluesky page can also be published, because its text is pseudonymized.

Usage:
    python -m src.analysis.coding_bench --build
    python -m src.analysis.coding_bench --build --hosted-bluesky <path>   # page body for publishing
    python -m src.analysis.coding_bench --build --hosted-topup <path>     # H4 comment top-up page body
    python -m src.analysis.coding_bench --build --topup-handoff kien      # offline top-up folder and zip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.analysis.artifacts import relative_path
from src.analysis.h1h3_sample import OUTPUT_ROOT, _key
from src.analysis.h4_comment_topup import OUTPUT as TOPUP_SAMPLE

TEMPLATE = Path(__file__).with_name("coding_bench_template.html")
PAGE_ROOT = OUTPUT_ROOT / "pages"
CODER_BUTTONS = [["kien", "Kiên"], ["duy", "Duy"]]
REDDIT_SETS = {"R1": "Set R1 · Australian posts", "R2": "Set R2 · Active commenters", "R3": "Set R3 · Community sample"}
RETURN_NOTE = "Send the downloaded CSV to Hung in the team Teams folder. It holds item ids and labels only, no post text."


def _items(packet: pd.DataFrame, text_column: str, assigned: pd.Series | None = None) -> list[dict]:
    items = []
    for position, row in enumerate(packet.itertuples(index=False)):
        item = {"id": row.item_id, "t": getattr(row, text_column)}
        if "parent_context" in packet:
            item["p"] = row.parent_context
            item["r"] = row.root_context
        if "set" in packet:
            item["s"] = row.set
        if assigned is not None:
            item["a"] = assigned.iloc[position]
        items.append(item)
    return items


def _script_json(value) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def render(config: dict, items: list[dict], standalone: bool) -> str:
    page = TEMPLATE.read_text(encoding="utf-8")
    page = page.replace("__TITLE__", config["title"]).replace("__CONFIG__", _script_json(config)).replace("__ITEMS__", _script_json(items))
    if standalone:
        head = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        return head + page.replace("</style>", "</style>\n</head>\n<body>", 1) + "\n</body>\n</html>\n"
    return page


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def bluesky_page(standalone: bool) -> str:
    packets = [pd.read_csv(OUTPUT_ROOT / coder / f"bluesky_h1b_{coder}.csv", keep_default_na=False).assign(assigned=coder) for coder, _ in CODER_BUTTONS]
    packet = pd.concat(packets, ignore_index=True).sort_values("item_id")
    config = {
        "title": "Bluesky Frame Bench",
        "short": "Bluesky H1-B",
        "subtitle": "Label Bluesky posts from the largest reply communities. Each of you codes your own half; Claude codes every post separately as the second coder.",
        "kind": "docs",
        "store": "bluesky_h1b",
        "assigned": True,
        "coders": CODER_BUTTONS,
        "facts": [[str(len(packets[0])) + " / " + str(len(packets[1])), "posts (Kiên / Duy)"], ["4", "questions each"], ["~1.5 h", "each"]],
        "returnNote": RETURN_NOTE,
    }
    return render(config, _items(packet, "text_for_annotation", packet["assigned"]), standalone)


def reddit_docs_page() -> str:
    packet = pd.read_csv(OUTPUT_ROOT / "kien" / "reddit_docs_kien.csv", keep_default_na=False)
    config = {
        "title": "Reddit Frame Bench",
        "short": "Reddit documents",
        "subtitle": "Label Reddit posts and comments for the H1-R, H2 and H3 tests. You both code every item, separately. Work through the sets in order: R1 first.",
        "kind": "docs",
        "store": "reddit_docs",
        "assigned": False,
        "coders": CODER_BUTTONS,
        "sets": REDDIT_SETS,
        "facts": [[str(len(packet)), "items"], ["3", "sets"], ["~9 h", "each"]],
        "returnNote": RETURN_NOTE + " This page contains Reddit text: keep it on your own computer and don't upload it anywhere.",
    }
    return render(config, _items(packet, "text_for_annotation"), standalone=True)


def reddit_threads_page() -> str:
    packet = pd.read_csv(OUTPUT_ROOT / "kien" / "reddit_threads_kien.csv", keep_default_na=False)
    config = {
        "title": "Reddit Thread Check",
        "short": "Reddit threads",
        "subtitle": "Decide whether each sampled Reddit thread is about age checks. Comments in threads you both reject drop out of the tests.",
        "kind": "threads",
        "store": "reddit_threads",
        "assigned": False,
        "coders": CODER_BUTTONS,
        "facts": [[str(len(packet)), "threads"], ["1", "question each"], ["~1 h", "each"]],
        "returnNote": RETURN_NOTE + " This page contains Reddit text: keep it on your own computer and don't upload it anywhere.",
    }
    return render(config, _items(packet, "thread_text"), standalone=True)


TOPUP_ASSIGNMENT = TOPUP_SAMPLE.with_name("youtube_topup_assignment.csv")
TOPUP_CODERS = [["hung", "Hung"], ["kien", "Kiên"]]
TOPUP_ROOT = TOPUP_SAMPLE.parent / "topup_handoff"


def _topup_packet() -> pd.DataFrame:
    sample = pd.read_csv(TOPUP_SAMPLE, keep_default_na=False)
    assignment = pd.read_csv(TOPUP_ASSIGNMENT, keep_default_na=False)
    sample = sample.merge(assignment, left_on="annotation_id", right_on="item_id", how="left", validate="one_to_one")
    if sample["assigned_to"].eq("").any() or sample["assigned_to"].isna().any():
        raise ValueError("every top-up comment needs a coder in youtube_topup_assignment.csv")
    order = sample["annotation_id"].map(lambda value: _key(f"topup|order|{value}"))
    return sample.assign(_order=order).sort_values(["_order", "annotation_id"]).drop(columns=["item_id"]).rename(columns={"annotation_id": "item_id"})


def youtube_topup_page(standalone: bool) -> str:
    """Page for the H4 comment top-up: YouTube comments with their video's title and description.

    Hung coded the first comments and Kiên codes the rest; youtube_topup_assignment.csv says who
    has which, and each coder sees only their own. Claude codes all of them separately.
    """

    packet = _topup_packet()
    counts = packet["assigned_to"].value_counts()
    config = {
        "title": "Comment Framing Bench",
        "short": "YouTube comments",
        "subtitle": "Label YouTube comments for the H4 test. Code each comment's own framing, not its video's. Claude codes the same comments separately as the second coder.",
        "kind": "docs",
        "store": "youtube_topup",
        "assigned": True,
        "coders": TOPUP_CODERS,
        "rootLabel": "Video title and description",
        "facts": [[f"{counts.get('hung', 0)} / {counts.get('kien', 0)}", "comments (Hung / Kiên)"], ["4", "questions each"], ["~1 min", "per comment"]],
        "returnNote": "Labels save to this page. If you coded offline, send the downloaded CSV to Hung in the team Teams folder; it holds comment ids and labels only.",
    }
    return render(config, _items(packet, "text_for_annotation", packet["assigned_to"].reset_index(drop=True)), standalone)


def topup_handoff(coder: str = "kien") -> dict:
    """Offline page and spreadsheet for one top-up coder, zipped for the Teams folder."""

    import shutil

    packet = _topup_packet()
    folder = TOPUP_ROOT / f"h4_comments_{coder}"
    if folder.exists():
        shutil.rmtree(folder)
    mine = packet[packet["assigned_to"].eq(coder)]
    columns = ["item_id", "text_for_annotation", "parent_context", "root_context"]
    sheet = mine[columns].assign(relevance="", language="", target_policy="", frame_labels="", notes="")
    folder.mkdir(parents=True)
    sheet.to_csv(folder / f"youtube_topup_{coder}.csv", index=False, encoding="utf-8-sig")
    _write(folder / "comment_framing_bench.html", youtube_topup_page(standalone=True))
    instructions = Path(__file__).resolve().parents[2] / "docs" / "h4_comment_coding_instructions.md"
    shutil.copy(instructions, folder / "README_comment_coding.md")
    (folder / "returned").mkdir()
    archive = shutil.make_archive(str(folder), "zip", folder)
    return {"items": int(len(mine)), "folder": relative_path(folder), "zip": relative_path(Path(archive))}

def build(hosted_bluesky: Path | None = None) -> dict:
    written = {
        "bluesky": _write(PAGE_ROOT / "bluesky_frame_bench.html", bluesky_page(standalone=True)),
        "reddit_docs": _write(PAGE_ROOT / "reddit_frame_bench.html", reddit_docs_page()),
        "reddit_threads": _write(PAGE_ROOT / "reddit_thread_check.html", reddit_threads_page()),
    }
    if hosted_bluesky:
        written["bluesky_hosted"] = _write(hosted_bluesky, bluesky_page(standalone=False))
    return {name: str(path) if not str(path).startswith(str(OUTPUT_ROOT)) else relative_path(path) for name, path in written.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true", help="write the offline coding pages")
    parser.add_argument("--hosted-bluesky", type=Path, help="also write the Bluesky page body for publishing as an artifact")
    parser.add_argument("--hosted-topup", type=Path, help="also write the YouTube comment top-up page body for publishing")
    parser.add_argument("--topup-handoff", metavar="CODER", help="also write the offline top-up folder and zip for one coder")
    args = parser.parse_args()
    if not args.build:
        parser.error("nothing to do; pass --build")
    print(json.dumps(build(args.hosted_bluesky), indent=2))
    if args.hosted_topup:
        print(_write(args.hosted_topup, youtube_topup_page(standalone=False)))
    if args.topup_handoff:
        print(json.dumps(topup_handoff(args.topup_handoff), indent=2))


if __name__ == "__main__":
    main()
