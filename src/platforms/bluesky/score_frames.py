# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

r"""Score the frame lexicons against hand-coded ground truth.

    python -m src.platforms.bluesky.score_frames coder_codes.txt [second_coder.txt]

Codes file format: one line per item, `item_id:FRAMES`, where FRAMES is any of
P C I S F concatenated (empty means no frame applies). Lines may be in any
order; every item in the sheet must appear exactly once.

    1:IS
    2:S
    3:
    4:F

With one codes file this reports precision, recall and F1 per frame. With two it
also reports Cohen's kappa between the coders, which is the statistic that shows
the coding is reproducible rather than one person's opinion. See
`docs/bluesky/frame_validation.md` for the coding protocol.

Why precision and recall are estimated on different pools
---------------------------------------------------------
The sample deliberately mixes a precision pool (posts the lexicon flags) with a
recall pool (posts it flags for nothing), shuffled and unlabelled. Precision is
computed on the first, recall on the second, and each is weighted back to its
pool's true size -- otherwise the base rates in the sample, which were chosen
for coding efficiency, would leak into the estimates.
"""

import sys
from pathlib import Path

import pandas as pd

from src.platforms.bluesky.config import PROJECT_ROOT

LETTER = {"P": "privacy", "C": "circumvent", "I": "id_upload",
          "S": "child_safety", "F": "free_speech"}
FRAMES = list(LETTER.values())


def read_codes(path: Path) -> pd.DataFrame:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        item, _, codes = raw.partition(":")
        codes = codes.strip().upper()
        bad = set(codes) - set(LETTER)
        if bad:
            raise SystemExit(f"item {item}: unknown code letter(s) {sorted(bad)}")
        row = {"item_id": int(item)}
        for letter, frame in LETTER.items():
            row["true_" + frame] = letter in codes
        rows.append(row)
    df = pd.DataFrame(rows)
    dupes = df.item_id[df.item_id.duplicated()].tolist()
    if dupes:
        raise SystemExit(f"duplicate item_id(s): {dupes[:10]}")
    return df


def kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa for two binary codings of the same items."""
    n = len(a)
    observed = (a == b).mean()
    pa1, pb1 = a.mean(), b.mean()
    expected = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if expected == 1:
        return float("nan")
    return (observed - expected) / (1 - expected)


def interpret(k: float) -> str:
    if pd.isna(k):
        return "undefined"
    for cut, label in ((0.81, "almost perfect"), (0.61, "substantial"),
                       (0.41, "moderate"), (0.21, "fair"), (0.0, "slight")):
        if k >= cut:
            return label
    return "poor (worse than chance)"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    root = PROJECT_ROOT
    interim = root / "data" / "interim"

    key = pd.read_csv(interim / "frame_validation_key.csv")
    coded = read_codes(Path(sys.argv[1]))
    df = key.merge(coded, on="item_id", how="left")
    missing = df.true_privacy.isna().sum()
    if missing:
        raise SystemExit(f"{missing} item(s) in the sheet have no code. "
                         "Every item needs a line, even if empty (e.g. `17:`).")

    # Pool sizes in the corpus, needed to weight the two pools back together.
    posts = pd.read_parquet(interim / "posts.parquet", columns=["uri"])
    print("=" * 66)
    print("FRAME LEXICON VALIDATION  (n = %d hand-coded items)" % len(df))
    print("=" * 66)
    print("%-14s %6s %6s %6s %7s %7s %6s" %
          ("frame", "TP", "FP", "FN", "prec", "recall", "F1"))

    out = []
    for frame in FRAMES:
        lex = df["lex_" + frame].astype(bool)
        true = df["true_" + frame].astype(bool)
        tp = int((lex & true).sum())
        fp = int((lex & ~true).sum())
        fn = int((~lex & true).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) else float("nan")
        print("%-14s %6d %6d %6d %6.1f%% %6.1f%% %6.2f"
              % (frame, tp, fp, fn, 100 * prec, 100 * rec, f1))
        out.append({"frame": frame, "tp": tp, "fp": fp, "fn": fn,
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4)})

    print()
    print("Precision is estimated on the flagged pool, recall on the unflagged")
    print("pool. FN counts here are posts the lexicon missed that a human coded")
    print("as the frame -- the recall pool exists to find exactly those.")

    if len(sys.argv) > 2:
        second = read_codes(Path(sys.argv[2])).rename(
            columns={"true_" + f: "b_" + f for f in FRAMES})
        both = df.merge(second, on="item_id", how="inner")
        print()
        print("=" * 66)
        print("INTER-RATER RELIABILITY  (n = %d items coded by both)" % len(both))
        print("=" * 66)
        print("%-14s %8s %10s   %s" % ("frame", "agree", "kappa", "interpretation"))
        for frame in FRAMES:
            a = both["true_" + frame].astype(bool)
            b = both["b_" + frame].astype(bool)
            k = kappa(a, b)
            print("%-14s %7.1f%% %10.3f   %s"
                  % (frame, 100 * (a == b).mean(), k, interpret(k)))
            for row in out:
                if row["frame"] == frame:
                    row["kappa"] = round(k, 4)
                    row["agreement"] = round(float((a == b).mean()), 4)
    else:
        print()
        print("NOTE: only one codes file given, so Cohen's kappa was not computed.")
        print("      Kappa needs two independent coders by definition. See")
        print("      docs/bluesky/frame_validation.md for how to add the second one.")

    pd.DataFrame(out).to_csv(interim / "frame_validation_scores.csv", index=False)
    print()
    print("wrote", interim / "frame_validation_scores.csv")


if __name__ == "__main__":
    main()
