# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

r"""Apply the frame lexicons and the circumvention indicator to the corpus.

    python -m src.platforms.bluesky.build_frames

Writes `data/interim/frames.parquet` (one row per post in the analysis corpus)
and `data/interim/frames_daily.parquet` (one row per day per frame).

Read `docs/bluesky/frame_validation.md` before using any number this produces. The
lexicons are measured instruments with known error rates, not ground truth, and
one of them (`child_safety`, recall 51%) is weak enough that its level should
not be reported without the caveat.

The circumvention indicator
---------------------------
`circumvention_act` is narrower than the `circumvent` frame and does something
different. The frame says a post DISCUSSES getting around age verification; the
indicator says the author reports PERSONALLY DOING IT -- "i forgot my vpn was
on", "i got a vpn", "i had to use a vpn to load my dms".

That distinction is what lets this project make a behavioural claim without a
stance classifier. A stance model would be inferring an attitude from text; this
is the author stating what they did. It is self-evidencing, so it cannot be
accused of putting words in anyone's mouth -- the usual and fair objection to
LLM-classified stance.

It is measured on the same corpus as everything else, so it inherits the same
exclusions (English, not meme, not spam) and the same day axis.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from src.platforms.bluesky.config import load_config
from src.platforms.bluesky.frames import COMPILED, is_first_person_circumvention

log = logging.getLogger("frames")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config()
    interim = cfg.interim_dir

    posts = pd.read_parquet(interim / "posts.parquet",
                            columns=["uri", "text_norm", "day", "author_did"])
    flags = pd.read_parquet(interim / "corpus_flags.parquet",
                            columns=["uri", "in_corpus"])
    pq_tbl = pd.read_parquet(interim / "post_query.parquet",
                             columns=["uri", "stratum", "phrase_exact"])

    # Frames are measured on the stratum-A measurement corpus: the neutral
    # backbone queries, phrase-exact, inside the analysis corpus. Stratum B is
    # excluded because it is ~98% a subset of A plus a keyword, so including it
    # would double-count the same posts under a different label.
    measurement = set(pq_tbl[(pq_tbl.stratum == "A") & pq_tbl.phrase_exact].uri)
    m = posts.merge(flags, on="uri")
    m = m[m.in_corpus & m.uri.isin(measurement)].reset_index(drop=True)
    log.info("measurement corpus: %s posts", f"{len(m):,}")

    for name, rx in COMPILED.items():
        m[name] = m.text_norm.map(lambda t, rx=rx: bool(rx.search(t)))
    frame_names = list(COMPILED)
    m["any_frame"] = m[frame_names].any(axis=1)
    m["n_frames"] = m[frame_names].sum(axis=1)
    m["circumvention_act"] = m.text_norm.map(is_first_person_circumvention)

    out = m[["uri", "day", "author_did"] + frame_names +
            ["any_frame", "n_frames", "circumvention_act"]]
    out.to_parquet(interim / "frames.parquet", compression="zstd", index=False)

    for name in frame_names:
        log.info("  %-14s %6s  %5.1f%%", name, f"{int(m[name].sum()):,}",
                 100 * m[name].mean())
    log.info("  %-14s %6s  %5.1f%%", "any frame", f"{int(m.any_frame.sum()):,}",
             100 * m.any_frame.mean())
    log.info("  %-14s %6s  %5.2f%%  (%s distinct authors)", "circumv. ACT",
             f"{int(m.circumvention_act.sum()):,}",
             100 * m.circumvention_act.mean(),
             f"{m.loc[m.circumvention_act, 'author_did'].nunique():,}")

    daily = m.groupby("day").agg(
        n_posts=("uri", "size"),
        **{f: (f, "sum") for f in frame_names},
        circumvention_act=("circumvention_act", "sum"),
    ).reset_index()
    for f in frame_names + ["circumvention_act"]:
        daily[f + "_per1k"] = (1000 * daily[f] / daily.n_posts).round(3)
    daily.to_parquet(interim / "frames_daily.parquet", compression="zstd", index=False)
    log.info("frames_daily: %s days", f"{len(daily):,}")

    # The headline behavioural result, printed so a run of this script is
    # self-documenting rather than needing a separate notebook to interpret.
    m["mo"] = m.day.str[:7]
    g = m.groupby("mo").agg(n=("uri", "size"), act=("circumvention_act", "sum"))
    g["per1k"] = (1000 * g.act / g.n).round(2)
    pre = g.loc[g.index < "2025-07", "act"].sum() / max(g.loc[g.index < "2025-07", "n"].sum(), 1)
    post = g.loc[g.index >= "2025-07", "act"].sum() / max(g.loc[g.index >= "2025-07", "n"].sum(), 1)
    log.info("-" * 60)
    log.info("Self-reported circumvention, before vs from UK enforcement month:")
    log.info("  2025-01..06  %.2f per 1,000 posts", 1000 * pre)
    log.info("  2025-07..end %.2f per 1,000 posts  (%.1fx)", 1000 * post, post / max(pre, 1e-9))
    log.info("Read docs/bluesky/frame_validation.md before quoting any frame level.")


if __name__ == "__main__":
    main()
