# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

r"""Materialise the analysis corpus and the reply graph the team agreed on.

    python -m src.platforms.bluesky.build_analysis_tables

Reads `data/interim/*.parquet`, writes four more tables there. Raw is untouched
and nothing upstream is modified.

This script is where the team's exclusion decisions of 2026-09-14 stop being
prose in a document and become a column other people can join on. Every filter
below traces to a recorded decision; none of them is invented here. See
`docs/bluesky/collection_and_cleaning_log.md` for the evidence behind each.

What it writes
--------------
  corpus_flags.parquet   uri -> in_corpus, with the reason it was excluded
  series_daily.parquet   (query, day) -> post count, plus the baseline denominator
  reply_edges.parquet    author -> author reply edges, weighted
  reply_nodes.parquet    author -> degrees and connected component

The analysis corpus
-------------------
A post is `in_corpus` when it is English, not a meme, and not from a spam or
repeat-burst account. That is decisions 3 and the English-only decision, applied
once so every downstream table agrees.

`phrase_exact` is deliberately NOT part of it. Phrase-exactness is a property of
a (post, query) PAIR, not of a post -- the same post can be an exact match for
`age verification` and a scattered-term match for `under-16 ban`. It therefore
belongs in the series definition, where it is applied, and not in a post-level
flag where it would be ambiguous.

Exclusions are recorded as a REASON, not just a boolean, so "how many posts did
we drop and why" is answerable from the data rather than from memory.
"""

import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.platforms.bluesky.config import load_config

log = logging.getLogger("analysis")

# Accounts whose posting is broadcast or spam rather than discourse. Decision 3
# keeps news bots and bridged accounts deliberately: institutional voices are
# part of public attention to a policy even though they are not opinion.
EXCLUDED_ACCOUNT_CLASSES = ("labelled_spam", "repeater")

# A query needs at least this many posts to be reported as its own series.
#
# The exact value barely matters because the gap is enormous: the smallest kept
# series has 633 posts and the largest dropped one has 110. Anything between
# those two gives the same answer, which is the useful kind of threshold.
MIN_SERIES_N = 500


def build_corpus_flags(cfg):
    posts = pd.read_parquet(cfg.interim_dir / "posts.parquet",
                            columns=["uri", "author_did", "day", "is_english",
                                     "is_meme", "in_search", "in_thread"])
    authors = pd.read_parquet(cfg.interim_dir / "authors.parquet",
                              columns=["did", "account_class"])
    posts = posts.merge(authors, left_on="author_did", right_on="did", how="left")

    spam = posts["account_class"].isin(EXCLUDED_ACCOUNT_CLASSES)
    # Reason is assigned in priority order, so each post carries the first
    # reason that applies rather than an arbitrary one.
    reason = pd.Series("included", index=posts.index, dtype=object)
    reason[~posts["is_english"]] = "not_english"
    reason[posts["is_english"] & posts["is_meme"]] = "meme"
    reason[posts["is_english"] & ~posts["is_meme"] & spam] = "spam_account"
    posts["exclusion_reason"] = reason
    posts["in_corpus"] = reason == "included"

    out = posts[["uri", "in_corpus", "exclusion_reason", "day",
                 "author_did", "in_search", "in_thread"]]
    out.to_parquet(cfg.interim_dir / "corpus_flags.parquet",
                   compression="zstd", index=False)
    log.info("corpus_flags: %s of %s posts in corpus (%.1f%%)",
             f"{int(out.in_corpus.sum()):,}", f"{len(out):,}",
             100 * out.in_corpus.mean())
    for k, v in reason.value_counts().items():
        log.info("    %-14s %9s", k, f"{v:,}")
    return out


def build_series(cfg, flags):
    """(query, day) counts over the agreed corpus, phrase-exact only."""
    pq_tbl = pd.read_parquet(cfg.interim_dir / "post_query.parquet",
                             columns=["uri", "query", "stratum", "phrase_exact", "day"])
    keep = set(flags.loc[flags.in_corpus, "uri"])
    m = pq_tbl[pq_tbl.phrase_exact & pq_tbl.uri.isin(keep)]

    per_query = m.groupby("query").uri.nunique().sort_values(ascending=False)
    reportable = set(per_query[per_query >= MIN_SERIES_N].index)
    log.info("series: %s of %s queries clear the %s-post floor",
             len(reportable), len(per_query), MIN_SERIES_N)
    for q, n in per_query.items():
        log.info("    %-26s %7s  %s", q, f"{n:,}",
                 "" if q in reportable else "<- dropped as a series")

    daily = (m.groupby(["query", "stratum", "day"]).uri.nunique()
             .rename("n_posts").reset_index())
    daily["reportable"] = daily["query"].isin(reportable)

    # The denominator. Platform activity fell 38% across the window, so a raw
    # count understates the topic's rise; normalising by the baseline basket is
    # what makes the series comparable end to end.
    #
    # A day containing a SATURATED cell loses its denominator entirely, rather
    # than being summed over the remaining terms. Dropping one term from a
    # five-term basket would make that day's total too SMALL and so inflate its
    # normalised rate -- the opposite of the intended correction. Nor can the
    # missing term be imputed from the others: it saturated precisely because it
    # was unusually busy, so the cap is a floor on a value we cannot see.
    #
    # Four days of 608 are affected. A null is the honest answer for them.
    base = pd.read_json(cfg.baseline_path, lines=True)
    sat = pd.read_csv(cfg.interim_dir / "baseline_saturated_cells.csv")
    incomplete = set(sat.day)
    denom = (base.groupby("day").hits_total.sum()
             .rename("baseline_total").reset_index())
    denom.loc[denom.day.isin(incomplete), "baseline_total"] = pd.NA
    daily = daily.merge(denom, on="day", how="left")
    daily["per_10k_baseline"] = (1e4 * daily.n_posts / daily.baseline_total).round(4)

    daily.to_parquet(cfg.interim_dir / "series_daily.parquet",
                     compression="zstd", index=False)
    null_days = daily.loc[daily.baseline_total.isna(), "day"].nunique()
    log.info("series_daily: %s rows, %s days | %s day(s) have no denominator "
             "(saturated baseline cell)", f"{len(daily):,}", daily.day.nunique(),
             null_days)
    return daily


def build_reply_graph(cfg, flags):
    """Author-to-author reply edges over the agreed corpus.

    `phrase_exact` plays no part here, and should not: replies were discovered
    by their POSITION in a conversation, not by matching a query, so there is no
    query for them to be exact against. The quality filter (English, not meme,
    not a spam account) still applies to both ends of every edge.
    """
    edges = pd.read_parquet(cfg.interim_dir / "thread_edges.parquet",
                            columns=["reply_uri", "parent_uri", "reply_author_did",
                                     "reply_day"])
    author_of = dict(zip(flags.uri, flags.author_did))
    in_corpus = set(flags.loc[flags.in_corpus, "uri"])

    edges["parent_author"] = edges.parent_uri.map(author_of)
    ok = (edges.reply_uri.isin(in_corpus) & edges.parent_uri.isin(in_corpus)
          & edges.parent_author.notna()
          & (edges.reply_author_did != edges.parent_author))
    dropped_self = int((edges.reply_author_did == edges.parent_author).sum())
    e = edges[ok]

    agg = (e.groupby(["reply_author_did", "parent_author"])
           .agg(weight=("reply_uri", "size"),
                first_day=("reply_day", "min"),
                last_day=("reply_day", "max"))
           .reset_index()
           .rename(columns={"reply_author_did": "source_did",
                            "parent_author": "target_did"}))

    # Reciprocity, computed here so nobody has to recompute it to describe the
    # graph. It is the number that says what KIND of graph this is: a low value
    # means strangers replying to strangers, not a mutual-relationship network.
    pairs = set(zip(agg.source_did, agg.target_did))
    recip = sum(1 for a, b in pairs if (b, a) in pairs)
    agg["reciprocated"] = [(b, a) in pairs for a, b in zip(agg.source_did, agg.target_did)]

    # Connected components on the undirected projection, via union-find.
    parent: dict[str, str] = {}

    def find(x):
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != x:       # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    nodes = set(agg.source_did) | set(agg.target_did)
    for n in nodes:
        parent[n] = n
    for a, b in pairs:
        union(a, b)
    comp = {n: find(n) for n in nodes}
    sizes = Counter(comp.values())
    biggest = sizes.most_common(1)[0][0] if sizes else None

    out_deg = agg.groupby("source_did").size()
    in_deg = agg.groupby("target_did").size()
    out_str = agg.groupby("source_did").weight.sum()
    in_str = agg.groupby("target_did").weight.sum()
    nodes_df = pd.DataFrame({"did": sorted(nodes)})
    nodes_df["out_degree"] = nodes_df.did.map(out_deg).fillna(0).astype(int)
    nodes_df["in_degree"] = nodes_df.did.map(in_deg).fillna(0).astype(int)
    nodes_df["out_strength"] = nodes_df.did.map(out_str).fillna(0).astype(int)
    nodes_df["in_strength"] = nodes_df.did.map(in_str).fillna(0).astype(int)
    nodes_df["component_size"] = nodes_df.did.map(lambda d: sizes[comp[d]])
    nodes_df["in_largest_component"] = nodes_df.did.map(lambda d: comp[d] == biggest)

    agg.to_parquet(cfg.interim_dir / "reply_edges.parquet", compression="zstd", index=False)
    nodes_df.to_parquet(cfg.interim_dir / "reply_nodes.parquet", compression="zstd", index=False)

    log.info("reply_edges: %s author->author edges from %s reply posts "
             "(%s self-replies dropped)", f"{len(agg):,}", f"{len(e):,}",
             f"{dropped_self:,}")
    log.info("reply_nodes: %s authors | largest component %s (%.1f%%) | "
             "reciprocity %.2f%% | mean weight %.2f",
             f"{len(nodes_df):,}", f"{sizes[biggest]:,}",
             100 * sizes[biggest] / len(nodes_df),
             100 * recip / max(len(pairs), 1), agg.weight.mean())
    return agg, nodes_df


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config()
    cfg.ensure_dirs()

    flags = build_corpus_flags(cfg)
    build_series(cfg, flags)
    build_reply_graph(cfg, flags)

    log.info("-" * 60)
    for name in ("corpus_flags", "series_daily", "reply_edges", "reply_nodes"):
        p = cfg.interim_dir / f"{name}.parquet"
        log.info("  %-22s %7.1f MB", name + ".parquet", p.stat().st_size / 1e6)
    log.info("Every filter here traces to a recorded decision; see "
             "docs/bluesky/collection_and_cleaning_log.md")


if __name__ == "__main__":
    main()
