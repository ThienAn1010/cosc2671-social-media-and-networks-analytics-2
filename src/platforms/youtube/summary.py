# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

"""Summary tables for the YouTube results/QA notebook (notebooks/analysis/youtube_results_qa.ipynb).

The notebook only calls these functions and explains the results in markdown; no logic lives in the notebook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.platforms.youtube.storage import REPO_ROOT

PROCESSED_ROOT = REPO_ROOT / "data" / "processed" / "youtube"
SEED_PATH = REPO_ROOT / "config" / "youtube" / "seed_videos.csv"
EVENT_ORDER = ["E1", "E2", "E3", "E4", "E5"]
# Reaction window shown in the notebook: a week before each event to two months after.
DAYS_BEFORE, DAYS_AFTER = 7, 60


def load_manifests(root: Path = PROCESSED_ROOT) -> dict[str, Any]:
    return {name: json.loads((root / f"{name}_manifest.json").read_text(encoding="utf-8")) for name in ("prepare", "edges")}


def comments_only(documents: pd.DataFrame) -> pd.DataFrame:
    return documents[documents["thing"] != "video"]


# D-015: strict = eligible only; inclusive = English top language, eligible or language_uncertain.
def inclusive_mask(documents: pd.DataFrame) -> pd.Series:
    return documents["is_english"].astype(bool) & documents["exclusion_status"].isin(["eligible", "language_uncertain"])


def seed_summary(seed_path: Path = SEED_PATH) -> pd.DataFrame:
    seed = pd.read_csv(seed_path)
    table = seed.pivot_table(index="event_id", columns="video_type", values="video_id", aggfunc="count", fill_value=0)
    table["videos"] = table.sum(axis=1)
    table["comments_at_screening"] = seed.groupby("event_id")["comment_count_at_screening"].sum()
    return table.reindex(EVENT_ORDER)


def status_by_thing(documents: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(documents["exclusion_status"], documents["thing"], margins=True, margins_name="total")


def analysable_by_event(documents: pd.DataFrame) -> pd.DataFrame:
    comments = comments_only(documents)
    grouped = pd.DataFrame({
        "comments": comments.groupby("yt_video_event_id").size(),
        "strict_eligible": (comments["exclusion_status"] == "eligible").groupby(comments["yt_video_event_id"]).sum(),
        "inclusive_english": inclusive_mask(comments).groupby(comments["yt_video_event_id"]).sum(),
    }).reindex(EVENT_ORDER)
    grouped["strict_share"] = (grouped["strict_eligible"] / grouped["comments"]).round(3)
    grouped["inclusive_share"] = (grouped["inclusive_english"] / grouped["comments"]).round(3)
    return grouped


# Comments per day relative to the event each video covers (day 0 = event date), inclusive English mask.
def reaction_curve(documents: pd.DataFrame) -> pd.DataFrame:
    comments = comments_only(documents)
    comments = comments[inclusive_mask(comments)]
    days = comments["yt_days_from_video_event"]
    in_range = comments[(days >= -DAYS_BEFORE) & (days <= DAYS_AFTER)]
    curve = in_range.groupby(["yt_days_from_video_event", "yt_video_event_id"]).size().unstack(fill_value=0)
    return curve.reindex(index=range(-DAYS_BEFORE, DAYS_AFTER + 1), columns=EVENT_ORDER, fill_value=0)


def bypass_by_event(documents: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    comments = comments_only(documents)
    comments = comments[inclusive_mask(comments)]
    has_hit = comments["bypass_term_hits"].fillna("") != ""
    table = pd.DataFrame({
        "comments": comments.groupby("yt_video_event_id").size(),
        "with_bypass_terms": has_hit.groupby(comments["yt_video_event_id"]).sum(),
    }).reindex(EVENT_ORDER)
    table["share"] = (table["with_bypass_terms"] / table["comments"]).round(3)
    terms = comments.loc[has_hit, ["yt_video_event_id", "bypass_term_hits"]].assign(term=lambda frame: frame["bypass_term_hits"].str.split("|")).explode("term")
    table["top_terms"] = terms.groupby("yt_video_event_id")["term"].agg(lambda series: ", ".join(series.value_counts().head(top_n).index))
    return table


def network_readiness(manifests: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(manifests["edges"]["readiness_summary"]).set_index("yt_video_event_id").reindex(EVENT_ORDER)


# Light-mode tokens from the dataviz reference palette; one series per panel, so only categorical slot 1 is used.
SERIES_COLOR = "#2a78d6"
SURFACE, INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID_COLOR, AXIS_COLOR = "#e1e0d9", "#c3c2b7"
EVENT_LABELS = {
    "E1": "E1 · US Supreme Court upholds Texas law (placebo)",
    "E2": "E2 · UK Online Safety Act age checks enforced",
    "E3": "E3 · Australia under-16 social media ban",
    "E4": "E4 · UK House of Lords vote (placebo)",
    "E5": "E5 · Australia adult-content age checks",
}


# Small multiples with independent y-scales: event volumes differ tenfold, and one shared or dual axis would hide the small events.
def plot_reaction_curves(curve: pd.DataFrame):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, axes = plt.subplots(len(EVENT_ORDER), 1, figsize=(9, 11), sharex=True, facecolor=SURFACE)
    for ax, event_id in zip(axes, EVENT_ORDER):
        series = curve[event_id]
        ax.set_facecolor(SURFACE)
        ax.plot(series.index, series.values, color=SERIES_COLOR, linewidth=2, solid_joinstyle="round", solid_capstyle="round")
        ax.axvline(0, color=INK_SECONDARY, linewidth=1)
        # Label only the peak; the table view in the notebook carries every other value.
        peak_day, peak = int(series.idxmax()), int(series.max())
        ax.plot([peak_day], [peak], marker="o", markersize=8, color=SERIES_COLOR, markeredgecolor=SURFACE, markeredgewidth=2)
        to_left = peak_day > DAYS_AFTER * 0.6
        # Above-and-beside the peak: the line always falls away below it, so the label never crosses the curve.
        ax.annotate(f"peak {peak:,} on day {peak_day}", (peak_day, peak), xytext=(-8 if to_left else 8, 4), textcoords="offset points",
                    ha="right" if to_left else "left", va="bottom", fontsize=9, color=INK_SECONDARY)
        ax.set_title(f"{EVENT_LABELS[event_id]} — {int(series.sum()):,} comments", loc="left", fontsize=10, color=INK_PRIMARY)
        ax.set_ylim(bottom=0, top=max(peak * 1.15, 1))
        ax.grid(axis="y", color=GRID_COLOR, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS_COLOR)
        ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axes[-1].set_xlabel("Days from event (0 = event date)", color=INK_SECONDARY, fontsize=9)
    fig.suptitle("YouTube comments per day around each event", x=0.01, ha="left", fontsize=12, color=INK_PRIMARY)
    fig.text(0.01, 0.955, "Inclusive English comments on each event's videos; y-scales differ per panel.", fontsize=9, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
