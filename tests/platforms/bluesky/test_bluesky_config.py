# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

"""Offline tests for the Bluesky config loader and its use of the shared calendar.

The validation rules here are not style checks: each one fires on a mistake that
would produce a plausible-looking but wrong corpus. A config that loads happily
and quietly collects the wrong thing is the expensive failure.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.platforms.bluesky import build_manifest, score_frames
from src.platforms.bluesky.config import Config, PROJECT_ROOT, load_config


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config()


class TestEventCalendar:
    def test_events_come_from_the_shared_file(self, cfg):
        """This workstream must not keep its own calendar. A second list drifts,
        and a Bluesky series dated off one calendar cannot be compared with a
        YouTube series dated off another."""
        assert {e.id for e in cfg.events} == {"E1", "E2", "E3", "E4", "E5"}

    def test_e5_uses_the_sourced_date(self, cfg):
        """Our own config had 2026-03-11 marked 'DATE UNVERIFIED placeholder'.
        The shared calendar has 2026-03-09 with a primary source."""
        e5 = next(e for e in cfg.events if e.id == "E5")
        assert e5.date == date(2026, 3, 9)
        assert e5.source_url

    def test_the_did_design_survives_the_load(self, cfg):
        """`role` and `control_jurisdiction` encode the quasi-experiment. If they
        are dropped in loading, an analysis silently treats placebos as
        treatments -- which is how E4's weak response got read as a problem
        rather than as the expected result."""
        by_id = {e.id: e for e in cfg.events}
        assert by_id["E1"].role == "placebo"
        assert by_id["E4"].role == "placebo"
        assert by_id["E2"].role == "treatment"
        assert by_id["E2"].control_jurisdiction == "IE"
        assert by_id["E3"].control_jurisdiction == "NZ"

    def test_every_event_sits_inside_the_collection_window(self, cfg):
        for event in cfg.events:
            assert cfg.date_start <= event.date <= cfg.date_end


class TestQueryStrata:
    def test_strata_do_not_overlap(self, cfg):
        """A term in both strata would be reported as A by `stratum_of`, letting
        the biased supplement contaminate the neutral backbone."""
        assert not set(cfg.queries_a) & set(cfg.queries_b)

    def test_stratum_lookup_agrees_with_the_lists(self, cfg):
        for query in cfg.queries_a:
            assert cfg.stratum_of(query) == "A"
        for query in cfg.queries_b:
            assert cfg.stratum_of(query) == "B"

    def test_baseline_basket_is_present(self, cfg):
        """Without a denominator, volume cannot be separated from Bluesky's own
        activity trend -- which fell 38% across the window."""
        assert len(cfg.baseline_queries) >= 3


class TestGrid:
    def test_day_window_is_half_open(self, cfg):
        """`until` must be the next midnight, not 23:59:59: a closed upper bound
        drops every post in the final second of the day, on every day."""
        since, until = cfg.day_window(date(2025, 6, 27))
        assert since == "2025-06-27T00:00:00Z"
        assert until == "2025-06-28T00:00:00Z"

    def test_cell_count_matches_queries_times_days(self, cfg):
        assert len(cfg.cells()) == (len(cfg.queries_a) + len(cfg.queries_b)) * len(cfg.days())

    def test_days_are_contiguous_and_inclusive(self, cfg):
        days = cfg.days()
        assert days[0] == cfg.date_start and days[-1] == cfg.date_end
        assert (days[-1] - days[0]).days + 1 == len(days)


def test_platform_tools_use_repository_root_after_platform_move():
    assert PROJECT_ROOT == build_manifest.PROJECT_ROOT
    assert PROJECT_ROOT == score_frames.PROJECT_ROOT
    assert (PROJECT_ROOT / "config" / "bluesky" / "config.yaml").is_file()
