# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

import unittest

import pandas as pd

from src.platforms.youtube.screen import MAX_PER_EVENT, accept_suggestions, freeze, suggest

PATTERNS = {"official": ["parliament"], "news": ["news"], "tech_explainer": ["tech"]}


def candidate(video_id, channel, views, flags=""):
    return {
        "video_id": video_id, "event_id": "E2", "title": "t", "channel_title": channel, "channel_id": "c",
        "published_at": "2025-07-26T00:00:00Z", "view_count": views, "comment_count": 100, "auto_flags": flags,
        "include_suggested": "", "video_type_suggested": "", "reason_suggested": "", "include": "", "video_type": "", "reviewer_note": "",
    }


# Screening rules must match the configured sampling frame exactly.
class ScreenTests(unittest.TestCase):
    def test_type_quota_keeps_small_channel_types_before_filling_by_views(self):
        rows = [candidate(f"n{i}", "Big News", 10_000 + i) for i in range(60)]
        rows += [candidate(f"t{i}", "Tiny Tech", i) for i in range(6)]
        rows.append(candidate("flagged", "Big News", 99_999, flags="low_comments"))
        frame = suggest(pd.DataFrame(rows), PATTERNS).set_index("video_id")
        chosen = frame[frame["include_suggested"] == "y"]
        self.assertEqual(len(chosen), MAX_PER_EVENT)
        self.assertEqual((chosen["video_type_suggested"] == "tech_explainer").sum(), 5)
        self.assertEqual(frame.loc["flagged", "include_suggested"], "n")

    def test_freeze_blocks_included_rows_without_valid_type(self):
        frame = pd.DataFrame([candidate("a", "News", 10), candidate("b", "News", 5)])
        frame.loc[0, ["include", "video_type"]] = ["y", "news"]
        frame.loc[1, ["include", "video_type"]] = ["y", ""]
        _, problems, warnings = freeze(frame, "kien")
        self.assertTrue(any("video_type" in problem for problem in problems))
        self.assertTrue(warnings)

    def test_accept_suggestions_only_fills_blanks(self):
        frame = pd.DataFrame([candidate("a", "News", 10)])
        frame.loc[0, ["include_suggested", "video_type_suggested", "include"]] = ["y", "news", "n"]
        accepted = accept_suggestions(frame)
        self.assertEqual((accepted.loc[0, "include"], accepted.loc[0, "video_type"]), ("n", "news"))

    # Regression: an all-blank include column read from CSV arrives as float NaN.
    def test_accept_suggestions_handles_all_blank_columns_read_as_nan(self):
        frame = pd.DataFrame([candidate("a", "News", 10), candidate("b", "News", 5)])
        frame["include"] = float("nan")
        frame["video_type"] = float("nan")
        frame["include_suggested"] = ["y", "n"]
        frame["video_type_suggested"] = ["news", "news"]
        accepted = accept_suggestions(frame)
        self.assertEqual(accepted["include"].tolist(), ["y", "n"])
        seed, problems, _ = freeze(accepted, "kien")
        self.assertEqual((len(seed), problems), (1, []))


if __name__ == "__main__":
    unittest.main()
