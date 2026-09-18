# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

import unittest

import pandas as pd

from src.platforms.youtube.summary import analysable_by_event, bypass_by_event, reaction_curve, status_by_thing


def row(thing, status="eligible", english=True, event="E2", day=0, bypass=""):
    return {"thing": thing, "exclusion_status": status, "is_english": english, "yt_video_event_id": event,
            "yt_days_from_video_event": day, "bypass_term_hits": bypass}


# Notebook numbers are copied into the report, so the summary arithmetic is pinned here.
class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.documents = pd.DataFrame([
            row("video"),
            row("comment", day=-8),
            row("comment", day=0, bypass="vpn|use a vpn"),
            row("reply", status="language_uncertain", day=1, bypass="vpn"),
            row("reply", status="language_uncertain", english=False, day=2),
            row("comment", status="spam", day=3),
        ])

    def test_strict_and_inclusive_counts(self):
        e2 = analysable_by_event(self.documents).loc["E2"]
        self.assertEqual((e2["comments"], e2["strict_eligible"], e2["inclusive_english"]), (5, 2, 3))

    def test_reaction_curve_keeps_window_and_inclusive_rows(self):
        curve = reaction_curve(self.documents)
        self.assertEqual(curve.loc[0, "E2"] + curve.loc[1, "E2"], 2)
        self.assertEqual(int(curve["E2"].sum()), 2)
        self.assertNotIn(-8, curve.index)

    def test_bypass_share_and_terms(self):
        e2 = bypass_by_event(self.documents).loc["E2"]
        self.assertEqual((e2["comments"], e2["with_bypass_terms"]), (3, 2))
        self.assertTrue(e2["top_terms"].startswith("vpn"))

    def test_reaction_plot_has_one_panel_per_event(self):
        import matplotlib

        matplotlib.use("Agg")
        from src.platforms.youtube.summary import plot_reaction_curves

        figure = plot_reaction_curves(reaction_curve(self.documents))
        self.assertEqual(len(figure.axes), 5)

    def test_status_table_has_totals(self):
        table = status_by_thing(self.documents)
        self.assertEqual(table.loc["total", "total"], 6)


if __name__ == "__main__":
    unittest.main()
