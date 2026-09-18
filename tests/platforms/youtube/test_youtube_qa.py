# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

import tempfile
import unittest
from pathlib import Path

from src.platforms.youtube.qa import g2_coverage, g3_duplicates, g4_reply_parents, g5_timestamps, g6_reply_reconciliation, g8_secret_scan, render_markdown, g1_manifest_status, g7_spot_check_sample


def top(comment_id, published="2025-12-11T00:00:00Z"):
    return {"id": comment_id, "snippet": {"publishedAt": published, "textOriginal": "@someone hello"}}


def reply(reply_id, parent, published="2025-12-11T01:00:00Z"):
    return {"id": reply_id, "snippet": {"parentId": parent, "publishedAt": published, "textOriginal": "reply"}}


def make_run():
    # v1 reports 5 comments: 2 threads + 3 unique replies (r1 is both embedded and fetched).
    threads = [
        {"id": "t1", "snippet": {"videoId": "v1", "totalReplyCount": 2, "topLevelComment": top("t1")}, "replies": {"comments": [reply("r1", "t1")]}},
        {"id": "t2", "snippet": {"videoId": "v1", "totalReplyCount": 4, "topLevelComment": top("t2")}, "replies": {"comments": []}},
    ]
    fetched = [{"collection": {"video_id": "v1"}, "data": reply(rid, parent)} for rid, parent in (("r1", "t1"), ("r2", "t1"), ("r3", "t2"))]
    return {
        "manifest": {"completion_status": "complete", "stop_reason": "", "videos": {"v1": {"event_id": "E3", "threads_status": "complete"}},
                     "replies": {"status_counts": {"complete": 2}}, "sessions": [{"started_at_utc": "x", "usage": {"units": 7}, "stop_reason": ""}]},
        "videos": [{"id": "v1", "snippet": {"publishedAt": "2025-12-10T00:00:00Z"}, "statistics": {"commentCount": "5"}}],
        "threads": threads,
        "replies": fetched,
    }


# QA numbers go into the report, so each gate's arithmetic is pinned on a tiny hand-checkable run.
class QaGateTests(unittest.TestCase):
    def test_coverage_counts_each_reply_once(self):
        row = g2_coverage(make_run()).iloc[0]
        self.assertEqual((row["threads"], row["replies"], row["collected"], row["coverage"]), (2, 3, 5, 1.0))

    def test_duplicates_and_parents(self):
        run = make_run()
        self.assertEqual(g3_duplicates(run)["inline_replies_also_fetched"], 1)
        self.assertEqual(g4_reply_parents(run)["missing_parent"], 0)

    def test_reply_reconciliation_flags_large_gap(self):
        result = g6_reply_reconciliation(make_run())
        # t1: 2 of 2 collected; t2: 1 of 4 collected -> one thread with a gap above 20%.
        self.assertEqual((result["expected_replies"], result["collected_replies"], result["threads_gap_over_20pct"]), (6, 3, 1))

    def test_timestamps_before_video_publish_are_counted(self):
        run = make_run()
        run["threads"][0]["snippet"]["topLevelComment"]["snippet"]["publishedAt"] = "2025-12-01T00:00:00Z"
        self.assertEqual(g5_timestamps(run)["before_video_publish"], 1)

    def test_secret_scan_finds_key_pattern(self):
        fake_key = "AI" + "za" + "x" * 35
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "clean.txt").write_text("nothing here", encoding="utf-8")
            (Path(tmp) / "leak.txt").write_text(f"key={fake_key}", encoding="utf-8")
            self.assertEqual([Path(hit).name for hit in g8_secret_scan([Path(tmp)])], ["leak.txt"])

    def test_spot_check_masks_handles_and_report_renders(self):
        run = make_run()
        sample = g7_spot_check_sample(run, size=3)
        self.assertFalse(sample["text_preview"].str.contains("someone").any())
        results = {"run_id": "run", "g1": g1_manifest_status(run), "g2": g2_coverage(run), "g3": g3_duplicates(run), "g4": g4_reply_parents(run),
                   "g5": g5_timestamps(run), "g6": g6_reply_reconciliation(run), "g7": sample, "g8": []}
        text = render_markdown(results, Path("/tmp/spot.csv"))
        self.assertIn("| G6 |", text)
        self.assertIn("FLAG", text)


if __name__ == "__main__":
    unittest.main()
