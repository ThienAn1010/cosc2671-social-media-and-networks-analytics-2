# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fake_youtube import FakeYouTube, fake_video

from src.shared.events import load_events
from src.platforms.youtube.discover import build_candidates, build_parser, parse_duration_seconds, plan_search_tasks, run_discovery, search_params

QUERIES = {
    "defaults": {"type": "video", "order": "viewCount", "relevanceLanguage": "en", "maxResults": 50},
    "scopes": ["region", "global"],
    "events": {"E2": {"region": "GB", "queries": [{"query_id": "E2-q1", "q": "Online Safety Act age verification"}]}},
}
LEXICONS = {"topic_keywords": ["online safety act", "age verification"]}


def search_item(video_id):
    return {"id": {"kind": "youtube#video", "videoId": video_id}, "snippet": {"title": "t"}}


# Discovery decides the sampling frame, so the window, scope and flag rules are pinned down here.
class DiscoverTests(unittest.TestCase):
    def setUp(self):
        self.events = load_events()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "queries.json").write_text(json.dumps(QUERIES), encoding="utf-8")
        (self.root / "lexicons.json").write_text(json.dumps(LEXICONS), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, *extra):
        return build_parser().parse_args([
            "--queries", str(self.root / "queries.json"), "--lexicons", str(self.root / "lexicons.json"),
            "--out-root", str(self.root / "raw"), "--screening-root", str(self.root / "screening"), *extra,
        ])

    def test_plan_runs_each_query_in_region_and_global_scope(self):
        tasks = plan_search_tasks(QUERIES)
        self.assertEqual([(task.scope, task.region_code) for task in tasks], [("region", "GB"), ("global", None)])
        self.assertEqual(plan_search_tasks(QUERIES, ["E3"]), [])

    def test_search_params_use_event_window_and_region_only_for_region_scope(self):
        region_task, global_task = plan_search_tasks(QUERIES)
        e2 = next(event for event in self.events if event.event_id == "E2")
        params = search_params(region_task, e2, QUERIES["defaults"])
        self.assertEqual((params["publishedAfter"], params["publishedBefore"]), ("2025-07-18T00:00:00Z", "2025-08-15T00:00:00Z"))
        self.assertEqual(params["regionCode"], "GB")
        self.assertNotIn("regionCode", search_params(global_task, e2, QUERIES["defaults"]))

    def test_parse_duration_seconds(self):
        self.assertEqual(parse_duration_seconds("PT1H2M3S"), 3723)
        self.assertIsNone(parse_duration_seconds(None))

    def test_build_candidates_flags_low_comments_off_topic_and_out_of_window(self):
        rows = [{"collection": {"event_id": "E2", "query_id": "E2-q1", "scope": "region"}, "data": search_item(vid)} for vid in ("ok", "few", "off", "late")]
        videos = [
            fake_video("ok", comment_count=500),
            fake_video("few", comment_count=5),
            fake_video("off", comment_count=500, title="Cooking pasta"),
            fake_video("late", comment_count=500, published_at="2025-09-30T00:00:00Z"),
        ]
        frame = build_candidates(rows, videos, self.events, LEXICONS["topic_keywords"]).set_index("video_id")
        self.assertEqual(frame.loc["ok", "auto_flags"], "")
        self.assertEqual(frame.loc["few", "auto_flags"], "low_comments")
        self.assertEqual(frame.loc["off", "auto_flags"], "no_topic_keyword")
        self.assertIn("out_of_window", frame.loc["late", "auto_flags"])

    def test_run_discovery_offline_end_to_end(self):
        fake = FakeYouTube({
            "search": lambda params: {"items": [search_item("v1"), search_item("v2")]},
            "videos": lambda params: {"items": [fake_video(vid, comment_count=500 if vid == "v1" else 3) for vid in params["id"].split(",")]},
        })
        with patch("src.platforms.youtube.api.urlopen", side_effect=fake):
            run_dir, manifest, _ = run_discovery(self.args(), api_key="test-key")
        self.assertEqual(manifest["completion_status"], "complete")
        self.assertEqual(manifest["sessions"][-1]["usage"]["search_calls"], 2)
        self.assertEqual(len(fake.calls_to("videos")), 1)
        sheet = pd.read_csv(self.root / "screening" / f"candidates_{run_dir.name}.csv", keep_default_na=False)
        self.assertEqual(sorted(sheet["video_id"]), ["v1", "v2"])
        self.assertEqual(sheet.set_index("video_id").loc["v2", "auto_flags"], "low_comments")

    def test_quota_on_search_marks_run_partial(self):
        fake = FakeYouTube({"search": lambda params: (403, "quotaExceeded")})
        with patch("src.platforms.youtube.api.urlopen", side_effect=fake):
            _, manifest, _ = run_discovery(self.args(), api_key="test-key")
        self.assertEqual((manifest["completion_status"], manifest["stop_reason"]), ("partial", "quota"))


if __name__ == "__main__":
    unittest.main()
