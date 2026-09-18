# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fake_youtube import FakeYouTube, fake_reply, fake_thread, fake_video

from src.platforms.youtube.collect import build_parser, run_collection, threads_needing_replies
from src.platforms.youtube.storage import read_jsonl


def videos_handler(params):
    return {"items": [fake_video(video_id) for video_id in params["id"].split(",")]}


# The collector must never lose or duplicate comments across quota stops and resumes.
class CollectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.seed = self.root / "seed.csv"
        self.seed.write_text("video_id,event_id,comment_count_at_screening\nv1,E2,300\nv2,E2,50\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, *extra):
        return build_parser().parse_args(["--seed", str(self.seed), "--out-root", str(self.root / "raw"), *extra])

    def thread_ids(self, run_dir):
        return [row["data"]["id"] for row in read_jsonl(run_dir / "comment_threads.jsonl")]

    def test_threads_pagination_and_comments_disabled(self):
        def threads(params):
            if params["videoId"] == "v2":
                return 403, "commentsDisabled"
            if params.get("pageToken") == "p2":
                return {"items": [fake_thread("t3", "v1")]}
            return {"items": [fake_thread("t1", "v1"), fake_thread("t2", "v1")], "nextPageToken": "p2"}

        fake = FakeYouTube({"videos": videos_handler, "commentThreads": threads})
        with patch("src.platforms.youtube.api.urlopen", side_effect=fake):
            run_dir, manifest = run_collection(self.args("--phase", "threads"), api_key="test-key")
        self.assertEqual(manifest["videos"]["v1"]["threads_status"], "complete")
        self.assertEqual(manifest["videos"]["v1"]["threads_written"], 3)
        self.assertEqual(manifest["videos"]["v2"]["threads_status"], "comments_disabled")
        self.assertEqual(self.thread_ids(run_dir), ["t1", "t2", "t3"])
        # Replies phase has not run yet, so the collection cannot be complete.
        self.assertEqual(manifest["completion_status"], "partial")

    def test_quota_stop_then_resume_without_duplicates(self):
        state = {"quota": True}

        def threads(params):
            if params["videoId"] == "v2":
                return {"items": [fake_thread("t4", "v2")]}
            if params.get("pageToken") == "p2":
                return (403, "quotaExceeded") if state["quota"] else {"items": [fake_thread("t3", "v1")]}
            return {"items": [fake_thread("t1", "v1"), fake_thread("t2", "v1")], "nextPageToken": "p2"}

        fake = FakeYouTube({"videos": videos_handler, "commentThreads": threads})
        with patch("src.platforms.youtube.api.urlopen", side_effect=fake):
            run_dir, manifest = run_collection(self.args("--phase", "threads"), api_key="test-key")
            self.assertEqual((manifest["stop_reason"], manifest["videos"]["v1"]["threads_status"]), ("quota", "partial"))
            state["quota"] = False
            _, manifest = run_collection(self.args("--phase", "threads", "--resume", str(run_dir)), api_key="test-key")
        self.assertEqual(manifest["videos"]["v1"]["threads_status"], "complete")
        ids = self.thread_ids(run_dir)
        self.assertEqual(sorted(ids), ["t1", "t2", "t3", "t4"])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(manifest["sessions"]), 2)

    def test_replies_largest_threads_first_and_done_threads_skipped(self):
        def threads(params):
            if params["videoId"] == "v2":
                return {"items": []}
            return {"items": [fake_thread("t1", "v1", total_replies=150, inline=5), fake_thread("t2", "v1", 3, 3), fake_thread("t3", "v1", 20, 5)]}

        def comments(params):
            if params["parentId"] == "t1" and not params.get("pageToken"):
                return {"items": [fake_reply("r1", "t1")], "nextPageToken": "x"}
            if params["parentId"] == "t1":
                return {"items": [fake_reply("r2", "t1")]}
            return {"items": [fake_reply("r3", params["parentId"])]}

        fake = FakeYouTube({"videos": videos_handler, "commentThreads": threads, "comments": comments})
        with patch("src.platforms.youtube.api.urlopen", side_effect=fake):
            run_dir, manifest = run_collection(self.args("--phase", "all"), api_key="test-key")
            self.assertEqual([params["parentId"] for params in fake.calls_to("comments")], ["t1", "t1", "t3"])
            self.assertEqual(manifest["completion_status"], "complete")
            self.assertEqual(manifest["replies"]["status_counts"], {"complete": 2})
            _, manifest = run_collection(self.args("--phase", "replies", "--resume", str(run_dir)), api_key="test-key")
        self.assertEqual(len(fake.calls_to("comments")), 3)
        self.assertEqual(manifest["replies"]["replies_on_disk"], 3)

    def test_threads_needing_replies_ignores_complete_threads_and_other_videos(self):
        threads = [fake_thread("a", "v1", 2, 2), fake_thread("b", "v1", 9, 1), fake_thread("c", "zz", 50, 0)]
        self.assertEqual(threads_needing_replies(threads, {"v1"}), [("b", "v1", 9)])


if __name__ == "__main__":
    unittest.main()
