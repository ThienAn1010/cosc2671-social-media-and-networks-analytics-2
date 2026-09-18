# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

import unittest

import pandas as pd

from src.platforms.youtube.edges import build_author_video, build_interactions, readiness_summary, resolve_thread, validate_tables


def entry(doc_id, author, handle, minute, text="plain reply"):
    return {"doc_id": doc_id, "author_id": author, "handle": handle, "published_at": f"2025-12-11T10:{minute:02d}:00Z", "text": text}


def raw_comment(comment_id, author, handle, minute, text, parent=None):
    snippet = {"authorChannelId": {"value": author}, "authorDisplayName": f"@{handle}", "publishedAt": f"2025-12-11T10:{minute:02d}:00Z", "textOriginal": text}
    if parent:
        snippet["parentId"] = parent
    return {"id": comment_id, "snippet": snippet}


# Reply targets decide who points at whom in the network, so each resolution path is tested.
class ResolveThreadTests(unittest.TestCase):
    def setUp(self):
        self.top = entry("yt:comment:t1", "UCtop", "alice", 0)

    def targets(self, replies):
        return [(edge["source"]["doc_id"], edge["target"]["doc_id"], edge["target_resolution"]) for edge in resolve_thread(self.top, replies)]

    def test_leading_handle_points_to_earlier_replier(self):
        replies = [entry("yt:reply:r1", "UCbob", "bob", 1), entry("yt:reply:r2", "UCcarol", "carol", 2, "@bob no, that is wrong")]
        self.assertEqual(self.targets(replies)[1], ("yt:reply:r2", "yt:reply:r1", "handle_match"))

    def test_no_handle_or_unknown_handle_goes_to_thread_root(self):
        replies = [entry("yt:reply:r1", "UCbob", "bob", 1), entry("yt:reply:r2", "UCcarol", "carol", 2, "@nobody hello")]
        self.assertEqual([target for _, target, _ in self.targets(replies)], ["yt:comment:t1", "yt:comment:t1"])
        self.assertEqual(self.targets(replies)[1][2], "thread_root")

    def test_handle_after_zero_width_space_is_read(self):
        replies = [entry("yt:reply:r1", "UCbob", "bob", 1), entry("yt:reply:r2", "UCcarol", "carol", 2, "​@bob agreed")]
        self.assertEqual(self.targets(replies)[1][2], "handle_match")

    def test_handle_used_by_two_people_resolves_to_most_recent(self):
        replies = [entry("yt:reply:r1", "UCbob1", "bob", 1), entry("yt:reply:r2", "UCbob2", "bob", 2), entry("yt:reply:r3", "UCcarol", "carol", 3, "@bob why")]
        self.assertEqual(self.targets(replies)[2][1], "yt:reply:r2")

    def test_reply_order_follows_time_not_input_order(self):
        replies = [entry("yt:reply:r2", "UCcarol", "carol", 2, "@bob ok"), entry("yt:reply:r1", "UCbob", "bob", 1)]
        self.assertEqual(self.targets(replies)[1], ("yt:reply:r2", "yt:reply:r1", "handle_match"))


class TableTests(unittest.TestCase):
    def setUp(self):
        top = raw_comment("t1", "UCalice", "alice", 0, "top level")
        self.threads = [{"id": "t1", "snippet": {"videoId": "v1", "topLevelComment": top}, "replies": {"comments": [raw_comment("r1", "UCbob", "bob", 1, "hi", parent="t1")]}}]
        self.replies = [{"collection": {"video_id": "v1"}, "data": raw_comment("r2", "UCalice", "alice", 2, "@bob thanks", parent="t1")}]
        self.videos = {"v1": {"id": "v1", "snippet": {"channelId": "UCchannel"}}}
        base = {"created_utc": "2025-12-11T10:00:00Z", "event_id_nearest": "E3", "event_window": "E3", "yt_video_event_id": "E3",
                "yt_video_type": "news", "exclusion_status": "eligible", "root_doc_id": "yt:video:v1", "container_id": "yt:channel:UCchannel"}
        self.documents = pd.DataFrame([
            {**base, "doc_id": "yt:video:v1", "thing": "video", "author_hash": "hv"},
            {**base, "doc_id": "yt:comment:t1", "thing": "comment", "author_hash": "ha"},
            {**base, "doc_id": "yt:reply:r1", "thing": "reply", "author_hash": "hb"},
            {**base, "doc_id": "yt:reply:r2", "thing": "reply", "author_hash": "ha", "exclusion_status": "language_uncertain"},
        ])

    def test_one_edge_per_reply_with_self_loop_flag_and_summary(self):
        interactions = build_interactions(self.threads, self.replies, self.videos, self.documents, "salt")
        author_video = build_author_video(self.documents)
        validate_tables(interactions, author_video, self.documents)
        edges = interactions.set_index("source_doc_id")
        self.assertEqual(edges.loc["yt:reply:r2", "target_doc_id"], "yt:reply:r1")
        self.assertFalse(edges["is_self_loop"].any())
        self.assertEqual(edges.loc["yt:reply:r2", "source_exclusion_status"], "language_uncertain")
        summary = readiness_summary(interactions).iloc[0]
        # alice -> bob (r2) and bob -> alice (r1) form one reciprocal pair.
        self.assertEqual((summary["nodes"], summary["directed_pairs"], summary["reciprocity"]), (2, 2, 1.0))
        self.assertEqual(author_video.set_index("author_hash").loc["ha", "n_comments"], 2)

    def test_validation_rejects_missing_reply_edge(self):
        interactions = build_interactions(self.threads, self.replies, self.videos, self.documents, "salt").iloc[:1]
        with self.assertRaises(ValueError):
            validate_tables(interactions, build_author_video(self.documents), self.documents)


if __name__ == "__main__":
    unittest.main()
