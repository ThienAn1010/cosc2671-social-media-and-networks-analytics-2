# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.platforms.youtube.sample import sample_tables, write_sample


def doc(doc_id, thing, root="yt:video:v1", parent="", status="eligible", event="E3", video_type="news", author="h1"):
    return {"doc_id": doc_id, "thing": thing, "root_doc_id": root, "parent_doc_id": parent, "exclusion_status": status,
            "yt_video_event_id": event, "yt_video_type": video_type, "author_hash": author, "created_utc": "2025-12-11T00:00:00Z",
            "container_id": "yt:channel:c"}


# The sample must keep whole reply trees and never carry sensitive rows or dangling edges.
class SampleTests(unittest.TestCase):
    def setUp(self):
        rows = [doc("yt:video:v1", "video")]
        for i in range(4):
            rows.append(doc(f"yt:comment:c{i}", "comment", author=f"h{i}"))
            rows.append(doc(f"yt:reply:r{i}", "reply", parent=f"yt:comment:c{i}", author="hr"))
        rows.append(doc("yt:comment:bad", "comment", status="sensitive"))
        self.documents = pd.DataFrame(rows)
        self.interactions = pd.DataFrame([
            {"source_doc_id": f"yt:reply:r{i}", "target_doc_id": f"yt:comment:c{i}"} for i in range(4)
        ])

    def test_threads_keep_their_replies_and_edges(self):
        tables = sample_tables(self.documents, self.interactions, threads_per_stratum=2)
        docs = tables["documents"]
        comments = set(docs.loc[docs["thing"] == "comment", "doc_id"])
        self.assertEqual(len(comments), 2)
        self.assertEqual(set(docs.loc[docs["thing"] == "reply", "parent_doc_id"]), comments)
        self.assertTrue(set(tables["interactions"]["source_doc_id"]) <= set(docs["doc_id"]))
        self.assertNotIn("sensitive", set(docs["exclusion_status"]))
        self.assertIn("yt:video:v1", set(docs["doc_id"]))
        # Regression: sampled comments must keep their stratum columns.
        self.assertEqual(set(docs.loc[docs["thing"] == "comment", "yt_video_event_id"]), {"E3"})
        self.assertEqual(set(docs.loc[docs["thing"] == "comment", "yt_video_type"]), {"news"})

    def test_written_sample_reports_size_under_limit(self):
        tables = sample_tables(self.documents, self.interactions, threads_per_stratum=2)
        with tempfile.TemporaryDirectory() as tmp:
            size = write_sample(tables, Path(tmp), threads_per_stratum=2)
            self.assertTrue((Path(tmp) / "README.md").exists())
        self.assertLess(size, 10 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
