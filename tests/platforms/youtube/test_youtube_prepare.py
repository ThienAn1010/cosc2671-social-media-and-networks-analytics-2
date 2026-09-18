# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from lingua import Language

from src.shared.events import load_events
from src.platforms.youtube.prepare import PRIVATE_COLUMNS, build_documents, build_parser, run_prepare, validate_documents

LEXICONS = {
    "spam_patterns": ["telegram", "crypto"], "sensitive_terms": ["nudes"], "adult_platform_terms": ["pornhub"],
    "bypass_terms": ["vpn", "use a vpn", "change your region"],
}
SEED = [{"video_id": "v1", "event_id": "E3", "video_type": "news"}]


# Deterministic stand-in for lingua: French for "bonjour" texts, confident English otherwise.
class FakeDetector:
    def compute_language_confidence_values_in_parallel(self, texts):
        results = []
        for text in texts:
            if not text:
                results.append([])
            elif "bonjour" in text:
                results.append([SimpleNamespace(language=Language.FRENCH, value=0.97), SimpleNamespace(language=Language.ENGLISH, value=0.03)])
            else:
                results.append([SimpleNamespace(language=Language.ENGLISH, value=0.95), SimpleNamespace(language=Language.FRENCH, value=0.05)])
        return results


def comment(comment_id, text, author="UCauthor", published="2025-12-11T10:00:00Z", updated=None, parent=None):
    snippet = {
        "textOriginal": text, "textDisplay": text, "authorDisplayName": "@someone", "authorChannelId": {"value": author},
        "authorChannelUrl": "http://www.youtube.com/@someone", "authorProfileImageUrl": "http://img", "likeCount": 2,
        "publishedAt": published, "updatedAt": updated or published,
    }
    if parent:
        snippet["parentId"] = parent
    return {"id": comment_id, "snippet": snippet}


def thread(top, video_id="v1", total=0, inline=()):
    return {"id": top["id"], "snippet": {"videoId": video_id, "totalReplyCount": total, "topLevelComment": top}, "replies": {"comments": list(inline)}}


def video(video_id="v1", channel="UCchannel"):
    return {
        "id": video_id,
        "snippet": {"channelId": channel, "title": "Australia social media ban begins today", "description": "News report on the under-16 ban.", "publishedAt": "2025-12-09T00:00:00Z"},
        "statistics": {"likeCount": "10", "commentCount": "3"},
    }


def build(threads, replies=()):
    frame, counts = build_documents({"v1": video()}, list(threads), list(replies), SEED, load_events(), LEXICONS, "salt", FakeDetector(), "run-test")
    return frame.set_index("doc_id"), counts


LONG = " is a long enough English comment about the ban"


# One test per cleaning rule C1-C17 in docs/youtube/cleaning_decisions.md, so every documented rule is pinned to behaviour.
class PrepareRuleTests(unittest.TestCase):
    def test_c1_fetched_reply_replaces_embedded_copy(self):
        inline = comment("r1", "old embedded copy" + LONG, parent="c1")
        fetched = comment("r1", "full fetched copy" + LONG, parent="c1")
        frame, counts = build([thread(comment("c1", "top" + LONG), total=1, inline=[inline])], [{"collection": {"video_id": "v1"}, "data": fetched}])
        self.assertEqual(frame.loc["yt:reply:r1", "yt_source"], "comments.list")
        self.assertTrue(frame.loc["yt:reply:r1", "text_clean"].startswith("full fetched copy"))
        self.assertEqual(counts["inline_replies_superseded_by_comments_list"], 1)

    def test_c2_html_unescaped_and_zero_width_removed(self):
        frame, _ = build([thread(comment("c1", "Tom &amp; Jerry​" + LONG))])
        self.assertTrue(frame.loc["yt:comment:c1", "text_clean"].startswith("Tom & Jerry is"))

    def test_c3_empty_text(self):
        frame, _ = build([thread(comment("c1", "   "))])
        self.assertEqual(frame.loc["yt:comment:c1", "exclusion_status"], "empty")

    def test_c4_leading_handle_removed_and_handles_masked(self):
        reply = comment("r1", "@john.doe-1 you are wrong about @jane" + LONG, parent="c1")
        frame, _ = build([thread(comment("c1", "top" + LONG), total=1, inline=[reply])])
        row = frame.loc["yt:reply:r1"]
        self.assertTrue(row["has_leading_mention"])
        self.assertTrue(row["text_clean"].startswith("you are wrong about @user"))
        self.assertTrue(row["text_raw"].startswith("@user you are wrong"))
        self.assertNotIn("john", row["text_raw"] + row["text_clean"])

    def test_c5_urls_and_personal_data_masked_years_kept(self):
        frame, _ = build([
            thread(comment("c1", "email a@b.com or call +61 412 345 678 see https://x.com between 2025-2026" + LONG)),
            thread(comment("c2", "https://spam.example/offer", author="UCother")),
        ])
        text = frame.loc["yt:comment:c1", "text_clean"]
        self.assertEqual(text.count("<PII>"), 2)
        self.assertIn("<URL>", text)
        self.assertIn("2025-2026", text)
        self.assertEqual((frame.loc["yt:comment:c2", "exclusion_status"], frame.loc["yt:comment:c2", "spam_reason"]), ("spam", "url_only"))

    def test_c6_language_statuses(self):
        frame, _ = build([
            thread(comment("c1", "bonjour tout le monde ceci est un commentaire")),
            thread(comment("c2", "ok lol", author="UCb")),
            thread(comment("c3", "This" + LONG, author="UCc")),
        ])
        self.assertEqual(frame.loc["yt:comment:c1", "exclusion_status"], "non_english")
        self.assertEqual(frame.loc["yt:comment:c2", "exclusion_status"], "language_uncertain")
        self.assertEqual(frame.loc["yt:comment:c3", "exclusion_status"], "eligible")

    def test_c7_video_owner_comment(self):
        frame, _ = build([thread(comment("c1", "Thanks for watching" + LONG, author="UCchannel"))])
        self.assertEqual(frame.loc["yt:comment:c1", "exclusion_status"], "bot_or_owner")

    # D-016: a slogan repeated by many authors is counted but stays eligible; promotion with a link is spam.
    def test_c8_shared_slogans_counted_and_promo_spam(self):
        text = "It was never about the children"
        frame, _ = build([thread(comment(f"c{i}", text, author=f"UC{i}")) for i in range(3)] + [thread(comment("p1", "join my telegram https://t.me/x" + LONG, author="UCp"))])
        self.assertEqual((frame.loc["yt:comment:c0", "exclusion_status"], frame.loc["yt:comment:c0", "copy_paste_author_count"]), ("eligible", 3))
        self.assertEqual(frame.loc["yt:comment:p1", "exclusion_status"], "spam")
        self.assertIn("promo_with_contact", frame.loc["yt:comment:p1", "spam_reason"])

    def test_c9_same_author_repeat_is_duplicate_first_kept(self):
        text = "Same author posting the same long comment again"
        frame, _ = build([thread(comment("c1", text, published="2025-12-11T10:00:00Z")), thread(comment("c2", text, published="2025-12-11T11:00:00Z"))])
        self.assertEqual(frame.loc["yt:comment:c1", "exclusion_status"], "eligible")
        self.assertEqual(frame.loc["yt:comment:c2", "exclusion_status"], "duplicate")

    # D-014: discussing adult platforms or grooming ("ask kids for nudes") is policy discourse; sharing a link is sensitive.
    def test_c10_sensitive_content(self):
        frame, _ = build([
            thread(comment("c1", "neither Roblox nor Pornhub are included in the ban" + LONG)),
            thread(comment("c2", "watch it here pornhub https://example.test/v" + LONG, author="UCb")),
            thread(comment("c3", "predators ask kids for nudes" + LONG, author="UCc")),
            thread(comment("c4", "free nudes https://example.test/n" + LONG, author="UCd")),
        ])
        self.assertEqual(frame.loc["yt:comment:c1", "exclusion_status"], "eligible")
        self.assertEqual(frame.loc["yt:comment:c2", "exclusion_status"], "sensitive")
        self.assertEqual(frame.loc["yt:comment:c3", "exclusion_status"], "eligible")
        self.assertEqual(frame.loc["yt:comment:c4", "exclusion_status"], "sensitive")

    # Screened news videos often link out while naming the platform in the description; that is not a sensitive comment.
    def test_c10_video_description_links_are_not_sensitive(self):
        news_video = video()
        news_video["snippet"]["description"] = "Pornhub blocks Australia under new codes. More: https://example.test/story"
        frame, _ = build_documents({"v1": news_video}, [], [], SEED, load_events(), LEXICONS, "salt", FakeDetector(), "run-test")
        self.assertFalse(frame.set_index("doc_id").loc["yt:video:v1", "is_sensitive_content"])

    def test_c11_edited_flag(self):
        frame, _ = build([thread(comment("c1", "Edited" + LONG, updated="2025-12-12T00:00:00Z"))])
        self.assertTrue(frame.loc["yt:comment:c1", "yt_is_edited"])

    def test_c12_event_window_edges(self):
        frame, _ = build([
            thread(comment("c1", "First" + LONG, published="2025-12-03T00:00:00Z")),
            thread(comment("c2", "Second" + LONG, author="UCb", published="2025-12-31T00:00:00Z")),
        ])
        self.assertEqual((frame.loc["yt:comment:c1", "event_window"], frame.loc["yt:comment:c1", "days_from_event"]), ("E3", -7))
        self.assertEqual(frame.loc["yt:comment:c2", "event_window"], "none")

    def test_c13_late_comment_keeps_video_event(self):
        frame, _ = build([thread(comment("c1", "Late" + LONG, published="2026-04-01T00:00:00Z"))])
        row = frame.loc["yt:comment:c1"]
        self.assertEqual((row["event_id_nearest"], row["yt_video_event_id"], row["yt_days_from_video_event"]), ("E5", "E3", 112))
        self.assertTrue(row["yt_is_late_comment"])

    def test_c14_jurisdiction_from_discovery_event(self):
        frame, _ = build([thread(comment("c1", "Hint" + LONG))])
        self.assertEqual((frame.loc["yt:comment:c1", "jurisdiction_hint"], frame.loc["yt:comment:c1", "jurisdiction_source"]), ("AU", "discovery_query"))

    def test_c15_bypass_terms(self):
        frame, _ = build([thread(comment("c1", "Kids just use a VPN or change your region" + LONG))])
        self.assertEqual(frame.loc["yt:comment:c1", "bypass_term_hits"], "change your region|use a vpn")

    def test_c16_pseudonymized_without_private_fields(self):
        frame, _ = build([thread(comment("c1", "Privacy" + LONG))])
        self.assertFalse(PRIVATE_COLUMNS & set(frame.columns))
        self.assertEqual(len(frame.loc["yt:comment:c1", "author_hash"]), 16)
        self.assertFalse(frame.reset_index().astype(str).apply(lambda column: column.str.contains("someone")).any().any())

    def test_c17_video_is_a_root_document(self):
        frame, _ = build([thread(comment("c1", "Root" + LONG))])
        row = frame.loc["yt:video:v1"]
        self.assertEqual((row["thing"], row["root_doc_id"], row["parent_doc_id"]), ("video", "yt:video:v1", ""))
        self.assertEqual(frame.loc["yt:comment:c1", "parent_doc_id"], "yt:video:v1")

    def test_validation_rejects_leaked_handle(self):
        frame, _ = build([thread(comment("c1", "Fine" + LONG))])
        frame = frame.reset_index()
        frame.loc[0, "text_raw"] = "hello @realperson"
        with self.assertRaises(ValueError):
            validate_documents(frame)


class PrepareRunTests(unittest.TestCase):
    def test_run_requires_complete_collection_unless_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "youtube_run"
            run.mkdir()
            (run / "videos.jsonl").write_text(json.dumps({"collection": {}, "data": video()}) + "\n", encoding="utf-8")
            (run / "comment_threads.jsonl").write_text(json.dumps({"collection": {}, "data": thread(comment("c1", "Run" + LONG))}) + "\n", encoding="utf-8")
            (run / "replies.jsonl").write_text("", encoding="utf-8")
            (run / "collection_manifest.json").write_text(json.dumps({"completion_status": "partial"}), encoding="utf-8")
            seed = root / "seed.csv"
            seed.write_text("video_id,event_id,video_type\nv1,E3,news\n", encoding="utf-8")
            lexicons = root / "lexicons.json"
            lexicons.write_text(json.dumps(LEXICONS), encoding="utf-8")
            base = ["--collect-run", str(run), "--seed", str(seed), "--lexicons", str(lexicons), "--out-root", str(root / "out")]

            with self.assertRaises(SystemExit):
                run_prepare(build_parser().parse_args(base), salt="salt", detector=FakeDetector())
            frame, manifest = run_prepare(build_parser().parse_args(base + ["--allow-partial"]), salt="salt", detector=FakeDetector())

            self.assertEqual(len(pd.read_parquet(root / "out" / "documents.parquet")), len(frame))
            self.assertEqual(manifest["counts"]["by_thing"], {"video": 1, "comment": 1})
            self.assertEqual(manifest["input"]["collection_completion_status"], "partial")


if __name__ == "__main__":
    unittest.main()
