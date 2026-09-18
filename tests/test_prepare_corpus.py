# Student name: Nguyen Hoang Thien An
# Student ID: s3825455
# Assignment 1 - Social Media and Networks Analytics

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.prepare_corpus import (
    add_corpus_flags,
    apply_language_overrides,
    build_language_detector,
    build_stopwords,
    finalize_analysis_masks,
    load_language_overrides,
    load_spacy_model,
    require_complete_collection_manifest,
    unwrap_comment,
    unwrap_post,
    write_submission_sample,
    write_outputs,
)
from src.prepare_corpus_records import research_window_from_manifest
from src.collect_arctic_shift_core import EVENT_WINDOWS
from src.prepare_corpus_text import (
    classify_relevance,
    classify_relevance_detail,
    clean_text_for_sentiment,
    clean_text_light,
    topic_tokens,
)


def build_post_wrapper(**data):
    base = {
        "id": "abc",
        "created_utc": 1731196800,
        "subreddit": "privacy",
        "author": "human_user",
        "author_fullname": "t2_human",
        "author_flair_text": "learner",
        "distinguished": None,
        "title": "Age assurance backlash",
        "selftext": "Privacy protections are being replaced by surveillance",
        "score": 12,
        "num_comments": 3,
        "url": "https://reddit.test/post",
    }
    base.update(data)
    event_id, event = next(iter(EVENT_WINDOWS.items()))
    return {
        "collection": {
            "source_corpus": "reddit",
            "subreddit": "privacy",
            "event_id": event_id,
            "jurisdiction": event["jurisdiction"],
            "window_start_utc": event["start_utc"],
            "window_end_utc": event["end_utc"],
            "query": "age verification",
        },
        "data": base,
    }


def build_comment_wrapper(**data):
    base = {
        "id": "def",
        "created_utc": 1731196800,
        "subreddit": "privacy",
        "author": "commenter",
        "author_fullname": "t2_commenter",
        "author_flair_text": "",
        "distinguished": None,
        "body": "Age checks are invasive",
        "score": 1,
        "link_id": "t3_abc",
        "parent_id": "t3_abc",
    }
    base.update(data)
    event_id, event = next(iter(EVENT_WINDOWS.items()))
    return {
        "collection": {
            "source_corpus": "reddit",
            "subreddit": "privacy",
            "event_id": event_id,
            "jurisdiction": event["jurisdiction"],
            "window_start_utc": event["start_utc"],
            "window_end_utc": event["end_utc"],
            "query": "age verification",
        },
        "data": base,
    }


# Tests for the cleaning, flagging and masking rules the analysis depends on.
class PrepareCorpusTests(unittest.TestCase):
    # Load spaCy, the detector and stopwords once for the whole test class.
    @classmethod
    def setUpClass(cls):
        cls.nlp = load_spacy_model()
        cls.detector = build_language_detector()
        cls.stopwords = build_stopwords()

    # Light view unescapes HTML and keeps link anchor text, dropping the URL.
    def test_text_light_repairs_html_and_keeps_markdown_link_text(self):
        raw = "Look &amp; [here](https://example.com)  now"

        self.assertEqual(clean_text_light(raw), "Look & here now")

    # Sentiment view must preserve the intensity cues VADER reads.
    def test_sentiment_text_keeps_case_punctuation_emoji_and_placeholders(self):
        raw = "I DON'T like this 😡!!! https://example.com u/test r/privacy"

        self.assertEqual(clean_text_for_sentiment(raw), "I DO NOT like this 😡!!! <URL> <USER> <SUBREDDIT>")

    # Lemmatisation runs, but negation words survive stopword removal.
    def test_topic_tokens_use_spacy_lemma_and_keep_negation(self):
        tokens = topic_tokens("People are running and not good", self.nlp, self.stopwords)

        self.assertEqual(tokens, "people run not good")

    # Plurals are resolved by the lemmatiser, not by chopping trailing letters.
    def test_topic_tokens_do_not_manually_strip_plural_s(self):
        tokens = topic_tokens("The series continues", self.nlp, self.stopwords)

        self.assertIn("series", tokens.split())
        self.assertNotIn("serie", tokens.split())

    # A record keeps a primary frame for compatibility and exposes all matching frames.
    def test_relevance_hierarchy(self):
        self.assertEqual(classify_relevance("The new age verification rule")[0], "age_policy")
        self.assertEqual(classify_relevance("This is about child safety online")[0], "child_safety")
        self.assertEqual(classify_relevance("Privacy and surveillance concerns for minors")[0], "privacy_surveillance")
        self.assertEqual(classify_relevance("People will use a VPN to bypass it for teenagers")[0], "circumvention_autonomy")
        self.assertEqual(classify_relevance("plain daily practice update")[0], "unrelated")

    def test_relevance_frames_are_multi_label_and_platform_alone_is_not_context(self):
        level, _, frames = classify_relevance_detail("Age verification raises privacy and surveillance concerns")
        self.assertEqual(level, "age_policy")
        self.assertEqual(frames, ["age_policy", "privacy_surveillance"])
        self.assertEqual(classify_relevance("This platform violates privacy")[0], "unrelated")

    # Every matched keyword is recorded, even those outside the assigned tier.
    def test_relevance_terms_keep_all_matched_keywords(self):
        level, terms = classify_relevance("Age verification raises privacy and surveillance concerns")

        self.assertEqual(level, "age_policy")
        self.assertIn("age verification", terms)
        self.assertIn("privacy", terms)
        self.assertIn("surveillance", terms)

    # Author fields stay internal-only and all three text views are produced.
    def test_unwrap_post_keeps_authors_internal_and_generates_views(self):
        row = unwrap_post(build_post_wrapper(), nlp=self.nlp, detector=self.detector, stopword_set=self.stopwords)

        self.assertEqual(row["record_id"], "t3_abc")
        self.assertEqual(row["text_raw"], "Age assurance backlash\n\nPrivacy protections are being replaced by surveillance")
        self.assertEqual(row["relevance_level"], "age_policy")
        self.assertEqual(row["relevance_frames"], "age_policy|privacy_surveillance")
        self.assertTrue(row["parent_is_root"])
        self.assertTrue(row["age_assurance_related"])
        self.assertIn("age", row["tokens_topic"])
        self.assertIn("assurance", row["tokens_topic"])
        self.assertTrue(row["schema_valid"])
        self.assertIn(row["exclusion_status"], {"eligible", "language_uncertain"})
        self.assertTrue(row["has_author_metadata"])
        self.assertIn("_author", row)

    def test_reddit_display_case_subreddit_is_normalized_before_validation(self):
        post = unwrap_post(
            build_post_wrapper(subreddit="Parenting"),
            nlp=self.nlp,
            detector=self.detector,
            stopword_set=self.stopwords,
        )
        comment = unwrap_comment(
            build_comment_wrapper(subreddit="Parenting"),
            nlp=self.nlp,
            detector=self.detector,
            stopword_set=self.stopwords,
        )

        self.assertEqual(post["subreddit"], "parenting")
        self.assertTrue(post["schema_valid"])
        self.assertEqual(comment["subreddit"], "parenting")
        self.assertTrue(comment["schema_valid"])

    # Network joins use normalized Reddit IDs and a pseudonymous actor key.
    def test_network_fields_are_usable_without_raw_author_names(self):
        row = unwrap_comment(build_comment_wrapper(), nlp=self.nlp, detector=self.detector, stopword_set=self.stopwords)

        self.assertEqual(row["record_id"], "t1_def")
        self.assertEqual(row["thread_id"], "t3_abc")
        self.assertEqual(row["parent_record_id"], "t3_abc")
        self.assertFalse(row["parent_is_root"])
        self.assertEqual(len(row["author_hash"]), 24)
        self.assertNotIn("commenter", row["author_hash"])

    def test_reply_parent_keeps_comment_type_prefix(self):
        row = unwrap_comment(
            build_comment_wrapper(parent_id="t1_parent"),
            nlp=self.nlp,
            detector=self.detector,
            stopword_set=self.stopwords,
        )

        self.assertEqual(row["parent_record_id"], "t1_parent")

    def test_empty_comment_link_id_is_invalid_but_empty_parent_is_a_root(self):
        root = unwrap_comment(
            build_comment_wrapper(parent_id=""),
            nlp=self.nlp,
            detector=self.detector,
            stopword_set=self.stopwords,
        )
        missing_link = unwrap_comment(
            build_comment_wrapper(link_id=""),
            nlp=self.nlp,
            detector=self.detector,
            stopword_set=self.stopwords,
        )

        self.assertTrue(root["schema_valid"])
        self.assertEqual(root["parent_record_id"], "")
        self.assertTrue(root["parent_is_root"])
        self.assertFalse(missing_link["schema_valid"])

    def test_numeric_only_text_is_not_substantive_or_analysis_eligible(self):
        row = unwrap_comment(
            build_comment_wrapper(body="123"),
            nlp=self.nlp,
            detector=self.detector,
            stopword_set=self.stopwords,
        )
        finalize_analysis_masks(add_corpus_flags([row]))

        self.assertTrue(row["is_no_substantive_text"])
        self.assertFalse(row["sentiment_eligible"])
        self.assertFalse(row["topic_eligible"])

    def test_unrelated_privacy_text_is_excluded_from_analysis_masks(self):
        row = unwrap_comment(
            build_comment_wrapper(body="This platform violates privacy today"),
            nlp=self.nlp,
            detector=self.detector,
            stopword_set=self.stopwords,
        )
        finalize_analysis_masks(add_corpus_flags([row]))

        self.assertFalse(row["age_assurance_related"])
        self.assertFalse(row["sentiment_eligible"])
        self.assertFalse(row["topic_eligible"])

    # Deleted content is kept as a row with a status, never dropped.
    def test_deleted_comment_retained_with_deleted_status(self):
        row = unwrap_comment(
            build_comment_wrapper(body="[deleted]"),
            nlp=self.nlp,
            detector=self.detector,
            stopword_set=self.stopwords,
        )

        self.assertEqual(row["exclusion_status"], "deleted")
        self.assertFalse(row["sentiment_eligible"] if "sentiment_eligible" in row else False)

    # An unusable timestamp flags the row instead of discarding it.
    def test_invalid_timestamp_row_retained(self):
        row = unwrap_comment(
            build_comment_wrapper(created_utc="oops"),
            nlp=self.nlp,
            detector=self.detector,
            stopword_set=self.stopwords,
        )

        self.assertEqual(row["exclusion_status"], "invalid_timestamp")
        self.assertFalse(row["schema_valid"])
        self.assertEqual(row["datetime_utc"], "")

    # A record carrying an unknown acquisition query indicates misrouted data.
    def test_research_schema_rejects_unknown_query(self):
        wrapper = build_post_wrapper()
        wrapper["collection"]["query"] = "duolingo AI"

        row = unwrap_post(wrapper, nlp=self.nlp, detector=self.detector, stopword_set=self.stopwords)

        self.assertFalse(row["schema_valid"])
        self.assertEqual(row["exclusion_status"], "invalid_schema")

    def test_research_schema_rejects_unknown_query_value(self):
        wrapper = build_post_wrapper(subreddit="technology")
        wrapper["collection"]["subreddit"] = "technology"
        wrapper["collection"]["query"] = "duolingo AI"

        row = unwrap_post(wrapper, nlp=self.nlp, detector=self.detector, stopword_set=self.stopwords)

        self.assertFalse(row["schema_valid"])
        self.assertEqual(row["exclusion_status"], "invalid_schema")

    # Preprocessing refuses to start without a complete collection manifest.
    def test_missing_collection_manifest_requires_override(self):
        with TemporaryDirectory() as tmp:
            raw_root = Path(tmp)
            with self.assertRaises(SystemExit) as exc:
                require_complete_collection_manifest(raw_root, allow_partial=False)

            self.assertIn("allow-partial", str(exc.exception))
            manifest = require_complete_collection_manifest(raw_root, allow_partial=True)
            self.assertEqual(manifest["status"], "missing")

    def test_complete_manifest_from_another_study_is_rejected(self):
        with TemporaryDirectory() as tmp:
            raw_root = Path(tmp)
            (raw_root / "collection_manifest.json").write_text(
                '{"study_name": "old_study", "completion_status": "complete"}',
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit) as exc:
                require_complete_collection_manifest(raw_root, allow_partial=False)

        self.assertIn("study_name", str(exc.exception))

    # A collector CLI override must become the cleaner's validation window.
    def test_research_window_comes_from_collection_manifest(self):
        self.assertEqual(
            research_window_from_manifest(
                {"start_utc": "2025-01-01T00:00:00Z", "end_utc": "2025-02-01T00:00:00Z"}
            ),
            (1735689600, 1738368000),
        )

    # Absent author metadata is not the same as an author who deleted their account.
    def test_missing_author_metadata_is_not_deleted_author(self):
        rows = [
            unwrap_comment(
                build_comment_wrapper(author=None, author_fullname=None, author_flair_text=None, body="Age assurance privacy protections changed badly"),
                nlp=self.nlp,
                detector=self.detector,
                stopword_set=self.stopwords,
            )
        ]

        flagged = add_corpus_flags(rows)

        self.assertFalse(flagged[0]["is_deleted_author"])
        self.assertFalse(flagged[0]["is_likely_bot_author"])

    # Exercises the bot, moderator, spam, exact-duplicate and SimHash flags together.
    def test_bot_mod_spam_duplicate_and_near_duplicate_flags(self):
        rows = [
            unwrap_comment(
                build_comment_wrapper(
                    id="a",
                    author="AutoModerator",
                    distinguished="moderator",
                    body="buy discount coupon free trial now https://x.test",
                ),
                nlp=self.nlp,
                detector=self.detector,
                stopword_set=self.stopwords,
            ),
            unwrap_comment(
                build_comment_wrapper(
                    id="b",
                    author="normal_user",
                    body="Age verification privacy protections dropped badly",
                ),
                nlp=self.nlp,
                detector=self.detector,
                stopword_set=self.stopwords,
            ),
            unwrap_comment(
                build_comment_wrapper(
                    id="c",
                    author="other_user",
                    body="Age verification privacy protections dropped badly",
                ),
                nlp=self.nlp,
                detector=self.detector,
                stopword_set=self.stopwords,
            ),
            unwrap_comment(
                build_comment_wrapper(
                    id="d",
                    author="other_user2",
                    body="Age verification privacy protection dropped badly",
                ),
                nlp=self.nlp,
                detector=self.detector,
                stopword_set=self.stopwords,
            ),
        ]

        flagged = add_corpus_flags(rows)

        self.assertTrue(flagged[0]["is_automoderator"])
        self.assertTrue(flagged[0]["is_mod_distinguished"])
        self.assertTrue(flagged[0]["is_likely_bot_author"])
        self.assertTrue(flagged[0]["is_likely_spam"])
        self.assertTrue(flagged[1]["is_exact_duplicate_text"])
        self.assertTrue(flagged[2]["is_exact_duplicate_text"])
        self.assertTrue(flagged[3]["is_near_duplicate_text"])

    # The core language decision: inclusive keeps uncertain records, strict does not.
    def test_uncertain_short_sentiment_enters_inclusive_mask_and_strict_sensitivity_excludes_it(self):
        rows = [
            unwrap_comment(
                build_comment_wrapper(id="z", body="Age checks suck"),
                nlp=self.nlp,
                detector=self.detector,
                stopword_set=self.stopwords,
            )
        ]

        finalize_analysis_masks(add_corpus_flags(rows))

        self.assertTrue(rows[0]["sentiment_eligible"])
        self.assertFalse(rows[0]["sentiment_eligible_strict"])
        self.assertTrue(rows[0]["language_review_required"])
        self.assertTrue(rows[0]["sentiment_review_candidate"])
        self.assertTrue(rows[0]["sentiment_analysis_eligible"])
        self.assertFalse(rows[0]["sentiment_analysis_eligible_strict"])
        self.assertFalse(rows[0]["topic_eligible"])
        self.assertFalse(rows[0]["is_low_information"])

    # A manual English override restores a record the detector was unsure about.
    def test_manual_language_override_reincludes_uncertain_sentiment(self):
        rows = [
            unwrap_comment(
                build_comment_wrapper(id="override", body="Age checks suck"),
                nlp=self.nlp,
                detector=self.detector,
                stopword_set=self.stopwords,
            )
        ]
        rows[0]["manual_language_override"] = "english"

        finalize_analysis_masks(add_corpus_flags(rows))

        self.assertTrue(rows[0]["language_review_required"])
        self.assertTrue(rows[0]["sentiment_eligible"])
        self.assertFalse(rows[0]["sentiment_eligible_strict"])
        self.assertTrue(rows[0]["sentiment_analysis_eligible"])
        self.assertFalse(rows[0]["sentiment_analysis_eligible_strict"])

    # Overrides loaded from CSV must reach the final analysis masks.
    def test_language_override_file_drives_finalize_masks(self):
        rows = [
            unwrap_comment(
                build_comment_wrapper(id="override-file", body="Age checks suck"),
                nlp=self.nlp,
                detector=self.detector,
                stopword_set=self.stopwords,
            )
        ]
        rows = add_corpus_flags(rows)

        with TemporaryDirectory() as tmp:
            override_path = Path(tmp) / "language_overrides.csv"
            override_path.write_text("thing,record_id,manual_language_override\ncomment,override-file,english\n", encoding="utf-8")
            applied = apply_language_overrides(rows, load_language_overrides(override_path))

        finalize_analysis_masks(rows)

        self.assertEqual(applied, 1)
        self.assertTrue(rows[0]["sentiment_eligible"])
        self.assertTrue(rows[0]["human_only"])
        self.assertTrue(rows[0]["sentiment_analysis_eligible"])

    # A non-English override removes the record from both sentiment masks.
    def test_non_english_override_excludes_strict_and_inclusive_sentiment(self):
        rows = [
            unwrap_comment(
                build_comment_wrapper(
                    id="non-english-override",
                    body="Age assurance privacy protections changed for families today",
                ),
                nlp=self.nlp,
                detector=self.detector,
                stopword_set=self.stopwords,
            )
        ]
        rows[0]["manual_language_override"] = "non_english"

        finalize_analysis_masks(add_corpus_flags(rows))

        self.assertFalse(rows[0]["sentiment_eligible"])
        self.assertFalse(rows[0]["sentiment_eligible_strict"])
        self.assertFalse(rows[0]["sentiment_analysis_eligible"])
        self.assertFalse(rows[0]["sentiment_analysis_eligible_strict"])

    # Language is detected on what the author wrote, not on text they quoted.
    def test_language_detection_uses_quote_stripped_authored_text(self):
        quoted_french = "\n".join(["> ceci est un long texte francais avec plusieurs phrases pour dominer la detection"] * 8)
        body = quoted_french + "\nI hate this age verification change because privacy feels worse for families today"

        row = unwrap_comment(
            build_comment_wrapper(id="quote-lang", body=body),
            nlp=self.nlp,
            detector=self.detector,
            stopword_set=self.stopwords,
        )

        self.assertNotIn("ceci est un long texte", row["text_topic"])
        self.assertEqual(row["language"], "english")

    # A bare link carries no sentiment to score.
    def test_url_only_row_is_not_sentiment_eligible(self):
        rows = [
            unwrap_comment(
                build_comment_wrapper(id="url", body="https://example.com"),
                nlp=self.nlp,
                detector=self.detector,
                stopword_set=self.stopwords,
            )
        ]

        finalize_analysis_masks(add_corpus_flags(rows))

        self.assertTrue(rows[0]["is_url_only"])
        self.assertFalse(rows[0]["sentiment_eligible"])

    # Moderator-distinguished is recorded but does not by itself mean bot.
    def test_moderator_comment_can_remain_human_only(self):
        rows = [
            unwrap_comment(
                build_comment_wrapper(
                    id="mod",
                    distinguished="moderator",
                    body="Age assurance privacy protections changed and users discuss it clearly today",
                ),
                nlp=self.nlp,
                detector=self.detector,
                stopword_set=self.stopwords,
            )
        ]

        finalize_analysis_masks(add_corpus_flags(rows))

        if rows[0]["exclusion_status"] == "eligible":
            self.assertTrue(rows[0]["human_only"])

    # Several invalid rows share a blank id; that must not trip the duplicate check.
    def test_multiple_invalid_rows_do_not_fail_duplicate_validation(self):
        rows = [
            unwrap_comment(build_comment_wrapper(id="", body="bad row one"), nlp=self.nlp, detector=self.detector, stopword_set=self.stopwords),
            unwrap_comment(build_comment_wrapper(id="", body="bad row two"), nlp=self.nlp, detector=self.detector, stopword_set=self.stopwords),
        ]
        finalize_analysis_masks(add_corpus_flags(rows))

        with TemporaryDirectory() as tmp:
            _, _, frame = write_outputs(rows, "invalid_rows", Path(tmp))

        self.assertEqual(len(frame), 2)
        self.assertFalse(frame["schema_valid"].any())

    # Written files contain only the pseudonymous network key, never raw authors.
    def test_write_outputs_strips_raw_author_columns(self):
        rows = add_corpus_flags(
            [
                unwrap_post(build_post_wrapper(), nlp=self.nlp, detector=self.detector, stopword_set=self.stopwords),
                unwrap_comment(build_comment_wrapper(id="ghi"), nlp=self.nlp, detector=self.detector, stopword_set=self.stopwords),
            ]
        )
        finalize_analysis_masks(rows)

        with TemporaryDirectory() as tmp:
            parquet_path, csv_path, frame = write_outputs(rows, "mixed", Path(tmp))

            self.assertTrue(parquet_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertIn("author_hash", frame.columns)
            self.assertNotIn("_author", frame.columns)
            self.assertNotIn("author", frame.columns)
            self.assertNotIn("author_fullname", frame.columns)
            self.assertNotIn("author_flair_text", frame.columns)

    def test_submission_sample_redacts_raw_text_and_reddit_ids(self):
        rows = add_corpus_flags(
            [
                unwrap_post(build_post_wrapper(), nlp=self.nlp, detector=self.detector, stopword_set=self.stopwords),
                unwrap_comment(build_comment_wrapper(id="ghi"), nlp=self.nlp, detector=self.detector, stopword_set=self.stopwords),
            ]
        )
        finalize_analysis_masks(rows)

        with TemporaryDirectory() as tmp:
            _, _, frame = write_outputs(rows, "mixed", Path(tmp))
            output = write_submission_sample(frame, Path(tmp) / "submission.csv", sample_size=2)
            columns = output.read_text(encoding="utf-8").splitlines()[0].split(",")
            content = output.read_text(encoding="utf-8")

        self.assertIn("submission_node_id", columns)
        self.assertNotIn("text_raw", columns)
        self.assertNotIn("record_id", columns)
        self.assertNotIn("Age assurance backlash", content)
        self.assertNotIn("t3_abc", content)


if __name__ == "__main__":
    unittest.main()
