# Student name: Nguyen Hoang Thien An
# Student ID: s3825455
# Assignment 1 - Social Media and Networks Analytics

import unittest

import pandas as pd

from src.sample_validation import build_sample_masks


def row(**overrides):
    base = {
        "schema_valid": True,
        "text_topic": "age verification privacy quality",
        "text_sentiment": "Age verification privacy is bad",
        "is_url_only": False,
        "is_no_substantive_text": False,
        "human_only": True,
        "source_corpus": "reddit",
        "age_assurance_related": True,
        "relevance_level": "age_policy",
        "relevance_frames": "age_policy|privacy_surveillance",
        "is_low_information": False,
        "is_likely_bot_author": False,
        "is_likely_spam": False,
        "is_exact_duplicate_text": False,
        "is_near_duplicate_text": False,
        "language_review_required": False,
        "sentiment_eligible": True,
        "sentiment_analysis_eligible": True,
        "sentiment_compound": -0.7,
        "unique_text_only": True,
    }
    base.update(overrides)
    return base


# Tests that unusable rows never enter the manual validation sample.
class SampleValidationTests(unittest.TestCase):
    # URL-only, empty and ineligible rows must be excluded from every stratum.
    def test_masks_exclude_unusable_rows_from_relevance_and_sentiment(self):
        frame = pd.DataFrame(
            [
                row(),
                row(schema_valid=False),
                row(is_url_only=True),
                row(human_only=False),
                row(sentiment_eligible=False, sentiment_analysis_eligible=False, sentiment_compound=-0.9),
                row(source_corpus="reddit"),
                row(age_assurance_related=False, relevance_level="unrelated"),
                row(language_review_required=True, sentiment_eligible=False, sentiment_analysis_eligible=False),
                row(is_exact_duplicate_text=True, unique_text_only=False),
            ]
        )

        masks = build_sample_masks(frame)

        self.assertEqual(masks["age_policy"].tolist(), [True, False, False, False, True, True, False, True, False])
        self.assertEqual(masks["privacy_surveillance"].tolist(), [True, False, False, False, True, True, False, True, False])
        self.assertEqual(masks["sentiment_edge_negative"].tolist(), [True, False, False, True, False, True, True, False, False])
        self.assertEqual(masks["cross_community"].tolist(), [True, False, False, False, True, True, False, True, False])
        self.assertEqual(masks["bot_spam_duplicate_language_uncertain"].tolist(), [False, False, False, False, False, False, False, True, True])


if __name__ == "__main__":
    unittest.main()
