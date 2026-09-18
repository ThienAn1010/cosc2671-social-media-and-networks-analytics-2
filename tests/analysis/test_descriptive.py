from __future__ import annotations

import pandas as pd

from src.analysis.descriptive import CORE_CASES, _author_balanced_frame_share, _language_masks, _timeline


def test_author_balanced_share_weights_authors_equally():
    frame = pd.DataFrame({"author_hash": ["a", "a", "b"], "flag": [True, False, False]})
    assert _author_balanced_frame_share(frame, frame["flag"]) == 0.25


def test_language_population_keeps_short_english_as_uncertain():
    frame = pd.DataFrame(
        {
            "is_english": [True, True, False],
            "exclusion_status": ["eligible", "language_uncertain", "non_english"],
        }
    )
    strict, inclusive = _language_masks(frame)
    assert strict.tolist() == [True, False, False]
    assert inclusive.tolist() == [True, True, False]


def test_inclusive_language_bound_keeps_uncertain_non_english_records():
    frame = pd.DataFrame(
        {
            "is_english": [False, False],
            "is_language_uncertain": [True, False],
            "exclusion_status": ["non_english", "non_english"],
        }
    )

    _, inclusive = _language_masks(frame)

    assert inclusive.tolist() == [True, False]


def test_timeline_marks_unsupported_days_missing():
    scope = next(iter(CORE_CASES))
    base = pd.DataFrame(
        {
            "scope": [scope, scope],
            "day": ["2025-01-01", "2025-01-03"],
            "inclusive": [True, True],
            "author_hash": ["a", "b"],
        }
    )
    timeline = _timeline(base, "test", "day", "scope")
    middle = timeline.loc[timeline["day"].eq("2025-01-02")].iloc[0]
    assert pd.isna(middle["all_documents"])
    assert middle["support_status"] == "missing"
