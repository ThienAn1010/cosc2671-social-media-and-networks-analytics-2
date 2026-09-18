from __future__ import annotations

from datetime import date

import pytest

from src.shared.case_windows import case_window_for_date, load_case_windows
from src.shared.ids import stable_analysis_doc_id
from src.shared.masking import contains_unmasked_pii, mask_text
from src.shared.time_axes import bluesky_day, reject_superseded_event_date


def test_mask_text_removes_contact_data_and_handles():
    masked = mask_text("Email a.person@example.org; call +61 412 345 678; see https://example.org/@name")
    assert masked == "Email <PII>; call <PII>; see <URL>"
    assert not contains_unmasked_pii(masked)


def test_contains_unmasked_pii_detects_phone_numbers_without_flagging_dates():
    assert contains_unmasked_pii("Call +61 412 345 678")
    assert not contains_unmasked_pii("The event ran from 2025-01-01 to 2025-01-08")


def test_short_hyphenated_numeric_dates_are_not_masked_as_phones():
    assert mask_text("The date is 2025-01-01") == "The date is 2025-01-01"


def test_case_window_assigns_australian_legislation():
    assert case_window_for_date(date(2024, 12, 1), load_case_windows()) == "AU_LEGISLATION"
    assert case_window_for_date(date(2025, 1, 11), load_case_windows()) == "none"


def test_stable_analysis_id_is_platform_prefixed():
    assert stable_analysis_doc_id("reddit", "comment", "t1_abc") == "rd:comment:t1_abc"


def test_bluesky_time_axis_uses_processed_day():
    assert bluesky_day({"day": "2026-03-09", "created_at": "2026-03-11T00:00:00Z"}) == date(2026, 3, 9)


def test_superseded_e5_date_is_rejected():
    with pytest.raises(ValueError, match="superseded E5"):
        reject_superseded_event_date("E5", "2026-03-11")
