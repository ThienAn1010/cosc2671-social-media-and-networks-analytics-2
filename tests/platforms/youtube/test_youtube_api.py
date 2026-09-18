# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

import unittest
from http.client import RemoteDisconnected
from unittest.mock import patch

from fake_youtube import FakeResponse, http_error

from src.platforms.youtube.api import ApiError, BudgetExceededError, Ledger, QuotaExceededError, api_get, iter_pages, parse_error

KEY = "test-key-123"


# The client must spend quota predictably, stop on quota errors, and never leak the key.
class ApiClientTests(unittest.TestCase):
    def test_ledger_keeps_search_in_its_own_bucket(self):
        ledger = Ledger()
        ledger.charge("search")
        ledger.charge("videos")
        self.assertEqual((ledger.search_calls, ledger.units), (1, 1))

    def test_budget_stops_before_any_request(self):
        with patch("src.platforms.youtube.api.urlopen") as mocked:
            with self.assertRaises(BudgetExceededError):
                api_get("videos", {"id": "x"}, Ledger(max_units=0), KEY)
            mocked.assert_not_called()

    def test_quota_error_is_not_retried(self):
        ledger = Ledger()
        with patch("src.platforms.youtube.api.urlopen", side_effect=[http_error(403, "quotaExceeded")]):
            with self.assertRaises(QuotaExceededError):
                api_get("commentThreads", {"videoId": "x"}, ledger, KEY)
        self.assertEqual(ledger.units, 1)

    def test_transient_error_is_retried_and_charged(self):
        ledger = Ledger()
        with patch("src.platforms.youtube.api.urlopen", side_effect=[http_error(500, "backendError"), FakeResponse({"items": []})]), \
                patch("src.platforms.youtube.api.time.sleep"):
            self.assertEqual(api_get("videos", {"id": "x"}, ledger, KEY), {"items": []})
        self.assertEqual(ledger.units, 2)

    # Regression: the full run on 2026-09-13 crashed when YouTube closed the connection without a response.
    def test_dropped_connection_is_retried(self):
        ledger = Ledger()
        with patch("src.platforms.youtube.api.urlopen", side_effect=[RemoteDisconnected("closed"), FakeResponse({"items": []})]), \
                patch("src.platforms.youtube.api.time.sleep"):
            self.assertEqual(api_get("comments", {"parentId": "x"}, ledger, KEY), {"items": []})
        self.assertEqual(ledger.units, 2)

    def test_config_error_stops_immediately_without_leaking_key(self):
        with patch("src.platforms.youtube.api.urlopen", side_effect=[http_error(400, "keyInvalid", f"bad key {KEY}")]):
            with self.assertRaises(ApiError) as caught:
                api_get("videos", {"id": "x"}, Ledger(), KEY)
        self.assertEqual(caught.exception.reason, "keyInvalid")
        self.assertNotIn(KEY, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)

    def test_iter_pages_follows_next_page_token(self):
        responses = [FakeResponse({"items": [1], "nextPageToken": "p2"}), FakeResponse({"items": [2]})]
        with patch("src.platforms.youtube.api.urlopen", side_effect=responses) as mocked:
            pages = list(iter_pages("commentThreads", {"videoId": "x"}, Ledger(), KEY))
        self.assertEqual([token for token, _ in pages], [None, "p2"])
        self.assertIn("pageToken=p2", mocked.call_args_list[1].args[0].full_url)

    def test_parse_error_handles_non_json_body(self):
        self.assertEqual(parse_error("<html>")[0], "unknown")


if __name__ == "__main__":
    unittest.main()
