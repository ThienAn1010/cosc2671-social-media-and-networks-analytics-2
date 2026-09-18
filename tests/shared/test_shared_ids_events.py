# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

import unittest
from datetime import date

from src.shared.events import NO_WINDOW, event_window, load_events, nearest_event, window_rfc3339
from src.shared.ids import author_hash, make_doc_id


# Identifiers must be identical across platforms and runs, or cross-source joins silently break.
class IdsTests(unittest.TestCase):
    def test_doc_id_uses_platform_prefix(self):
        self.assertEqual(make_doc_id("youtube", "comment", "Ugx1"), "yt:comment:Ugx1")
        with self.assertRaises(ValueError):
            make_doc_id("youtube", "comment", "")

    def test_author_hash_is_stable_salted_and_short(self):
        first = author_hash("salt", "youtube", "UC123")
        self.assertEqual(first, author_hash("salt", "youtube", "UC123"))
        self.assertEqual(len(first), 16)
        self.assertNotEqual(first, author_hash("other-salt", "youtube", "UC123"))
        self.assertEqual(author_hash("salt", "youtube", ""), "")
        with self.assertRaises(ValueError):
            author_hash("", "youtube", "UC123")


# Window edges decide which event a document belongs to, so the half-open boundaries are tested exactly.
class EventsTests(unittest.TestCase):
    def setUp(self):
        self.events = load_events()

    def test_calendar_has_five_sorted_events(self):
        self.assertEqual([event.event_id for event in self.events], ["E1", "E2", "E3", "E4", "E5"])

    def test_window_is_half_open_and_non_overlapping(self):
        self.assertEqual(event_window(date(2025, 7, 17), self.events), "E1")
        self.assertEqual(event_window(date(2025, 7, 18), self.events), "E2")
        self.assertEqual(event_window(date(2025, 8, 14), self.events), "E2")
        self.assertEqual(event_window(date(2025, 8, 15), self.events), NO_WINDOW)

    def test_nearest_event_returns_signed_day_offset(self):
        event, days = nearest_event(date(2025, 12, 1), self.events)
        self.assertEqual((event.event_id, days), ("E3", -9))

    def test_window_rfc3339_bounds(self):
        e2 = next(event for event in self.events if event.event_id == "E2")
        self.assertEqual(window_rfc3339(e2), ("2025-07-18T00:00:00Z", "2025-08-15T00:00:00Z"))


if __name__ == "__main__":
    unittest.main()
