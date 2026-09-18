# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

"""Offline tests for Bluesky pseudonymisation.

Uses a fixed TEST salt, never the real one. These check the rules, not any real
hash value, so they need no secret and pass on any machine.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.platforms.bluesky import anonymise as an
from src.shared.events import load_events
from src.shared.ids import author_hash

SALT = "test-salt-not-the-real-salt"
DID_A = "did:plc:aaaaaaaaaaaaaaaaaaaaaaaa"
URI_A = f"at://{DID_A}/app.bsky.feed.post/3lslyh7ofk226"


class TestSalt:
    def test_refuses_to_run_without_a_salt(self, monkeypatch):
        """Unsalted hashes are reversible by anyone with a DID list."""
        monkeypatch.setattr(an, "load_dotenv", lambda *a, **k: None)
        monkeypatch.delenv("PSEUDONYM_SALT", raising=False)
        with pytest.raises(SystemExit, match="PSEUDONYM_SALT missing"):
            an.load_salt()

    def test_uses_the_shared_hash_rule(self):
        """author_hash must be the shared helper, so anyone re-running this
        step with the same salt reproduces the same hashes."""
        frame = pd.DataFrame({"author_did": [DID_A]})
        out = an.transform(frame, SALT, {})
        assert out["author_hash"].iloc[0] == author_hash(SALT, "bluesky", DID_A)
        assert len(out["author_hash"].iloc[0]) == 16

    def test_different_salt_different_hash(self):
        """Why a salt is reproduced by sharing its value, not the command that made it."""
        assert author_hash("one", "bluesky", DID_A) != author_hash("two", "bluesky", DID_A)


class TestDocIds:
    def test_doc_id_follows_the_contract_form(self):
        posts = pd.DataFrame({"uri": [URI_A], "is_reply": [False], "embed_type": [None]})
        assert an.build_doc_ids(posts)[URI_A] == "bs:post:3lslyh7ofk226"

    def test_thing_reply_quote_post(self):
        assert an.thing_of(True, None) == "reply"
        assert an.thing_of(False, "app.bsky.embed.record#view") == "quote"
        assert an.thing_of(False, "app.bsky.embed.recordWithMedia#view") == "quote"
        assert an.thing_of(False, "app.bsky.embed.external#view") == "post"

    def test_reply_that_quotes_is_a_reply(self):
        assert an.thing_of(True, "app.bsky.embed.record#view") == "reply"

    def test_native_id_carries_no_did(self):
        assert "did:plc" not in an.rkey_of(URI_A)

    def test_collision_is_refused_not_silently_merged(self):
        """rkeys are unique per author, not globally. A collision would merge
        two different people's posts under one doc_id."""
        posts = pd.DataFrame({
            "uri": [f"at://did:plc:one/app.bsky.feed.post/SAME",
                    f"at://did:plc:two/app.bsky.feed.post/SAME"],
            "is_reply": [False, False], "embed_type": [None, None]})
        with pytest.raises(SystemExit, match="collision"):
            an.build_doc_ids(posts)

    def test_uncollected_root_defaults_to_post(self):
        uri = "at://did:plc:elsewhere/app.bsky.feed.post/3zzz"
        assert an.doc_id_of(uri, {}) == "bs:post:3zzz"


class TestForbiddenFields:
    def test_handles_and_display_names_are_dropped(self):
        frame = pd.DataFrame({"did": [DID_A], "handle": ["alice.bsky.social"],
                              "display_name": ["Alice"]})
        out = an.transform(frame, SALT, {})
        assert "handle" not in out.columns and "display_name" not in out.columns

    def test_mentions_become_user(self):
        text, _ = an.clean_text("thanks @alice.bsky.social for this")
        assert text == "thanks @user for this"

    def test_email_and_phone_become_pii(self):
        text, n = an.clean_text("email me at a.b@example.com or call +61 412 345 678")
        assert "example.com" not in text and "412" not in text
        assert n == 2

    def test_short_digit_runs_are_not_phones(self):
        """Years, dates and bill numbers must survive -- same 9-digit floor as
        the YouTube pipeline."""
        text, n = an.clean_text("the 2025-07-25 deadline for HR 8250")
        assert n == 0 and "2025-07-25" in text

    def test_profile_links_and_dids_are_removed(self):
        text, _ = an.clean_text(f"see bsky.app/profile/alice.bsky.social and {DID_A}")
        assert "alice" not in text and "did:plc" not in text


class TestSensitiveContent:
    def test_moderation_labels_flag_it(self):
        assert an.is_sensitive("porn")
        assert an.is_sensitive("rude,graphic-media")

    def test_non_adult_labels_do_not(self):
        assert not an.is_sensitive("rude,intolerant")
        assert not an.is_sensitive(None)


class TestEvents:
    def test_window_is_minus_seven_to_plus_twentyone(self):
        """The shared half-open window, decision D-005."""
        events = load_events()
        frame = pd.DataFrame({"day": ["2025-07-18", "2025-07-17", "2025-08-14", "2025-08-15"]})
        out = an.add_event_columns(frame, events)
        assert list(out["event_window"]) == ["E2", "E1", "E2", "none"]

    def test_nearest_event_ties_go_earlier(self):
        events = load_events()
        eid, days = an.nearest_event(date(2025, 7, 25), events)
        assert eid == "E2" and days == 0

    def test_days_from_event_is_signed(self):
        events = load_events()
        _, before = an.nearest_event(date(2025, 7, 20), events)
        assert before < 0
