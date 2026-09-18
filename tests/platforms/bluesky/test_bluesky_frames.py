# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

"""Offline tests for the Bluesky frame lexicons and the circumvention indicator.

These pin the behaviours that were established by measurement rather than by
choice, so a later edit that quietly undoes one of them fails here instead of in
a result. Each test names the evidence it protects; the full record is in
`docs/bluesky/frame_validation.md`.

No network, no data files: every case is a literal string.
"""

from __future__ import annotations

from src.platforms.bluesky.frames import (
    ACT_PROXIMITY_CHARS,
    COMPILED,
    frames_of,
    is_first_person_circumvention,
)


class TestFrameLexicons:
    def test_every_frame_is_independent(self):
        """Frames are not mutually exclusive; a post can engage several.

        Note the wording: "upload your passport", not "want my passport". The
        lexicon matches the former and not the latter, which is a real recall
        gap -- but the lexicons are FROZEN. Their precision, recall and Cohen's
        kappa were all measured against this exact version, so widening a
        pattern now would silently invalidate every figure in
        docs/bluesky/frame_validation.md. Tests here pin what was validated, not
        what would be nice.
        """
        got = frames_of("upload your passport to protect children")
        assert got["id_upload"] and got["child_safety"]

    def test_no_frame_on_unrelated_text(self):
        assert not any(frames_of("thames water ceo went unchallenged on the bbc").values())

    def test_privacy_catches_objection_without_the_word_privacy(self):
        """Recall fix: 86 posts objected on privacy grounds using no such word."""
        assert frames_of("having to scan my face for age verification")["privacy"]
        assert frames_of("i refuse to hand over my id to some company")["privacy"]

    def test_censor_stem_does_not_match_image_censor_bars(self):
        """v1 defect: bare `censor` matched an unrelated image-editing post."""
        assert not frames_of("tip at least $5 to remove the silly censor")["free_speech"]
        assert frames_of("this is state censorship, plain and simple")["free_speech"]

    def test_safeguard_is_restricted_to_children(self):
        """v1 defect: `safeguard` matched media standards and AI-wellbeing posts,
        and would also have matched 'safeguard free speech' -- the OPPOSING frame."""
        assert not frames_of("ofcom was designed to safeguard broadcasting standards")["child_safety"]
        assert not frames_of("we must safeguard free speech online")["child_safety"]
        assert frames_of("measures safeguarding children from harm")["child_safety"]

    def test_definite_article_defeats_child_safety(self):
        """A KNOWN GAP, pinned deliberately rather than fixed.

        "protect children" matches; "protect **the** children" does not, because
        the pattern allows only "of" between the verb and the noun. The second
        phrasing is at least as common as the first, so this is a concrete
        cause of the 51.4% recall measured for this frame -- the weakest of the
        five.

        The test asserts the CURRENT behaviour so that anyone widening the
        pattern is forced to notice they are invalidating the published recall
        figure and Cohen's kappa, both measured against this exact version.
        Fixing it means re-validating, not editing.
        """
        assert not frames_of("it was never about protecting the children")["child_safety"]
        assert frames_of("it was never about protecting children")["child_safety"]

    def test_another_age_is_not_circumvention(self):
        """v1 defect: `another age` matched 154 posts about nothing."""
        assert not frames_of("that's another age verification step isn't it")["circumvent"]
        assert frames_of("i used a fake id to get past it")["circumvent"]


class TestCircumventionIndicator:
    def test_first_person_report_counts(self):
        assert is_first_person_circumvention("i just use a vpn and set it to ireland")
        assert is_first_person_circumvention("i had to turn on my vpn to read my dms")

    def test_discussion_is_not_an_act(self):
        """The frame says a post DISCUSSES circumvention; the indicator says the
        author reports DOING it. Keeping them apart is the whole point."""
        assert COMPILED["circumvent"].search("kids will just use a vpn to get around it")
        assert not is_first_person_circumvention("kids will just use a vpn to get around it")
        assert not is_first_person_circumvention("vpn signups jumped 1400% after the act")

    def test_denial_is_not_an_admission(self):
        """v1 inverted the author's stated meaning on posts like this one. That is
        the error the indicator can least afford, since its claim over a stance
        classifier is that it reports what people say they did."""
        assert not is_first_person_circumvention(
            "i use a vpn precisely because i want to feel safe online, "
            "not to evade the online safety act"
        )

    def test_compliance_is_not_circumvention(self):
        assert not is_first_person_circumvention(
            "i verified. i got the emails. i shouldn't have had to, but i did, "
            "and my vpn had nothing to do with it"
        )

    def test_proximity_is_required(self):
        """v1 had no distance limit, so an act verb anywhere in the post paired
        with a circumvention term anywhere else. Precision was 72%."""
        far = ("i have to fight with this app every time i use it. "
               + "x" * (ACT_PROXIMITY_CHARS + 40)
               + " anyway someone mentioned a vpn in another thread")
        assert not is_first_person_circumvention(far)
        assert is_first_person_circumvention("i use a vpn for this")

    def test_indicator_implies_the_frame(self):
        """An act is always also a discussion; the reverse does not hold."""
        text = "i got a vpn to get around the age verification"
        assert is_first_person_circumvention(text)
        assert frames_of(text)["circumvent"]
