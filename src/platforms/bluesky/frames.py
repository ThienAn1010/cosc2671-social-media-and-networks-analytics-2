# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

r"""Frame lexicons for the age-verification corpus, and the circumvention indicator.

Importable definitions only -- `build_frames.py` applies them. Kept separate so
the patterns can be inspected, cited and version-controlled without reading the
code that uses them.

What a "frame" is here
----------------------
A frame is the ANGLE a post takes on age verification: privacy, circumvention,
child safety, free speech, or the mechanics of handing over ID. Frames are not
mutually exclusive -- "they want my passport to protect the children" is both --
and they are not stance. `privacy` means the post engages the privacy angle,
NOT that it objects on privacy grounds. See `docs/bluesky/frame_validation.md`.

Design rule
-----------
Every term must be specific to its frame, not merely topic-adjacent. The draft
version failed this in three measurable ways, all found by asking which posts a
term was the SOLE reason for flagging:

* `censorship` contributed **0** unique posts, because the pattern `\bcensor`
  already matched it. Dead weight that inflated the apparent term count.
* Bare `censor` matched image censor-bars -- "tip at least $5 to remove the
  silly censor" -- which is not a speech-policy claim.
* `safeguard` matched "how anthropic's claude ai protects user mental health"
  and "ofcom was designed to safeguard standards in an era of media plurality",
  neither of which is child-safety framing. Worse, "safeguard free speech" would
  have counted toward the OPPOSING frame.

v2 fixes those and adds recall terms for privacy objection expressed without the
word "privacy" -- "scan my face", "hand over my id" -- which the draft missed on
86 posts in a single spot check.
"""

# --------------------------------------------------------------------- frames
FRAMES: dict[str, str] = {
    # The privacy/surveillance angle. Recall additions in the second half:
    # people object on privacy grounds without ever using the word.
    "privacy": r"""\b(
        privacy | surveillance | data\ breach | breached | personal\ data
      | biometric | facial\ recognition | anonymity | anonymous
      | data\ (harvest|mining|retention) | honeypot
      | hacked | leaked
      | (hand|handing)\ over\ (my|your)\ (id|data|passport|details)
      | (send|sending|give|giving|upload(ing)?)\ (them\ )?(my|your)\ (id|face|passport|data|details)
      | scan\ (my|your)\ face | who\ (is\ |s\ )?(storing|keeping)\ (my|your|this)
      | don.t\ trust\ (them|him|her|it|these)\ with
    )""",

    # Getting around the mandate. `vpn` alone carries most of this and is highly
    # specific in context; the rest catch the non-VPN vocabulary.
    "circumvent": r"""\b(
        vpn | bypass(ing|ed)? | circumvent(ing|ed)? | work\ ?around | get(ting)?\ around
      | proxy | tor\ browser | loophole | evade | side\ ?step
      # NOT `another`: "that's another age verification step isn't it?" matched
      # 154 posts on `another age`, none of them about faking anything.
      | (fake|false)\ (id|birthday|age|dob)
    )""",

    # The mechanics of proving age -- distinct from the privacy angle, because a
    # post can describe the process without objecting to it.
    "id_upload": r"""\b(
        upload(ing)?\ (my\ |your\ |an?\ |the\ )?(id|passport|licence|license|document)
      | photo\ id | government\ id | digital\ id | id\ check | show\ (my|your)\ id
      | selfie | face\ scan | facial\ scan | (scan|picture|photo)\ of\ (my|your)
    )""",

    # The protective rationale. `safeguard` is restricted to child-safeguarding:
    # unrestricted, it matched media-standards and AI-wellbeing posts, and would
    # also have matched "safeguard free speech".
    "child_safety": r"""\b(
        child\ safety | child\ protection | protect(ing|ion)?\ (of\ )?(children|kids|minors|young)
      | safeguard(ing)?\ (children|kids|minors|young) | child\ safeguard
      | grooming | csam | harmful\ content | age[- ]appropriate
      | keep(ing)?\ (children|kids|minors)\ safe
    )""",

    # The speech/liberty angle. `censorship|censoring|censored` replaces the bare
    # `censor` stem, which matched censor-bars on images.
    "free_speech": r"""\b(
        free\ speech | freedom\ of\ (expression|speech) | censorship | censoring | censored
      | first\ amendment | authoritarian | overreach(ing)? | nanny\ state
      | chilling\ effect | slippery\ slope
    )""",
}

# ------------------------------------------------------- circumvention as ACT
# A separate, deliberately narrow indicator: the post reports the author
# PERSONALLY circumventing, rather than discussing circumvention.
#
# This needs no stance model because it is self-evidencing -- "i forgot my vpn
# was on" reports behaviour, it does not express a position. That makes it the
# most direct measurement available of the "circumvention behaviour" the project
# set out to study, and it is immune to the objection that a classifier put
# words in respondents' mouths.
#
# It must be ANDed with the `circumvent` frame: "i just got a" on its own is
# obviously not circumvention.
FIRST_PERSON_ACT = r"""\b(
    i\ (just\ |already\ |simply\ |finally\ |immediately\ |now\ )?(use|used|using|installed|
        switched|turned\ on|flipped|got|downloaded|bought|set\ up|enabled|cycled|reset|fired\ up)
  | i.m\ using | i\ have\ (a|my)\ vpn | my\ vpn | i\ bypass(ed)? | i\ got\ around
  | i\ had\ to\ (use|get|install|switch|figure\ out|turn\ on)
  | i\ ain.t\ playing\ ball | has\ been\ bypassed
)"""

# Two things had to be added after measuring, because the AND of the two
# patterns above was too loose on its own (precision 72% on a 100-item blind
# sample).
#
# PROXIMITY. The act verb and the circumvention term must be near each other.
# Without it, "twitter doesn't have this horseshit that i have to fight with
# every time i **use** the app" counted, because the post mentioned a VPN
# somewhere else entirely. 80 characters is about one clause either way.
ACT_PROXIMITY_CHARS = 80

# DENIAL. Some authors describe using a VPN and explicitly say it is NOT to get
# around age verification -- "i use a vpn precisely *because* i want to feel safe
# on the internet, not to evade the online safety act". Counting those inverts
# the author's stated meaning, which is the one error this indicator most needs
# to avoid: its whole claim to being better than a stance classifier is that it
# reports what people say they did.
DENIAL = r"""(
    not\ to\ (evade|circumvent|bypass|get\ around|dodge)
  | isn.t\ to\ (evade|circumvent|bypass)
  | not\ because\ i.m\ underage
  | i\ verified | i\ confirmed\ my\ id | i\ did\ the\ age\ verification
)"""

# Everything above is written with re.VERBOSE, so whitespace in the patterns is
# ignored and escaped spaces (`\ `) are literal.
import re

FLAGS = re.IGNORECASE | re.VERBOSE

COMPILED = {name: re.compile(pat, FLAGS) for name, pat in FRAMES.items()}
COMPILED_ACT = re.compile(FIRST_PERSON_ACT, FLAGS)
COMPILED_DENIAL = re.compile(DENIAL, FLAGS)


def frames_of(text: str) -> dict[str, bool]:
    """Which frames a post engages. Not mutually exclusive, not stance."""
    return {name: bool(rx.search(text)) for name, rx in COMPILED.items()}


def is_first_person_circumvention(text: str) -> bool:
    """True when the author reports personally getting around age verification.

    Requires a first-person act verb and a circumvention term within
    ACT_PROXIMITY_CHARS of one another, and no explicit denial anywhere in the
    post. See the notes on those two constants for what each one fixed.
    """
    if COMPILED_DENIAL.search(text):
        return False
    acts = [m.span() for m in COMPILED_ACT.finditer(text)]
    if not acts:
        return False
    circs = [m.span() for m in COMPILED["circumvent"].finditer(text)]
    if not circs:
        return False
    for a_start, a_end in acts:
        for c_start, c_end in circs:
            # Distance between the two spans, zero when they overlap.
            gap = max(c_start - a_end, a_start - c_end, 0)
            if gap <= ACT_PROXIMITY_CHARS:
                return True
    return False
