# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

r"""Pseudonymise the Bluesky analysis tables into the processed layer.

    python -m src.platforms.bluesky.anonymise

Reads `data/interim/`, writes `data/processed/bluesky/` and a
`prepare_manifest.json`. Nothing upstream is modified; `data/interim/` keeps
real identifiers and never leaves the machine.

Relationship to other platform workstreams
-------------------------------------------
This workstream keeps its own table structure and is analysed independently.
There is no cross-platform join. It adopts the following conventions for report
consistency:

  * pseudonymised users, via the shared `src.shared.ids` helpers and a secret salt
  * the shared event calendar and its [date-7, date+21) window
  * UTC throughout
  * no silent drop -- excluded rows stay, flagged
  * a manifest
  * sensitive identity fields are excluded from processed output

The salt
--------
Read from `PSEUDONYM_SALT` in the environment, exactly as `src.platforms.youtube.prepare`
does, and the run refuses to start without it. It lives in `SMFR/.env` --
never in code, never in the repo, never alongside the shared data.

Two properties matter. It must be SECRET: DIDs are public, so anyone holding the
salt can hash a DID list and match accounts back. And it must be STABLE: every
run over this workstream's data must use the same value, or `author_hash` stops
joining across outputs.

This workstream uses its own salt rather than the team value. Nothing is lost
for analysis: there is no cross-platform join, and the platform name is inside
the hash input, so no Bluesky hash is ever compared with another platform's.
The cost is reproducibility by others -- a teammate re-running this
step gets matching hashes only with the Bluesky salt, obtained privately from
the owner. `secrets.token_hex(32)` returns a fresh value on every call, so a
salt is reproduced by sharing its VALUE, never by re-running the command.

Pseudonymisation, not anonymisation
-----------------------------------
Post text is retained, and post text is public: a distinctive sentence can be
searched on Bluesky to find its author. This removes bulk identifiability, not
individual re-identification. Say "pseudonymised, with post text retained" in
the report, never "anonymous".
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.platforms.bluesky.config import PROJECT_ROOT, load_config
from src.shared.events import event_window, load_events
from src.shared.ids import author_hash, make_doc_id

PLATFORM = "bluesky"

# Sensitive identity fields are excluded from processed data: display name,
# handle, avatar, profile URL, email and phone. Dropped outright.
DROP_COLS = {"handle", "seed_handle", "display_name", "cid", "avatar", "banner"}

# Same patterns and threshold as src.platforms.youtube.prepare, so "<PII>" means the same
# thing in both workstreams. Copied rather than imported: the workstreams are
# independent, and this one should not break when that one refactors.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
PHONE_MIN_DIGITS = 9
HANDLE_RE = re.compile(r"@[A-Za-z0-9][A-Za-z0-9._-]*\.[A-Za-z]{2,}")
PROFILE_LINK_RE = re.compile(r"(?:https?://)?(?:staging\.)?bsky\.app/profile/\S+", re.I)
DID_RE = re.compile(r"did:plc:[a-z0-9]+", re.I)

# Bluesky moderation labels marking adult or graphic content. The platform's own
# labellers applied these, which is a stronger signal than the keyword lexicon
# YouTube had to rely on for the same flag.
SENSITIVE_LABELS = {"porn", "sexual", "nudity", "graphic-media", "sexual-figurative"}

QUOTE_EMBEDS = {"app.bsky.embed.record#view", "app.bsky.embed.recordWithMedia#view"}


def load_salt() -> str:
    """The workstream salt, from SMFR/.env. Refuses to run without it."""
    load_dotenv(PROJECT_ROOT / ".env")
    salt = os.environ.get("PSEUDONYM_SALT", "").strip()
    if not salt:
        raise SystemExit(
            "PSEUDONYM_SALT missing: add the Bluesky salt to SMFR/.env.\n"
            "Use the value the existing outputs were built with (ask the "
            "workstream owner). A new salt gives hashes that do not match "
            "any data already shared."
        )
    return salt


def rkey_of(uri) -> str:
    """Record key: the last segment of at://<did>/app.bsky.feed.post/<rkey>.

    Used as native_id because the full AT-URI embeds the author's DID, which the
    privacy policy excludes from processed data. An rkey cannot be resolved to a post
    without the DID.
    """
    return uri.rsplit("/", 1)[-1] if isinstance(uri, str) and uri else ""


def thing_of(is_reply: bool, embed_type) -> str:
    """Contract section 5 allows bluesky: post / reply / quote.

    A post that both replies and quotes is filed as `reply`: its position in a
    conversation is the structural fact the interaction tables are built on.
    """
    if is_reply:
        return "reply"
    if embed_type in QUOTE_EMBEDS:
        return "quote"
    return "post"


def build_doc_ids(posts: pd.DataFrame) -> dict[str, str]:
    """uri -> doc_id for every post held. Fails loudly on a collision."""
    things = [thing_of(bool(r), e) for r, e in zip(posts["is_reply"], posts["embed_type"])]
    mapping = {u: make_doc_id(PLATFORM, t, rkey_of(u)) for u, t in zip(posts["uri"], things)}
    if len(set(mapping.values())) != len(mapping):
        raise SystemExit(
            "doc_id collision: two posts share a record key. TIDs are unique in "
            "practice (0 collisions across 423,628 posts at build time) but not by "
            "guarantee; native_id must be widened before this can run."
        )
    return mapping


def doc_id_of(uri, mapping: dict[str, str]) -> str:
    """A URI outside the corpus -- a thread root never collected -- has unknown
    `thing`. A root is never a reply, so `post` is the safe default."""
    if not isinstance(uri, str) or not uri:
        return ""
    return mapping.get(uri) or make_doc_id(PLATFORM, "post", rkey_of(uri))


def _mask_phone(match: re.Match) -> str:
    return "<PII>" if sum(ch.isdigit() for ch in match.group(0)) >= PHONE_MIN_DIGITS else match.group(0)


def clean_text(text) -> tuple:
    """Strip what names or contacts a person. Returns (text, pii_count)."""
    if not isinstance(text, str) or not text:
        return text, 0
    text = EMAIL_RE.sub("<PII>", text)
    text = PHONE_RE.sub(_mask_phone, text)
    pii = text.count("<PII>")
    text = PROFILE_LINK_RE.sub("<URL>", text)
    text = HANDLE_RE.sub("@user", text)
    text = DID_RE.sub("<PII>", text)
    return text, pii


def is_sensitive(labels) -> bool:
    if not isinstance(labels, str) or not labels:
        return False
    return bool(SENSITIVE_LABELS & set(labels.split(",")))


def nearest_event(day: date, events) -> tuple:
    """Calendar-nearest event; ties go to the EARLIER event, matching the rule in
    docs/youtube/data_dictionary.md."""
    best = min(events, key=lambda e: (abs((day - e.event_date).days), e.event_date))
    return best.event_id, (day - best.event_date).days


def add_event_columns(df: pd.DataFrame, events, day_col: str = "day") -> pd.DataFrame:
    """event_id_nearest, days_from_event, event_window from the shared calendar.

    Keyed on `day`, not `created_date_utc`. Bluesky's
    `created_at` is set by the posting client -- 899 posts here claim dates
    before Bluesky existed, the earliest 2004 -- so dating events off it would
    place those posts two decades from any event. `day` is the collection cell
    for search posts and the index day for replies. This divergence is recorded
    in the manifest so nobody assumes otherwise.
    """
    days = pd.to_datetime(df[day_col]).dt.date
    nearest = days.map(lambda d: nearest_event(d, events))
    df["event_id_nearest"] = nearest.map(lambda t: t[0])
    df["days_from_event"] = nearest.map(lambda t: t[1]).astype("int32")
    df["event_window"] = days.map(lambda d: event_window(d, events))
    return df


DID_COLS = {"author_did": "author_hash", "did": "author_hash",
            "source_did": "source_author_hash", "target_did": "target_author_hash",
            "seed_did": "seed_author_hash", "reply_author_did": "reply_author_hash"}
URI_COLS = {"uri": "doc_id", "reply_uri": "source_doc_id", "parent_uri": "target_doc_id",
            "reply_parent_uri": "parent_doc_id", "reply_root_uri": "root_doc_id"}


def transform(df: pd.DataFrame, salt: str, doc_ids: dict[str, str]) -> pd.DataFrame:
    df = df.drop(columns=[c for c in df.columns if c in DROP_COLS], errors="ignore")
    for col, new in DID_COLS.items():
        if col in df.columns:
            df[new] = df[col].map(lambda v: author_hash(salt, PLATFORM, v) if isinstance(v, str) else "")
            df = df.drop(columns=col)
    for col, new in URI_COLS.items():
        if col in df.columns:
            df[new] = df[col].map(lambda u: doc_id_of(u, doc_ids))
            df = df.drop(columns=col)
    if "text" in df.columns:
        cleaned = df["text"].map(clean_text)
        df["text_raw"] = cleaned.map(lambda t: t[0])
        df["pii_count"] = cleaned.map(lambda t: t[1]).astype("int32")
        df = df.drop(columns="text")
    if "text_norm" in df.columns:
        df["text_norm"] = df["text_norm"].map(lambda t: clean_text(t)[0])
    return df


def audit(out_dir: Path) -> list:
    """Fail loudly if any forbidden identifier survived."""
    problems = []
    for f in sorted(out_dir.glob("*")):
        if f.suffix == ".parquet":
            frame = pd.read_parquet(f)
            blob = "\n".join(frame[c].astype(str).str.cat(sep="\n")
                             for c in frame.columns if frame[c].dtype == object)
        elif f.suffix in (".csv", ".txt"):
            blob = f.read_text(encoding="utf-8", errors="replace")
        else:
            continue
        if DID_RE.search(blob):
            problems.append(f"{f.name}: contains a DID")
        if re.search(r"\bat://", blob):
            problems.append(f"{f.name}: contains an AT-URI")
        if re.search(r"@[A-Za-z0-9][A-Za-z0-9._-]*\.bsky\.social", blob):
            problems.append(f"{f.name}: contains an @handle")
        if EMAIL_RE.search(blob):
            problems.append(f"{f.name}: contains an email-like string")
    return problems


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    cfg = load_config()
    salt = load_salt()
    events = load_events()

    interim = cfg.interim_dir
    out_dir = PROJECT_ROOT / "data" / "processed" / PLATFORM
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    posts = pd.read_parquet(interim / "posts.parquet")
    doc_ids = build_doc_ids(posts)
    posts["thing"] = [thing_of(bool(r), e) for r, e in zip(posts["is_reply"], posts["embed_type"])]
    posts["native_id"] = posts["uri"].map(rkey_of)
    posts["platform"] = PLATFORM
    posts["is_sensitive_content"] = posts["labels"].map(is_sensitive)
    posts = add_event_columns(posts, events)

    counts = {}
    skip = {"frame_validation_sheet.csv", "frame_validation_key.csv"}
    for src in sorted(interim.glob("*.parquet")) + sorted(interim.glob("*.csv")):
        if src.name in skip:
            continue
        if src.name == "posts.parquet":
            frame = posts
        elif src.suffix == ".parquet":
            frame = pd.read_parquet(src)
        else:
            frame = pd.read_csv(src)
        frame = transform(frame.copy(), salt, doc_ids)
        dst = out_dir / src.name
        if src.suffix == ".parquet":
            frame.to_parquet(dst, compression="zstd", index=False)
        else:
            frame.to_csv(dst, index=False, encoding="utf-8")
        counts[src.name] = len(frame)
        print(f"  {src.name:<34} {len(frame):>9,} rows")

    problems = audit(out_dir)
    if problems:
        raise SystemExit("identifiers survived; not safe to share:\n  " + "\n  ".join(problems))

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": PLATFORM,
        "script": "src.platforms.bluesky.anonymise",
        "salt_source": "env:PSEUDONYM_SALT",
        "doc_id_rule": "bs:<post|reply|quote>:<rkey>",
        "author_hash_rule": "sha256(PSEUDONYM_SALT + platform + did)[:16]",
        "event_calendar": "config/events.csv",
        "event_window_days": [-7, 21],
        "event_date_basis": "day (collection cell / index day), not created_at",
        "counts": counts,
        "sensitive_content_rows": int(posts["is_sensitive_content"].sum()),
        "outputs_sha256": {f.name: sha256_file(f) for f in sorted(out_dir.glob("*"))},
    }
    (out_dir / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Audit clean. Wrote {out_dir}")


if __name__ == "__main__":
    main()
