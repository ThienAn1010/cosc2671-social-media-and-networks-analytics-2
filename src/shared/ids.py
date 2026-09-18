# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

"""Shared document IDs and pseudonymous author hashes for platform pipelines."""

from __future__ import annotations

import hashlib

# Short platform prefixes keep doc_id readable and unique across the whole project.
PLATFORM_PREFIXES = {"reddit": "rd", "youtube": "yt", "bluesky": "bs"}


# doc_id = "<prefix>:<thing>:<native_id>", e.g. "yt:comment:Ugx123".
def make_doc_id(platform: str, thing: str, native_id: str) -> str:
    if not native_id:
        raise ValueError("native_id must not be empty")
    return f"{PLATFORM_PREFIXES[platform]}:{thing}:{native_id}"


def stable_analysis_doc_id(platform: str, thing: str, native_id: str) -> str:
    """Return a stable platform-prefixed ID for derived analysis joins."""

    return make_doc_id(platform, thing, native_id)


# Salted SHA-256: the same user always gets the same hash, but it cannot be reversed without the team salt.
def author_hash(salt: str, platform: str, native_author_id: str | None) -> str:
    if not native_author_id:
        return ""
    if not salt:
        raise ValueError("PSEUDONYM_SALT is empty; refusing to produce unsalted hashes")
    return hashlib.sha256(f"{salt}{platform}{native_author_id}".encode("utf-8")).hexdigest()[:16]
