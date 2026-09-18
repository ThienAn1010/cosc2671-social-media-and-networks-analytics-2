# Student name: Hung
# Student ID: s4199510
# Assignment 2 - Social Media and Networks Analytics

"""Talking to the Bluesky API safely.

Both collection passes -- searching for posts and walking reply trees -- need
the same two things: convert an SDK response into plain JSON, and retry the
handful of failures that are actually worth retrying. They live here so the
behaviour is identical in both, rather than drifting apart in two copies.
"""

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from atproto_client.exceptions import BadRequestError, NetworkError, RequestErrorBase

log = logging.getLogger("api")

# The atproto SDK talks over httpx, which logs a full URL line for every single
# request at INFO. Over a 7,296-cell backfill that buries the one line per cell
# that actually tells you whether collection is going well, at a ratio of
# roughly ten to one. Its warnings and errors still come through.
logging.getLogger("httpx").setLevel(logging.WARNING)

# Failures worth retrying: rate limiting, and server-side errors that may not
# recur. Everything else (bad request, expired session) is not fixed by trying
# again, so it propagates.
RETRYABLE_STATUS = (429, 500, 502, 503, 504)

# ...AND network-level failures, which is a bug fix rather than a refinement.
#
# FOUND THE HARD WAY, 2026-09-13: the thread-enrichment pass died after 2,435
# threads with an uncaught InvokeTimeoutError. The exception hierarchy is
#
#     InvokeTimeoutError -> NetworkError -> RequestErrorBase
#
# so it WAS caught by the `except RequestErrorBase` below. The problem was the
# next two lines: a timeout carries no HTTP response, so `status` came out None,
# `None not in RETRYABLE_STATUS` was true, and it re-raised on the first
# attempt. The single most retryable class of failure was the one class never
# retried.
#
# This went unnoticed in Assignment 1 because its Bluesky collection ran in
# minutes. A pass that runs for hours will meet a dropped connection eventually;
# over ~8 hours of collection it is a certainty, not a risk.
#
# Note A1's Arctic Shift helper (common/http.py) always handled this correctly
# for Reddit -- it catches httpx.TransportError explicitly. The atproto side
# simply never grew the equivalent, because nothing had forced it to.
MAX_RETRIES = 5

# Login is retried harder: a failed request costs one cell, a failed login
# costs the whole run.
LOGIN_MAX_RETRIES = 8


def to_dict(model: Any) -> dict:
    """Convert an SDK response model into a JSON-safe dictionary.

    `mode="json"` converts datetimes to strings during the dump rather than
    leaving Python objects for json.dumps to coerce later. The raw file is then
    genuinely JSON-native, which matters because raw is the one layer we never
    regenerate.
    """
    try:
        return model.model_dump(mode="json")
    except TypeError:  # older pydantic
        return json.loads(model.json())


def call_with_retry(fn: Callable[[dict], Any], params: dict) -> Any:
    """Call an SDK endpoint, backing off on rate limits and transient errors.

    On 429 we honour the server's `Retry-After` header when it sends one -- it
    knows better than any number we would invent.

    Anything non-retryable propagates. A collector that quietly swallows errors
    and reports success while returning nothing is the worst possible outcome
    here: you would not discover it until analysis, by which point the posts
    are gone.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return fn(params)
        except RequestErrorBase as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)

            # A NetworkError (timeout, dropped connection, DNS blip) has no HTTP
            # response at all, so it can never match a status code. It is
            # retryable by nature -- nothing about the request was wrong.
            retryable = isinstance(exc, NetworkError) or status in RETRYABLE_STATUS
            if not retryable or attempt == MAX_RETRIES - 1:
                raise

            # Exponential backoff: 2s, 4s, 8s, 16s. Overridden by Retry-After.
            wait = 2 ** (attempt + 1)
            headers = getattr(response, "headers", {}) or {}
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after:
                try:
                    wait = max(wait, int(retry_after))
                except (TypeError, ValueError):
                    pass

            # Name the failure. "HTTP None" told us nothing while this bug was
            # being diagnosed; the exception class is what identifies a timeout.
            what = f"HTTP {status}" if status else type(exc).__name__
            log.warning("%s -- backing off %ss (attempt %s/%s)",
                        what, wait, attempt + 1, MAX_RETRIES)
            time.sleep(wait)

    raise RuntimeError("unreachable")  # loop either returns or raises


def login_with_retry(client: Any, handle: str, password: str) -> None:
    """Log in, retrying transient network failures.

    Separate from `call_with_retry` because `client.login()` is not an endpoint
    call -- it takes no params dict and returns a session, so it does not fit
    that signature.

    FOUND THE HARD WAY, 2026-09-13, immediately after fixing the timeout bug
    below: the enrichment pass was restarted and died again in under two
    minutes, this time inside `com.atproto.server.createSession`. Every endpoint
    call in this project was wrapped in a retry; the login that precedes them
    all was not, so a blip during startup killed a run before it did any work at
    all. That is the most annoying possible time to fail, because a resumable
    job that cannot start is not resumable in any useful sense.

    Login is retried harder than a normal call. A failed request costs one cell;
    a failed login costs the entire run.
    """
    for attempt in range(LOGIN_MAX_RETRIES):
        try:
            client.login(handle, password)
            return
        except RequestErrorBase as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            retryable = isinstance(exc, NetworkError) or status in RETRYABLE_STATUS
            # A bad password is not a blip. Retrying it four more times just
            # delays the error and risks tripping account protections.
            if not retryable or attempt == LOGIN_MAX_RETRIES - 1:
                raise
            wait = min(60, 2 ** (attempt + 1))
            what = f"HTTP {status}" if status else type(exc).__name__
            log.warning("Login failed (%s) -- retrying in %ss (attempt %s/%s)",
                        what, wait, attempt + 1, LOGIN_MAX_RETRIES)
            time.sleep(wait)


def is_gone(exc: Exception) -> bool:
    """True when a 400 means "this subject no longer exists", not "bad request".

    FOUND THE HARD WAY, 2026-09-13: enrichment died after 3,938 threads on

        BadRequestError: 400 NotFound: Post not found: at://did:plc:.../3meja2fupfk27

    `call_with_retry` is right to refuse to retry a 400 -- nothing about the
    request will change. But a root post deleted between collection and
    enrichment is not a programming error, it is the survivorship attrition this
    project already measures elsewhere. It arrives as an ERROR rather than as
    the tombstone the walker was written to expect, which is why it killed the
    run instead of being counted.

    The check is deliberately narrow. A 400 carrying NotFound/Could not find
    means the subject is gone; any other 400 is a genuine bug in the request and
    must still crash loudly, because silently skipping malformed requests would
    hide a fault that affects every record.
    """
    if not isinstance(exc, BadRequestError):
        return False
    text = str(getattr(exc, "response", None) or exc)
    lowered = text.lower()
    return ("notfound" in lowered.replace(" ", "")
            or "could not find" in lowered
            or "not found" in lowered)
