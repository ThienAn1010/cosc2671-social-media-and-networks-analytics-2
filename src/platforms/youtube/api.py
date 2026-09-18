# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

"""Minimal YouTube Data API v3 client: GET with retries, error classification and a quota ledger.

The API key is read from SMFR/.env and is never written to logs, exceptions or manifests.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from http.client import HTTPException
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv

API_BASE = "https://www.googleapis.com/youtube/v3"
REPO_ROOT = Path(__file__).resolve().parents[3]
# search.list draws from its own 100-calls/day bucket; every other read shares 10,000 units/day.
SEARCH_RESOURCES = {"search"}
QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded"}
TRANSIENT_REASONS = {"backendError", "internalError", "processingFailure", "rateLimitExceeded", "userRateLimitExceeded"}
MAX_RETRIES = 6


class ApiError(RuntimeError):
    def __init__(self, status: int, reason: str, message: str, resource: str):
        super().__init__(f"{resource}: HTTP {status} {reason}: {message}")
        self.status = status
        self.reason = reason
        self.resource = resource


# Out of daily quota: callers stop cleanly and resume after the reset.
class QuotaExceededError(ApiError):
    pass


# Local --max-units budget reached (used for smoke runs).
class BudgetExceededError(RuntimeError):
    pass


# Counts requests per quota bucket; every request costs quota, including ones that fail.
@dataclass
class Ledger:
    max_units: int | None = None
    units: int = 0
    search_calls: int = 0
    calls_by_resource: dict[str, int] = field(default_factory=dict)

    def check_budget(self) -> None:
        if self.max_units is not None and self.units + self.search_calls >= self.max_units:
            raise BudgetExceededError(f"local budget of {self.max_units} requests reached")

    def charge(self, resource: str) -> None:
        if resource in SEARCH_RESOURCES:
            self.search_calls += 1
        else:
            self.units += 1
        self.calls_by_resource[resource] = self.calls_by_resource.get(resource, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {"units": self.units, "search_calls": self.search_calls, "calls_by_resource": dict(self.calls_by_resource)}


def load_api_key() -> str:
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not key:
        raise SystemExit("YOUTUBE_API_KEY missing: add it to SMFR/.env (docs/10-youtube-api-setup-guide.md)")
    return key


# Pull the machine-readable reason out of a Google error body.
def parse_error(body: str) -> tuple[str, str]:
    try:
        error = json.loads(body).get("error", {})
    except (json.JSONDecodeError, AttributeError):
        return "unknown", body[:200]
    errors = error.get("errors") or [{}]
    return str(errors[0].get("reason") or error.get("status") or "unknown"), str(error.get("message", ""))[:300]


def api_get(resource: str, params: dict[str, Any], ledger: Ledger, api_key: str | None = None) -> dict[str, Any]:
    key = api_key or load_api_key()
    url = f"{API_BASE}/{resource}?{urlencode({**params, 'key': key})}"
    for attempt in range(MAX_RETRIES + 1):
        ledger.check_budget()
        ledger.charge(resource)
        try:
            with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            exc.close()
            reason, message = parse_error(body)
            message = message.replace(key, "<redacted>")
            # "from None" drops the original exception, which carries the URL (and so the key).
            if reason in QUOTA_REASONS:
                raise QuotaExceededError(exc.code, reason, message, resource) from None
            if (exc.code >= 500 or exc.code == 429 or reason in TRANSIENT_REASONS) and attempt < MAX_RETRIES:
                time.sleep(min(60, 5 * (attempt + 1)))
                continue
            raise ApiError(exc.code, reason, message, resource) from None
        # OSError covers URLError, timeouts and dropped connections (RemoteDisconnected); HTTPException covers malformed responses.
        except (OSError, HTTPException) as exc:
            if attempt < MAX_RETRIES:
                time.sleep(min(60, 5 * (attempt + 1)))
                continue
            raise ApiError(0, "network", type(exc).__name__, resource) from None
    raise AssertionError("unreachable")


# Yield (page_token_used, payload) for every page of a list endpoint.
def iter_pages(
    resource: str,
    params: dict[str, Any],
    ledger: Ledger,
    api_key: str | None = None,
    max_pages: int | None = None,
) -> Iterator[tuple[str | None, dict[str, Any]]]:
    token: str | None = None
    pages = 0
    while True:
        payload = api_get(resource, {**params, "pageToken": token} if token else dict(params), ledger, api_key)
        yield token, payload
        pages += 1
        token = payload.get("nextPageToken")
        if not token or (max_pages is not None and pages >= max_pages):
            return


def run_self_test() -> None:
    ledger = Ledger()
    ledger.charge("search")
    ledger.charge("commentThreads")
    ledger.charge("commentThreads")
    assert (ledger.search_calls, ledger.units) == (1, 2)
    assert parse_error('{"error": {"message": "m", "errors": [{"reason": "quotaExceeded"}]}}') == ("quotaExceeded", "m")
    assert parse_error("not json")[0] == "unknown"
    try:
        Ledger(max_units=0).check_budget()
    except BudgetExceededError:
        pass
    else:
        raise AssertionError("a budget of 0 must stop before any request")
    print("self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run offline checks and exit")
    parser.add_argument("--test-video", help="Live check: one videos.list call (1 unit) for this video ID")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0
    if args.test_video:
        ledger = Ledger()
        payload = api_get("videos", {"part": "snippet,statistics", "id": args.test_video}, ledger)
        items = payload.get("items", [])
        if not items:
            print("no video returned")
            return 1
        print(f"OK: {items[0]['snippet']['title']} | commentCount: {items[0]['statistics'].get('commentCount')}")
        print(f"ledger: {ledger.as_dict()}")
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
