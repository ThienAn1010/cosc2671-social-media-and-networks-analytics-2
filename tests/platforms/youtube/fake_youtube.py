# Student name: Le Minh Trung Kien
# Student ID: s3651471
# Assignment 2 - Social Media and Networks Analytics

"""Offline stand-in for the YouTube Data API used by the YouTube tests; nothing here touches the network."""

import io
import json
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.payload


def http_error(status, reason, message="fake error"):
    body = json.dumps({"error": {"code": status, "message": message, "errors": [{"reason": reason}]}}).encode("utf-8")
    return HTTPError("https://example.test", status, reason, {}, io.BytesIO(body))


# Routes each request to a handler by resource name; a handler returns a payload dict or (status, reason).
class FakeYouTube:
    def __init__(self, handlers):
        self.handlers = handlers
        self.calls = []

    def __call__(self, request, timeout=None):
        parsed = urlparse(request.full_url)
        resource = parsed.path.rsplit("/", 1)[-1]
        params = {key: values[0] for key, values in parse_qs(parsed.query).items()}
        self.calls.append((resource, params))
        result = self.handlers[resource](params)
        if isinstance(result, tuple):
            raise http_error(*result)
        return FakeResponse(result)

    def calls_to(self, resource):
        return [params for name, params in self.calls if name == resource]


def fake_video(video_id, comment_count=100, published_at="2025-07-26T10:00:00Z", title="Online Safety Act age verification explained", channel="Fake News"):
    return {
        "id": video_id,
        "snippet": {"title": title, "description": "", "channelTitle": channel, "channelId": f"ch-{video_id}", "publishedAt": published_at},
        "statistics": {"viewCount": "1000", "commentCount": str(comment_count)},
        "contentDetails": {"duration": "PT5M"},
    }


def fake_thread(thread_id, video_id, total_replies=0, inline=0):
    return {
        "id": thread_id,
        "snippet": {"videoId": video_id, "totalReplyCount": total_replies, "topLevelComment": {"id": thread_id, "snippet": {"textOriginal": "fake"}}},
        "replies": {"comments": [{"id": f"{thread_id}.inline{index}"} for index in range(inline)]},
    }


def fake_reply(reply_id, parent_id):
    return {"id": reply_id, "snippet": {"parentId": parent_id, "textOriginal": "fake reply"}}
