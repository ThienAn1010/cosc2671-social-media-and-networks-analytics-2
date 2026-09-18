# YouTube collection QA

- Run: `youtube_20260913_160448`
- Generated: 2026-09-13T23:40:37Z by `python -m src.platforms.youtube.qa`
- Sessions: [{'started_at_utc': '2026-09-13T16:04:48Z', 'units': 1173, 'stop_reason': ''}, {'started_at_utc': '2026-09-13T16:31:10Z', 'units': 960, 'stop_reason': ''}]

| Gate | Check | Result | Evidence |
|---|---|---|---|
| G1 | Manifest complete, no failed videos | PASS | completion=complete, videos={'complete': 134}, reply threads={'complete': 1493} |
| G2 | Coverage per video ≥ 0.7 | PASS | min=1.0, median=1.0, max=1.0, flagged=0 |
| G3 | No duplicate comment IDs | PASS | {'thread_rows': 55305, 'thread_duplicate_rows': 0, 'fetched_reply_rows': 25226, 'fetched_reply_duplicate_rows': 0, 'inline_replies_also_fetched': 7455} |
| G4 | Every reply has its top-level comment | PASS | {'replies': 38289, 'missing_parent': 0, 'share_with_parent': 1.0} |
| G5 | Timestamps parse; not before video publish | FLAG | {'comments': 93594, 'parse_failures': 0, 'before_video_publish': 28, 'earliest_utc': '2025-06-27T03:22:53Z', 'latest_utc': '2026-09-13T12:17:20Z'} |
| G6 | Replies vs totalReplyCount (gap > 20% flagged) | PASS | {'threads_with_replies': 8900, 'expected_replies': 38289, 'collected_replies': 38289, 'ratio': 1.0, 'threads_gap_over_20pct': 0} |
| G7 | Manual spot-check of 30 comments in the YouTube UI | PENDING (manual) | sheet: `data/interim/youtube/qa/spot_check_g7.csv` |
| G8 | No API key pattern in raw, processed or docs | PASS | hits=[] |

## Coverage by event

| event_id | videos | api_comments | collected | min_coverage | coverage |
|---|---|---|---|---|---|
| E1 | 13 | 2014 | 2014 | 1.0 | 1.0 |
| E2 | 50 | 40534 | 40534 | 1.0 | 1.0 |
| E3 | 50 | 43254 | 43254 | 1.0 | 1.0 |
| E4 | 14 | 4044 | 4044 | 1.0 | 1.0 |
| E5 | 7 | 3748 | 3748 | 1.0 | 1.0 |

## Videos below coverage threshold

None.

## Notes

- Coverage can exceed 1.0 slightly when comments arrive between the videos.list refresh and the thread crawl.
- Replies counted once across embedded thread replies and comments.list results.
- G5: a video's publishedAt is when it became public. Creators who give members or patrons early access receive comments before that time, so a small count on non-live creator videos is expected and is not a collection error.
