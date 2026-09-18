# YouTube data dictionary

Tables in `data/processed/youtube/` (full) and `data/sample/youtube/` (submission sample, same columns).
This document is the authoritative schema for the YouTube tables. `yt_*` and diagnostic columns are YouTube-specific.
API field names refer to YouTube Data API v3 resources (`video`, `commentThread`, `comment`).

## 1. `documents` — one row per video, top-level comment or reply (46 columns)

| Column | Type | Meaning | Source field | Rule |
|---|---|---|---|---|
| `doc_id` | str | Project-wide unique ID `yt:<thing>:<id>` | `id` | `make_doc_id` |
| `platform` | str | Always `youtube` | — | constant |
| `thing` | str | `video`, `comment` (top-level) or `reply` | resource type | C1, C17 |
| `native_id` | str | YouTube ID of the video/comment | `id` | — |
| `parent_doc_id` | str | Document replied to: video for top-level comments, top-level comment for replies, empty for videos | `snippet.parentId`, `snippet.videoId` | C1 |
| `root_doc_id` | str | Video the thread belongs to | `snippet.videoId` | — |
| `container_id` | str | Publishing channel `yt:channel:<id>` (organisation, not a user) | video `snippet.channelId` | — |
| `author_hash` | str | Salted SHA-256 (16 hex) of the author channel ID | `snippet.authorChannelId.value` | C16 |
| `created_utc` | str | Publish time, ISO-8601 UTC | `snippet.publishedAt` | C11 |
| `created_date_utc` | str | Publish date (UTC) | `snippet.publishedAt` | — |
| `text_raw` | str | Original text with handles → `@user`, emails/phones → `<PII>` (URLs kept) | `snippet.textOriginal` (fallback `textDisplay`); video: title + description | C4, C5 |
| `text_clean` | str | Repaired text for NLP: leading reply handle removed, URLs → `<URL>`, handles → `@user`, PII masked; case/punctuation/emoji kept | derived | C2, C4, C5 |
| `lang` | str | Top language detected by lingua (lowercase) | derived | C6 |
| `lang_confidence` | float | Confidence of the top language (0–1) | derived | C6 |
| `is_english` | bool | Top language is English | derived | C6 |
| `engagement` | Int64 | Likes on the comment/video | `snippet.likeCount` / `statistics.likeCount` | — |
| `reply_count` | Int64 | Top-level: `totalReplyCount`; video: `commentCount`; replies: null | API | — |
| `event_id_nearest` | str | Calendar-nearest event to `created_date_utc` (ties → earlier event) | `config/events.csv` | C12 |
| `days_from_event` | int | `created_date − event_date` for `event_id_nearest` (negative = before) | derived | C12 |
| `event_window` | str | Event whose window `[date−7, date+21)` contains the date, else `none` | derived | C12 |
| `jurisdiction_hint` | str | Jurisdiction of the video's event (`GB`, `AU`, `US`, `unknown`) | seed list + events | C14 |
| `jurisdiction_source` | str | `discovery_query` (from the event the video was sampled for) or `none` | derived | C14 |
| `exclusion_status` | str | Highest-priority reason: `empty` > `sensitive` > `spam` > `bot_or_owner` > `duplicate` > `non_english` > `language_uncertain` > `eligible` | derived | all |
| `is_exact_duplicate` | bool | Same author already posted this text (≥30 chars); first copy is False | derived | C9 |
| `is_near_duplicate` | boolean (null) | Not computed for YouTube; always null | — | — |
| `is_bot_or_owner` | bool | Author is the channel that published the video | derived | C7 |
| `is_spam` | bool | Link-only, or promotion term together with a link/contact | derived | C5, C8 |
| `is_url_only` | bool | Text is nothing but links | derived | C5 |
| `is_sensitive_content` | bool | Comment shares a URL together with an adult term or platform name; videos exempt | derived | C10 |
| `bypass_term_hits` | str | Pipe-joined bypass terms found (weak pre-tag) | derived | C15 |
| `collection_run_id` | str | Raw collection run folder | manifest | — |
| `yt_video_id` | str | YouTube video ID | `snippet.videoId` | — |
| `yt_video_type` | str | `news`, `commentary`, `tech_explainer`, `official`, `other` (from screening) | `config/youtube/seed_videos.csv` | screening |
| `yt_video_event_id` | str | Event the video was sampled for | seed list | C12 |
| `yt_days_from_video_event` | Int64 | `created_date − date of yt_video_event_id` | derived | C12 |
| `yt_is_edited` | bool | `updatedAt != publishedAt` | `snippet.updatedAt` | C11 |
| `yt_is_late_comment` | bool | More than 90 days after the video's event | derived | C13 |
| `yt_like_count` | Int64 | Same as `engagement` (kept for clarity in YouTube-only analyses) | `likeCount` | — |
| `yt_source` | str | Where the row came from: `videos.list`, `commentThreads.list`, `commentThreads.list:inline`, `comments.list` | collection | C1 |
| `lang_en_confidence` | float | lingua confidence for English specifically | derived | C6, D-015 |
| `is_language_uncertain` | bool | Top confidence < 0.80 or text < 24 characters | derived | C6 |
| `has_leading_mention` | bool | Reply started with an `@handle` (removed from `text_clean`) | derived | C4 |
| `url_count` | int | URLs found in the text | derived | C5 |
| `pii_count` | int | Emails/phone numbers masked | derived | C5 |
| `spam_reason` | str | Pipe-joined: `url_only`, `promo_with_contact` | derived | C8 |
| `copy_paste_author_count` | int | Number of distinct authors who posted this exact normalised text (≥20 chars) | derived | C8, D-016 |

Recommended analysis masks:
- Strict: `exclusion_status == "eligible"`.
- Inclusive English (D-015): `is_english & exclusion_status.isin(["eligible", "language_uncertain"])`.

## 2. `interactions` — one directed edge per reply (17 columns)

| Column | Type | Meaning | Rule |
|---|---|---|---|
| `edge_id` | str | `<source_doc_id>-><target_doc_id>` | — |
| `platform` | str | `youtube` | — |
| `edge_type` | str | `reply` | — |
| `source_author_hash` | str | Author of the reply | C16 |
| `target_author_hash` | str | Author being replied to | C16 |
| `source_doc_id` | str | The reply | — |
| `target_doc_id` | str | Comment replied to: most recent earlier comment by the `@handle` author, else the top-level comment | edges.py |
| `root_doc_id` | str | Video of the thread | — |
| `container_id` | str | Publishing channel | — |
| `created_utc` | str | Time of the reply | — |
| `event_id_nearest` | str | From the reply document | C12 |
| `event_window` | str | From the reply document | C12 |
| `is_self_loop` | bool | Source and target are the same author | — |
| `target_resolution` | str | `handle_match` (leading `@handle` matched an earlier commenter) or `thread_root` | edges.py |
| `yt_video_event_id` | str | Event of the video | C12 |
| `yt_video_type` | str | Video type | — |
| `source_exclusion_status` | str | `exclusion_status` of the reply, for filtering in Section 5 | — |

Direction: replier → replied-to. Edges are not aggregated or filtered; weights and filters are analysis decisions.

## 3. `author_video` — commenter ↔ video bipartite counts (8 columns)

| Column | Type | Meaning |
|---|---|---|
| `author_hash` | str | Commenter |
| `video_doc_id` | str | Video (`yt:video:<id>`) |
| `n_comments` | int | Comments + replies by this author on this video (all statuses) |
| `n_eligible` | int | Of which `exclusion_status == "eligible"` |
| `first_created_utc` | str | Author's first comment time on this video |
| `yt_video_event_id` | str | Event of the video |
| `yt_video_type` | str | Video type |
| `container_id` | str | Publishing channel |
