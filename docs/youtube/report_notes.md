# YouTube — notes for the report (not final prose)

Answers the data-statement bullets of the brief (§3.3) for the YouTube source. Every number comes from
`collection_manifest.json`, `prepare_manifest.json`, `edges_manifest.json` or `collection_qa.md`.
Decision labels D-xxx are provenance labels from the YouTube collection work; cleaning rules C1–C17 are documented in [cleaning_decisions.md](cleaning_decisions.md).

## 1. Where the data came from
- YouTube Data API v3 (official): `search.list`, `videos.list`, `commentThreads.list`, `comments.list`.
- Public comments on 134 English-language videos about age-verification policy, published inside the five event windows.

## 2. How it was collected
- **Discovery:** 22 queries across the five events (`config/youtube/queries.json`) + 8 supplementary E4/E5 queries (D-013), each run with the event's region code and globally, `order=viewCount`, `relevanceLanguage=en`, restricted to the event window `[event−7d, event+21d)`. 60 search calls; 546 candidate videos.
- **Screening:** rule-based auto-flags (outside window, <20 comments, no topic keyword, non-English audio) + title review; up to 50 videos per event by views with ≥5 per main channel type where available (D-012). Seed list frozen in `config/youtube/seed_videos.csv`.
- **Comments:** every top-level thread (`commentThreads.list`, 100/page, full pagination) and every reply (`comments.list` for threads whose replies were not fully embedded), largest threads first. Resume-safe, deduplicated by ID.
- **Quota:** 2,133 units for the comment crawl (limit 10,000/day); one crash on a dropped connection was fixed and resumed without duplicates.

## 3. Collection date and period covered
- Collected 2026-09-13 (UTC); search results are a snapshot at that time.
- Videos published in the five windows between 2025-06-20 and 2026-03-30.
- Comment timestamps span 2025-06-27 to 2026-09-13; 2,132 comments arrived more than 90 days after their video's event (flagged).

| Event | Videos | Comments + replies | In event window |
|---|---|---|---|
| E1 (US, placebo) | 13 | 2,014 | 1,892 |
| E2 (UK enforcement) | 50 | 40,534 | 35,508 |
| E3 (AU under-16 ban) | 50 | 43,254 | 42,308 |
| E4 (UK Lords, placebo) | 14 | 4,044 | 4,142* |
| E5 (AU adult-content codes) | 7 | 3,748 | 3,984* |

\* "In event window" counts rows by comment date, so comments on other events' videos that fall inside this window are included.

## 4. Main fields used
- Text: `text_clean` (NLP), `text_raw` (audit).
- Time: `created_utc`, `event_window`, `yt_video_event_id`, `yt_days_from_video_event`.
- Structure: `doc_id`, `parent_doc_id`, `root_doc_id`, `container_id`, `author_hash`.
- Context and filters: `yt_video_type`, `jurisdiction_hint`, `exclusion_status`, language fields, `bypass_term_hits`, `engagement`.
- Full definitions: `data_dictionary.md`.

## 5. How the data supports the network component
- **Reply graph** (`interactions.parquet`): 38,289 directed edges, replier → author replied to. 25% of targets are resolved from the reply's leading `@handle` (reply to a reply), 75% point to the thread's top-level author. 1,732 self-loops kept and flagged.
- Per video event (self-loops removed): E2 10,561 users / 13,856 directed pairs; E3 9,901 / 14,417; E1 672 / 940; E4 1,119 / 1,265; E5 808 / 1,040. Reciprocity 0.10–0.15; largest weak component holds 72–88% of users.
- **Bipartite commenter ↔ video** (`author_video.parquet`): 56,120 users, 64,230 user-video pairs; 5,645 users commented on ≥2 videos — the basis for channel/video audience-overlap projections and cross-video bridges.
- Suggested framing for the Network lead: directed (who answers whom), weighted by repeated replies between a pair; projection undirected, weighted by shared commenters.

## 6. How the data supports the NLP component
- 93,594 comments and replies + 134 video titles/descriptions.
- Strict eligible 52,553 comments; inclusive English 87,221 (D-015) — enough per event for sentiment trajectories and topic models (smallest event E5: 2,188 strict / 3,556 inclusive).
- Text kept with case, punctuation and emoji for transformer/lexicon sentiment; URLs, handles and personal data masked.
- 4,256 comments contain bypass vocabulary (E2 2,414, E3 1,015) for the diffusion analysis; video type (news / commentary / tech explainer / official) allows frame comparison across channel types.

## 7. Limitations, missing data, sampling issues
1. **No user location:** jurisdiction is inferred from the event a video was found for, not from the commenter.
2. **Search sampling:** `search.list` returns a ranked, non-exhaustive, non-reproducible sample; most-viewed videos and large channels are favoured. Query bank expanded after the first pass for E4/E5 (D-013).
3. **Uneven events:** E1 (13 videos), E4 (14) and E5 (7) are below the 20-video target; E5 — a treatment event — is the thinnest arm, so YouTube results for E5 should be treated as indicative.
4. **Comment accumulation:** comments keep arriving for months; 2,132 are more than 90 days after the event. Time-series work should use comment time and test sensitivity to late comments.
5. **Language detection:** 42% of rows are `language_uncertain` (87% have English as top language) because short comments split lingua's confidence; results should be shown under strict and inclusive masks.
6. **Reply targets are inferred:** YouTube threads are two levels deep; replies to replies are resolved from the leading `@handle`, which misses replies that don't tag anyone.
7. **Unobservable comments:** comments held for review, deleted, or filtered by YouTube are not returned; coverage against `commentCount` was 1.00 for every video at collection time, but earlier deletions cannot be seen.
8. **Timing oddity:** 28 comments are timestamped before their video's public release (creator early access); kept and documented (QA G5).
9. **Platform demographics:** YouTube commenters are not a representative sample of UK/AU/US publics.
10. **Terms of service:** raw API data is stored locally only and deleted by 2026-10-13; the submission includes a pseudonymised sample (D-009).
11. **Rule changes after precision checks:** sensitive-content and copy-paste rules were narrowed after inspecting flagged rows, before any analysis (D-014, D-016).
