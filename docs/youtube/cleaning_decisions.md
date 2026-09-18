# YouTube cleaning decisions (task Y7)

- Input run: `youtube_20260913_160448` (134 seed videos, collection `complete`, QA in `collection_qa.md`)
- Script: `python -m src.platforms.youtube.prepare --collect-run data/raw/youtube/youtube_20260913_160448`
- Output: `data/processed/youtube/documents.parquet` + `prepare_manifest.json` (all numbers below come from that manifest or from the parquet)
- Rules: C1–C17 are documented below; schema: [YouTube data dictionary](data_dictionary.md)
- Rule changes after precision checks are recorded below (D-014, D-015, D-016)

## Overview

| | Comments | Replies | Videos | Total |
|---|---|---|---|---|
| Rows | 55,305 | 38,289 | 134 | 93,728 |
| eligible | 31,626 | 20,927 | 126 | 52,679 |
| language_uncertain | 23,204 | 16,446 | 8 | 39,658 |
| duplicate | 375 | 407 | 0 | 782 |
| bot_or_owner | 27 | 355 | 0 | 382 |
| empty | 5 | 118 | 0 | 123 |
| non_english | 58 | 28 | 0 | 86 |
| spam | 10 | 8 | 0 | 18 |
| sensitive | 0 | 0 | 0 | 0 |

No row is deleted. `exclusion_status` is the highest-priority reason in this order:
empty → sensitive → spam → bot_or_owner → duplicate → non_english → language_uncertain → eligible.

### Analysable comments per video event (comments + replies)

| Event | Comments | Strict eligible | Share | Inclusive English* | Share |
|---|---|---|---|---|---|
| E1 | 2,014 | 1,076 | 0.534 | 1,865 | 0.926 |
| E2 | 40,534 | 22,663 | 0.559 | 37,907 | 0.935 |
| E3 | 43,254 | 24,255 | 0.561 | 40,100 | 0.927 |
| E4 | 4,044 | 2,371 | 0.586 | 3,793 | 0.938 |
| E5 | 3,748 | 2,188 | 0.584 | 3,556 | 0.949 |
| **Total** | **93,594** | **52,553** | 0.561 | **87,221** | 0.932 |

\* Inclusive mask (D-015): `is_english & exclusion_status in {eligible, language_uncertain}`.

---

## C1 — One row per comment ID
### Decision
Merge top-level comments, replies embedded in threads, and replies fetched with `comments.list`; when a reply exists in both places, keep the `comments.list` copy.
### Why this, not the alternative
Threads with ≤5 replies only have embedded replies, so ignoring them would lose replies; keeping both copies would double-count.
### Effect on the data
20,518 embedded + 25,226 fetched replies → 38,289 unique replies; 7,455 embedded copies replaced. Matches the sum of `totalReplyCount` exactly (QA G6).
### How to say it in the presentation
"Every comment appears exactly once, and the reply count matches what YouTube reports."

## C2 — Light text repair
### Decision
`textOriginal` (fallback `textDisplay`) → ftfy mojibake repair → Unicode NFC → HTML unescape → remove zero-width characters → collapse whitespace. Case, punctuation and emoji are kept.
### Why this, not the alternative
Transformer and lexicon sentiment models use capitals, "!!!" and emoji as intensity; lowercasing or stripping punctuation would remove signal.
### Effect on the data
Applied to all 93,728 rows. Zero-width spaces were common: they precede the `@handle` on most replies-to-replies.
### How to say it in the presentation
"We repaired broken characters but kept how people actually wrote, because tone matters for sentiment."

## C3 — Empty text
### Decision
Rows whose cleaned text is empty get `empty`.
### Why this, not the alternative
Nothing to analyse; keeping the row preserves the reply structure for the network.
### Effect on the data
123 rows (118 replies that consisted only of an `@handle`).
### How to say it in the presentation
"A reply that only tags someone has no text, but it still counts as an interaction in the network."

## C4 — Leading reply address
### Decision
For replies, remove a leading `@handle` from `text_clean`; replace every handle with `@user` in `text_raw` and `text_clean`.
### Why this, not the alternative
The leading handle is addressing, not content, and it is personal data. The handle is still used in memory by `edges.py` to find who was replied to.
### Effect on the data
16,322 replies had a leading handle. 0 unmasked handles remain (validated on every run).
### How to say it in the presentation
"We used '@name' to know who replied to whom, then removed names from the text."

## C5 — URLs and personal data
### Decision
URLs → `<URL>`; emails and phone numbers (≥9 digits) → `<PII>`. A comment with nothing but links is `spam` (`url_only`).
### Why this, not the alternative
Keeps sentence structure for models while removing identifiers. The digit threshold keeps year ranges like "2025-2026".
### Effect on the data
106 comments with URLs, 16 with personal data, 18 link-only spam.
### How to say it in the presentation
"Links and contact details are masked, so the text is safe to analyse and share."

## C6 — Language (D-015)
### Decision
lingua over all languages, same thresholds as the Reddit pipeline: confidence < 0.80 or text < 24 characters → `language_uncertain`; confident non-English → `non_english`. Stored: `lang`, `lang_confidence`, `lang_en_confidence`, `is_english`, `is_language_uncertain`.
### Why this, not the alternative
Same rule across platforms keeps English filtering comparable. Short YouTube comments spread confidence across similar languages, so "uncertain" is kept separate instead of being treated as non-English.
### Effect on the data
39,658 uncertain (87.4% have English as top language; 10,139 are shorter than 24 characters); only 86 confidently non-English. Strict eligible 52,553 vs inclusive English 87,221 comments.
### How to say it in the presentation
"Almost everything is English; we mark short or ambiguous comments separately and show results hold with and without them."

## C7 — Channel owner comments
### Decision
Comments whose author is the channel that published the video → `bot_or_owner`; kept as edges in the network.
### Why this, not the alternative
Publishers pinning or replying are not public discourse, but their replies are real interactions.
### Effect on the data
382 rows (355 replies), on 41 videos, mostly commentary creators answering their audience.
### How to say it in the presentation
"We separate creators talking to their audience from the audience itself."

## C8 — Promotion spam and shared slogans (D-016)
### Decision
`spam` = link-only comments or promotion terms (telegram, crypto, …) together with a link or contact. Identical text from ≥3 authors is recorded in `copy_paste_author_count`, not excluded.
### Why this, not the alternative
The first version excluded 91 "copy-paste" comments that were widely shared slogans ("It was never about the children", "The road to hell is paved with good intentions") — frame signals, not bots.
### Effect on the data
Spam fell from 109 to 18. 91 comments have `copy_paste_author_count ≥ 3` for sensitivity checks.
### How to say it in the presentation
"We removed advertising, not people repeating the same protest slogan."

## C9 — Same-author repeats
### Decision
The same author posting the same text (≥30 characters) again → later copies are `duplicate`; the first stays eligible.
### Why this, not the alternative
One person pasting a comment under many videos should not count as many opinions.
### Effect on the data
782 duplicates.
### How to say it in the presentation
"Each person's repeated comment counts once."

## C10 — Sensitive content (D-014)
### Decision
`sensitive` only when a comment shares a URL together with an adult term or adult platform name. Mentioning them without a link stays in the corpus. Videos are exempt.
### Why this, not the alternative
Term-only matching removed policy discussion ("neither Roblox nor Pornhub are included in the ban", "predators ask kids for nudes") that carries the child-safety and futility frames. News video descriptions link out routinely.
### Effect on the data
43 flagged in the first full run → 0 after the change; links to adult content would still be excluded.
### How to say it in the presentation
"For this topic, talking about adult sites is the debate itself; we only exclude people sharing links to them."

## C11 — Edited comments
### Decision
`yt_is_edited = updatedAt != publishedAt`; flag only.
### Why this, not the alternative
Edits can happen after the event, so timing analyses can test robustness without them.
### Effect on the data
6,986 edited comments.
### How to say it in the presentation
"We know which comments were edited later and can check they don't change the timeline."

## C12–C14 — Time and place tags
### Decision
Calendar tags from comment time: `event_id_nearest`, `days_from_event`, `event_window` (`[date−7, date+21)`). Sampling tags from the video: `yt_video_event_id`, `yt_days_from_video_event`, `yt_is_late_comment` (>90 days). `jurisdiction_hint` = jurisdiction of the video's event (`discovery_query`).
### Why this, not the alternative
Comments keep arriving after an event; analyses of reaction use the video event, daily series use comment time. YouTube has no user location, so jurisdiction is a hint only.
### Effect on the data
Inside event windows: E1 1,905 · E2 35,558 · E3 42,358 · E4 4,156 · E5 3,991 rows; 5,760 outside any window. 2,132 comments arrived more than 90 days after their video's event (3.1% E1, 4.0% E2, 0.8% E3, 2.1% E4, 0.6% E5).
### How to say it in the presentation
"Each comment knows both when it was written and which policy event its video covered."

## C15 — Bypass vocabulary
### Decision
Word-boundary matches of `bypass_terms` (vpn, proxy, fake id, get around, …) into `bypass_term_hits`.
### Why this, not the alternative
Input for the bypass-knowledge diffusion analysis; a weak pre-tag, not a frame label.
### Effect on the data
4,256 comments mention at least one term (E1 184 · E2 2,414 · E3 1,015 · E4 355 · E5 288); "vpn" dominates (2,916).
### How to say it in the presentation
"We tagged every mention of circumvention tools so we can see how that knowledge spreads."

## C16 — Pseudonymisation
### Decision
`author_hash = sha256(salt + "youtube" + channel ID)[:16]`; no display names, avatars, profile URLs, handles, emails or phone numbers in outputs.
### Why this, not the alternative
Keeps network analysis possible (same person = same hash) without identifying anyone; required by RULES R-B4 and the brief.
### Effect on the data
Validation fails the run if a private column or unmasked handle appears; both are 0.
### How to say it in the presentation
"We can follow a user's interactions without ever knowing who they are."

## C17 — Videos as documents
### Decision
Each video is a row (`thing=video`, title + description) and the root of its comment tree.
### Why this, not the alternative
Gives topic context per video and a root node for tree and bipartite networks.
### Effect on the data
134 rows: 126 eligible, 8 language_uncertain.
### How to say it in the presentation
"Videos are part of the dataset too, so every comment connects back to what people were reacting to."

## Not produced here
- `is_near_duplicate` is null: near-duplicate detection is not part of the YouTube rules.
- `text_topic` (lemmatised text) is left to the NLP lead.
