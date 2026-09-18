# Bluesky workstream

Owner: Hung. Code `src/platforms/bluesky/`, config `config/bluesky/`, tests `tests/platforms/bluesky/`.

## Documents

| File | What it answers |
|---|---|
| `collection_and_cleaning_log.md` | What has been done to the data, in order |
| `data_quality_audit.md` | What is wrong with the data, and how bad |
| `frame_validation.md` | How good the frame lexicons are — **read before quoting any frame figure** |
| `probe_findings.md` | What the pre-flight probe measured about the platform |

## Run order

```bash
python -m src.platforms.bluesky.collect_bluesky           # 1  posts
python -m src.platforms.bluesky.collect_baseline          # 2  denominator
python -m src.platforms.bluesky.enrich_threads            # 3  replies
python -m src.platforms.bluesky.recollect_capped_threads  # 3b repair; run before 4
python -m src.platforms.bluesky.collect_profiles          # 4  authors
python -m src.platforms.bluesky.collect_follows           # 5  follow graph
python -m src.platforms.bluesky.normalise                 #    raw -> analysis tables
python -m src.platforms.bluesky.build_analysis_tables
python -m src.platforms.bluesky.build_frames
python -m src.platforms.bluesky.anonymise                 #    -> data/processed/bluesky
```

## Relationship to other platform workstreams

This workstream keeps its own table structure and is analysed independently.
It adopts the conventions below for report consistency, but there is no
cross-platform raw-data join:

| Convention | How |
|---|---|
| Pseudonymised users | `src.shared.ids.author_hash` with a Bluesky-specific `PSEUDONYM_SALT` |
| Document ids | `bs:<post\|reply\|quote>:<rkey>`, uniqueness asserted at runtime |
| Event calendar and window | `config/events.csv`, `[date-7, date+21)` via `src.shared.events` |
| No silent drop | excluded rows stay, with `exclusion_reason` |
| Forbidden fields | handles, display names, avatars, profile URLs dropped; emails and phones → `<PII>` |
| Sensitive content | `is_sensitive_content` from Bluesky's own moderation labels |
| Manifest | `data/processed/bluesky/prepare_manifest.json` |

**One deliberate divergence.** Event columns are keyed on `day`, not
`created_date_utc`. Bluesky's `created_at` is set by the posting client and 899
posts claim dates before Bluesky existed; `day` is the collection cell or index
day. The manifest records this so nobody assumes event dates are derived from
post timestamps.

`python -m src.platforms.bluesky.anonymise` needs `PSEUDONYM_SALT` in `SMFR/.env` and
refuses to run without it. The salt is this workstream's own, not the team
value: nothing joins across platforms, so no hash needs to match another
workstream's. Re-running this step to reproduce the shared hashes needs the same
value, obtained privately from the owner.

**Not in the processed output:** the follow graph (raw only, in
`data/raw/follows/`) and the hand-coding sheets and codes used for frame
validation. Only the resulting scores ship.
