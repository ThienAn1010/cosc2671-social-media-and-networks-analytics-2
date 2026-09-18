# Data quality audit — Bluesky corpus

**Audited 2026-09-14, read-only.** Nothing in `data/` was modified. Every figure
below was recomputed from the files on disk rather than taken from a collection
log, because two of the issues found are cases where the log and the disk
disagree.

**Scope:** all 608 `posts_*.jsonl`, all 607 `thread_*.jsonl`, `profiles.jsonl`,
`follows.jsonl`, `daily_counts.jsonl`, and all five checkpoints — 2.5 GB,
338,637 post rows, 133,443 reply rows, 160,268 profiles, 3,717,888 follow edges,
3,040 baseline cells.

**Nothing here is fixed.** These are findings for the team to decide on. Several
of them are properties of the platform, not defects in the collection, and the
right response to those is a documented decision in the methods section rather
than a change to the data.

---

## Summary

| # | Issue | Severity | Status |
|---|---|---|---|
| 1 | Multi-word queries match scattered terms, not phrases | **Blocking** | **Decided** — series use phrase-exact rows only |
| 2 | Three viral meme waves, not one — the second-largest day in the corpus is undocumented | **Blocking** | **Decided** — excluded from the series; both enforcement events surface once removed |
| 3 | One spam account manufactures a three-day volume spike | **Blocking** | **Decided** — spam/repeater accounts excluded from the series |
| 4 | The documented reply count is wrong — 92,556 vs 133,732 actual | **Blocking** | **Resolved 2026-09-14** |
| 5 | `created_at` is unreliable as the time axis | Material | **Resolved** — `day` is the axis; `indexed_at` also found unstable |
| 6 | 9,036 replies are also in the posts layer | Material | **Resolved** — `discovered_via`, `in_search`/`in_thread` |
| 7 | 39,794 duplicate post rows; dedup conflicts with per-query series | Material | **Resolved** — `posts` / `post_query` split |
| 8 | 10.4% of unique posts are exact-duplicate text | Material | Instrumented (`dup_text_n`, `is_dup_text`) |
| 9 | News-bot and bridged-account concentration | Material | **Decided** — kept; institutional voices are part of public attention |
| 10 | Engagement counts are a single snapshot, not a fixed-age measure | Material | **Withdrawn** — no age effect detectable; see correction below |
| 11 | Follow graph: censored out-degree, sparse induced subgraph | Material | **Decided** — reply graph is primary; follow graph in-degree only |
| 12 | 33 threads truncated at the 200-reply cap | Material | **Resolved 2026-09-14** |
| 13–20 | Minor issues and stale documentation | Minor | 13, 14, 16, 17, 19, 20 resolved or instrumented; 14 closed by `is_english` |

---

# Blocking issues

## 1. Multi-word queries are AND-of-terms, not phrases

`app.bsky.feed.search_posts` does not treat a multi-word query as a phrase. It
matches posts containing **all the terms anywhere**, in any order, across the
post text *and* its embed metadata (link titles, descriptions). Measured by
testing every collected row against its own query:

| Query | Rows | Contains the phrase | All terms, scattered | Only some terms |
|---|---:|---:|---:|---:|
| `age verification privacy` | 8,222 | **1.5%** | 98.5% | 0.0% |
| `age verification VPN` | 3,592 | **1.5%** | 98.4% | 0.1% |
| `porn ID check` | 266 | **3.4%** | 96.6% | 0.0% |
| `under-16 ban` | 11,084 | **6.0%** | 94.0% | 0.0% |
| `age check` | 35,369 | **10.6%** | 89.2% | 0.2% |
| `social media ban` | 70,624 | **45.7%** | 54.3% | 0.0% |
| `verify your age` | 13,435 | 84.8% | 15.2% | 0.0% |
| `Online Safety Act` | 23,685 | 91.0% | 9.0% | 0.0% |
| `age assurance` | 7,802 | 95.0% | 4.6% | 0.3% |
| `age verification` | 126,245 | 97.0% | 2.9% | 0.1% |
| `Ofcom` | 35,114 | 100.0% | — | — |
| `eSafety` | 3,199 | 100.0% | — | — |

Single-token queries are exact by construction. The longer the query, the less
it means. Real examples of rows that matched without containing the phrase:

- `under-16 ban` → *"There are bans that dumbfuck little gov'ts can declare, & then there are those that they can enforce…"*
- `age check` → *"ALRIGHT, I've finally got it all set up and filled; I'm opening the art gallery server…"*
- `porn ID check` → *"Hey, 18+ Florida people, get your ID and come check out my pages 🤣"*
- `age verification privacy` → *"This won't be the only thing they ban in the name 'The kids.' Plus sending you…"*

**Why it matters.** Three things the project currently assumes stop holding:

1. **The per-query series do not measure what their names say.** An `under-16 ban`
   series is a series of posts containing "under", "16" and "ban" somewhere —
   94% of the time not the phrase. Treating it as the Australian under-16
   measure would be wrong.
2. **Stratum B is not independent of stratum A.** `age verification privacy` and
   `age verification VPN` are, 98.5% of the time, just "`age verification` posts
   that also happen to contain the word privacy/VPN". The two-stratum design
   was meant to test whether frame-share is an artefact of the query list; as
   collected, the B terms are near-subsets of A rather than a contrast.
3. **The probe's 86–92% precision figures were measured on `age verification`**,
   which is 97% phrase-exact and therefore the best-behaved query in the set.
   They do not transfer to the short and multi-word queries.

Note a small counter-effect: a handful of rows (≤0.1%) contain no query term at
all, mostly French posts where *vérification* folds to *verification*. So the
backend also does accent folding and light stemming, and the phrase percentages
above are very slightly pessimistic. It does not change the picture.

**Nothing was lost in collection** — every row is on disk with its query
recorded, so any phrase-level filter can be applied at normalisation. The
decision the team needs: which queries are treated as measures and which are
treated only as recall nets.

## 2. Three viral meme waves, not one

The meme is `Age verification? [something that dates me]`. The existing
documentation describes it as a July 2025 event. It recurs three times:

| Day | Rows | Meme share | Corpus rank |
|---|---:|---:|---|
| **2025-07-11** | 9,082 | **80.2%** | #1 day |
| **2026-02-10** | 9,053 | **75.0%** | #2 day |
| **2025-09-10** | 3,238 | **75.1%** | — |
| 2025-07-12 | 1,025 | 68.4% | — |
| 2026-02-11 | 2,717 | 62.4% | — |
| 2025-09-15 | 795 | 37.4% | — |
| 2025-07-30 | 3,449 | 30.7% | — |

Monthly: **2025-07 = 28.4%**, **2025-09 = 20.2%**, **2026-02 = 28.1%** of all
rows. Corpus-wide **28,478 rows, 8.4%**.

**The 2026-02 wave is not mentioned anywhere in the current documentation**, and
it is the more dangerous of the two. It peaks on 2026-02-10, **twenty days
after event E4** (2026-01-21, UK announcement). An interrupted time series on
E4 would read a 9,053-post day as sustained public response to the
announcement. It is a nostalgia meme.

The existing note that the meme is "81% of 2025-07-11" is confirmed (80.2%),
but the row counts beside it are stale — that cell held 4,664 posts when the
probe ran and holds 9,082 after the page cap was raised and the cell
re-collected.

**Detection:** `^\s*age\s*verification\b` (case-insensitive) at the start of the
post. Matching only the colon form `age verification:` catches just 2,991 rows —
about a tenth of the wave — because the dominant form uses a question mark.

## 3. A single spam account manufactures a three-day spike

One account, `metapsyk.bsky.social`, posted the **identical 300-character text
2,361 times** across three consecutive days:

| Day | Total rows | Spam rows | Share |
|---|---:|---:|---:|
| 2025-10-25 | 754 | 513 | 68% |
| 2025-10-26 | 788 | 578 | 73% |
| 2025-10-27 | 1,522 | 1,270 | **83%** |

All 2,361 rows landed under `social media ban`, and none of them are about
social media bans. The text is *"IS DONALD TRUMP THE ANTI-CHRIST? CLICK
trumpantichrist.wixsite.com/adelajas/ban… SWITCH SOCIAL MEDIA …"* — it matched
because "social", "media" and "ban" all appear somewhere, the last of them
inside a URL path. This is issue 1 producing a concrete false spike.

Three days of the `social media ban` series are majority spam from one account.
Left in, late October 2025 shows a burst of ban discourse that did not happen.

## 4. The documented reply count is wrong — RESOLVED 2026-09-14

Documentation and the final log line both say **92,556 replies**. On disk:

| Source | Replies |
|---|---:|
| `docs/bluesky/probe_findings.md`, `README.md` | 92,556 |
| `thread_*.jsonl` rows | **133,443** |
| Unique reply URIs | **132,340** |
| Enrichment checkpoint, summed | **133,443** |

The checkpoint and the files agree exactly, so the data is intact and complete;
92,556 is simply the tally from the **last** of several enrichment runs, after
the pass had been restarted following the timeout, login and NotFound crashes.
Each restart resumed correctly from the checkpoint and appended its own share.

This matters because the figure is already quoted in two documents and would go
into the report — it understates the reply corpus by 31%.

**Corrected** in `README.md` and `docs/bluesky/probe_findings.md` on 2026-09-14, with a
note explaining the cause so the discrepancy is not "fixed" back by anyone
reading the old log. Post-repair the corpus holds **135,034 reply rows /
133,732 unique**.

---

# Material issues

## 5. `created_at` is not a trustworthy time axis; the file day is

Bluesky lets the client set `createdAt` to any value. In this corpus:

- **899 posts are dated before Bluesky existed**, ranging back to **2004-02-05**.
  Spread across every year 2004–2021.
- 1,234 rows in total have a `created_at` day different from the day-file they
  were collected into (908 far off, 326 off by one — the latter are ordinary
  midnight-boundary and timezone effects).
- No post is dated in the future.

The largest contributors are archive-import accounts (520 of the 899 are by
authors whose handle no longer resolves, plus `pranesh-archive.bsky.social`,
`mobileworldlive.com`).

Search indexes on `indexed_at`, and the collector files each row by the cell day
it was retrieved for, so **the filename day is the reliable axis** and
`created_at` is not. A daily series built on `record.created_at` would place 899
posts between 2004 and 2021. 99.6% of rows agree either way, so this is small in
volume and total in consequence if it is missed.

## 6. 9,036 replies are also in the posts layer

Posts and replies live in the same directory and share the same
`{_meta, post}` envelope, distinguishable only by `_meta.discovery`
(`"search"` vs `"thread"`). **9,036 reply rows (6.8%) have a URI that also
appears in the posts layer** — they are replies that independently matched a
search query.

Concatenating the two layers double-counts them. They are also legitimately
both things at once, so the right handling depends on the analysis: for volume
they are one post; for reply-graph work they are an edge and a node.

Related, and in the corpus's favour: **0 orphan replies** — every one of the
13,042 distinct parent URIs is present in the posts layer.

## 7. 39,794 duplicate post rows, and dedup fights the per-query series

| | |
|---|---:|
| Rows | 338,637 |
| Unique URIs | 298,843 |
| Duplicate rows | **39,794 (11.8%)** |
| URIs appearing under more than one query | **32,225** |

Multiplicity: 262,814 URIs once, 32,665 twice, 2,993 three times, 342 four
times, 28 five times, 1 six times.

This is by design — "raw is a log, not a set" — but it forces an order of
operations that is easy to get wrong. Deduplicating by URI **destroys the
per-query series**, because the surviving row keeps only one query. Any
frame-share or stratum comparison has to be computed before dedup, and any
volume or sentiment measure after it. The two cannot run off the same table.

## 8. 10.4% of unique posts are exact-duplicate text

After deduplicating by URI, **31,175 posts in 6,507 groups** share text
byte-identical to another post. Excluding the spam account from issue 3, the
largest groups are news-bot boilerplate reposted verbatim:

| Count | Text |
|---:|---|
| 2,361 | *IS DONALD TRUMP THE ANTI-CHRIST?…* (issue 3) |
| 1,165 | *On 22 January, GB News presenter Josh Howie repeated a homophobic slur…* |
| 535 | *Verify your age* |
| 387 | *🚨 Fire every 5 hours. Cheap goods from online marketplaces…* |
| 359 | *TalkTV is airing attacks on trans people without broadcasting the views…* |
| 264 | *Thames Water CEO went unchallenged on the BBC…* |

Any sentiment or topic model run on the raw text will weight these by their
repetition count.

## 9. News-bot and bridged-account concentration

Among unique posts:

- Top 10 authors: 9,786 posts (**3.3%**)
- Top 100 authors: 28,964 posts (**9.7%**)
- 137 accounts have ≥100 posts each
- **13,570 posts (4.0%) come from 2,017 bridged accounts** (`*.brid.gy` —
  Mastodon and RSS relays, not native Bluesky users)
- Profile labels flag **367 bot** and **150 spam** accounts, plus 2,217 carrying
  a bridged-from-ActivityPub/web label

The single largest non-spam poster is `longtail-news.bsky.social` at 2,677
posts. These accounts are news feeds: they inflate volume, carry no opinion, and
their followers/following patterns are not social ties. For network analysis
they are structurally different objects from ordinary accounts.

## 10. Engagement counts are one snapshot, not a fixed-age measure

`like_count`, `repost_count`, `reply_count` and `quote_count` were all captured
at collection time (2026-09-13/14). A post from January 2025 had 20 months to
accumulate them; a post from 2026-08-31 had 13 days.

Mean likes are surprisingly flat across the window (2025-01: 11.2 vs 2026-08:
10.7), which suggests most engagement lands quickly. But the share of
zero-like posts rises through the final weeks:

| Window before the cut-off | n | Mean likes | Zero-like share |
|---|---:|---:|---:|
| T−60…T−41 | 19,853 | 8.65 | **30.0%** |
| T−40…T−21 | 7,636 | 10.75 | 42.6% |
| T−20…T−1 | 7,866 | 10.85 | **44.2%** |

That gradient is suggestive rather than conclusive — the earliest window
overlaps a high-engagement period, so content mix is confounded with age. The
safe statement is the structural one: engagement is not comparable across the
window without normalising for age at collection, and the last ~3 weeks are
the part to be most careful with.

`reply_count` has a second problem: it is the platform's count at collection
time, and will not match the number of replies actually enriched (see issue 12).

> **WITHDRAWN 2026-09-14.** The concern above is not supported once measured on
> the agreed analysis corpus. Engagement is flat across age at collection from 7
> to 900 days — mean likes 10.0–12.1, median 1 in every bucket, zero-like share
> 34–47% with no trend. The gradient quoted above compared windows that differed
> in **content mix**, not in age: the earliest window overlapped a high-engagement
> period. Engagement on Bluesky lands within days and then stops accumulating.
>
> Only 2.2% of posts were under 30 days old at collection, all within
> 2026-08-14…31. No correction is applied; the bucket table is reported in
> limitations as measured evidence instead.
>
> The structural point still stands and should still be stated: these are
> snapshot counts, not fixed-age measures. It simply does not bite in this
> corpus, and that is now a finding rather than an assumption.

## 11. Follow graph: censored out-degree, sparse induced subgraph

| | |
|---|---:|
| Seeds | 5,000 (all complete, 0 duplicates) |
| Edges | 3,717,888 |
| Distinct targets | 902,244 |
| **Truncated seeds (hit the 2,000 cap)** | **804 (16.1%)** |
| Seeds returning zero edges | 239 — **230 genuine, 9 unavailable** (see correction below) |
| Untruncated seeds collecting fewer than their profile reports | **3,302 of 4,194 (79%)**, median gap −12 |
| Self-loops | 36 |
| Targets that are also seeds | 4,309 |
| **Edges falling inside the seed set** | **158,284 (4.3% of all edges)** |
| Targets who are authors anywhere in the corpus | 95,285 (10.6%) |
| Targets with in-degree 1 | 496,826 (55%) |

Four separate things here:

1. **Out-degree is censored at 2,000** for 16.1% of seeds — already known and
   documented. It must not be used as an influence measure. In-degree is
   unaffected.
2. **239 seeds returned zero edges — but only 9 are missing data.**

   > **Corrected 2026-09-14.** This section originally claimed all 239 were
   > deleted or deactivated accounts. That was wrong. Cross-checking each seed
   > against its own profile record (`follows_count`) separates them cleanly:
   > **230 seeds genuinely follow nobody** — their profile independently reports
   > `follows_count = 0` — and only **9** report a non-zero follow count while
   > returning no edges, which is the real missing-data case. The 230 are
   > overwhelmingly news bots and bridged RSS accounts, which broadcast without
   > following anyone; that is consistent with the seed set being the most
   > prolific posters rather than the most social ones.

   So out-degree is a real measurement for 230 of them and must be NA for 9.
   The classification is materialised in `data/interim/follows_seed_status.csv`.
3. **79% of untruncated seeds collected fewer follows than their own profile
   reports**, median 12 short. Small per seed, but it means `follows_count`
   from the profile layer and the edge count from the follow layer will never
   reconcile. Expected cause is follows pointing at deleted accounts, which the
   count includes and the listing omits.
4. **The induced subgraph is thin.** Only 4.3% of edges connect two seeds. Any
   network analysis restricted to the seed set is working with 158,284 edges,
   not 3.7 million, and over half of all targets are followed by exactly one
   seed. The graph is a 2026-09 snapshot regardless — it cannot support
   time-varying network analysis across the event window.

## 12. 33 threads truncated at the 200-reply cap — RESOLVED 2026-09-14

Of 13,093 enriched threads, **33 hit `max_replies_per_thread: 200`** and were
incomplete. 51 returned zero replies and 4 roots had been deleted by enrichment
time.

The bias was not random: the threads that exceed 200 replies are the largest and
most contentious, which is to say exactly the ones a conflict or polarisation
analysis would care about most.

**Repaired.** `src/platforms/bluesky/recollect_capped_threads.py` re-fetched all 33 with the cap
lifted, appending only replies not already on disk. **1,591 new replies**; the
largest thread went from 200 to 493. No thread hit the lifted cap, so every one
is now complete to depth 2. Rows from the repair carry
`_meta.recollect_of_capped = true`.

One honest caveat: some of those 1,591 were posted *between* the original
enrichment (2026-09-13/14) and the repair, so they are new replies rather than
previously-capped ones. The two are indistinguishable, and both are legitimate
replies to a collected post. The provenance flag lets any analysis that cares
separate them.

---

# Minor issues

13. **10,465 posts (3.1%) have empty text.** Not corrupt — they are link-share
    posts; a sample shows 97% carry an `app.bsky.embed.external` embed, the rest
    images. They contribute nothing to text analysis but are valid
    link-sharing behaviour, and the URL is retrievable from the embed.

14. **55,513 rows (16.4%) have no `langs` field** — RESOLVED 2026-09-14. Of
    those that do: 273,567 en, then fr (1,530), de (1,519), en-US (1,443), ja
    (1,135), es (1,102), with `en`, `en-US` and `en-GB` as separate values.

    The declared tag turned out to be worse than "incomplete". It reports the
    posting **client's** language, not the post's: **3,630 posts declaring `de`,
    `fr`, `ja` or `pt` are written in English**, across 2,378 distinct authors.
    A naive `langs == "en"` filter would have dropped those, plus the untagged
    sixth of the corpus, plus every `en-GB`/`en-US` row — and it would have kept
    897 posts that declare English but are confidently Finnish, Chinese or
    Japanese.

    Replaced by `is_english` (95.3% of posts), built from the declared tag and
    langdetect together, with detection allowed to overrule the tag only at
    ≥60 characters and p ≥ 0.999. `english_basis` records the reason per post.

15. **54 replies are dated after the study window** (2026-09-01 to 2026-09-12),
    collected because enrichment ran after the window closed. Thread depth for
    the final days of the window is still accruing and is right-censored.

16. **1,276 posts and 242 accounts carry `handle.invalid`** — the DID resolves
    but the handle does not. Across the whole corpus **22 DIDs appear under more
    than one handle**; 17 of those are a real handle alternating with
    `handle.invalid`, and 5 are genuine renames (e.g.
    `r-m.northsky.social` → `spetsdad.bsky.social`). No handle is shared by two
    different DIDs once `handle.invalid` is excluded.

    > **Corrected 2026-09-14.** The first pass reported "one handle shared by two
    > DIDs, one DID under two handles" because it scanned only `posts_*.jsonl`.
    > Including the thread and profile layers raises it to 22 DIDs. The
    > conclusion is unchanged and now better supported: **join on DID, never on
    > handle** — a handle-keyed join would split 22 accounts in two and merge
    > every `handle.invalid` account into one.

17. **4 baseline cells saturated** at the 10,000 `hits_total` cap — `dinner` on
    2026-04-26/27 and `coffee` on 2025-01-26/27. Out of 3,040, so masking those
    cells is sufficient; no term needs dropping. Listed in
    `data/interim/baseline_saturated_cells.csv`.

18. **21 profile tombstones** (`gone: true`) — accounts deleted between the post
    and profile passes. Author→profile coverage is otherwise complete.

19. **368 posts (0.1%) contain styled or fullwidth Unicode letters**
    (𝗕𝗼𝗹𝗱, 𝑖𝑡𝑎𝑙𝑖𝑐, ｆｕｌｌｗｉｄｔｈ). These break tokenisers and keyword filters
    unless NFKC-normalised. Small, but they cluster in exactly the
    engagement-bait and campaign posts a discourse analysis would want.

20. **Stale figures in the existing documentation**, beyond issue 4 — all
    corrected 2026-09-14:
    - Meme contamination recorded as 22,845 rows / 6.7%; actual **28,478 /
      8.4%** after the truncated cells were re-collected.
    - 2025-07-11 recorded as 4,664 posts; actual **9,082**.
    - Profile count recorded as 160,218; actual **160,268** rows.

---

# What passed

These were checked and are clean. Worth recording, because several are the
failure modes that would have been most expensive to find later.

| Check | Result |
|---|---|
| Malformed JSON, all five layers | **0** across 338,637 + 133,443 + 160,268 + 5,000 + 3,040 rows |
| Envelope violations in the post directory | **0** — no baseline or profile rows leaked in |
| Day coverage | **608 / 608**, no gaps, no empty days |
| Cells hitting the 200-page cap | **0** — no post-level truncation anywhere |
| Orphan replies (parent not in corpus) | **0** of 13,042 parents |
| Duplicate DIDs in the profile layer | **0** |
| Author → profile coverage | **100%** — all 160,268 post and reply authors resolved |
| Duplicate (seed, target) edges | **0** |
| Duplicate seeds | **0** of 5,000 |
| Duplicate (term, day) baseline cells | **0**; all 5 terms present on all 608 days |
| Baseline zero-hit cells | **0** |
| Posts dated in the future | **0** |

---

# Changelog — what has been applied

**2026-09-14, groups A and B.** Only corrections and one source repair. No row
was excluded, no exclusion criterion was chosen, and nothing that depends on a
team decision was touched.

| Item | Action | Artifact |
|---|---|---|
| B1 | Re-fetched the 33 capped threads with the cap lifted; +1,591 replies, none still capped | `src/platforms/bluesky/recollect_capped_threads.py`, `logs_recollect_capped.txt` |
| A1 | Reply count corrected, 92,556 → **133,732 unique**, with the cause recorded | `README.md`, `docs/bluesky/probe_findings.md` |
| A2 | Meme figure 22,845/6.7% → **28,478/8.4%**; 2025-07-11 4,664 → **9,082**; profiles 160,218 → **160,268**; all three meme waves now named | `README.md`, `docs/bluesky/probe_findings.md` |
| A3 | Every seed classified `complete` / `capped` / `genuine_zero` / `unavailable`, with `out_degree` blank where it is not a measurement | `data/interim/follows_seed_status.csv` |
| A4 | Handle↔DID collisions enumerated; join-on-DID convention recorded | this document, issue 16 |
| A5 | The 4 saturated baseline cells named | `data/interim/baseline_saturated_cells.csv` |

**Corpus after the repair:**

| Layer | Volume |
|---|---|
| Posts | 338,637 rows → 298,843 unique |
| Replies | **135,034 rows → 133,732 unique** (was 133,443 / 132,340) |
| Profiles | 160,268 |
| Follow edges | 3,717,888 |
| Baseline | 3,040 |

## 2026-09-14, groups C, D and E — normalisation

`src/platforms/bluesky/normalise.py` built four Parquet tables in `data/interim/` (`posts`,
`post_query`, `thread_edges`, `authors`) and `src/platforms/bluesky/build_manifest.py` wrote
`data/manifest.json`. Every issue above that could be turned into a column was.
**Nothing was excluded** — the pass annotates only, so each analysis still
chooses its own filters.

Two things changed during that build:

* **`meme_wave_day` was redefined.** Share alone flagged 21 days, conflating a
  meme that *inflates a day's volume* (2025-07-11: 78.6% of 5,044 posts, 14.3x
  the median day) with one merely *present* on an ordinary day (2026-08-22:
  32.7% of 272 posts, 0.8x median). It now requires share ≥20% **and** volume
  ≥2x median — 10 days. `day_meme_share` and `day_volume_ratio` ship alongside
  so the threshold can be changed without re-running.
* **The repair pass had opened a new gap.** Re-fetching the 33 capped threads
  introduced 380 authors with no profile record, because pass 4 had already
  finished. `collect_profiles.py` was re-run to close it. Pass 4 must now follow
  pass 3b.

**Two findings in this document were corrected by the work itself**, both
because the first pass drew a conclusion from one layer that a second layer
contradicted — see issue 11 (239 zero-edge seeds are mostly genuine, not
missing) and issue 16 (22 DIDs carry multiple handles, not 1).

---

# Open decisions for the team

None of these are mine to make alone; all of them change what the numbers mean.

1. **Which queries count as measures?** Given issue 1, `under-16 ban`,
   `age check`, `porn ID check`, `age verification privacy` and
   `age verification VPN` are recall nets rather than measures unless a
   phrase filter is applied at normalisation.
2. **Is stratum B still a valid contrast**, given that its terms are ~98%
   subsets of stratum A plus a keyword?
3. **How are the three meme waves handled** — excluded, modelled as a separate
   series, or reported as a confound? This decides whether E4 can carry an ITS.
4. **Bot and bridged-account policy**: exclude, flag, or weight. It affects
   volume, sentiment and network results differently.
5. **Which time axis** — filed day (`indexed_at`) or `created_at`. Recommend
   filed day, per issue 5.
6. **Dedup order of operations**, per issue 7.
