# Bluesky pre-flight probe — findings

**Run:** 2026-09-13, `src/platforms/bluesky/probe.py` (atproto 0.0.72)
**Scope:** Bluesky arm only. Nothing written to `data/`.
**Verdict: proceed with the full daily backfill, Jan 2025 – Jun 2026.**

---

## Q1. Does Bluesky's search index reach back to January 2025?

**Yes.** This was the biggest feasibility risk in the Bluesky workstream and it is cleared.

The test I originally designed didn't work as intended, so it's worth recording why. I ran a
high-volume control term (`the`) at monthly intervals back to Nov 2024, expecting a roughly flat
series in which any cliff would expose an index gap. Instead **every single month reported exactly
10,000 hits** — `hits_total` saturates at a server-side cap of 10,000. The test therefore
degenerated from a cliff detector into a floor check ("were there ≥10,000 indexed posts that
day?"). Useful, but much weaker than planned: it proves no month is *missing*, not that no month is
*thin*.

The real evidence comes from paginating actual topic queries to exhaustion on early-2025 days,
where counts come back small and precise (and therefore uncapped and trustworthy):

| Day | `age verification` | `Online Safety Act` |
|---|---|---|
| 2025-01-15 | 320 | 16 |
| 2025-02-15 | 16 | 2 |
| 2025-03-15 | 15 | 8 |
| 2025-04-15 | 31 | 9 |
| 2025-05-15 | 92 | 64 |
| 2025-06-15 | 4 | 6 |

The key point is the **shape**. A failing index decays smoothly toward zero the further back you
go. This doesn't — it jumps around (320 → 16 → 15 → 31 → 92 → 4) in the irregular way a
news-driven topic does. That is a genuinely quiet pre-enforcement period being measured correctly,
not an index running out. The window can start at 2025-01-01 as the proposal assumes.

One anomaly to keep an eye on: **2025-01-15 returned 320 posts**, roughly 20× the surrounding
days. Could be a real news event, could be a spam or bot cluster. Worth checking once the corpus
lands, before anyone reads it as a pre-period trend.

## Q2. Is there enough volume?

**Comfortably.** Posts per day, fully paginated:

| Query | Stratum | Quiet day | Biggest event day | Mean over 10 sample days |
|---|---|---|---|---|
| `age verification` | A | 414 | 1,429 (E2) | 523.6 |
| `Online Safety Act` | A | 37 | 597 (E2) | 140.4 |
| `age check` | A | 65 | 192 (E2) | 90.3 |
| `under-16 ban` | A | 13 | 582 (E3) | 88.0 |
| `Ofcom` | A | 23 | 224 (E3) | 73.6 |
| `age assurance` | A | 14 | 91 (E2) | 32.8 |
| `eSafety` | A | 9 | 42 (E3) | 9.6 |
| `age verification privacy` | B | 26 | 70 (E2) | 29.5 |
| `age verification VPN` | B | 3 | 82 (E2) | 22.2 |
| `porn ID check` | B | 0 | 8 (E3) | 1.6 |

Two things matter here beyond the totals:

1. **The baseline is not zero.** A quiet September 2025 day still yields 414 posts on the backbone
   query. A continuous daily series is viable, which is what the timing analysis needs — event
   windows alone would have forced a different design.
2. **The events show up in the right places.** E2 (UK enforcement) dominates `Online Safety Act`
   and `Ofcom`; E3 (AU under-16 ban) dominates `under-16 ban` and `eSafety`. The queries are
   tracking the events they should, which is a sanity check on the whole collection design.

Note E4 (Lords VPN vote, 2026-01-21) is weak across the board — 110 posts on the backbone query,
below the quiet-day baseline. If that holds across the surrounding days, E4 may be too small to
support event-level analysis on Bluesky.

## Q2c. Are the queries actually precise?

This wasn't in the original plan. I added it after spotting a false positive in the very first
sampled post — a photograph of a standing stone, matching "age verification" in the archaeological
sense. Recall is only half the question; a backbone query that drags in unrelated posts silently
changes the denominator of every share-based statistic computed from that stratum.

`policy%` = share of returned posts containing any policy-adjacent term:

| Query | Day | n | policy% |
|---|---|---|---|
| `age verification` | 2025-01-15 | 320 | 86% |
| `age verification` | 2025-07-25 | 1,429 | **91%** |
| `age verification` | 2025-09-17 | 414 | **92%** |
| `age verification` | 2026-03-11 | 298 | 89% |
| `age check` | 2025-07-25 | 192 | **52%** |
| `age check` | 2025-09-17 | 65 | **25%** |

- **`age verification` is clean** — 86–92%, and stable between quiet days and event days, which
  means precision doesn't degrade when volume spikes. The archaeology worry was unfounded (0%).
- **`age check` is not** — 52% on an event day and 25% on a quiet day. It catches "age vibe check",
  generic tech-news tagging, and similar. **Recommend demoting it out of the neutral backbone.**

A caveat on reading that number: a large share of the "off-topic" remainder are **link-only posts**
whose text is just "Interesting article!" with the substance sitting in the link card. Those are
not actually off-topic, and A1's flattener already captures `card_title` / `card_description`
separately — so they're recoverable. True precision on `age verification` is therefore somewhat
better than 86–92%.

## Q3. Do the Assignment 1 flatteners still work?

**Yes, unchanged.** Every field path `flatten_bluesky()` reads is present and correctly named under
atproto 0.0.72 (A1 was written against 0.0.69): `post.uri`, `post.author.did`, `post.indexed_at`,
`post.like_count`, `post.reply_count`, `post.repost_count`, `post.quote_count`, `post.labels`,
`post.embed`, `record.text`, `record.created_at`, `record.langs`, `record.facets`.

`record.reply` and `record.tags` came back null, but only because the sampled post was neither a
reply nor explicitly tagged — the flattener already guards both with `or {}` / `or []`.

The SDK has **gained** two fields worth capturing that A1's schema doesn't have:

- **`bookmark_count`** — free additional engagement signal.
- **`threadgate`** — indicates the author restricted who may reply. This one matters for the
  network component: a threadgated post has missing reply edges *by design*, not by sampling
  failure. Without it we'd be unable to tell the two apart when building the reply graph.

## Q4. What jurisdiction signal actually exists?

Measured over **10,116 posts from 6,699 distinct authors**.

| Signal | Result | Usable? |
|---|---|---|
| Country-TLD handle (`.uk`, `.au`, …) | 101 of 6,699 authors = **1.5%** (.uk 83, .au 9, .scot 5, .ie 4) | Too sparse alone |
| Custom domain handle (any) | 1,315 = 19.6% | Weak proxy at best |
| Declared language regional variant | `en-GB` 10, `en-AU` 1, against `en` 8,438 | **Useless** |
| Explicit profile location/country field | **Does not exist** | — |
| Profile bio text | **24 of 25 sampled authors have one** | **Yes — main route** |

This confirms the §3.1 concern empirically: there is no direct country signal on Bluesky, and the
two free ones (handle TLD, language tag) are far too sparse to carry a jurisdiction split on their
own. The profile record's available fields are `description`, `display_name`, `website`, `pronouns`,
`followers_count`, `follows_count`, `created_at`, `posts_count`, `labels` — no location.

**Consequence for collection:** bio text plus the follow graph are the only viable routes to
country tagging, and neither is in the search payload. Both must be collected deliberately or the
option closes permanently. Profile collection is therefore confirmed as its own pipeline step, as
planned.

## Cost of the full backfill

10 queries × 546 days = **5,460 cells**. The naive estimate of ~0.5 h assumed one request per cell,
but the big queries need 4–15 pages on event days. Realistic estimate: **~11,000 requests, ~1–1.5
hours** of wall clock, resumable and checkpointed so it runs unattended. Pacing stays at A1's
measured 0.05 s sleep — about 2.9 req/s against a 10 req/s limit.

Expected corpus size is hard to pin down because the sample days are deliberately event-weighted
(7 of 10 are event days), so the probe's own extrapolation (~286k–552k pre-dedup) is an optimistic
ceiling. A more honest floor: the quiet-day backbone rate of 414/day × 546 days ≈ **226,000 posts
before deduplication** on `age verification` alone. This corpus will be large — an order of
magnitude bigger than A1's 15,710. Worth planning storage and processing around.

---

## Decisions this produces

1. **Window:** 2025-01-01 → 2026-06-30, daily cells. Confirmed viable.
2. **Demote `age check`** from stratum A to stratum B on precision grounds (25–52%).
3. **Add `bookmark_count` and `threadgate`** to the schema; propose `threadgate` to the team as a
   shared column, since it affects any reply-graph construction on any platform with the concept.
4. **Profile collection confirmed** as a required pipeline step — it is the only route that keeps
   jurisdiction tagging possible later.
5. **Flag E4** (Lords VPN vote) as possibly too small on Bluesky for event-level analysis.
6. **Check the 2025-01-15 spike** for bot/spam contamination once the corpus lands.
7. **Fixed in the probe itself:** the shortfall warning now requires the gap to be large in both
   relative *and* absolute terms. The first run emitted nine "pagination fell short" warnings
   (e.g. "got 3 of 4 hits") that were pure small-cell noise — a proportional-only test misfires
   constantly on a topic with genuinely quiet days, which A1's consumer-launch topic never had.

---

# Addendum — findings from the live backfill (2026-09-13)

Two things surfaced once the real collection started that the probe's sampling
design could not have caught. Both were found in the first 200 cells.

## A. The page cap was too low, and event-based sizing is why

`age verification` on **2025-07-11** truncated at the 40-page cap. Paginating
that cell to exhaustion takes **49 pages / 4,664 unique posts** (96.8% of the
reported `hits_total` — the remainder is normal attrition, not a cursor
failure). So the cursor goes deeper than 40 pages; the cap was simply wrong.

The instructive part is *why* it was wrong. I sized it from the busiest cell the
probe measured — 1,621 posts on the AU ban day — and thought 40 pages gave 2.5×
headroom. But the probe sampled **policy events**, because those are the days
the research is about. The largest cell in the corpus is not a policy event at
all. Sizing a cap from the days you expect to be busy will miss the days that
are busy for reasons outside your event calendar.

**Fixed:** `max_pages_per_cell` raised 40 → 200 (4× headroom over the largest
cell actually observed). The truncated cell was removed from the checkpoint so
the re-run collects it properly — a cell recorded as done at the old cap would
never be revisited, and its missing posts would stay invisible for the rest of
the project. The cap costs nothing on ordinary cells, since pagination stops
when the cursor runs out rather than when the cap is reached.

## B. A viral meme is contaminating the volume series — badly

The 4,664 posts on 2025-07-11 are not policy discourse. They are a joke format:
`"Age verification:"` followed by something that dates the poster.

> *"Age verification: when I lived in Germany, it had been one country for less than a decade"*
> *"Age verification? I remember the Carter presidency."*

Measuring how much of each day is that format (text beginning `age
verification:` or `age verification?`):

| Day | posts | meme format | what it is |
|---|---|---|---|
| 2025-07-10 | 673 | 10% | ramp-up |
| **2025-07-11** | **4,664** | **81%** | meme peak |
| 2025-07-12 | 819 | 79% | meme tail |
| **2025-07-25** | 1,429 | **1%** | **E2, real policy discourse** |
| 2025-09-17 | 414 | **48%** | "quiet baseline" day |

Three consequences, in descending order of how much they matter:

1. **It would fake a finding.** A 4,664-post spike two weeks *before* UK
   enforcement is exactly the shape an interrupted time series would read as
   anticipatory discourse building ahead of the policy. It is a joke about
   remembering the Carter presidency. Any timing analysis run on unfiltered
   volume would report this as a result.
2. **The probe's volume figures are overstated.** The "quiet day baseline" of
   414 posts is roughly half meme. Real policy discourse on an ordinary day is
   nearer 200. The corpus will still be large, but the earlier extrapolations
   should be read down accordingly.
3. **My precision check was fooled, and the reason is worth noting.** The
   meme day scores **99% "policy-related"** because the policy regex contains
   `verif` — and every meme post contains the phrase "age verification" by
   construction. A keyword test whose vocabulary overlaps the query it is
   testing will validate anything that query returns. The 86–92% precision
   figures reported above are therefore ceilings, not estimates.

**This does not change collection.** The posts genuinely match the query, raw
stays a log of what the API returned, and filtering at collection time would
destroy the ability to measure the contamination later. It is a **normalisation
requirement**, and it needs to be handled before any volume series is built:

- Flag the meme format explicitly (leading `age verification:` / `age
  verification?`) as its own column rather than deleting the rows.
- Rebuild any precision estimate with a keyword list that excludes the query
  terms themselves.
- Re-check the 2025-01-15 spike (320 posts against neighbours of 15–31,
  flagged earlier as possible bot activity) — the same meme is the likelier
  explanation.
- Tell the team. If a meme can put a 10× spike in a Bluesky volume series, the
  Reddit arm should check whether the same format travelled there, and the
  cross-source comparison needs to treat it consistently.

## C. The denominator series is in, and the drift is bigger than expected

Pass 2 completed: 3,040 (term, day) cells, **zero missing values**, and only 4
cells (0.13%) hit the 10,000 `hits_total` ceiling — those must be masked when
building the denominator rather than used as values, and rather than causing
their whole term to be dropped.

| Term | min | max | Jan 2025 avg | Aug 2026 avg |
|---|---|---|---|---|
| cooking | 1,392 | 7,621 | 2,733 | 1,777 |
| dinner | 2,007 | 10,000 | 3,573 | 2,325 |
| breakfast | 1,594 | 5,024 | 2,804 | 1,873 |
| tired | 3,165 | 9,966 | 6,387 | 4,052 |
| coffee | 3,554 | 10,000 | 7,886 | 4,524 |

**Basket total: 724,881 → 451,091, a 38% decline.** Every term falls, and they
fall together, which is what a platform-level trend looks like rather than five
independent vocabulary shifts.

This is larger than the spot checks suggested and it cuts in the direction that
matters: general Bluesky chatter contracted by more than a third over exactly
the window in which age-verification discourse grew. **Raw post counts therefore
understate the real rise.** Any volume series built without dividing by this
denominator is measuring the topic and the platform at once, and would
understate the effect it is trying to find.

## D. The follow-graph pass, and what its cap costs

Built and smoke-tested. Seeds are the most active authors (>=3 posts), capped at
5,000, with each seed's follow list capped at 2,000 — all three numbers measured
rather than guessed, and justified in `config/bluesky/config.yaml` and the script docstring.

Two things surfaced in the smoke test worth recording:

- **`getFollows` cannot tell you how much it truncated.** Its `subject` is a
  `ProfileView`, which has no `follows_count`; only `getProfiles`'
  `ProfileViewDetailed` carries it. Read naively, every seed records `None` and
  truncation becomes detectable but not measurable. Counts are now prefetched
  in batches of 25 (~200 requests against ~28,000).
- **Even untruncated seeds lose a few edges.** Reported vs kept: 909→881,
  492→484, 259→255, 1,291→1,272. That is the same survivorship story as the
  post layer — followed accounts deleted or suspended since the follow was
  made — now measurable at the edge level, at roughly 2–3%.

The most active authors are also visibly *not* all human: the top seed by post
count is `longtail-news` (866 posts, follows 4 accounts), and several bridged
fediverse accounts follow nobody at all. Activity rank is a good cheap proxy for
participation but it is not a filter for automation, and community detection
will need to handle that.

## E. Pass 1 complete — corpus verified

| | |
|---|---|
| Raw rows | 338,637 |
| Unique posts | 298,843 (dedup removes 12%) |
| Days covered | **608 of 608**, 2025-01-01 → 2026-08-31, no gaps |
| Malformed lines | **0** |
| Stratum A / B | 291,188 / 47,449 (86% / 14%) |
| Meme-format rows | **28,478 (8.4% of all rows)** — corrected 2026-09-14, see `data_quality_audit.md` |

Yield by query, as a share of raw rows:

| Query | Rows | Share |
|---|---|---|
| `age verification` | 126,245 | 37.3% |
| `social media ban` | 70,624 | 20.9% |
| `age check` | 35,369 | 10.4% |
| `Ofcom` | 35,114 | 10.4% |
| `Online Safety Act` | 23,685 | 7.0% |
| `verify your age` | 13,435 | 4.0% |
| `under-16 ban` | 11,084 | 3.3% |
| `age verification privacy` | 8,222 | 2.4% |
| `age assurance` | 7,802 | 2.3% |
| `age verification VPN` | 3,592 | 1.1% |
| `eSafety` | 3,199 | 0.9% |
| `porn ID check` | 266 | 0.1% |

Two things worth noting. **Only 12% of rows are duplicates**, which means the
queries overlap far less than expected — they are genuinely complementary
rather than twelve views of the same posts. And `social media ban`, the query
added late on measured evidence, is the **second-largest contributor at 21%**;
without it the AU under-16 event would rest on `under-16 ban` alone at 3.3%.

`porn ID check` returned nothing on 467 of its 608 days and contributed 266
rows in total (0.1%). It earns its place only as insurance against missing the
"upload your ID to watch porn" conversation entirely; it is not a series.

## F. Correction — the shortfall is cell-specific, not query-specific

Addendum B of this document and an earlier version of the collector's comments
claimed `hits_total` was unreliable *for the query* `social media ban`. **That
was wrong.** The corrected finding, from three consecutive runs of each cell:

| Cell | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| `social media ban` / 2025-10-24 | 53 / 264 (20%) | 53 / 264 | 53 / 264 |
| `social media ban` / 2025-12-10 | 1,621 / 1,711 (95%) | 1,621 / 1,711 | 1,621 / 1,711 |

Same query, opposite behaviour, and **perfectly deterministic in both
directions**. So the effect belongs to the cell, not the query and not the
moment. On an affected cell the backend serves one short page, withdraws the
cursor, and declares the set exhausted while still reporting a much larger
count. Re-running the largest event-day cells across five queries returns
95–97% with pagination exhausting naturally, which is ordinary attrition.

Scale: 150 of 7,273 cells (2.1%) tripped the test; roughly **3% of indexed
posts could not be retrieved**. That 3% is the figure to quote as survivorship
bias. Because the behaviour reproduces exactly, re-collection does not recover
those posts.

The lesson for anyone reading a shortfall warning here: it is a diagnostic, not
a measurement. Re-run the individual cell before concluding anything.

## G. Passes 3 and 4 complete

**Enrichment:** 13,093 threads → **133,732 unique replies** (135,034 rows).

> **Corrected 2026-09-14.** This section previously read "92,556 replies". That
> was the tally printed by the *last* of several enrichment runs, after the pass
> had been restarted following the timeout, login and NotFound crashes. Each
> restart resumed correctly from the checkpoint and appended its own share, so
> the data was always complete — only the headline figure was short, by 31%.
> The count above is recomputed from disk and agrees exactly with the sum of the
> enrichment checkpoint.

Only 4 root posts had been deleted by enrichment time. 33 threads hit the
200-reply cap; all 33 were re-fetched on 2026-09-14 with the cap lifted
(`src/platforms/bluesky/recollect_capped_threads.py`), adding 1,591 replies, and **no thread is
capped any longer**. These
replies are the graph: they are, by construction, the posts search cannot see,
and without them a reply network built from search results alone would have
almost no edges.

**Profiles:** **160,268 author profiles**, 21 tombstoned (0.0%), **90.2% carry
a bio**.

### What the bios are actually worth for jurisdiction

This is the measurement the deferred §3.1 decision was waiting on. A crude
regex for UK/AU/IE/NZ places and nationalities over all 160,268 bios matches
**9,404 authors — 5.9%**.

| Signal | Coverage |
|---|---|
| Regional language tag (`en-GB`/`en-AU`) | ~0.1% |
| Country-TLD handle | 1.5% |
| **Bio mentions a UK/AU/IE/NZ place** | **5.9%** |

So bio text is roughly **four times better than the handle TLD** — and still
leaves 94% of authors untagged. A naive keyword rule will not carry a
jurisdiction split, and a smarter classifier would have to be validated against
a hand-labelled sample with its error rate reported before anyone leaned on it.

That is not a reason to regret collecting the profiles: the data now exists, the
option is open, and the number above is the evidence the team needs to decide.
But it does strengthen the original recommendation. **Bluesky is the
cross-jurisdiction discourse-and-network layer; Reddit and Google Trends should
carry the UK-vs-Australia comparison.** The follow graph is the remaining
candidate for doing better, since follow homophily does not depend on people
volunteering where they live.

## H. The follow-graph seed set is cap-bound, not threshold-bound

On the finished corpus, **37,308 authors clear the 3-post threshold** — four
times the 9,176 measured mid-collection. With `max_seeds: 5000`, the cap now
binds hard: the least active seed has **11 posts, not 3**, and 32,308 eligible
authors are not seeded.

Two consequences:

1. **Describe it accurately.** The follow graph covers the 5,000 most active
   authors, which in practice means authors with 11 or more posts in the
   corpus. It is not a sample of participants, and calling it one would be
   wrong.
2. **Extending is cheap and incremental.** Raising `max_seeds` and re-running
   fetches only the new seeds, because completed ones are checkpointed. If the
   community-detection results look thin at 5,000, going to 10,000 costs about
   another five hours and nothing already collected.

## I. Pass 5 complete — the follow graph, and the cap's real cost

Ran 441 minutes (7.4 h), **zero errors and zero retries** across the entire pass.

| | |
|---|---|
| Seeds | 5,000 |
| Follow edges | **3,717,888** |
| Distinct followed accounts | 902,244 |
| Truncated seeds | 804 (16.1%) |
| Zero-edge seeds | 239 (4.8%) |
| **Edges lost to the cap** | **7,862,262 of 11,580,149 reported — 67.9%** |

### The figure that matters, and it is not the one I expected

"16% of seeds were truncated" sounds mild. It is not. Those 16% of seeds
account for **68% of all the follow edges that exist** — 7.86M of 11.58M. The
distribution is so heavy-tailed that a cap touching one seed in six removes
two thirds of the raw edge mass.

Both numbers must appear together in the writeup, because either alone
misleads in the opposite direction:

- *"16% of seeds truncated"* alone understates it badly.
- *"68% of edges dropped"* alone implies the graph is a fragment, when in fact
  84% of seeds have their complete follow list and the median seed is
  unaffected.

The accurate statement: **the collected graph is every follow for 84% of
seeds, and the 2,000 most recent follows for the other 16%.**

This does not change the judgement that the cap was correct — an account
following hundreds of thousands of others is an aggregator whose edges connect
everything to everything, and including them would smear community structure
rather than reveal it. But it does mean the cap is a *substantive analytical
choice*, not a technicality, and anyone running centrality on this graph needs
to know that the highest-degree nodes are capped by construction. Out-degree
is therefore **censored at 2,000 and must not be interpreted as an influence
measure**; in-degree (how many seeds follow an account) is unaffected and is
the sounder basis for centrality work.

My pre-collection estimate was 12% truncation, taken from a 200-seed sample.
The true figure is 16.1%. The sample was too small and skewed optimistic;
quote 16.1%.

## J. Collection phase complete — final corpus

| Layer | Volume |
|---|---|
| Posts (search) | 338,637 rows → 298,843 unique, 608/608 days |
| Replies (thread enrichment) | 135,034 rows → **133,732 unique** from 13,093 threads |
| Author profiles | 160,268 (90.2% with a bio) |
| Follow edges | 3,717,888 across 5,000 seeds |
| Baseline denominator | 3,040 (term, day) counts |

Attrition, all measured rather than assumed:

- ~3% of indexed posts were not servable (concentrated in 2.1% of cells)
- 4 thread roots deleted between collection and enrichment
- 0 threads truncated at the reply cap (33 were, and were repaired 2026-09-14)
- 21 author accounts (0.0%) deleted between posting and profile collection
- ~2-3% of follow edges point at accounts already gone
