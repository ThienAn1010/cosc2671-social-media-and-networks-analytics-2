# Collection and cleaning log — Bluesky workstream

**Owner:** Bluesky collection (one of three platform workstreams).
**Status:** collection complete; normalisation complete; all exclusion criteria decided; analysis tables built; frame lexicons validated by two coders.
**Last updated:** 2026-09-14.

This is the running record of what was done to the Bluesky data, when, and why.
It is the file to update whenever cleaning work happens — append to the
changelog at the bottom and revise the status tables above it.

**How it relates to the other documents.** They do not overlap; each answers a
different question.

| Document | Answers |
|---|---|
| `README.md` | How do I run this? |
| `docs/bluesky/probe_findings.md` | What did we measure about the platform before and during collection? |
| `docs/bluesky/data_quality_audit.md` | What is wrong with the data and how bad is it? |
| **this file** | What has actually been done to the data, in what order? |

---

# Current state

| Layer | Volume | Path |
|---|---|---|
| Posts (search) | 338,637 rows → **298,843 unique**, 608/608 days | `data/raw/bluesky/posts_*.jsonl` |
| Replies (thread enrichment) | 135,034 rows → **133,732 unique**, from 13,093 threads | `data/raw/bluesky/thread_*.jsonl` |
| Author profiles | **160,648** (21 tombstoned, 90.2% with a bio) | `data/raw/profiles/profiles.jsonl` |
| Follow edges | **3,717,888** across 5,000 seeds, 902,244 distinct targets | `data/raw/follows/follows.jsonl` |
| Baseline denominator | **3,040** (term, day) counts | `data/raw/baseline/daily_counts.jsonl` |
| Correction tables | 2 CSVs, regenerable | `data/interim/` |
| **Normalised tables** | `posts` 423,628 (46 cols) · `post_query` 338,637 · `thread_edges` 135,034 · `authors` 160,648 | `data/interim/*.parquet` |
| **Analysis corpus** | `is_english` → **403,898 posts (95.3%)**; searched corpus 283,327 over **608/608 days** | filter on `posts.parquet` |
| Manifest | row counts, date ranges, checksums, code hashes | `data/manifest.json` |
| **Processed dataset** | pseudonymised analysis tables, 99 MB, verified shareable (supersedes the old `data/shared/`) | `data/processed/bluesky/` → team drive |

Raw ~2.6 GB, interim ~123 MB. All timestamps inside the data are **UTC**; the
run logs are local time (UTC+7), so the two differ by 7 hours — they are not
inconsistent.

**Group by `day`**, filter on `is_english`. Those two columns carry the only
decisions made so far that change what a series contains.

---

# Part 1 — Collection (complete)

## What ran, in order

| Pass | Script | Output | Ran (UTC) |
|---|---|---|---|
| 1 | `collect_bluesky.py` | 7,296 cells (12 queries × 608 days) | 2026-09-13 10:38–12:15 |
| 2 | `collect_baseline.py` | 3,040 cells (5 terms × 608 days) | 2026-09-13 10:39–11:04 |
| 3 | `enrich_threads.py` | 13,093 threads | 2026-09-13 10:47 – 09-14 00:12 |
| 3b | `recollect_capped_threads.py` | 33 threads repaired | 2026-09-14 06:25 |
| 4 | `collect_profiles.py` | 160,268 profiles | 2026-09-13 10:47–19:23 |
| 4b | `collect_profiles.py` (re-run) | +380 profiles for authors the repair pass introduced | 2026-09-14 06:50 |
| 5 | `collect_follows.py` | 5,000 seeds | 2026-09-13 11:04 – 09-14 02:47 |

Passes overlap because several were run concurrently once pass 1 had produced
enough of the raw layer to feed them.

## Parameters that shape the data

These are the choices a reader needs in order to interpret any number derived
from this corpus. All live in `config/bluesky/config.yaml` with justification comments.

- **Window:** 2025-01-01 → 2026-08-31 (608 days). Deliberately extended past the
  proposal's June 2026 end to cover the E5 collection attempt; the authoritative
  frozen event date is 2026-03-09.
- **Queries:** 12, in two strata — 8 neutral backbone (A), 4 known-biased
  supplement (B). The two-stratum design was meant to test whether frame-share
  is an artefact of the query list. *(Audit issue 1 found this design does not
  survive contact with how Bluesky search actually matches — see Part 2.)*
- **Sort order:** `latest`, never `top`. Engagement ranking would bias a
  sentiment sample toward whatever the platform amplified.
- **Page cap:** 200 pages/cell. Raised from 40 mid-run after `age verification`
  / 2025-07-11 was found truncated at the old cap; affected cells were un-marked
  from the checkpoint and re-collected. **No cell reached the 200-page cap**, so
  there is no post-level truncation anywhere in the corpus.
- **Enrichment:** threads with ≥3 replies, depth 2, originally capped at 200
  replies/thread (cap since removed for the 33 affected threads).
- **Follow seeds:** authors with ≥3 posts, capped at 5,000 seeds and 2,000
  follows each. 37,308 authors clear the 3-post threshold, so the seed set is
  cap-bound: the least active seed has 11 posts.
- **Baseline basket:** 5 deliberately mundane terms (`cooking`, `dinner`,
  `breakfast`, `tired`, `coffee`) chosen for stability, to measure platform
  drift independent of the topic.

## Measured during collection

Recorded in full in `docs/bluesky/probe_findings.md`. The load-bearing results:

- **Index depth is fine** — search reaches back to 2025-01-01; the thin early
  volume is a real quiet period, not decay in the index.
- **Platform drift is large** — the baseline basket fell **38%** over the
  window (724,881 → ~451,000 monthly). Raw topic counts therefore *understate*
  the topic's rise and must be normalised by the denominator.
- **Retrieval attrition ~3.1%** of indexed posts, concentrated in 2.1% of cells
  where the backend serves one short page and withdraws the cursor. Verified
  deterministic — re-running an affected cell returns the identical shortfall,
  so it is not recoverable by re-collection.
- **No geolocation exists on Bluesky.** Bios give 5.9% jurisdiction coverage,
  country TLDs 1.5%, regional language tags ~0.1%. 94% untagged. Bluesky is
  therefore the cross-jurisdiction discourse and network layer; UK-vs-AU
  comparison has to come from Reddit and Google Trends.

## Incidents during collection

All resolved; each left a fix in the code. Recorded because they explain why
some figures in the logs disagree with the data on disk.

| What happened | Cause | Fix |
|---|---|---|
| Enrichment died after 2,435 threads | `InvokeTimeoutError` has no HTTP response, so `status` was `None` and failed the retryable-status test — the most retryable failure class was the one never retried | `isinstance(exc, NetworkError)` in `src/platforms/bluesky/api.py` |
| Restart died in <2 min | `client.login()` was the only call in the project not wrapped in a retry | `login_with_retry()`, applied to all six scripts |
| Enrichment died after 3,938 threads | A root post deleted between collection and enrichment returns `400 NotFound` — an error, not the tombstone the walker expected | `is_gone()` predicate, applied to enrichment, profiles and follows |
| Enrichment crashed on `KeyError: '_meta'` | Baseline and profile JSONL were written into the post directory, which `iter_raw_records()` globs wholesale | Separate directories per record shape |
| Collector killed at 6,547/7,296 cells | System-wide memory pressure, not the collector | Checkpoint resumed cleanly |
| Follow counts all `None` | `getFollows`' subject is a `ProfileView`, which has no `follows_count` | `fetch_follow_counts()` prefetch, ~200 requests instead of ~28,000 |

**Consequence for the numbers:** because enrichment was restarted several times
and the raw layer is append-only, the final log line reports only the *last*
run's tally. That is the origin of the 92,556-vs-133,732 reply discrepancy
corrected on 2026-09-14.

## Deliberately not done

- **No filtering at collection time.** Memes, spam and off-topic matches are all
  collected. Filtering at collection would have destroyed the ability to measure
  the contamination, which is now a finding in its own right.
- **No `top` sorting, no engagement thresholds** — see above.
- **No geolocation inference.** Measured as unsupportable, not assumed.

---

# Part 2 — Cleaning

## Approach

**Flag, don't delete. `data/raw/` is immutable and append-only.**

Different analyses need different exclusions — the meme is noise for an
interrupted time series but is itself the object for a virality analysis. All
cleaning therefore produces *annotations* in `data/interim/`, never edits or
removals in `data/raw/`.

A second rule follows from the team workflow: **normalisation is local,
exclusion is shared.** Parsing, dedup, schema and encoding are decided by
whoever owns the platform. What counts as relevant, bot, or spam must be agreed
across all three platforms, or cross-platform comparisons end up confounded by
cleaning method rather than platform.

| | Who decides | Status |
|---|---|---|
| Normalisation — parsing, dedup, schema, encoding, timestamps | This workstream | In progress |
| Exclusion — relevance, bots, memes, spam, language | Whole team | Not yet raised |

## Done — 2026-09-14, groups A and B

Corrections and one source repair only. No row was excluded and no exclusion
criterion was chosen.

| Item | What | Artifact |
|---|---|---|
| **B1** | Re-fetched the 33 threads truncated at the 200-reply cap; +1,591 replies, none still capped, all now complete to depth 2 | `src/platforms/bluesky/recollect_capped_threads.py` |
| **A1** | Reply count corrected 92,556 → **133,732 unique**, with the cause recorded so it is not "fixed" back from the old log | `README.md`, `probe_findings.md` |
| **A2** | Meme figure 22,845/6.7% → **28,478/8.4%**; 2025-07-11 4,664 → **9,082**; profiles 160,218 → **160,268**; all three meme waves named | `README.md`, `probe_findings.md` |
| **A3** | Every follow seed classified `complete` / `capped` / `genuine_zero` / `unavailable`; `out_degree` left blank where it is not a measurement (813 of 5,000) | `data/interim/follows_seed_status.csv` |
| **A4** | Handle↔DID collisions enumerated; **join on DID, never handle** recorded as a convention | `data_quality_audit.md` issue 16 |
| **A5** | The 4 saturated baseline cells named, to mask before using the denominator | `data/interim/baseline_saturated_cells.csv` |

Both CSVs are regenerable with `src/platforms/bluesky/build_correction_tables.py`.

## Done — 2026-09-14, groups C, D and E

The normalisation pass. It **annotates, never excludes** — every filter is a
column, so no analysis's judgement is imposed on any other, and no decision the
team has not made has been made here.

| Item | What | Where |
|---|---|---|
| **C1** | Four-table split: `posts` (423,628 unique) / `post_query` (338,637 pairs) / `thread_edges` (135,034) / `authors` (160,648) | `src/platforms/bluesky/normalise.py` |
| **C2** | `day` as the canonical axis (collection cell for search posts, index day for replies); `cell_day`, `index_day`, `created_at` and both drift flags kept | `posts.parquet` |
| **C3** | NFKC normalisation, `text_norm`, `text_hash`, `has_text`, `n_chars` | `posts.parquet` |
| **C4** | `is_reply`, `discovered_via`, `in_search`, `in_thread` — the 8,947 posts that are both a search hit and a thread reply are now countable once | `posts.parquet` |
| **D1** | `phrase_exact` and `all_terms` per (post, query) | `post_query.parquet` |
| **D2** | `is_meme`, `meme_wave_day`, plus `day_meme_share` and `day_volume_ratio` so the threshold can be revisited without re-running | `posts.parquet` |
| **D3** | `dup_text_n`, `is_dup_text` | `posts.parquet` |
| **D4** | `is_repeat_burst` — 47 (author, text) pairs repeated ≥20 times | `posts.parquet` |
| **D5** | `account_class` with its underlying signals kept alongside | `authors.parquet` |
| **D6** | `lang_norm` + `lang_source`; langdetect resolved 38,956 of 39,017 untagged posts | `posts.parquet` |
| **E1** | Manifest: per-layer rows, bytes, date range, rolled-up sha256, plus hashes of `config/bluesky/config.yaml` and all 15 scripts | `data/manifest.json` |

**Integrity, verified after the build:** 0 duplicate URIs in `posts`; 0 dangling
joins from `post_query`, `thread_edges` or `authors`; 0 authors without a
profile.

Flag prevalence on the searched corpus (298,843 posts): `is_meme` 8.2%,
`meme_wave_day` 9.7%, `is_dup_text` 10.5%, `is_repeat_burst` 1.4%, `is_reply`
27.4%, `has_text` 97.2%, `created_at_backdated` 0.3%.

`all_terms` is ≥99.7% for **every** query while `phrase_exact` ranges from 1.5%
to 100%, which settles audit issue 1 beyond doubt: Bluesky search is
AND-of-terms, not phrase matching.

## Decided — English only (2026-09-14)

The team settled decision 4: **this project works in English only.**

Implemented as `is_english` on `posts.parquet`, with `english_basis` recording
why each post was kept or dropped. **403,898 of 423,628 posts are English
(95.3%)** — 283,327 of the searched corpus (94.8%) and 129,454 replies (96.8%).

The rule is a hybrid, because neither available signal is trustworthy alone.
Bluesky's `langs` tag is set by the posting client and generally reports the
user's *interface* language rather than the language they typed in; langdetect
is unreliable on short text. Both were measured before the rule was written:

| Evidence | Measurement |
|---|---|
| Declared `en` that detects as English | 94.9% of a 15,000-post sample |
| Detection disagreement by length | 49.1% under 20 chars → 0.3% above 120 |
| Detection confidence on disagreements | bimodal: 618 below p=0.90, 897 above p=0.999, only 4 between |

So detection may overrule a declared tag only where it is both long enough
(≥60 chars) and confident (p ≥ 0.999), and the rule is **asymmetric by design**:
a post leaves the English corpus only on confident evidence. The corpus is ~95%
English, so a wrongly-dropped post is a silent loss correlated with short,
informal, emoji-heavy text — the register a discourse study most wants — while a
few hundred stray non-English posts are negligible noise in 400,000.

| `english_basis` | Posts | Meaning |
|---|---:|---|
| `declared_confirmed` | 283,813 | tag and detection agree |
| `declared` | 79,309 | tag only; too short or too uncertain to second-guess |
| `detected` | 37,146 | no tag; detected English |
| `no_text` | 10,671 | link-only posts — recoverable with `~has_text` |
| `declared_other` | 6,291 | declared non-English, not confidently English |
| `detected_override` | 3,630 | **declared non-English but confidently English** |
| `detected_other` | 1,810 | no tag; detected non-English |
| `declared_overridden` | 897 | declared English but confidently not |
| `undetectable` | 61 | no tag, detection failed |

Note that `detected_override` **reclaims four times more posts than
`declared_overridden` removes**. A naive `langs == "en"` filter would have
silently discarded 3,630 genuinely English posts — typically English written by
users whose client reported `de`, `fr`, `ja` or `pt` — across 2,378 distinct
authors. That asymmetry is the main practical argument for not filtering on the
declared tag.

English share by query ranges from 96.6% (`age check`) to 89.2% (`eSafety`) and
62.4% (`porn ID check`, 266 rows), so the filter is close to uniform and does
not reshape the query mix.

## Decided — all five remaining items (2026-09-14)

Every open exclusion question is now settled. Each was measured before it was
decided; the evidence is recorded here so the choices can be defended or
revisited.

### 1. Per-query series use phrase-exact rows only

One uniform rule for every query. Series falling below a usable n are dropped as
series; **their posts stay in the corpus**, and most survive in another query's
series anyway because 32,225 posts matched more than one query.

| Series reported | English + phrase-exact |
|---|---:|
| `age verification` | 117,076 |
| `Ofcom` | 33,103 |
| `social media ban` | 29,835 |
| `Online Safety Act` | 20,027 |
| `verify your age` | 10,925 |
| `age assurance` | 6,966 |
| `age check` | 3,486 |
| `eSafety` | 2,853 |
| `under-16 ban` | 633 — thin, flag in any AU claim |
| *dropped as series* | `age verification privacy` (110), `age verification VPN` (45), `porn ID check` (5) |

### 2. Stratum B replaced by frame lexicons over corpus A

Stratum B was never independent of A (98%+ AND-matches) and collapses to ~160
posts under phrase-exactness. Frames are now measured from post **text** across
the 209,320-post stratum-A measurement corpus:

| Frame | Posts | Share |
|---|---:|---:|
| privacy / surveillance | 13,300 | 6.4% |
| circumvention / VPN | 6,428 | 3.1% |
| free speech | 5,632 | 2.7% |
| child safety | 3,888 | 1.9% |
| ID upload | 1,947 | 0.9% |

This measures frame prevalence without any query-list artefact, which is what
the two-stratum design was originally meant to guard against — but it is **not**
the pre-registered design and the report must say so.

**The result that justified the choice.** Daily frame share around E2 (UK
enforcement, 2025-07-25), memes excluded, shows three frames peaking at three
different times:

| Day | n | circumvent | privacy | ID upload |
|---|---:|---:|---:|---:|
| 2025-07-18 | 226 | 0.9% | 4.4% | 1.3% |
| 2025-07-24 | 1,986 | 6.9% | 5.2% | **5.1%** |
| 2025-07-25 | 2,024 | 7.1% | 5.6% | 3.7% |
| 2025-07-28 | 1,494 | **9.2%** | 5.7% | 1.8% |
| 2025-08-05 | 664 | 3.0% | **11.0%** | 0.5% |

ID-upload talk peaks the day *before* enforcement; circumvention peaks three
days *after*; privacy is still climbing two weeks later. Across events,
enforcement moves circumvention while announcements move privacy and free-speech
talk. E5 is excluded from the frozen core because its earlier 2026-03-11 date was
superseded by the authoritative 2026-03-09 event registry.

**Still owed:** the five lexicons are first-draft regexes written to size the
option. They need refinement and a hand-coded validation sample (300–500 posts,
reported agreement) before the report. They also find *vocabulary, not stance* —
"age verification protects privacy" and "destroys privacy" both match.

### 3. Volume series exclude memes and spam, keep news and bridged accounts

| Filter | Posts | |
|---|---:|---|
| stratum-A measurement corpus | 209,320 | |
| **− meme posts − spam/repeater accounts** | **181,625** | −13.2%, all 608 days kept |

News bots and bridged RSS accounts are **kept**: institutional voices are part of
public attention to a policy, even though they are not opinion.

The evidence that settled it — excluding memes reorders the peaks:

| | Top 6 days |
|---|---|
| All posts | 2026-02-10, 2026-07-13, 2025-07-11, 2026-06-15, 2026-02-09, 2025-09-10 |
| Memes excluded | 2026-07-13, 2026-06-15, 2026-02-09, **2025-12-10 (E3)**, **2025-07-25 (E2)**, 2025-07-24 |

Both enforcement events surface into the top six only once the meme is removed.
Exclusion rather than dummy regressors because `is_meme` identifies individual
posts — a day-level dummy would also absorb the genuine discourse on those days.

### 4. Engagement used as collected, with the check reported

**This corrects audit issue 10.** That issue flagged right-censoring as material
on the strength of a zero-like share rising 30% → 44% across the final weeks.
Measured properly on the agreed corpus, there is **no detectable age effect**:

| Age at collection | n | Mean likes | Median | Zero-like |
|---|---:|---:|---:|---:|
| 7–14 days | 290 | 11.24 | 1 | 44.1% |
| 14–30 | 3,753 | 11.50 | 1 | 41.7% |
| 30–60 | 7,646 | 10.89 | 1 | 39.1% |
| 60–120 | 31,854 | 10.00 | 1 | 34.2% |
| 120–240 | 45,411 | 11.47 | 1 | 42.5% |
| 240–480 | 78,678 | 11.07 | 1 | 41.7% |
| 480–900 | 13,363 | 12.14 | 1 | 46.6% |

The earlier gradient was confounded by content mix — the windows compared
differed in what was in them, not in how old they were. Engagement on Bluesky
lands within days and then stops. Only 2.2% of posts were under 30 days old at
collection, all in 2026-08-14…31. No correction applied; the table above goes in
limitations as measured evidence rather than an assurance.

### 5. Reply graph primary, follow graph for homophily

They cover **different populations** — only 3,365 of 71,242 reply-graph
participants are follow seeds — so this is not two views of one network.

| | Reply graph | Follow graph (seed-induced) |
|---|---|---|
| Nodes | 71,242 | 5,000 |
| Edges | 101,874 author→author | 158,284 |
| Largest component | 66,220 (93.0%) | — |
| Density | 2.0e-05 | 6.3e-03 |
| Reciprocity | 0.3% | — |

The 0.3% reciprocity confirms the reply graph is a discussion/conflict structure
rather than a friendship one: people reply to strangers. It carries the
diffusion and cross-community questions. The follow graph supplies community
structure and homophily on the seeds, **in-degree only** — out-degree is NA for
813 seeds (`data/interim/follows_seed_status.csv`), and the whole graph is a
2026-09 snapshot with no history.

## Work these decisions create

| | Task | Status |
|---|---|---|
| 1 | Materialise the series filter as a documented view | **done** — `src/platforms/bluesky/build_analysis_tables.py` |
| 2 | Refine the five frame lexicons; hand-code 300–500 posts; report agreement | **done** — κ 0.40–0.65, two coders |
| 3 | Build the author-level reply graph as a table | **done** — same script |
| 4 | Stance classification within frames | **dropped** — replaced by the circumvention indicator |

### Item 2 — frame lexicons refined and validated, 2026-09-14

`src/platforms/bluesky/frames.py` holds the lexicons; `src/platforms/bluesky/build_frames.py` applies them;
`src/platforms/bluesky/score_frames.py` scores them. Protocol and the remaining κ step are in
`docs/bluesky/frame_validation.md`.

**Refinement fixed three measurable defects**, all found by asking which posts a
term was the *sole* reason for flagging:

| Defect | Evidence |
|---|---|
| `censorship` contributed **0** unique posts | the pattern `censor` already matched it |
| bare `censor` matched image censor-bars | *"tip at least $5 to remove the silly censor"* |
| `safeguard` matched unrelated posts | *"how anthropic's claude ai protects user mental health"*; would also have matched "safeguard free speech", the opposing frame |
| `another age` matched 154 posts about nothing | *"that's another age verification step isn't it?"* — caught during v2 testing |

Recall terms were added for privacy objection expressed without the word
"privacy" — "scan my face", "hand over my id" — which the draft missed.

**Validated on 400 blind-coded posts** (precision and recall pools shuffled
together, all flags hidden from the coder):

| Frame | Precision | Recall | F1 |
|---|---:|---:|---:|
| `circumvent` | 82.8% | **91.4%** | **0.87** |
| `id_upload` | **94.9%** | 73.7% | 0.83 |
| `privacy` | 79.3% | 75.8% | 0.78 |
| `free_speech` | 78.1% | 78.1% | 0.78 |
| `child_safety` | 93.4% | **51.4%** | **0.66** |

`child_safety` is precise but misses half of what a human codes, because the
framing often carries no lexicon vocabulary — "Australia starts social media ban
for under-16s" invokes the protective rationale without saying "protect". **Its
trend is usable; its level is an undercount and must be labelled as such.**

**Cohen's κ, two coders, 100 shared items** — `circumvent` 0.647, `child_safety`
0.622, `privacy` 0.565, `free_speech` 0.407, `id_upload` 0.397.

The two weak frames fail differently, and the difference decides what is
fixable. **`id_upload` is a codebook failure**: 22 of 23 disagreements run one
way, because the boundary between "age verification generally" and "the
mechanics of proving age" was under-specified in my codebook. **`free_speech` is
genuine construct ambiguity**: the disagreement is balanced, over whether
identity-exposure and bare-hashtag posts count.

Scoring the lexicon against each coder shows the resulting uncertainty —
`privacy` F1 is 0.77 against one coder and 0.61 against the other, `id_upload`
0.78 against one and 0.41 against the other. **That spread, not the single
figure, is what the frame numbers actually carry.**

The codebook was deliberately not sharpened and re-coded until κ improved:
iterating a validation until it yields an acceptable number fits the test to the
answer. Full detail in `docs/bluesky/frame_validation.md`.

### Circumvention indicator — built and validated 2026-09-14

`circumvention_act` flags posts where the author reports **personally** getting
around age verification. Self-evidencing, so no stance model is involved: the
author states what they did rather than an attitude being inferred.

**555 posts across 526 distinct authors** — not a handful of accounts.

| Period | per 1,000 posts |
|---|---:|
| 2025-01 … 06, before UK enforcement | **0.29** |
| 2025-07 … 2026-08 | **3.35** |

A **10.9× step change that never returns to baseline.** This is the project's
most direct measurement of circumvention as behaviour rather than as discussion,
and it is what made dropping stance classification affordable.

**Validated separately, on its own held-out sample.** v1 scored precision 72.0%
on 100 blind-coded items. Two faults showed up in the false positives: the act
verb and the circumvention term could sit anywhere in the post (so *"every time
i **use** the app"* plus a stray "vpn" counted), and explicit denials were
counted as admissions (*"i use a vpn ... **not to evade** the online safety
act"*) — inverting the author's stated meaning, which is the one error this
indicator most needs to avoid.

| | Precision | Recall (within frame) | F1 |
|---|---:|---:|---:|
| v1 | 72.0% | 83.7% | 0.77 |
| v2, tuning set (100) | 84.8% | 90.7% | 0.88 |
| **v2, held-out (60 fresh)** | **86.7%** | **86.7%** | **0.87** |

The held-out figures are the quotable ones — measured on items drawn after
tuning and never used to adjust anything. That they match the tuning set (0.87
vs 0.88) is the evidence the fix generalises rather than fitting its own sample.

The correction removed 36 of 555 flagged posts and moved the headline from 11.6×
to 10.9×, which is the useful part: **a step change that size did not depend on
the instrument being perfect.** Evidence kept in
`data/interim/act_validation_tuning.csv` and `act_validation_heldout.csv`.

### Items 1 and 3 — built 2026-09-14

`src/platforms/bluesky/build_analysis_tables.py` turns the decisions into four joinable tables.

| Table | Rows | Grain |
|---|---:|---|
| `corpus_flags` | 423,628 | uri → `in_corpus` + `exclusion_reason` |
| `series_daily` | 4,811 | (query, day) → count, denominator, normalised rate |
| `reply_edges` | 90,269 | author → author, weighted |
| `reply_nodes` | 63,408 | author → degrees, component |

**Analysis corpus: 372,674 of 423,628 posts (88.0%).** Exclusions are recorded
as a reason rather than a bare boolean, so "what did we drop and why" is
answerable from the data: 24,162 meme, 19,730 not English, 7,062 spam account.

`phrase_exact` is deliberately *not* in `in_corpus`. It is a property of a
(post, query) pair, not of a post — the same post can be exact for
`age verification` and scattered for `under-16 ban` — so it belongs in the
series definition, where it is applied, and nowhere else.

**Nine reportable series**, phrase-exact over the corpus:

| Query | Posts | | Query | Posts |
|---|---:|---|---|---:|
| `age verification` | 87,989 | | `age check` | 3,431 |
| `Ofcom` | 32,418 | | `eSafety` | 2,540 |
| `social media ban` | 28,825 | | `under-16 ban` | 613 |
| `Online Safety Act` | 19,498 | | *dropped:* `age verification privacy` | 96 |
| `verify your age` | 10,900 | | *dropped:* `age verification VPN` | 44 |
| `age assurance` | 6,838 | | *dropped:* `porn ID check` | 5 |

`age verification` falls from 117,076 to 87,989 because the meme posts begin
with the literal phrase and were therefore phrase-exact matches for it. Removing
them takes a quarter of that series — which is exactly the contamination the
meme flag exists to remove, landing where it was always going to land hardest.

**Baseline normalisation matters more than expected.** Platform drift does not
merely dampen the trend, it conceals its size, and it changes the ranking of
events:

| | First 3 months | Last 3 months | Growth |
|---|---:|---:|---:|
| Raw posts | 3,435 | 14,111 | 4.1× |
| Per 10k baseline | 51.4 | 315.9 | **6.1×** |

| Event | Raw posts | Per 10k |
|---|---:|---:|
| E1 2025-06-27 US announce | 732 | 334.7 |
| **E2 2025-07-25 UK enforce** | **2,243** | 1,186.8 |
| **E3 2025-12-10 AU enforce** | 2,126 | **1,329.5** |
| E4 2026-01-21 UK announce | 338 | 189.4 |
| E5 2026-03-11 (unverified) | 373 | 216.2 |

On raw counts E2 is the larger enforcement event; **normalised, E3 overtakes
it.** Any claim about which mandate provoked more discourse depends on the
denominator, so the denominator has to be reported alongside it. Four days of
608 carry a null denominator because a saturated baseline cell makes the basket
incomplete for that day — summing the remaining four terms would understate the
day and inflate its rate, and the missing term cannot be imputed because it
saturated by being unusually busy.

**Reply graph**, both ends filtered to the corpus, self-replies dropped
(17,591 of them):

| | |
|---|---|
| Author→author edges | 90,269 (from 104,455 reply posts) |
| Authors | 63,408 |
| Largest component | 58,747 (**92.6%**) |
| Reciprocity | **0.23%** (212 edges) |
| Mean edge weight | 1.16; 9,573 edges repeat |

The 0.23% reciprocity confirms what kind of object this is: a discussion
structure where people reply to strangers, not a mutual-relationship network.
The most-replied-to account has in-degree 2,661 against out-degree 2 — a
broadcaster being answered en masse, never answering back.

## Techniques used, for the methods section

Worth stating accurately rather than generically:

| Tier | Technique | Status |
|---|---|---|
| Deterministic / rule-based | URI dedup, MD5 text hashing, anchored regex flags, relational normalisation into four tables, threshold rules, DID joins, NFKC folding, sha256 manifest | **done** — the large majority of the work |
| Classical ML / statistical | **langdetect** (n-gram language identification, seeded for determinism) combined with the platform's declared tag under a measured confidence rule | **done** |
| Statistical (planned) | interrupted time series with meme-wave and spam dummies rather than deletion | not started |
| LLM-based | topical relevance classification for the loose queries; adjudicating ambiguous meme cases | **not started** — awaiting team decision |

Note for the methods section: the language step is **not** "we used langdetect".
Detection alone is wrong below 60 characters (49% error under 20), and the
declared tag alone is wrong in both directions (it drops 3,630 genuinely English
posts and keeps 897 non-English ones). What was actually built is a hybrid whose
threshold was set from measurement, and which is deliberately asymmetric because
the two error costs are unequal. That is the defensible claim, and it is
stronger than either simple one.

The data-quality audit itself was **LLM-assisted auditing**, not LLM-based
cleaning: every detection mechanism is a regex, a set operation or a counter.
What an LLM contributed was hypothesis generation, reading samples to tell
detector noise from genuine signal, and noticing when two measurements of the
same quantity disagreed. If an LLM classifier is later used for relevance, it
needs a hand-coded validation sample (300–500 posts) with reported agreement, or
it is not defensible.

---

# Changelog

Newest first. Append here as work happens.

## 2026-09-14 — processed dataset built; code moved to the platforms layout

- Ran `python -m src.platforms.bluesky.anonymise` over `data/interim/` into
  `data/processed/bluesky/` (99 MB, 15 tables + `prepare_manifest.json`). This
  **supersedes the `data/shared/` output** described below, which predates email
  and phone masking and used a local salt; that folder must not be shared.
- Independent check of the output, not relying on the script's own audit: row
  counts equal source for every table (`posts` 423,628, `authors` 160,648,
  `reply_edges` 90,269); no `did`, `handle`, `uri` or parent/root URI columns;
  every `*author_hash` is 16 hex; post authors all present in `authors` and every
  reply-edge source present in `posts`; `doc_id` unique. In text: 1,776 `<PII>`
  in 1,770 posts, 12,480 `@user`, 1,939 `<URL>`, 1,324 sensitive posts. A broad
  scan found no DIDs, handles or emails; ~1,200 phone-shaped strings were
  inspected and are dates, plain numbers and article/DOI/ISBN identifiers; the
  one `at://` hit is a truncated `did:plc...` with no identifier.
- **Salt.** The owner chose a Bluesky-specific `PSEUDONYM_SALT` rather than the
  team value. Nothing is lost for analysis: D-017 means no cross-platform join,
  and the platform name is inside the hash input. Reproducing these hashes needs
  this salt, shared privately by the owner.
- **Not in the processed output:** the follow graph (the earlier `data/shared/`
  had `follow_edges.parquet`, built from raw; the new step reads `data/interim/`
  only) and the frame hand-coding sheets and codes (scores ship, codes do not).
- Rebased onto `main` after the YouTube and Reddit merges and moved to the team
  layout: `src/platforms/bluesky/`, `tests/platforms/bluesky/`, shared helpers
  from `src.shared`. Module paths in the entries below were updated to match.
  Fixed `probe.py`'s repository root, wrong since the port (it pointed at `src/`).

## 2026-09-14 — dataset pseudonymised and split for sharing

- `src/platforms/bluesky/anonymise.py` writes `data/shared/` (~130 MB): every account identifier
  replaced by a keyed hash, handles/display names/bios/avatars/post URLs removed,
  `@mentions` and profile links scrubbed from post text. **967,409 accounts and
  503,551 posts** pseudonymised; joins verified intact across all tables.
- Built `follow_edges.parquet` (3,717,888 edges) from raw, which had existed
  nowhere in `data/interim/`.
- A self-audit fails the run if any DID, AT-URI or handle survives. It passes.
- **This is pseudonymisation, not anonymisation**, and the docs say so: post text
  is retained and is publicly searchable, so individual posts stay
  re-identifiable. Removing text would defeat the dataset's purpose. The claim to
  make is "pseudonymised, with post text retained".
- The salt lives in `.anon_salt`, gitignored, and must never travel with the
  data — with it, anyone holding a DID list can test membership.
- Repo/drive split enforced in `.gitignore`: code and docs to the repo, `data/`
  and `logs_*.txt` excluded, `data/manifest.json` deliberately re-included as the
  link between the two. (The negation needed `data/*` rather than `data/` — git
  cannot re-include a file inside an excluded directory, so the first attempt
  silently did nothing.)
- Verified by dry run: **31 files, 442 KB, zero sensitive paths.**

## 2026-09-14 — Cohen's κ computed, frame validation complete

- Second coder finished 100 items on the coding bench artifact; codes read back
  from the page's store and scored.
- κ: `circumvent` 0.647, `child_safety` 0.622, `privacy` 0.565,
  `free_speech` 0.407, `id_upload` 0.397.
- Diagnosed the two weak frames as failing for different reasons — `id_upload` a
  one-directional codebook failure (22 of 23 disagreements one way),
  `free_speech` a balanced construct ambiguity.
- Added a sensitivity table scoring the lexicon against each coder, since the
  single-coder F1 figures understate the uncertainty by roughly 0.15–0.37.
- Did **not** re-code to improve κ; recorded why.

## 2026-09-14 — circumvention indicator validated and refined

- Blind-coded 100 items: v1 precision 72.0%, recall 83.7%.
- Fixed two faults — added an 80-character proximity requirement between the act
  verb and the circumvention term, and a denial exclusion so "not to evade the
  online safety act" no longer counts as evading it.
- Re-validated on **60 fresh held-out items**: precision 86.7%, recall 86.7%,
  F1 0.87 — matching the tuning set, so the fix generalises.
- Headline moved 11.6x → **10.9x**; 555 → 519 flagged posts, 493 distinct authors.
- Only `FIRST_PERSON_ACT` was touched, not the five frame patterns, so the
  in-progress κ coding sheet and its answer key remain valid.
- Closed four README gaps (`build_analysis_tables`, `build_frames`, `frames`,
  `score_frames` were undocumented) and corrected the README status line, which
  still claimed no analysis existed.

## 2026-09-14 — item 2 (frames) and the circumvention indicator

- Refined the five lexicons into `src/platforms/bluesky/frames.py`, fixing four defective terms
  found by sole-reason analysis (`censorship` contributing 0 unique posts, bare
  `censor` matching image censor-bars, `safeguard` matching AI-wellbeing posts,
  `another age` matching 154 irrelevant posts).
- Built a 400-item blind validation sample, coded it, and scored:
  F1 0.87 `circumvent`, 0.83 `id_upload`, 0.78 `privacy`, 0.78 `free_speech`,
  0.66 `child_safety`.
- **`child_safety` recall is 51.4%** — report its trend, not its level.
- Built `src/platforms/bluesky/build_frames.py` → `frames.parquet`, `frames_daily.parquet`.
- **Circumvention indicator: 0.29 → 3.35 per 1,000 posts (11.6x)** from the UK
  enforcement month, across 526 distinct authors. Self-evidencing, no classifier.
- Wrote `docs/bluesky/frame_validation.md` with the codebook and the κ procedure.
  **κ remains outstanding and needs a second coder.**

## 2026-09-14 — analysis tables built (decision items 1 and 3)

- `src/platforms/bluesky/build_analysis_tables.py` writes `corpus_flags`, `series_daily`,
  `reply_edges`, `reply_nodes`. Analysis corpus 372,674 posts (88.0%); nine
  reportable series; reply graph 90,269 edges over 63,408 authors, 92.6% in one
  component.
- **Baseline normalisation changes a substantive conclusion.** Raw counts make
  E2 (UK enforcement) the larger event at 2,243 posts against E3's 2,126;
  normalised per 10k baseline, E3 overtakes it (1,329.5 vs 1,186.8). Growth
  across the window is 4.1x raw but 6.1x normalised.
- **Fixed a denominator bug before it shipped.** Masking a saturated baseline
  cell had left the day summed over the remaining four terms, making the basket
  too small and the normalised rate too high — the opposite of the intended
  correction. Those four days now carry a null denominator; the missing term
  cannot be imputed because it saturated by being unusually busy.

## 2026-09-14 — all five remaining exclusion decisions settled

- Phrase-exact series, frame lexicons replacing stratum B, meme+spam excluded
  from the series, engagement used as-is, reply graph primary. Full evidence in
  Part 2 above.
- **Corrected audit issue 10.** Engagement right-censoring was flagged as
  material on a zero-like gradient that turned out to be confounded by content
  mix. Measured across age buckets from 7 to 900 days, engagement is flat
  (mean 10.0–12.1, median 1 throughout). No correction needed.
- Measured for the first time: author-level reply graph (71,242 nodes, 101,874
  edges, 93% in one component, 0.3% reciprocity) and the overlap between the
  reply and follow populations (3,365 shared nodes, 4.7%).

## 2026-09-14 — English-only decision implemented

- Added `is_english`, `english_basis`, `lang_detected`, `lang_detect_prob` and
  `declares_english` to `posts.parquet`. 403,898 of 423,628 posts (95.3%).
- Measured before choosing the rule rather than after: declared-vs-detected
  agreement (94.9%), disagreement by text length (49.1% under 20 chars, 0.3%
  above 120), and detector confidence on the disagreements (bimodal at p=0.90
  and p=0.999). Detection may overrule a tag only at ≥60 chars and p ≥ 0.999.
- **Tightened the rule after reading samples.** A length-only threshold
  overrode 1,519 declared-English posts, but samples showed several were plainly
  English — "denmark to ban social media for under-15s" scored Danish,
  "[ai + swearing + online safety act] #granny" scored Welsh. Adding the
  confidence requirement cut the overrides to 897 and rescued 622 posts.
- **Changed the time axis.** `indexed_at` turned out not to be stable: 1,750
  search posts (0.59%) carry an index day outside the cell that retrieved them,
  one nine days past the window end, because the AppView re-indexes some posts.
  That put a 609th day in a 608-day window. The canonical column is now `day` —
  collection cell for search posts, index day for replies — giving exactly 608
  days with no empty ones. `cell_day`, `index_day` and `index_day_drift` are
  kept for audit.
- Rebuilt `data/manifest.json`.

## 2026-09-14 — cleaning groups C, D and E (normalisation)

- Installed `pandas`, `pyarrow`, `langdetect` (already listed in
  `requirements.txt` for this phase).
- Built `src/platforms/bluesky/normalise.py`: four Parquet tables in `data/interim/` —
  `posts` (423,628), `post_query` (338,637), `thread_edges` (135,034),
  `authors` (160,648). Annotates only; excludes nothing.
- Built `src/platforms/bluesky/build_manifest.py` → `data/manifest.json`: per-layer rows, bytes,
  date range and rolled-up sha256, plus hashes of `config/bluesky/config.yaml` and all 15
  scripts, so a copy on the shared drive can be verified.
- Verified: 0 duplicate URIs, 0 dangling joins, 0 authors without a profile.
- **Found and closed a gap the repair pass opened.** Re-collecting the 33 capped
  threads pulled in replies from **380 authors the corpus had never seen**, and
  they had no profile because pass 4 had already finished. Re-ran
  `collect_profiles.py` (checkpointed, fetched only the 380). Pass 4 must now run
  after pass 3b; noted in the README.
- **Revised the `meme_wave_day` definition mid-build.** A share-only threshold
  flagged 21 days, but that conflates two different things: on 2026-08-22 the
  meme is 32.7% of a 272-post day (0.8x median — present, harmless), while on
  2025-07-11 it is 78.6% of 5,044 unique posts (14.3x median — the case the flag
  exists for). Now requires share ≥20% **and** volume ≥2x median, giving 10 wave
  days. `day_meme_share` and `day_volume_ratio` are emitted per post so the
  threshold can be revisited without re-running the pass.
- **Fixed a config bug.** `paths.interim` pointed at a single file
  (`data/interim/bluesky.parquet`), inherited from an earlier single-table
  design, so the first run wrote the tables *inside* a directory of that name.
  Renamed to `interim_dir: "data/interim"` and corrected `ensure_dirs()`.

## 2026-09-14 — cleaning groups A and B

- Ran `src/platforms/bluesky/recollect_capped_threads.py`: 33 capped threads repaired, +1,591
  replies, none still capped. Reply corpus 133,443 → **135,034 rows /
  133,732 unique**.
- Corrected the reply count, meme figures, 2025-07-11 post count and profile
  count across `README.md` and `docs/bluesky/probe_findings.md`.
- Built `data/interim/follows_seed_status.csv` and
  `data/interim/baseline_saturated_cells.csv` via new
  `src/platforms/bluesky/build_correction_tables.py`.
- **Corrected two of my own audit findings**, both because building the artifact
  forced a cross-check against a second layer:
  - The 239 zero-edge follow seeds are **not** all deleted accounts. 230 have a
    profile independently reporting `follows_count = 0` and genuinely follow
    nobody (largely news bots and bridged RSS accounts); only **9** are missing
    data.
  - DIDs carrying more than one handle: **22**, not 1. The first count scanned
    only `posts_*.jsonl`; including threads and profiles raises it.
- Fixed a bug in `recollect_capped_threads.py`: it selected threads by
  `replies >= 200`, which would re-select repaired threads forever (one now
  holds 493). Now also requires `not recollected`. Verified idempotent.

## 2026-09-14 — data quality audit

- Full read-only audit of all five layers, 2.5 GB. Wrote
  `docs/bluesky/data_quality_audit.md`: 20 issues, 4 blocking, plus 12 checks that
  passed.
- Headline findings: multi-word queries match scattered terms not phrases;
  three meme waves rather than one, including an undocumented wave peaking
  2026-02-10 at 75% of a 9,053-post day; a single spam account accounting for
  68–83% of three days in October 2025; and the reply count in the
  documentation understating the corpus by 31%.

## 2026-09-13 / 14 — collection

- All five passes completed. See Part 1.
