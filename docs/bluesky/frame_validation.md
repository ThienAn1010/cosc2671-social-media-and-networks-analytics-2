# Frame lexicons — validation, and how to finish it

**Status: complete.** Two independent coders, κ computed 2026-09-14.
**Read the reliability section before quoting any frame figure** — two of the
five frames fall below κ 0.5 and carry a caveat.

---

## What has been done

Five frame lexicons (`src/platforms/bluesky/frames.py`) were refined from a draft, then scored
against 400 hand-coded posts. The sample was **blind**: precision and recall
items were pooled, shuffled, and stripped of all flags before coding, so the
coder could not tell which pool an item came from or which frame had flagged it.

| Frame | Precision | Recall | F1 |
|---|---:|---:|---:|
| `circumvent` | 82.8% | **91.4%** | **0.87** |
| `id_upload` | **94.9%** | 73.7% | 0.83 |
| `privacy` | 79.3% | 75.8% | 0.78 |
| `free_speech` | 78.1% | 78.1% | 0.78 |
| `child_safety` | 93.4% | **51.4%** | **0.66** |

**Read `child_safety` with care.** It is precise but misses about half of what a
human codes as child-safety framing.

**One concrete cause was found while writing the tests:** the pattern allows only
`of` between the verb and the noun, so *"protect children"* matches but
*"protect **the** children"* does not. The definite-article form is at least as
common as the bare one, and it is the exact phrasing of the rhetorical move the
frame exists to capture — "it was never about protecting the children". That
alone plausibly accounts for a large share of the missing recall.

It has **not** been fixed, deliberately. Precision, recall and Cohen's κ were all
measured against the current patterns; widening one now would invalidate every
figure in this document. `tests/platforms/bluesky/test_bluesky_frames.py` pins the current
behaviour so that anyone who widens it is forced to notice they owe a
re-validation. That is the honest order: re-measure, then change, then republish
the numbers.

The rest of the miss is framing that carries no lexicon vocabulary at all — "Australia starts social media ban for under-16s"
engages the protective rationale without saying "protect", "safeguard" or
"child safety". Its *level* is therefore an undercount. Its *trend* is more
trustworthy than its level, provided the miss rate is stable over time.

## Inter-rater reliability — the headline result

Two coders, 100 shared items, coded independently and blind.

| Frame | Agreement | κ | Band |
|---|---:|---:|---|
| `circumvent` | 92.0% | **0.647** | substantial |
| `child_safety` | 83.0% | **0.622** | substantial |
| `privacy` | 83.0% | **0.565** | moderate |
| `free_speech` | 87.0% | **0.407** | fair |
| `id_upload` | 77.0% | **0.397** | fair |

### The two low scores fail for different reasons

This distinction matters more than the numbers, because it decides what can be
fixed and what has to be lived with.

**`id_upload` (κ 0.397) is a codebook failure, and the codebook is mine.** The
disagreement is almost entirely one-directional: coder A marked it 34 times,
coder C 13, and **22 of the 23 disagreements are A's marks, not C's**. A applied
the frame to posts about age verification in general; C applied it only where
documents, face scans or digital ID are the actual subject. The codebook said
"the mechanics of proving age" with three examples, which was not enough to draw
that boundary. The instrument's *documentation* failed, not the coders.

**`free_speech` (κ 0.407) is genuine construct ambiguity.** Here the
disagreement is balanced — 8 one way, 5 the other. A counted "trans people
forced to out themselves" as a speech issue; C filed it under privacy. C counted
a bare `#Censorship #Fascism` hashtag post; A did not. Neither reading is wrong.
The construct has a soft edge, and that is a finding about the concept rather
than a defect to repair.

### What this does to the precision and recall figures

Scoring the lexicon against each coder as gold standard, on the same 100 items:

| Frame | F1 vs coder C | F1 vs coder A |
|---|---:|---:|
| `privacy` | 0.77 | 0.61 |
| `circumvent` | 0.83 | 0.69 |
| `child_safety` | 0.67 | 0.52 |
| `free_speech` | 0.74 | **0.42** |
| `id_upload` | 0.78 | **0.41** |

**That spread is the real uncertainty in the single-coder figures reported
above.** Quoting `privacy` F1 as 0.78 would overstate what is known: it is
0.61–0.77 depending on whose judgement is taken as truth. Report the range, or
report the single-coder figure with this table beside it. Do not report one
number as if it were a fixed property of the lexicon.

### What was deliberately NOT done

The codebook was **not** sharpened and re-coded until κ improved. Iterating a
validation until it produces an acceptable number is fitting the test to the
answer, and it would make every figure here meaningless. The results stand as
first measured.

### Recommendation

1. **Report as-is** — κ 0.40–0.65 with the diagnosis above. `circumvent` and
   `child_safety` are reportable without qualification; `id_upload` and
   `free_speech` carry an explicit reliability caveat.
2. Note as a limitation that **`id_upload` and `privacy` may not be separable**
   in practice. They are the two coders conflated most, and "handing over your
   documents" is arguably the concrete form of the privacy frame rather than a
   frame of its own.
3. `circumvent` is the strongest frame at κ 0.647 — which is fortunate, since it
   carries the project's circumvention finding.

## Why a second coder was needed

The scores above compare the lexicon to **one** person's judgement. That shows
the instrument matches a human, but not that the human is reproducible — and I
wrote the lexicons, so my notion of "privacy frame" is the same notion that
produced the regex. Blind coding removes the crudest bias but cannot remove that
one.

Cohen's κ measures agreement between two independent coders, correcting for
agreement that would happen by chance. It is what turns "we checked" into "we
measured", and it is the statistic a marker will look for.

---

## What to do — about 45 minutes

### 1. Open the coding sheet

```
data/interim/frame_validation_sheet.csv
```

400 rows: `item_id`, `text`, then five empty columns. **Code only the first
100 rows** — that is enough for a reportable κ and keeps the task short. Do not
open `frame_validation_key.csv`; it holds the answers.

### 2. Code each post against the codebook below

Put any non-empty mark (`1`, `x`, `y`) in a frame's column if that frame applies;
leave it blank if not. A post can have **any number of frames, including none** —
many are neither, and that is expected.

### 3. The codebook

Code what the post **engages**, not whether you agree with it. A post arguing
*against* a privacy objection still engages the privacy frame.

| Frame | Code it when the post… | Examples |
|---|---|---|
| **privacy** | treats age verification as a data, surveillance, anonymity or identity-exposure issue | "they'll have my passport in a database"; "this is surveillance under another name"; "could force trans people to out themselves" |
| **circumvent** | discusses getting around age verification — by anyone, including hypothetically | "kids will just use a VPN"; "you can bypass it with a browser extension"; "1,400% jump in VPN signups" |
| **id_upload** | concerns the *mechanics* of proving age: documents, face scans, selfies, digital ID | "upload your ID or a selfie"; "Discord will require a face scan"; "digital ID plan" |
| **child_safety** | invokes protecting children/minors as a rationale, whether endorsing or attacking it | "to keep kids safe"; "it was never about protecting children"; "ban for under-16s" |
| **free_speech** | frames it as censorship, speech restriction, or state overreach | "this is censorship"; "violates the First Amendment"; "authoritarian creep" |

**Rules of thumb**

- **Engagement, not agreement.** "It was never about protecting children" *is*
  child_safety — it argues about that rationale.
- **Explicit beats implied.** If you have to construct an argument for why a
  frame applies, it probably does not.
- **Descriptive news counts.** "EU regulators open child-protection probe" is
  child_safety even with no opinion in it.
- **A ban on under-16s is child_safety** even when protection is not stated —
  that is the rationale the policy exists under.
- **When genuinely torn, leave it blank** and note why in `notes`. Systematic
  blanks are more informative than coin-flips.

### 4. Save your codes as a text file

One line per item, `item_id:LETTERS`, where letters are any of
**P**rivacy, **C**ircumvent, **I**d_upload, child **S**afety, **F**ree speech.
An item with no frames still needs a line:

```
1:IS
2:S
3:
4:F
5:I
```

Save as `data/interim/frame_codes_<yourname>.txt`. Order does not matter; every
item you coded must appear exactly once.

### 5. Run the scorer

```bash
python -m src.platforms.bluesky.score_frames \
    data/interim/frame_codes_claude.txt \
    data/interim/frame_codes_<yourname>.txt
```

It prints precision/recall/F1 against the first file and **Cohen's κ between the
two**, with the conventional interpretation band, and writes
`data/interim/frame_validation_scores.csv`.

### 6. What to report

Quote κ per frame with its band. The conventional reading:

| κ | Interpretation |
|---|---|
| ≥ 0.81 | almost perfect |
| 0.61–0.80 | substantial |
| 0.41–0.60 | moderate |
| 0.21–0.40 | fair |
| < 0.21 | slight or worse |

**If a frame lands below ~0.6, say so rather than quietly dropping it.** A
disagreement that size usually means the construct is ambiguous, not that a
coder was careless, and that is a finding about the concept worth reporting.

---

## Honest limitations to carry into the writeup

1. **Lexicons find vocabulary, not stance.** "Age verification protects privacy"
   and "age verification destroys privacy" both count as `privacy`. Frame
   prevalence is therefore *salience*, not opposition. Stance classification was
   scoped out of this project deliberately; see the cleaning log.
2. **`child_safety` recall is 51.4%.** Report its trend, not its level, and say
   why.
3. **Reliability is moderate at best.** κ ranges 0.40–0.65 across the frames.
   Two frames are in the "fair" band, which is weak, and the honest reading is
   that frame assignment is reproducible for `circumvent` and `child_safety`
   and only partly so for `id_upload` and `free_speech`.
4. **The sample is from the stratum-A measurement corpus**, so these error rates
   apply there and not to the raw corpus.

---

## The circumvention indicator is separate, and does not need this

`circumvention_act` flags posts where the author reports **personally** getting
around age verification — "i forgot my vpn was on", "i had to use a vpn to load
my dms". It is self-evidencing: the author states what they did, so no stance
model and no attitude inference is involved.

**519 posts across 493 distinct authors**, so it is not a handful of accounts.

| Period | per 1,000 posts |
|---|---:|
| 2025-01 … 2025-06 (before UK enforcement) | **0.29** |
| 2025-07 … 2026-08 | **3.13** |

A **10.9× step change** that never returns to baseline. It is the most direct
measurement this project has of circumvention as behaviour rather than as talk.

### It has been validated separately — 2026-09-14

A first version scored **precision 72.0%, recall 83.7%** on 100 blind-coded
items. The false positives showed one structural fault and one conceptual one:

* **No proximity requirement.** The act verb and the circumvention term could sit
  anywhere in the post, so *"twitter doesn't have this horseshit that i have to
  fight with every time i **use** the app"* counted because a VPN was mentioned
  in a different sentence.
* **Denials counted as admissions.** *"i use a vpn precisely because i want to
  feel safe on the internet, **not to evade** the online safety act"* was scored
  as circumvention — inverting what the author said. That is the single error
  this indicator most needs to avoid, since its whole claim over a stance
  classifier is that it reports what people say they did.

Both were fixed: the act verb and the circumvention term must now fall within 80
characters of each other, and an explicit denial disqualifies the post.

| | Precision | Recall\* | F1 |
|---|---:|---:|---:|
| v1 | 72.0% | 83.7% | 0.77 |
| v2, tuning set (100 items) | 84.8% | 90.7% | 0.88 |
| **v2, held-out set (60 fresh items)** | **86.7%** | **86.7%** | **0.87** |

\*Recall is measured *within the `circumvent` frame* — the question is how many
genuine personal reports the indicator finds among posts that discuss
circumvention at all, not among the whole corpus.

The held-out figures are the ones to quote. They were measured on 60 items drawn
after tuning and never used to adjust anything, which is why they are
trustworthy where the tuning-set figures are optimistic by construction. That
they barely differ (0.87 vs 0.88) is the evidence that the fix generalises
rather than fitting the sample it was built on.

Both samples, with codes attached, are kept as evidence in
`data/interim/act_validation_tuning.csv` and `act_validation_heldout.csv`.

**What survives the correction.** The refinement removed 36 of 555 flagged posts
and moved the headline from 11.6× to **10.9×**. A step change of that size is not
sensitive to a precision error of this magnitude, which is worth stating: the
result did not depend on the instrument being perfect.

**Still single-coder.** Like the frame lexicons, this was coded by me alone. It
is a smaller claim than the frames carry, and it would benefit from the same
second-coder treatment, but it is not blocked on the κ exercise above.
