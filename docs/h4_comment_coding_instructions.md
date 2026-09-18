# H4 Comment Coding Instructions (Kiên)

## What this is for

H4 asks whether YouTube comments follow their video's framing more closely under news videos than under commentary videos. The first run found that pattern, but most videos had only about four coded comments, so the result didn't hold up under stricter checks. These 292 extra comments bring most videos up to about five usable comments.

Hung coded the first 30. You're coding the other **262**, about 4.5 hours at roughly a minute each. Claude codes all 292 separately as the second coder, and Hung settles the disagreements afterwards.

Please do this before your H1 to H3 Reddit coding: it's shorter, and H4 is the test closest to done.

## Getting started

Choose one option and stick with it.

### Option A: online page (recommended)

1. Open the Comment Framing Bench link Hung shares with you: https://claude.ai/artifact/38RCqTswguJUEsRrzjvHrh
2. Press **Kiên**, then **Start coding**. You'll see only your 262 comments.
3. Labels save as you go and sync across devices, so there's nothing to send back. Press **Save and pause** when you stop.

### Option B: offline files

Hung puts `h4_comments_kien.zip` in your Teams folder. It contains:

| File | What it is |
|---|---|
| `comment_framing_bench.html` | The same page, working offline. Double-click to open it in Chrome, Edge or Firefox. |
| `youtube_topup_kien.csv` | Your 262 comments as a spreadsheet |
| `README_comment_coding.md` | This file |

- **Page:** pick **Kiên** and code. Labels save in that browser. At the end of every session, press **Save and pause**, then **Download my labels (CSV)**, as a backup. To move to another browser, use **Restore from a CSV**.
- **Spreadsheet:** open it in Excel through **Data > From Text/CSV** with UTF-8. Fill in only `relevance`, `language`, `target_policy`, `frame_labels` and `notes`, then save as **CSV UTF-8**. Don't change `item_id`, the text columns or the row order.

Return offline files to `h4_comments_kien/returned/` in the Teams folder and tell Hung.

## What you see

For each comment:

1. **Video title and description:** the video the comment sits under.
2. **Replying to:** the comment above it, for replies only.
3. **Code this:** the comment you label.

Usernames and links are already masked (`@user`, `<URL>`, `<PII>`).

## What to label

These are the same four questions as the H1 to H3 pages. Full rules are in `docs/age_gate_manual_annotation_instructions.md`.

| Question | Values | Rule of thumb |
|---|---|---|
| `relevance` | `relevant`, `adjacent_contextual`, `irrelevant` | Relevant means the comment itself discusses age checks, the policy or a direct consequence of it. Adjacent means related background (online safety, privacy, speech) that doesn't touch the policy. |
| `language` | `english`, `mixed`, `non_english`, `too_short_ambiguous` | Use too short only for emoji-only, link-only or garbled text. |
| `target_policy` | `UK_OSA`, `AU_SOCIAL_MINIMUM_AGE`, `OTHER_EXTENDED_EVENT`, `multiple`, `unclear` | See the rule below. |
| `frame_labels` | any of the five frames, or none | Tick only what the comment itself argues. |

| Frame | Use when the comment argues about |
|---|---|
| `policy_assurance` | What the rule or age check is, how it works, its rollout |
| `child_safety` | Protecting children, harmful content, parents' role |
| `privacy_surveillance` | ID uploads, face scans, data leaks, tracking, anonymity |
| `governance_platform_responsibility` | Who enforces it, fines, regulators, platform or government duty |
| `circumvention_censorship_autonomy` | Getting around it as an argument, censorship, overreach, whether it can work |

Three rules matter most for H4:

1. **Code the comment's own framing, not the video's.** The test measures whether comments follow the video. Carrying the video's frames over to its comments would inflate the result.
2. **Use the video to identify the policy.** If a comment says "this ban" or "the law" under a video about Australia's under-16 ban, choose `AU_SOCIAL_MINIMUM_AGE`. Only comments that name a specific policy count toward H4, so don't leave these as unclear. Keep `unclear` for videos that cover several countries, or comments that could mean any policy.
3. **Be strict about frames.** Tick policy and assurance when the comment is about the rule or the check itself. Tick another frame only when the comment argues it outright. A bare VPN mention isn't circumvention unless it's part of an argument about avoiding the rule or whether it can work. Jokes, insults, shout-outs and pure reactions get no frame.

On the page, picking Irrelevant fills in Unclear and No frame for you. In a spreadsheet, leave `frame_labels` blank and write `no_supported_frame` in `notes` when no frame applies.

## Keyboard shortcuts

| Question | Keys |
|---|---|
| Relevance: relevant / adjacent / irrelevant | 1 2 3 |
| Language: English / mixed / not English / too short | Q W E R |
| Policy: UK / Australia / other / several / unclear | A S D F G |
| Frames: policy, child safety, privacy, governance, circumvention | Z X C V B |
| No frame | N |
| Next / previous | Enter / ← |

## Rules

- Code on your own and don't look at anyone else's labels. Hung settles disagreements after all 292 are in.
- Don't paste comment text into chats, documents or AI tools.
- Use notes for real problems: sarcasm you can't resolve, broken text, or a comment that seems to reply to something not shown.
- Sessions of about an hour keep labels consistent.

## After you finish

Tell Hung. Claude then compares the labels, Hung settles the differences, and H4 is rerun with the extra comments. The report will say that two team members shared the first coder role for these comments.
