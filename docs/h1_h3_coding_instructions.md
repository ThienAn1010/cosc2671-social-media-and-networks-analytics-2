# H1 to H3 Coding Instructions (Kiên and Duy)

## Why we're coding

Our planned H1, H2 and H3 tests were meant to read frames from model labels, and no model passed validation. We'll run all three as smaller exploratory tests on human labels instead, and the report will say the samples are small. Every item needs two independent labels, so the work is:

| Packet | Items | Tests it feeds | Coder A | Coder B | Time per person |
|---|---:|---|---|---|---:|
| Bluesky posts | 220 | H1-B | Kiên (113) or Duy (107) | Claude (done) | ~1.5 h |
| Reddit threads | 280 | Thread check for H1-R, H2, H3 | Kiên (all) | Duy (all) | ~1 h |
| Reddit posts and comments | 780 | H2 (set R1), H3 (set R2), H1-R (set R3) | Kiên (all) | Duy (all) | ~9 h |

Each of you has about 11.5 hours of coding in total. Hung settles the items you disagree on.

Claude can't be the second Reddit coder, because Reddit text can't be shared with an AI service until the teaching team approves it in writing. That's why you both code every Reddit item.

## Order to work in

1. **Reddit threads** first (about 1 hour). They decide which Reddit comments count.
2. **Reddit set R1**, 300 items (about 3.5 hours). This feeds H2.
3. **Reddit set R2**, 288 items (about 3.5 hours). This feeds H3.
4. **Reddit set R3**, 192 items (about 2 hours). This feeds H1-R.
5. **Your half of the Bluesky posts** (about 1.5 hours), at any point.

Each finished set is usable on its own, so if we run short of time, we still have results for the sets you completed. Sessions of about 100 items (an hour or so) keep labels consistent. Take a break when you notice yourself speeding up.

## Getting your files

Hung puts one folder per person in the team Teams folder: `h1h3_coding/kien` and `h1h3_coding/duy`. Download yours. It contains:

| File | What it is |
|---|---|
| `reddit_thread_check.html` | Coding page for the Reddit threads |
| `reddit_frame_bench.html` | Coding page for the Reddit posts and comments |
| `bluesky_frame_bench.html` | Offline coding page for the Bluesky posts |
| `reddit_threads_<you>.csv` | The same threads as a spreadsheet |
| `reddit_docs_<you>.csv` | The same Reddit items as a spreadsheet |
| `bluesky_h1b_<you>.csv` | Your Bluesky half as a spreadsheet |

Each task comes both as a coding page and as a spreadsheet. Use whichever you prefer (option A or B below), but stick to one per packet.

## Option A: coding pages (recommended)

The pages show one item at a time with its reply context, check that every question is answered, and support keyboard shortcuts.

### Reddit pages (offline files)

1. Double-click `reddit_thread_check.html` or `reddit_frame_bench.html`. It opens in your browser (use Chrome, Edge or Firefox). No internet or login is needed.
2. Choose your name and press **Start coding**.
3. Answer the questions and press **Next** (or Enter). Labels save in that browser automatically.
4. To stop, press **Save and pause**, then **Download my labels (CSV)**. Do this at the end of every session, because clearing browser data would delete your labels otherwise. The download is your backup.
5. To continue, open the same file in the same browser. If you switch computers or browsers, open the page, press **Save and pause**, then **Restore from a CSV** and pick your latest download.
6. When every set is done, download the final CSV.

Keep the Reddit pages on your own computer. Don't upload them to Google Drive, email or any AI tool, because they contain Reddit text.

### Bluesky page (online or offline)

- **Online:** Hung shares the Bluesky Frame Bench link from claude.ai (https://claude.ai/artifact/3Eax3BVD98JEAvwVj9mCwS). Open it, choose your name, and code. It saves as you go and syncs across devices, so you can stop and continue anywhere, and Hung can read your labels directly. You'll only see your own half.
- **Offline:** if the link doesn't open for you, use `bluesky_frame_bench.html` exactly like the Reddit pages.

### Keyboard shortcuts

| Question | Keys |
|---|---|
| Relevance: relevant / adjacent / irrelevant | 1 2 3 |
| Language: English / mixed / not English / too short | Q W E R |
| Policy: UK / Australia / other / several / unclear | A S D F G |
| Frames: policy, child safety, privacy, governance, circumvention | Z X C V B |
| No frame | N |
| Next / previous | Enter / ← |
| Thread page: yes / no (moves on by itself) | Y N |

Shortcuts are off while you type in the note box.

## Option B: spreadsheets

1. In Excel, open a blank workbook and use **Data > From Text/CSV**, choosing UTF-8 (65001) if Excel asks for an encoding. Double-clicking the file can garble emoji and accents.
2. Fill in only the label columns: `relevance`, `language`, `target_policy`, `frame_labels` and `notes` (for threads: `thread_relevant` and `notes`).
3. Don't change `item_id`, the text columns, the row order or the column names.
4. Type values exactly as listed below, lower case where shown. Separate several frames with `|` and no spaces, for example `policy_assurance|privacy_surveillance`.
5. If no frame applies, leave `frame_labels` blank and write `no_supported_frame` in `notes`.
6. Save as **CSV UTF-8 (Comma delimited)** with the same file name.

## What to label

These are short versions of the team codebook. The full rules are in `docs/age_gate_manual_annotation_instructions.md`. Our pages skip stance, sentiment and bypass technique, because H1 to H3 don't use them.

**Label the current author only.** The thread opening and the post being replied to are there so you understand the author. Don't use outside knowledge, and don't search for users, links or subreddits.

| Question | Values | Rule of thumb |
|---|---|---|
| `relevance` | `relevant`, `adjacent_contextual`, `irrelevant` | Relevant means the author discusses age checks, age verification, a named age policy or a direct consequence of one. Adjacent means related background (online safety, privacy, speech) that doesn't touch the policy. One keyword isn't enough. |
| `language` | `english`, `mixed`, `non_english`, `too_short_ambiguous` | Use too short only for emoji-only, link-only or garbled text. |
| `target_policy` | `UK_OSA`, `AU_SOCIAL_MINIMUM_AGE`, `OTHER_EXTENDED_EVENT`, `multiple`, `unclear` | Pick the main policy discussed. Use the reply context, not the subreddit or date. If you can't tell, use unclear. |
| `frame_labels` | any of the five frames below, or none | Tick every frame the author clearly uses. |

| Frame | Use when the author argues about |
|---|---|
| `policy_assurance` | What the rule or age check is, how it works, its rollout |
| `child_safety` | Protecting children, harmful content, parents' role |
| `privacy_surveillance` | ID uploads, face scans, data leaks, tracking, anonymity |
| `governance_platform_responsibility` | Who enforces it, fines, regulators, platform or government duty |
| `circumvention_censorship_autonomy` | Getting around it as an argument, censorship, overreach, whether it can work |

A bare VPN mention isn't circumvention unless it's part of an argument about avoiding the rule or whether it can work. Naming the under-16 ban is policy and assurance; add child safety only when protecting children is invoked. Jokes, insults and pure reactions get no frame.

On the pages, choosing Irrelevant fills in Unclear and No frame for you; change them if they're wrong.

**Thread check (`thread_relevant`).** Answer `yes` when the thread's title and opening are about age checks, age verification, the UK Online Safety Act, Australia's under-16 ban or a similar age rule, including news about enforcement. Answer `no` when the thread is about something else and only mentions a keyword, or has no usable text.

## Rules

- **Code on your own.** Don't discuss items with each other until both files are in. Hung settles the disagreements afterwards.
- **Don't change a label because of what the other coder might pick.** Disagreements are expected and useful.
- **Treat the text as private.** Usernames and links are already masked (`@user`, `<URL>`, `<PII>`). Don't try to recover them, and don't paste any text into chats, documents or AI tools.
- **Use notes for real problems:** sarcasm you can't resolve, broken text, or an item that seems to appear twice.
- **Keep to one coding method per packet.** If you start on the page, finish on the page.

## Returning your labels

Put these in your Teams folder under `h1h3_coding/<you>/returned/`:

| Packet | File the page downloads | Spreadsheet option |
|---|---|---|
| Reddit threads | `reddit_threads_<you>_labels.csv` | `reddit_threads_<you>.csv` |
| Reddit posts and comments | `reddit_docs_<you>_labels.csv` | `reddit_docs_<you>.csv` |
| Bluesky posts | `bluesky_h1b_<you>_labels.csv`, or nothing if you used the online page | `bluesky_h1b_<you>.csv` |

Files downloaded from the pages hold only item ids and labels, so they're safe to share inside the team. Spreadsheets you filled in still contain the text, so keep them in the Teams folder only.

Tell Hung in the team chat when each packet is in. You can return the thread file and set R1 early, so Hung can check agreement before you do the rest.

## What happens next

1. Hung runs the agreement check. It reads only ids and labels:
   `python -m src.analysis.h1h3_agreement --packet reddit_docs --coder-a kien.csv --coder-b duy.csv`
   The Bluesky files are each compared with Claude's labels.
2. Hung settles the disagreements. Reddit disagreements are settled offline.
3. The settled labels feed the exploratory H1-R, H1-B, H2 and H3 runs. The report states the sample sizes and treats the results as exploratory.

If agreement on a set is low (kappa below about 0.6 on relevance or on a frame), Hung will ask you both to reread the codebook and recode a small batch together before continuing.

## Questions

- **The text is cut off or empty.** Code what you can see and add a note. Use `too_short_ambiguous` if nothing is readable.
- **The comment replies to something I can't see.** Code from what's shown. Parent context is sometimes missing; that's normal.
- **The same words appear twice.** Code both and add a note; exact duplicates were already removed.
- **I made a mistake on an earlier item.** Use ← on the page, or the id buttons on the pause screen, to go back and fix it.
- **Excel turned `|` separators or accents into something else.** Reopen the CSV through Data > From Text/CSV with UTF-8, or switch to the coding page.

## For Hung: preparing and rebuilding the folders

The samples and pages are rebuilt deterministically from the frozen data:

```bash
python -m src.analysis.h1h3_sample --build     # packets in data/analysis/annotation/h1h3/<coder>/
python -m src.analysis.coding_bench --build    # pages in data/analysis/annotation/h1h3/pages/
```

Copy `pages/` and each coder's folder into that coder's Teams folder. Don't share `data/analysis/annotation/h1h3/coordinator/`: it maps items to accounts, threads and roles, and coders must not see it.
