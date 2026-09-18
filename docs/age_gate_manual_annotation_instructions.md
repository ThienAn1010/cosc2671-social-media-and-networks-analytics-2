# Age-Gate Paradox Manual Annotation Instructions

## Material Passport

- **Origin:** Academic research experiment/study protocol workflow
- **Verification status:** Verified against `age-gate-codebook-v2` and the current validation packet schema
- **Version:** `team_annotation_instructions_v1`
- **Codebook:** `age-gate-codebook-v2`

## TL;DR for coders

Read `text_for_annotation` and the supplied `parent_context`/`root_context`, then edit only these columns:

| Column | Label what | Allowed values |
|---|---|---|
| `relevance` | Relation to the age-assurance topic | `relevant`, `adjacent_contextual`, `irrelevant` |
| `language` | Main language of the text | `english`, `mixed`, `non_english`, `too_short_ambiguous` |
| `target_policy` | Policy being discussed | `UK_OSA`, `AU_SOCIAL_MINIMUM_AGE`, `OTHER_EXTENDED_EVENT`, `multiple`, `unclear` |
| `stance` | Author's position toward the policy | `support`, `oppose`, `mixed_conditional`, `neutral_descriptive`, `unclear_ambiguous` |
| `sentiment` | Emotional/evaluative tone | `positive`, `negative`, `neutral`, `mixed_ambiguous` |
| `frame_labels` | Clearly supported themes/arguments; select all that apply | `policy_assurance`, `child_safety`, `privacy_surveillance`, `governance_platform_responsibility`, `circumvention_censorship_autonomy` |
| `bypass_techniques` | Concrete bypass methods mentioned | `VPN`, `proxy_Tor`, `DNS_change`, `false_borrowed_ID`, `face_spoofing`, `parent_account`, `age_misstatement`, `platform_migration`, `none_unclear` |
| `notes` | Brief ambiguity or data-problem note | Free text; optional |

Separate multiple `frame_labels` or `bypass_techniques` with `|` and no spaces. If no frame is supported, leave `frame_labels` blank and write `no_supported_frame` in `notes`. Label the current author only; do not infer from keywords or outside information. Do not change IDs, text/context, row order, column names, `label_version`, or `annotation_status`. Each item must be labelled independently by two coders.

## 1. Purpose

This task creates human labels for the frozen Age-Gate Paradox validation sample. The labels will be used to validate measurements of:

- relevance;
- language;
- policy target;
- stance;
- sentiment;
- theory-led frames; and
- circumvention/bypass techniques.

This is a validation task, not an exercise in proving the research hypothesis. Label only what the supplied text and context support. Do not make assumptions about what the author intended.

## 2. Independent coding requirement

Every item must be labelled independently by two coders.

Coder A labels only:

- `data/analysis/validation/coder_a_labels_development.csv`
- `data/analysis/validation/coder_a_labels_evaluation.csv`
- `data/analysis/validation/coder_a_labels_reserve.csv`

Coder B labels only:

- `data/analysis/validation/coder_b_labels_development.csv`
- `data/analysis/validation/coder_b_labels_evaluation.csv`
- `data/analysis/validation/coder_b_labels_reserve.csv`

The files contain the same annotation IDs so the two independent decisions can be compared later.

Do not discuss individual items, share decisions, or inspect the other coder's file before submitting your own labels. A third person resolves remaining disagreements after both independent labels have been saved.

If more than two people are available, the coordinator may assign different coder pairs to different items. Each official item must still have two independent labels, and a third person should be used for adjudication rather than ordinary third-pass coding.

## 3. Sample structure

The frozen sample contains:

| Split | Total documents | Per platform | Intended use |
|---|---:|---:|---|
| Development | 600 | 200 | Codebook, feature, model, and threshold development |
| Evaluation | 600 | 200 | One final evaluation after the pipeline is frozen |
| Reserve | 300 | 100 | Contingency coverage only under the predeclared rule |

Do not move items between files, add rows, delete rows, or combine splits. The evaluation split must not be used to choose a model or threshold. The reserve must not be used for ordinary model selection.

## 4. What the coder may use

Each row contains:

- `text_for_annotation`: the current author's post, comment, or reply;
- `parent_context`: the direct parent, when available; and
- `root_context`: the root post, thread, or video context, when available.

Parent and root context may be blank. That is normal.

The coder packet intentionally does not show platform, split metadata, sampling stratum, author ID, sampling weights, or model predictions. Do not request or use the coordinator register.

The text has already been privacy-masked. You may see placeholders such as `@user`, `<URL>`, and `<PII>`. Do not try to recover masked information.

Do not:

- open links or external media;
- search usernames, profiles, communities, or videos;
- infer age, residence, identity, or intent;
- use outside information to decide a label; or
- paste raw posts into online services, public documents, or team chat.

Use the supplied parent/root context only to understand what the current author is referring to. The label must represent the current row's author, not the author of the context.

## 5. Columns to edit

Edit only these columns:

- `relevance`
- `language`
- `target_policy`
- `stance`
- `sentiment`
- `frame_labels`
- `bypass_techniques`
- `notes`

Do not edit:

- `annotation_id`;
- `text_for_annotation`;
- `parent_context`;
- `root_context`;
- `label_version`;
- `annotation_status`;
- column names; or
- row order.

Leave `label_version` as `age-gate-codebook-v2`. Leave `annotation_status` unchanged; the pipeline determines coding status from the completed label fields.

## 6. Recommended coding order

For each row:

1. Read `text_for_annotation`.
2. Read `parent_context` and `root_context`, if present.
3. Label `relevance`.
4. Label `language`.
5. Label `target_policy`.
6. Label `stance`.
7. Label `sentiment`.
8. Label `frame_labels`.
9. Label `bypass_techniques`.
10. Add a short note only when the decision is ambiguous or there is a data problem.

## 7. Relevance

Use exactly one value:

| Value | Use when |
|---|---|
| `relevant` | The author directly discusses age assurance, age-gating, age verification, a named age-policy event, or a direct consequence of that policy. |
| `adjacent_contextual` | The text provides related background about online safety, privacy, speech, platforms, or regulation but does not directly discuss the target policy. |
| `irrelevant` | The text is unrelated, spam, keyword-only, or uses terms such as “age”, “child”, or “VPN” in an unrelated way. |

Do not label a row `relevant` merely because it contains one matching keyword.

## 8. Language

Use exactly one value:

| Value | Use when |
|---|---|
| `english` | Substantive content is primarily English and understandable. |
| `mixed` | Meaningful English and non-English content are both present. |
| `non_english` | Substantive content is primarily not English. |
| `too_short_ambiguous` | There is too little, corrupted, link-only, emoji-only, or otherwise unclear text to determine the language reliably. |

Do not use `too_short_ambiguous` merely because a text is short if its language and meaning are clear.

## 9. Target policy

Use exactly one value:

| Value | Use when |
|---|---|
| `UK_OSA` | The text discusses the UK Online Safety Act or its age-assurance/enforcement requirements. |
| `AU_SOCIAL_MINIMUM_AGE` | The text discusses Australia's social-media minimum-age policy or implementation. |
| `OTHER_EXTENDED_EVENT` | The text clearly identifies another age-assurance event outside those two targets. |
| `multiple` | Two or more target policies are materially discussed. |
| `unclear` | The target cannot be identified from the supplied text and context. |

If one policy is clearly the main subject and another is only incidental, label the main policy. Use `multiple` when both are materially discussed.

Do not infer a target from the platform, date, username, community, or outside knowledge. If the supplied text and context are insufficient, use `unclear`.

## 10. Stance

Stance means the author's position toward the identifiable policy or target. Use exactly one value:

| Value | Use when |
|---|---|
| `support` | The author endorses or approves the policy, mechanism, or argument. |
| `oppose` | The author rejects or criticises the policy, mechanism, or argument. |
| `mixed_conditional` | The author supports one aspect but opposes another, or supports the policy only under conditions. |
| `neutral_descriptive` | The author describes facts, events, or procedures without expressing a position. |
| `unclear_ambiguous` | The author's position cannot be determined. |

Important rules:

- Negative sentiment is not automatically opposition.
- Supporting child safety while opposing ID collection is `mixed_conditional`.
- A question is not automatically support or opposition.
- A VPN or other bypass mention is not automatically opposition.
- A regulation mention is not automatically censorship.

If the target is unclear and the text does not provide a policy position, use `unclear_ambiguous` rather than forcing `support` or `oppose`.

## 11. Sentiment

Sentiment describes emotional or evaluative tone, not policy stance. Use exactly one value:

| Value | Use when |
|---|---|
| `positive` | Approval, optimism, praise, or positive affect is expressed. |
| `negative` | Anger, fear, concern, frustration, criticism, or negative affect is expressed. |
| `neutral` | The wording is factual or non-evaluative. |
| `mixed_ambiguous` | Emotions conflict, sarcasm cannot be resolved, or context is insufficient. |

For example, an author can have negative sentiment about a policy but still support it as necessary. Label sentiment and stance separately.

Sarcasm may only be interpreted when the supplied context makes it clear. Otherwise use `mixed_ambiguous` sentiment and/or `unclear_ambiguous` stance as appropriate.

## 12. Frame labels

Frames are non-exclusive. Select every frame clearly supported by the current author's contribution.

Use these exact labels:

| Label | Definition |
|---|---|
| `policy_assurance` | Discussion of the existence, design, operation, or implementation of age-policy or age-assurance requirements. |
| `child_safety` | Protection of children, harmful-content exposure, parental safeguarding, or harm reduction. |
| `privacy_surveillance` | Identification, tracking, biometric checks, ID exposure, data retention, data breaches, anonymity, or surveillance concerns. |
| `governance_platform_responsibility` | Accountability, enforcement, transparency, platform responsibility, government responsibility, appeals, audits, or institutional competence. |
| `circumvention_censorship_autonomy` | Arguments about bypassing rules, censorship, speech, access, paternalism, exclusion, overreach, autonomy, or technical feasibility. |

Multiple frame labels must be separated with `|` and no spaces. Use the following order:

```text
policy_assurance|child_safety|privacy_surveillance|governance_platform_responsibility|circumvention_censorship_autonomy
```

Only include labels that apply. Do not duplicate labels.

Do not assign a frame from a keyword alone. In particular, a VPN mention is not automatically a `circumvention_censorship_autonomy` frame. It must be part of an argument about avoidance, feasibility, access, autonomy, or overreach.

If no frame is supported, leave `frame_labels` blank and write `no_supported_frame` in `notes`. Do not enter `none_unclear` in `frame_labels`; that value is valid only in `bypass_techniques`.

## 13. Bypass techniques

Use one or more exact values, separated with `|` and no spaces:

- `VPN`
- `proxy_Tor`
- `DNS_change`
- `false_borrowed_ID`
- `face_spoofing`
- `parent_account`
- `age_misstatement`
- `platform_migration`
- `none_unclear`

Use `none_unclear` when no bypass technique is present or the technique cannot be determined.

Only label a technique when the text supports it. Do not infer that someone used a technique merely because they mention privacy, anonymity, or another platform.

A technique is coded separately from the circumvention frame. For example, a supported VPN technique can be labelled under `bypass_techniques` without assigning the circumvention frame if the text does not make an argument about avoidance, autonomy, or access.

## 14. Quoted text and replies

Label the current author's contribution, not another person's quoted speech.

Example:

> Someone says we should upload ID, but that is unacceptable.

Code the current author's opposition and privacy concern. Do not treat the quoted speaker's position as the current author's position.

For a short reply such as “Exactly”, use the parent context to understand what is being endorsed. If the referenced position is clear, label it. If it is not clear, use `unclear_ambiguous` and leave `frame_labels` blank.

Context can resolve pronouns and references, but the final label must represent the current row's author.

## 15. Legacy labels

Do not use old labels such as:

- `privacy`;
- `free_speech`;
- `circumvent`; or
- `id_upload`.

Use the v2 labels instead:

| Old concept | Current v2 treatment |
|---|---|
| Privacy concern | `privacy_surveillance` |
| Free speech, access, or autonomy argument | `circumvention_censorship_autonomy` |
| Concrete bypass method | Appropriate `bypass_techniques` value |
| ID upload as a policy mechanism | `policy_assurance`; add `privacy_surveillance` when privacy, biometric, surveillance, or data risk is discussed |
| Child protection | `child_safety` |

## 16. Difficult or unlabelable rows

Do not delete, skip, rewrite, or replace a difficult row.

Use the available ambiguity values:

| Situation | Label |
|---|---|
| Target cannot be identified | `target_policy=unclear` |
| Stance cannot be identified | `stance=unclear_ambiguous` |
| Sentiment cannot be identified | `sentiment=mixed_ambiguous` |
| Language cannot be identified | `language=too_short_ambiguous` |
| No bypass technique or unclear technique | `bypass_techniques=none_unclear` |
| No supported frame | Leave `frame_labels` blank and note `no_supported_frame` |

Suggested short notes include:

- `missing_context`
- `quoted_only`
- `sarcasm`
- `multiple_targets`
- `no_supported_frame`
- `possible_language_issue`

Notes are optional and should be brief. Do not put raw post text, usernames, or external personal information in `notes`.

If the codebook appears unclear, finish the current item independently and note the issue. Do not silently change the codebook or invent a new label. The coordinator will issue any written clarification to all coders.

## 17. Final quality check before submission

Before returning the file:

- confirm the row count is unchanged;
- confirm every `annotation_id` is present exactly once;
- confirm no text or context column was changed;
- confirm the column names are unchanged;
- confirm `label_version` is still `age-gate-codebook-v2`;
- confirm labels use the exact spelling and capitalization above;
- confirm multi-label values use `|` with no spaces;
- confirm there are no old or invented labels;
- confirm all applicable label fields are completed;
- confirm `frame_labels` is blank only when no frame is supported; and
- confirm `bypass_techniques` uses `none_unclear` when no technique applies.

Save the file using its original filename and UTF-8 CSV format. If using Excel or another spreadsheet application, import the CSV as UTF-8 text so annotation IDs are not converted or reformatted. Do not sort, filter-and-save, or auto-fill in a way that changes row order or values.

Return the completed file privately to the coordinator. Do not paste raw rows into team chat.

## 18. Coordinator handoff

The coordinator must keep the two original coder files unchanged. After both coder files for a split are received, combine them without fabricating adjudication:

```bash
.venv/bin/python -m src.analysis.run validation merge --split development
.venv/bin/python -m src.analysis.run validation merge --split evaluation
.venv/bin/python -m src.analysis.run validation merge --split reserve
```

The merge preserves both independent labels and creates a coordinator artifact. It does not resolve disagreements.

The coordinator then:

1. compares the two coder labels;
2. keeps both original coder decisions;
3. sends remaining disagreements to a third person;
4. records final `adjudicated_*` values separately; and
5. records any exclusions, replacements, and adjudication decisions.

Do not give coders `coordinator_register_*.csv`. It contains hidden platform, sampling, author, and probability metadata.

### Coordinator-only implementation note

The current schema has no explicit no-frame token. Preserve a blank `frame_labels` value when no frame is supported; do not invent a token. The frame model explicitly excludes these no-supported-frame rows and reports both the evaluated denominator and the excluded count.
