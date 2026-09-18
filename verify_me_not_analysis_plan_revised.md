# Verify Me Not: Analysis and Report Plan

This is the analysis protocol for the project proposed in `docs/age_gate_paradox_project_proposal.md`. It treats the assessment specification as the minimum standard and is designed to produce defensible new findings, not a catalogue of methods.

## Material Passport

| Field | Value |
|---|---|
| Artifact | Implementation-ready analysis protocol |
| Authority | `ASSESSMENT_SPEC.md` and `docs/age_gate_paradox_project_proposal.md` |
| Data status | Frozen; cleaning, annotation and derived analysis only |
| Core empirical scope | The proposal's three case windows and H1-H4 |
| Extended scope | E1/E4/E5-specific claims, Google Trends modelling, change points, Granger tests and forecasting; retained but approval-gated |
| Result status | Planned analyses only; no result is asserted by this document |
| Protocol status | Freeze before final confirmatory analysis; disclose any results already inspected |

## Non-negotiable decisions

- **The data is frozen.** No new platform records, Google Trends windows, petition histories, VPN-provider figures, or replacement samples will be scraped. All research questions must be answered with the collected data.
- **Cleaning and annotation may continue.** We may audit relevance, language, missingness, model error and graph construction; derive new variables; and create human-labelled validation data from the frozen corpus.
- **The 20-page limit does not restrict the analysis.** We run every analysis in this plan that passes its validity gate. Page count affects only whether evidence appears in the main report, an allowed appendix, or the reproducible repository supplement.
- **The approved proposal controls the assessed core.** E1, E4, E5, inferential Google Trends work, change-point analysis, Granger tests and forecasting remain in the research programme, but cannot support assessed core claims unless the proposal is formally amended or the lecturer approves the extension. If approval is absent, their code and outputs remain clearly separated exploratory research artifacts.
- **This is a time-stamped analysis protocol, not a preregistration.** Collection and preliminary quality checks already occurred. Any result seen before this protocol is frozen must be disclosed; confirmatory, secondary and exploratory results must remain visibly separated.
- **NetworkX is the analytical backbone.** Use it for graph construction, descriptive measures, PageRank, projections, null models and most robustness checks. Use `python-igraph`/`leidenalg` where Leiden or scale makes them materially better. A degree-corrected stochastic block model is an independent robustness model if it can be installed and reproduced reliably.
- **No method earns space merely by sounding advanced.** Each method must answer a research question, outperform or add something to a transparent baseline, pass its assumptions, and change the interpretation. A failed or null advanced model is reported honestly.
- **The report will not call visibility, centrality, association, temporal ordering or prediction “causal influence.”** The data supports observed interaction, attention, brokerage, association and predictive value—not individual identity, exposure, persuasion or policy effectiveness.

## 1. Research outcome and narrative

### Central research question

How are age-policy/assurance, child-safety, privacy/surveillance, governance/platform-responsibility and circumvention/censorship/autonomy frames distributed, connected and propagated across Reddit, YouTube and Bluesky around the proposal's UK and Australian age-assurance cases?

### The report's narrative spine

The analysis should answer five questions in order:

1. **What did people argue?** Identify frames, stance, sentiment and unanticipated topics.
2. **Where did those arguments concentrate?** Compare platforms, events, communities and video audiences.
3. **Who connected otherwise separated conversations?** Distinguish attention from brokerage and activity.
4. **How and when did arguments travel through observed interactions?** Measure cascades, frame transitions and community timing; add external search association only if scope-approved.
5. **So what should change first?** Convert the strongest replicated findings into one or two priorities with owners, mechanisms, success indicators and guardrails.

The working theoretical proposition is deliberately falsifiable: **age-assurance debate is organised by competing values, while policy design and platform architecture shape whether those values cluster, meet, separate, gain visibility and propagate.** This is a proposition to test, not a conclusion to assume.

### Intended contributions

The final report should aim to contribute all five layers below, even if some yield null results:

1. A validated cross-platform measurement of frames and stance that does not confuse opposition with negative emotion.
2. A platform-aware comparison using equivalent concepts without pretending the platforms have identical data-generating processes.
3. A network account of communities, attention, brokerage and observed propagation rather than a centrality leaderboard.
4. A temporal comparison of proposal-core legislation, enforcement and implementation discourse, with any approval-gated prediction clearly separated from causation.
5. An evidence-to-action synthesis that tells a specified actor what to address first and how success would be observed.

### Assessment alignment

| Course requirement | Direct evidence in this plan |
|---|---|
| Apply data science to social media data | Frozen manifests, audited populations, uncertainty, hypothesis tests and reproducible Python outputs |
| Analyse communities, important nodes and influence propagation | Explicit actor/message/bipartite graphs, Leiden stability, PageRank, brokerage, H1/H3 and bounded cascade analysis |
| Apply NLP for sentiment and topics | Human-validated sentiment, stance, five-frame classification, NMF and BERTopic discovery |
| Synthesize and present insights | Finding cards, cross-platform evidence matrix, claim-led visuals and one-to-two measurable action priorities |

Network analysis remains central: every graph states nodes, edges, direction, weight, filter and inferential meaning, and structural opportunity is never mislabeled as persuasion.

## 2. Research questions, hypotheses and evidential status

### Overarching question and confirmatory hypotheses

SQ1—how frames, stance and sentiment differ across platforms, cases and communities—is the overarching descriptive/comparative question. It is not itself a confirmatory test.

H1-H4 are inherited from the proposal. The six primary endpoint tests below form one family, `PRIMARY-6`, corrected together with Holm's method. All 0-1 effect thresholds are predeclared interpretation anchors, not universal benchmarks; continuous estimates and intervals are reported even when a decision threshold is missed.

| Hypothesis | Platform | `population_id` | Unit | Primary outcome | Estimand | Primary contrast | Null model | Missing-data rule | Correction family | Minimum meaningful effect | Pass/fail decision rule | Verification command |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1-R | Reddit | `R_REPLY_CORE` | Case-specific community; author frame vector is permutation unit | Activity-weighted mean Jensen-Shannon divergence between each community frame profile and its case-wide profile, scaled 0-1 | Observed case-weighted divergence minus median null divergence | Observed fixed case partitions versus shuffled author profiles | Hold each case graph/Leiden partition fixed; shuffle complete author frame vectors within case × predeclared degree × activity bins; merge bins with fewer than 20 authors; 9,999 joint permutations | Authors need at least 3 classifiable documents in a case; excluded authors/documents and coverage are reported | `PRIMARY-6` Holm | Excess divergence ≥ 0.05 | Holm-adjusted p < .05 and excess divergence ≥ 0.05 | `python -m src.analysis.run hypothesis --id H1 --platform reddit --check` |
| H1-B | Bluesky | `B_REPLY_CORE` | E2/E3-specific community; author frame vector is permutation unit | Same case/event-weighted outcome as H1-R | Same as H1-R | Same as H1-R | Hold each event graph/partition fixed; shuffle whole author vectors within event × degree × activity bins; merge bins <20; 9,999 joint permutations | Same ≥3-document-within-event rule; only observed descendants of phrase-exact roots; `day` is the time axis | `PRIMARY-6` Holm | Excess divergence ≥ 0.05 | Same rule as H1-R; cross-platform recurrence requires both H1-R and H1-B to pass | `python -m src.analysis.run hypothesis --id H1 --platform bluesky --check` |
| H2-P | Reddit | `R_AU_LEGISLATION` + `R_AU_IMPLEMENTATION` | Document nested in author and thread | Author-balanced prevalence of `privacy_surveillance` | Implementation minus legislation risk difference | AU implementation window versus AU legislation window | H0: risk difference ≤ 0; inference uses 9,999 thread-cluster bootstrap draws with author-balanced weights, not exchangeability-based event relabelling | Unusable text excluded with counts; uncertain-language bounds and audited-thread population are mandatory sensitivities | `PRIMARY-6` Holm | +0.05 absolute share | Holm-adjusted one-sided p < .05, lower 95% interval > 0 and difference ≥ .05 | `python -m src.analysis.run hypothesis --id H2 --outcome privacy --check` |
| H2-C | Reddit | `R_AU_LEGISLATION` + `R_AU_IMPLEMENTATION` | Document nested in author and thread | Author-balanced prevalence of proposal-aligned `circumvention_censorship_autonomy` | Implementation minus legislation risk difference | Same as H2-P | Same as H2-P | Same as H2-P | `PRIMARY-6` Holm | +0.05 absolute share | Same as H2-P; H2 as a whole is supported only if H2-P and H2-C both pass | `python -m src.analysis.run hypothesis --id H2 --outcome circumvention_autonomy --check` |
| H3 | Reddit | `R_BROKER_CORE` | Author | Normalized Shannon entropy of the author's five-frame profile | Activity/degree-matched mean entropy difference: top-decile betweenness minus below-median betweenness | High-betweenness brokers versus matched non-brokers | Match 1:3 within case on log document count, degree and subreddit breadth; permute broker label within matched sets 9,999 times | Require ≥5 classifiable documents and membership in an analysed component; report unmatched/excluded authors | `PRIMARY-6` Holm | +0.10 normalized entropy | Holm-adjusted p < .05, lower 95% interval > 0 and difference ≥ .10 | `python -m src.analysis.run hypothesis --id H3 --platform reddit --check` |
| H4 | YouTube | `Y_VIDEO_E2E3` | Selected video | Frame alignment = 1 − Jensen-Shannon divergence between adjudicated video-metadata profile and contributor-balanced audience profile | Event-adjusted news-minus-commentary mean alignment difference | `news` versus `commentary` within E2/E3 selected videos | H0: adjusted difference = 0; permute channel-level source labels within event and use channel-cluster bootstrap, 9,999 replicates | Require adjudicated metadata and ≥20 strict-eligible comments; exclusions reported; comment-weighted audience is sensitivity | `PRIMARY-6` Holm | Absolute difference ≥ 0.05 | Holm-adjusted two-sided p < .05, interval excludes 0 and absolute difference ≥ .05 | `python -m src.analysis.run hypothesis --id H4 --check` |

H1 is reported separately for Reddit and Bluesky; it is called cross-platform recurrent only if both pass. H2 is explicitly Reddit-only. H3 is confirmatory on Reddit's complete observed reply trees, with Bluesky and defensible YouTube results as secondary replications. H4 is a bounded comparison among purposively selected E2/E3 videos, never a population-wide YouTube effect.

Before joining final labels, run a simulation/precision gate using the frozen cluster sizes, graph topology, source-type counts and the thresholds above. Target at least 80% power under the declared minimum effect and record the minimum detectable effect. If a test is underpowered—especially H4 after channel clustering—the estimate and interval are still reported, but the result is labelled inconclusive rather than “no difference.” The power script may diagnose feasibility; it cannot change populations or thresholds after outcomes are inspected.

### Secondary and exploratory questions

| ID | Question | Status |
|---|---|---|
| SQ2 | Do discourse shifts and Google Trends move together around E2, E3 and E5? | Extended, associational and approval-gated |
| SQ3 | Are PageRank attention, brokerage and non-redundant spreader roles held by the same accounts? | Secondary; expected disagreement is itself a finding |
| SQ4 | Do particular frames or bypass techniques precede larger or more structurally viral observed reply cascades? | Secondary; no persuasion claim |
| SQ5 | Does social discourse improve out-of-sample prediction of VPN search interest beyond lag-only baselines? | Extended predictive test; approval-gated |
| SQ6 | What stable topics fall outside the theory-led frame codebook? | Exploratory discovery |
| SQ7 | Do community roles and frame mixtures persist or reorganise across event phases? | Exploratory temporal-network analysis |

The report must label every result as confirmatory, secondary or exploratory. Secondary and exploratory tests use separate, named false-discovery-rate families by workstream; they never alter the `PRIMARY-6` decision rules after results are seen.

## 3. Frozen data, events and analytical populations

### 3.1 Frozen-data registry

Before modelling, create a machine-readable population manifest containing file paths, row counts, date ranges, schema versions, checksums and exclusion counts. The current snapshot to verify against the files is:

| Source | Current frozen snapshot | Role |
|---|---:|---|
| Bluesky | 423,628 processed posts; 372,674 currently marked `in_corpus` | Search prevalence, conversation networks and event dynamics |
| YouTube | 134 videos; 93,594 comments/replies | Video-audience alignment, commenter-video network and reply network |
| Reddit | 1,764 posts; 129,940 comments; 2,034/2,034 tasks complete | Thread discussion, subreddit comparison, reply and author-subreddit networks |
| Google Trends | Daily data, 2025-01-01 to 2026-06-30; four countries and three search terms | Frozen contextual signal; inferential/predictive use is approval-gated |

These counts are checkpoints, not numbers to copy manually into the report. Every final number must be generated from the frozen manifest and analysis scripts.

### 3.2 Scope and event registries

Two calendars answer different questions and must not be silently substituted for each other.

#### Proposal case windows: confirmatory authority

Create `config/case_windows.csv` from the frozen Reddit manifest and proposal. Its half-open ranges define the assessed core:

| `case_window_id` | UTC interval | Core role |
|---|---|---|
| `AU_LEGISLATION` | 2024-11-10 to 2025-01-11 | H2 formation/legislation comparison |
| `UK_ENFORCEMENT_CLUSTER` | 2025-06-14 to 2025-08-26 | UK policy discussion centred on E2, 2025-07-25 |
| `AU_IMPLEMENTATION` | 2025-11-10 to 2026-01-11 | Australian implementation discussion centred on E3, 2025-12-10 |

`src/shared/events.py` currently loads only E1-E5 and therefore cannot assign the Australian legislation case. Add a separate case-window loader rather than forcing the Reddit collector labels into E1-E5.

#### Point-event calendar: contextual and approval-gated extensions

`config/events.csv` remains the single authority for exact point dates and the common [day −7, day +21) descriptive windows:

| Event | Date | Status |
|---|---|---|
| E1 | 2025-06-27 | Context only; Texas/adult-content event is outside proposal core |
| E2 | 2025-07-25 | Core UK anchor inside `UK_ENFORCEMENT_CLUSTER` |
| E3 | 2025-12-10 | Core AU anchor inside `AU_IMPLEMENTATION` |
| E4 | 2026-01-21 | Extended legislative comparison requiring scope approval |
| E5 | 2026-03-09 | Extended Australian adult-content-code comparison requiring scope approval |

E5's authoritative date is 2026-03-09. The old Bluesky 2026-03-11 placeholder is superseded by `config/events.csv`, `src/platforms/bluesky/config.py` and its test. Collection remains valid because it was continuous, but any derived artifact using 2026-03-11 must be regenerated or marked stale.

#### Platform-event availability

| Window/event | Reddit | YouTube | Bluesky | Google Trends | Permitted claim |
|---|---|---|---|---|---|
| `AU_LEGISLATION` | Yes | No | Incomplete tail only; no H2 use | Yes | Reddit-only H2 baseline |
| `UK_ENFORCEMENT_CLUSTER` / E2 | Yes | Yes | Yes | Yes | Core cross-platform comparison |
| `AU_IMPLEMENTATION` / E3 | Yes | Yes | Yes | Yes | Core cross-platform comparison and Reddit H2 endpoint |
| E1 | Falls inside Reddit UK collection but is not a separate case | Yes | Yes | Yes | Context/negative-control only; approval-gated if analysed separately |
| E4 | No | Yes | Yes | Yes | Extended only |
| E5 | No | Seven selected videos | Yes | Yes | Extended only; YouTube descriptive |

No absent platform-event cell is imputed, borrowed or described as a null result.

### 3.3 Platform-specific analytical populations

A single broad eligibility flag is not a valid universal population. Freeze a population registry with one named population per research purpose.

| Platform | Prevalence/NLP population | Network population | Required sensitivity |
|---|---|---|---|
| Bluesky | Phrase-exact search posts for prevalence; relevant replies/descendants for conversation language | Replies descending from phrase-exact relevant roots; replies are incomplete beyond collected depth | Non-exact search hits for recall; search-only versus search-plus-replies; remove memes/repeaters |
| YouTube | Strict eligible English comments/replies; video titles/descriptions labelled separately | Commenter-video bipartite graph is primary; observed reply graph is secondary | Inclusive English/uncertain-language population; handle-match-only replies; exclude channel-owner replies |
| Reddit | Substantive human comments in manually accepted relevant threads | All valid human reply edges inside accepted threads; author-subreddit bipartite graph for cross-subreddit structure | Existing keyword screens; post-only versus comment-inclusive; leave-one-subreddit-out |

Freeze these hypothesis populations in the manifest:

| `population_id` | Exact definition |
|---|---|
| `R_AU_LEGISLATION` | Substantive English human documents inside audited-relevant Reddit threads in `AU_LEGISLATION` |
| `R_AU_IMPLEMENTATION` | Same rules inside `AU_IMPLEMENTATION` |
| `R_REPLY_CORE` | Valid author-to-author reply edges inside audited-relevant threads across the three proposal case windows; H1 authors require ≥3 classifiable documents |
| `B_REPLY_CORE` | Valid observed replies descending from phrase-exact relevant roots inside E2/E3 core windows; spam/repeater endpoints removed; H1 authors require ≥3 classifiable documents |
| `R_BROKER_CORE` | `R_REPLY_CORE` authors with ≥5 classifiable documents in an analysed connected component |
| `Y_VIDEO_E2E3` | Purposively selected E2/E3 videos with adjudicated metadata and ≥20 strict-eligible comments; audience profiles are contributor-balanced |

The manifest stores the executable Boolean rule, source checksum, row/node count and exclusion counts for each ID. Hypothesis code accepts a `population_id`; it may not rebuild eligibility ad hoc.

#### Reddit relevance gate

The broad existing filter cannot be the main population because many records currently marked eligible are unrelated and keyword masks contain the constructs being measured. Audit root posts, freeze `thread_relevant`, record two-coder decisions on borderline roots, and propagate acceptance to descendants. Existing `age_assurance_related`, topic and sentiment masks become sensitivity specifications, not ground truth. Retain the 2024 Australian legislation window for H2.

#### Bluesky relevance gate

`in_corpus` is primarily a quality flag, not proof of substantive relevance. Phrase-exact search posts define the prevalence population. Conversation analyses use their observed descendants. Non-exact search matches are a recall sensitivity, not part of the headline prevalence estimate. Do not construct quote/repost diffusion because the frozen data does not support it adequately.

#### YouTube validity gate

Complete the pending manual UI/sample audit before treating comment or interaction completeness as verified. Treat the seven E5 videos as descriptive and approval-gated, with leave-one-video-out uncertainty if analysed. YouTube reaction time uses the video's assigned event and `yt_days_from_video_event`, not the nearest event to each late comment.

The screening workflow permits `news`, `commentary`, `tech_explainer`, `official` and `other`, but the frozen selected videos contain only the first three. Audit ambiguous classifications blind to audience results and retain the original screened label plus any adjudicated revision. No absent source type enters H4; `tech_explainer` is descriptive because it is sparse and absent in some core events.

### 3.4 Language uncertainty as a measured threat

Do not silently drop “language uncertain” records. The unified human-validation sample must include a dedicated uncertain-language stratum and label each item as English, mixed, non-English or too short/ambiguous. Report:

- uncertain-language prevalence by platform and text length;
- human-estimated English share within the uncertain bucket;
- strict-English, inclusive and human-calibrated/bounded estimates;
- whether any headline result changes sign, rank or substantive conclusion.

If it changes, language selection becomes a finding and a central limitation rather than a footnote.

## 4. Shared measurement and inference protocol

### 4.1 Unified construct codebook and migration

The confirmatory taxonomy follows the proposal's five non-exclusive frame families. Richer subcodes preserve analytical precision without silently changing H1-H4:

| Confirmatory frame | Definition | Analytical subcodes/tags |
|---|---|---|
| `policy_assurance` | The existence, design or operation of age-policy and assurance requirements | Policy/event named; assurance mechanism; implementation stage |
| `child_safety` | Protection of children, exposure prevention, parental safeguarding or harm reduction | Harm type; protection rationale; parental responsibility |
| `privacy_surveillance` | Identification, tracking, biometric/ID exposure, data retention, breach or surveillance concern | Data minimisation; anonymity; biometric/ID concern |
| `governance_platform_responsibility` | Accountability, enforcement design, transparency, platform duty or institutional competence | Government/platform/assurance-provider responsibility; appeal and audit |
| `circumvention_censorship_autonomy` | Argumentative claims about bypass, speech, access, paternalism, exclusion, overreach or autonomy | `censorship_autonomy`, `technical_efficacy_futility`, and argumentative circumvention |

Literal bypass techniques and policy features are coded separately. A VPN mention is not automatically a circumvention frame; it must be used as part of an argument about avoidance, feasibility or autonomy. Techniques include VPN, proxy/Tor, DNS change, false/borrowed ID, face spoofing, parent account, age misstatement and platform migration. Policy features include face/age estimation, ID upload, third-party assurance, data retention, VPN gating, parental consent, appeals and enforcement.

#### Frozen legacy-to-v2 crosswalk

| Existing Bluesky construct | v2 destination | Action |
|---|---|---|
| `privacy` | `privacy_surveillance` | Candidate seed only; revalidate against v2 labels |
| `free_speech` | `circumvention_censorship_autonomy` → `censorship_autonomy` | Candidate seed only; revalidate because legacy reliability was low |
| `circumvent` | Bypass-technique tag; sometimes argumentative circumvention | Never equate automatically with `technical_efficacy_futility` or the umbrella frame |
| `id_upload` | Policy-feature tag; may co-occur with privacy | Remove as a frame; do not reuse its legacy score as v2 validity |
| `child_safety` | `child_safety` | Revalidate because legacy recall was weak |
| No legacy equivalent | `policy_assurance`, `governance_platform_responsibility`, `technical_efficacy_futility` | Fresh annotation and validation required |

Existing Bluesky scores remain evidence about the legacy lexicons only. Every v2 confirmatory construct is evaluated on the shared untouched v2 sample.

#### Annotation context and target

Every coding item carries a fixed context packet without model output:

- standalone post: document text, proposal case/event label and no parent;
- Reddit reply: root-post title plus direct-parent text plus reply text;
- Bluesky reply: phrase-exact root plus direct-parent text plus reply text, using observed context only;
- YouTube comment: video title/description plus comment; a reply also includes its resolved parent when available.

Coders label the author's contribution, not quoted speech. Sarcasm is coded only when the supplied context supports it; otherwise it is ambiguous. Each item first receives `relevant`, `adjacent/contextual` or `irrelevant`, then one `target_policy` (`UK_OSA`, `AU_SOCIAL_MINIMUM_AGE`, `OTHER_EXTENDED_EVENT`, `multiple`, or `unclear`). Primary stance inference excludes `multiple`/`unclear` targets but reports them.

Stance toward the labelled target is `support`, `oppose`, `mixed/conditional`, `neutral/descriptive`, or `unclear/ambiguous`. Support for child safety alongside opposition to ID collection is `mixed/conditional`, not forced into support or oppose. A bypass mention is not automatically technical futility; negative sentiment is not automatically opposition; a regulation mention is not automatically censorship.

### 4.2 One strong human-validation programme

Use one shared sample to validate relevance, language, sentiment, stance and frames instead of several small disconnected samples. Target **1,500 usable, independently double-coded documents: 500 per platform**. Draw deterministic replacements from the frozen corpus until the target is met; report attempted, unusable, replaced and final counts.

#### Split allocation

| Split | Per platform | Total | Permitted use |
|---|---:|---:|---|
| Development | 200 | 600 | Codebook revision, feature/model selection, threshold selection and calibration |
| Untouched evaluation | 200 | 600 | One final evaluation of the frozen v2 pipeline only |
| Sealed contingency reserve | 100 | 300 | Open only by a predeclared coverage/failure rule; never use for ordinary winner selection |
| **Total usable** | **500** | **1,500** | All items double-coded |

Within each platform's development and evaluation split, assign candidates to exactly one stratum in this precedence order:

| Stratum | Development n | Evaluation n | Definition |
|---|---:|---:|---|
| `L_LANGUAGE` | 30 | 30 | Language detector uncertain; not already unusable |
| `S_SHORT_CONTEXT` | 30 | 30 | Short, reply-dependent, quoted or sarcasm-prone; not `L_LANGUAGE` |
| `R_RARE_DISAGREE` | 60 | 60 | Rare predicted frame/stance or disagreement between frozen candidate instruments; not earlier strata |
| `G_GENERAL` | 80 | 80 | Probability sample of all remaining eligible candidates |

The 100-item reserve per platform uses 15/15/30/40 across the same four strata. Within each stratum, allocate proportionally across that platform's available proposal-core windows, with a floor of 10 where enough candidates exist. Any unfillable floor transfers by a written deterministic order before coding.

Assign Reddit threads, YouTube videos and Bluesky root threads to development/evaluation/reserve before document sampling so no context container crosses splits. Exact and near-duplicate texts may appear in only one split. The sampling program then uses a frozen seed and a documented two-stage draw: retain at most two documents per author per split, then at most ten documents per context container per split. Each stage samples by a frozen random key and records its conditional probability; their product is the inclusion probability. If an item is unusable, replace it with the next frozen-key candidate from the same platform/split/stratum/case; if that cell is exhausted, transfer using the predeclared case-floor rule and log the move. Quantify author overlap across splits and add unseen-author evaluation metrics. A fixture test must reproduce split assignment, selected IDs, replacements and weights.

#### Coding and evaluation rules

1. Two coders label every item independently while blind to model outputs and split membership. Adjudication occurs only after independent labels are saved; a third person resolves remaining disagreements.
2. Report raw agreement, Cohen's kappa for single-label tasks, per-frame agreement for multilabel coding, and Krippendorff's alpha where appropriate.
3. Report per-platform and per-class precision, recall and F1 with cluster-bootstrap intervals; also report design-weighted performance using inclusion probabilities.
4. Target at least 40 evaluation positives and require at least 30 positives and 30 negatives per frame/platform. Require at least 30 evaluation examples per sentiment/stance class/platform where that class exists.
5. The reserve may top up a deficient cell according to the frozen stratum order before final metrics are computed. If the minimum still fails, no automated full-corpus claim is made for that cell; use the human-coded estimate or narrow the claim.
6. All codebook, model, feature and threshold choices are made using development data. Freeze one pipeline per platform before opening evaluation results.
7. The untouched evaluation set is used once. Do not switch to the runner-up after seeing evaluation performance.
8. If evaluation reveals a necessary codebook or pipeline revision, version it as v3 and evaluate it on a new untouched sample from unused frozen-corpus records; never recycle the failed evaluation set as proof of the revision.

#### Frozen validity decisions

- Development selection maximizes the task's primary macro-F1. If candidates differ by less than 0.02, choose the more transparent/lower-complexity instrument.
- A codebook construct is usable for automated inference only when independent-coder raw agreement is at least 0.80 and kappa/alpha is at least 0.60. Report prevalence-sensitive disagreement even when this gate passes.
- A frozen automated instrument may support full-corpus primary claims on a platform only when untouched macro-F1 is at least 0.70, its lower cluster-bootstrap 95% bound is at least 0.60, and every claim-bearing class/frame has precision and recall of at least 0.60.
- A failed cell falls back to weighted human-sample inference or a narrower descriptive claim. It is never repaired by selecting another model on evaluation data.
- These thresholds are validity gates, not claims that 0.70 is universally “good.” Publish the complete per-class metrics and uncertainty.

This programme directly addresses the lecturer's concerns about small usable validation sets, uncertain-language selection and keyword completeness while preserving honest fallbacks.

### 4.3 Statistical hierarchy

- Report effect sizes and uncertainty before p-values.
- Cluster resampling or standard errors at the dependency unit: Reddit thread, YouTube video, and Bluesky author/thread as appropriate.
- Show both document-weighted and author-balanced estimates. A highly active account must not silently represent many people.
- Use Holm correction across the six `PRIMARY-6` endpoint tests and false-discovery-rate control within separately named secondary/exploratory workstream families.
- Use degree- and activity-preserving permutations for network hypotheses.
- Never pool raw platform counts as though collection coverage were equal. Compare standardized within-platform effects and then synthesize direction, magnitude and uncertainty.
- Keep all exclusions, transformations, seeds and model revisions in result metadata.

### 4.4 Privacy, ethics and reproducibility

- Create and test one shared `mask_text()` transformation, then apply it before any model or analyst-facing export. Mask URLs, emails and phone numbers consistently.
- Keep Reddit text local until written teaching-team approval permits otherwise. Agents may inspect schemas, code and aggregates but not restricted rows.
- Never join identities across platforms or imply that pseudonymous accounts are people.
- Use paraphrased, non-searchable examples in the report; no handles or long verbatim text.
- Pin environments, seeds, model revisions and codebook versions.
- Test population assignment, masking, graph direction, event assignment, label joins and temporal leakage.
- Generate every reported table and figure from scripts; no hand-edited result values.

## 5. Analysis workstreams

### 5.0 Population, validity and measurement gate — An and Hung

This gate completes before confirmatory modelling.

1. Build the frozen-data and population manifests.
2. Audit Reddit root relevance and Bluesky search/reply relevance; freeze named populations.
3. Complete YouTube collection-quality checks already possible from the frozen data.
4. Run missingness, duplication, author concentration and language-uncertainty audits.
5. Freeze the codebook and double-coded sample protocol.
6. Publish a compact validity dashboard: candidate records, analytical records, exclusions, uncertainty, usable labels and known coverage gaps.

**Output:** one authoritative population table used by every workstream. No owner may silently redefine eligibility.

### 5.1 Coverage and exploratory structure — An

1. Produce platform-by-event counts, unique authors, date ranges, document types and exclusions.
2. Plot volume with collection gaps shown as gaps, never zeroes.
3. Describe jurisdiction only through query/video/subreddit context; do not infer user location.
4. Quantify concentration: top-author share, top-thread/video share and effective number of contributors.
5. Compare raw and normalized event prominence within each platform.
6. Decompose volume change into more active contributors versus more contributions per active contributor.
7. Use weekly aggregation where daily support is inadequate; label thin AU evidence and keep all E5 results approval-gated.

**Finding test:** does an apparent event surge represent broad participation or concentrated activity by a small number of accounts/videos?

### 5.2 Sentiment and stance — An, with shared coding

#### Sentiment instruments

Run three interpretable competitors on the same masked input where data policy permits:

- VADER as the transparent social-text baseline;
- pinned `twitter-roberta-base-sentiment-latest` as the transformer baseline;
- a pinned, documented LLM classifier as an optional comparator, never as ground truth.

Compare instruments only on development data, using grouped cross-validation by thread/video/author as appropriate. Freeze the primary instrument, preprocessing and thresholds per platform before opening the untouched evaluation results. Evaluation measures the frozen choice; it never selects a different winner. If evaluation quality is inadequate, restrict full-corpus inference or start a separately versioned pipeline with a new untouched evaluation sample.

#### Stance models

Sentiment and stance answer different questions. Compare:

- TF-IDF logistic regression as an interpretable supervised baseline;
- a pinned zero-shot NLI transformer;
- a fine-tuned compact transformer only if the labelled sample and class coverage support it;
- optional LLM classification where privacy rules permit.

Use grouped cross-validation for all development choices and the untouched set once for final evaluation. Report macro-F1, per-class recall, calibration and confusion matrices by platform. Primary stance results come from the development-selected model only when its untouched performance passes the frozen validity rule; otherwise use human-coded estimates and narrow the claim.

#### Analyses

1. Estimate sentiment and stance by platform, event and phase with clustered bootstrap intervals.
2. Cross-tab sentiment and stance to expose cases such as negative language supporting stricter policy or positive language opposing it sarcastically.
3. Compare document-weighted with author-balanced trajectories.
4. Test whether sentiment adds information after frame and stance are known; do not let sentiment dominate the substantive interpretation.
5. Run strict/inclusive language and search-only/reply-inclusive sensitivities.

**Potential finding:** “sentiment is not stance” becomes a headline only if the mismatch is large, validated and consequential for interpreting policy support.

### 5.3 Theory-led frames and data-led topics — Hung

#### Frame measurement

1. Implement the five-frame codebook with transparent lexicons as the auditable baseline.
2. Train/evaluate a multilabel supervised classifier using the shared human labels.
3. Compare lexicon, classifier and optional LLM labels on the untouched set.
4. Choose and freeze the primary instrument per frame/platform using development data only. Preserve lexicon estimates as sensitivity results; use untouched evaluation solely to quantify the frozen instrument's performance.
5. Estimate frame prevalence and co-occurrence with inclusion-probability correction and author-balanced versions.
6. Analyse bypass techniques and policy features separately from frames.
7. For privacy-feature association, remove overlapping feature terms from the privacy instrument and report the sensitivity of odds ratios.

#### Topic discovery

Use two complementary models:

- NMF as the sparse, interpretable baseline;
- BERTopic as semantic discovery.

Fit a stable topic space **per platform across events**, then estimate topic prevalence over time. Do not fit unrelated event-specific topic spaces for the main comparison. Event-specific fits may be exploratory only.

For both models:

- pin preprocessing, representation and model revisions;
- run multiple seeds and report stability/coherence rather than choosing the prettiest run;
- inspect representative documents under the data-handling rules;
- have humans name topics and record the naming rationale;
- label topics that map to no theory-led frame as unanticipated findings;
- report topic prevalence with uncertainty and minimum-support rules.

**Finding test:** do semantic topics reveal a stable concern absent from the proposal's frame taxonomy, or do they merely restate the seeded lexicon?

### 5.4 Network structure, communities and roles — Kien

#### Explicit graph definitions

Create a graph registry stating nodes, edge meaning, direction, weight, time assignment, exclusions and inferential scope.

| Graph | Definition | Main use and limit |
|---|---|---|
| Actor attention graph | `replier -> replied-to author`, weight = valid reply count | In-strength and PageRank as received attention/prestige; not exposure or persuasion |
| Actor reply-cascade orientation | `replied-to author -> replier`, weight = valid reply count | Directional structural proxy for potential conversational flow; never called observed exposure |
| Message reply tree | Document nodes with observed parent → child edges and child timestamp/`day` | Cascade size, depth, breadth, duration, structural virality and frame transitions |
| YouTube commenter-video bipartite graph | strict-eligible commenter ↔ selected core video, weight = eligible comment count | Audience overlap and H4 alignment; inclusive-language/all-status graph is sensitivity only |
| Reddit author-subreddit bipartite graph | author ↔ subreddit, weight = accepted core activity | Cross-subreddit bridges that reply edges cannot capture |

Reddit and Bluesky message trees use observed parent links. Bluesky temporal order always uses processed `day`, never client-supplied `created_at`. YouTube's commenter-video graph is primary because it captures the platform's creator/audience structure. Its actor reply graph is secondary and inferential analyses use only strict-eligible source documents with `target_resolution == "handle_match"`, valid endpoints and no self-loop. Root-fallback edges are reported separately as a structural sensitivity, not silently mixed into the primary graph.

#### Structure and communities

1. Report nodes, edges, components, density, reciprocity, strength/degree distributions and concentration by platform/event.
2. Use Leiden as primary community detection and Louvain as a cross-check on the appropriate undirected weighted view.
3. For the primary partition, evaluate Leiden resolution γ in {0.50, 0.75, 1.00, 1.25, 1.50, 2.00} across 20 fixed seeds. Eligible resolutions require median pairwise ARI ≥0.80, at least three communities, and no single community above 80% of analysed nodes. Select the eligible resolution with highest mean modularity; break ties within 0.01 toward the lower resolution. If none qualifies, H1 for that platform is inconclusive. Report NMI/ARI, modularity and community-size distribution for the chosen and neighbouring resolutions.
4. Compare modularity with degree-preserving null networks; raw modularity alone is not evidence of meaningful polarisation. H1 uses its separate frozen author-vector null below rather than substituting graph rewiring.
5. If reproducible, fit a degree-corrected stochastic block model as an independent model-based robustness check. Do not force it if installation, convergence or comparability fails.
6. Track communities across event periods by member overlap with a declared matching threshold; label splits, merges, births and deaths rather than pretending labels persist automatically.

#### Attention, brokerage and structural roles

Compute and interpret distinct roles rather than naming all of them “influence”:

- weighted in/out-strength and PageRank for received attention;
- personalized PageRank seeded by each validated frame for frame-specific attention pathways;
- betweenness and participation coefficient for cross-community brokerage;
- within-module degree z-score for local hubs;
- k-core/coreness for embeddedness;
- VoteRank on the reply-cascade-oriented actor graph as a non-redundant structural coverage set, not evidence of actual spreading;
- bipartite degree and a tested co-ranking measure such as BiRank/HITS for YouTube videos and commenters.

Test PageRank under damping factors 0.75, 0.85 and 0.95, binary versus weighted edges, bootstrap resamples and hub removal. Report rank correlations and top-k stability. If rankings are unstable, instability is the result.

Create an actor-role map using within-module z-score and participation coefficient, then compare it with PageRank and VoteRank. The primary insight should be whether attention, local authority, brokerage and non-redundant structural coverage coincide or separate.

#### H1 frame segregation

1. Freeze the topology-only Leiden partition and its resolution before joining frame labels.
2. Aggregate each eligible author's complete five-frame vector using the frozen primary instrument and minimum-document rule.
3. Use activity-weighted mean community-to-platform Jensen-Shannon divergence as the sole `PRIMARY-6` segregation statistic.
4. Hold each case/event graph and partition fixed; shuffle whole author vectors within case/event × predeclared degree × activity bins for 9,999 joint permutations. Merge adjacent bins containing fewer than 20 authors before any frame result is inspected.
5. Report dominant-frame assortativity, edge-level profile similarity, community entropy and alternative partitions as secondary diagnostics only.
6. Repeat under author-balanced labels, uncertain-label bounds and decisive population specifications.
7. Interpret platform differences through observable network affordances, not universal user behaviour.

#### H3 broker diversity

1. Define the confirmatory Reddit broker group before joining frames: top-decile normalized betweenness within case among `R_BROKER_CORE`; comparison authors are below the case median.
2. Standardize covariates within case and match 1:3 without replacement using Mahalanobis distance, with 0.20-SD calipers on log document count and log degree and an absolute subreddit-breadth difference ≤1. Require post-match standardized mean differences below 0.10; otherwise H3 is inconclusive. Report balance and unmatched authors.
3. Use normalized Shannon entropy of the five-frame author profile as the primary outcome. Effective number of frames, participation coefficient and mixed/conditional stance are secondary.
4. Permute broker labels within matched sets 9,999 times and bootstrap matched sets for uncertainty.
5. Repeat on Bluesky and defensible YouTube structures as secondary replication, never as a substitute for the frozen Reddit test.
6. Report alternative broker thresholds and whether a few high-volume accounts drive the result.

#### Observed propagation and cascades

1. Construct document-level message trees before author aggregation. Measure cascade size, maximum depth, breadth, duration and structural virality only on Reddit and Bluesky trees with observed parent-child links.
2. Compare cascade structure by source frame, stance and event with thread/video-clustered uncertainty.
3. Estimate parent-to-reply frame and stance transition matrices against shuffled-label nulls.
4. Trace bypass techniques by first sustained community uptake, not simply first mention.
5. Treat YouTube thread size and observed top-level/reply structure as descriptive; its heuristic reply-to-reply resolution cannot support primary structural-virality claims.
6. Use a relational event model only where timestamps, candidate risk sets and edge coverage are defensible. Model the rate of observed replies as a function of prior interaction, community relation, role and frame; otherwise stop at descriptive event transitions.
7. Describe temporal order as diffusion only when an observed path and timing support it. Reddit cross-subreddit timing is not a reply-path diffusion claim.

### 5.5 Video-audience alignment — Kien and Hung

This is the explicit YouTube test required by H4. Its inferential population is the purposively selected E2/E3 videos, so every conclusion begins “among the selected videos.” The primary source-type contrast is `news` versus `commentary`. The frozen seed checkpoint is E2: 10 news and 34 commentary videos; E3: 35 news and 15 commentary videos. Reproduce those counts before exclusions. `tech_explainer`, E1, E4 and the seven-video E5 arm are descriptive or approval-gated only.

1. Outside the 1,500-document validation sample, have two coders independently label all 134 video title/description packets for frames and stance and audit their frozen source type; adjudicate before audience results are opened.
2. Estimate each video's audience frame/stance distribution from validated comment labels with commenter-balanced and comment-weighted versions.
3. Measure metadata-audience alignment as 1 − Jensen-Shannon divergence, with contributor-balanced audience profiles primary and comment-weighted profiles as sensitivity.
4. Estimate the event-adjusted news-minus-commentary difference using E2/E3 videos, channel-cluster bootstrap uncertainty and the frozen `PRIMARY-6` rule. Partial pooling may be exploratory but cannot create replication or rescue sparse cells.
5. Map alignment onto the commenter-video bipartite network to test whether shared audiences connect similarly framed or conflicting videos.
6. Run leave-one-video-out and leave-one-channel-out checks. E5 remains a seven-video descriptive case and cannot support a source-type interaction.

**Finding test:** among selected E2/E3 videos, is the news-versus-commentary alignment difference reproducible, or is it a few-channel artifact?

### Approval gate for extended analysis

Sections 5.6 and 5.7 are intentionally retained because they may yield valuable new knowledge from already frozen data. They are not part of H1-H4 or the proposal-approved empirical core. Before their results enter the assessed report, record written lecturer approval or a formally accepted proposal amendment covering the exact event, outcome and claim type. Without that approval:

- Google Trends appears only as clearly labelled descriptive context after the three-platform analysis;
- E1, E4 and E5 do not support core empirical conclusions;
- change points, cross-correlation, Granger tests and forecasts remain reproducible exploratory outputs outside the assessed claims;
- no core page, conclusion or recommendation depends on them.

This is a scope-authority gate, not a time or page-limit cut.

### 5.6 Extended event timing and external association — Duy

Google Trends is a comparative external signal, not a causal outcome measure.

1. Rebuild the daily series from the frozen raw windows when available; otherwise checksum and freeze the stitched file. Do not download replacements.
2. Audit window scaling. Re-run results with raw/local-window normalization and alternative stitch anchors to show whether compounded scaling changes conclusions.
3. Estimate segmented interrupted time-series models with day-of-week terms, trend terms and HAC uncertainty for E2, E3 and E5.
4. Treat E1 as contextual/negative control and E4 as a genuine legislative comparison, not a placebo.
5. Use Ireland and New Zealand only as imperfect comparators. Check pre-trends and explicitly avoid “clean control,” “treatment effect” or “lower bound” claims that cannot be identified.
6. Compare fixed event dates with data-driven change points using PELT or a Bayesian change-point model. A discovered break is corroboration only when temporally close and robust; it does not prove the event caused the change.
7. Build daily discourse panels from phrase-exact Bluesky search posts, validated frame/stance/sentiment series and volume per baseline. Use AU series only where support passes the declared threshold.
8. Run cross-correlation and distributed-lag/Granger predictive tests on stationary or differenced series, with pre-whitening and corrected significance.
9. Use false event dates and unaffected search terms/country-event pairs as placebo checks where the frozen Trends data permits.

No petition JSON or publicly reported VPN surge series will be added as analysed data. Such material may be cited as background only and must be kept separate from empirical results.

### 5.7 Extended predictive modelling — Duy

Prediction is restored as a secondary research contribution, not cut for time.

#### Target and design

- Predict UK and AU Google Trends `VPN` interest at **t+1** and **t+7**.
- Features may include lagged target values, calendar terms, event phase, Bluesky volume, validated frame/stance/sentiment proportions and network summaries available before the forecast origin.
- Every feature must be timestamped and available before its prediction cutoff. Add an automated leakage test.

#### Models

1. Seasonal-naive forecast.
2. Autoregressive lag-only model.
3. Elastic Net with lagged discourse features.
4. Histogram gradient boosting with the same permissible information set.

Use rolling-origin evaluation, never a random split. Tune only inside each training window. Compare MAE and RMSE, report uncertainty in improvement over the lag-only baseline, and inspect permutation importance or stable coefficients. The claim is limited to incremental predictive information. If social features do not improve out-of-sample forecasts, that is a strong negative finding.

### 5.8 Robustness and falsification programme — all owners

Run a predeclared, estimand-specific specification curve across the choices most likely to change each conclusion. This is not a full Cartesian product: each hypothesis lists a small set of theoretically defensible variants, and combinations are added only when two choices plausibly interact.

- strict, inclusive and human-calibrated language populations;
- phrase-exact versus recall-oriented Bluesky populations;
- Reddit audited-thread versus keyword-screen populations;
- document-weighted versus author-balanced estimates;
- event-window widths and overlapping-event rules;
- lexicon, supervised and optional LLM label instruments;
- NMF/BERTopic seeds and topic solutions;
- Leiden/Louvain, resolution and community seeds;
- directed/undirected and binary/weighted graphs where substantively valid;
- PageRank damping factors and hub removal;
- leave-one-thread/video/subreddit/community-out checks;
- alternative Google Trends stitching and false event dates.

Summarize specifications by effect direction, magnitude and decision stability—not a wall of p-values. The main report shows the decisive sensitivity; the complete targeted matrix remains a reproducible output.

### 5.9 Cross-platform synthesis and action — all, integrated by Duy

Build an evidence matrix with one row per candidate finding:

| Field | Required content |
|---|---|
| Claim | One precise sentence with population and event scope |
| Evidence | Effect size, interval, sample/graph size and primary figure |
| Replication | Platforms/events where direction repeats or conflicts |
| Robustness | Specifications passed and failed |
| Alternative explanation | Strongest plausible rival interpretation |
| Evidential status | Confirmatory, secondary or exploratory |
| Bounded conclusion | Strongest wording the design supports |
| Decision implication | Actor, mechanism and candidate response |

The final decision question is:

> Which implementation risk should age-assurance policymakers and platform implementers address first: privacy/surveillance, technical circumvention/efficacy, governance/accountability, autonomy/censorship, or insufficient child protection?

Do not create an opaque composite priority score. Select no more than two action priorities using transparent rules:

1. replicated across at least two platforms or events, unless a platform-specific mechanism is the finding;
2. substantively meaningful effect with uncertainty shown;
3. stable across the decisive robustness specifications;
4. connected to high-attention, brokered or propagating parts of the observed network;
5. actionable by a named actor with a measurable success indicator;
6. accompanied by a guardrail against foreseeable harm.

For each final priority, state:

- **who acts;**
- **what should change first;**
- **why this evidence outranks competing issues;**
- **the mechanism by which the change should help;**
- **what observable indicator would show improvement;**
- **what result would show the intervention is failing;**
- **what privacy, equity or rights guardrail must not be traded away.**

This is where the report must exceed earlier work: extensive robustness must end in a clear, testable decision rather than “more research is needed.”

### 5.10 Model selection boundaries

The plan is method-rich without treating novelty as validity. PageRank remains because it cleanly measures prestige/attention under an explicit reply direction; it is strengthened by personalized PageRank, stability tests and comparison with brokerage and spreader roles.

Do not add graph neural networks, temporal graph networks, node embeddings or a single opaque “influence score” unless a predeclared supervised task and valid held-out target require them. The frozen networks do not provide ground-truth influence, exposure or persuasion labels. Likewise, do not force synthetic control or causal difference-in-differences from a few contaminated comparator countries. These exclusions are about identification and interpretability, not time or page limits.

## 6. Report architecture and evidence hierarchy

### 6.1 Finding-led structure

The final report is not organised as six mini-reports titled EDA, sentiment, topics, networks, timing and prediction. Methods live in the method section; results are organised around three to five integrated findings.

Each main finding must contain:

1. a claim-style heading;
2. one effect size or comparison with uncertainty;
3. one primary figure or table;
4. the network/NLP/temporal mechanism that explains it;
5. the strongest robustness challenge;
6. the strongest alternative explanation;
7. a bounded implication.

Possible headline forms—only if supported—include:

- Attention and brokerage identify different actors.
- Privacy concern receives attention, while technical workarounds cross communities.
- Negative sentiment substantially overstates opposition to age assurance.
- Implementation produces a different debate from legislation.
- Platform architecture changes where frames segregate and where they meet.
- If scope-approved, social discourse does, or does not, improve short-horizon VPN-search prediction.

These are candidate stories, not conclusions to engineer.

### 6.2 Hero evidence

Aim for six to eight main visuals, chosen after results exist:

1. data/event coverage and validity map;
2. frame and stance change around policy stages;
3. stable/unanticipated topic evidence;
4. community-frame map with a null comparison;
5. attention-versus-brokerage role map and stability;
6. propagation/cascade or frame-transition result;
7. YouTube video-audience alignment;
8. timing/prediction evidence only if scope-approved and it changes the conclusion.

Every title should state the finding, not the chart type. Every visual must have readable units, denominator, population, uncertainty and sample/graph size. Decorative network hairballs, redundant robustness plots and unranked model dashboards do not enter the main narrative.

### 6.3 The 20-page packaging rule

The page ceiling is handled only after the full evidence base exists. A provisional main-report budget is:

| Section | Pages |
|---|---:|
| Executive summary and two decisions | 0.75 |
| Problem, research gap, RQs and contribution | 1.25 |
| Data, populations and validity | 2.00 |
| Core methods and inference | 2.00 |
| Finding 1: frames, stance and topics | 2.50 |
| Finding 2: communities, attention and brokerage | 3.50 |
| Finding 3: propagation and YouTube alignment | 2.50 |
| Timing and predictive value | 1.50 |
| Synthesis and action priorities | 1.50 |
| Limitations and conclusion | 1.00 |
| References | 1.50 |
| **Total** | **20.00** |

This allocation is adjustable and does not cancel an analysis. If extended analysis lacks approval, its 1.50 pages return to the strongest core findings rather than leaving a gap. Full validation tables, model diagnostics, null distributions, specification curves, extra network views and reproducibility details go to an allowed appendix or repository supplement. Until the marking guide confirms otherwise, assume every page inside the submitted PDF counts; the analysis remains unchanged either way.

## 7. Implementation sequence, command contract and ownership

Plan one thin entry point, `python -m src.analysis.run`, whose subcommands call testable workstream functions. The command names below are the implementation contract; they do not imply that code already exists.

| Task | Verifiable slice | Owner | Depends on | Acceptance criteria | Verification |
|---|---|---|---|---|---|
| T0 | Freeze core/extended scope authority | All | None | Proposal-core and approval-gated analyses recorded; no ambiguous event status | `python -m src.analysis.run scope --check` |
| T1 | Freeze source and population manifests | An | T0 | Checksums/counts/exclusions and all six `population_id` rules reproduce | `python -m src.analysis.run populations --check` |
| T2 | Freeze shared masking, case dates and time axes | An | T1 | PII fixture masked; AU legislation assigned; Bluesky uses `day`; E5 old-date artifact rejected | `pytest -q tests/analysis/test_shared_contracts.py` |
| T3 | Freeze v2 codebook, crosswalk and annotation packet | Hung | T0 | Every legacy construct has a disposition; stance target/context examples pass adjudication review | `python -m src.analysis.run codebook --check` |
| T4 | Draw and freeze validation splits | Hung + An | T1, T3 | Exact strata, IDs, probabilities, cluster caps and 600/600/300 split reproduce from seed | `python -m src.analysis.run validation sample --check` |
| T5 | Select sentiment pipeline on development data | An | T4 | Grouped development metrics saved; one pipeline/platform frozen without evaluation access | `python -m src.analysis.run validation select --task sentiment --check` |
| T6 | Select stance and frame pipelines on development data | Hung | T4 | One stance and per-frame pipeline/platform frozen; thresholds and fallbacks recorded | `python -m src.analysis.run validation select --task stance,frames --check` |
| T7 | Open evaluation once and publish validity report | All | T5, T6 | Reliability, per-class metrics, positive counts and pass/fallback decisions immutable | `python -m src.analysis.run validation evaluate --check` |
| T8 | Produce coverage, concentration and descriptive timelines | An | T1, T2 | Denominators reconcile; gaps/minimum-volume warnings shown; core/extended panels separated | `python -m src.analysis.run descriptive --check` |
| T9 | Build and audit graph registry | Kien | T1, T2 | Actor graphs, message trees and bipartite graphs reproduce; all endpoint/filter counts reported | `python -m src.analysis.run networks build --check` |
| T10 | Freeze topology-only communities and structural roles | Kien | T9 | Leiden stability/null diagnostics and PageRank/role stability complete before frame join | `python -m src.analysis.run networks structure --check` |
| T11 | Execute H1-R/H1-B | Kien + Hung | T7, T10 | Exact author-vector null, 9,999 permutations and `PRIMARY-6` outputs reproduce | `python -m src.analysis.run hypothesis --id H1 --check` |
| T12 | Execute H2-P/H2-C | Hung + An | T1, T7 | Reddit-only contrasts, thread-cluster uncertainty and both primary endpoints reproduce | `python -m src.analysis.run hypothesis --id H2 --check` |
| T13 | Execute H3 and secondary role comparisons | Kien | T7, T10 | Matching balance, entropy estimand, permutations and hub sensitivity reproduce | `python -m src.analysis.run hypothesis --id H3 --check` |
| T14 | Execute H4 selected-video comparison | Kien + Hung | T7, T9 | E2/E3 news-commentary contrast and channel/video sensitivities reproduce | `python -m src.analysis.run hypothesis --id H4 --check` |
| T15 | Run message-tree cascades and temporal communities | Kien | T7, T9, T10 | Only valid observed trees used; proxy/heuristic analyses visibly separated | `python -m src.analysis.run networks temporal --check` |
| T16 | Run stable NMF/BERTopic discovery | Hung | T7 | Platform-level topic spaces, seed stability and human naming ledger reproduce | `python -m src.analysis.run topics --check` |
| T17 | Run approval-gated timing models | Duy | T0, T7, T8 | Approval receipt present or command exits without assessed output; stitch/placebo diagnostics pass | `python -m src.analysis.run extended timing --check` |
| T18 | Run approval-gated forecasts | Duy | T17 | Rolling-origin splits and leakage tests pass; baselines and uncertainty saved | `python -m src.analysis.run extended prediction --check` |
| T19 | Run targeted specification curves | Each owner | T11-T18 as applicable | Every primary finding has its predeclared decisive variants; no Cartesian fishing | `python -m src.analysis.run robustness --check` |
| T20 | Build evidence/action matrix and report bundle | All; Duy integrates | T11-T19 | Three-to-five finding cards, one-to-two actions, traceable figures and number audit complete | `python -m src.analysis.run report --check` |

### Checkpoints

- **Foundation checkpoint after T4:** approve scope, populations, time axes, codebook and sample before any final modelling.
- **Measurement checkpoint after T7:** freeze validity decisions; failed cells receive fallbacks before hypothesis execution.
- **Core-evidence checkpoint after T14:** H1-H4 complete before advanced temporal work can influence the story.
- **Research checkpoint after T18:** classify every advanced result as core-approved, extended-approved or repository-only.
- **Submission checkpoint after T20:** regenerate the complete report bundle in a clean environment and reconcile all values.

After T7, T8/T9/T16 can proceed independently; H1-H4 follow their listed dependencies. This preserves parallel work without allowing labels, populations or event definitions to drift.

Add `requirements/analysis.txt` to the existing requirements chain. Pin only dependencies used by activated methods: `scipy`, `statsmodels`, the selected transformer stack, `python-igraph`/`leidenalg`, and BERTopic dependencies when BERTopic is run. Keep NetworkX as the default. Record an isolated environment recipe for any optional degree-corrected block-model implementation rather than forcing it into the shared environment.

The minimum global verification command is `pytest -q tests/analysis`; every analysis subcommand also supports `--check` to validate inputs and existing outputs without recomputing expensive models.

### Ownership and handoffs

| Lead | Primary responsibility | Required handoff |
|---|---|---|
| An | Population/coverage, language audit, sentiment and stance | Frozen population manifest; validated daily/document labels; number-consistency audit |
| Hung | Codebook, frames, topics and validation coordination | Versioned frame/topic labels; validation report; YouTube metadata labels |
| Kien | Graph registry, communities, roles, propagation and YouTube alignment | Versioned edges/partitions/measures; explicit network-construction statement |
| Duy | Synthesis/report integration plus approval-gated timing and prediction | Evidence/action matrix; frozen daily panel and rolling forecasts only under the scope gate |
| All | Double coding, interpretation, limitations and action priorities | Independent labels, finding cards and adversarial review |

Use logical modules under `src/analysis/` and versioned outputs under `data/analysis/`; do not create extra abstractions merely to mirror this document. Shared definitions belong in the existing shared analysis code. Notebooks may explore but cannot be the only source of a reported result.

Each owner delivers a **finding card** rather than an isolated mini-report:

- precise claim and evidential status;
- population, sample/graph size and label version;
- effect size, interval and corrected test where applicable;
- one primary visual and source-data path;
- decisive robustness result;
- strongest alternative explanation;
- bounded conclusion and decision relevance.

### Minimum reproducible artifacts

1. Frozen-data and population manifest.
2. Codebook, sampling manifest, independent labels and adjudicated gold set.
3. Model cards with revision, inputs, metrics and known failure modes.
4. Graph registry and edge-construction audit.
5. Hypothesis table with primary outcomes and correction families.
6. Full robustness/specification matrix.
7. Figure/table source-data files and generation commands.
8. Evidence-to-action matrix and final number-consistency check.

### Operational rules carried forward

- Read Bluesky `text_raw` from `posts.parquet`, YouTube `text_clean` from `documents.parquet`, and Reddit `text_sentiment` for sentiment or `text_topic` for lexical/topic work. Create one shared `mask_text()` helper with focused tests and apply it after loading so every instrument receives the same protected input.
- Do not rerun the frozen Reddit preparation merely to regenerate analysis files: its random salt would change `author_hash` values and break graph/result joins. Freeze the processed run and record its checksum.
- Give every record a stable analysis `doc_id`. Assign Reddit proposal cases with the new case-window registry and use `src/shared/events.py` only for E1-E5 point-event overlays; do not reinterpret collector windows as E1-E5.
- Use the common descriptive window from day -7 through, but not including, day +21 unless a workstream declares and justifies another window. Timing models must also report segment lengths and overlapping-event handling.
- A daily estimate based on fewer than 20 eligible documents is missing, not zero. If more than one-third of a displayed daily series is unsupported, use weekly aggregation. Revisit this threshold in the specification curve.
- Count unique Bluesky `doc_id` values after joining phrase-exact query membership; never sum per-query counts because one post can match several queries. Normalize volume using the frozen daily baseline and expose baseline gaps.
- Treat `eSafety`, `under-16 ban`, `social media ban` and `Online Safety Act` as query context, not geography. Preliminary thin-cell warnings—such as low E3/E5 counts for `eSafety` and `under-16 ban`—must be regenerated and attached to affected estimates. Re-run UK interpretations with `Ofcom` alone and AU interpretations without globally ambiguous queries.
- Build YouTube reaction curves with `yt_video_event_id` and `yt_days_from_video_event`. The primary commenter-video graph uses strict-eligible documents only. Keep channel-owner replies in the structural graph when they represent creator-audience interaction, then exclude them in sensitivity analysis.
- Filter spam/bot endpoints with each platform's frozen flags and report every exclusion. Primary YouTube actor-reply inference uses strict-eligible sources and `handle_match` targets only; thread-root fallbacks are a separate sensitivity. For Bluesky, remove labelled spam/repeater accounts in the primary network and show the inclusive sensitivity.
- Every Bluesky temporal field, ordering and event join uses processed `day`. `created_at` may appear only in the data-quality audit because client-supplied timestamps include backdated records.
- Store versioned labels, graphs, models, tables and figures under `data/analysis/<workstream>/`. Every artifact records population version, codebook/model version, seed and source checksum.
- Owners may begin on label-free graph structure or frozen descriptive panels, but no confirmatory frame, stance or sentiment result is final until the shared held-out validation is frozen.

## 8. Decision gates and stopping rules

- Freeze populations, codebook, primary outcomes and hypothesis families before final confirmatory results.
- Do not tune, select or switch a classifier/lexicon using the untouched evaluation set.
- Do not promote approval-gated events or models into assessed claims without a recorded scope decision.
- Do not interpret a frame/platform estimate when validation is inadequate; bound or narrow it.
- Do not interpret a community partition until stability and null comparisons are available.
- Do not call PageRank influence; state the edge semantics beside the result.
- Do not run the relational event model if the risk set or timestamp coverage is indefensible.
- Do not claim cross-platform replication when the construct or population differs materially.
- Do not use prediction results without rolling-origin improvement over a lag-only baseline.
- Do not choose final recommendations until the evidence matrix is complete.
- Do not remove a valid analysis because of page count. Decide only whether it belongs in the main narrative or supplement.

## 9. Known limitations to carry into every interpretation

- The platforms were collected differently and are not representative samples of their user populations.
- Pseudonymous accounts are not verified individuals; author-balanced estimates still describe accounts.
- Bluesky lacks adequate quote/repost edges, has replies only for selected threads and depth, and requires index/file `day` rather than client-supplied `created_at` for time.
- Reddit reply edges cannot connect subreddits; cross-subreddit structure comes from shared authors.
- YouTube video selection is purposive/view-ranked, actor reply targets are partly heuristic, source-type findings are bounded to selected videos, and E5 has only seven videos.
- Platform/query/subreddit context is not user geography.
- Language, relevance, frame and stance labels contain measured error even after validation.
- Google Trends values are relative and stitched across windows; comparator countries share media and policy exposure.
- Event studies remain vulnerable to concurrent events, anticipation and selection.
- Reply order and cascade shape show observed interaction, not exposure, belief change or causal diffusion.
- No cross-platform identity linkage is attempted, so “cross-platform propagation” means comparable aggregate timing and framing, not traced users.

## 10. Completion criteria

The analysis is complete only when:

- the frozen population manifest reproduces every headline denominator;
- 1,500 usable documents have independent double coding across the frozen 600/600/300 allocation, or any shortfall and its consequence is explicitly reported;
- sentiment, stance and frame claims cite held-out per-platform validity;
- all six `PRIMARY-6` endpoints reproduce their frozen population, estimand, null, effect threshold, uncertainty, multiplicity and decision result;
- every network result states nodes, edges, direction, weight, filters and what the measure means;
- PageRank, community and broker findings survive their declared stability/null checks or are reported as unstable;
- timing language remains associational and prediction is evaluated out of sample without leakage;
- three to five integrated findings are supported by finding cards;
- one or two action priorities name the actor, first action, mechanism, success indicator, failure indicator and guardrail;
- every number and figure is script-generated and traceable;
- the main report is readable within 20 pages while the complete evidence base remains available in the permitted supplement/repository.

The intended standard is not “more analysis than the land-pricing report.” It is stronger construct validity, clearer network science, harder falsification, more honest inference, more consequential findings and a tighter evidence-to-decision story.
