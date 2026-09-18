# The Age-Gate Paradox: Child Safety, Privacy and Circumvention Across Three Platform Networks

*Project proposal for Social Media and Network Analytics (four-person team)*

---

## 1. Project summary

Age-assurance policies ask platforms to distinguish children from adults in order to reduce young people's exposure to harmful content. The same systems can require identity documents, facial age estimation, behavioural inference or other personal data from adults as well as children. This creates the **age-gate paradox**: a policy designed to make participation safer may also make participation less private, less anonymous and easier to monitor.

The policy conflict became concrete during two recent cases. Under the United Kingdom's Online Safety Act, relevant services were required to act on risks to children from 25 July 2025. Australia's social-media minimum-age obligation took effect on 10 December 2025 and requires covered platforms to take reasonable steps to prevent people under 16 from holding accounts. These policies differ, so they will be analysed as separate cases rather than treated as equivalent interventions.

This project investigates how child-safety, privacy/surveillance, governance and circumvention/autonomy frames are organised across **Reddit, YouTube and Bluesky**. Each platform answers a different question and produces a different network. User identities and graphs will never be merged across platforms. Comparison occurs only after each platform has been analysed on its own terms.

The intended contribution is not a prediction system or a causal estimate of whether age assurance works. It is an evidence-led account of how arguments are distributed within platform structures, which communities remain separated, and which actors or content objects connect competing interpretations of the same policy debate.

## 2. Problem and analytical contribution

Public counts of posts, comments or sentiment cannot show where an argument appears or who connects it to other parts of an online community. A privacy objection voiced inside a specialist community has a different social role from the same objection appearing across national discussions, news-video audiences or repost networks. Network structure is therefore part of the research problem, not an illustrative add-on.

The project makes three connected contributions:

1. **Substantive:** it maps the tension between child protection and privacy, surveillance, governance and circumvention around clearly dated UK and Australian policy events.
2. **Network-analytic:** it tests whether competing frames are structurally segregated, connected by brokers or amplified through observable platform interactions.
3. **Comparative:** it explains which findings are platform-specific and which recur across different community architectures, without pretending that unlike networks or user populations are directly interchangeable.

The final report will distinguish observed interaction, supported inference and speculation. Replying, quoting or reposting is observable. Exposure, persuasion, residence, age and offline behaviour are not.

## 3. Research questions and hypotheses

### Central research question

**How are child-safety, privacy/surveillance, governance and circumvention/autonomy frames distributed across the interaction networks of Reddit, YouTube and Bluesky around UK and Australian age-assurance policy events?**

### Platform-specific questions

| Platform | Question | Why this platform is needed |
|---|---|---|
| Reddit | How do frames differ between privacy, technology, parenting and national communities, and which participants bridge otherwise separated discussions? | Subreddits and full comment trees provide explicit community boundaries and reply-to relations. |
| YouTube | How do frames signalled in video titles and descriptions relate to the frames and sentiment expressed by their audiences? | Video metadata provides a reproducible framing surface, while comments and shared audiences connect content and viewers. |
| Bluesky | Which frames are amplified through replies, quotes and reposts, and which accounts connect policy, safety and digital-rights conversations? | Typed interaction records permit analysis of conversational response and amplification. |
| Cross-platform synthesis | Which frame and network patterns recur across policy events, and which appear to follow platform affordances? | Comparison can separate event-consistent patterns from platform-specific organisation while retaining each platform's analytical boundaries. |

### Pre-specified hypotheses

- **H1 (frame segregation):** interaction communities will contain more frame concentration than expected under a permutation baseline that preserves network structure and frame frequencies.
- **H2 (event-stage shift):** within the Australian case, privacy/surveillance and circumvention/autonomy frames will occupy a larger share of relevant discourse around implementation than around legislation.
- **H3 (brokerage and frame diversity):** high-betweenness participants will engage with a wider range of frames than activity-matched participants with low betweenness.
- **H4 (media–audience alignment):** YouTube comment sections will partly reproduce the dominant frame signalled by their video's title and description, but the strength and direction of alignment will vary by source type.

These are hypotheses, not expected findings. Null or contradictory results answer the research question and will be reported rather than treated as failed analyses.

## 4. Case selection and event windows

The empirical scope is English-language discussion around three bounded windows already encoded in the Reddit collector. Intervals are half-open UTC ranges: the start is included and the end is excluded.

| Event window | UTC interval | Policy anchor | Analytical purpose |
|---|---|---|---|
| Australian legislative debate | 10 Nov 2024 to 11 Jan 2025 | Enactment of the *Online Safety Amendment (Social Media Minimum Age) Act 2024* | Establishes how the policy was framed during formation and legislative debate. |
| UK enforcement and European policy cluster | 14 Jun 2025 to 26 Aug 2025 | UK child-safety duties applying from 25 Jul 2025, with contemporaneous European discussion retained as context | Examines debate when age-gating duties became operational for relevant services. |
| Australian implementation | 10 Nov 2025 to 11 Jan 2026 | Social-media minimum-age obligation taking effect on 10 Dec 2025 | Compares anticipatory and immediate implementation discourse with the earlier legislative window. |

The UK and Australian instruments are not identical. The UK case concerns risk-based child-safety duties and highly effective age assurance for relevant content and services. The Australian case concerns a minimum age for accounts on covered social-media platforms. Results will first be interpreted within each policy context; only frame and network patterns that are conceptually comparable will enter the synthesis.

Subreddit names, profile text and language will not be used as evidence of residence. Geographic labels describe policy-event context or community participation, not user location.

## 5. Data collection

### 5.1 Common collection principles

- Collect only public material available through documented interfaces.
- Preserve immutable raw records and a machine-readable collection manifest before cleaning.
- Record platform, query, event window, UTC timestamp, content identifier, parent/root relation, public engagement fields and acquisition route.
- Use high-recall search followed by an audited relevance screen. Search matches are candidates, not automatically relevant observations.
- Deduplicate records while preserving every query and route through which each record was discovered.
- Keep platform-specific identifiers separate and prohibit cross-platform identity matching.
- Flag deleted, removed, automated, duplicate, promotional and uncertain-language material before defining analysis samples.

### 5.2 Reddit: peer-to-peer community discourse

Reddit is the most mature collection stream. Arctic Shift is used to search public posts and retrieve the complete available comment tree for every matched post.

**Communities:** `r/privacy`, `r/technology`, `r/parenting`, `r/unitedkingdom`, `r/europe` and `r/australia`.

**Search design:** fifteen age-assurance queries spanning age verification, age assurance, age checks, age estimation, under-16 restrictions, social-media bans, child safety, online safety, digital identity, biometrics, privacy/surveillance and VPN-related age verification. Broader terms enter a second relevance screen so that generic child-safety or digital-identity posts are not silently treated as age-assurance evidence.

**Collected units:** post title and self-text, post metadata, comment text, author identifier, `parent_id`, `link_id`, score, subreddit and timestamp. Full trees preserve replies to replies rather than only top-level comments.

**Feasibility evidence:** a retry-enabled stratified pilot completed all six selected comment trees. In one Australian implementation window and one query, it found 66 matched posts and 609 raw comment rows in `r/privacy`, 14 posts and 777 raw comment rows in `r/australia`, and no matches in `r/parenting`. Two search timeouts were recovered by bounded retries; the completed pilot recorded no HTTP 429 responses, collapsed nodes or unknown tree nodes. These are raw collection counts, not final relevance-eligible counts. The zero-result parenting stratum is a substantive negative observation, not a collection failure.

### 5.3 YouTube: media framing and audience response

YouTube is a core platform, not a stretch goal. A reproducible, purposive sample of approximately 8–12 eligible English-language videos per event window will be frozen before comment collection. Selection will balance mainstream news, public or regulatory sources, civil-society/digital-rights sources and creator commentary where those categories are available. Relevance, publication timing, public availability and enabled comments are eligibility criteria; view count alone will not determine inclusion.

For each selected video, the project will collect title, description, channel, publication time, public engagement metadata, top-level comments and all retrievable replies. `commentThreads.list` supplies threads and limited embedded replies; `comments.list` will retrieve remaining replies when `totalReplyCount` shows that the embedded set is incomplete. The selection log will retain included and excluded videos and the reason for each decision.

Comments are a snapshot of what remains public at collection time. Deleted comments, disabled comment sections and ranking effects will be documented.

### 5.4 Bluesky: conversation and amplification

Bluesky will be collected through the public AT Protocol/Bluesky interfaces using policy and age-assurance query terms. Matched posts will retain author DID-derived research identifiers, record URI, text, timestamp, reply references, quote references and available repost metadata. Post threads will be retrieved for matched roots where the interface permits.

Historical search completeness is a known feasibility risk. Before full collection, a dated-search pilot must confirm query behaviour, pagination, temporal coverage and thread completeness for each event window. If complete historical recall cannot be established, Bluesky will be described as a bounded query sample rather than a census. Its findings will remain valid for the collected sample, but no volume comparison will imply platform-wide prevalence.

## 6. Network construction

Each network is defined before analysis so that nodes and edges have a defensible social meaning.

| Platform network | Nodes | Edges | Direction and weight | Interpretation |
|---|---|---|---|---|
| Reddit reply network | Pseudonymised users | User A replies to content authored by User B | Directed A → B; weight is number of replies | Participation, attention, repeated interaction and brokerage within threaded communities. |
| YouTube commenter–video network | Pseudonymised commenters and selected videos | A commenter posts on a video | Bipartite and undirected; weight is comment count | Audience overlap, participation around videos and relations between metadata framing and audience response. |
| YouTube reply subgraph | Pseudonymised commenters | User A replies to User B's top-level comment | Directed A → B; weight is reply count | Conversational structure where reply density is sufficient. This is secondary to the bipartite network. |
| Bluesky conversation network | Pseudonymised accounts | User A replies to or quotes User B | Directed; edge type retained and repeated interactions weighted | Conversation, contestation and cross-frame brokerage. |
| Bluesky amplification network | Pseudonymised accounts | User A reposts User B | Directed A → B; weight is repost count | Observable amplification. It is analysed separately from replies and quotes before any sensitivity analysis combines edge types. |

Deleted or unavailable authors cannot become ordinary user nodes. Self-loops, bots and low-frequency edges will be reported and handled through documented sensitivity analyses rather than removed by an unexplained threshold.

No cross-platform supergraph will be built. Centrality values from networks of different size and meaning will not be compared as if they share one scale.

## 7. Text and NLP analysis

### 7.1 Preprocessing

The pipeline will normalise Unicode and timestamps, remove markup while retaining analytically meaningful punctuation and emoji, identify language, flag uncertain cases, and create separate text views for topics and sentiment. English is the primary language. Non-English records will be counted for coverage reporting but excluded from automated English-language inference unless manually adjudicated.

### 7.2 Frame analysis

A multi-label codebook will use five groups already represented in preprocessing:

1. age policy and assurance;
2. child safety;
3. privacy and surveillance;
4. governance and platform responsibility;
5. circumvention, censorship and autonomy.

Multi-label coding matters because the paradox often appears when child-safety and privacy claims coexist in the same document. Rule-based terms provide a transparent candidate baseline. They do not become ground truth.

A stratified gold set of approximately 300 items will cover platforms, event windows, document types, frame groups and difficult cases. Two team members will independently code each item; a third will adjudicate disagreements. Inter-rater reliability, per-frame precision, recall and F1 will be reported. If a frame cannot reach usable agreement, its definition will be revised or merged before full-corpus claims are made. If automated frame detection remains unreliable, inference will be restricted to the human-coded sample.

### 7.3 Topic and sentiment analysis

TF-IDF with NMF will provide an interpretable inductive topic baseline separately for each platform. Topic number will be selected using stability, coherence and human interpretability rather than a single optimisation score. Topics will be compared with the deductive frame codebook to identify both expected and unanticipated themes.

VADER will be used only as an English social-text sentiment baseline. Sentiment is secondary to frame and stance because negative language does not necessarily mean opposition to age assurance. A stratified human-labelled sample and, if feasible, a documented transformer baseline will test whether the sentiment pattern is stable. If instruments disagree materially, the report will present the disagreement and avoid a single definitive sentiment series.

No LLM-generated label will be treated as ground truth. Any model-assisted interpretation must retain the original evidence and be reproducible from a documented prompt and model version.

## 8. Analysis plan

Collection, preprocessing, NLP, network construction, statistical analysis and assessed visualisation will be implemented primarily in Python. R will not be used.

### 8.1 Coverage and descriptive analysis

The analysis begins with a platform-by-event coverage table: candidate records, relevant records, deleted or unavailable text, unique pseudonymised actors, observed relations, connected components and collection completeness. Volume is reported as sample activity, not public opinion.

### 8.2 Structure, communities and influence

For each appropriate network, the project will report node and edge counts, density, reciprocity, component structure and degree distributions. Weighted Louvain community detection will identify interaction communities on an undirected projection where required; modularity and sensitivity to the resolution parameter will be reported.

Influence measures will be tied to specific interpretations:

- weighted in-degree or PageRank: attention within the observed network;
- weighted out-degree: participation directed toward others;
- betweenness centrality: potential brokerage between otherwise weakly connected regions;
- bipartite degree and audience overlap: participation breadth and shared audiences in the YouTube sample.

These measures describe structural opportunity, not persuasion or offline influence. Rankings will be accompanied by frame profiles and local network context rather than presented as unsupported “top influencer” lists.

### 8.3 Connecting network and NLP evidence

- Compare frame distributions across detected communities and predefined Reddit community strata.
- Measure edge-level similarity between users' multi-label frame profiles and, as a sensitivity check, dominant-frame assortativity; compare both with label-permutation baselines.
- Test whether brokers produce or connect a broader set of frames after accounting for activity level.
- Compare the frame profile of each YouTube video's title and description with the aggregate profile of its comments.
- Trace observed Bluesky reply, quote and repost paths for selected high-engagement claims.
- Plot daily or weekly frame share around event anchors with uncertainty intervals and minimum-volume warnings.

First appearance across communities will be called **temporal propagation** unless an observed reply, quote or repost path supports a stronger diffusion claim.

### 8.4 Cross-platform synthesis

Platforms will be compared through a common results matrix containing frame prevalence, frame co-occurrence, community concentration, brokerage patterns and temporal direction of change. Raw centrality scores, raw sentiment levels and identities will not be pooled. The synthesis will ask whether a pattern is event-consistent, platform-specific or too uncertain to classify.

Event-window comparisons are observational. The report will use “associated with”, “coincides with” and “precedes” unless a design directly supports stronger language.

## 9. Success criteria and decision rules

| Criterion | Convincing evidence | Failure or fallback rule |
|---|---|---|
| Collection integrity | Versioned manifests, exact query/window provenance, deduplicated routes and explicit incomplete-task counts | An incomplete stratum is labelled incomplete and excluded from prevalence comparison; it is never silently treated as zero. |
| NLP validity | Reported inter-rater reliability and per-frame validation against a stratified gold set | Unstable labels are revised or merged; weak automation is replaced by inference from the coded subset. |
| Meaningful network | Observed relations support stable structure, community or brokerage interpretation under sensitivity checks | Sparse reply data uses the pre-specified bipartite design or remains a documented null network finding. |
| Research-question coverage | Every core platform contributes at least one text result and one network result tied to its platform question | Methods that do not answer a stated question are cut from the final report. |
| Comparative validity | Platform-specific results remain separate and only conceptually comparable quantities enter synthesis | No pooled actor graph, pooled sentiment score or implied common population. |
| Reproducibility | Python scripts regenerate analysis tables and figures from frozen processed data; table, figure and report values agree | Hand-edited figures and untraceable numbers do not enter the submission. |

The project succeeds if it produces a defensible explanation of where frames cluster, interact or fail to appear. It does not require confirmation of every hypothesis.

## 10. Expected analytical outputs and story

The final report will be organised around one argument: **age assurance is debated not only through competing values, but through platform-specific network patterns that show where those values meet, separate or gain visibility.**

Planned evidence includes:

1. an event timeline that separates legislation, enforcement and implementation;
2. a transparent data-coverage and exclusion table;
3. one readable network map and structural summary per platform;
4. community-by-frame heatmaps, edge-level frame similarity and dominant-frame assortativity sensitivity results;
5. profiles of structurally important brokers with anonymised textual evidence;
6. event-aligned frame and sentiment trajectories with uncertainty and volume warnings;
7. a YouTube title/description-frame versus audience-frame comparison;
8. a cross-platform synthesis matrix identifying convergent, divergent and inconclusive patterns.

The report will preserve negative results. For example, a parenting community with no relevant posts, a sparse YouTube reply graph or a Bluesky window with incomplete historical coverage can reveal boundaries of attention or data access when reported precisely.

## 11. Limitations and non-goals

- Query-based samples do not represent all platform discourse or general public opinion.
- Deleted, moderated, private and algorithmically hidden content is unavailable.
- English-only analysis excludes affected users who discuss policy in other languages.
- Community membership, language and profile text do not establish age, nationality or residence.
- Platform affordances and sampling methods differ; cross-platform contrasts are interpretive, not population estimates.
- Replies, quotes and reposts show interaction, not exposure, belief change or persuasion.
- Circumvention **discourse** does not measure actual VPN use or successful bypass behaviour.
- The study does not estimate policy effectiveness or causal effects. Difference-in-differences, Granger causality and forecasting are outside the core design.
- Google Trends or official implementation statistics may be used only as clearly labelled context after the three-platform analysis is complete.
- Texas adult-content regulation, Australian adult-content-only rules and Vietnam are outside the empirical scope because they would introduce different policy constructs or language requirements.

## 12. Ethics and data governance

The study will follow institutional requirements and use the Association of Internet Researchers' Internet Research Ethics guidance as a decision framework.

- No private groups, messages, credentials or inferred sensitive attributes will be collected.
- Raw author fields and source identifiers remain restricted local research data.
- Analysis uses platform-specific, run-scoped pseudonyms. The pseudonymisation secret is not stored with outputs.
- User identities will not be linked across platforms.
- Public deliverables prioritise aggregates. Direct quotations will be minimised, paraphrased where appropriate and checked for searchability and harm.
- A representative submission sample will remove raw text, URLs, raw account metadata and source identifiers while retaining enough structure to demonstrate the method.
- Raw data will be deleted after assessment and reproducibility obligations are satisfied, subject to institutional policy.

Ethical review or course approval, where required, remains an institutional decision; public availability alone is not treated as automatic ethical permission.

## 13. Implementation plan: team of four, 12 weeks

### Roles

| Member | Primary ownership | Shared responsibility |
|---|---|---|
| A: Data and provenance lead | Reddit full collection, manifests, raw/processed schema and reproducibility | Cross-platform schema review |
| B: YouTube and NLP lead | Video selection, comment collection, topic/sentiment baselines and frame codebook | Gold-set coordination |
| C: Bluesky and network lead | Bluesky feasibility/collection, graph construction and network sensitivity checks | Network interpretation review |
| D: Validation and synthesis lead | Annotation reliability, statistical comparisons, figures and cross-platform synthesis | Report consistency and presentation |

All four members annotate the gold set and review at least one workstream they do not own.

### Timeline and gates

**Weeks 1–2: lock scope and prove access**

- Freeze the three event windows, Reddit queries and platform-specific sampling rules.
- Run YouTube and Bluesky pilots; document coverage, pagination, reply density and API limitations.
- Complete the Reddit full scrape using conservative delay, retry and resume settings.
- Draft frame codebook v1 and annotation guide.
- Gate: no full YouTube or Bluesky collection until the pilot produces a valid text table, node table, edge table and completeness report.

**Weeks 3–5: collect, clean and validate**

- Complete platform collections and freeze immutable raw snapshots.
- Produce the coverage/exclusion table before substantive analysis.
- Annotate the gold set, calculate agreement and revise weak frame definitions.
- Freeze the processed analysis corpus and data dictionary.

**Weeks 6–9: core analysis**

- Run topic/frame analyses and validated sentiment baselines.
- Build platform-specific networks, communities, centralities and permutation tests.
- Connect frame evidence to community and brokerage structure.
- Conduct an internal results review in which every result is paired with its strongest validity threat.

**Weeks 10–12: synthesis and delivery**

- Write the report around the central argument rather than around a list of methods.
- Generate every table and figure from Python.
- Check that each figure supports a stated claim and that every value agrees across notebook, report and slides.
- Rehearse the presentation around problem, evidence, alternative explanation, limitation and conclusion.

If time contracts, optional transformer comparison and external contextual indicators are cut first. Frame–network integration, validation and the three platform-specific questions remain the core.

## 14. Primary sources and technical documentation

- Australian Government, Federal Register of Legislation. [*Online Safety Amendment (Social Media Minimum Age) Act 2024*](https://www.legislation.gov.au/C2024A00127/asmade).
- eSafety Commissioner. [Social media age restrictions](https://www.esafety.gov.au/about-us/industry-regulation/social-media-age-restrictions).
- Ofcom. [Age assurance duties under the Online Safety Act](https://www.ofcom.org.uk/online-safety/illegal-and-harmful-content/age-assurance).
- Association of Internet Researchers. [Internet Research Ethics](https://aoir.org/ethics/).
- Arctic Shift. [API documentation](https://github.com/ArthurHeitmann/arctic_shift/blob/master/api/README.md).
- Bluesky. [HTTP API reference](https://docs.bsky.app/docs/api/app-bsky-feed-get-feed).
- Google for Developers. [YouTube Data API `commentThreads.list`](https://developers.google.com/youtube/v3/docs/commentThreads/list).
