# The Age-Gate Paradox

This repository contains the frozen Python code, configuration, documentation and verification tests for the COSC2671 Social Media and Network Analytics project. It studies how child-safety, privacy and surveillance, governance, and circumvention or autonomy arguments appear around age-assurance policy events on Reddit, YouTube and Bluesky. The repository is a reproducible research submission, not a live dashboard.

Start with the [project proposal](docs/age_gate_paradox_project_proposal.md) for the research design, then use this README to install the environment and verify the frozen outputs. The assessed report and presentation are submitted separately through Canvas. The frozen data bundle is supplied through OneDrive and referenced in the Canvas submission; it is intentionally not committed to Git.

## Team

| Student | Student ID |
|---|---|
| Nguyen Hoang Thien An | s3825455 |
| Le Minh Trung Kien | s3651471 |
| Dang Quoc Hung | s4199510 |
| Bui Thanh Duy | s4198464 |

## Read this first

The checkout is frozen for submission. The full test suite passed with `320 passed` on 18 September 2026. The descriptive report bundle and fallback evidence bundle validate successfully. The confirmatory measurement lifecycle remains deliberately gated, so a blocked endpoint is not a null result.

For marking, download the approved frozen data bundle from the OneDrive link included with the Canvas submission, extract it at the repository root, then use the existing artifacts and the read-only `--check` commands below. Do not run a collector or an analysis `--build` command unless the team has authorised a new immutable analysis cycle. Build commands write files, change checksums and can invalidate the frozen evidence.

### Current frozen statuses

| Artifact | Status | Meaning |
|---|---|---|
| `data/analysis/report/report_manifest.json` | `descriptive_bundle_ready_confirmatory_blocked` | The generated report contains traceable descriptive and exploratory evidence. The full confirmatory family is not available. |
| `data/analysis/fallback/analysis_manifest.json` | `--check: valid` | The report-ready fallback uses 600 evaluation records, 285 eligible records and 9,999 cluster-bootstrap replicates. |
| `data/analysis/hypotheses/primary6.json` | `blocked_until_all_six_results_exist` | The six planned endpoints, H1-R, H1-B, H2-P, H2-C, H3 and H4, have no complete PRIMARY-6 result in this snapshot. |
| `data/analysis/hypotheses/h1*.json`, `h2*.json`, `h3.json`, `h4.json` | `blocked` | Required measurement or audit prerequisites are missing. |
| `data/analysis/topics/topics_manifest.json` | `exploratory_named` | Topic discovery is exploratory. Topic stability does not turn topic names into human-coded frame evidence. |
| `data/analysis/extended/timing.json` and `prediction.json` | `approval-required` | Timing and forecasting are approval-gated extensions, not assessed findings. |
| `data/analysis/models/reserve_assessment.json` | `fallback-human-sample` | The reserve records the measurement lifecycle boundary and does not authorise unsupported automated claims. |

The authoritative status is always the current manifest. Do not infer a result from a filename, a passing unit test, a generated figure or a non-zero sample count.

## Reproduce the frozen submission

Run every command from the repository root. The commands below validate existing files and do not contact platform APIs.

### Requirements

- Recommended runtime: CPython 3.14. The supplied development environment was verified with Python 3.14.7.
- The approved frozen data bundle downloaded from the team's OneDrive link in the Canvas submission. Git intentionally ignores `data/`, so a clean clone does not contain the data artifacts.
- Internet access only for installing packages or downloading the spaCy and NLTK resources. No API credentials are needed for the read-only checks.

### Obtain the frozen data bundle

The code repository and the data bundle are separate parts of the submission. Use the OneDrive link recorded in the Canvas assignment submission, download the approved bundle, and extract it at the repository root. Preserve the `data/` directory layout so paths such as `data/analysis/report/report_manifest.json`, `data/analysis/fallback/analysis_manifest.json` and `data/processed/youtube/documents.parquet` exist before running the checks. Do not replace the bundle with live downloads, partial pilot data or a newly collected dataset.

Create the environment and install the pinned dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/python -c 'import nltk; nltk.download("stopwords"); nltk.download("vader_lexicon")'
```

The complete requirements file includes the platform collectors, preprocessing, network analysis, topic analysis, notebooks and report-validation tooling. The packages used by the validators are therefore installed even though the assessed report is submitted separately.

### Verify the code and offline entry points

Run the unit tests first:

```bash
.venv/bin/python -m pytest tests -q
```

The frozen checkout currently reports `320 passed`. The platform self-tests do not use credentials or write research data:

```bash
.venv/bin/python -m src.platforms.reddit.cli --self-test
.venv/bin/python -m src.platforms.youtube.discover --self-test
.venv/bin/python -m src.platforms.youtube.collect --self-test
.venv/bin/python -m src.platforms.youtube.prepare --self-test
```

### Validate the frozen analysis bundle

These checks validate manifests, checksums, input contracts and report safety. They do not rebuild outputs:

```bash
.venv/bin/python -m src.analysis.run scope --check
.venv/bin/python -m src.analysis.run populations --check
.venv/bin/python -m src.analysis.run codebook --check
.venv/bin/python -m src.analysis.run descriptive --check
```

```bash
.venv/bin/python -m src.analysis.run networks build --check
.venv/bin/python -m src.analysis.run networks structure --check
.venv/bin/python -m src.analysis.run networks temporal --check
.venv/bin/python -m src.analysis.run topics --check
.venv/bin/python -m src.analysis.run robustness --check
```

```bash
.venv/bin/python -m src.analysis.run report --check
.venv/bin/python -m src.analysis.fallback_analysis --check
```

Expected successful results include `status=valid` for the core validators, `status=exploratory_named` for topics, `status=blocked_until_prerequisites_are_available` for the robustness specification, `status=descriptive_bundle_ready_confirmatory_blocked` for the report, and `status=valid` for the fallback bundle.

### Checks that remain blocked by design

The following commands expose the frozen measurement boundary. They are useful when auditing the submission, but a non-zero exit or a `blocked` status is expected:

```bash
.venv/bin/python -m src.analysis.run validation sample --check
.venv/bin/python -m src.analysis.run validation select --task sentiment --check
.venv/bin/python -m src.analysis.run validation select --task stance,frames --check
.venv/bin/python -m src.analysis.run validation evaluate --check
.venv/bin/python -m src.analysis.run validation reserve --check
.venv/bin/python -m src.analysis.run predictions --check
```

The fail-closed message says that the measurement selection is stale or incomplete. Starting a new cycle would require `validation sample --build --revision revision_20260918`, where `revision_20260918` is only an example of a new team-approved identifier. The repository intentionally has no current `measurement_cycle.json`. Do not bypass this guard or create a new revision during marking.

Hypothesis checks return a JSON status rather than silently fabricating a result:

```bash
.venv/bin/python -m src.analysis.run hypothesis --id H1 --check
.venv/bin/python -m src.analysis.run hypothesis --id H2 --check
.venv/bin/python -m src.analysis.run hypothesis --id H3 --check
.venv/bin/python -m src.analysis.run hypothesis --id H4 --check
```

The missing measurement-validity and Reddit thread-relevance prerequisites are recorded in the returned JSON. The fallback bundle keeps all six corresponding endpoints `unavailable`. Neither state means that the hypothesis is false.

## Research question and scope

The central question is: **How are child-safety, privacy or surveillance, governance, and circumvention or autonomy frames distributed across the interaction networks of Reddit, YouTube and Bluesky around UK and Australian age-assurance policy events?**

The project treats each platform as a separate research setting. It compares derived frame and structural summaries only after platform-specific analysis. It never merges raw users, pseudonymous identities or graphs across platforms.

### Frames

The frozen annotation codebook uses five non-exclusive frame labels:

- `policy_assurance`
- `child_safety`
- `privacy_surveillance`
- `governance_platform_responsibility`
- `circumvention_censorship_autonomy`

Frame labels can overlap. Their shares therefore do not sum to 100%. Stance and sentiment are separate variables: negative language does not necessarily oppose a policy, and support does not necessarily sound positive.

### Planned hypotheses

These are the pre-specified endpoints. Their current status comes from the manifests above, not from the descriptions here:

| Endpoint | Planned comparison |
|---|---|
| H1-R | Frame segregation across Reddit interaction communities |
| H1-B | Frame segregation across Bluesky interaction communities |
| H2-P | Privacy or surveillance framing in the Australian implementation window versus the Australian legislative window |
| H2-C | Circumvention, censorship or autonomy framing in the same Australian comparison |
| H3 | Frame diversity of high-betweenness Reddit brokers versus matched lower-betweenness authors |
| H4 | Alignment between YouTube video metadata frames and audience frames, comparing source types |

### Core case windows

The core windows are defined in `config/case_windows.csv` and use half-open Coordinated Universal Time (UTC) intervals, meaning the start is included and the end is excluded:

| Case | UTC interval | Purpose |
|---|---|---|
| `AU_LEGISLATION` | 2024-11-10 to 2025-01-11 | Australian legislative debate |
| `UK_ENFORCEMENT_CLUSTER` | 2025-06-14 to 2025-08-26 | UK Online Safety Act enforcement |
| `AU_IMPLEMENTATION` | 2025-11-10 to 2026-01-11 | Australian social-media minimum-age implementation |

The broader event registry in `config/events.csv` also records E1 to E5, including US and UK placebo or legislative events and the Australian age-restricted-materials event. Platform-specific analysis uses the event and case identifiers recorded in each manifest.

## Data sources and ethical boundary

The project uses public, platform-specific snapshots. YouTube Data API means the YouTube Data application programming interface:

| Source | Collection and analysis role | Documentation |
|---|---|---|
| Reddit via Arctic Shift | Posts and complete available comment trees across six subreddits and the three core windows | `docs/age_gate_manual_annotation_instructions.md`, `docs/data_handling.md`, `src/platforms/reddit/` |
| YouTube Data API v3 | Selected video metadata, comments and replies across the event registry | `docs/youtube/collection_qa.md`, `docs/youtube/cleaning_decisions.md`, `docs/youtube/data_dictionary.md` |
| Bluesky public AT Protocol interfaces | Search posts, thread replies and platform-specific interaction tables | `docs/bluesky/README.md`, `docs/bluesky/collection_and_cleaning_log.md` |
| Google Trends | Contextual search series for `VPN`, `free VPN` and `age verification` | `docs/google_trends/README.md`, `notebooks/google_trends/preprocessing.ipynb` |

The data layers have different access rules:

- `data/raw/` contains private collection inputs, source identifiers and acquisition provenance. Never submit or publish it.
- `data/interim/` contains restricted cleaning and validation tables. Treat it as research data.
- `data/processed/` contains pseudonymised but still restricted text, actor hashes and network relations.
- `data/analysis/` contains frozen manifests, labels, derived tables, figures and report-safe outputs. It can still include restricted annotation or model-support files.
- `data/sample/` contains the approved submission exports. The Reddit sample has 500 data rows. The YouTube sample contains pseudonymised documents, interactions and author-video rows.

The local working copy may contain `data.zip`, but Git ignores it by design. That archive includes raw and processed data and is not a safe submission artifact. The approved frozen bundle is supplied through OneDrive and referenced in the Canvas submission. Download that bundle for marking, keep it outside Git, and do not replace it with a live collection or partial pilot dataset. Never include `.env`, API keys, access tokens, passwords, raw author fields, platform identifiers or private annotation credentials.

All platform identifiers are kept separate. Pseudonymisation supports within-platform structure analysis; it does not establish identity, residence, age, persuasion or offline behaviour. Subreddit, video-event and language tags are sampling or context variables, not proof of a user's location or demographic.

## Repository map

| Path | Purpose |
|---|---|
| `ASSESSMENT_SPEC.md` | Assessment requirements and constraints |
| `verify_me_not_analysis_plan_revised.md` | Frozen scope authority and analysis-plan contract |
| `config/events.csv` | Event registry, dates and treatment or placebo roles |
| `config/case_windows.csv` | Three core analysis windows |
| `config/youtube/` | YouTube queries, lexicons and frozen seed-video selection |
| `config/bluesky/` | Bluesky collection configuration |
| `src/platforms/reddit/` | Arctic Shift collection and Reddit CLI |
| `src/platforms/youtube/` | YouTube discovery, collection, preparation, QA, edges and sample export |
| `src/platforms/bluesky/` | Bluesky collection, enrichment, normalisation and pseudonymisation |
| `src/shared/` | Shared identifiers, event windows, masking and time helpers |
| `src/analysis/` | Frozen scope, validation, descriptive, network, topic, hypothesis and report stages |
| `tests/` | Offline unit and contract tests |
| `data/analysis/` | Frozen derived artifacts and manifests, supplied through OneDrive and Canvas |
| `data/sample/` | Sanitised reviewer or submission samples, supplied through OneDrive and Canvas |
| `docs/` | Collection logs, data dictionaries, coding rules and retention policy |
| `notebooks/` | Supporting Google Trends, analysis and YouTube QA notebooks |
| `reports/` | Excluded from this repository; the assessed report and presentation are submitted through Canvas |
| `historical_reports/` | Excluded from this repository; retained only as local design references |

## Pipeline stages and entry points

The stages are explicit. Each stage writes a manifest with source or code checksums where the output is part of the frozen evidence.

### Collection and preparation

Live collection requires network access, current platform terms and private credentials. Do not run these commands to inspect the submission:

- Reddit: `python -m src.platforms.reddit.cli`. Use `--self-test` or `--plan-only` for offline inspection. The collector targets the subreddits and queries defined in `src/platforms/reddit/collector.py` and `src/platforms/reddit/cli.py`.
- YouTube: follow the documented order in `docs/youtube/report_notes.md` and the module help for `discover`, `screen`, `collect`, `qa`, `prepare`, `edges` and `sample`. The frozen video list is `config/youtube/seed_videos.csv`.
- Bluesky: follow the order in `docs/bluesky/README.md`: collect posts, collect the baseline, enrich threads, repair capped threads, collect profiles and follows, normalise tables, build analysis tables and frames, then anonymise.

For the frozen Reddit preparation entry point, the command shape is:

```bash
.venv/bin/python -m src.prepare_corpus \
  --raw-root data/reddit_age_gate_full/arctic_shift_20260915_111812 \
  --out-root data/processed/reddit_age_gate \
  --submission-sample data/sample/reddit_age_gate_sample.csv
```

Do not rerun this command over the frozen inputs during marking. Preprocessing generates run-scoped pseudonyms and can change joins or output checksums. `--allow-partial` and `--skip-topics` are not valid substitutes for the intended full run.

### Frozen analysis orchestration

`src.analysis.run` is the single entry point for the analysis contracts:

| Command family | Role |
|---|---|
| `scope`, `populations`, `codebook` | Validate the frozen scope, population rules and annotation codebook |
| `descriptive` | Validate coverage, concentration and timeline artifacts |
| `networks build`, `structure`, `temporal` | Validate graph tables, topology, communities, roles and cascades |
| `validation sample`, `select`, `evaluate`, `reserve` | Manage the immutable human-measurement lifecycle; currently fail closed at the frozen boundary |
| `hypothesis --id H1` through `H4` | Gate the six PRIMARY-6 endpoints; blocked prerequisites are reported explicitly |
| `topics` | Validate exploratory non-negative matrix factorisation (NMF) and BERTopic sensitivity outputs |
| `extended timing`, `prediction` | Validate approval-gated extensions |
| `robustness` | Validate the predeclared specification matrix; current status is blocked by prerequisites |
| `predictions` | Validate full-corpus measurement labels; current status is blocked by the measurement cycle |
| `report` | Validate the traceable report bundle and generated evidence matrix |

The report-ready fallback entry point is `src.analysis.fallback_analysis`. It uses adjudicated evaluation labels, platform-specific clusters and normalised Hájek inverse-inclusion weighting. Its outputs are descriptive or exploratory and deliberately keep H1-B, H1-R, H2-C, H2-P, H3 and H4 unavailable when the confirmatory prerequisites are not met.

### Human annotation artifacts

Human coding instructions are in [docs/h1_h3_coding_instructions.md](docs/h1_h3_coding_instructions.md), [docs/h4_comment_coding_instructions.md](docs/h4_comment_coding_instructions.md) and [docs/age_gate_manual_annotation_instructions.md](docs/age_gate_manual_annotation_instructions.md). The frozen validation design contains development, evaluation and reserve splits with 600, 600 and 300 records. Coder files, coordinator labels, agreement reports and result manifests live under `data/analysis/validation/`, `data/analysis/annotation/`, `data/analysis/h1h3_human/` and `data/analysis/h4_human/`.

The stored human-sample H1 to H3 and H4 outputs are exploratory or fallback evidence. They do not replace the formal `src.analysis.hypotheses` gates. H3 requires complete 1:3 matched sets; a result with fewer complete sets is `not_estimable`, not a null finding. H4 excludes mixed-source channels rather than relabelling them.

## Network definitions

The project analyses interaction objects separately by platform. The table describes the intended interpretation of each frozen graph family:

| Graph | Nodes | Edge meaning | Direction or weight |
|---|---|---|---|
| Reddit actor attention | Pseudonymised authors | A replies to content by B | Directed A → B, repeated replies weighted |
| Reddit author-subreddit | Authors and subreddits | An author participates in a subreddit | Bipartite, document-count weight |
| YouTube actor attention | Pseudonymised commenters | A replies to B or to a thread author | Directed, repeated replies weighted |
| YouTube commenter-video | Commenters and selected videos | A comments on a video | Bipartite, comment-count weight |
| Bluesky actor attention | Pseudonymised accounts | A replies to B | Directed, observed reply multiplicity |
| Bluesky actor cascade | Pseudonymised accounts | Observed reply cascade relation | Directed, kept separate from attention interpretation |
| Message trees | Documents | A document is a reply to its parent | Directed, structural depth and breadth |

Community structure, PageRank, betweenness, participation and cascade measures describe observed network position. They do not prove influence, exposure, persuasion, ideology or causality. The frozen report does not name or target individual accounts.

## Reports and presentation

The assessed Word report and presentation are submitted separately through Canvas. They are not required to run the tests or validate the frozen analysis artifacts in this repository. The `src/analysis/report.py` module remains in the codebase because it validates the report-facing analysis bundle supplied with the frozen data.

## How to interpret the results

The report separates four evidence classes:

- **Descriptive**: counts, coverage, exclusions, frame profiles and observed timelines within the frozen sample
- **Exploratory**: topology, community, centrality and topic diagnostics that support bounded interpretation
- **Unavailable or blocked**: an endpoint whose design, measurement or source-audit prerequisite is not satisfied
- **Approval-required**: an extension that has no current approval receipt or assessed output

The fallback bundle currently records five report-safe finding cards, but it does not create a complete confirmatory claim. In particular:

- sample activity is not public-opinion prevalence;
- `chosen / pool` is not a valid population inclusion probability after subreddit quotas, author deduplication and thread caps;
- H2 remains an unweighted fixed-quota exploratory comparison unless the sampling design is rebuilt;
- automated predictions are not ground truth when the measurement gate fails;
- unavailable H2 must not be read as evidence of no change;
- cross-platform centrality, sentiment and frame shares are not pooled into one ranking;
- event association is not causal policy effectiveness;
- topic stability does not guarantee semantic validity.

The report-facing outputs remove raw text and direct identifiers. See `data/analysis/fallback/analysis_manifest.json` and `data/analysis/report/report_manifest.json` for source registries, checksums, output checksums and the exact status recorded at freeze time.

## Troubleshooting

### Imports fail

Run from the repository root and use `.venv/bin/python`. Recreate the environment from `requirements.txt` if a pinned package or spaCy model is missing.

### A validator says data is missing

The data directories are intentionally outside Git. Download the approved frozen bundle from the OneDrive link in the Canvas submission and extract it at the repository root so paths such as `data/analysis/report/report_manifest.json`, `data/analysis/fallback/analysis_manifest.json` and `data/processed/youtube/documents.parquet` exist. Do not substitute live downloads or partial pilot data.

### A checksum or freshness check fails

Treat `source checksum changed`, `analysis configuration changed` and `analysis code changed` as freeze failures. Do not delete the guard or edit a manifest by hand. Restore the exact frozen inputs, or start a new documented revision with explicit team approval.

### A measurement command says the selection is stale

That is the expected fail-closed boundary in this submission. Read the stored `report_manifest.json`, `primary6.json` and `reserve_assessment.json`; do not run a new validation cycle during marking.

### Report or presentation files are unavailable

The assessed report and presentation are submitted through Canvas, not this code repository. They are not required for the unit tests or read-only frozen-analysis checks.

## Submission checklist

Before creating the hand-in archive:

1. Include the tracked source, configuration, documentation and tests in the code repository.
2. Download the approved frozen data bundle from OneDrive using the link recorded in the Canvas submission.
3. Submit the repository link, report, presentation and OneDrive data-bundle reference through Canvas.
4. Keep `data/raw/`, `.venv/`, `.env` and credentials out of the Git repository and public links.
5. Run `.venv/bin/python -m pytest tests -q` and record the result.
6. Run the read-only core validators and retain their JSON output with the submission notes.
7. Confirm that `data/analysis/report/report_manifest.json` and `data/analysis/fallback/analysis_manifest.json` still validate after extracting the OneDrive bundle.
8. Explain any unavailable or blocked endpoint as a deliberate evidence boundary, never as a successful null result.

The data-handling and retention rules in [docs/data_handling.md](docs/data_handling.md) apply after submission. Do not publicly upload raw or processed social-media data.
