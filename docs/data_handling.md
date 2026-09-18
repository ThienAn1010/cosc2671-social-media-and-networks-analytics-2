# Data handling

Install the declared Python dependencies from `requirements.txt`, then install the spaCy model and NLTK resources required by preprocessing:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/python -c 'import nltk; nltk.download("stopwords"); nltk.download("vader_lexicon")'
```

The files under `data/raw/` are private working inputs. They contain the original Reddit author fields, Reddit-derived IDs, and acquisition provenance, and must not be submitted, committed, or uploaded. `acquisition_routes.jsonl` records every distinct subreddit/query/endpoint route for a deduplicated record.

The normal processed outputs remove raw author fields but still contain run-scoped actor hashes, Reddit-derived network relations, and analysis text. Treat them as restricted research data unless the teaching team explicitly approves their release.

Create the hand-in sample with:

```bash
python3 src/prepare_corpus.py \
  --raw-root data/raw/<complete-run> \
  --out-root data/processed \
  --submission-sample data/submission/analysis_sample.csv
```

The submission export contains a deterministic representative sample, run-scoped actor hashes, relabelled sample node IDs, relation fields only when the target is also sampled, and topic tokens. It removes raw text, titles, bodies, URLs, Reddit record IDs, and raw author metadata. The HMAC salt is generated per preprocessing run and is never written to disk.

Retain raw and normal processed data only for the period required by the assessment and institutional ethics policy. Delete local raw data after the submitted outputs and reproducibility record have been verified.
