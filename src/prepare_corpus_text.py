# Student name: Nguyen Hoang Thien An
# Student ID: s3825455
# Assignment 1 - Social Media and Networks Analytics

"""Shared text cleaning and enrichment for the Reddit preparation pipeline.

The functions here produce separate light, sentiment, and topic views, then
add language, emoji, sentiment, relevance, and topic metadata while retaining
exclusion diagnostics.  Record assembly and schema validation live in
:mod:`src.prepare_corpus_records`.
"""

from __future__ import annotations

import hashlib
import html
import importlib.metadata
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import emoji
import ftfy
import spacy
from lingua import Language, LanguageDetectorBuilder
from nltk import data as nltk_data
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Small English spaCy model, used only for lemmas.
SPACY_MODEL_NAME = "en_core_web_sm"
# Language detection below this confidence is marked uncertain rather than trusted.
LANGUAGE_CONFIDENCE_THRESHOLD = 0.80
SHORT_UNCERTAIN_TEXT_CHARS = 24
# SimHash settings; very short texts are skipped because they collide by chance.
NEAR_DUPLICATE_HAMMING_THRESHOLD = 3
NEAR_DUPLICATE_MIN_TOKENS = 5
REQUIRED_NLTK_RESOURCES = {
    "corpora/stopwords": "python -m nltk.downloader stopwords",
    "sentiment/vader_lexicon.zip": "python -m nltk.downloader vader_lexicon",
}

# Study vocabulary is intentionally interpretable: these labels are screening
# strata, not final human-coded claims.
AGE_POLICY_TERMS = [
    "age verification",
    "age-verification",
    "age assurance",
    "age-assurance",
    "age check",
    "age checks",
    "age gate",
    "age-gating",
    "age estimation",
    "age-estimation",
    "age inference",
    "under 16",
    "under-16",
    "minimum age",
    "social media ban",
]
CHILD_SAFETY_TERMS = [
    "child safety",
    "children online",
    "online safety",
    "online harms",
    "protect children",
    "protect minors",
    "minor safety",
    "grooming",
]
PRIVACY_SURVEILLANCE_TERMS = [
    "privacy",
    "surveillance",
    "data collection",
    "facial recognition",
    "facial age estimation",
    "biometric",
    "biometrics",
    "digital id",
    "digital identity",
    "identity document",
    "anonymous",
    "anonymity",
]
GOVERNANCE_TERMS = [
    "platform accountability",
    "platform responsibility",
    "content moderation",
    "age-appropriate design",
    "age appropriate design",
    "parental controls",
    "regulation",
    "regulatory",
    "legislation",
    "bill",
]
CIRCUMVENTION_AUTONOMY_TERMS = [
    "vpn",
    "circumvent",
    "circumvention",
    "bypass",
    "false age",
    "free speech",
    "censorship",
    "digital rights",
]
AGE_CONTEXT_TERMS = [
    "age",
    "minor",
    "minors",
    "child",
    "children",
    "teen",
    "teenager",
    "teenagers",
    "social media",
    "online platform",
]
ALL_RELEVANCE_TERMS = (
    AGE_POLICY_TERMS
    + CHILD_SAFETY_TERMS
    + PRIVACY_SURVEILLANCE_TERMS
    + GOVERNANCE_TERMS
    + CIRCUMVENTION_AUTONOMY_TERMS
)
RELEVANCE_GROUPS = [
    ("age_policy", AGE_POLICY_TERMS),
    ("child_safety", CHILD_SAFETY_TERMS),
    ("privacy_surveillance", PRIVACY_SURVEILLANCE_TERMS),
    ("governance", GOVERNANCE_TERMS),
    ("circumvention_autonomy", CIRCUMVENTION_AUTONOMY_TERMS),
]
# Known automated posters, flagged rather than deleted.
BOT_AUTHORS = {"automoderator"}
PROMO_KEYWORDS = [
    "ad",
    "affiliate",
    "coupon",
    "deal",
    "discount",
    "dm me",
    "free trial",
    "limited offer",
    "promo",
    "referral",
    "sale",
    "sponsored",
    "subscribe now",
    "use my code",
]
# Kept out of the stopword list: removing them would invert the meaning of a sentence.
NEGATION_KEEPWORDS = {"no", "not", "nor", "never"}
# Also kept, because standard stopword lists would strip the study vocabulary.
DOMAIN_KEEPWORDS = {
    "age",
    "assurance",
    "biometric",
    "biometrics",
    "bypass",
    "censorship",
    "child",
    "circumvention",
    "digital",
    "gating",
    "grooming",
    "identity",
    "minor",
    "minors",
    "online",
    "platform",
    "privacy",
    "regulation",
    "surveillance",
    "teen",
    "teenager",
    "teenagers",
    "verification",
    "vpn",
}
# Archive placeholders left behind when content was taken down.
REMOVED_MARKERS = {"[removed]", "removed"}
DELETED_MARKERS = {"[deleted]", "deleted", "deleted by user"}

# Precompiled once at import; these patterns run over the full corpus.
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
WHITESPACE_RE = re.compile(r"\s+")
USER_RE = re.compile(r"(?<!\w)(?:/?u/|u/)([A-Za-z0-9_-]+)", re.IGNORECASE)
SUBREDDIT_RE = re.compile(r"(?<!\w)(?:/?r/|r/)([A-Za-z0-9_-]+)", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z]+(?:-[a-z]+)?|\d+")
# A number alone is not substantive text for language, sentiment or topic review.
SUBSTANTIVE_RE = re.compile(r"[^\W\d_]", re.UNICODE)
PROMO_PATTERNS = [
    re.compile(rf"(?<![a-z0-9]){re.escape(keyword).replace(r'\\ ', r'\\s+')}(?![a-z0-9])", re.IGNORECASE)
    for keyword in PROMO_KEYWORDS
]
RELEVANCE_PATTERNS = {
    term: re.compile(rf"(?<![a-z0-9]){re.escape(term).replace(r'\\ ', r'\\s+')}(?![a-z0-9])", re.IGNORECASE)
    for term in ALL_RELEVANCE_TERMS
}
AGE_CONTEXT_PATTERNS = [
    re.compile(rf"(?<![a-z0-9]){re.escape(term).replace(r'\ ', r'\s+')}(?![a-z0-9])", re.IGNORECASE)
    for term in AGE_CONTEXT_TERMS
]
# Expanded before scoring so negation survives tokenisation (can't -> cannot).
CONTRACTIONS = [
    (re.compile(r"\bwon't\b", re.IGNORECASE), "will not"),
    (re.compile(r"\bcan't\b", re.IGNORECASE), "cannot"),
    (re.compile(r"\bshan't\b", re.IGNORECASE), "shall not"),
    (re.compile(r"\bain't\b", re.IGNORECASE), "is not"),
    (re.compile(r"n't\b", re.IGNORECASE), " not"),
]
# Records with no usable text; skipped by the enrichment pass but still kept as rows.
STRUCTURAL_EXCLUSION_STATUSES = {"deleted", "removed", "empty", "invalid_schema", "invalid_timestamp"}


# Look up an installed package version for the manifest, or 'missing' if absent.
def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


# Hash a file in 1 MB chunks so large inputs do not have to fit in memory.
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Rewrite negated contractions to full words, preserving all-caps emphasis.
def expand_negation_contractions(text: str) -> str:
    expanded = text
    for pattern, replacement in CONTRACTIONS:
        expanded = pattern.sub(lambda match: replacement.upper() if match.group(0).isupper() else replacement, expanded)
    return expanded


# Light view: repair mojibake, normalise Unicode, unescape HTML, flatten links, collapse spaces.
def clean_text_light(value: Any) -> str:
    if value is None:
        return ""
    text = ftfy.fix_text(str(value))
    text = unicodedata.normalize("NFC", text)
    text = html.unescape(text)
    # Keep the anchor text of a Markdown link and drop the URL.
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    return WHITESPACE_RE.sub(" ", text).strip()


# Drop lines starting with '>' so a commenter is not scored on text they quoted.
def remove_quote_blocks(value: Any) -> str:
    if value is None:
        return ""
    return "\n".join(line for line in str(value).splitlines() if not line.lstrip().startswith(">"))


# Sentiment view: keeps case, punctuation and emoji because VADER reads them as intensity.
def clean_text_for_sentiment(value: Any) -> str:
    text = clean_text_light(remove_quote_blocks(value))
    if not text:
        return ""
    text = expand_negation_contractions(text)
    # Replace rather than delete, so sentence structure survives for the model.
    text = URL_RE.sub("<URL>", text)
    text = USER_RE.sub("<USER>", text)
    text = SUBREDDIT_RE.sub("<SUBREDDIT>", text)
    return WHITESPACE_RE.sub(" ", text).strip()


# Topic view starts from the light view with quotes removed; lemmatisation happens later.
def clean_text_for_topic(value: Any) -> str:
    return clean_text_light(remove_quote_blocks(value))


# Count emoji and record their names, used as a descriptive flag.
def emoji_metadata(text: str) -> tuple[bool, int, str]:
    labels = [emoji.demojize(entry["emoji"]).strip(":") for entry in emoji.emoji_list(text)]
    return bool(labels), len(labels), "|".join(labels)


# Preload all language models once; per-call construction would be far slower.
def build_language_detector():
    return LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()


# Names of every language the detector can return, recorded in the manifest.
def language_set_names() -> list[str]:
    return [language.name.lower() for language in Language.all()]


# Return top language plus confidence, marking low-confidence results uncertain.
def detect_language(detector, text: str) -> dict[str, Any]:
    if not text.strip():
        return {
            "language": "",
            "language_confidence": 0.0,
            "is_english": False,
            "is_language_uncertain": False,
        }
    confidence_values = detector.compute_language_confidence_values(text)
    if not confidence_values:
        return {
            "language": "unknown",
            "language_confidence": 0.0,
            "is_english": False,
            "is_language_uncertain": True,
        }
    top = confidence_values[0]
    confidence = float(top.value)
    return {
        "language": str(top.language.name).lower(),
        "language_confidence": round(confidence, 6),
        "is_english": top.language == Language.ENGLISH,
        # Uncertain is a separate outcome from non-English; the pipeline treats them differently.
        "is_language_uncertain": confidence < LANGUAGE_CONFIDENCE_THRESHOLD,
    }


# Union of the NLTK and scikit-learn lists, minus negation and domain keepwords.
def build_stopwords() -> set[str]:
    return (set(stopwords.words("english")) | set(ENGLISH_STOP_WORDS)) - NEGATION_KEEPWORDS - DOMAIN_KEEPWORDS


# Load spaCy with NER, parser and text categoriser disabled; only lemmas are needed.
def load_spacy_model():
    try:
        return spacy.load(SPACY_MODEL_NAME, disable=["ner", "parser", "textcat"])
    except OSError as exc:
        try:
            model_module = __import__(SPACY_MODEL_NAME)
            return model_module.load(disable=["ner", "parser", "textcat"])
        except Exception:
            raise SystemExit(
                f"missing spaCy model {SPACY_MODEL_NAME}. install with: python -m spacy download {SPACY_MODEL_NAME}"
            ) from exc


# Fail early with the exact download command if an NLTK resource is missing.
def ensure_nltk_resources() -> dict[str, str]:
    present: dict[str, str] = {}
    for resource, install_cmd in REQUIRED_NLTK_RESOURCES.items():
        try:
            nltk_data.find(resource)
        except LookupError as exc:
            raise SystemExit(f"missing NLTK resource {resource}. install with: {install_cmd}") from exc
        present[resource] = "present"
    return present


# VADER baseline scorer, compared against RoBERTa in the notebook.
def build_vader_analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


# VADER's standard +/-0.05 compound cutoffs for three-class labels.
def sentiment_label_from_compound(compound: float) -> str:
    if compound <= -0.05:
        return "negative"
    if compound >= 0.05:
        return "positive"
    return "neutral"


# Lowercase, whitespace-collapsed form used only for placeholder detection.
def normalize_for_status(text: str) -> str:
    return clean_text_light(text).strip().lower()


# Classify a record as eligible, deleted, removed or empty from its text alone.
def text_presence_status(*parts: str) -> tuple[str, str]:
    normalized_parts = [normalize_for_status(part) for part in parts if part is not None]
    non_empty_parts = [part for part in normalized_parts if part]
    if not non_empty_parts:
        return "empty", "no text content"
    if all(part in DELETED_MARKERS for part in non_empty_parts):
        return "deleted", "text marked deleted"
    if all(part in REMOVED_MARKERS for part in non_empty_parts):
        return "removed", "text marked removed"
    combined = " ".join(part for part in non_empty_parts if part not in DELETED_MARKERS | REMOVED_MARKERS).strip()
    if not combined:
        return "empty", "no substantive text after archive markers"
    return "eligible", ""


# SHA-256 of the text, used for exact-duplicate detection and caching.
def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Derive UTC date, ISO week and month from the epoch, flagging unusable timestamps.
def timestamp_fields(created_utc: Any) -> tuple[bool, dict[str, Any], str]:
    try:
        epoch = int(created_utc)
    except (TypeError, ValueError):
        return False, {
            "created_utc": None,
            "datetime_utc": "",
            "date": "",
            "week": "",
            "month": "",
        }, "created_utc missing or not integer"

    dt = datetime.fromtimestamp(epoch, timezone.utc)
    iso_year, iso_week, _ = dt.isocalendar()
    return True, {
        "created_utc": epoch,
        "datetime_utc": dt.isoformat(),
        "date": dt.date().isoformat(),
        "week": f"{iso_year}-W{iso_week:02d}",
        "month": dt.strftime("%Y-%m"),
    }, ""


# Topic tokens for a single text; strips URLs and mentions before lemmatising.
def topic_tokens(text: str, nlp, stopword_set: set[str]) -> str:
    if not text:
        return ""
    prepared = SUBREDDIT_RE.sub(" ", USER_RE.sub(" ", URL_RE.sub(" ", text)))
    return topic_tokens_from_doc(nlp(prepared), stopword_set)


# Keep lemmatised tokens, retaining common age thresholds because 13/16/18 are
# analytically meaningful in this study while unrelated numbers remain noise.
def topic_tokens_from_doc(doc, stopword_set: set[str]) -> str:
    tokens: list[str] = []
    for token in doc:
        if token.is_space or token.is_punct or token.like_url:
            continue
        raw = token.text.strip().lower()
        lemma = token.lemma_.strip().lower()
        value = lemma or raw
        if token.like_num and value not in {"13", "16", "18"}:
            continue
        if not value or not TOKEN_RE.fullmatch(value):
            continue
        if value in stopword_set and value not in NEGATION_KEEPWORDS and value not in DOMAIN_KEEPWORDS:
            continue
        tokens.append(value)
    return " ".join(tokens)


# Promotional keywords present in the text, one input to the spam flag.
def promo_hits(text: str) -> list[str]:
    return [keyword for keyword, pattern in zip(PROMO_KEYWORDS, PROMO_PATTERNS) if pattern.search(text)]


# Assign all supported study frames and retain one primary frame for legacy consumers.
def classify_relevance_detail(text: str) -> tuple[str, list[str], list[str]]:
    hits = [term for term, pattern in RELEVANCE_PATTERNS.items() if pattern.search(text)]
    has_age_context = any(pattern.search(text) for pattern in AGE_CONTEXT_PATTERNS)
    frames: list[str] = []
    for level, terms in RELEVANCE_GROUPS:
        level_hits = [term for term in hits if term in terms]
        if level_hits and (level == "age_policy" or has_age_context):
            frames.append(level)
    return (frames[0] if frames else "unrelated"), (hits if frames else []), frames


# Backward-compatible two-value view; use classify_relevance_detail for analysis outputs.
def classify_relevance(text: str) -> tuple[str, list[str]]:
    level, terms, _ = classify_relevance_detail(text)
    return level, terms


# Add tokens, language, VADER scores and study-frame screening to every row.
def enrich_rows(
    rows: list[dict[str, Any]],
    nlp,
    detector,
    stopword_set: set[str],
    batch_size: int = 256,
    vader_analyzer: SentimentIntensityAnalyzer | None = None,
) -> list[dict[str, Any]]:
    active_indexes = [
        index for index, row in enumerate(rows) if str(row.get("exclusion_status") or "") not in STRUCTURAL_EXCLUSION_STATUSES
    ]
    prepared_texts = [
        SUBREDDIT_RE.sub(" ", USER_RE.sub(" ", URL_RE.sub(" ", str(rows[index].get("text_topic") or rows[index].get("text_light") or ""))))
        for index in active_indexes
    ]
    token_cache: dict[str, str] = {}
    language_cache: dict[str, dict[str, Any]] = {}
    # nlp.pipe batches the spaCy work; running it per row would be far slower.
    docs = nlp.pipe(prepared_texts, batch_size=batch_size) if nlp is not None else [None] * len(prepared_texts)

    for row in rows:
        text_light = str(row.get("text_light") or "")
        text_topic = str(row.get("text_topic") or text_light)
        exclusion_status = str(row.get("exclusion_status") or "")
        row["char_count"] = len(text_light)
        row["has_emoji"] = False
        row["emoji_count"] = 0
        row["emoji_labels"] = ""
        row["tokens_topic"] = ""
        row["token_count"] = 0
        row["is_low_information"] = True
        row["is_url_only"] = bool(URL_RE.sub("", text_topic).strip() == "" and URL_RE.search(text_topic))
        row["is_no_substantive_text"] = not bool(SUBSTANTIVE_RE.search(URL_RE.sub(" ", text_topic)))
        row["language"] = ""
        row["language_confidence"] = 0.0
        row["is_english"] = False
        row["is_language_uncertain"] = False
        row["sentiment_neg"] = 0.0
        row["sentiment_neu"] = 0.0
        row["sentiment_pos"] = 0.0
        row["sentiment_compound"] = 0.0
        row["sentiment_label"] = ""
        row["relevance_level"] = "unrelated"
        row["relevance_frames"] = ""
        row["relevance_terms"] = ""
        row["age_assurance_related"] = False
        # Placeholder rows keep their default values and are skipped, not dropped.
        if exclusion_status in STRUCTURAL_EXCLUSION_STATUSES:
            continue
        row["has_emoji"], row["emoji_count"], row["emoji_labels"] = emoji_metadata(str(row.get("text_sentiment") or ""))
        relevance_level, relevance_terms, relevance_frames = classify_relevance_detail(text_topic)
        row["relevance_level"] = relevance_level
        row["relevance_frames"] = "|".join(relevance_frames)
        row["relevance_terms"] = "|".join(relevance_terms)
        row["age_assurance_related"] = bool(relevance_frames)
        if vader_analyzer is not None:
            sentiment_scores = vader_analyzer.polarity_scores(str(row.get("text_sentiment") or ""))
            row["sentiment_neg"] = round(float(sentiment_scores.get("neg", 0.0)), 6)
            row["sentiment_neu"] = round(float(sentiment_scores.get("neu", 0.0)), 6)
            row["sentiment_pos"] = round(float(sentiment_scores.get("pos", 0.0)), 6)
            row["sentiment_compound"] = round(float(sentiment_scores.get("compound", 0.0)), 6)
            row["sentiment_label"] = sentiment_label_from_compound(row["sentiment_compound"])

    for index, prepared_text, doc in zip(active_indexes, prepared_texts, docs):
        row = rows[index]
        sentiment_text = str(row.get("text_sentiment") or "")
        tokens_topic = ""
        if doc is not None:
            # Cache by text, since duplicate posts are common and lemmatising is the slow step.
            cached_tokens = token_cache.get(prepared_text)
            if cached_tokens is None:
                cached_tokens = topic_tokens_from_doc(doc, stopword_set)
                token_cache[prepared_text] = cached_tokens
            tokens_topic = cached_tokens
        row["tokens_topic"] = tokens_topic
        row["token_count"] = len(tokens_topic.split()) if tokens_topic else 0
        # Under three tokens carries too little signal to analyse.
        row["is_low_information"] = row["token_count"] < 3

        language_text = str(row.get("text_topic") or "") or sentiment_text
        language_info = language_cache.get(language_text)
        if language_info is None:
            language_info = detect_language(detector, language_text)
            language_cache[language_text] = language_info
        row.update(language_info)

        # Language outcome can downgrade an otherwise eligible record.
        if row["exclusion_status"] == "eligible":
            if row["is_language_uncertain"]:
                row["exclusion_status"] = "language_uncertain"
                row["exclusion_reason"] = f"language={row['language']} confidence={row['language_confidence']}"
            elif not row["is_english"]:
                row["exclusion_status"] = "non_english"
                row["exclusion_reason"] = f"language={row['language']} confidence={row['language_confidence']}"
    return rows
