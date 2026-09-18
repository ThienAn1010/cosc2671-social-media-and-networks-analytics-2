"""Stable, label-independent NMF topic discovery with an explicit naming gate."""

from __future__ import annotations

import itertools
import importlib.util
import re
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import MiniBatchNMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.analysis.artifacts import ANALYSIS_ROOT, REPO_ROOT, relative_path, sha256_file, utc_now_iso, write_json, read_json
from src.analysis.codebook import FRAME_DEFINITIONS
from src.shared.case_windows import REDDIT_CASE_BY_EVENT
from src.shared.masking import mask_text

TOPICS_ROOT = ANALYSIS_ROOT / "topics"
TOPICS_MANIFEST = TOPICS_ROOT / "topics_manifest.json"
TOPIC_COUNTS = (6, 8, 10)
SEEDS = (17, 29, 41)
BERTopic_SEEDS = (17, 29, 41)
TOP_TERMS = 15
PREPROCESSING_VERSION = "tfidf-word-1-2gram-min_df5-max_df0.98-text_raw-mask-v2"
TOPIC_STABILITY_THRESHOLD = 0.70
TOPICS_SCHEMA_VERSION = "topics.v3"
TOPIC_EMBEDDING_VERSION = "tfidf-lsa-v1"
BERTOPIC_REQUIRED_VERSION = "0.17.4"


def _load_documents() -> dict[str, pd.DataFrame]:
    reddit = pd.read_parquet(
        REPO_ROOT / "data" / "processed" / "reddit_age_gate" / "analysis_corpus.parquet",
        columns=["event_id", "date", "text_topic", "schema_valid", "human_only", "is_english", "is_url_only", "is_no_substantive_text"],
    )
    reddit_text = reddit["text_topic"].fillna("").astype(str).str.strip()
    reddit = reddit[
        reddit["schema_valid"].fillna(False).astype(bool)
        & reddit["human_only"].fillna(False).astype(bool)
        & reddit["is_english"].fillna(False).astype(bool)
        & reddit["is_url_only"].fillna(False).eq(False)
        & reddit["is_no_substantive_text"].fillna(False).eq(False)
        & reddit_text.ne("")
    ].copy()
    reddit["scope"] = reddit["event_id"].map(REDDIT_CASE_BY_EVENT)
    reddit["text"] = reddit["text_topic"].map(mask_text)
    reddit = reddit.rename(columns={"date": "day"})[["scope", "day", "text"]]

    bluesky = pd.read_parquet(
        REPO_ROOT / "data" / "processed" / "bluesky" / "posts.parquet",
        columns=["event_window", "day", "text_raw", "is_english", "is_meme", "is_repeat_burst"],
    )
    bluesky = bluesky[
        bluesky["event_window"].isin(["E2", "E3"])
        & bluesky["is_english"].fillna(False).astype(bool)
        & ~bluesky["is_meme"].fillna(False).astype(bool)
        & ~bluesky["is_repeat_burst"].fillna(False).astype(bool)
    ].copy()
    bluesky["text"] = bluesky["text_raw"].fillna("").astype(str).map(mask_text)
    bluesky = bluesky[bluesky["text"].str.strip().ne("")].rename(columns={"event_window": "scope"})[["scope", "day", "text"]]

    youtube = pd.read_parquet(
        REPO_ROOT / "data" / "processed" / "youtube" / "documents.parquet",
        columns=["thing", "event_window", "created_date_utc", "text_clean", "exclusion_status", "is_english"],
    )
    youtube = youtube[
        youtube["thing"].ne("video")
        & youtube["event_window"].isin(["E2", "E3"])
        & youtube["exclusion_status"].eq("eligible")
        & youtube["is_english"].fillna(False).astype(bool)
    ].copy()
    youtube["text"] = youtube["text_clean"].fillna("").astype(str).map(mask_text)
    youtube = youtube[youtube["text"].str.strip().ne("")].rename(columns={"event_window": "scope", "created_date_utc": "day"})[["scope", "day", "text"]]
    documents = {"reddit": reddit, "bluesky": bluesky, "youtube": youtube}
    for platform, frame in documents.items():
        frame.insert(0, "document_ref", [f"{platform}-{index:08d}" for index in range(len(frame))])
    return documents


def _top_terms(model: MiniBatchNMF, names: np.ndarray) -> list[list[str]]:
    return [[str(names[index]) for index in row.argsort()[::-1][:TOP_TERMS]] for row in model.components_]


def _topic_alignment(left: list[list[str]], right: list[list[str]]) -> float:
    if not left or not right:
        return 1.0 if not left and not right else 0.0
    scores = np.asarray(
        [[len(set(a) & set(b)) / len(set(a) | set(b)) if set(a) | set(b) else 1.0 for b in right] for a in left],
        dtype=float,
    )
    rows, columns = linear_sum_assignment(-scores)
    return float(scores[rows, columns].mean()) if len(rows) else 1.0


def _topic_diversity(terms: list[list[str]]) -> float:
    flattened = [term for topic in terms for term in topic]
    return float(len(set(flattened)) / len(flattened)) if flattened else 0.0


def _topic_coherence(matrix, model: MiniBatchNMF, top_terms: int = 10) -> float:
    scores = []
    for weights in model.components_:
        indexes = weights.argsort()[::-1][:top_terms]
        presence = matrix[:, indexes].sign().astype(np.int32)
        cooccurrence = (presence.T @ presence).toarray().astype(float)
        document_frequency = np.asarray(presence.sum(axis=0)).ravel().astype(float)
        pair_scores = [
            np.log((cooccurrence[left, right] + 1) / max(document_frequency[right], 1))
            for left in range(len(indexes))
            for right in range(left + 1, len(indexes))
        ]
        scores.append(float(np.mean(pair_scores)) if pair_scores else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _normalise_metric(values: pd.Series, higher_is_better: bool) -> pd.Series:
    low, high = float(values.min()), float(values.max())
    if high == low:
        return pd.Series(1.0, index=values.index)
    scaled = (values - low) / (high - low)
    return scaled if higher_is_better else 1.0 - scaled


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    share = successes / total
    denominator = 1 + z**2 / total
    centre = (share + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((share * (1 - share) + z**2 / (4 * total)) / total) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _select_topic_count(candidate_metrics: pd.DataFrame) -> int:
    if candidate_metrics.empty:
        raise ValueError("topic model selection needs at least one candidate")
    scored = candidate_metrics.copy()
    scored["selection_score"] = (
        0.40 * _normalise_metric(scored["mean_seed_stability"], True)
        + 0.25 * _normalise_metric(scored["mean_top_term_coherence"], True)
        + 0.20 * _normalise_metric(scored["topic_diversity"], True)
        + 0.15 * _normalise_metric(scored["reconstruction_error"], False)
    )
    candidate_metrics["selection_score"] = scored["selection_score"]
    return int(
        scored.sort_values(
            ["selection_score", "mean_seed_stability", "mean_top_term_coherence", "topic_count"],
            ascending=[False, False, False, True],
        ).iloc[0]["topic_count"]
    )


def _fit_platform(
    platform: str, documents: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if documents.empty:
        raise ValueError(f"no documents available for {platform} topic model")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.98,
        max_features=5_000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(documents["text"])
    names = vectorizer.get_feature_names_out()
    models: dict[tuple[int, int], tuple[MiniBatchNMF, np.ndarray, list[list[str]]]] = {}
    candidate_rows = []
    stability_by_count: dict[int, float] = {}
    terms_by_count: dict[int, dict[int, list[list[str]]]] = {}
    for topic_count in TOPIC_COUNTS:
        for seed in SEEDS:
            model = MiniBatchNMF(
                n_components=topic_count,
                init="nndsvda",
                batch_size=1024,
                max_iter=100,
                random_state=seed,
            )
            weights = model.fit_transform(matrix)
            terms = _top_terms(model, names)
            models[(topic_count, seed)] = (model, weights, terms)
            terms_by_count.setdefault(topic_count, {})[seed] = terms
        pairwise = [
            _topic_alignment(terms_by_count[topic_count][left], terms_by_count[topic_count][right])
            for left, right in itertools.combinations(SEEDS, 2)
        ]
        stability_by_count[topic_count] = float(np.mean(pairwise)) if pairwise else 1.0
        for seed in SEEDS:
            model, weights, terms = models[(topic_count, seed)]
            topic_support = weights.argmax(axis=1)
            candidate_rows.append(
                {
                    "platform": platform,
                    "topic_count": topic_count,
                    "seed": seed,
                    "reconstruction_error": float(model.reconstruction_err_),
                    "mean_top_term_coherence": _topic_coherence(matrix, model),
                    "topic_diversity": _topic_diversity(terms),
                    "mean_seed_stability": stability_by_count[topic_count],
                    "minimum_topic_support": int(np.bincount(topic_support, minlength=topic_count).min()),
                }
            )
    candidates = pd.DataFrame(candidate_rows)
    count_metrics = (
        candidates.groupby(["platform", "topic_count"], as_index=False)
        .agg(
            reconstruction_error=("reconstruction_error", "min"),
            mean_top_term_coherence=("mean_top_term_coherence", "mean"),
            topic_diversity=("topic_diversity", "mean"),
            mean_seed_stability=("mean_seed_stability", "mean"),
        )
        .sort_values("topic_count")
        .reset_index(drop=True)
    )
    selected_count = _select_topic_count(count_metrics)
    selected_seed = int(
        candidates[candidates["topic_count"].eq(selected_count)]
        .sort_values(["reconstruction_error", "seed"], ascending=[True, True])
        .iloc[0]["seed"]
    )
    selected_model, selected_weights, selected_terms = models[(selected_count, selected_seed)]
    assignments = selected_weights.argmax(axis=1)
    support = pd.Series(assignments).value_counts().reindex(range(selected_count), fill_value=0)
    scopes = documents["scope"].drop_duplicates().sort_values().tolist()
    observed_counts = (
        documents.assign(topic_id=assignments)
        .groupby(["scope", "topic_id"], sort=True)
        .size()
        .rename("documents")
    )
    full_index = pd.MultiIndex.from_product([scopes, range(selected_count)], names=["scope", "topic_id"])
    prevalence = observed_counts.reindex(full_index, fill_value=0).reset_index()
    totals = prevalence.groupby("scope")["documents"].transform("sum")
    prevalence["share"] = prevalence["documents"] / totals
    intervals = [_wilson_interval(int(count), int(total)) for count, total in zip(prevalence["documents"], totals)]
    prevalence["share_lower_95"] = [lower for lower, _ in intervals]
    prevalence["share_upper_95"] = [upper for _, upper in intervals]
    prevalence["uncertainty_method"] = "wilson_95_normal_approximation"
    prevalence["platform"] = platform
    prevalence["topic_count"] = selected_count
    prevalence["model_seed"] = selected_seed
    terms = pd.DataFrame(
        [
            {"platform": platform, "topic_id": topic, "rank": rank + 1, "term": term, "topic_count": selected_count, "model_seed": selected_seed}
            for topic, topic_terms in enumerate(selected_terms)
            for rank, term in enumerate(topic_terms)
        ]
    )
    stability_rows = []
    for left_seed, right_seed in itertools.combinations(SEEDS, 2):
        stability_rows.append(
            {
                "platform": platform,
                "topic_count": selected_count,
                "left_seed": left_seed,
                "right_seed": right_seed,
                "mean_top_term_jaccard": _topic_alignment(terms_by_count[selected_count][left_seed], terms_by_count[selected_count][right_seed]),
                "status": "multi_seed_top_term_stability",
            }
        )
    review_rows = []
    for topic_id in range(selected_count):
        for rank, row_index in enumerate(np.argsort(selected_weights[:, topic_id])[::-1][:3], start=1):
            row = documents.iloc[row_index]
            review_rows.append(
                {
                    "platform": platform,
                    "scope": row["scope"],
                    "topic_id": topic_id,
                    "rank": rank,
                    "document_ref": row["document_ref"],
                    "text_for_review": row["text"],
                    "topic_count": selected_count,
                    "model_seed": selected_seed,
                }
            )
    review = pd.DataFrame(review_rows)
    diagnostics = {
        "documents": int(len(documents)),
        "vocabulary": int(len(names)),
        "reconstruction_error": float(selected_model.reconstruction_err_),
        "selected_topic_count": selected_count,
        "selected_seed": selected_seed,
        "minimum_topic_support": int(support.min()),
        "minimum_support_rule": "report support for every selected topic; no topic is silently dropped",
        "representative_review_rows": int(selected_count * 3),
        "model_selection": "0.40 stability + 0.25 coherence + 0.20 diversity + 0.15 inverse reconstruction error; seed minimizes reconstruction error within selected count",
        "candidate_metrics": count_metrics.to_dict(orient="records"),
        "status": "nmf_selected_human_topic_naming_and_representative_review_required",
    }
    return prevalence, terms, pd.DataFrame(stability_rows), review, diagnostics


def _topic_novelty(terms: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    theory_terms = {
        token
        for frame in FRAME_DEFINITIONS
        for token in re.findall(r"[a-z]+", f"{frame['label']} {frame['definition']}".replace("_", " ").casefold())
    }
    stability_by_platform = stability.groupby("platform")["mean_top_term_jaccard"].mean().to_dict()
    rows = []
    for (platform, topic_id), group in terms.groupby(["platform", "topic_id"], sort=True):
        top_terms = group.sort_values("rank")["term"].astype(str).tolist()
        overlap = sorted(set(top_terms) & theory_terms)
        score = float(stability_by_platform.get(platform, 0.0))
        rows.append(
            {
                "platform": platform,
                "topic_id": int(topic_id),
                "top_terms": "|".join(top_terms),
                "theory_term_overlap": "|".join(overlap),
                "mean_topic_stability": score,
                "stability_threshold": TOPIC_STABILITY_THRESHOLD,
                "status": "stable_novel_topic" if score >= TOPIC_STABILITY_THRESHOLD and not overlap else "theory_aligned" if overlap else "unstable_novel_candidate",
            }
        )
    return pd.DataFrame(rows)


def _bertopic_sensitivity(documents: dict[str, pd.DataFrame]) -> pd.DataFrame:
    columns = [
        "platform",
        "seed",
        "topic_id",
        "documents",
        "top_terms",
        "mean_seed_stability",
        "embedding_backend",
        "representation_version",
        "status",
    ]
    unavailable = [
        {
            "platform": platform,
            "seed": seed,
            "topic_id": None,
            "documents": None,
            "top_terms": "",
            "mean_seed_stability": None,
            "embedding_backend": TOPIC_EMBEDDING_VERSION,
            "representation_version": "bertopic-min_size10-count-word-1-2gram",
            "status": "runtime-unavailable",
        }
        for platform in sorted(documents)
        for seed in BERTopic_SEEDS
    ]
    if importlib.util.find_spec("bertopic") is None:
        return pd.DataFrame(unavailable, columns=columns)
    try:
        if package_version("bertopic") != BERTOPIC_REQUIRED_VERSION:
            for row in unavailable:
                row["status"] = "runtime-version-mismatch"
            return pd.DataFrame(unavailable, columns=columns)
    except PackageNotFoundError:
        return pd.DataFrame(unavailable, columns=columns)
    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from umap import UMAP
    except ImportError:
        return pd.DataFrame(unavailable, columns=columns)

    rows = []
    for platform, frame in documents.items():
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.98,
            max_features=5_000,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(frame["text"].astype(str))
        components = min(50, matrix.shape[1] - 1, matrix.shape[0] - 1)
        if components < 2:
            rows.extend(
                {
                    "platform": platform,
                    "seed": seed,
                    "topic_id": None,
                    "documents": None,
                    "top_terms": "",
                    "mean_seed_stability": None,
                    "embedding_backend": TOPIC_EMBEDDING_VERSION,
                    "representation_version": "bertopic-min_size10-count-word-1-2gram",
                    "status": "execution-failed:insufficient-embedding-rank",
                }
                for seed in BERTopic_SEEDS
            )
            continue
        embeddings = TruncatedSVD(n_components=components, random_state=BERTopic_SEEDS[0]).fit_transform(matrix)
        run_topics: dict[int, list[list[str]]] = {}
        run_rows: dict[int, list[dict[str, Any]]] = {}
        for seed in BERTopic_SEEDS:
            try:
                reducer = UMAP(
                    n_neighbors=min(15, max(2, len(frame) - 1)),
                    n_components=5,
                    min_dist=0.0,
                    metric="cosine",
                    random_state=seed,
                )
                clusterer = HDBSCAN(min_cluster_size=10, metric="euclidean", prediction_data=True)
                model = BERTopic(
                    calculate_probabilities=False,
                    embedding_model=None,
                    hdbscan_model=clusterer,
                    min_topic_size=10,
                    representation_model=None,
                    umap_model=reducer,
                    vectorizer_model=CountVectorizer(ngram_range=(1, 2), min_df=5, max_features=5_000),
                    verbose=False,
                )
                assignments, _ = model.fit_transform(frame["text"].astype(str).tolist(), embeddings=embeddings)
                info = model.get_topic_info()
                topic_rows = []
                topic_terms = []
                for topic in info.loc[info["Topic"].ge(0), "Topic"].astype(int):
                    words = [word for word, _ in model.get_topic(topic)[:TOP_TERMS]]
                    topic_terms.append(words)
                    topic_rows.append(
                        {
                            "platform": platform,
                            "seed": seed,
                            "topic_id": topic,
                            "documents": int(sum(value == topic for value in assignments)),
                            "top_terms": "|".join(words),
                            "embedding_backend": TOPIC_EMBEDDING_VERSION,
                            "representation_version": "bertopic-min_size10-count-word-1-2gram",
                            "status": "executed_multi_seed_sensitivity",
                        }
                    )
                if not topic_rows:
                    topic_rows.append(
                        {
                            "platform": platform,
                            "seed": seed,
                            "topic_id": None,
                            "documents": 0,
                            "top_terms": "",
                            "embedding_backend": TOPIC_EMBEDDING_VERSION,
                            "representation_version": "bertopic-min_size10-count-word-1-2gram",
                            "status": "execution-failed:no-topics",
                        }
                    )
                run_topics[seed] = topic_terms
                run_rows[seed] = topic_rows
            except Exception as error:
                rows.append(
                    {
                        "platform": platform,
                        "seed": seed,
                        "topic_id": None,
                        "documents": None,
                        "top_terms": "",
                        "mean_seed_stability": None,
                        "embedding_backend": TOPIC_EMBEDDING_VERSION,
                        "representation_version": "bertopic-min_size10-count-word-1-2gram",
                        "status": f"execution-failed:{type(error).__name__}",
                    }
                )
        pairwise = [
            _topic_alignment(run_topics[left], run_topics[right])
            for left, right in itertools.combinations(sorted(run_topics), 2)
        ]
        mean_stability = float(np.mean(pairwise)) if pairwise else None
        for topic_rows in run_rows.values():
            for row in topic_rows:
                row["mean_seed_stability"] = mean_stability
            rows.extend(topic_rows)
    return pd.DataFrame(rows or unavailable, columns=columns)


def _bertopic_status(frame: pd.DataFrame, platforms: list[str]) -> str:
    expected = {(str(platform), int(seed)) for platform in platforms for seed in BERTopic_SEEDS}
    required = {"platform", "seed", "topic_id", "documents", "status"}
    if frame.empty or not required <= set(frame.columns):
        return "execution-failed"
    observed = {(str(platform), int(seed)) for platform, seed in frame[["platform", "seed"]].itertuples(index=False, name=None)}
    if observed != expected:
        return "execution-failed"
    successful = {
        key
        for key, group in frame.groupby(["platform", "seed"], sort=True)
        if set(group["status"]) == {"executed_multi_seed_sensitivity"}
        and group["topic_id"].notna().any()
        and pd.to_numeric(group["documents"], errors="coerce").fillna(0).gt(0).any()
    }
    if successful == expected:
        return "executed_multi_seed_sensitivity"
    all_statuses = set(frame["status"].astype(str))
    if all_statuses and all_statuses == {"runtime-unavailable"}:
        return "runtime-unavailable"
    if all_statuses and all_statuses == {"runtime-version-mismatch"}:
        return "runtime-version-mismatch"
    return "execution-failed"


def _topic_naming_complete(naming: pd.DataFrame, review: pd.DataFrame) -> bool:
    naming_columns = {"platform", "topic_id", "topic_count", "human_name", "naming_rationale"}
    review_columns = {"platform", "topic_id", "topic_count", "rank", "document_ref", "text_for_review"}
    if not naming_columns <= set(naming.columns) or not review_columns <= set(review.columns) or naming.empty:
        return False
    if not naming["human_name"].fillna("").astype(str).str.strip().ne("").all() or not naming["naming_rationale"].fillna("").astype(str).str.strip().ne("").all():
        return False
    keys = ["platform", "topic_id", "topic_count"]
    naming_keys = set(naming[keys].itertuples(index=False, name=None))
    review_keys = set(review[keys].itertuples(index=False, name=None))
    if naming_keys != review_keys or review["text_for_review"].fillna("").astype(str).str.strip().eq("").any():
        return False
    ranks = pd.to_numeric(review["rank"], errors="coerce")
    if ranks.isna().any():
        return False
    review = review.assign(_rank=ranks.astype(int))
    return all(set(group["_rank"]) == {1, 2, 3} for _, group in review.groupby(keys, sort=False))


def _input_checksums() -> dict[str, str]:
    return {
        "reddit": sha256_file(REPO_ROOT / "data" / "processed" / "reddit_age_gate" / "analysis_corpus.parquet"),
        "bluesky": sha256_file(REPO_ROOT / "data" / "processed" / "bluesky" / "posts.parquet"),
        "youtube": sha256_file(REPO_ROOT / "data" / "processed" / "youtube" / "documents.parquet"),
    }


def build_topic_artifacts() -> dict[str, Any]:
    documents = _load_documents()
    prevalence, terms, stability, reviews, diagnostics = [], [], [], [], {}
    for platform, frame in documents.items():
        p, t, s, r, d = _fit_platform(platform, frame)
        prevalence.append(p)
        terms.append(t)
        stability.append(s)
        reviews.append(r)
        diagnostics[platform] = d
    prevalence_frame = pd.concat(prevalence, ignore_index=True).sort_values(["platform", "scope", "topic_id"])
    terms_frame = pd.concat(terms, ignore_index=True).sort_values(["platform", "topic_id", "rank"])
    stability_frame = pd.concat(stability, ignore_index=True).sort_values(["platform", "topic_count", "left_seed", "right_seed"])
    novelty_frame = _topic_novelty(terms_frame, stability_frame)
    bertopic_frame = _bertopic_sensitivity(documents)
    TOPICS_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "prevalence": TOPICS_ROOT / "topic_prevalence.csv",
        "terms": TOPICS_ROOT / "topic_terms.csv",
        "stability": TOPICS_ROOT / "topic_stability.csv",
        "review": TOPICS_ROOT / "topic_review_packets.csv",
        "naming_ledger": TOPICS_ROOT / "topic_naming_ledger.csv",
        "novelty": TOPICS_ROOT / "topic_novelty.csv",
        "bertopic": TOPICS_ROOT / "bertopic_sensitivity.csv",
    }
    prevalence_frame.to_csv(outputs["prevalence"], index=False, lineterminator="\n")
    terms_frame.to_csv(outputs["terms"], index=False, lineterminator="\n")
    stability_frame.to_csv(outputs["stability"], index=False, lineterminator="\n")
    review_frame = pd.concat(reviews, ignore_index=True).sort_values(["platform", "topic_id", "rank"])
    review_frame.to_csv(outputs["review"], index=False, lineterminator="\n")
    selected_counts = {platform: diagnostics[platform]["selected_topic_count"] for platform in documents}
    ledger = pd.concat(
        [
            pd.DataFrame({"platform": [platform] * selected_counts[platform], "topic_id": range(selected_counts[platform]), "topic_count": selected_counts[platform]})
            for platform in documents
        ],
        ignore_index=True,
    )
    if outputs["naming_ledger"].exists():
        previous = pd.read_csv(outputs["naming_ledger"], keep_default_na=False)
        required = {"platform", "topic_id", "topic_count", "human_name", "naming_rationale"}
        if required <= set(previous.columns):
            ledger = ledger.merge(
                previous[list(required)],
                on=["platform", "topic_id", "topic_count"],
                how="left",
                validate="one_to_one",
            )
    for column in ("human_name", "naming_rationale"):
        if column not in ledger:
            ledger[column] = ""
        ledger[column] = ledger[column].fillna("").astype(str)
    ledger["status"] = np.where(ledger["human_name"].str.strip().ne(""), "human_named", "human_topic_naming_required")
    ledger.to_csv(outputs["naming_ledger"], index=False, lineterminator="\n")
    novelty_frame.to_csv(outputs["novelty"], index=False, lineterminator="\n")
    bertopic_frame.to_csv(outputs["bertopic"], index=False, lineterminator="\n")
    bertopic_status = _bertopic_status(bertopic_frame, sorted(documents))
    naming_complete = _topic_naming_complete(ledger, review_frame)
    manifest = {
        "schema_version": TOPICS_SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "command": "python -m src.analysis.run topics --check",
        "preprocessing_version": PREPROCESSING_VERSION,
        "embedding_backend": TOPIC_EMBEDDING_VERSION,
        "topic_counts": list(TOPIC_COUNTS),
        "selected_topic_counts": selected_counts,
        "seeds": list(SEEDS),
        "bertopic_seeds": list(BERTopic_SEEDS),
        "platform_diagnostics": diagnostics,
        "input_checksums": _input_checksums(),
        "bertopic": {
            "status": bertopic_status,
            "installed": bool(importlib.util.find_spec("bertopic")),
            "required_runtime": f"bertopic=={BERTOPIC_REQUIRED_VERSION}",
            "embedding_backend": TOPIC_EMBEDDING_VERSION,
            "seeds": list(BERTopic_SEEDS),
            "representation_version": "bertopic-min_size10-count-word-1-2gram",
        },
        "prevalence_uncertainty": {"method": "wilson_95_normal_approximation", "level": 0.95, "zero_topic_rows_included": True},
        "naming": {"status": "complete" if naming_complete else "human_topic_naming_required", "complete": bool(naming_complete)},
        "outputs": {
            name: {"path": relative_path(path), "sha256": sha256_file(path), "rows": int(len(pd.read_csv(path)))}
            for name, path in outputs.items()
        },
        "limitations": [
            "NMF is exploratory until human naming and representative-document review are complete.",
            "Topic terms are not proposal frame labels and do not replace v2 frame annotation.",
            "BERTopic is an optional, pinned multi-seed semantic-discovery sensitivity and is not silently substituted or treated as a missing NMF result.",
        ],
        "status": "exploratory_named" if naming_complete else "exploratory_review_required",
    }
    write_json(TOPICS_MANIFEST, manifest)
    return manifest


def check_topics() -> dict[str, Any]:
    if not TOPICS_MANIFEST.exists():
        raise FileNotFoundError(f"topic manifest is missing: {TOPICS_MANIFEST}; run the explicit topics build first")
    manifest = read_json(TOPICS_MANIFEST)
    if manifest.get("schema_version") != TOPICS_SCHEMA_VERSION or manifest.get("topic_counts") != list(TOPIC_COUNTS) or manifest.get("seeds") != list(SEEDS):
        raise ValueError("topic manifest configuration is stale; run the explicit topics build")
    bertopic = manifest.get("bertopic", {})
    if (
        manifest.get("bertopic_seeds") != list(BERTopic_SEEDS)
        or manifest.get("embedding_backend") != TOPIC_EMBEDDING_VERSION
        or bertopic.get("required_runtime") != f"bertopic=={BERTOPIC_REQUIRED_VERSION}"
        or bertopic.get("embedding_backend") != TOPIC_EMBEDDING_VERSION
        or bertopic.get("seeds") != list(BERTopic_SEEDS)
    ):
        raise ValueError("topic sensitivity contract is stale; run the explicit topics build")
    if manifest.get("input_checksums") != _input_checksums():
        raise ValueError("topic input checksum changed")
    for record in manifest.get("outputs", {}).values():
        path = REPO_ROOT / record["path"]
        if not path.exists() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"topic artifact changed: {record['path']}")
    prevalence_path = REPO_ROOT / manifest["outputs"]["prevalence"]["path"]
    prevalence = pd.read_csv(prevalence_path)
    required_prevalence = {"share_lower_95", "share_upper_95", "uncertainty_method"}
    if not required_prevalence <= set(prevalence.columns):
        raise ValueError("topic prevalence uncertainty fields are missing; run the explicit topics build")
    if not prevalence["share_lower_95"].le(prevalence["share"]).all() or not prevalence["share"].le(prevalence["share_upper_95"]).all():
        raise ValueError("topic prevalence uncertainty interval is invalid")
    naming_path = REPO_ROOT / manifest["outputs"]["naming_ledger"]["path"]
    naming = pd.read_csv(naming_path, keep_default_na=False)
    review_path = REPO_ROOT / manifest["outputs"]["review"]["path"]
    review = pd.read_csv(review_path, keep_default_na=False)
    if not _topic_naming_complete(naming, review):
        raise ValueError("topic naming gate is incomplete; representative review, names, and rationale are required")
    naming_complete = True
    expected_status = "exploratory_named"
    if manifest.get("status") != expected_status or manifest.get("naming", {}).get("complete") != bool(naming_complete):
        raise ValueError("topic naming gate is stale; run the explicit topics build or complete the naming ledger")
    bertopic_path = REPO_ROOT / manifest["outputs"]["bertopic"]["path"]
    bertopic_frame = pd.read_csv(bertopic_path)
    expected_bertopic_status = _bertopic_status(bertopic_frame, sorted(manifest.get("selected_topic_counts", {})))
    if bertopic.get("status") != expected_bertopic_status:
        raise ValueError("BERTopic sensitivity status is overstated; run the explicit topics build")
    return {
        "status": manifest["status"],
        "artifact": relative_path(TOPICS_MANIFEST),
        "outputs": len(manifest.get("outputs", {})),
        "naming_status": manifest.get("naming", {}).get("status", "unknown"),
        "bertopic_status": bertopic.get("status", "unknown"),
    }
