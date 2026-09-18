"""Held-out measurement selection, reliability and fallback gates.

The module is deliberately model-lazy: without human labels it emits a
machine-readable blocked validity artifact and never treats existing legacy
scores or predictions as gold.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from src.analysis.artifacts import ANALYSIS_ROOT, REPO_ROOT, read_json, relative_path, sha256_file, utc_now_iso, write_json
from src.analysis.codebook import CODEBOOK_VERSION, FRAME_DEFINITIONS
from src.analysis.validation import _frame_digest, _key, active_model_root, active_validation_root, load_label_packet

VALIDATION_ROOT = active_validation_root()
MODEL_ROOT = active_model_root()
SENTIMENT_SELECTION = MODEL_ROOT / "selection_sentiment.json"
STANCE_FRAME_SELECTION = MODEL_ROOT / "selection_stance_frames.json"
EVALUATION_REPORT = MODEL_ROOT / "evaluation_report.json"
RESERVE_ACTIVATION = MODEL_ROOT / "reserve_activation.json"
RESERVE_ASSESSMENT = MODEL_ROOT / "reserve_assessment.json"
RESERVE_ASSESSMENT_SEAL = MODEL_ROOT / "reserve_assessment.seal.json"
RESERVE_PREDICTIONS = MODEL_ROOT / "reserve_predictions.csv"
RESERVE_ACTIVATION_REASON = "premature_evaluation_opened_before_selection_freeze"
SELECTION_SCHEMA_VERSION = "measurement-selection.v3"
MEASUREMENT_CONTRACT_VERSION = "measurement-contract-v2-all-negative-frame-rows"
ROBERTA_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ROBERTA_REVISION = None  # Filled only by an immutable, user-approved model receipt before activation.
NLI_MODEL_ID = "MoritzLaurer/deberta-v3-base-zeroshot-v1.1"
NLI_REVISION = None  # Filled only by an immutable, user-approved model receipt before activation.
TRANSFORMER_RUNTIME = {
    "transformers": "4.57.6",
    "tokenizers": "0.22.1",
    "torch": "2.14.0",
}
MODEL_RECEIPT_FIELDS = [
    "model_id",
    "revision",
    "local_path",
    "local_sha256",
    "tokenizer_path",
    "tokenizer_revision",
    "model_card_url",
    "source_url",
]

FRAME_LABELS = [frame["label"] for frame in FRAME_DEFINITIONS]
FRAME_COMPLETION_FIELDS = ("relevance", "language", "target_policy", "stance", "sentiment", "bypass_techniques")
STANCE_LABELS = ["support", "oppose", "mixed_conditional", "neutral_descriptive", "unclear_ambiguous"]
SENTIMENT_LABELS = ["positive", "negative", "neutral", "mixed_ambiguous"]
SENTIMENT_AUTOMATED_LABELS = ["positive", "negative", "neutral"]
SENTIMENT_THRESHOLDS = ((-0.05, 0.05), (-0.10, 0.10), (-0.15, 0.15), (-0.20, 0.20))
FRAME_THRESHOLD_CANDIDATES = (0.30, 0.40, 0.50, 0.60, 0.70)
BOOTSTRAP_REPLICATES = 9_999
MODEL_RECEIPTS = MODEL_ROOT / "model_receipts.json"
PREDICTION_CACHE_ROOT = MODEL_ROOT / "prediction_cache"
PREDICTION_CACHE_VERSION = "measurement-prediction-cache-v1"
PREPROCESSING_VERSION = "masked-text-v2"
NLI_HYPOTHESIS_TEMPLATE = "The author's stance toward age-assurance policy is {}."
NLI_CANDIDATE_LABELS = (
    "supporting the policy",
    "opposing the policy",
    "taking a mixed or conditional position on the policy",
    "describing the policy neutrally",
    "having an unclear or ambiguous stance toward the policy",
)
NLI_LABEL_MAPPING = dict(zip(NLI_CANDIDATE_LABELS, STANCE_LABELS))
FRAME_LEXICON_TERMS = {
    "policy_assurance": ("age verification", "age assurance", "age check", "online safety", "minimum age", "age limit"),
    "child_safety": ("child", "children", "kids", "minor", "groom", "harmful content", "protect"),
    "privacy_surveillance": ("privacy", "surveillance", "biometric", "facial recognition", "personal data", "id upload", "tracking", "retention"),
    "governance_platform_responsibility": ("government", "regulator", "platform", "enforcement", "accountability", "transparency", "appeal", "audit"),
    "circumvention_censorship_autonomy": ("vpn", "proxy", "tor", "bypass", "circumvent", "censor", "free speech", "autonomy", "overreach"),
}
_TRANSFORMER_PIPELINES: dict[tuple[str, str], Any] = {}


def _packet(split: str) -> pd.DataFrame:
    return load_label_packet(split)


def _labelled(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].astype(str).str.strip().ne("")


def _frame_evaluation_mask(frame: pd.DataFrame) -> pd.Series:
    """Rows with an adjudicated frame field; blank means no supported frame."""

    if "adjudicated_frame_labels" not in frame:
        return pd.Series(False, index=frame.index)
    return frame["adjudicated_frame_labels"].notna()


def _frame_labels(value: str) -> set[str]:
    values = {label.strip() for label in str(value).split("|") if label.strip()}
    unknown = values - set(FRAME_LABELS)
    if unknown:
        raise ValueError(f"unknown frame labels: {sorted(unknown)}")
    return values


def _cohen_kappa(left: pd.Series, right: pd.Series) -> float:
    if len(set(left.astype(str)) | set(right.astype(str))) < 2:
        return 1.0 if bool((left == right).all()) else 0.0
    value = float(cohen_kappa_score(left, right))
    if np.isfinite(value):
        return value
    return 1.0 if bool((left == right).all()) else 0.0


def reliability_report(frame: pd.DataFrame, field: str, multilabel: bool = False) -> dict[str, Any]:
    left, right = f"coder_a_{field}", f"coder_b_{field}"
    if multilabel:
        present = pd.Series(True, index=frame.index)
        for coder in ("a", "b"):
            for required_field in FRAME_COMPLETION_FIELDS:
                present &= _labelled(frame, f"coder_{coder}_{required_field}")
    else:
        present = _labelled(frame, left) & _labelled(frame, right)
    if not present.any():
        return {"field": field, "usable_rows": 0, "status": "human-labels-required"}
    if not present.all():
        return {"field": field, "usable_rows": int(present.sum()), "missing_rows": int((~present).sum()), "status": "incomplete-double-coding"}
    left_values, right_values = frame[left].astype(str), frame[right].astype(str)
    if not multilabel:
        return {
            "field": field,
            "usable_rows": int(present.sum()),
            "status": "available",
            "raw_agreement": float((left_values == right_values).mean()),
            "cohen_kappa": _cohen_kappa(left_values, right_values),
        }
    agreements = []
    for label in FRAME_LABELS:
        left_set = left_values.map(_frame_labels).map(lambda values: label in values)
        right_set = right_values.map(_frame_labels).map(lambda values: label in values)
        raw_agreement = float((left_set == right_set).mean())
        kappa = _cohen_kappa(left_set, right_set)
        agreements.append({"label": label, "raw_agreement": raw_agreement, "cohen_kappa": kappa})
    report = {
        "field": field,
        "usable_rows": int(present.sum()),
        "status": "available",
        "raw_agreement": float((left_values == right_values).mean()),
        "cohen_kappa": float(np.mean([item["cohen_kappa"] for item in agreements])),
        "per_frame_agreement": agreements,
        "mean_cohen_kappa": float(np.mean([item["cohen_kappa"] for item in agreements])),
    }
    if multilabel:
        any_frame = _labelled(frame, left) | _labelled(frame, right)
        report["frame_rows_excluded_no_frame"] = int((present & ~any_frame).sum())
    return report


def reliability_gates(frame: pd.DataFrame) -> dict[str, Any]:
    fields = {
        "relevance": reliability_report(frame, "relevance"),
        "language": reliability_report(frame, "language"),
        "target_policy": reliability_report(frame, "target_policy"),
        "stance": reliability_report(frame, "stance"),
        "sentiment": reliability_report(frame, "sentiment"),
        "frame_labels": reliability_report(frame, "frame_labels", multilabel=True),
    }
    failures = []
    for field, report in fields.items():
        if report.get("status") != "available" or report.get("usable_rows", 0) == 0:
            failures.append(field)
            continue
        if field != "frame_labels":
            if report.get("raw_agreement") is None or report.get("cohen_kappa") is None:
                failures.append(field)
            elif report["raw_agreement"] < 0.80 or report["cohen_kappa"] < 0.60:
                failures.append(field)
        else:
            frame_agreement = report.get("per_frame_agreement", [])
            if not frame_agreement or any(
                item.get("raw_agreement") is None
                or item.get("cohen_kappa") is None
                or item["raw_agreement"] < 0.80
                or item["cohen_kappa"] < 0.60
                for item in frame_agreement
            ):
                failures.append(field)
    return {"fields": fields, "passed": not failures, "failed_fields": sorted(set(failures))}


def _grouped_folds(frame: pd.DataFrame, labels: pd.Series, n_splits: int = 5):
    groups = frame["cluster_id"].fillna(frame["doc_id"]).astype(str)
    class_counts = labels.value_counts()
    n = min(n_splits, int(class_counts.min()) if not class_counts.empty else 0, int(groups.nunique()))
    if n < 2:
        return []
    splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=20260915)
    return list(splitter.split(frame, labels, groups))


def _grouped_multilabel_fold_plan(
    frame: pd.DataFrame, labels: np.ndarray, n_splits: int = 5
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Create deterministic group-separated multilabel folds with support checks."""

    values = np.asarray(labels, dtype=int)
    if values.ndim != 2 or len(values) != len(frame):
        raise ValueError("multilabel fold labels must be a row-aligned two-dimensional array")
    groups = frame["cluster_id"].fillna(frame.get("doc_id", pd.Series(frame.index, index=frame.index))).astype(str)
    group_names = sorted(groups.unique(), key=lambda value: (_key(f"multilabel-group|{value}"), value))
    group_indexes = {group: np.flatnonzero(groups.to_numpy() == group) for group in group_names}
    group_values = {group: values[indexes] for group, indexes in group_indexes.items()}
    group_positive_support = np.asarray([sum(bool(group_values[group][:, label].any()) for group in group_names) for label in range(values.shape[1])])
    group_negative_support = np.asarray([sum(bool((group_values[group][:, label] == 0).any()) for group in group_names) for label in range(values.shape[1])])
    maximum = min(
        int(n_splits),
        len(group_names),
        int(group_positive_support.min()) if len(group_positive_support) else 0,
        int(group_negative_support.min()) if len(group_negative_support) else 0,
    )
    info: dict[str, Any] = {
        "requested_fold_count": int(n_splits),
        "fold_count": 0,
        "group_field": "cluster_id",
        "strict_group_separation": True,
        "reduction_reason": None,
        "positive_group_support": group_positive_support.tolist(),
        "negative_group_support": group_negative_support.tolist(),
    }
    for candidate in range(maximum, 1, -1):
        target_rows = len(values) / candidate
        target_positive = values.sum(axis=0) / candidate
        ordered_groups = sorted(
            group_names,
            key=lambda group: (
                -float(group_values[group].sum()),
                -len(group_indexes[group]),
                _key(f"multilabel-order|{group}"),
                group,
            ),
        )
        best: tuple[float, list[np.ndarray]] | None = None
        rng = np.random.default_rng(_key(f"multilabel-folds|{candidate}"))
        for attempt in range(1000):
            group_order = ordered_groups if attempt == 0 else rng.permutation(ordered_groups).tolist()
            fold_rows = np.zeros(candidate, dtype=float)
            fold_positive = np.zeros((candidate, values.shape[1]), dtype=float)
            fold_groups: list[list[str]] = [[] for _ in range(candidate)]
            for group_number, group in enumerate(group_order):
                group_positive_values = group_values[group].sum(axis=0).astype(float)
                row_count = len(group_indexes[group])
                fold = group_number % candidate
                fold_groups[fold].append(group)
                fold_rows[fold] += row_count
                fold_positive[fold] += group_positive_values
            fold_indexes = [
                np.concatenate([group_indexes[group] for group in groups_in_fold]) if groups_in_fold else np.asarray([], dtype=int)
                for groups_in_fold in fold_groups
            ]
            support_ok = all(
                np.all(values[indexes].sum(axis=0) > 0)
                and np.all(values[indexes].sum(axis=0) < len(indexes))
                for indexes in fold_indexes
            )
            if not support_ok:
                continue
            score = float(
                np.mean(((fold_rows - target_rows) / max(target_rows, 1.0)) ** 2)
                + np.mean(((fold_positive - target_positive) / np.maximum(target_positive, 1.0)) ** 2)
            )
            if best is None or score < best[0]:
                best = (score, fold_indexes)
        if best is None:
            continue
        folds = []
        all_indexes = np.arange(len(values))
        for indexes in best[1]:
            test = np.sort(indexes)
            train = np.setdiff1d(all_indexes, test, assume_unique=False)
            folds.append((train, test))
        info["fold_count"] = candidate
        if candidate != n_splits:
            info["reduction_reason"] = "requested_count_exceeded_grouped_per_label_support"
        return folds, info
    info["reduction_reason"] = "no_grouped_fold_count_preserves_positive_and_negative_support"
    return [], info


def _grouped_multilabel_folds(frame: pd.DataFrame, labels: np.ndarray, n_splits: int = 5):
    """Compatibility wrapper returning only the fold list."""

    return _grouped_multilabel_fold_plan(frame, labels, n_splits)[0]


def _vader():
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except LookupError:
        return None


def vader_scores(texts: pd.Series) -> np.ndarray:
    analyzer = _vader()
    if analyzer is None:
        raise RuntimeError("nltk vader_lexicon is unavailable")
    return np.asarray([analyzer.polarity_scores(str(text))["compound"] for text in texts], dtype=float)


def sentiment_from_compound(scores: np.ndarray, negative: float, positive: float) -> np.ndarray:
    return np.where(scores <= negative, "negative", np.where(scores >= positive, "positive", "neutral"))


def _sentiment_contract() -> dict[str, Any]:
    return {
        "labels": list(SENTIMENT_LABELS),
        "automated_labels": list(SENTIMENT_AUTOMATED_LABELS),
        "unsupported_labels": ["mixed_ambiguous"],
        "abstention_policy": "three-class models abstain from mixed_ambiguous claims; use inverse-probability-weighted human labels for that class",
        "automated_claims_allowed": True,
        "mixed_ambiguous_fallback": "weighted_human_sample",
    }


def macro_metrics(actual: pd.Series, predicted: np.ndarray, labels: list[str] | None = None) -> dict[str, Any]:
    classes = labels or sorted(set(actual.astype(str)) | set(map(str, predicted)))
    precision, recall, f1, support = precision_recall_fscore_support(actual.astype(str), predicted, labels=classes, zero_division=0)
    return {
        "macro_f1": float(f1_score(actual.astype(str), predicted, labels=classes, average="macro", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(score),
                "support": int(n),
                "predicted_support": int(np.count_nonzero(np.asarray(predicted).astype(str) == label)),
            }
            for label, p, r, score, n in zip(classes, precision, recall, f1, support)
        },
    }


def _sentiment_metrics_with_abstention(actual: pd.Series, predicted: np.ndarray) -> dict[str, Any]:
    """Report the full four-class view and the supported three-class view."""

    actual = actual.astype(str).reset_index(drop=True)
    predicted = np.asarray(predicted).astype(str)
    full = macro_metrics(actual, predicted, SENTIMENT_LABELS)
    supported = actual.isin(SENTIMENT_AUTOMATED_LABELS)
    if supported.any():
        supported_metrics = macro_metrics(actual[supported], predicted[supported], SENTIMENT_AUTOMATED_LABELS)
    else:
        supported_metrics = None
    full["supported_class_metrics"] = supported_metrics
    full["supported_rows"] = int(supported.sum())
    full["abstained_mixed_rows"] = int((~supported).sum())
    full["selection_macro_f1"] = supported_metrics["macro_f1"] if supported_metrics else None
    return full


def _multilabel_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("multilabel metrics need equal two-dimensional arrays")
    per_frame: dict[str, Any] = {}
    for index, label in enumerate(FRAME_LABELS):
        precision, recall, f1, support = precision_recall_fscore_support(
            actual[:, index], predicted[:, index], labels=[0, 1], average=None, zero_division=0
        )
        per_frame[label] = {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
            "negative_support": int(support[0]),
        }
    return {"macro_f1": float(np.mean([item["f1"] for item in per_frame.values()])), "per_frame": per_frame, "sample_rows": int(len(actual))}


def cluster_bootstrap_macro_f1(
    actual: pd.Series,
    predicted: np.ndarray,
    clusters: pd.Series,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 20260915,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Return a cluster bootstrap interval for macro-F1."""

    actual = actual.astype(str).reset_index(drop=True)
    predicted = np.asarray(predicted).astype(str)
    clusters = clusters.astype(str).reset_index(drop=True)
    unique_clusters = np.asarray(sorted(clusters.unique()))
    if len(actual) != len(predicted) or len(actual) != len(clusters) or not len(unique_clusters):
        raise ValueError("cluster bootstrap inputs must have equal non-empty lengths")
    rng = np.random.default_rng(seed)
    rows_by_cluster = {cluster: np.flatnonzero(clusters.to_numpy() == cluster) for cluster in unique_clusters}
    values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        selected = np.concatenate([rows_by_cluster[cluster] for cluster in sampled])
        values[index] = f1_score(actual.iloc[selected], predicted[selected], labels=labels, average="macro", zero_division=0)
    return {
        "replicates": int(replicates),
        "seed": seed,
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def evaluation_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
    clusters: pd.Series,
    inclusion_probability: pd.Series | None = None,
    labels: list[str] | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Expose unweighted, design-weighted and cluster-bootstrap diagnostics."""

    actual = actual.astype(str).reset_index(drop=True)
    predicted = np.asarray(predicted).astype(str)
    classes = labels or sorted(set(actual) | set(predicted))
    sample_weight = None
    weighted = None
    if inclusion_probability is not None:
        probabilities = pd.to_numeric(inclusion_probability, errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities <= 0):
            raise ValueError("inclusion probabilities must be finite and positive")
        sample_weight = 1.0 / probabilities
        weighted_precision, weighted_recall, weighted_f1, weighted_support = precision_recall_fscore_support(
            actual, predicted, labels=classes, sample_weight=sample_weight, zero_division=0
        )
        weighted = {
            "macro_f1": float(np.mean(weighted_f1)),
            "support_weighted_f1": float(np.average(weighted_f1, weights=np.maximum(weighted_support, 1e-12))),
            "per_class": {
                label: {"precision": float(precision), "recall": float(recall), "f1": float(score), "weighted_support": float(support)}
                for label, precision, recall, score, support in zip(classes, weighted_precision, weighted_recall, weighted_f1, weighted_support)
            },
        }
    return {
        "unweighted": macro_metrics(actual, predicted, classes),
        "design_weighted": weighted,
        "confusion_matrix": {"labels": classes, "counts": confusion_matrix(actual, predicted, labels=classes).tolist()},
        "cluster_bootstrap_macro_f1": cluster_bootstrap_macro_f1(actual, predicted, clusters, replicates=bootstrap_replicates, labels=classes),
        "sample_rows": int(len(actual)),
        "cluster_count": int(clusters.nunique()),
        "weights": "inverse_inclusion_probability" if sample_weight is not None else None,
    }


def add_sentiment_abstention_metrics(
    metrics: dict[str, Any],
    actual: pd.Series,
    predicted: np.ndarray,
    clusters: pd.Series,
    inclusion_probability: pd.Series | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Attach a separately scored three-class view and an explicit mixed fallback."""

    actual = actual.astype(str).reset_index(drop=True)
    predicted = np.asarray(predicted).astype(str)
    clusters = clusters.astype(str).reset_index(drop=True)
    supported = actual.isin(SENTIMENT_AUTOMATED_LABELS)
    if supported.any():
        probabilities = inclusion_probability.reset_index(drop=True) if inclusion_probability is not None else None
        supported_metrics = evaluation_metrics(
            actual[supported],
            predicted[supported.to_numpy()],
            clusters[supported],
            probabilities[supported] if probabilities is not None else None,
            SENTIMENT_AUTOMATED_LABELS,
            bootstrap_replicates,
        )
        metrics["automated_macro_f1"] = supported_metrics["unweighted"]["macro_f1"]
        metrics["automated_per_class"] = supported_metrics["unweighted"]["per_class"]
        metrics["cluster_bootstrap_automated_macro_f1"] = supported_metrics["cluster_bootstrap_macro_f1"]
    else:
        metrics["automated_macro_f1"] = None
        metrics["automated_per_class"] = {}
        metrics["cluster_bootstrap_automated_macro_f1"] = {"lower_95": -1.0}
    metrics["mixed_ambiguous_human_fallback"] = {
        "status": "weighted_human_sample",
        "rows": int((~supported).sum()),
        "claim_scope": "mixed_ambiguous is not predicted by the three-class automated instrument",
    }
    return metrics


def multilabel_evaluation_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    clusters: pd.Series,
    inclusion_probability: pd.Series | None = None,
    labels: list[str] | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    if actual.shape != predicted.shape or actual.ndim != 2 or len(clusters) != actual.shape[0]:
        raise ValueError("multilabel evaluation arrays must have the same two-dimensional shape")
    labels = labels or [str(index) for index in range(actual.shape[1])]
    if len(labels) != actual.shape[1]:
        raise ValueError("multilabel evaluation labels do not match the array width")
    per_label = {
        label: evaluation_metrics(
            pd.Series(actual[:, index]),
            predicted[:, index],
            clusters,
            inclusion_probability,
            ["0", "1"],
            bootstrap_replicates,
        )
        for index, label in enumerate(labels)
    }
    macro_f1_values = np.asarray([result["unweighted"]["macro_f1"] for result in per_label.values()], dtype=float)
    clusters = clusters.astype(str).reset_index(drop=True)
    unique_clusters = np.asarray(sorted(clusters.unique()))
    rng = np.random.default_rng(20260915)
    rows_by_cluster = {cluster: np.flatnonzero(clusters.to_numpy() == cluster) for cluster in unique_clusters}
    macro_bootstrap = np.empty(bootstrap_replicates, dtype=float)
    for index in range(bootstrap_replicates):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        selected = np.concatenate([rows_by_cluster[cluster] for cluster in sampled])
        macro_bootstrap[index] = float(
            np.mean(
                [
                    f1_score(actual[selected, label_index], predicted[selected, label_index], zero_division=0)
                    for label_index in range(actual.shape[1])
                ]
            )
        )
    return {
        "macro_f1": float(macro_f1_values.mean()),
        "per_label": per_label,
        "sample_rows": int(actual.shape[0]),
        "label_count": int(actual.shape[1]),
        "cluster_bootstrap_macro_f1": {
            "replicates": int(bootstrap_replicates),
            "seed": 20260915,
            "lower_95": float(np.quantile(macro_bootstrap, 0.025)),
            "median": float(np.quantile(macro_bootstrap, 0.5)),
            "upper_95": float(np.quantile(macro_bootstrap, 0.975)),
        },
    }


def _vader_candidate_results(actual: pd.Series, scores: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> list[dict[str, Any]]:
    """Score each named threshold on its own grouped out-of-fold predictions."""

    configs = []
    for negative, positive in SENTIMENT_THRESHOLDS:
        out_actual, out_pred = [], []
        for _, test_index in folds:
            out_actual.extend(actual.iloc[test_index].tolist())
            out_pred.extend(sentiment_from_compound(scores[test_index], negative, positive).tolist())
        if out_actual:
            actual_values = pd.Series(out_actual)
            predicted_values = np.asarray(out_pred)
            configs.append(
                {
                    "negative_threshold": negative,
                    "positive_threshold": positive,
                    "metrics": _sentiment_metrics_with_abstention(actual_values, predicted_values),
                    "label_contract": _sentiment_contract(),
                }
            )
    if configs:
        return configs
    return []


def select_vader(frame: pd.DataFrame, platform: str = "all") -> dict[str, Any]:
    actual = frame["adjudicated_sentiment"].astype(str)
    scores = vader_scores(frame["text_for_annotation"])
    folds = _grouped_folds(frame, actual)
    configs = _vader_candidate_results(actual, scores, folds)
    if not configs:
        return {"name": None, "status": "insufficient_grouped_folds", "group_field": "cluster_id", "fold_count": 0, "label_contract": _sentiment_contract(), "roberta_comparison": _model_comparison_status("sentiment")}
    selected = max(configs, key=lambda item: item["metrics"].get("selection_macro_f1", -1.0))
    comparison = _transformer_comparison("sentiment", frame, platform)
    return {
        "name": _selected_pipeline("vader", selected["metrics"], comparison),
        "status": "available",
        "label_contract": _sentiment_contract(),
        "automated_claims_allowed": comparison.get("status") == "available",
        "fallback_policy": "mixed_ambiguous rows use weighted human inference; supported three-class rows may use the frozen selected model",
        "metrics": selected["metrics"],
        "fold_count": len(folds),
        "selected": selected,
        "candidates": configs,
        "roberta_comparison": comparison,
    }


def _stance_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260915)),
    ])


def select_stance(frame: pd.DataFrame, platform: str = "all") -> dict[str, Any]:
    labels = frame["adjudicated_stance"].astype(str)
    folds = _grouped_folds(frame, labels)
    actual, predicted = [], []
    for train_index, test_index in folds:
        model = _stance_pipeline().fit(frame.iloc[train_index]["text_for_annotation"], labels.iloc[train_index])
        actual.extend(labels.iloc[test_index].tolist())
        predicted.extend(model.predict(frame.iloc[test_index]["text_for_annotation"]).tolist())
    if not actual:
        return {"name": None, "status": "insufficient_grouped_folds", "group_field": "cluster_id", "fold_count": 0, "nli_comparison": _model_comparison_status("stance")}
    baseline_metrics = macro_metrics(pd.Series(actual), np.asarray(predicted), STANCE_LABELS)
    comparison = _transformer_comparison("stance", frame, platform)
    return {
        "name": _selected_pipeline("tfidf_logistic_regression", baseline_metrics, comparison),
        "status": "available",
        "automated_claims_allowed": comparison.get("status") == "available",
        "fallback_policy": "weighted human-sample inference when the required DeBERTa comparison is unavailable",
        "fold_count": len(folds),
        "metrics": baseline_metrics,
        "group_field": "cluster_id",
        "config": {"ngram_range": [1, 2], "min_df": 2, "class_weight": "balanced"},
        "nli_contract": {
            "candidate_labels": list(NLI_CANDIDATE_LABELS),
            "hypothesis_template": NLI_HYPOTHESIS_TEMPLATE,
            "label_mapping": NLI_LABEL_MAPPING,
        },
        "nli_comparison": comparison,
    }


def _frame_pipeline() -> tuple[Pipeline, list[str]]:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("classifier", OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260915))),
    ]), FRAME_LABELS


def _validate_frame_threshold_values(values: Any) -> np.ndarray:
    thresholds = np.asarray(values, dtype=float)
    if thresholds.shape != (len(FRAME_LABELS),):
        raise ValueError("selected frame thresholds have the wrong shape")
    if not np.isfinite(thresholds).all() or any(float(value) not in FRAME_THRESHOLD_CANDIDATES for value in thresholds):
        raise ValueError("selected frame thresholds are outside the frozen candidate set")
    return thresholds


def _selected_frame_thresholds(candidate: dict[str, Any]) -> np.ndarray:
    values = candidate.get("thresholds")
    if not isinstance(values, dict) or set(values) != set(FRAME_LABELS):
        raise ValueError("selected frame thresholds are missing or incomplete")
    return _validate_frame_threshold_values([values[label] for label in FRAME_LABELS])


def _frame_lexicon_predictions(texts: pd.Series) -> np.ndarray:
    lowered = texts.fillna("").astype(str).str.casefold()
    return np.asarray(
        [[int(any(term in text for term in FRAME_LEXICON_TERMS[label])) for label in FRAME_LABELS] for text in lowered],
        dtype=int,
    )


def select_frames(frame: pd.DataFrame, platform: str = "all") -> dict[str, Any]:
    frame_rows = frame.loc[_frame_evaluation_mask(frame)].copy().reset_index(drop=True)
    denominator = {
        "frame_rows_total": int(len(frame)),
        "frame_rows_evaluated": int(len(frame_rows)),
        "frame_rows_excluded_no_frame": int(len(frame) - len(frame_rows)),
        "frame_exclusion_policy": "blank adjudicated frame_labels are valid no-supported-frame rows and are included in frame model evaluation",
    }
    if frame_rows.empty:
        return {"name": None, "status": "no_frame_rows_for_evaluation", "group_field": "cluster_id", "fold_count": 0, **denominator}
    labels = np.asarray([[_frame_labels(value).__contains__(label) for label in FRAME_LABELS] for value in frame_rows["adjudicated_frame_labels"]], dtype=int)
    folds, fold_plan = _grouped_multilabel_fold_plan(frame_rows, labels)
    actual_oof = np.zeros_like(labels, dtype=int)
    probability_oof = np.full(labels.shape, np.nan, dtype=float)
    evaluated = np.zeros(len(frame_rows), dtype=bool)
    for train_index, test_index in folds:
        model, _ = _frame_pipeline()
        model.fit(frame_rows.iloc[train_index]["text_for_annotation"], labels[train_index])
        if evaluated[test_index].any():
            raise ValueError("grouped frame folds reuse an evaluation row")
        actual_oof[test_index] = labels[test_index]
        probability_oof[test_index] = model.predict_proba(frame_rows.iloc[test_index]["text_for_annotation"])
        evaluated[test_index] = True
    if not evaluated.any():
        return {"name": None, "status": "insufficient_grouped_folds", "group_field": "cluster_id", "fold_count": 0, "fold_plan": fold_plan, **denominator}
    evaluation_indexes = np.flatnonzero(evaluated)
    actual_array = actual_oof[evaluation_indexes].astype(float)
    probability_array = probability_oof[evaluation_indexes]
    evaluation_rows = frame_rows.iloc[evaluation_indexes]
    thresholds: dict[str, float] = {}
    per_frame = {}
    for index, label in enumerate(FRAME_LABELS):
        candidates = []
        for threshold in FRAME_THRESHOLD_CANDIDATES:
            predicted = (probability_array[:, index] >= threshold).astype(int)
            precision, recall, f1, support = precision_recall_fscore_support(
                actual_array[:, index], predicted, labels=[0, 1], average=None, zero_division=0
            )
            candidates.append((float(f1[1]), abs(threshold - 0.5), threshold, precision, recall, support))
        _, _, threshold, precision, recall, support = max(candidates, key=lambda item: (item[0], -item[1], -item[2]))
        thresholds[label] = float(threshold)
        per_frame[label] = {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(max(candidates, key=lambda item: (item[0], -item[1], -item[2]))[0]),
            "support": int(support[1]),
            "negative_support": int(support[0]),
        }
    lexicon_metrics = _multilabel_metrics(actual_array, _frame_lexicon_predictions(evaluation_rows["text_for_annotation"]))
    return {
        "name": "tfidf_ovr_logistic_regression",
        "status": "available",
        "automated_claims_allowed": True,
        "fold_count": len(folds),
        "fold_plan": fold_plan,
        "macro_f1": float(np.mean([item["f1"] for item in per_frame.values()])),
        "selection_macro_f1": float(np.mean([item["f1"] for item in per_frame.values()])),
        "per_frame": per_frame,
        "thresholds": thresholds,
        "threshold_candidates": list(FRAME_THRESHOLD_CANDIDATES),
        "lexicon_sensitivity": {"name": "transparent_v2_lexicon", "metrics": lexicon_metrics, "terms": FRAME_LEXICON_TERMS},
        "group_field": "cluster_id",
        "config": {"ngram_range": [1, 2], "min_df": 2, "class_weight": "balanced", "threshold_selection": "development grouped out-of-fold F1; ties closest to 0.50"},
        **denominator,
    }


def _model_receipt(model_id: str) -> dict[str, Any] | None:
    if not MODEL_RECEIPTS.exists():
        return None
    payload = read_json(MODEL_RECEIPTS)
    record = payload.get("models", {}).get(model_id, payload.get(model_id))
    if not isinstance(record, dict) or record.get("model_id", model_id) != model_id:
        return None
    if any(not isinstance(record.get(field), str) or not record[field].strip() for field in MODEL_RECEIPT_FIELDS):
        return None
    if not all(_is_immutable_revision(record[field]) for field in ("revision", "tokenizer_revision")):
        return None
    if not _is_sha256(record["local_sha256"]):
        return None
    if not all(_is_https_url(record[field]) for field in ("model_card_url", "source_url")):
        return None
    local_path = _receipt_path(record["local_path"])
    if not local_path.exists() or _sha256_path(local_path) != record["local_sha256"].casefold():
        return None
    tokenizer_path = _receipt_path(record["tokenizer_path"])
    if not tokenizer_path.exists():
        return None
    return {**record, "local_path": str(local_path), "tokenizer_path": str(tokenizer_path)}


def _receipt_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _is_immutable_revision(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdefABCDEF" for character in value)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def _is_https_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def _sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        return ""
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _model_runtime_status(model_id: str) -> dict[str, Any]:
    receipt = _model_receipt(model_id)
    if receipt is None:
        return {"status": "missing_immutable_receipt", "model_id": model_id}
    missing = [name for name in TRANSFORMER_RUNTIME if importlib.util.find_spec(name) is None]
    if missing:
        return {"status": "transformers_runtime_unavailable", "model_id": model_id, "revision": receipt["revision"], "missing_packages": missing}
    mismatches = {}
    for name, expected in TRANSFORMER_RUNTIME.items():
        try:
            actual = package_version(name)
        except PackageNotFoundError:
            return {"status": "transformers_runtime_unavailable", "model_id": model_id, "revision": receipt["revision"], "missing_packages": [name]}
        if actual != expected:
            mismatches[name] = {"expected": expected, "actual": actual}
    if mismatches:
        return {"status": "transformers_runtime_version_mismatch", "model_id": model_id, "revision": receipt["revision"], "mismatches": mismatches}
    return {"status": "ready", "model_id": model_id, "revision": receipt["revision"]}


def _prediction_row_ids(frame: pd.DataFrame) -> list[str]:
    for column in ("annotation_id", "doc_id"):
        if column in frame:
            return frame[column].fillna("").astype(str).tolist()
    return [f"row-{index}" for index in range(len(frame))]


def _prediction_cache_spec(task: str, frame: pd.DataFrame, split: str, platform: str) -> tuple[Path, dict[str, Any], list[str]]:
    model_id = ROBERTA_MODEL_ID if task == "sentiment" else NLI_MODEL_ID
    receipt = _model_receipt(model_id)
    if receipt is None:
        raise RuntimeError("missing_immutable_receipt")
    row_ids = _prediction_row_ids(frame)
    digest_columns = tuple(column for column in ("annotation_id", "doc_id", "cluster_id", "text_for_annotation") if column in frame)
    source_digest = _frame_digest(frame, digest_columns) if digest_columns else hashlib.sha256(str(len(frame)).encode("utf-8")).hexdigest()
    configuration: dict[str, Any] = {"batch_size": 16 if task == "sentiment" else 8, "max_length": 512, "seed": 20260915}
    if task == "stance":
        configuration.update({"candidate_labels": list(NLI_CANDIDATE_LABELS), "hypothesis_template": NLI_HYPOTHESIS_TEMPLATE, "label_mapping": NLI_LABEL_MAPPING})
    descriptor = {
        "cache_version": PREDICTION_CACHE_VERSION,
        "task": task,
        "model_id": model_id,
        "revision": receipt["revision"],
        "local_sha256": receipt["local_sha256"],
        "split": split,
        "platform": platform,
        "source_digest": source_digest,
        "preprocessing_version": PREPROCESSING_VERSION,
        "configuration": configuration,
    }
    key = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode("utf-8")).hexdigest()
    return PREDICTION_CACHE_ROOT / task / f"{key}.json", descriptor, row_ids


def _read_cached_predictions(path: Path, descriptor: dict[str, Any], row_ids: list[str]) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        payload = read_json(path)
        predicted = payload.get("predicted")
        if payload.get("descriptor") != descriptor or payload.get("row_ids") != row_ids or not isinstance(predicted, list) or len(predicted) != len(row_ids):
            return None
        return np.asarray([str(value) for value in predicted])
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _transformer_pipeline(task: str, receipt: dict[str, Any]) -> Any:
    key = (task, receipt["revision"])
    if key not in _TRANSFORMER_PIPELINES:
        from transformers import pipeline

        task_name = "text-classification" if task == "sentiment" else "zero-shot-classification"
        _TRANSFORMER_PIPELINES[key] = pipeline(
            task_name,
            model=receipt["local_path"],
            tokenizer=receipt["tokenizer_path"],
            revision=receipt["revision"],
            truncation=True,
            device=-1,
        )
    return _TRANSFORMER_PIPELINES[key]


def _normalise_stance_label(value: Any) -> str:
    normalized = " ".join(str(value).casefold().replace("_", " ").split())
    mapping = {" ".join(key.casefold().replace("_", " ").split()): label for key, label in NLI_LABEL_MAPPING.items()}
    if normalized not in mapping:
        raise ValueError(f"unexpected NLI candidate label: {value!r}")
    return mapping[normalized]


def _candidate_registry() -> dict[str, list[dict[str, Any]]]:
    roberta_receipt = _model_receipt(ROBERTA_MODEL_ID)
    nli_receipt = _model_receipt(NLI_MODEL_ID)
    return {
        "sentiment": [
            {"name": "vader", "kind": "transparent_baseline", "required_for_comparison": True},
            {"name": ROBERTA_MODEL_ID, "kind": "transformer_baseline", "revision": roberta_receipt.get("revision") if roberta_receipt else ROBERTA_REVISION, "required_for_comparison": True, "activation": "requires_immutable_model_receipt_and_local_runtime"},
        ],
        "stance": [
            {"name": "tfidf_logistic_regression", "kind": "interpretable_supervised_baseline", "required_for_comparison": True},
            {"name": NLI_MODEL_ID, "kind": "zero_shot_nli", "revision": nli_receipt.get("revision") if nli_receipt else NLI_REVISION, "required_for_comparison": True, "activation": "requires_immutable_model_receipt_and_local_runtime"},
        ],
        "frames": [
            {"name": "tfidf_ovr_logistic_regression", "kind": "supervised_multilabel", "required_for_comparison": True},
            {"name": "transparent_v2_lexicon", "kind": "auditable_sensitivity_only", "required_for_comparison": False},
        ],
    }


def _model_comparison_status(task: str) -> dict[str, Any]:
    candidates = _candidate_registry()[task]
    required = [candidate for candidate in candidates if candidate.get("required_for_comparison") and candidate["name"] in {ROBERTA_MODEL_ID, NLI_MODEL_ID}]
    runtime = {candidate["name"]: _model_runtime_status(candidate["name"]) for candidate in required}
    inactive = [name for name, status in runtime.items() if status["status"] != "ready"]
    return {
        "required": True,
        "status": "model-comparison-required" if inactive else "ready_for_benchmark",
        "inactive_required_candidates": inactive,
        "candidate_status": runtime,
        "required_receipt_fields": MODEL_RECEIPT_FIELDS,
        "receipt_path": relative_path(MODEL_RECEIPTS),
    }


def _transformer_predictions(task: str, frame: pd.DataFrame, split: str = "unknown", platform: str = "all") -> tuple[np.ndarray, str]:
    model_id = ROBERTA_MODEL_ID if task == "sentiment" else NLI_MODEL_ID
    status = _model_runtime_status(model_id)
    if status["status"] != "ready":
        raise RuntimeError(status["status"])
    receipt = _model_receipt(model_id)
    cache_path, descriptor, row_ids = _prediction_cache_spec(task, frame, split, platform)
    cached = _read_cached_predictions(cache_path, descriptor, row_ids)
    if cached is not None:
        return cached, receipt["revision"]
    classifier = _transformer_pipeline(task, receipt)
    texts = frame["text_for_annotation"].astype(str).tolist()
    if task == "sentiment":
        outputs = classifier(texts, batch_size=descriptor["configuration"]["batch_size"], truncation=True, max_length=descriptor["configuration"]["max_length"])
        label_map = {"label_0": "negative", "label_1": "neutral", "label_2": "positive", "negative": "negative", "neutral": "neutral", "positive": "positive"}
        predicted = np.asarray([label_map.get(str(output["label"]).casefold()) for output in outputs])
    else:
        outputs = classifier(
            texts,
            candidate_labels=list(NLI_CANDIDATE_LABELS),
            hypothesis_template=NLI_HYPOTHESIS_TEMPLATE,
            multi_label=False,
            batch_size=descriptor["configuration"]["batch_size"],
            truncation=True,
            max_length=descriptor["configuration"]["max_length"],
        )
        predicted = np.asarray([_normalise_stance_label(output["labels"][0]) for output in outputs])
    if len(predicted) != len(row_ids) or any(value is None for value in predicted):
        raise ValueError("transformer prediction output is incomplete or unmapped")
    write_json(
        cache_path,
        {
            "schema_version": PREDICTION_CACHE_VERSION,
            "created_at_utc": utc_now_iso(),
            "descriptor": descriptor,
            "row_ids": row_ids,
            "predicted": predicted.tolist(),
        },
    )
    return predicted, receipt["revision"]


def _transformer_comparison(task: str, frame: pd.DataFrame, platform: str = "all") -> dict[str, Any]:
    model_id = ROBERTA_MODEL_ID if task == "sentiment" else NLI_MODEL_ID
    status = _model_runtime_status(model_id)
    contract = _sentiment_contract() if task == "sentiment" else {}
    if status["status"] != "ready":
        return {"model": model_id, **status, **contract, "required_for_comparison": True}
    try:
        label_column = "adjudicated_sentiment" if task == "sentiment" else "adjudicated_stance"
        actual = frame[label_column].astype(str)
        if task == "stance":
            actual = actual.str.replace("/", " ", regex=False).str.replace(" ", "_", regex=False)
        folds = _grouped_folds(frame, actual)
        if not folds:
            return {"model": model_id, **status, **contract, "status": "insufficient_grouped_folds", "required_for_comparison": True, "fold_count": 0}
        actual_oof, predicted_oof = [], []
        revisions = set()
        for _, test_index in folds:
            predicted, revision = _transformer_predictions(task, frame.iloc[test_index], "development_oof", platform)
            actual_oof.extend(actual.iloc[test_index].tolist())
            predicted_oof.extend(predicted.tolist())
            revisions.add(revision)
        if len(revisions) != 1:
            raise ValueError("transformer revision changed across grouped folds")
        metrics = _sentiment_metrics_with_abstention(pd.Series(actual_oof), np.asarray(predicted_oof)) if task == "sentiment" else macro_metrics(pd.Series(actual_oof), np.asarray(predicted_oof), STANCE_LABELS)
        return {
            "model": model_id,
            "revision": revisions.pop(),
            "status": "available",
            "required_for_comparison": True,
            "metrics": metrics,
            **contract,
            "automated_claims_allowed": True,
            "mixed_ambiguous_fallback": "weighted_human_sample" if task == "sentiment" else None,
            "preprocessing_version": PREPROCESSING_VERSION,
            "group_field": "cluster_id",
            "fold_count": len(folds),
            "out_of_fold": True,
        }
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        return {"model": model_id, **status, **contract, "status": "comparison_failed", "error_type": type(error).__name__, "required_for_comparison": True}


def _selected_pipeline(baseline_name: str, baseline_metrics: dict[str, Any], comparison: dict[str, Any]) -> str:
    """Prefer a frozen transformer only when it clears the declared 0.02 margin."""

    if comparison.get("status") != "available" or not comparison.get("metrics") or comparison.get("automated_claims_allowed") is False:
        return baseline_name
    baseline_f1 = float(baseline_metrics.get("selection_macro_f1", baseline_metrics.get("macro_f1", 0.0)) or 0.0)
    comparison_f1 = float(comparison["metrics"].get("selection_macro_f1", comparison["metrics"].get("macro_f1", 0.0)) or 0.0)
    return str(comparison["model"]) if comparison_f1 > baseline_f1 + 0.02 else baseline_name


def _label_source_checksums(split: str) -> dict[str, str | None]:
    paths = {
        "coordinator": VALIDATION_ROOT / f"coordinator_labels_{split}.csv",
        "coder_a": VALIDATION_ROOT / f"coder_a_labels_{split}.csv",
        "coder_b": VALIDATION_ROOT / f"coder_b_labels_{split}.csv",
    }
    return {name: sha256_file(path) if path.exists() else None for name, path in paths.items()}


def _selection_checksums() -> dict[str, str | None]:
    return {
        "sentiment": sha256_file(SENTIMENT_SELECTION) if SENTIMENT_SELECTION.exists() else None,
        "stance_frames": sha256_file(STANCE_FRAME_SELECTION) if STANCE_FRAME_SELECTION.exists() else None,
    }


def _labels_ready(split: str, fields: tuple[str, ...]) -> bool:
    frame = _packet(split)
    if not len(frame):
        return False
    for field in fields:
        if field == "frame_labels":
            if not all(_labelled(frame, f"adjudicated_{required_field}").all() for required_field in FRAME_COMPLETION_FIELDS):
                return False
        elif not _labelled(frame, f"adjudicated_{field}").all():
            return False
    return True


def _platform_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "platform" not in frame:
        return {"all": frame}
    return {str(platform): group.reset_index(drop=True) for platform, group in frame.groupby("platform", sort=True)}


def _selection_payload(task: str, fields: tuple[str, ...], selector: Callable[[pd.DataFrame], dict[str, Any]] | None) -> dict[str, Any]:
    frame = _packet("development")
    platform_frames = _platform_frames(frame)
    reliability = {
        "overall": reliability_gates(frame),
        "by_platform": {platform: reliability_gates(group) for platform, group in platform_frames.items()},
    }
    payload: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "measurement_contract_version": MEASUREMENT_CONTRACT_VERSION,
        "created_at_utc": utc_now_iso(),
        "task": task,
        "codebook_version": CODEBOOK_VERSION,
        "split": "development",
        "grouped_cross_validation": {"group_field": "cluster_id", "seed": 20260915, "untouched_evaluation_accessed": False},
        "candidate_pipelines": _candidate_registry()[task],
        "selection_rule": "development macro-F1; transparent lower-complexity tie-break within 0.02",
        "reliability": reliability,
        "label_source_checksums": _label_source_checksums("development"),
        "model_receipt_sha256": sha256_file(MODEL_RECEIPTS) if MODEL_RECEIPTS.exists() else None,
        "platforms": sorted(platform_frames),
        "model_comparison": _model_comparison_status(task),
    }
    if task == "frames":
        payload["frame_evaluation"] = {
            "status": "human-labels-required",
            "frame_rows_total": int(len(frame)),
            "frame_rows_evaluated": None,
            "frame_rows_excluded_no_frame": None,
            "frame_exclusion_policy": "blank adjudicated frame_labels are valid no-supported-frame rows and are included in frame model evaluation",
        }
    if not _labels_ready("development", fields):
        payload.update({"status": "human-labels-required", "gold_label_rows": 0, "primary_pipeline": None, "candidate_result": None, "required_input": "independent coder packets plus coordinator_labels_development.csv with adjudicated labels"})
        return payload
    payload["gold_label_rows"] = int(len(frame))
    if task == "frames":
        payload["frame_evaluation"].update({
            "status": "available",
            "frame_rows_evaluated": int(_frame_evaluation_mask(frame).sum()),
            "frame_rows_excluded_no_frame": int((~_frame_evaluation_mask(frame)).sum()),
        })
    payload["candidate_result"] = {
        "by_platform": {platform: selector(group, platform) for platform, group in platform_frames.items()} if selector else {"status": "selector-unavailable"}
    }
    payload["primary_pipeline"] = {
        platform: result.get("name")
        for platform, result in payload["candidate_result"].get("by_platform", {}).items()
    }
    payload["automated_claims_allowed"] = all(
        result.get("automated_claims_allowed", True)
        for result in payload["candidate_result"].get("by_platform", {}).values()
    )
    unavailable_comparison = payload["model_comparison"]["status"] != "ready_for_benchmark"
    if unavailable_comparison:
        payload["predeclared_fallback"] = {
            "status": "fallback-human-sample",
            "reason": "required_transformer_comparison_unavailable_or_failed",
            "model_comparison": payload["model_comparison"],
        }
    reliability_passed = reliability["overall"]["passed"] and all(item["passed"] for item in reliability["by_platform"].values())
    candidate_results = payload["candidate_result"].get("by_platform", {})
    candidates_ready = bool(candidate_results) and all(
        result.get("status") == "available" and (result.get("metrics") is not None or result.get("macro_f1") is not None)
        for result in candidate_results.values()
    )
    payload["status"] = (
        "selected"
        if reliability_passed and candidates_ready and payload["automated_claims_allowed"] and payload["model_comparison"]["status"] == "ready_for_benchmark"
        else "fallback-human-sample"
        if not reliability_passed or not payload["automated_claims_allowed"]
        else "grouped-development-required"
        if not candidates_ready
        else "fallback-human-sample"
    )
    return payload


def _selection_result_stale(result: dict[str, Any], task: str) -> bool:
    return (
        result.get("schema_version") != SELECTION_SCHEMA_VERSION
        or result.get("measurement_contract_version") != MEASUREMENT_CONTRACT_VERSION
        or result.get("codebook_version") != CODEBOOK_VERSION
        or result.get("candidate_pipelines") != _candidate_registry()[task]
        or result.get("label_source_checksums") != _label_source_checksums("development")
        or result.get("model_receipt_sha256") != (sha256_file(MODEL_RECEIPTS) if MODEL_RECEIPTS.exists() else None)
        or result.get("platforms") != sorted(_platform_frames(_packet("development")))
    )


def _selection_configuration(task: str) -> tuple[Path, tuple[str, ...], Callable[[pd.DataFrame], dict[str, Any]]]:
    if task == "sentiment":
        return SENTIMENT_SELECTION, ("sentiment",), select_vader
    elif task == "stance":
        return STANCE_FRAME_SELECTION, ("stance",), select_stance
    elif task == "frames":
        return STANCE_FRAME_SELECTION, ("frame_labels",), select_frames
    raise ValueError(f"unknown measurement task: {task}")


def build_selection(task: str) -> dict[str, Any]:
    if RESERVE_ACTIVATION.exists() or RESERVE_ASSESSMENT.exists():
        raise ValueError("development selection belongs to a consumed measurement cycle; run validation sample --build --revision <id> to start a new immutable cycle")
    path, fields, selector = _selection_configuration(task)
    if task in {"stance", "frames"}:
        payload = read_json(path) if path.exists() else {"schema_version": SELECTION_SCHEMA_VERSION, "created_at_utc": utc_now_iso(), "codebook_version": CODEBOOK_VERSION, "results": {}}
        payload.update({"schema_version": SELECTION_SCHEMA_VERSION, "codebook_version": CODEBOOK_VERSION, "measurement_contract_version": MEASUREMENT_CONTRACT_VERSION})
        payload.setdefault("results", {})[task] = _selection_payload(task, fields, selector)
        write_json(path, payload)
        result = payload["results"][task]
    else:
        result = _selection_payload(task, fields, selector)
        write_json(path, result)
    return {"status": result["status"], "artifact": relative_path(path), "task": task}


def check_selection(task: str) -> dict[str, Any]:
    path, _, _ = _selection_configuration(task)
    if not path.exists():
        raise FileNotFoundError(f"measurement selection artifact is missing: {path}; run the explicit validation select build first")
    payload = read_json(path)
    result = payload if task == "sentiment" else payload.get("results", {}).get(task)
    if not isinstance(result, dict) or _selection_result_stale(result, task):
        recovery = (
            "run validation sample --build --revision <id> to start a new immutable cycle"
            if RESERVE_ACTIVATION.exists() or RESERVE_ASSESSMENT.exists()
            else "run the explicit validation select build"
        )
        raise ValueError(f"measurement selection is stale or incomplete for {task}; {recovery}")
    return {"status": result["status"], "artifact": relative_path(path), "task": task}


def _stored_selection(task: str) -> dict[str, Any]:
    path = SENTIMENT_SELECTION if task == "sentiment" else STANCE_FRAME_SELECTION
    payload = read_json(path)
    return payload if task == "sentiment" else payload["results"][task]


def _support_deficits(task: str, metrics: dict[str, Any]) -> dict[str, Any]:
    minimum = 30
    if task == "frames":
        deficits = {}
        for label in FRAME_LABELS:
            result = metrics.get("per_label", {}).get(label, {})
            per_class = result.get("unweighted", {}).get("per_class", {})
            deficits[label] = {
                "positive": max(minimum - int(per_class.get("1", {}).get("support", 0)), 0),
                "negative": max(minimum - int(per_class.get("0", {}).get("support", 0)), 0),
            }
        return deficits
    expected = SENTIMENT_LABELS if task == "sentiment" else STANCE_LABELS
    per_class = metrics.get("unweighted", {}).get("per_class", {})
    return {label: max(minimum - int(per_class.get(label, {}).get("support", 0)), 0) for label in expected}


def _evaluation_gate(task: str, metrics: dict[str, Any], *, reserve: bool = False) -> dict[str, Any]:
    failures: list[str] = []
    required_rows_for_support = {
        "sentiment": len(SENTIMENT_LABELS) * 30,
        "stance": len(STANCE_LABELS) * 30,
        "frames": 2 * 30,
    }[task]
    sample_rows = int(metrics.get("sample_rows", 0) or 0)
    support_deficits = _support_deficits(task, metrics)
    if task == "frames":
        top_up_rows_required = max(
            max((value for deficit in support_deficits.values() for value in deficit.values()), default=0),
            required_rows_for_support - sample_rows,
        )
    else:
        top_up_rows_required = max(sum(support_deficits.values()), required_rows_for_support - sample_rows)
    if reserve and top_up_rows_required:
        failures.append("reserve_support_capacity")
    macro_key = "automated_macro_f1" if task == "sentiment" and metrics.get("automated_macro_f1") is not None else "macro_f1"
    macro_f1 = float(metrics.get(macro_key, -1.0))
    bootstrap_key = "cluster_bootstrap_automated_macro_f1" if task == "sentiment" and metrics.get("cluster_bootstrap_automated_macro_f1") else "cluster_bootstrap_macro_f1"
    bootstrap = metrics.get(bootstrap_key, {})
    lower_95 = float(bootstrap.get("lower_95", -1.0))
    if macro_f1 < 0.70:
        failures.append("macro_f1")
    if lower_95 < 0.60:
        failures.append("lower_cluster_bootstrap_bound")
    if task == "frames":
        if int(metrics.get("frame_rows_evaluated", metrics.get("sample_rows", 0)) or 0) < 1:
            failures.append("frame_evaluation_rows")
        per_label = metrics.get("per_label", {})
        if set(per_label) != set(FRAME_LABELS):
            failures.append("per_label_metrics")
        for label, result in per_label.items():
            classes = result.get("unweighted", {}).get("per_class", {})
            positive = classes.get("1", {})
            negative = classes.get("0", {})
            if positive.get("support", 0) < 30:
                failures.append(f"{label}:positive_support")
            if negative.get("support", 0) < 30:
                failures.append(f"{label}:negative_support")
            if positive.get("precision", 0.0) < 0.60 or positive.get("recall", 0.0) < 0.60:
                failures.append(f"{label}:precision_recall")
    else:
        per_class = metrics.get("unweighted", {}).get("per_class", {})
        expected_classes = SENTIMENT_LABELS if task == "sentiment" else STANCE_LABELS
        if not per_class or (task == "sentiment" and set(per_class) != set(SENTIMENT_LABELS)):
            failures.append("per_class_metrics")
        for label in expected_classes:
            result = per_class.get(label, {})
            support = int(result.get("support", 0))
            if support < 30:
                failures.append(f"{label}:support")
            if label != "mixed_ambiguous" and (result.get("precision", 0.0) < 0.60 or result.get("recall", 0.0) < 0.60):
                failures.append(f"{label}:precision_recall")
        if task == "sentiment" and metrics.get("mixed_ambiguous_human_fallback", {}).get("status") != "weighted_human_sample":
            failures.append("mixed_ambiguous_human_fallback")
    return {
        "passed": not failures,
        "failed_gates": failures,
        "task": task,
        "minimum_support_per_class": 30,
        "required_rows_for_support": required_rows_for_support,
        "support_capacity_sufficient": top_up_rows_required == 0,
        "reserve_support_deficits": support_deficits,
        "reserve_top_up_rows_required": int(top_up_rows_required),
        "reserve_top_up_policy": "top_up_deficient_cells_before automated reserve claims" if top_up_rows_required else "no_support_top_up_required",
    }


def _unseen_author_metrics(
    task: str,
    development_authors: set[str],
    evaluation_group: pd.DataFrame,
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
    *,
    reserve: bool = False,
) -> dict[str, Any]:
    if "author_key" not in evaluation_group:
        raise ValueError("evaluation rows must include author_key for unseen-author metrics")
    unseen = ~evaluation_group["author_key"].astype(str).isin(development_authors)
    indexes = np.flatnonzero(unseen.to_numpy())
    authors = set(evaluation_group.iloc[indexes]["author_key"].astype(str))
    if not len(indexes):
        return {"status": "insufficient_unseen_author_rows", "author_count": 0, "sample_rows": 0, "metrics": None, "gates": {"passed": False, "failed_gates": ["unseen_author_rows"], "task": task}}
    clusters = evaluation_group.iloc[indexes]["cluster_id"]
    probabilities = evaluation_group.iloc[indexes]["inclusion_probability"]
    if task == "frames":
        metrics = multilabel_evaluation_metrics(np.asarray(actual)[indexes], np.asarray(predicted)[indexes], clusters, probabilities, FRAME_LABELS)
    else:
        actual_values = actual.iloc[indexes] if isinstance(actual, pd.Series) else pd.Series(np.asarray(actual)[indexes])
        metrics = evaluation_metrics(actual_values, np.asarray(predicted)[indexes], clusters, probabilities, SENTIMENT_LABELS if task == "sentiment" else None)
        if task == "sentiment":
            add_sentiment_abstention_metrics(metrics, actual_values, np.asarray(predicted)[indexes], clusters, probabilities)
            metrics.update(_sentiment_contract())
    metrics["author_count"] = len(authors)
    metrics["status"] = "available"
    metrics["gates"] = _evaluation_gate(task, metrics, reserve=reserve)
    return metrics


def _predict_evaluation(
    task: str,
    development: pd.DataFrame,
    evaluation: pd.DataFrame,
    selection: dict[str, Any],
    split: str = "evaluation",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    results = selection.get("candidate_result", {}).get("by_platform", {})
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: dict[str, Any] = {}
    for platform, eval_group in _platform_frames(evaluation).items():
        dev_group = _platform_frames(development).get(platform)
        candidate = results.get(platform, {})
        if dev_group is None or not candidate:
            raise ValueError(f"selection result is missing platform {platform} for {task}")
        if selection.get("status") != "selected":
            metric_rows[platform] = {
                "status": "fallback-human-sample",
                "sample_rows": 0,
                "fallback_reason": selection.get("predeclared_fallback", {}).get("reason", "selection_status_is_not_selected"),
                "gates": {"passed": False, "failed_gates": ["selection_status"], "task": task},
                "unseen_author": {"status": "not_run_selection_fallback", "author_count": 0, "sample_rows": 0, "metrics": None, "gates": {"passed": False, "failed_gates": ["selection_status"], "task": task}},
            }
            continue
        development_authors = set(dev_group["author_key"].astype(str))
        if "author_key" not in eval_group:
            raise ValueError("evaluation rows must include author_key for unseen-author metrics")
        if task == "sentiment":
            if candidate.get("name") == ROBERTA_MODEL_ID:
                predicted, revision = _transformer_predictions(task, eval_group, split, platform)
            else:
                scores = vader_scores(eval_group["text_for_annotation"])
                thresholds = candidate["selected"]
                predicted = sentiment_from_compound(scores, thresholds["negative_threshold"], thresholds["positive_threshold"])
                revision = None
            actual = eval_group["adjudicated_sentiment"].astype(str)
            metric_rows[platform] = evaluation_metrics(
                actual,
                predicted,
                eval_group["cluster_id"],
                eval_group["inclusion_probability"],
                SENTIMENT_LABELS,
            )
            add_sentiment_abstention_metrics(metric_rows[platform], actual, predicted, eval_group["cluster_id"], eval_group["inclusion_probability"])
            metric_rows[platform].update(candidate.get("label_contract", _sentiment_contract()))
            metric_rows[platform]["automated_claims_allowed"] = candidate.get("automated_claims_allowed", False)
            metric_rows[platform]["model_revision"] = revision
            metric_rows[platform]["gates"] = _evaluation_gate(task, metric_rows[platform], reserve=split == "reserve")
            metric_rows[platform]["unseen_author"] = _unseen_author_metrics(task, development_authors, eval_group, actual, predicted, reserve=split == "reserve")
            for row, value in zip(eval_group.itertuples(index=False), predicted):
                prediction_rows.append({"annotation_id": row.annotation_id, "platform": platform, "task": task, "author_evaluation_scope": "unseen" if str(row.author_key) not in development_authors else "seen", "actual": row.adjudicated_sentiment, "predicted": value})
        elif task == "stance":
            if candidate.get("name") == NLI_MODEL_ID:
                predicted, revision = _transformer_predictions(task, eval_group, split, platform)
            else:
                model = _stance_pipeline().fit(dev_group["text_for_annotation"], dev_group["adjudicated_stance"].astype(str))
                predicted = model.predict(eval_group["text_for_annotation"])
                revision = None
            actual = eval_group["adjudicated_stance"].astype(str)
            metric_rows[platform] = evaluation_metrics(actual, predicted, eval_group["cluster_id"], eval_group["inclusion_probability"], STANCE_LABELS)
            metric_rows[platform]["model_revision"] = revision
            metric_rows[platform]["gates"] = _evaluation_gate(task, metric_rows[platform], reserve=split == "reserve")
            metric_rows[platform]["unseen_author"] = _unseen_author_metrics(task, development_authors, eval_group, actual, predicted, reserve=split == "reserve")
            for row, value in zip(eval_group.itertuples(index=False), predicted):
                prediction_rows.append({"annotation_id": row.annotation_id, "platform": platform, "task": task, "author_evaluation_scope": "unseen" if str(row.author_key) not in development_authors else "seen", "actual": row.adjudicated_stance, "predicted": value})
        elif task == "frames":
            dev_frame_group = dev_group.loc[_frame_evaluation_mask(dev_group)].copy()
            eval_frame_group = eval_group.loc[_frame_evaluation_mask(eval_group)].copy()
            if dev_frame_group.empty:
                raise ValueError("frame evaluation has no adjudicated development rows")
            if eval_frame_group.empty:
                blocked = {"macro_f1": -1.0, "sample_rows": 0, "frame_rows_evaluated": 0, "per_label": {}, "cluster_bootstrap_macro_f1": {"lower_95": -1.0}}
                metric_rows[platform] = {
                    **blocked,
                    "status": "no_frame_rows_for_evaluation",
                    "frame_rows_total": int(len(eval_group)),
                    "frame_rows_excluded_no_frame": 0,
                    "gates": _evaluation_gate("frames", blocked, reserve=split == "reserve"),
                    "unseen_author": {"status": "insufficient_unseen_author_rows", "author_count": 0, "sample_rows": 0, "metrics": None, "gates": {"passed": False, "failed_gates": ["frame_evaluation_rows"], "task": "frames"}},
                }
                continue
            labels = np.asarray([[_frame_labels(value).__contains__(label) for label in FRAME_LABELS] for value in dev_frame_group["adjudicated_frame_labels"]], dtype=int)
            model, _ = _frame_pipeline()
            model.fit(dev_frame_group["text_for_annotation"], labels)
            probabilities = np.asarray(model.predict_proba(eval_frame_group["text_for_annotation"]), dtype=float)
            thresholds = _selected_frame_thresholds(candidate)
            predicted = (probabilities >= thresholds).astype(int)
            actual = np.asarray([[_frame_labels(value).__contains__(label) for label in FRAME_LABELS] for value in eval_frame_group["adjudicated_frame_labels"]], dtype=int)
            metric_rows[platform] = multilabel_evaluation_metrics(actual, predicted, eval_frame_group["cluster_id"], eval_frame_group["inclusion_probability"], FRAME_LABELS)
            metric_rows[platform].update({
                "frame_rows_total": int(len(eval_group)),
                "frame_rows_evaluated": int(len(eval_frame_group)),
                "frame_rows_excluded_no_frame": int(len(eval_group) - len(eval_frame_group)),
                "frame_exclusion_policy": "blank adjudicated frame_labels are valid no-supported-frame rows and are included in frame model evaluation",
            })
            metric_rows[platform]["gates"] = _evaluation_gate(task, metric_rows[platform], reserve=split == "reserve")
            metric_rows[platform]["unseen_author"] = _unseen_author_metrics(task, development_authors, eval_frame_group, actual, predicted, reserve=split == "reserve")
            for row, value in zip(eval_frame_group.itertuples(index=False), predicted):
                prediction_rows.append({"annotation_id": row.annotation_id, "platform": platform, "task": task, "author_evaluation_scope": "unseen" if str(row.author_key) not in development_authors else "seen", "actual": row.adjudicated_frame_labels, "predicted": "|".join(label for label, active in zip(FRAME_LABELS, value) if active)})
        else:
            raise ValueError(f"unknown evaluation task: {task}")
    return pd.DataFrame(prediction_rows), {"by_platform": metric_rows}


def build_evaluation() -> dict[str, Any]:
    sentiment = check_selection("sentiment")
    stance = check_selection("stance")
    frames = check_selection("frames")
    evaluation = _packet("evaluation")
    evaluation_sources = _label_source_checksums("evaluation")
    selection_status = {"sentiment": sentiment["status"], "stance": stance["status"], "frames": frames["status"]}
    selection_checksums = _selection_checksums()
    if EVALUATION_REPORT.exists():
        raise ValueError("the original evaluation opening is preserved; use validation evaluate --check and validation reserve --build")
    ready = all(_labels_ready("evaluation", fields) for fields in (("sentiment",), ("stance",), ("frame_labels",)))
    payload = {
        "schema_version": "measurement-evaluation.v2",
        "created_at_utc": utc_now_iso(),
        "evaluation_access": "single_opening_only",
        "selection_status": selection_status,
        "selection_checksums": selection_checksums,
        "evaluation_rows": int(len(evaluation)),
        "gold_labels_present": ready,
        "evaluation_label_source_checksums": evaluation_sources,
        "gates": {
            "macro_f1": {"minimum": 0.70, "lower_cluster_bootstrap_bound": 0.60},
            "claim_class_precision_recall": {"minimum": 0.60},
            "frame_minimum_positives": 30,
            "sentiment_stance_minimum_examples_per_class": 30,
        },
        "fallback": "weighted human-sample inference or narrower descriptive claim when a gate fails",
    }
    if not ready:
        payload.update({"status": "human-labels-required", "metrics": None, "prediction_artifact": None})
    elif not all(value == "selected" for value in selection_status.values()):
        payload.update({"status": "model-comparison-required", "metrics": None, "prediction_artifact": None})
    else:
        development = _packet("development")
        predictions = []
        metrics = {}
        for task, selection_result in (("sentiment", _stored_selection("sentiment")), ("stance", _stored_selection("stance")), ("frames", _stored_selection("frames"))):
            task_predictions, task_metrics = _predict_evaluation(task, development, evaluation, selection_result)
            predictions.append(task_predictions)
            metrics[task] = task_metrics
        prediction_path = MODEL_ROOT / "evaluation_predictions.csv"
        pd.concat(predictions, ignore_index=True).to_csv(prediction_path, index=False, lineterminator="\n")
        gate_results = {
            task: {
                platform: {
                    "overall": result["gates"],
                    "unseen_author": result["unseen_author"]["gates"],
                }
                for platform, result in task_metrics["by_platform"].items()
            }
            for task, task_metrics in metrics.items()
        }
        passed = all(gate["passed"] for task in gate_results.values() for platform in task.values() for gate in platform.values())
        payload.update({"status": "evaluated" if passed else "fallback-human-sample", "metrics": metrics, "gate_results": gate_results, "prediction_artifact": {"path": relative_path(prediction_path), "sha256": sha256_file(prediction_path), "rows": int(sum(len(part) for part in predictions))}})
    write_json(EVALUATION_REPORT, payload)
    return {"status": payload["status"], "artifact": relative_path(EVALUATION_REPORT), "evaluation_rows": payload["evaluation_rows"]}


def check_evaluation() -> dict[str, Any]:
    selection = {task: check_selection(task) for task in ("sentiment", "stance", "frames")}
    if not EVALUATION_REPORT.exists():
        raise FileNotFoundError(f"evaluation report is missing: {EVALUATION_REPORT}; run the explicit validation evaluate build first")
    evaluation = _packet("evaluation")
    evaluation_sources = _label_source_checksums("evaluation")
    actual = read_json(EVALUATION_REPORT)
    if actual.get("schema_version") != "measurement-evaluation.v2":
        raise ValueError("evaluation report schema is stale; run the explicit validation evaluate build")
    expected_status = {task: value["status"] for task, value in selection.items()}
    if actual.get("evaluation_label_source_checksums") != evaluation_sources:
        raise ValueError("evaluation inputs changed; run the explicit validation evaluate build")
    if actual.get("evaluation_rows") != len(evaluation):
        raise ValueError("evaluation row count changed; run the explicit validation evaluate build")
    prediction = actual.get("prediction_artifact")
    incomplete_opening = prediction is None and actual.get("status") in {"model-comparison-required", "human-labels-required", "fallback-human-sample"}
    if incomplete_opening:
        if not isinstance(actual.get("selection_status"), dict) or not isinstance(actual.get("selection_checksums"), dict):
            raise ValueError("incomplete evaluation opening has no historical selection receipt")
        return {
            "status": actual["status"],
            "opening_status": "preserved_incomplete_opening",
            "artifact": relative_path(EVALUATION_REPORT),
            "evaluation_rows": actual["evaluation_rows"],
            "current_selection_status": expected_status,
        }
    if actual.get("selection_status") != expected_status or actual.get("selection_checksums") != _selection_checksums():
        raise ValueError("evaluation inputs changed; run the explicit validation evaluate build")
    if prediction:
        path = REPO_ROOT / prediction["path"]
        if not path.exists() or sha256_file(path) != prediction.get("sha256"):
            raise ValueError("evaluation prediction artifact changed")
    if actual.get("status") in {"evaluated", "fallback-human-sample"} and not actual.get("gate_results"):
        raise ValueError("evaluation gate results are missing; run the explicit validation evaluate build")
    return {"status": actual["status"], "artifact": relative_path(EVALUATION_REPORT), "evaluation_rows": actual["evaluation_rows"]}


def _reserve_selection_receipt() -> tuple[dict[str, dict[str, Any]], dict[str, str | None]]:
    selection_status = {task: check_selection(task)["status"] for task in ("sentiment", "stance", "frames")}
    invalid = sorted(task for task, status in selection_status.items() if status not in {"selected", "fallback-human-sample"})
    if invalid:
        raise ValueError(f"development selection is not frozen for: {', '.join(invalid)}")
    selections = {task: _stored_selection(task) for task in selection_status}
    for task, selection in selections.items():
        if selection.get("split") not in {None, "development"}:
            raise ValueError(f"selection split is not development for {task}")
        if selection.get("grouped_cross_validation", {}).get("untouched_evaluation_accessed") is not False:
            raise ValueError(f"selection does not prove evaluation isolation for {task}")
        if not selection.get("primary_pipeline"):
            raise ValueError(f"selection has no primary pipeline for {task}")
        if selection.get("status") not in {"selected", "fallback-human-sample"}:
            raise ValueError(f"selection result is not frozen for {task}")
    return selections, _selection_checksums()


def build_reserve_assessment() -> dict[str, Any]:
    """Open and assess the sealed reserve exactly once after development freeze."""

    if RESERVE_ACTIVATION.exists() or RESERVE_ASSESSMENT.exists():
        raise ValueError("reserve assessment has already been opened; use validation reserve --check")
    if not EVALUATION_REPORT.exists():
        raise FileNotFoundError("the preserved evaluation opening is missing")
    evaluation_status = check_evaluation()
    if evaluation_status.get("opening_status") != "preserved_incomplete_opening":
        raise ValueError("reserve contingency requires the preserved incomplete evaluation opening")
    selections, selection_checksums = _reserve_selection_receipt()
    reserve = _packet("reserve")
    if not len(reserve):
        raise ValueError("reserve packet is empty")
    if not all(_labels_ready("reserve", fields) for fields in (("sentiment",), ("stance",), ("frame_labels",))):
        raise ValueError("reserve packet does not contain complete adjudicated labels")
    reserve_sources = _label_source_checksums("reserve")
    activation = {
        "schema_version": "reserve-activation.v1",
        "activated_at_utc": utc_now_iso(),
        "reason": RESERVE_ACTIVATION_REASON,
        "evaluation_artifact": {"path": relative_path(EVALUATION_REPORT), "sha256": sha256_file(EVALUATION_REPORT)},
        "selection_status": {task: selection["status"] for task, selection in selections.items()},
        "selection_checksums": selection_checksums,
        "reserve_label_source_checksums": reserve_sources,
        "model_receipt_sha256": sha256_file(MODEL_RECEIPTS) if MODEL_RECEIPTS.exists() else None,
        "reserve_rows": int(len(reserve)),
        "single_opening": True,
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    write_json(RESERVE_ACTIVATION, activation)
    try:
        development = _packet("development")
        prediction_parts: list[pd.DataFrame] = []
        metrics: dict[str, Any] = {}
        for task in ("sentiment", "stance", "frames"):
            task_predictions, task_metrics = _predict_evaluation(task, development, reserve, selections[task], split="reserve")
            if not task_predictions.empty:
                prediction_parts.append(task_predictions)
            metrics[task] = task_metrics
        prediction_columns = ["annotation_id", "platform", "task", "author_evaluation_scope", "actual", "predicted"]
        predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame(columns=prediction_columns)
        predictions.to_csv(RESERVE_PREDICTIONS, index=False, lineterminator="\n")
        gate_results = {
            task: {
                platform: {
                    "overall": result["gates"],
                    "unseen_author": result["unseen_author"]["gates"],
                }
                for platform, result in task_metrics["by_platform"].items()
            }
            for task, task_metrics in metrics.items()
        }
        passed = all(gate["passed"] for task in gate_results.values() for platform in task.values() for gate in platform.values())
        status = "evaluated" if passed else "fallback-human-sample"
        payload = {
            "schema_version": "measurement-reserve.v1",
            "measurement_contract_version": MEASUREMENT_CONTRACT_VERSION,
            "assessed_at_utc": utc_now_iso(),
            "evaluation_access": "reserve_single_opening_only",
            "opening_status": "opened_once",
            "status": status,
            "activation_reason": RESERVE_ACTIVATION_REASON,
            "activation_receipt": {"path": relative_path(RESERVE_ACTIVATION), "sha256": sha256_file(RESERVE_ACTIVATION)},
            "evaluation_opening": evaluation_status,
            "selection_status": activation["selection_status"],
            "selection_checksums": selection_checksums,
            "reserve_rows": int(len(reserve)),
            "reserve_label_source_checksums": reserve_sources,
            "model_receipt_sha256": sha256_file(MODEL_RECEIPTS) if MODEL_RECEIPTS.exists() else None,
            "preprocessing_version": PREPROCESSING_VERSION,
            "metrics": metrics,
            "gate_results": gate_results,
            "calibration": {"status": "not_applicable", "reason": "selected instruments persist hard labels; no calibrated probability model was declared"},
            "prediction_artifact": {"path": relative_path(RESERVE_PREDICTIONS), "sha256": sha256_file(RESERVE_PREDICTIONS), "rows": int(len(predictions))},
            "fallback": "weighted human-sample inference or narrower descriptive claim when a reserve gate fails",
        }
    except Exception as error:
        payload = {
            "schema_version": "measurement-reserve.v1",
            "measurement_contract_version": MEASUREMENT_CONTRACT_VERSION,
            "assessed_at_utc": utc_now_iso(),
            "evaluation_access": "reserve_single_opening_only",
            "opening_status": "opened_once",
            "status": "assessment_failed",
            "activation_reason": RESERVE_ACTIVATION_REASON,
            "activation_receipt": {"path": relative_path(RESERVE_ACTIVATION), "sha256": sha256_file(RESERVE_ACTIVATION)},
            "selection_status": activation["selection_status"],
            "selection_checksums": selection_checksums,
            "reserve_rows": int(len(reserve)),
            "reserve_label_source_checksums": reserve_sources,
            "model_receipt_sha256": sha256_file(MODEL_RECEIPTS) if MODEL_RECEIPTS.exists() else None,
            "error_type": type(error).__name__,
            "error": str(error),
            "prediction_artifact": None,
            "interpretation": "The reserve opening is consumed and must not be repeated; resolve the recorded failure manually before making measurement claims.",
        }
    write_json(RESERVE_ASSESSMENT, payload)
    seal = seal_reserve_assessment()
    return {"status": payload["status"], "artifact": relative_path(RESERVE_ASSESSMENT), "reserve_rows": payload["reserve_rows"], "seal": seal["sha256"]}


def _reserve_assessment_content(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in ("metrics", "gate_results", "reserve_label_source_checksums", "activation_receipt", "model_receipt_sha256", "prediction_artifact")}


def _reserve_content_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(_reserve_assessment_content(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def seal_reserve_assessment() -> dict[str, Any]:
    """Write a detached digest for the immutable, single-opening assessment."""

    if not RESERVE_ACTIVATION.exists() or not RESERVE_ASSESSMENT.exists():
        raise FileNotFoundError("cannot seal a reserve assessment before activation and assessment exist")
    payload = read_json(RESERVE_ASSESSMENT)
    if payload.get("schema_version") != "measurement-reserve.v1":
        raise ValueError("cannot seal an invalid reserve assessment")
    seal = {
        "schema_version": "reserve-assessment-seal.v1",
        "assessment": {"path": relative_path(RESERVE_ASSESSMENT), "sha256": sha256_file(RESERVE_ASSESSMENT)},
        "assessment_payload_sha256": sha256_file(RESERVE_ASSESSMENT),
        "activation": {"path": relative_path(RESERVE_ACTIVATION), "sha256": sha256_file(RESERVE_ACTIVATION)},
        "measurement_contract_version": payload.get("measurement_contract_version"),
        "selection_checksums": payload.get("selection_checksums"),
        "content_sha256": _reserve_content_sha256(payload),
        "reserve_rows": payload.get("reserve_rows"),
        "status": payload.get("status"),
        "sealed_at_utc": utc_now_iso(),
    }
    write_json(RESERVE_ASSESSMENT_SEAL, seal)
    return {"path": relative_path(RESERVE_ASSESSMENT_SEAL), "sha256": sha256_file(RESERVE_ASSESSMENT_SEAL)}


def check_reserve_assessment() -> dict[str, Any]:
    check_evaluation()
    _, selection_checksums = _reserve_selection_receipt()
    if not RESERVE_ACTIVATION.exists():
        raise FileNotFoundError(f"reserve activation receipt is missing: {RESERVE_ACTIVATION}; run the explicit validation reserve build")
    activation = read_json(RESERVE_ACTIVATION)
    if activation.get("schema_version") != "reserve-activation.v1" or activation.get("reason") != RESERVE_ACTIVATION_REASON or activation.get("single_opening") is not True:
        raise ValueError("reserve activation receipt is invalid")
    if activation.get("selection_checksums") != selection_checksums:
        raise ValueError("development selection changed after reserve activation")
    current_model_receipt = sha256_file(MODEL_RECEIPTS) if MODEL_RECEIPTS.exists() else None
    if activation.get("model_receipt_sha256") != current_model_receipt:
        raise ValueError("reserve activation model receipt changed")
    if activation.get("evaluation_artifact", {}).get("sha256") != sha256_file(EVALUATION_REPORT):
        raise ValueError("preserved evaluation opening changed after reserve activation")
    reserve = _packet("reserve")
    if activation.get("reserve_rows") != len(reserve) or activation.get("reserve_label_source_checksums") != _label_source_checksums("reserve"):
        raise ValueError("reserve labels or row count changed after activation")
    if not RESERVE_ASSESSMENT.exists():
        raise ValueError("reserve was opened but its final assessment artifact is missing; refusing a second opening")
    payload = read_json(RESERVE_ASSESSMENT)
    if payload.get("schema_version") != "measurement-reserve.v1" or payload.get("measurement_contract_version") != MEASUREMENT_CONTRACT_VERSION or payload.get("activation_reason") != RESERVE_ACTIVATION_REASON or payload.get("selection_checksums") != selection_checksums:
        raise ValueError("reserve assessment is stale or missing its activation contract")
    if payload.get("model_receipt_sha256") != current_model_receipt:
        raise ValueError("reserve assessment model receipt changed")
    expected_activation_receipt = {"path": relative_path(RESERVE_ACTIVATION), "sha256": sha256_file(RESERVE_ACTIVATION)}
    if payload.get("activation_receipt") != expected_activation_receipt:
        raise ValueError("reserve assessment activation receipt changed")
    if payload.get("reserve_label_source_checksums") != activation.get("reserve_label_source_checksums"):
        raise ValueError("reserve assessment label sources changed")
    prediction = payload.get("prediction_artifact")
    if prediction:
        if prediction.get("path") != relative_path(RESERVE_PREDICTIONS):
            raise ValueError("reserve prediction artifact path is invalid")
        path = REPO_ROOT / prediction["path"]
        if not path.exists() or sha256_file(path) != prediction.get("sha256"):
            raise ValueError("reserve prediction artifact changed")
    if not RESERVE_ASSESSMENT_SEAL.exists():
        raise ValueError("reserve assessment seal is missing; refusing to trust editable reserve metrics")
    seal = read_json(RESERVE_ASSESSMENT_SEAL)
    if (
        seal.get("schema_version") != "reserve-assessment-seal.v1"
        or seal.get("assessment", {}).get("path") != relative_path(RESERVE_ASSESSMENT)
        or seal.get("assessment", {}).get("sha256") != sha256_file(RESERVE_ASSESSMENT)
        or seal.get("assessment_payload_sha256") != sha256_file(RESERVE_ASSESSMENT)
        or seal.get("activation", {}).get("path") != relative_path(RESERVE_ACTIVATION)
        or seal.get("activation", {}).get("sha256") != sha256_file(RESERVE_ACTIVATION)
        or seal.get("measurement_contract_version") != MEASUREMENT_CONTRACT_VERSION
        or seal.get("selection_checksums") != selection_checksums
        or seal.get("content_sha256") != _reserve_content_sha256(payload)
        or seal.get("reserve_rows") != payload.get("reserve_rows")
        or seal.get("status") != payload.get("status")
    ):
        raise ValueError("reserve assessment seal does not match the activated assessment")
    return {"status": payload["status"], "artifact": relative_path(RESERVE_ASSESSMENT), "reserve_rows": payload["reserve_rows"]}
