from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.analysis.measurement as measurement
from src.analysis.measurement import evaluation_metrics, macro_metrics, reliability_gates, sentiment_from_compound


def test_vader_thresholds_are_candidate_config_not_sentiment_truth():
    scores = np.array([-0.8, -0.01, 0.0, 0.8])
    assert sentiment_from_compound(scores, -0.05, 0.05).tolist() == ["negative", "neutral", "neutral", "positive"]


def test_missing_double_coding_fails_closed():
    report = reliability_gates(pd.DataFrame({"doc_id": ["a"]}))
    assert report["passed"] is False
    assert "frame_labels" in report["failed_fields"]


def test_multilabel_reliability_has_overall_and_per_frame_agreement():
    frame = pd.DataFrame(
        {
            **{
                f"coder_{coder}_{field}": ["relevant"] * 4
                for coder in ("a", "b")
                for field in ("relevance", "language", "target_policy", "stance", "sentiment")
            },
            "coder_a_bypass_techniques": ["none_unclear"] * 4,
            "coder_b_bypass_techniques": ["none_unclear"] * 4,
            "coder_a_frame_labels": ["privacy_surveillance"] * 4,
            "coder_b_frame_labels": ["privacy_surveillance"] * 4,
        }
    )
    report = reliability_gates(frame)
    assert report["fields"]["frame_labels"]["raw_agreement"] == 1.0
    assert report["fields"]["frame_labels"]["cohen_kappa"] == 1.0
    assert report["passed"] is True


def test_multilabel_exact_set_agreement_is_descriptive_not_a_gate():
    rows = []
    for index in range(20):
        labels = [label for position, label in enumerate(measurement.FRAME_LABELS) if (index + position) % 2 == 0]
        rows.append(
            {
                **{
                    f"coder_{coder}_{field}": "relevant"
                    for coder in ("a", "b")
                    for field in ("relevance", "language", "target_policy", "stance", "sentiment")
                },
                **{
                    f"coder_{coder}_bypass_techniques": "none_unclear"
                    for coder in ("a", "b")
                },
                "coder_a_frame_labels": "|".join(labels),
                "coder_b_frame_labels": "|".join(labels),
            }
        )
    for index, label in enumerate(measurement.FRAME_LABELS):
        values = set(rows[index]["coder_b_frame_labels"].split("|"))
        values.remove(label) if label in values else values.add(label)
        rows[index]["coder_b_frame_labels"] = "|".join(frame for frame in measurement.FRAME_LABELS if frame in values)

    report = reliability_gates(pd.DataFrame(rows))

    assert report["fields"]["frame_labels"]["raw_agreement"] == 0.75
    assert report["passed"] is True


def test_blank_frame_labels_are_valid_no_frame_rows_for_reliability():
    frame = pd.DataFrame(
        {
            **{
                f"coder_{coder}_{field}": ["relevant"]
                for coder in ("a", "b")
                for field in ("relevance", "language", "target_policy", "stance", "sentiment")
            },
            "coder_a_bypass_techniques": ["none_unclear"],
            "coder_b_bypass_techniques": ["none_unclear"],
            "coder_a_frame_labels": [""],
            "coder_b_frame_labels": [""],
        }
    )

    report = reliability_gates(frame)

    assert report["fields"]["frame_labels"]["status"] == "available"
    assert report["fields"]["frame_labels"]["frame_rows_excluded_no_frame"] == 1


def test_frame_readiness_allows_explicit_blank_no_frame_rows(monkeypatch):
    frame = pd.DataFrame(
        {
            **{f"adjudicated_{field}": ["relevant"] for field in measurement.FRAME_COMPLETION_FIELDS},
            "adjudicated_frame_labels": [""],
        }
    )
    monkeypatch.setattr(measurement, "_packet", lambda split: frame)

    assert measurement._labels_ready("development", ("frame_labels",)) is True


def test_partial_double_coding_fails_closed():
    frame = pd.DataFrame(
        {
            "coder_a_sentiment": ["positive", ""],
            "coder_b_sentiment": ["positive", "negative"],
        }
    )
    assert reliability_gates(frame)["fields"]["sentiment"]["status"] == "incomplete-double-coding"


def test_evaluation_gate_enforces_zero_support_and_class_metrics():
    metrics = {
        "macro_f1": 0.80,
        "cluster_bootstrap_macro_f1": {"lower_95": 0.70},
        "unweighted": {"per_class": {"positive": {"support": 0, "precision": 1.0, "recall": 1.0}}},
    }
    gate = measurement._evaluation_gate("sentiment", metrics)
    assert gate["passed"] is False
    assert "positive:support" in gate["failed_gates"]


def test_macro_metrics_exposes_per_class_support():
    metrics = macro_metrics(pd.Series(["positive", "negative", "neutral"]), np.array(["positive", "negative", "negative"]), ["positive", "negative", "neutral"])
    assert metrics["per_class"]["neutral"]["support"] == 1
    assert 0 <= metrics["macro_f1"] <= 1


def test_four_class_sentiment_metrics_keep_mixed_ambiguous_support():
    metrics = macro_metrics(
        pd.Series(["positive", "negative", "neutral", "mixed_ambiguous"]),
        np.array(["positive", "negative", "neutral", "neutral"]),
        measurement.SENTIMENT_LABELS,
    )

    assert set(metrics["per_class"]) == set(measurement.SENTIMENT_LABELS)
    assert metrics["per_class"]["mixed_ambiguous"]["support"] == 1
    assert metrics["per_class"]["mixed_ambiguous"]["predicted_support"] == 0


def test_three_class_sentiment_candidate_explicitly_abstains_on_mixed_class():
    actual = pd.Series(["positive", "negative", "neutral", "mixed_ambiguous"])
    configs = measurement._vader_candidate_results(actual, np.array([0.8, -0.8, 0.0, 0.01]), [(np.arange(4), np.arange(4))])

    assert configs
    assert configs[0]["metrics"]["per_class"]["mixed_ambiguous"]["support"] == 1
    assert configs[0]["label_contract"]["automated_claims_allowed"] is True
    assert "mixed_ambiguous" in configs[0]["label_contract"]["unsupported_labels"]
    assert configs[0]["label_contract"]["mixed_ambiguous_fallback"] == "weighted_human_sample"


def test_vader_candidate_metrics_evaluate_the_candidate_being_named(monkeypatch):
    frame = pd.DataFrame(
        {
            "adjudicated_sentiment": ["negative", "neutral", "positive", "neutral", "negative", "positive", "neutral", "positive"],
            "text_for_annotation": [str(index) for index in range(8)],
            "cluster_id": [str(index) for index in range(8)],
            "doc_id": [str(index) for index in range(8)],
        }
    )
    monkeypatch.setattr(measurement, "vader_scores", lambda texts: np.array([-0.8, -0.12, -0.06, 0.0, -0.08, 0.08, 0.12, 0.8]))
    monkeypatch.setattr(measurement, "_grouped_folds", lambda frame, labels: [(np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7]))])

    result = measurement.select_vader(frame)

    scores = {tuple((item["negative_threshold"], item["positive_threshold"])): item["metrics"]["macro_f1"] for item in result["candidates"]}
    assert len(set(scores.values())) > 1


def test_frame_selection_reports_positive_support(monkeypatch):
    rows = []
    for index in range(50):
        labels = [label for position, label in enumerate(measurement.FRAME_LABELS) if (index + position) % 2 == 0]
        rows.append({"text_for_annotation": f"document token {index % 5}", "adjudicated_frame_labels": "|".join(labels), "cluster_id": str(index)})
    frame = pd.DataFrame(rows)
    monkeypatch.setattr(measurement, "_grouped_multilabel_fold_plan", lambda frame, labels: ([(np.arange(0, 25), np.arange(25, 50)), (np.arange(25, 50), np.arange(0, 25))], {"fold_count": 2, "group_field": "cluster_id", "strict_group_separation": True}))

    result = measurement.select_frames(frame)

    assert result["status"] == "available"
    assert all(item["support"] == 25 for item in result["per_frame"].values())


def test_frame_selection_includes_blank_no_frame_rows_and_reports_denominator(monkeypatch):
    rows = []
    for index in range(50):
        labels = [label for position, label in enumerate(measurement.FRAME_LABELS) if (index + position) % 2 == 0]
        rows.append({"text_for_annotation": f"document token {index % 5}", "adjudicated_frame_labels": "|".join(labels), "cluster_id": str(index)})
    rows.extend(
        {"text_for_annotation": "document token no frame", "adjudicated_frame_labels": "", "cluster_id": f"blank-{index}"}
        for index in range(2)
    )
    frame = pd.DataFrame(rows)
    monkeypatch.setattr(measurement, "_grouped_multilabel_fold_plan", lambda frame, labels: ([(np.arange(0, 26), np.arange(26, 52)), (np.arange(26, 52), np.arange(0, 26))], {"fold_count": 2, "group_field": "cluster_id", "strict_group_separation": True}))

    result = measurement.select_frames(frame)

    assert result["status"] == "available"
    assert result["frame_rows_total"] == 52
    assert result["frame_rows_evaluated"] == 52
    assert result["frame_rows_excluded_no_frame"] == 0


def test_design_weighted_macro_f1_is_unweighted_across_classes():
    metrics = evaluation_metrics(
        pd.Series(["positive"] * 9 + ["negative"]),
        np.array(["positive"] * 10),
        pd.Series([f"thread-{index}" for index in range(10)]),
        pd.Series([1.0] * 10),
        ["positive", "negative"],
        bootstrap_replicates=20,
    )

    weighted = metrics["design_weighted"]
    assert weighted["macro_f1"] == pytest.approx(0.4736842105)
    assert weighted["support_weighted_f1"] > weighted["macro_f1"]


def test_reserve_gate_reports_support_capacity_limit():
    gate = measurement._evaluation_gate("stance", {"sample_rows": 100, "macro_f1": 1.0, "cluster_bootstrap_macro_f1": {"lower_95": 1.0}, "unweighted": {"per_class": {}}}, reserve=True)

    assert gate["support_capacity_sufficient"] is False
    assert gate["required_rows_for_support"] == 150
    assert gate["reserve_top_up_rows_required"] == 150
    assert "reserve_support_capacity" in gate["failed_gates"]


def test_selected_frame_thresholds_are_complete_and_frozen():
    candidate = {"thresholds": {label: threshold for label, threshold in zip(measurement.FRAME_LABELS, measurement.FRAME_THRESHOLD_CANDIDATES)}}

    assert measurement._selected_frame_thresholds(candidate).tolist() == list(measurement.FRAME_THRESHOLD_CANDIDATES)
    with pytest.raises(ValueError, match="missing or incomplete"):
        measurement._selected_frame_thresholds({"thresholds": {measurement.FRAME_LABELS[0]: 0.50}})
    with pytest.raises(ValueError, match="outside the frozen candidate set"):
        measurement._selected_frame_thresholds({"thresholds": {label: 0.55 for label in measurement.FRAME_LABELS}})


def test_reserve_seal_binds_metrics_and_gate_content(tmp_path, monkeypatch):
    activation_path = tmp_path / "reserve_activation.json"
    assessment_path = tmp_path / "reserve_assessment.json"
    seal_path = tmp_path / "reserve_assessment.seal.json"
    monkeypatch.setattr(measurement, "RESERVE_ACTIVATION", activation_path)
    monkeypatch.setattr(measurement, "RESERVE_ASSESSMENT", assessment_path)
    monkeypatch.setattr(measurement, "RESERVE_ASSESSMENT_SEAL", seal_path)
    measurement.write_json(activation_path, {"schema_version": "reserve-activation.v1", "single_opening": True})
    measurement.write_json(
        assessment_path,
        {
            "schema_version": "measurement-reserve.v1",
            "measurement_contract_version": measurement.MEASUREMENT_CONTRACT_VERSION,
            "metrics": {"frames": {"by_platform": {"reddit": {"gates": {"passed": False}}}}},
            "gate_results": {"frames": {"reddit": {"overall": {"passed": False}}}},
            "reserve_label_source_checksums": {"labels": "a"},
            "activation_receipt": {"path": str(activation_path), "sha256": measurement.sha256_file(activation_path)},
            "model_receipt_sha256": None,
            "prediction_artifact": None,
            "reserve_rows": 100,
            "status": "fallback-human-sample",
        },
    )

    measurement.seal_reserve_assessment()
    seal = measurement.read_json(seal_path)
    assert seal["content_sha256"] == measurement._reserve_content_sha256(measurement.read_json(assessment_path))

    changed = measurement.read_json(assessment_path)
    changed["metrics"]["frames"]["by_platform"]["reddit"]["gates"]["passed"] = True
    measurement.write_json(assessment_path, changed)
    assert seal["content_sha256"] != measurement._reserve_content_sha256(changed)


def test_grouped_multilabel_folds_preserve_groups_and_binary_support():
    frame = pd.DataFrame({"cluster_id": [f"cluster-{index}" for index in range(30)]})
    labels = np.asarray(
        [[int((index + offset) % 3 == 0) for offset in range(len(measurement.FRAME_LABELS))] for index in range(len(frame))]
    )

    folds, plan = measurement._grouped_multilabel_fold_plan(frame, labels)

    assert plan["strict_group_separation"] is True
    assert len(folds) == 5
    test_groups = [set(frame.iloc[test]["cluster_id"]) for _, test in folds]
    assert len(set.union(*test_groups)) == len(frame)
    assert sum(map(len, test_groups)) == len(frame)
    assert all(np.all(labels[test].sum(axis=0) > 0) and np.all(labels[test].sum(axis=0) < len(test)) for _, test in folds)


def test_transformer_predictions_reuse_checksum_keyed_cache(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "weights.bin").write_bytes(b"frozen")
    revision = "b" * 40
    receipt = {
        "model_id": measurement.ROBERTA_MODEL_ID,
        "revision": revision,
        "local_path": str(model_path),
        "local_sha256": measurement._sha256_path(model_path),
        "tokenizer_path": str(model_path),
        "tokenizer_revision": revision,
        "model_card_url": "https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest",
        "source_url": "https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest/commit/" + revision,
    }
    receipts_path = tmp_path / "model_receipts.json"
    measurement.write_json(receipts_path, {"models": {measurement.ROBERTA_MODEL_ID: receipt}})
    monkeypatch.setattr(measurement, "MODEL_RECEIPTS", receipts_path)
    monkeypatch.setattr(measurement, "PREDICTION_CACHE_ROOT", tmp_path / "cache")
    measurement._TRANSFORMER_PIPELINES.clear()

    class FakeClassifier:
        calls = 0

        def __call__(self, texts, **kwargs):
            self.calls += 1
            return [{"label": "LABEL_0"} for _ in texts]

    classifier = FakeClassifier()
    monkeypatch.setattr(measurement, "_transformer_pipeline", lambda task, model_receipt: classifier)
    frame = pd.DataFrame({"annotation_id": ["a1", "a2"], "text_for_annotation": ["masked one", "masked two"]})

    first, _ = measurement._transformer_predictions("sentiment", frame, "reserve", "bluesky")
    second, _ = measurement._transformer_predictions("sentiment", frame, "reserve", "bluesky")

    assert first.tolist() == ["negative", "negative"]
    assert second.tolist() == first.tolist()
    assert classifier.calls == 1
    assert len(list((tmp_path / "cache").rglob("*.json"))) == 1


def test_reserve_assessment_does_not_open_before_selection_freeze(tmp_path, monkeypatch):
    activation = tmp_path / "reserve_activation.json"
    monkeypatch.setattr(measurement, "RESERVE_ACTIVATION", activation)
    monkeypatch.setattr(measurement, "RESERVE_ASSESSMENT", tmp_path / "reserve_assessment.json")
    monkeypatch.setattr(measurement, "check_evaluation", lambda: {"opening_status": "preserved_incomplete_opening"})
    monkeypatch.setattr(measurement, "_reserve_selection_receipt", lambda: (_ for _ in ()).throw(ValueError("selection is not frozen")))

    with pytest.raises(ValueError, match="selection is not frozen"):
        measurement.build_reserve_assessment()

    assert not activation.exists()


def test_reserve_assessment_cannot_be_opened_twice(tmp_path, monkeypatch):
    activation = tmp_path / "reserve_activation.json"
    activation.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(measurement, "RESERVE_ACTIVATION", activation)
    monkeypatch.setattr(measurement, "RESERVE_ASSESSMENT", tmp_path / "reserve_assessment.json")

    with pytest.raises(ValueError, match="already been opened"):
        measurement.build_reserve_assessment()


def test_consumed_cycle_reports_versioned_recovery_path(tmp_path, monkeypatch):
    activation = tmp_path / "reserve_activation.json"
    activation.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(measurement, "RESERVE_ACTIVATION", activation)
    monkeypatch.setattr(measurement, "RESERVE_ASSESSMENT", tmp_path / "reserve_assessment.json")

    with pytest.raises(ValueError, match="validation sample --build --revision"):
        measurement.build_selection("sentiment")


def test_evaluation_metrics_exposes_confusion_weights_and_cluster_interval():
    metrics = evaluation_metrics(
        pd.Series(["positive", "negative", "positive", "negative"]),
        np.array(["positive", "positive", "positive", "negative"]),
        pd.Series(["thread-a", "thread-a", "thread-b", "thread-b"]),
        pd.Series([0.5, 0.5, 1.0, 1.0]),
        ["positive", "negative"],
        bootstrap_replicates=20,
    )

    assert metrics["confusion_matrix"]["counts"] == [[2, 0], [1, 1]]
    assert metrics["design_weighted"] is not None
    assert metrics["cluster_bootstrap_macro_f1"]["replicates"] == 20


def test_unseen_author_metrics_separate_overlapping_authors():
    evaluation = pd.DataFrame(
        {
            "author_key": ["seen", "new", "new"],
            "cluster_id": ["c1", "c2", "c3"],
            "inclusion_probability": [1.0, 1.0, 1.0],
        }
    )

    result = measurement._unseen_author_metrics(
        "sentiment",
        {"seen"},
        evaluation,
        pd.Series(["positive", "negative", "negative"]),
        np.array(["positive", "negative", "positive"]),
    )

    assert result["status"] == "available"
    assert result["author_count"] == 1
    assert result["sample_rows"] == 2


def test_unseen_author_metrics_explicitly_blocks_when_all_authors_overlap():
    evaluation = pd.DataFrame({"author_key": ["seen"], "cluster_id": ["c1"], "inclusion_probability": [1.0]})

    result = measurement._unseen_author_metrics(
        "sentiment",
        {"seen"},
        evaluation,
        pd.Series(["positive"]),
        np.array(["positive"]),
    )

    assert result["status"] == "insufficient_unseen_author_rows"
    assert result["gates"]["passed"] is False


def test_required_transformer_comparison_is_explicitly_receipt_gated(tmp_path, monkeypatch):
    monkeypatch.setattr(measurement, "MODEL_RECEIPTS", tmp_path / "missing-model-receipts.json")
    result = measurement._model_comparison_status("sentiment")

    assert result["status"] == "model-comparison-required"
    assert measurement.ROBERTA_MODEL_ID in result["inactive_required_candidates"]
    assert result["required_receipt_fields"] == measurement.MODEL_RECEIPT_FIELDS


def test_model_receipt_requires_immutable_artifact_and_source_metadata(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "weights.bin").write_bytes(b"frozen model")
    receipt_path = tmp_path / "model_receipts.json"
    monkeypatch.setattr(measurement, "MODEL_RECEIPTS", receipt_path)
    revision = "a" * 40
    record = {
        "model_id": measurement.ROBERTA_MODEL_ID,
        "revision": revision,
        "local_path": str(model_path),
        "local_sha256": measurement._sha256_path(model_path),
        "tokenizer_path": str(model_path),
        "tokenizer_revision": revision,
        "model_card_url": "https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest",
        "source_url": "https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest/commit/" + revision,
    }
    measurement.write_json(receipt_path, {"models": {measurement.ROBERTA_MODEL_ID: record}})

    assert measurement._model_receipt(measurement.ROBERTA_MODEL_ID)["revision"] == revision
    record["local_sha256"] = "0" * 64
    measurement.write_json(receipt_path, {"models": {measurement.ROBERTA_MODEL_ID: record}})
    assert measurement._model_receipt(measurement.ROBERTA_MODEL_ID) is None
