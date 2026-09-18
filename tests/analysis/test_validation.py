from __future__ import annotations

import pandas as pd
import pytest

import src.analysis.validation as validation
from src.analysis.validation import _blind_packet_frame, _coder_packet_frame, assign_context_splits, load_label_packet, select_split_samples


def test_context_container_has_one_split():
    candidates = pd.DataFrame(
        [
            {"doc_id": "d1", "cluster_id": "cluster", "author_key": "a1", "text_digest": "t1", "stratum": "G_GENERAL"},
            {"doc_id": "d2", "cluster_id": "cluster", "author_key": "a2", "text_digest": "t2", "stratum": "G_GENERAL"},
        ]
    )
    assigned = assign_context_splits(candidates)
    assert assigned["split"].nunique() == 1


def test_reddit_candidate_context_uses_unfiltered_unified_text(monkeypatch):
    rows = [
        {
            "record_id": "root",
            "thing": "post",
            "event_id": "australia_legislation",
            "author_hash": "root-author",
            "thread_id": "root",
            "parent_record_id": "",
            "schema_valid": False,
            "human_only": False,
            "is_english": True,
            "is_language_uncertain": False,
            "text_topic": "root topic view",
            "text_sentiment": "root don't view",
            "is_url_only": False,
            "is_no_substantive_text": False,
            "sentiment_compound": 0.0,
        },
        {
            "record_id": "reply",
            "thing": "comment",
            "event_id": "australia_legislation",
            "author_hash": "reply-author",
            "thread_id": "root",
            "parent_record_id": "root",
            "schema_valid": True,
            "human_only": True,
            "is_english": True,
            "is_language_uncertain": False,
            "text_topic": "reply topic view",
            "text_sentiment": "reply don't view",
            "is_url_only": False,
            "is_no_substantive_text": False,
            "sentiment_compound": 0.0,
        },
    ]
    monkeypatch.setattr(validation.pd, "read_parquet", lambda path, columns=None: pd.DataFrame(rows))

    result = validation._base_frame("reddit")

    row = result.iloc[0]
    assert row["text_for_annotation"] == "reply don't view"
    assert row["parent_context"] == "root don't view"
    assert row["root_context"] == "root don't view"


def test_sampling_metadata_reports_empty_parent_context_counts():
    sample = pd.DataFrame(
        {
            "platform": ["reddit", "reddit", "reddit"],
            "split": ["development", "evaluation", "reserve"],
            "stratum": ["G_GENERAL"] * 3,
            "author_key": ["a", "b", "c"],
            "parent_context": ["", "parent", ""],
        }
    )

    metadata = validation._sampling_metadata(sample, {})

    assert metadata["empty_parent_context_counts"]["reddit"] == {"development": 1, "evaluation": 0, "reserve": 1}


def test_global_caps_hold_when_sampling_a_split():
    candidates = pd.DataFrame(
        [
            {"doc_id": f"d{i}", "cluster_id": f"c{i // 3}", "author_key": f"a{i // 2}", "text_digest": f"t{i}", "stratum": "G_GENERAL"}
            for i in range(900)
        ]
    )
    assigned = assign_context_splits(candidates)
    sampled, _ = select_split_samples(assigned, "development")
    assert len(sampled) == 200
    assert sampled.groupby("author_key").size().max() <= 2
    assert sampled.groupby("cluster_id").size().max() <= 10


def test_sampling_draw_and_weights_are_reproducible_and_multiplicative():
    candidates = pd.DataFrame(
        [
            {"doc_id": f"d{i}", "cluster_id": f"c{i // 4}", "author_key": f"a{i // 2}", "text_digest": f"t{i}", "stratum": "G_GENERAL", "case_window_id": "E2" if i % 2 else "E3"}
            for i in range(900)
        ]
    )
    assigned = assign_context_splits(candidates)
    first, _ = select_split_samples(assigned, "development")
    second, _ = select_split_samples(assigned, "development")
    assert first[["doc_id", "inclusion_probability"]].equals(second[["doc_id", "inclusion_probability"]])
    expected = first["p_author_cap"] * first["p_context_cap"] * first["p_quota"]
    assert (first["inclusion_probability"] - expected).abs().lt(1e-12).all()
    assert (first["inclusion_probability"] < 1).any()


def test_coder_packets_hide_sampling_metadata_and_other_coder_fields():
    selected = pd.DataFrame(
        [
            {
                "doc_id": "rd:post:1",
                "platform": "reddit",
                "split": "evaluation",
                "stratum": "G_GENERAL",
                "cluster_id": "thread-1",
                "author_key": "author-1",
                "case_window_id": "AU_LEGISLATION",
                "text_digest": "digest",
                "text_for_annotation": "masked text",
                "parent_context": "masked parent",
                "root_context": "masked root",
                "random_key": 1,
                "p_author_cap": 1.0,
                "p_context_cap": 1.0,
                "p_quota": 1.0,
                "inclusion_probability": 1.0,
            }
        ]
    )

    packet = _blind_packet_frame(selected, ["annotation-000001"])
    coder = _coder_packet_frame(packet)

    assert set(packet.columns) == {"annotation_id", "text_for_annotation", "parent_context", "root_context"}
    assert not set(packet.columns) & {"platform", "split", "stratum", "random_key", "inclusion_probability", "doc_id", "author_key"}
    assert set(coder.columns) == {"annotation_id", "text_for_annotation", "parent_context", "root_context", "label_version", "annotation_status", "relevance", "language", "target_policy", "stance", "sentiment", "frame_labels", "bypass_techniques", "notes"}
    assert not set(coder.columns) & {"coder_a_relevance", "coder_b_relevance", "adjudicated_relevance"}


def test_label_loader_sees_coder_edits_after_an_old_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "VALIDATION_ROOT", tmp_path)
    packet = pd.DataFrame({"annotation_id": ["annotation-000001"], "text_for_annotation": ["masked"], "parent_context": [""], "root_context": [""]})
    packet.to_csv(tmp_path / "labels_development.csv", index=False)
    register = {column: "" for column in validation.REGISTER_COLUMNS}
    register.update({"annotation_id": "annotation-000001", "split": "development", "doc_id": "doc-1", "text_digest": "digest", "cluster_id": "cluster-1", "author_key": "author-1"})
    pd.DataFrame([register]).to_csv(tmp_path / "coordinator_register_development.csv", index=False)
    for coder in ("a", "b"):
        values = validation._coder_packet_frame(packet)
        values.to_csv(tmp_path / f"coder_{coder}_labels_development.csv", index=False)
    pd.DataFrame([{**register, "coder_a_relevance": "relevant", **{f"coder_b_{field}": "" for field in validation.CODER_LABEL_FIELDS}, **{f"coder_a_{field}": "" for field in validation.CODER_LABEL_FIELDS}, **{f"adjudicated_{field}": "" for field in validation.CODER_LABEL_FIELDS}}]).to_csv(tmp_path / "coordinator_labels_development.csv", index=False)
    edited = pd.read_csv(tmp_path / "coder_a_labels_development.csv", dtype=str)
    edited.loc[0, "relevance"] = "relevant"
    edited.to_csv(tmp_path / "coder_a_labels_development.csv", index=False)

    loaded = load_label_packet("development")

    assert loaded.loc[0, "coder_a_relevance"] == "relevant"


def test_merge_refreshes_only_agreed_adjudications(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "VALIDATION_ROOT", tmp_path)
    packet = pd.DataFrame({"annotation_id": ["annotation-000001"], "text_for_annotation": ["masked"], "parent_context": [""], "root_context": [""]})
    packet.to_csv(tmp_path / "labels_development.csv", index=False)
    register = {column: "" for column in validation.REGISTER_COLUMNS}
    register.update({"annotation_id": "annotation-000001", "split": "development", "doc_id": "doc-1", "text_digest": "digest", "cluster_id": "cluster-1", "author_key": "author-1", "label_version": validation.CODEBOOK_VERSION})
    pd.DataFrame([register]).to_csv(tmp_path / "coordinator_register_development.csv", index=False)

    for coder in ("a", "b"):
        values = validation._coder_packet_frame(packet)
        values.loc[0, "stance"] = "support"
        values.loc[0, "sentiment"] = "positive" if coder == "a" else "negative"
        values.loc[0, "frame_labels"] = "policy_assurance|child_safety" if coder == "a" else "policy_assurance|privacy_surveillance"
        values.to_csv(tmp_path / f"coder_{coder}_labels_development.csv", index=False)

    previous = {**register, "text_for_annotation": "masked", "parent_context": "", "root_context": ""}
    previous.update({f"adjudicated_{field}": "" for field in validation.CODER_LABEL_FIELDS})
    previous.update({"adjudicated_relevance": "irrelevant", "adjudicated_stance": "oppose", "adjudicated_sentiment": "neutral", "adjudicated_frame_labels": "child_safety", "annotation_status": "adjudicated"})
    pd.DataFrame([previous]).to_csv(tmp_path / "coordinator_labels_development.csv", index=False)

    result = validation.merge_coder_labels("development")
    merged = pd.read_csv(tmp_path / "coordinator_labels_development.csv", keep_default_na=False)

    assert result["status"] == "merged"
    assert merged.loc[0, "adjudicated_relevance"] == "irrelevant"
    assert merged.loc[0, "adjudicated_stance"] == "support"
    assert merged.loc[0, "adjudicated_sentiment"] == "neutral"
    assert merged.loc[0, "adjudicated_frame_labels"] == "policy_assurance|child_safety"


def test_label_loader_rejects_unknown_codebook_values(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "VALIDATION_ROOT", tmp_path)
    packet = pd.DataFrame({"annotation_id": ["annotation-000001"], "text_for_annotation": ["masked"], "parent_context": [""], "root_context": [""]})
    packet.to_csv(tmp_path / "labels_development.csv", index=False)
    register = {column: "" for column in validation.REGISTER_COLUMNS}
    register.update({"annotation_id": "annotation-000001", "split": "development", "doc_id": "doc-1", "text_digest": "digest", "cluster_id": "cluster-1", "author_key": "author-1"})
    pd.DataFrame([register]).to_csv(tmp_path / "coordinator_register_development.csv", index=False)
    for coder in ("a", "b"):
        values = validation._coder_packet_frame(packet)
        values.loc[0, "stance"] = "not-a-codebook-value" if coder == "a" else ""
        values.to_csv(tmp_path / f"coder_{coder}_labels_development.csv", index=False)
    with pytest.raises(ValueError, match="unknown stance"):
        load_label_packet("development")


def test_missing_validation_check_is_read_only(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "VALIDATION_ROOT", tmp_path)
    monkeypatch.setattr(validation, "VALIDATION_MANIFEST", tmp_path / "validation_manifest.json")

    with pytest.raises(FileNotFoundError, match="validation sample --build"):
        validation.check_validation_samples()

    assert list(tmp_path.iterdir()) == []


def test_labeled_validation_cycle_requires_a_new_revision(tmp_path, monkeypatch):
    root = tmp_path / "validation"
    root.mkdir()
    (root / "coordinator_labels_development.csv").write_text("labels\n", encoding="utf-8")
    monkeypatch.setattr(validation, "VALIDATION_ROOT", root)

    with pytest.raises(ValueError, match="new cycle"):
        validation.build_validation_artifacts()


def test_validation_revision_isolated_from_consumed_sample(tmp_path, monkeypatch):
    old_root = tmp_path / "validation" / "old"
    old_root.mkdir(parents=True)
    pd.DataFrame([{"doc_id": "old-doc", "duplicate_key": "old-text", "cluster_id": "old-cluster"}]).to_csv(
        old_root / "coordinator_register_development.csv", index=False
    )
    base_root = tmp_path / "validation"
    older_root = base_root / "revisions" / "cycle-v1"
    older_root.mkdir(parents=True)
    pd.DataFrame([{"doc_id": "older-doc", "duplicate_key": "older-text", "cluster_id": "older-cluster"}]).to_csv(
        older_root / "coordinator_register_reserve.csv", index=False
    )
    model_root = tmp_path / "models"
    state_path = tmp_path / "measurement_cycle.json"
    monkeypatch.setattr(validation, "BASE_VALIDATION_ROOT", base_root)
    monkeypatch.setattr(validation, "BASE_MODEL_ROOT", model_root)
    monkeypatch.setattr(validation, "MEASUREMENT_CYCLE_STATE", state_path)
    monkeypatch.setattr(validation, "VALIDATION_ROOT", old_root)

    rows = []
    for index, split in enumerate(validation.SPLIT_ORDER):
        rows.append(
            {
                **{column: "" for column in validation.REGISTER_COLUMNS},
                "annotation_id": f"annotation-{index:06d}",
                "platform": "reddit",
                "split": split,
                "stratum": "G_GENERAL",
                "doc_id": f"new-doc-{index}",
                "author_key": f"author-{index}",
                "cluster_id": f"new-cluster-{index}",
                "case_window_id": "AU_LEGISLATION",
                "text_digest": f"digest-{index}",
                "text_owner_split": split,
                "random_key": index,
                "p_author_cap": 1.0,
                "p_context_cap": 1.0,
                "p_quota": 1.0,
                "inclusion_probability": 1.0,
                "duplicate_key": f"new-text-{index}",
                "sampling_seed": 1,
                "selection_key": index,
                "label_version": validation.CODEBOOK_VERSION,
                "text_for_annotation": f"text-{index}",
                "parent_context": "",
                "root_context": "",
            }
        )
    sample = pd.DataFrame(rows)
    captured = {}
    monkeypatch.setattr(
        validation,
        "_expected_sample",
        lambda **kwargs: (captured.update(kwargs) or sample.copy(), {}),
    )
    monkeypatch.setattr(validation, "_source_checksums", lambda: {"reddit": "r", "bluesky": "b", "youtube": "y"})

    result = validation.build_validation_revision("cycle-v3")

    output_root = base_root / "revisions" / "cycle-v3"
    manifest = validation.read_json(output_root / "validation_manifest.json")
    state = validation.read_json(state_path)
    assert result["status"] == "new_measurement_cycle"
    assert captured["excluded_doc_ids"] == {"old-doc", "older-doc"}
    assert captured["excluded_duplicate_keys"] == {"old-text", "older-text"}
    assert captured["excluded_clusters"] == {"old-cluster", "older-cluster"}
    assert captured["seed"] == validation._revision_seed("cycle-v3")
    assert manifest["revision"] == "cycle-v3"
    assert state["validation_root"] == validation.relative_path(output_root)
    assert state["model_root"] == validation.relative_path(model_root / "revisions" / "cycle-v3")
    assert (old_root / "coordinator_register_development.csv").read_text(encoding="utf-8").count("old-doc") == 1
