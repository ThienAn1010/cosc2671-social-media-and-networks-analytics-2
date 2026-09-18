from __future__ import annotations

import json

import pandas as pd
import pytest

import src.analysis.h1h3_agreement as agreement
import src.analysis.h1h3_human_sample as h1h3
import src.analysis.h1h3_settle as settle
import src.analysis.h4_human_sample as h4


def test_h3_requires_complete_three_comparison_sets():
    rows = []
    for match_set, comparisons in (("complete", 3), ("partial", 1)):
        rows.append({"match_set": match_set, "role": "broker", "author_key": f"{match_set}-b", "entropy": 0.8})
        for index in range(comparisons):
            rows.append({"match_set": match_set, "role": "comparison", "author_key": f"{match_set}-c{index}", "entropy": 0.2})
    contrasts = h1h3._set_contrasts(pd.DataFrame(rows))

    assert len(contrasts) == 2
    assert contrasts.loc[contrasts["match_set"].eq("complete"), "comparisons"].iat[0] == 3
    assert contrasts.loc[contrasts["match_set"].eq("partial"), "comparisons"].iat[0] == 1


def test_h4_excludes_mixed_source_channels(tmp_path, monkeypatch):
    seed_path = tmp_path / "seed_videos.csv"
    seed_path.write_text(
        "video_id,event_id,channel_id\n"
        "mixed-news,E2,mixed-channel\n"
        "mixed-commentary,E3,mixed-channel\n"
        "news-e2,E2,news-channel\n"
        "commentary-e2,E2,commentary-channel\n",
        encoding="utf-8",
    )
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "video_id,adjudicated_source_type,adjudicated_frame\n"
        "mixed-news,news,child_safety\n"
        "mixed-commentary,commentary,child_safety\n"
        "news-e2,news,child_safety\n"
        "commentary-e2,commentary,child_safety\n",
        encoding="utf-8",
    )
    comments = pd.DataFrame(
        {
            "video_id": ["yt:video:mixed-news", "yt:video:mixed-commentary", "yt:video:news-e2", "yt:video:commentary-e2"],
            "author_key": ["a", "b", "c", "d"],
            **{label: [1, 1, 1, 1] for label in h4.FRAME_LABELS},
            "comments": [1, 1, 1, 1],
            "authors": [1, 1, 1, 1],
        }
    )
    monkeypatch.setattr(h4, "SEED_VIDEOS", seed_path)

    table, notes = h4.build_table(metadata_path, 1, comments)

    assert set(table["video_id"]) == {"yt:video:news-e2", "yt:video:commentary-e2"}
    assert notes["dropped_mixed_source_channels"] == ["mixed-channel"]
    assert notes["dropped_mixed_source_videos"] == ["yt:video:mixed-news", "yt:video:mixed-commentary"]


def test_coder_tools_reject_mismatched_item_sets(tmp_path):
    coder_a = tmp_path / "a.csv"
    coder_b = tmp_path / "b.csv"
    coder_a.write_text("item_id,thread_relevant\none,yes\ntwo,no\n", encoding="utf-8")
    coder_b.write_text("item_id,thread_relevant\none,yes\n", encoding="utf-8")

    with pytest.raises(ValueError, match="same item ids"):
        agreement.compare("reddit_threads", coder_a, coder_b)
    with pytest.raises(ValueError, match="same item ids"):
        settle._pairs("reddit_threads", coder_a, coder_b)


def test_h4_result_rejects_changed_input(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("video_id,adjudicated_source_type,adjudicated_frame\nv,news,child_safety\n", encoding="utf-8")
    seed = tmp_path / "seed_videos.csv"
    seed.write_text("video_id,event_id,channel_id\nv,E2,c\n", encoding="utf-8")
    validation_root = tmp_path / "validation"
    validation_root.mkdir()
    (validation_root / "coordinator_labels_development.csv").write_text("labels\n", encoding="utf-8")
    monkeypatch.setattr(h4, "SEED_VIDEOS", seed)
    monkeypatch.setattr(h4, "VALIDATION_ROOT", validation_root)
    table = tmp_path / "h4_video_table.csv"
    table.write_text("video_id\nv\n", encoding="utf-8")
    config = h4._analysis_config(metadata, 1, 2, ("development",), None)

    payload = {
        "schema_version": h4.H4_SCHEMA_VERSION,
        "status": "estimated",
        "metadata_path": str(metadata),
        "topup_labels_path": None,
        "label_splits": ["development"],
        "input_checksums": h4._input_checksums(metadata, ("development",), None),
        "analysis_config": config,
        "analysis_config_sha256": h4._analysis_config_sha256(config),
        "analysis_code_checksums": h4._analysis_code_checksums(),
        "analysis_code_sha256": h4._analysis_code_sha256(),
        "meets_nominal_rule": False,
        "result": {"p_value": 0.2, "observed": 0.1, "lower_95": -0.1, "upper_95": 0.3},
        "output_checksums": {"video_table": h4._output_record(table, 1, ["video_id"])},
    }
    result_path = tmp_path / "h4_result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    assert h4.validate_h4_result(result_path)["status"] == "estimated"

    metadata.write_text("video_id,adjudicated_source_type,adjudicated_frame\nv,commentary,child_safety\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale or incomplete"):
        h4.validate_h4_result(result_path)
