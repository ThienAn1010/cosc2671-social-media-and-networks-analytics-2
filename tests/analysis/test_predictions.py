from __future__ import annotations

import pandas as pd

import src.analysis.predictions as predictions
from src.analysis.predictions import _base_columns, _reserve_cell_status
from src.analysis.predictions import _bluesky_reply_document_ids


def test_prediction_rows_drop_annotation_text_and_require_both_reserve_gates():
    frame = pd.DataFrame(
        {
            "platform": ["reddit"],
            "doc_id": ["doc-1"],
            "population_id": ["R_REPLY_CORE"],
            "case_window_id": ["E2"],
            "event_id": ["E2"],
            "author_key": ["author-1"],
            "cluster_id": ["thread-1"],
            "container_id": ["subreddit"],
            "day": ["2025-01-01"],
            "text_digest": ["digest"],
            "text_for_annotation": ["masked text"],
        }
    )
    selections = {"status": "selected"}
    passed = {"gates": {"passed": True}, "unseen_author": {"gates": {"passed": True}}}
    failed_unseen = {"gates": {"passed": True}, "unseen_author": {"gates": {"passed": False}}}

    assert "text_for_annotation" not in _base_columns(frame).drop(columns=["text_for_annotation"]).columns
    assert _reserve_cell_status("frames", "reddit", {"frames": selections}, {"metrics": {"frames": {"by_platform": {"reddit": passed}}}})[0] == "validated_automated"
    assert _reserve_cell_status("frames", "reddit", {"frames": selections}, {"metrics": {"frames": {"by_platform": {"reddit": failed_unseen}}}})[0] == "fallback_human_sample"


def test_bluesky_reply_population_keeps_eligible_descendants(monkeypatch):
    posts = pd.DataFrame(
        {
            "doc_id": ["root", "reply"],
            "event_window": ["E2", "E2"],
            "in_search": [True, False],
            "is_english": [True, True],
            "is_meme": [False, False],
            "is_repeat_burst": [False, False],
            "is_reply": [False, True],
            "account_class": ["ordinary", "ordinary"],
            "author_hash": ["author-root", "author-reply"],
        }
    )

    def fake_read_parquet(path, columns=None):
        if str(path).endswith("post_query.parquet"):
            return pd.DataFrame({"doc_id": ["root"], "phrase_exact": [True]})
        return pd.DataFrame({"source_doc_id": ["reply"], "target_doc_id": ["root"]})

    monkeypatch.setattr(predictions.pd, "read_parquet", fake_read_parquet)

    assert _bluesky_reply_document_ids(posts) == {"reply"}
