from __future__ import annotations

import pandas as pd

from src.analysis.populations import POPULATION_IDS, POPULATION_RULES, REDDIT_EVENT_BY_CASE
from src.analysis import scope
import src.analysis.populations as populations
from src.shared.case_windows import REDDIT_CASE_BY_EVENT


def test_all_six_population_ids_have_explicit_rules():
    assert tuple(POPULATION_RULES) == POPULATION_IDS
    assert all(rule["definition"] and rule["rule"] for rule in POPULATION_RULES.values())


def test_reddit_case_registry_is_not_the_point_event_calendar():
    assert REDDIT_EVENT_BY_CASE["AU_LEGISLATION"] == "australia_legislation"
    assert "E2" not in REDDIT_EVENT_BY_CASE.values()


def test_reddit_case_registry_has_one_shared_definition():
    assert REDDIT_EVENT_BY_CASE is getattr(scope, "REDDIT_EVENT_BY_CASE", None)
    assert REDDIT_CASE_BY_EVENT["australia_legislation"] == "AU_LEGISLATION"


def test_reddit_counts_apply_relevance_and_reply_component_rules(tmp_path, monkeypatch):
    rows = []

    def add_row(record_id, event_id, thread_id, author_hash, thing="comment", parent_record_id=""):
        rows.append(
            {
                "record_id": record_id,
                "event_id": event_id,
                "thing": thing,
                "human_only": True,
                "is_english": True,
                "is_language_uncertain": False,
                "schema_valid": True,
                "text_topic": "text",
                "is_url_only": False,
                "is_no_substantive_text": False,
                "author_hash": author_hash,
                "thread_id": thread_id,
                "parent_record_id": parent_record_id,
                "exclusion_status": "eligible",
            }
        )

    add_row("p1", "australia_legislation", "t1", "root-1", thing="post")
    for index in range(10):
        add_row(f"c1-{index}", "australia_legislation", "t1", f"author-{index % 2}", parent_record_id="p1")
    add_row("p2", "australia_implementation", "t2", "root-2", thing="post")
    for index in range(5):
        add_row(f"c2-{index}", "australia_implementation", "t2", f"rejected-{index}", parent_record_id="p2")
    for index in range(5):
        add_row(f"c3-{index}", "australia_legislation", "t3", "disconnected-{index}")

    processed = tmp_path / "processed"
    corpus_path = processed / "reddit_age_gate" / "analysis_corpus.parquet"
    corpus_path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(corpus_path)
    audit_path = tmp_path / "reddit_thread_relevance.csv"
    pd.DataFrame(
        {
            "thread_id": ["t1", "t2", "t3"],
            "coder_a_relevant": ["yes", "no", "yes"],
            "coder_b_relevant": ["yes", "no", "yes"],
            "adjudicated_relevant": ["yes", "no", "yes"],
        }
    ).to_csv(audit_path, index=False)
    monkeypatch.setattr(populations, "PROCESSED_ROOT", processed)
    monkeypatch.setattr(populations, "REDDIT_RELEVANCE_AUDIT", audit_path)

    counts = populations._reddit_candidate_counts()

    assert counts == {"R_AU_LEGISLATION": 16, "R_AU_IMPLEMENTATION": 0, "R_REPLY_CORE": 10, "R_BROKER_CORE": 2}
