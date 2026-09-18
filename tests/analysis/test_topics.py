from __future__ import annotations

import pandas as pd

from src.analysis.topics import BERTopic_SEEDS, _bertopic_status, _select_topic_count, _topic_alignment, _topic_diversity, _topic_naming_complete, _topic_novelty, _wilson_interval


def test_topic_alignment_matches_topic_order_permutation():
    left = [["privacy", "id"], ["safety", "children"]]
    right = [["children", "safety"], ["id", "privacy"]]
    assert _topic_alignment(left, right) == 1.0
    assert _topic_alignment([], []) == 1.0
    assert _topic_alignment([["topic"]], []) == 0.0


def test_topic_selection_uses_declared_metrics_instead_of_first_seed_or_count():
    candidates = pd.DataFrame(
        [
            {"platform": "test", "topic_count": 6, "reconstruction_error": 3.0, "mean_top_term_coherence": -2.0, "topic_diversity": 0.5, "mean_seed_stability": 0.4},
            {"platform": "test", "topic_count": 8, "reconstruction_error": 1.0, "mean_top_term_coherence": -1.0, "topic_diversity": 0.8, "mean_seed_stability": 0.9},
            {"platform": "test", "topic_count": 10, "reconstruction_error": 2.0, "mean_top_term_coherence": -1.5, "topic_diversity": 0.7, "mean_seed_stability": 0.6},
        ]
    )

    assert _select_topic_count(candidates) == 8
    assert candidates.loc[candidates["topic_count"].eq(8), "selection_score"].iat[0] > candidates["selection_score"].min()


def test_topic_diversity_counts_repeated_top_terms():
    assert _topic_diversity([["privacy", "safety"], ["privacy", "policy"]]) == 0.75


def test_topic_novelty_flags_stable_topics_without_theory_terms():
    terms = pd.DataFrame(
        {
            "platform": ["test"] * 4,
            "topic_id": [0, 0, 1, 1],
            "rank": [1, 2, 1, 2],
            "term": ["privacy", "identity", "bananas", "orchards"],
        }
    )
    stability = pd.DataFrame({"platform": ["test"], "mean_top_term_jaccard": [0.8]})

    result = _topic_novelty(terms, stability)

    assert result.loc[result["topic_id"].eq(0), "status"].iat[0] == "theory_aligned"
    assert result.loc[result["topic_id"].eq(1), "status"].iat[0] == "stable_novel_topic"


def test_topic_prevalence_wilson_interval_is_bounded():
    lower, upper = _wilson_interval(0, 100)
    assert 0.0 <= lower <= upper <= 1.0
    assert _wilson_interval(100, 100)[1] > 0.99


def test_bertopic_completion_requires_every_platform_seed_run():
    rows = [
        {"platform": platform, "seed": seed, "topic_id": 0, "documents": 1, "status": "executed_multi_seed_sensitivity"}
        for platform in ("reddit", "bluesky")
        for seed in BERTopic_SEEDS
    ]
    complete = pd.DataFrame(rows)
    partial = complete.copy()
    partial.loc[partial.index[-1], "status"] = "execution-failed:RuntimeError"

    assert _bertopic_status(complete, ["reddit", "bluesky"]) == "executed_multi_seed_sensitivity"
    assert _bertopic_status(partial, ["reddit", "bluesky"]) == "execution-failed"


def test_bertopic_zero_topic_runs_are_not_successful():
    rows = [
        {"platform": platform, "seed": seed, "topic_id": None, "documents": 0, "status": "executed_multi_seed_sensitivity"}
        for platform in ("reddit", "bluesky")
        for seed in BERTopic_SEEDS
    ]

    assert _bertopic_status(pd.DataFrame(rows), ["reddit", "bluesky"]) == "execution-failed"


def test_topic_naming_requires_representative_review_and_rationale():
    naming = pd.DataFrame(
        [{"platform": "reddit", "topic_id": 0, "topic_count": 6, "human_name": "", "naming_rationale": ""}]
    )
    review = pd.DataFrame(
        [
            {"platform": "reddit", "topic_id": 0, "topic_count": 6, "rank": rank, "document_ref": f"d{rank}", "text_for_review": "text"}
            for rank in (1, 2, 3)
        ]
    )

    assert _topic_naming_complete(naming, review) is False
    naming.loc[0, ["human_name", "naming_rationale"]] = ["Policy concern", "The three representatives describe the concern."]
    assert _topic_naming_complete(naming, review) is True
