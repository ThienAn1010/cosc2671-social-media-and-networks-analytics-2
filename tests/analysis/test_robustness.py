from __future__ import annotations

import pandas as pd

import src.analysis.robustness as robustness
from src.analysis.robustness import SPECIFICATIONS, execute_named_variant, summarize_variant_results


def test_robustness_matrix_is_named_not_cartesian():
    frame = pd.DataFrame(SPECIFICATIONS)
    assert not frame["variant_id"].is_unique
    assert frame.groupby("hypothesis_id").size().to_dict() == {"H1": 4, "H2": 3, "H3": 3, "H4": 4}
    assert frame["decisive"].sum() == 7


def test_robustness_summary_reports_direction_and_nominal_gate():
    result = summarize_variant_results(
        pd.DataFrame(
            [
                {"hypothesis_id": "H2", "variant_id": "primary", "effect": 0.10, "p_value": 0.01},
                {"hypothesis_id": "H2", "variant_id": "sensitivity", "effect": -0.01, "p_value": 0.90},
            ]
        )
    )

    assert result["direction"].tolist() == ["positive", "negative"]
    assert result["passes_nominal_gate"].tolist() == [True, False]
    assert {"lower_95", "upper_95", "decision", "interval_excludes_zero", "decision_stable"} <= set(result.columns)


def test_h2_primary_endpoints_are_stable_independently(tmp_path, monkeypatch):
    robustness_root = tmp_path / "robustness"
    hypothesis_root = tmp_path / "hypotheses"
    hypothesis_root.mkdir()
    monkeypatch.setattr(robustness, "ROBUSTNESS_ROOT", robustness_root)
    monkeypatch.setattr(robustness, "ROBUSTNESS_MANIFEST", robustness_root / "robustness_manifest.json")
    monkeypatch.setattr(robustness, "HYPOTHESIS_ROOT", hypothesis_root)
    for hypothesis_id in robustness.HYPOTHESIS_SPECS:
        (hypothesis_root / f"{hypothesis_id.lower()}.json").write_text('{"status": "ready_for_execution"}\n', encoding="utf-8")

    frames = {
        f"{spec['hypothesis_id']}/{spec['variant_id']}": pd.DataFrame({"value": [1]})
        for spec in robustness.SPECIFICATIONS
    }

    def fake_execute(hypothesis_id, variant_id, frame):
        mixed_endpoint = hypothesis_id == "H2" and variant_id == "circumvention_primary"
        effect = 0.0 if mixed_endpoint else 0.10
        return {
            "observed": effect,
            "lower_95": effect - 0.01,
            "upper_95": effect + 0.01,
            "p_value": 0.90 if mixed_endpoint else 0.01,
        }

    monkeypatch.setattr(robustness, "execute_named_variant", fake_execute)

    robustness.build_robustness_artifacts(frames)
    results = pd.read_csv(robustness_root / "variant_results.csv")
    h2 = results[
        results["hypothesis_id"].eq("H2")
        & results["variant_id"].isin(("privacy_primary", "circumvention_primary"))
    ]

    assert h2["decision"].astype(str).str.casefold().tolist() == ["true", "false"]
    assert h2["decision_stable"].astype(str).str.casefold().tolist() == ["true", "true"]


def test_named_variant_dispatches_to_the_h2_estimator():
    frame = pd.DataFrame(
        [
            {"case": case, "author": f"a{index}", "thread": f"{case}-{index}", "outcome": value}
            for case, value in (("AU_IMPLEMENTATION", 1.0), ("AU_LEGISLATION", 0.0))
            for index in range(4)
        ]
    )

    result = execute_named_variant("H2", "privacy_primary", frame, replicates=20)

    assert result["hypothesis_id"] == "H2"
    assert result["variant_id"] == "privacy_primary"
    assert {"observed", "lower_95", "upper_95", "p_value"} <= set(result)


def test_named_h2_variants_consume_different_declared_inputs():
    rows = []
    for case, offset in (("AU_IMPLEMENTATION", 1.0), ("AU_LEGISLATION", 0.0)):
        for index in range(4):
            rows.append(
                {
                    "case": case,
                    "author": f"{case}-{index}",
                    "thread": f"{case}-thread-{index}",
                    "outcome": offset if index % 2 else 0.0,
                    "privacy_surveillance": offset if index % 2 else 0.0,
                    "legacy_revalidated": 1.0 if index < 2 else 0.0,
                    "language_population": "strict_english" if index < 3 else "inclusive_uncertain_bound",
                    "thread_population": "keyword_screen" if index == 0 else "audited_relevant",
                }
            )
    frame = pd.DataFrame(rows)

    primary = execute_named_variant("H2", "privacy_primary", frame, replicates=20)
    sensitivity = execute_named_variant("H2", "strict_keyword_sensitivity", frame, replicates=20)

    assert primary["weighting"] == "author_balanced"
    assert sensitivity["weighting"] == "document_weighted"
    assert primary["input_contract"]["rows_after"] != sensitivity["input_contract"]["rows_after"]
    assert primary["variant_spec"]["thread_population"] != sensitivity["variant_spec"]["thread_population"]


def test_h4_leave_out_variant_runs_every_declared_unit():
    rows = []
    for event in ("E2", "E3"):
        for source, channel, metadata, audience in (
            ("news", f"{event}-n1", [1, 0], [1, 0]),
            ("news", f"{event}-n2", [1, 0], [0, 1]),
            ("commentary", f"{event}-c1", [0, 1], [0, 1]),
            ("commentary", f"{event}-c2", [0, 1], [1, 0]),
        ):
            rows.append(
                {
                    "video_id": f"{channel}-video",
                    "event": event,
                    "channel_id": channel,
                    "source_type": source,
                    "m1": metadata[0],
                    "m2": metadata[1],
                    "a1": audience[0],
                    "a2": audience[1],
                }
            )

    result = execute_named_variant(
        "H4",
        "leave_one_channel_out",
        pd.DataFrame(rows),
        metadata_profile_columns=["m1", "m2"],
        audience_profile_columns=["a1", "a2"],
        permutations=20,
    )

    assert result["leave_out_unit"] == "channel_id"
    assert result["excluded_units"] == sorted(pd.DataFrame(rows)["channel_id"].unique())
    assert len(result["leave_out_results"]) == len(result["excluded_units"])
