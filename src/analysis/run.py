"""Command-line entry point for the frozen-data analysis contracts.

The command tree follows the workflow documented in the repository README:
establish the scope and population contracts, validate descriptive and network
artifacts, gate measurement-dependent hypotheses, and then validate the report
bundle.  Commands that build or check artifacts require an explicit
``--build`` or ``--check`` choice.  Hypotheses additionally expose
``--execute`` once their prerequisites are ready; ``validation merge`` is the
one direct coder-file merge operation.
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from src.analysis.codebook import check_codebook, write_codebook_artifacts
from src.analysis.descriptive import build_descriptive_artifacts, check_descriptive
from src.analysis.extended import build_extended, check_extended
from src.analysis.hypotheses import build_hypothesis_artifact, check_hypothesis, execute_hypothesis
from src.analysis.measurement import (
    build_evaluation,
    build_reserve_assessment,
    build_selection,
    check_evaluation,
    check_reserve_assessment,
    check_selection,
)
from src.analysis.networks import build_network_artifacts, check_networks
from src.analysis.populations import check_populations, write_population_manifest
from src.analysis.predictions import build_predictions, check_predictions
from src.analysis.report import build_report_artifacts, check_report
from src.analysis.robustness import build_robustness_artifacts, check_robustness
from src.analysis.scope import check_scope, write_scope_artifact
from src.analysis.structure import build_structure_artifacts, check_structure
from src.analysis.temporal import build_temporal_artifacts, check_temporal
from src.analysis.topics import build_topic_artifacts, check_topics
from src.analysis.validation import build_validation_artifacts, check_validation_samples, merge_coder_labels


def _build_or_check(parser: argparse.ArgumentParser) -> None:
    """Add the standard mutually exclusive read-only/write modes."""

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="validate existing artifacts without writing")
    group.add_argument("--build", action="store_true", help="explicitly build or refresh artifacts")


def _hypothesis_mode(parser: argparse.ArgumentParser) -> None:
    """Add check, prerequisite-build, and execution modes for a hypothesis."""

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="validate an existing hypothesis artifact without writing")
    group.add_argument("--build", action="store_true", help="write the prerequisite-gate artifact")
    group.add_argument("--execute", action="store_true", help="execute a ready hypothesis and write its result")


def build_parser() -> argparse.ArgumentParser:
    """Build the command parser without reading or writing analysis artifacts."""

    parser = argparse.ArgumentParser(
        prog="python -m src.analysis.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    # Foundational contracts and descriptive artifacts.
    scope = commands.add_parser("scope", help="validate the core/extended scope contract")
    _build_or_check(scope)
    populations = commands.add_parser("populations", help="inventory frozen sources and population rules")
    _build_or_check(populations)
    codebook = commands.add_parser("codebook", help="validate the frozen v2 annotation codebook")
    _build_or_check(codebook)
    descriptive = commands.add_parser("descriptive", help="build or validate label-independent descriptive artifacts")
    _build_or_check(descriptive)

    # Network artifacts are split by the question they answer: edge tables,
    # topology/roles, and observed temporal or cascade summaries.
    networks = commands.add_parser("networks", help="build and validate the frozen graph registry")
    network_commands = networks.add_subparsers(dest="network_command", required=True)
    network_build = network_commands.add_parser("build", help="build or validate graph edge tables")
    _build_or_check(network_build)
    network_structure = network_commands.add_parser("structure", help="freeze topology-only communities and structural roles")
    _build_or_check(network_structure)
    network_temporal = network_commands.add_parser("temporal", help="validate observed cascade and temporal-community summaries")
    _build_or_check(network_temporal)

    # Confirmatory hypotheses and approval-gated extensions are kept separate
    # from descriptive and exploratory outputs.
    hypothesis = commands.add_parser("hypothesis", help="run or gate a frozen confirmatory hypothesis")
    hypothesis.add_argument("--id", required=True, choices=("H1", "H2", "H3", "H4"))
    hypothesis.add_argument("--platform", help="optional platform selector for the hypothesis endpoint")
    hypothesis.add_argument("--outcome", help="optional named outcome selector for H2")
    _hypothesis_mode(hypothesis)
    topics = commands.add_parser("topics", help="build or validate stable exploratory topic artifacts")
    _build_or_check(topics)
    extended = commands.add_parser("extended", help="validate approval-gated timing and prediction specifications")
    extended_commands = extended.add_subparsers(dest="extended_command", required=True)
    timing = extended_commands.add_parser("timing", help="validate the approval-gated timing artifact")
    _build_or_check(timing)
    prediction = extended_commands.add_parser("prediction", help="validate the approval-gated prediction artifact")
    _build_or_check(prediction)
    robustness = commands.add_parser("robustness", help="validate the predeclared robustness specification matrix")
    _build_or_check(robustness)
    predictions = commands.add_parser("predictions", help="materialise frozen-corpus measurement labels after reserve validation")
    _build_or_check(predictions)

    # Reporting and human-measurement lifecycle commands are last because they
    # consume the contracts and artifacts established above.
    report = commands.add_parser("report", help="assemble and validate the final evidence/report bundle")
    _build_or_check(report)
    validation = commands.add_parser("validation", help="human-validation sampling and model gates")
    validation_commands = validation.add_subparsers(dest="validation_command", required=True)
    sample = validation_commands.add_parser("sample", help="generate or validate blind annotation packets")
    sample.add_argument("--revision", help="write or validate an explicit versioned measurement cycle")
    _build_or_check(sample)
    select = validation_commands.add_parser("select", help="select a primary pipeline using development labels only")
    select.add_argument("--task", required=True, help="sentiment, stance, or frames")
    _build_or_check(select)
    evaluate = validation_commands.add_parser("evaluate", help="open the untouched evaluation split once")
    _build_or_check(evaluate)
    reserve = validation_commands.add_parser("reserve", help="open and assess the sealed reserve contingency exactly once")
    _build_or_check(reserve)
    merge = validation_commands.add_parser("merge", help="combine independent coder files without adjudicating")
    merge.add_argument("--split", required=True, choices=("development", "evaluation", "reserve"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one explicit analysis command and return a shell status."""

    args = build_parser().parse_args(argv)

    # Foundational contracts and label-independent artifacts.
    if args.command == "scope":
        if args.build:
            print(json.dumps({"status": "built", "artifact": str(write_scope_artifact())}, sort_keys=True))
        else:
            print(json.dumps(check_scope(), sort_keys=True))
        return 0
    if args.command == "populations":
        print(json.dumps(write_population_manifest() if args.build else check_populations(), sort_keys=True))
        return 0
    if args.command == "codebook":
        print(json.dumps(write_codebook_artifacts() if args.build else check_codebook(), sort_keys=True))
        return 0
    if args.command == "descriptive":
        print(json.dumps(build_descriptive_artifacts() if args.build else check_descriptive(), sort_keys=True))
        return 0

    # Network artifacts, from interaction tables through structure and time.
    if args.command == "networks" and args.network_command == "build":
        print(json.dumps(build_network_artifacts() if args.build else check_networks(), sort_keys=True))
        return 0
    if args.command == "networks" and args.network_command == "structure":
        print(json.dumps(build_structure_artifacts() if args.build else check_structure(), sort_keys=True))
        return 0
    if args.command == "networks" and args.network_command == "temporal":
        print(json.dumps(build_temporal_artifacts() if args.build else check_temporal(), sort_keys=True))
        return 0

    # Hypotheses, topics, approval-gated extensions, and robustness.
    if args.command == "hypothesis":
        if args.build:
            print(json.dumps(build_hypothesis_artifact(args.id, args.platform, args.outcome), sort_keys=True))
        elif args.execute:
            print(json.dumps(execute_hypothesis(args.id, args.platform, args.outcome), sort_keys=True))
        else:
            print(json.dumps(check_hypothesis(args.id, args.platform, args.outcome), sort_keys=True))
        return 0
    if args.command == "topics":
        print(json.dumps(build_topic_artifacts() if args.build else check_topics(), sort_keys=True))
        return 0
    if args.command == "extended":
        print(json.dumps(build_extended(args.extended_command) if args.build else check_extended(args.extended_command), sort_keys=True))
        return 0
    if args.command == "robustness":
        print(json.dumps(build_robustness_artifacts() if args.build else check_robustness(), sort_keys=True))
        return 0
    if args.command == "predictions":
        print(json.dumps(build_predictions() if args.build else check_predictions(), sort_keys=True))
        return 0
    if args.command == "report":
        print(json.dumps(build_report_artifacts() if args.build else check_report(), sort_keys=True))
        return 0

    # Human-measurement lifecycle.  ``merge`` deliberately has no build/check
    # switch because it combines already-produced coder files only.
    if args.command == "validation" and args.validation_command == "sample":
        print(json.dumps(build_validation_artifacts(args.revision) if args.build else check_validation_samples(args.revision), sort_keys=True))
        return 0
    if args.command == "validation" and args.validation_command == "select":
        function = build_selection if args.build else check_selection
        statuses = [function(task.strip()) for task in args.task.split(",") if task.strip()]
        print(json.dumps(statuses, sort_keys=True))
        return 0
    if args.command == "validation" and args.validation_command == "evaluate":
        print(json.dumps(build_evaluation() if args.build else check_evaluation(), sort_keys=True))
        return 0
    if args.command == "validation" and args.validation_command == "reserve":
        print(json.dumps(build_reserve_assessment() if args.build else check_reserve_assessment(), sort_keys=True))
        return 0
    if args.command == "validation" and args.validation_command == "merge":
        print(json.dumps(merge_coder_labels(args.split), sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
