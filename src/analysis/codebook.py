"""Frozen v2 annotation codebook and legacy-label crosswalk."""

from __future__ import annotations

from typing import Any

from src.analysis.artifacts import ANALYSIS_ROOT, read_json, utc_now_iso, write_json

CODEBOOK_VERSION = "age-gate-codebook-v2"
CODEBOOK_ARTIFACT = ANALYSIS_ROOT / "codebook" / "codebook_v2.json"
ANNOTATION_SCHEMA_ARTIFACT = ANALYSIS_ROOT / "codebook" / "annotation_schema_v1.json"

FRAME_DEFINITIONS = [
    {
        "label": "policy_assurance",
        "definition": "The existence, design or operation of age-policy and assurance requirements.",
        "subcodes": ["policy_event_named", "assurance_mechanism", "implementation_stage"],
    },
    {
        "label": "child_safety",
        "definition": "Protection of children, exposure prevention, parental safeguarding or harm reduction.",
        "subcodes": ["harm_type", "protection_rationale", "parental_responsibility"],
    },
    {
        "label": "privacy_surveillance",
        "definition": "Identification, tracking, biometric or ID exposure, retention, breach or surveillance concern.",
        "subcodes": ["data_minimisation", "anonymity", "biometric_or_id_concern"],
    },
    {
        "label": "governance_platform_responsibility",
        "definition": "Accountability, enforcement design, transparency, platform duty or institutional competence.",
        "subcodes": ["government_responsibility", "platform_responsibility", "assurance_provider", "appeal_or_audit"],
    },
    {
        "label": "circumvention_censorship_autonomy",
        "definition": "Argumentative claims about bypass, speech, access, paternalism, exclusion, overreach or autonomy.",
        "subcodes": ["censorship_autonomy", "technical_efficacy_futility", "argumentative_circumvention"],
    },
]

LEGACY_CROSSWALK = [
    {"legacy": "privacy", "destination": "privacy_surveillance", "action": "candidate_seed_revalidate"},
    {"legacy": "free_speech", "destination": "circumvention_censorship_autonomy.censorship_autonomy", "action": "candidate_seed_revalidate"},
    {"legacy": "circumvent", "destination": "bypass_technique_or_argumentative_circumvention", "action": "never_equate_automatically"},
    {"legacy": "id_upload", "destination": "privacy_surveillance.biometric_or_id_concern_or_policy_feature", "action": "remove_as_frame_revalidate"},
    {"legacy": "child_safety", "destination": "child_safety", "action": "revalidate"},
    {"legacy": "no_legacy_equivalent", "destination": "policy_assurance|governance_platform_responsibility|circumvention_censorship_autonomy.technical_efficacy_futility", "action": "fresh_annotation_required"},
]

ANNOTATION_SCHEMA = {
    "schema_version": "annotation.v1",
    "label_version": CODEBOOK_VERSION,
    "fields": {
        "relevance": ["relevant", "adjacent_contextual", "irrelevant"],
        "language": ["english", "mixed", "non_english", "too_short_ambiguous"],
        "target_policy": ["UK_OSA", "AU_SOCIAL_MINIMUM_AGE", "OTHER_EXTENDED_EVENT", "multiple", "unclear"],
        "stance": ["support", "oppose", "mixed_conditional", "neutral_descriptive", "unclear_ambiguous"],
        "sentiment": ["positive", "negative", "neutral", "mixed_ambiguous"],
        "frame_labels": [frame["label"] for frame in FRAME_DEFINITIONS],
        "bypass_techniques": ["VPN", "proxy_Tor", "DNS_change", "false_borrowed_ID", "face_spoofing", "parent_account", "age_misstatement", "platform_migration", "none_unclear"],
    },
    "coding_rules": [
        "Code the author's contribution, not quoted speech.",
        "Label relevance and target before stance or frame labels.",
        "A VPN mention is not a circumvention frame without an argument about avoidance, feasibility or autonomy.",
        "Negative sentiment is not opposition; support for child safety alongside opposition to ID collection is mixed_conditional.",
        "Do not infer age, residence, identity or intent from a profile or community.",
    ],
    "independent_coding": {
        "required_coders": 2,
        "adjudicator": "third_person_for_remaining_disagreements",
        "model_output_in_packet": False,
    },
}


def build_codebook() -> dict[str, Any]:
    return {
        "schema_version": "codebook.v2",
        "codebook_version": CODEBOOK_VERSION,
        "created_at_utc": utc_now_iso(),
        "frames": FRAME_DEFINITIONS,
        "legacy_crosswalk": LEGACY_CROSSWALK,
        "stance_target": ANNOTATION_SCHEMA["fields"]["target_policy"],
        "annotation_schema_path": "data/analysis/codebook/annotation_schema_v1.json",
        "approval_gated_labels": ["E1_context", "E4", "E5"],
        "prohibitions": [
            "legacy scores are not v2 gold labels",
            "human labels are required before automated full-corpus claims",
            "identities are never joined across platforms",
        ],
    }


def write_codebook_artifacts() -> dict[str, Any]:
    CODEBOOK_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    write_json(CODEBOOK_ARTIFACT, build_codebook())
    write_json(ANNOTATION_SCHEMA_ARTIFACT, ANNOTATION_SCHEMA)
    return {"status": "built", "codebook_version": CODEBOOK_VERSION, "frames": len(FRAME_DEFINITIONS)}


def validate_codebook(payload: dict[str, Any]) -> None:
    labels = [frame["label"] for frame in payload.get("frames", [])]
    if labels != [frame["label"] for frame in FRAME_DEFINITIONS]:
        raise ValueError("codebook frame order or labels do not match frozen v2")
    if {row["legacy"] for row in payload.get("legacy_crosswalk", [])} != {row["legacy"] for row in LEGACY_CROSSWALK}:
        raise ValueError("legacy crosswalk is incomplete")
    if payload.get("codebook_version") != CODEBOOK_VERSION:
        raise ValueError("unexpected codebook version")


def check_codebook() -> dict[str, Any]:
    if not CODEBOOK_ARTIFACT.exists() or not ANNOTATION_SCHEMA_ARTIFACT.exists():
        raise FileNotFoundError("codebook artifacts are incomplete; run the explicit codebook build first")
    payload = read_json(CODEBOOK_ARTIFACT)
    validate_codebook(payload)
    schema = read_json(ANNOTATION_SCHEMA_ARTIFACT)
    if schema != ANNOTATION_SCHEMA:
        raise ValueError("annotation schema is stale or inconsistent")
    return {"status": "valid", "codebook_version": CODEBOOK_VERSION, "frames": len(FRAME_DEFINITIONS)}
