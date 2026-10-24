from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from emerguav_bench.config.vocab import (
    ABSTENTION_DECISIONS,
    ACTION_TOKENS,
    APPLICATION_THEMES,
    COLLABORATION_REQUEST_TYPES,
    DEGRADATION_SEVERITIES,
    LANGUAGE_ENTROPY_LABELS,
    LANGUAGE_ENTROPY_LEVELS,
    POLICY_FAMILIES,
    RISK_LEVELS,
    SOURCE_DOMAIN_BUCKETS,
    TASK_ID_TO_NAME,
    TRUSTED_MODALITIES,
    OBSERVATION_SEVERITY_CODES,
)


@dataclass(slots=True)
class ValidationErrorDetail:
    field: str
    message: str


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_string_list(value: Any, *, non_empty: bool = False) -> bool:
    if not isinstance(value, list):
        return False
    if non_empty and not value:
        return False
    return all(_is_non_empty_string(item) for item in value)


def _validate_pose_6dof(value: Any) -> list[ValidationErrorDetail]:
    errors: list[ValidationErrorDetail] = []
    if not isinstance(value, dict):
        return [ValidationErrorDetail("pose_6dof", "must be an object")]

    for key in ("position", "orientation"):
        if key not in value:
            errors.append(ValidationErrorDetail("pose_6dof", f"missing key '{key}'"))
        elif not isinstance(value[key], dict):
            errors.append(ValidationErrorDetail("pose_6dof", f"'{key}' must be an object"))

    position = value.get("position", {})
    if isinstance(position, dict):
        for axis in ("x", "y", "z"):
            if axis not in position or not isinstance(position[axis], (int, float)):
                errors.append(ValidationErrorDetail("pose_6dof", f"position.{axis} must be numeric"))

    orientation = value.get("orientation", {})
    if isinstance(orientation, dict):
        numeric_keys = [key for key, item in orientation.items() if isinstance(item, (int, float))]
        if len(numeric_keys) < 3:
            errors.append(
                ValidationErrorDetail(
                    "pose_6dof",
                    "orientation must include at least 3 numeric components",
                )
            )

    return errors


def _validate_observation_protocol(value: Any) -> list[ValidationErrorDetail]:
    errors: list[ValidationErrorDetail] = []
    if not isinstance(value, dict):
        return [ValidationErrorDetail("observation_protocol_gt", "must be an object")]

    if not _is_non_empty_string(value.get("protocol_context_profile")):
        errors.append(ValidationErrorDetail("observation_protocol_gt.protocol_context_profile", "must be a non-empty string"))
    if not _is_non_empty_string(value.get("primary_degradation")):
        errors.append(ValidationErrorDetail("observation_protocol_gt.primary_degradation", "must be a non-empty string"))
    usability = value.get("usability")
    if not _is_non_empty_string(usability):
        errors.append(ValidationErrorDetail("observation_protocol_gt.usability", "must be a non-empty string"))
    trusted_modality = value.get("trusted_modality")
    if trusted_modality not in TRUSTED_MODALITIES:
        errors.append(
            ValidationErrorDetail(
                "observation_protocol_gt.trusted_modality",
                f"must be one of {sorted(TRUSTED_MODALITIES)}",
            )
        )
    risk_level = value.get("risk_level")
    if risk_level not in RISK_LEVELS:
        errors.append(
            ValidationErrorDetail(
                "observation_protocol_gt.risk_level",
                f"must be one of {sorted(RISK_LEVELS)}",
            )
        )
    if not _is_non_empty_string(value.get("risk_zone_label")):
        errors.append(ValidationErrorDetail("observation_protocol_gt.risk_zone_label", "must be a non-empty string"))
    if not _is_non_empty_string(value.get("viewpoint_summary")):
        errors.append(ValidationErrorDetail("observation_protocol_gt.viewpoint_summary", "must be a non-empty string"))
    if not _is_non_empty_string(value.get("cause_label")):
        errors.append(ValidationErrorDetail("observation_protocol_gt.cause_label", "must be a non-empty string"))

    return errors


def _validate_intent_extraction(value: Any) -> list[ValidationErrorDetail]:
    errors: list[ValidationErrorDetail] = []
    if not isinstance(value, dict):
        return [ValidationErrorDetail("intent_extraction_gt", "must be an object")]

    if not _is_non_empty_string(value.get("cleaned_instruction")):
        errors.append(ValidationErrorDetail("intent_extraction_gt.cleaned_instruction", "must be a non-empty string"))

    target_object = value.get("target_object")
    if target_object is not None and not _is_non_empty_string(target_object):
        errors.append(ValidationErrorDetail("intent_extraction_gt.target_object", "must be null or a non-empty string"))

    constraints = value.get("constraints")
    if not _is_string_list(constraints):
        errors.append(ValidationErrorDetail("intent_extraction_gt.constraints", "must be a list of strings"))
    if not _is_non_empty_string(value.get("target_goal")):
        errors.append(ValidationErrorDetail("intent_extraction_gt.target_goal", "must be a non-empty string"))

    return errors


def _validate_collaboration(value: Any, *, field_name: str = "collaboration_gt") -> list[ValidationErrorDetail]:
    errors: list[ValidationErrorDetail] = []
    if not isinstance(value, dict):
        return [ValidationErrorDetail(field_name, "must be an object")]
    if not isinstance(value.get("required"), bool):
        errors.append(ValidationErrorDetail(f"{field_name}.required", "must be a boolean"))
    request_type = value.get("request_type")
    if request_type not in COLLABORATION_REQUEST_TYPES:
        errors.append(
            ValidationErrorDetail(
                f"{field_name}.request_type",
                f"must be one of {sorted(COLLABORATION_REQUEST_TYPES)}",
            )
        )
    if value.get("trusted_modality") not in TRUSTED_MODALITIES:
        errors.append(
            ValidationErrorDetail(
                f"{field_name}.trusted_modality",
                f"must be one of {sorted(TRUSTED_MODALITIES)}",
            )
        )
    if value.get("required") is True and request_type == "none":
        errors.append(ValidationErrorDetail(f"{field_name}.request_type", "must not be 'none' when required is true"))
    if value.get("required") is False and request_type not in {None, "none"}:
        errors.append(ValidationErrorDetail(f"{field_name}.request_type", "must be 'none' when required is false"))
    return errors


def _validate_paper_task_truth(value: Any) -> list[ValidationErrorDetail]:
    errors: list[ValidationErrorDetail] = []
    if not isinstance(value, dict):
        return [ValidationErrorDetail("paper_task_gt", "must be an object")]

    for task_id in TASK_ID_TO_NAME:
        if task_id not in value:
            errors.append(ValidationErrorDetail("paper_task_gt", f"missing key '{task_id}'"))

    if errors:
        return errors

    for task_id in ("T1", "T2", "T3", "T4", "T7", "T9"):
        if not _is_non_empty_string(value.get(task_id)):
            errors.append(ValidationErrorDetail(f"paper_task_gt.{task_id}", "must be a non-empty string"))

    if value.get("T5") not in TRUSTED_MODALITIES:
        errors.append(ValidationErrorDetail(f"paper_task_gt.T5", f"must be one of {sorted(TRUSTED_MODALITIES)}"))

    t6 = value.get("T6")
    if not isinstance(t6, dict):
        errors.append(ValidationErrorDetail("paper_task_gt.T6", "must be an object"))
    else:
        if not _is_non_empty_string(t6.get("usability")):
            errors.append(ValidationErrorDetail("paper_task_gt.T6.usability", "must be a non-empty string"))
        if not _is_non_empty_string(t6.get("cause_label")):
            errors.append(ValidationErrorDetail("paper_task_gt.T6.cause_label", "must be a non-empty string"))

    if not _is_string_list(value.get("T8")):
        errors.append(ValidationErrorDetail("paper_task_gt.T8", "must be a list of strings"))
    if value.get("T10") not in ACTION_TOKENS:
        errors.append(ValidationErrorDetail("paper_task_gt.T10", f"must be one of {sorted(ACTION_TOKENS)}"))
    if value.get("T11") not in ABSTENTION_DECISIONS:
        errors.append(
            ValidationErrorDetail(
                "paper_task_gt.T11",
                f"must be one of {sorted(ABSTENTION_DECISIONS)}",
            )
        )
    errors.extend(_validate_collaboration(value.get("T12"), field_name="paper_task_gt.T12"))
    return errors


def _validate_geometry_anchor(value: Any) -> list[ValidationErrorDetail]:
    if not isinstance(value, dict):
        return [ValidationErrorDetail("geometry_anchor", "must be an object")]
    return []


def validate_sample_record(sample: dict[str, Any]) -> list[ValidationErrorDetail]:
    if not isinstance(sample, dict):
        return [ValidationErrorDetail("sample", "must be an object")]

    errors: list[ValidationErrorDetail] = []
    required_string_fields = [
        "sample_id",
        "protocol_context",
        "application_theme",
        "policy_family",
        "sequence_id",
        "frame_id",
        "rgb_frame_path",
        "lidar_frame_path",
        "map_id",
        "language_entropy_level",
        "observation_severity_code",
        "clean_instruction",
        "noisy_instruction",
        "action_token_gt",
        "risk_level_gt",
    ]

    for field in required_string_fields:
        if not _is_non_empty_string(sample.get(field)):
            errors.append(ValidationErrorDetail(field, "must be a non-empty string"))

    if sample.get("source_domain_bucket") not in SOURCE_DOMAIN_BUCKETS:
        errors.append(ValidationErrorDetail("source_domain_bucket", f"must be one of {sorted(SOURCE_DOMAIN_BUCKETS)}"))
    if sample.get("application_theme") not in APPLICATION_THEMES:
        errors.append(ValidationErrorDetail("application_theme", f"must be one of {sorted(APPLICATION_THEMES)}"))
    if sample.get("policy_family") not in POLICY_FAMILIES:
        errors.append(ValidationErrorDetail("policy_family", f"must be one of {sorted(POLICY_FAMILIES)}"))

    if sample.get("degradation_severity") not in DEGRADATION_SEVERITIES:
        errors.append(
            ValidationErrorDetail(
                "degradation_severity",
                f"must be one of {sorted(DEGRADATION_SEVERITIES)}",
            )
        )
    if sample.get("observation_severity_code") not in OBSERVATION_SEVERITY_CODES:
        errors.append(
            ValidationErrorDetail(
                "observation_severity_code",
                f"must be one of {sorted(OBSERVATION_SEVERITY_CODES)}",
            )
        )
    if sample.get("language_entropy_level") not in LANGUAGE_ENTROPY_LEVELS:
        errors.append(
            ValidationErrorDetail(
                "language_entropy_level",
                f"must be one of {sorted(LANGUAGE_ENTROPY_LEVELS)}",
            )
        )

    if not _is_string_list(sample.get("degradation_labels"), non_empty=True):
        errors.append(ValidationErrorDetail("degradation_labels", "must be a non-empty list of strings"))
    language_entropy_labels = sample.get("language_entropy_labels")
    if not _is_string_list(language_entropy_labels, non_empty=True):
        errors.append(ValidationErrorDetail("language_entropy_labels", "must be a non-empty list of strings"))
    elif any(label not in LANGUAGE_ENTROPY_LABELS for label in language_entropy_labels):
        errors.append(
            ValidationErrorDetail(
                "language_entropy_labels",
                f"must contain only values in {sorted(LANGUAGE_ENTROPY_LABELS)}",
            )
        )

    if not _is_string_list(sample.get("mission_constraints")):
        errors.append(ValidationErrorDetail("mission_constraints", "must be a list of strings"))

    safe_alternatives = sample.get("acceptable_safe_alternatives")
    if not _is_string_list(safe_alternatives, non_empty=True):
        errors.append(
            ValidationErrorDetail(
                "acceptable_safe_alternatives",
                "must be a non-empty list of strings",
            )
        )

    forbidden_actions = sample.get("forbidden_actions")
    if not _is_string_list(forbidden_actions):
        errors.append(ValidationErrorDetail("forbidden_actions", "must be a list of strings"))
    elif sample.get("action_token_gt") in forbidden_actions:
        errors.append(ValidationErrorDetail("forbidden_actions", "must not contain action_token_gt"))

    action_token = sample.get("action_token_gt")
    if action_token not in ACTION_TOKENS:
        errors.append(ValidationErrorDetail("action_token_gt", f"must be one of {sorted(ACTION_TOKENS)}"))

    risk_level = sample.get("risk_level_gt")
    if risk_level not in RISK_LEVELS:
        errors.append(ValidationErrorDetail("risk_level_gt", f"must be one of {sorted(RISK_LEVELS)}"))

    if not isinstance(sample.get("protocol_metadata"), dict):
        errors.append(ValidationErrorDetail("protocol_metadata", "must be an object"))

    errors.extend(_validate_pose_6dof(sample.get("pose_6dof")))
    errors.extend(_validate_observation_protocol(sample.get("observation_protocol_gt")))
    errors.extend(_validate_intent_extraction(sample.get("intent_extraction_gt")))
    errors.extend(_validate_collaboration(sample.get("collaboration_gt")))
    errors.extend(_validate_paper_task_truth(sample.get("paper_task_gt")))
    errors.extend(_validate_geometry_anchor(sample.get("geometry_anchor")))

    return errors
