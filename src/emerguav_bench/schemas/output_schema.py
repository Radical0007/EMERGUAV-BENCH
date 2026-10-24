from __future__ import annotations

from typing import Any

from emerguav_bench.config.vocab import (
    ABSTENTION_DECISIONS,
    ACTION_TOKENS,
    COLLABORATION_REQUEST_TYPES,
    RISK_LEVELS,
    TRUSTED_MODALITIES,
)
from emerguav_bench.schemas.sample_schema import ValidationErrorDetail


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str)


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _validate_collaboration_request(value: Any) -> list[ValidationErrorDetail]:
    errors: list[ValidationErrorDetail] = []
    if not isinstance(value, dict):
        return [ValidationErrorDetail("Action_Output.collaboration_request", "must be an object")]
    if not isinstance(value.get("required"), bool):
        errors.append(ValidationErrorDetail("Action_Output.collaboration_request.required", "must be a boolean"))
    if value.get("request_type") not in COLLABORATION_REQUEST_TYPES:
        errors.append(
            ValidationErrorDetail(
                "Action_Output.collaboration_request.request_type",
                f"must be one of {sorted(COLLABORATION_REQUEST_TYPES)}",
            )
        )
    if value.get("trusted_modality") not in TRUSTED_MODALITIES:
        errors.append(
            ValidationErrorDetail(
                "Action_Output.collaboration_request.trusted_modality",
                f"must be one of {sorted(TRUSTED_MODALITIES)}",
            )
        )
    return errors


def validate_model_output(output: dict[str, Any]) -> list[ValidationErrorDetail]:
    if not isinstance(output, dict):
        return [ValidationErrorDetail("output", "must be an object")]

    errors: list[ValidationErrorDetail] = []

    for field in ("Observation_Protocol_Assessment", "Intent_Extraction", "Action_Output", "Rationale_Short"):
        if field not in output:
            errors.append(ValidationErrorDetail(field, "is required"))

    assessment = output.get("Observation_Protocol_Assessment")
    if not isinstance(assessment, dict):
        errors.append(ValidationErrorDetail("Observation_Protocol_Assessment", "must be an object"))
    else:
        if not _is_non_empty_string(assessment.get("protocol_context_profile")):
            errors.append(ValidationErrorDetail("Observation_Protocol_Assessment.protocol_context_profile", "must be a string"))
        if not _is_non_empty_string(assessment.get("primary_degradation")):
            errors.append(ValidationErrorDetail("Observation_Protocol_Assessment.primary_degradation", "must be a string"))
        if not _is_non_empty_string(assessment.get("usability")):
            errors.append(ValidationErrorDetail("Observation_Protocol_Assessment.usability", "must be a string"))
        if assessment.get("trusted_modality") not in TRUSTED_MODALITIES:
            errors.append(
                ValidationErrorDetail(
                    "Observation_Protocol_Assessment.trusted_modality",
                    f"must be one of {sorted(TRUSTED_MODALITIES)}",
                )
            )
        if assessment.get("risk_level") not in RISK_LEVELS:
            errors.append(
                ValidationErrorDetail(
                    "Observation_Protocol_Assessment.risk_level",
                    f"must be one of {sorted(RISK_LEVELS)}",
                )
            )
        if not _is_non_empty_string(assessment.get("risk_zone_label")):
            errors.append(ValidationErrorDetail("Observation_Protocol_Assessment.risk_zone_label", "must be a string"))
        if not _is_non_empty_string(assessment.get("viewpoint_summary")):
            errors.append(ValidationErrorDetail("Observation_Protocol_Assessment.viewpoint_summary", "must be a string"))
        if not _is_non_empty_string(assessment.get("cause_label")):
            errors.append(ValidationErrorDetail("Observation_Protocol_Assessment.cause_label", "must be a string"))

    intent = output.get("Intent_Extraction")
    if not isinstance(intent, dict):
        errors.append(ValidationErrorDetail("Intent_Extraction", "must be an object"))
    else:
        if not _is_non_empty_string(intent.get("cleaned_instruction")):
            errors.append(ValidationErrorDetail("Intent_Extraction.cleaned_instruction", "must be a string"))
        target_object = intent.get("target_object")
        if target_object is not None and not _is_non_empty_string(target_object):
            errors.append(ValidationErrorDetail("Intent_Extraction.target_object", "must be null or a string"))
        if not _is_string_list(intent.get("constraints")):
            errors.append(ValidationErrorDetail("Intent_Extraction.constraints", "must be a list of strings"))
        if not _is_non_empty_string(intent.get("target_goal")):
            errors.append(ValidationErrorDetail("Intent_Extraction.target_goal", "must be a string"))

    action_output = output.get("Action_Output")
    if not isinstance(action_output, dict):
        errors.append(ValidationErrorDetail("Action_Output", "must be an object"))
    else:
        if action_output.get("action_token") not in ACTION_TOKENS:
            errors.append(
                ValidationErrorDetail(
                    "Action_Output.action_token",
                    f"must be one of {sorted(ACTION_TOKENS)}",
                )
            )
        confidence = action_output.get("confidence")
        if not isinstance(confidence, (int, float)):
            errors.append(ValidationErrorDetail("Action_Output.confidence", "must be numeric"))
        elif not 0.0 <= float(confidence) <= 1.0:
            errors.append(ValidationErrorDetail("Action_Output.confidence", "must be between 0 and 1"))
        if action_output.get("abstention_decision") not in ABSTENTION_DECISIONS:
            errors.append(
                ValidationErrorDetail(
                    "Action_Output.abstention_decision",
                    f"must be one of {sorted(ABSTENTION_DECISIONS)}",
                )
            )
        errors.extend(_validate_collaboration_request(action_output.get("collaboration_request")))

    if not isinstance(output.get("Rationale_Short"), str):
        errors.append(ValidationErrorDetail("Rationale_Short", "must be a string"))

    return errors
