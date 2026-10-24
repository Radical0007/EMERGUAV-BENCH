from __future__ import annotations

from typing import Any


def derive_intent_truth(
    *,
    clean_instruction: str,
    mission_constraints: list[str],
    protocol_context: str | None = None,
    application_theme: str | None = None,
    policy_family: str | None = None,
    target_object: str | None = None,
    target_goal: str | None = None,
) -> dict[str, Any]:
    resolved_target_goal = target_goal
    if resolved_target_goal is None:
        if protocol_context == "Sensitive-Site Respect":
            resolved_target_goal = "maintain_sensitive_site_buffer"
        elif protocol_context == "Landmark & Media Compliance":
            resolved_target_goal = "capture_compliant_landmark_view"
        elif protocol_context == "Smart Infrastructure Interaction":
            resolved_target_goal = "complete_dispatch_or_docking_sequence"
        elif application_theme == "Multilingual & High-Entropy Operations":
            resolved_target_goal = "recover_operator_intent_under_entropy"
        elif policy_family == "Government / Critical Facility Sensitivity":
            resolved_target_goal = "respect_critical_facility_buffer"
        else:
            resolved_target_goal = "preserve_safe_mission_progress"

    return {
        "cleaned_instruction": clean_instruction,
        "target_object": target_object,
        "constraints": list(mission_constraints),
        "target_goal": resolved_target_goal,
    }
