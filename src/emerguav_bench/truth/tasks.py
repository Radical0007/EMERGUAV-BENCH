from __future__ import annotations

from typing import Any


def derive_task_truth(
    *,
    observation_protocol_truth: dict[str, Any],
    intent_truth: dict[str, Any],
    action_truth: dict[str, Any],
) -> dict[str, Any]:
    return {
        "T1": observation_protocol_truth["protocol_context_profile"],
        "T2": observation_protocol_truth["risk_zone_label"],
        "T3": observation_protocol_truth["viewpoint_summary"],
        "T4": observation_protocol_truth["primary_degradation"],
        "T5": observation_protocol_truth["trusted_modality"],
        "T6": {
            "usability": observation_protocol_truth["usability"],
            "cause_label": observation_protocol_truth["cause_label"],
        },
        "T7": intent_truth["cleaned_instruction"],
        "T8": list(intent_truth["constraints"]),
        "T9": intent_truth["target_goal"],
        "T10": action_truth["action_token_gt"],
        "T11": action_truth["abstention_decision"],
        "T12": dict(action_truth["collaboration_gt"]),
    }
