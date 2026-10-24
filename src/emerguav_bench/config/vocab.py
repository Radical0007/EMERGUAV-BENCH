from __future__ import annotations

# Canonical benchmark action tokens and legacy aliases.
BENCHMARK_ACTION_TOKENS = {
    "proceed",
    "hover",
    "reobserve",
    "request_collaboration",
    "request_collab",
    "abort",
    "alert_escalation",
    "alert",
}

LEGACY_ACTION_TOKENS = {
    "Hover",
    "Proceed_Slow",
    "Ascend_Cautiously",
    "Descend_Cautiously",
    "Turn_Left_Slow",
    "Turn_Right_Slow",
    "Reobserve",
    "Ask_Other_UAV",
    "Abort_Mission",
    "Hover_and_Alert",
}

ACTION_TOKENS = BENCHMARK_ACTION_TOKENS | LEGACY_ACTION_TOKENS

# Normalize accepted action spellings to canonical tokens.
ACTION_TOKEN_CANONICAL_MAP = {
    "proceed": "proceed",
    "Proceed_Slow": "proceed",
    "Ascend_Cautiously": "proceed",
    "Descend_Cautiously": "proceed",
    "Turn_Left_Slow": "proceed",
    "Turn_Right_Slow": "proceed",
    "hover": "hover",
    "Hover": "hover",
    "reobserve": "reobserve",
    "Reobserve": "reobserve",
    "request_collaboration": "request_collaboration",
    "request_collab": "request_collaboration",
    "Ask_Other_UAV": "request_collaboration",
    "abort": "abort",
    "Abort_Mission": "abort",
    "alert_escalation": "alert_escalation",
    "alert": "alert_escalation",
    "Hover_and_Alert": "alert_escalation",
}

RISK_LEVELS = {"low", "medium", "high", "critical"}
DEGRADATION_SEVERITIES = {"low", "medium", "high", "critical"}
OBSERVATION_SEVERITY_CODES = {"SV1", "SV2", "SV3", "SV4", "SV5"}
TRUSTED_MODALITIES = {"rgb", "lidar", "pose", "multi", "visual_rgb", "lidar_3d", "fused"}
SOURCE_DOMAIN_BUCKETS = {
    "town_like",
    "valley_open_area_like",
    "airport_like",
    "island_coastal_like",
}

EMERGENCY_MANAGEMENT_PILLARS = {
    "industrial_safety": "Industrial Operational Safety",
    "disaster_prevention": "Disaster Prevention & Mitigation",
    "rescue_response": "Emergency Rescue Response",
}

APPLICATION_THEMES = {
    "Extreme Climate & Low-Visibility Operations",
    "Hyper-Dense Urban Airspace",
    "Cultural & Religious Sensitivity",
    "Landmark, Tourism & Media Permission",
    "Smart-City Infrastructure Interaction",
    "Multilingual & High-Entropy Operations",
    "Emergency, Ethical & Public-Safety Events",
    "Cross-Border / Cross-Culture Generalization",
}

POLICY_FAMILIES = {
    "General Flight Operations",
    "Worship-Site Respect",
    "Memorial & Cemetery Respect",
    "Crowd-Sensitive Ceremony",
    "Privacy-Sensitive Institutions",
    "Permit-Controlled Landmark Media",
    "Government / Critical Facility Sensitivity",
}

# Controlled 6-level Language Entropy Taxonomy (L0 ~ L5)
LANGUAGE_CONDITION_TAXONOMY = {
    "L0": "L0_canonical_baseline",
    "L1": "L1_orthographic_perturbation",
    "L2": "L2_lexical_code_switching",
    "L3": "L3_temporal_disfluency",
    "L4": "L4_tactical_abbreviation",
    "L5": "L5_compound_high_entropy",
}

LANGUAGE_ENTROPY_LABELS = {
    "clean_prompt",
    "typo_omission_abbreviation",
    "code_switching",
    "hesitation_revision",
    "operator_shorthand",
    "mixed_entropy",
    *LANGUAGE_CONDITION_TAXONOMY.values(),
}
LANGUAGE_ENTROPY_LEVELS = {"EL1", "EL2", "EL3", "EL4", "EL5", "L0", "L1", "L2", "L3", "L4", "L5"}

# Legacy report aliases; public D1-D6 indices are defined in taxonomy.
CORE_DIAGNOSTIC_DIMENSIONS = {
    "D_vis": "Scene condition",
    "D_lidar": "Mission viewpoint",
    "D_trust": "Evidence selection",
    "D_intent": "Instruction recovery",
    "D_constr": "Emergency constraints",
    "D_action": "Safe action",
}

ABSTENTION_DECISIONS = {
    "continue_execution",
    "reobserve",
    "hold_position",
    "abort_mission",
}

COLLABORATION_REQUEST_TYPES = {
    "none",
    "reobserve_context",
    "traffic_check",
    "policy_clarification",
    "crowd_assessment",
    "docking_confirmation",
}

TASK_ID_TO_NAME = {
    "T1": "Protocol Context Reporting",
    "T2": "Risk Zone Awareness",
    "T3": "Viewpoint Situation Summary",
    "T4": "Observation Degradation Assessment",
    "T5": "Modality Trust Selection",
    "T6": "Usability & Causal Assessment",
    "T7": "Intent Recovery",
    "T8": "Constraint Extraction",
    "T9": "Target / Goal Disambiguation",
    "T10": "Safe Next-Action Selection",
    "T11": "Abstention / Reobserve Decision",
    "T12": "Collaboration Triggering",
}

RISK_LEVEL_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}

DEGRADATION_CAUSE_LABELS = {
    "clean_visual": "clean_observation",
    "dust_veil": "airborne_visibility_loss",
    "heat_shimmer": "thermal_distortion",
    "high_glare": "optical_saturation",
    "blur": "motion_blur",
    "small_target_long_distance": "scale_limit",
    "occlusion": "occluded_view",
    "noise": "low_snr",
    "partial_masking": "masked_region",
    "data_loss": "sensor_dropout",
}

SEVERITY_TO_OBSERVATION_CODE = {
    "low": "SV2",
    "medium": "SV3",
    "high": "SV4",
    "critical": "SV5",
}

LANGUAGE_LABEL_TO_DEFAULT_LEVEL = {
    "clean_prompt": "EL1",
    "typo_omission_abbreviation": "EL3",
    "code_switching": "EL4",
    "hesitation_revision": "EL4",
    "operator_shorthand": "EL3",
    "mixed_entropy": "EL5",
}

ABSTENTION_DECISION_BY_ACTION = {
    "Hover": "hold_position",
    "Hover_and_Alert": "hold_position",
    "Reobserve": "reobserve",
    "Ask_Other_UAV": "hold_position",
    "Abort_Mission": "abort_mission",
    "Proceed_Slow": "continue_execution",
    "Ascend_Cautiously": "continue_execution",
    "Descend_Cautiously": "continue_execution",
    "Turn_Left_Slow": "continue_execution",
    "Turn_Right_Slow": "continue_execution",
}

DEFAULT_LANGUAGE_ENTROPY_LABEL = "typo_omission_abbreviation"


def max_risk_level(left: str, right: str) -> str:
    return left if RISK_LEVEL_RANK[left] >= RISK_LEVEL_RANK[right] else right


def resolve_observation_severity_code(
    degradation_severity: str,
    degradation_labels: list[str] | None = None,
) -> str:
    primary_degradation = (degradation_labels or ["clean_visual"])[0]
    if primary_degradation == "clean_visual":
        return "SV1"
    return SEVERITY_TO_OBSERVATION_CODE.get(degradation_severity, "SV3")


def resolve_language_entropy_level(
    language_entropy_labels: list[str] | None,
    requested_level: str | None = None,
) -> str:
    labels = list(language_entropy_labels or ["clean_prompt"])
    if requested_level and requested_level in LANGUAGE_ENTROPY_LEVELS:
        return requested_level
    if len(labels) == 1:
        return LANGUAGE_LABEL_TO_DEFAULT_LEVEL.get(labels[0], "EL3")
    return "EL5"


def derive_abstention_decision(action_token: str) -> str:
    return ABSTENTION_DECISION_BY_ACTION.get(action_token, "continue_execution")
