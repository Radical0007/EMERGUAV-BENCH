from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class PolicyPack:
    family_name: str
    risk_zone_label: str
    default_constraints: list[str]
    preferred_actions: list[str]
    forbidden_actions: list[str]
    collaboration_request_type: str
    overlay_profiles: list[str] = field(default_factory=list)
    risk_floor: str | None = None
    safety_priority: bool = False
    default_collaboration_required: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


POLICY_PACKS: dict[str, PolicyPack] = {
    "General Flight Operations": PolicyPack(
        family_name="General Flight Operations",
        risk_zone_label="general_flight_zone",
        default_constraints=["respect_base_mission_constraints"],
        preferred_actions=["Proceed_Slow", "Reobserve", "Hover"],
        forbidden_actions=[],
        collaboration_request_type="none",
        overlay_profiles=["baseline_operation"],
    ),
    "Worship-Site Respect": PolicyPack(
        family_name="Worship-Site Respect",
        risk_zone_label="sensitive_zone",
        default_constraints=["maintain_sensitive_site_buffer", "avoid_close_shot", "maintain_quiet_altitude"],
        preferred_actions=["Reobserve", "Hover", "Abort_Mission"],
        forbidden_actions=["Descend_Cautiously", "Proceed_Slow"],
        collaboration_request_type="policy_clarification",
        overlay_profiles=["sensitive_context", "quiet_operation_window"],
        risk_floor="high",
        safety_priority=True,
    ),
    "Memorial & Cemetery Respect": PolicyPack(
        family_name="Memorial & Cemetery Respect",
        risk_zone_label="memorial_sensitive_zone",
        default_constraints=["avoid_close_filming", "maintain_memorial_buffer"],
        preferred_actions=["Hover", "Reobserve", "Abort_Mission"],
        forbidden_actions=["Descend_Cautiously", "Proceed_Slow"],
        collaboration_request_type="policy_clarification",
        overlay_profiles=["sensitive_context", "dignity_preserving_operation"],
        risk_floor="high",
        safety_priority=True,
    ),
    "Crowd-Sensitive Ceremony": PolicyPack(
        family_name="Crowd-Sensitive Ceremony",
        risk_zone_label="crowd_sensitive_zone",
        default_constraints=["avoid_overflight_on_crowd", "maintain_safe_buffer"],
        preferred_actions=["Hover", "Ask_Other_UAV", "Abort_Mission"],
        forbidden_actions=["Descend_Cautiously", "Proceed_Slow"],
        collaboration_request_type="crowd_assessment",
        overlay_profiles=["crowd_sensitive_event", "temporary_buffer_zone"],
        risk_floor="high",
        safety_priority=True,
        default_collaboration_required=True,
    ),
    "Privacy-Sensitive Institutions": PolicyPack(
        family_name="Privacy-Sensitive Institutions",
        risk_zone_label="privacy_sensitive_zone",
        default_constraints=["avoid_close_shot", "maintain_privacy_buffer"],
        preferred_actions=["Reobserve", "Hover", "Abort_Mission"],
        forbidden_actions=["Descend_Cautiously"],
        collaboration_request_type="policy_clarification",
        overlay_profiles=["privacy_sensitive_context"],
        risk_floor="medium",
        safety_priority=True,
    ),
    "Permit-Controlled Landmark Media": PolicyPack(
        family_name="Permit-Controlled Landmark Media",
        risk_zone_label="permit_controlled_zone",
        default_constraints=["confirm_media_permit", "respect_crowd_adjacent_buffer"],
        preferred_actions=["Proceed_Slow", "Hover", "Ask_Other_UAV"],
        forbidden_actions=["Descend_Cautiously"],
        collaboration_request_type="policy_clarification",
        overlay_profiles=["permit_controlled_context", "media_capture_window"],
        risk_floor="medium",
        default_collaboration_required=True,
    ),
    "Government / Critical Facility Sensitivity": PolicyPack(
        family_name="Government / Critical Facility Sensitivity",
        risk_zone_label="restricted_sensitive_zone",
        default_constraints=["maintain_standoff_distance", "respect_critical_facility_buffer"],
        preferred_actions=["Hover_and_Alert", "Hover", "Abort_Mission"],
        forbidden_actions=["Descend_Cautiously", "Proceed_Slow"],
        collaboration_request_type="traffic_check",
        overlay_profiles=["critical_facility_context", "restricted_airspace"],
        risk_floor="high",
        safety_priority=True,
        default_collaboration_required=True,
    ),
}
