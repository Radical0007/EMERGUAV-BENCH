from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class ProtocolContextTemplate:
    protocol_context_id: str
    protocol_context: str
    source_domain_bucket: str
    supported_source_domain_buckets: list[str]
    representative_sequence_placeholder: str
    deployment_framing: str
    application_theme: str
    default_policy_family: str
    default_target_goal: str
    typical_risks: list[str]
    typical_preferred_actions: list[str]
    default_viewpoint_summary: str = "constrained_viewpoint"
    is_overlay: bool = False
    overlay_target: str | None = None
    default_constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def supports_source_domain_bucket(self, source_domain_bucket: str) -> bool:
        return source_domain_bucket in self.supported_source_domain_buckets


PROTOCOL_CONTEXT_TEMPLATES: dict[str, ProtocolContextTemplate] = {
    "Airport Perimeter Monitoring": ProtocolContextTemplate(
        protocol_context_id="S1",
        protocol_context="Airport Perimeter Monitoring",
        source_domain_bucket="airport_like",
        supported_source_domain_buckets=["airport_like"],
        representative_sequence_placeholder="SEQ_AIRPORT_01",
        deployment_framing="机场周界巡检 / restricted corridor / runway-adjacent monitoring",
        application_theme="Hyper-Dense Urban Airspace",
        default_policy_family="Government / Critical Facility Sensitivity",
        default_target_goal="monitor_airport_perimeter_safely",
        typical_risks=["nfz_adjacency", "high_glare", "local_occlusion", "vehicle_misidentification"],
        typical_preferred_actions=["Hover", "Reobserve", "Hover_and_Alert", "Abort_Mission"],
        default_viewpoint_summary="restricted_viewpoint",
        default_constraints=["avoid_nfz", "maintain_corridor_clearance"],
    ),
    "Smart-City Boulevard Patrol": ProtocolContextTemplate(
        protocol_context_id="S2",
        protocol_context="Smart-City Boulevard Patrol",
        source_domain_bucket="town_like",
        supported_source_domain_buckets=["town_like"],
        representative_sequence_placeholder="SEQ_TOWN_01",
        deployment_framing="智慧城市主干路巡检 / boulevard patrol / roadside monitoring",
        application_theme="Hyper-Dense Urban Airspace",
        default_policy_family="General Flight Operations",
        default_target_goal="maintain_boulevard_patrol_coverage",
        typical_risks=["building_reflection", "complex_background", "target_confusion", "road_occlusion"],
        typical_preferred_actions=["Proceed_Slow", "Turn_Left_Slow", "Turn_Right_Slow", "Ask_Other_UAV"],
        default_viewpoint_summary="navigation_limited_viewpoint",
        default_constraints=["maintain_lane_context", "slow_navigation"],
    ),
    "Coastal / Port-side Inspection": ProtocolContextTemplate(
        protocol_context_id="S3",
        protocol_context="Coastal / Port-side Inspection",
        source_domain_bucket="island_coastal_like",
        supported_source_domain_buckets=["island_coastal_like"],
        representative_sequence_placeholder="SEQ_COAST_01",
        deployment_framing="滨海设施与港口巡检 / coastal infrastructure inspection",
        application_theme="Extreme Climate & Low-Visibility Operations",
        default_policy_family="General Flight Operations",
        default_target_goal="inspect_coastal_or_portside_assets",
        typical_risks=["water_glare", "boundary_constraint", "restricted_proximity", "low_texture_region"],
        typical_preferred_actions=["Reobserve", "Proceed_Slow", "Hover_and_Alert"],
        default_viewpoint_summary="boundary_sensitive_viewpoint",
        default_constraints=["respect_boundary_zone", "avoid_restricted_proximity"],
    ),
    "Desert-Edge Logistics Corridor": ProtocolContextTemplate(
        protocol_context_id="S4",
        protocol_context="Desert-Edge Logistics Corridor",
        source_domain_bucket="valley_open_area_like",
        supported_source_domain_buckets=["valley_open_area_like"],
        representative_sequence_placeholder="SEQ_DESERT_01",
        deployment_framing="沙漠边缘物流走廊监测 / low-visibility corridor monitoring",
        application_theme="Extreme Climate & Low-Visibility Operations",
        default_policy_family="General Flight Operations",
        default_target_goal="preserve_low_visibility_corridor_safety",
        typical_risks=["dust_veil", "heat_shimmer", "low_visibility", "small_target_long_distance"],
        typical_preferred_actions=["Hover", "Ascend_Cautiously", "Abort_Mission"],
        default_viewpoint_summary="low_visibility_viewpoint",
        default_constraints=["maintain_visibility_margin", "preserve_corridor_safety"],
    ),
    "NFZ Edge Caution": ProtocolContextTemplate(
        protocol_context_id="S5",
        protocol_context="NFZ Edge Caution",
        source_domain_bucket="overlay_on_any_bucket",
        supported_source_domain_buckets=["town_like", "valley_open_area_like", "airport_like", "island_coastal_like"],
        representative_sequence_placeholder="SEQ_OVERLAY_01",
        deployment_framing="虚拟限制区邻接飞行 / no-approach / caution-sector operation",
        application_theme="Hyper-Dense Urban Airspace",
        default_policy_family="Government / Critical Facility Sensitivity",
        default_target_goal="respect_virtual_nfz_boundary",
        typical_risks=["corridor_violation", "nfz_proximity", "no_approach_sector", "high_risk_proximity"],
        typical_preferred_actions=["Hover", "Reobserve", "Abort_Mission", "Hover_and_Alert"],
        default_viewpoint_summary="restricted_viewpoint",
        is_overlay=True,
        overlay_target="any_source_domain_bucket",
        default_constraints=["avoid_nfz", "respect_no_approach_sector"],
    ),
    "Sensitive-Site Respect": ProtocolContextTemplate(
        protocol_context_id="S6",
        protocol_context="Sensitive-Site Respect",
        source_domain_bucket="town_like",
        supported_source_domain_buckets=["town_like", "island_coastal_like", "airport_like"],
        representative_sequence_placeholder="SEQ_SENSITIVE_01",
        deployment_framing="宗教/纪念/隐私敏感区域 / policy-driven respectful operation",
        application_theme="Cultural & Religious Sensitivity",
        default_policy_family="Worship-Site Respect",
        default_target_goal="maintain_sensitive_site_buffer",
        typical_risks=["privacy_intrusion", "culturally_unsafe_filming", "buffer_violation", "quiet_zone_breach"],
        typical_preferred_actions=["Reobserve", "Hover", "Abort_Mission"],
        default_viewpoint_summary="policy_limited_viewpoint",
        default_constraints=["respect_sensitive_site_policy", "prefer_conservative_operation"],
    ),
    "Landmark & Media Compliance": ProtocolContextTemplate(
        protocol_context_id="S7",
        protocol_context="Landmark & Media Compliance",
        source_domain_bucket="town_like",
        supported_source_domain_buckets=["town_like", "island_coastal_like"],
        representative_sequence_placeholder="SEQ_LANDMARK_01",
        deployment_framing="地标拍摄与旅游媒体任务 / permit-controlled filming",
        application_theme="Landmark, Tourism & Media Permission",
        default_policy_family="Permit-Controlled Landmark Media",
        default_target_goal="capture_compliant_landmark_view",
        typical_risks=["unauthorized_filming", "crowd_adjacent_capture", "unsafe_orbit", "permit_mismatch"],
        typical_preferred_actions=["Proceed_Slow", "Hover", "Ask_Other_UAV"],
        default_viewpoint_summary="permit_constrained_viewpoint",
        default_constraints=["respect_media_policy", "confirm_capture_window"],
    ),
    "Smart Infrastructure Interaction": ProtocolContextTemplate(
        protocol_context_id="S8",
        protocol_context="Smart Infrastructure Interaction",
        source_domain_bucket="town_like",
        supported_source_domain_buckets=["town_like", "airport_like"],
        representative_sequence_placeholder="SEQ_INFRA_01",
        deployment_framing="无人机港、设施 dispatch、道路应急联动 / infrastructure interaction",
        application_theme="Smart-City Infrastructure Interaction",
        default_policy_family="General Flight Operations",
        default_target_goal="complete_dispatch_or_docking_sequence",
        typical_risks=["wrong_docking_approach", "unsafe_descent", "dispatch_conflict", "command_mismatch"],
        typical_preferred_actions=["Descend_Cautiously", "Hover", "Proceed_Slow"],
        default_viewpoint_summary="task_constrained_viewpoint",
        default_constraints=["respect_dispatch_priority", "confirm_target_zone_before_descent"],
    ),
    "EmergUAV Protocol": ProtocolContextTemplate(
        protocol_context_id="S9",
        protocol_context="EmergUAV Protocol",
        source_domain_bucket="town_like",
        supported_source_domain_buckets=["town_like", "valley_open_area_like", "airport_like", "island_coastal_like"],
        representative_sequence_placeholder="SEQ_EMERGUAV_01",
        deployment_framing="应急无人机决策基准 / emergency UAV decision benchmark",
        application_theme="Emergency, Ethical & Public-Safety Events",
        default_policy_family="General Flight Operations",
        default_target_goal="maintain_emergency_mission_safety",
        typical_risks=["airborne_visibility_loss", "nfz_proximity", "thermal_distortion", "operator_command_ambiguity"],
        typical_preferred_actions=["Hover", "Proceed_Slow", "Reobserve", "Ask_Other_UAV", "Abort_Mission", "Hover_and_Alert"],
        default_viewpoint_summary="constrained_viewpoint",
        default_constraints=["avoid_nfz", "maintain_corridor_safety", "slow_navigation"],
    ),
    "EmergUAV-Bench Emergency Protocol": ProtocolContextTemplate(
        protocol_context_id="S9",
        protocol_context="EmergUAV-Bench Emergency Protocol",
        source_domain_bucket="town_like",
        supported_source_domain_buckets=["town_like", "valley_open_area_like", "airport_like", "island_coastal_like"],
        representative_sequence_placeholder="SEQ_EMERGUAV_01",
        deployment_framing="应急无人机决策基准 / emergency UAV decision benchmark",
        application_theme="Emergency, Ethical & Public-Safety Events",
        default_policy_family="General Flight Operations",
        default_target_goal="maintain_emergency_mission_safety",
        typical_risks=["airborne_visibility_loss", "nfz_proximity", "thermal_distortion", "operator_command_ambiguity"],
        typical_preferred_actions=["Hover", "Proceed_Slow", "Reobserve", "Ask_Other_UAV", "Abort_Mission", "Hover_and_Alert"],
        default_viewpoint_summary="constrained_viewpoint",
        default_constraints=["avoid_nfz", "maintain_corridor_safety", "slow_navigation"],
    ),
    "Industrial Operational Safety": ProtocolContextTemplate(
        protocol_context_id="S10",
        protocol_context="Industrial Operational Safety",
        source_domain_bucket="town_like",
        supported_source_domain_buckets=["town_like", "valley_open_area_like", "airport_like", "island_coastal_like"],
        representative_sequence_placeholder="SEQ_IND_01",
        deployment_framing="工业安全生产应急巡检 / industrial operational safety emergency",
        application_theme="Emergency, Ethical & Public-Safety Events",
        default_policy_family="General Flight Operations",
        default_target_goal="maintain_industrial_safety_corridor",
        typical_risks=["infrastructure_hazard", "restricted_proximity", "smoke_dispersion", "structural_failure"],
        typical_preferred_actions=["Hover", "Proceed_Slow", "Reobserve", "Abort_Mission"],
        default_viewpoint_summary="constrained_viewpoint",
        default_constraints=["avoid_hazard_zone", "maintain_safety_clearance", "slow_navigation"],
    ),
    "Disaster Prevention & Mitigation": ProtocolContextTemplate(
        protocol_context_id="S11",
        protocol_context="Disaster Prevention & Mitigation",
        source_domain_bucket="valley_open_area_like",
        supported_source_domain_buckets=["town_like", "valley_open_area_like", "airport_like", "island_coastal_like"],
        representative_sequence_placeholder="SEQ_DISASTER_01",
        deployment_framing="自然灾害防灾减灾应急巡检 / disaster prevention and mitigation",
        application_theme="Emergency, Ethical & Public-Safety Events",
        default_policy_family="General Flight Operations",
        default_target_goal="execute_disaster_mitigation_patrol",
        typical_risks=["wildfire_spread", "flood_boundary", "airborne_visibility_loss", "sudden_wind_shear"],
        typical_preferred_actions=["Hover", "Reobserve", "Hover_and_Alert", "Ask_Other_UAV"],
        default_viewpoint_summary="low_visibility_viewpoint",
        default_constraints=["avoid_nfz", "preserve_corridor_safety", "report_hotspot"],
    ),
    "Emergency Rescue Response": ProtocolContextTemplate(
        protocol_context_id="S12",
        protocol_context="Emergency Rescue Response",
        source_domain_bucket="island_coastal_like",
        supported_source_domain_buckets=["town_like", "valley_open_area_like", "airport_like", "island_coastal_like"],
        representative_sequence_placeholder="SEQ_RESCUE_01",
        deployment_framing="应急救援搜救与现场处置 / emergency rescue response",
        application_theme="Emergency, Ethical & Public-Safety Events",
        default_policy_family="General Flight Operations",
        default_target_goal="locate_and_protect_trapped_survivors",
        typical_risks=["crowd_proximity", "obstacle_proximity", "missing_gps", "survivor_distress"],
        typical_preferred_actions=["Hover", "Proceed_Slow", "Descend_Cautiously", "Ask_Other_UAV"],
        default_viewpoint_summary="restricted_viewpoint",
        default_constraints=["protect_personnel", "maintain_standoff", "request_collaboration"],
    ),
}
