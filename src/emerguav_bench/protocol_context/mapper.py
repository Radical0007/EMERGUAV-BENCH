from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from emerguav_bench.config.vocab import APPLICATION_THEMES
from emerguav_bench.protocol_context.policies import POLICY_PACKS
from emerguav_bench.protocol_context.templates import PROTOCOL_CONTEXT_TEMPLATES, ProtocolContextTemplate

SCENE_FAMILY_TO_SOURCE_DOMAIN_BUCKET = {
    "AMtown": "town_like",
    "AMvalley": "valley_open_area_like",
    "HKairport": "airport_like",
    "HKisland": "island_coastal_like",
}

SOURCE_DOMAIN_BUCKET_TO_PROTOCOL_CONTEXT = {
    "town_like": "Smart-City Boulevard Patrol",
    "valley_open_area_like": "Desert-Edge Logistics Corridor",
    "airport_like": "Airport Perimeter Monitoring",
    "island_coastal_like": "Coastal / Port-side Inspection",
}


@dataclass(slots=True)
class ProtocolContextAssignment:
    sequence_id: str
    scene_family: str
    source_domain_bucket: str
    protocol_context: str
    application_theme: str
    policy_family: str
    overlay_profiles: list[str]
    mission_constraints: list[str]
    protocol_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_source_domain_bucket(scene_family: str) -> str:
    try:
        return SCENE_FAMILY_TO_SOURCE_DOMAIN_BUCKET[scene_family]
    except KeyError as exc:
        raise KeyError(f"Unsupported UAVScenes scene family: {scene_family}") from exc


def get_protocol_context(
    source_domain_bucket: str, protocol_context: str | None = None
) -> ProtocolContextTemplate:
    protocol_context_name = protocol_context or SOURCE_DOMAIN_BUCKET_TO_PROTOCOL_CONTEXT.get(source_domain_bucket)
    if protocol_context_name is None or protocol_context_name not in PROTOCOL_CONTEXT_TEMPLATES:
        raise KeyError(f"Unsupported protocol context for source_domain_bucket: {source_domain_bucket}")
    protocol_context_template = PROTOCOL_CONTEXT_TEMPLATES[protocol_context_name]
    if protocol_context_template.is_overlay:
        raise KeyError(f"Overlay protocol context '{protocol_context_name}' cannot be selected as the primary context")
    if not protocol_context_template.supports_source_domain_bucket(source_domain_bucket):
        raise KeyError(
            f"Protocol context '{protocol_context_name}' does not support source_domain_bucket '{source_domain_bucket}'"
        )
    return protocol_context_template


def assign_protocol_context(
    *,
    sequence_id: str,
    scene_family: str,
    protocol_context: str | None = None,
    application_theme: str | None = None,
    policy_family: str | None = None,
    enable_nfz_overlay: bool = False,
    overlay_profiles: list[str] | None = None,
    extra_constraints: list[str] | None = None,
) -> ProtocolContextAssignment:
    source_domain_bucket = get_source_domain_bucket(scene_family)
    protocol_context_template = get_protocol_context(source_domain_bucket, protocol_context=protocol_context)
    resolved_application_theme = application_theme or protocol_context_template.application_theme
    if resolved_application_theme not in APPLICATION_THEMES:
        raise KeyError(f"Unsupported application theme: {resolved_application_theme}")
    resolved_policy_family = policy_family or protocol_context_template.default_policy_family
    try:
        policy = POLICY_PACKS[resolved_policy_family]
    except KeyError as exc:
        raise KeyError(f"Unsupported policy family: {resolved_policy_family}") from exc

    mission_constraints = list(protocol_context_template.default_constraints)
    mission_constraints.extend(policy.default_constraints)
    metadata: dict[str, Any] = {
        "protocol_context_id": protocol_context_template.protocol_context_id,
        "deployment_framing": protocol_context_template.deployment_framing,
        "typical_risks": list(protocol_context_template.typical_risks),
        "typical_preferred_actions": list(protocol_context_template.typical_preferred_actions),
        "default_target_goal": protocol_context_template.default_target_goal,
        "default_viewpoint_summary": protocol_context_template.default_viewpoint_summary,
        "application_theme": resolved_application_theme,
        "policy_pack": policy.to_dict(),
        "overlay_protocol_contexts": [],
        "overlay_profiles": list(policy.overlay_profiles),
    }

    if extra_constraints:
        mission_constraints.extend(extra_constraints)

    requested_overlay_profiles = list(overlay_profiles or [])
    if enable_nfz_overlay or "NFZ Edge Caution" in requested_overlay_profiles:
        overlay = PROTOCOL_CONTEXT_TEMPLATES["NFZ Edge Caution"]
        mission_constraints.extend(overlay.default_constraints)
        metadata["overlay_protocol_contexts"].append(overlay.protocol_context)
        metadata["overlay_risks"] = list(overlay.typical_risks)
        metadata["overlay_profiles"].append("nfz_edge_caution")

    for profile in requested_overlay_profiles:
        if profile != "NFZ Edge Caution":
            metadata["overlay_profiles"].append(profile)

    deduped_constraints: list[str] = []
    for constraint in mission_constraints:
        if constraint not in deduped_constraints:
            deduped_constraints.append(constraint)

    deduped_overlay_profiles: list[str] = []
    for profile in metadata["overlay_profiles"]:
        if profile not in deduped_overlay_profiles:
            deduped_overlay_profiles.append(profile)

    return ProtocolContextAssignment(
        sequence_id=sequence_id,
        scene_family=scene_family,
        source_domain_bucket=source_domain_bucket,
        protocol_context=protocol_context_template.protocol_context,
        application_theme=resolved_application_theme,
        policy_family=resolved_policy_family,
        overlay_profiles=deduped_overlay_profiles,
        mission_constraints=deduped_constraints,
        protocol_metadata=metadata,
    )
