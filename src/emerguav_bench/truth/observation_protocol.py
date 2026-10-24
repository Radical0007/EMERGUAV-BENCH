from __future__ import annotations

from emerguav_bench.config.vocab import DEGRADATION_CAUSE_LABELS, max_risk_level
from emerguav_bench.protocol_context.policies import POLICY_PACKS


def derive_observation_protocol_truth(
    *,
    degradation_labels: list[str],
    degradation_severity: str,
    source_domain_bucket: str,
    protocol_context: str | None = None,
    application_theme: str | None = None,
    policy_family: str = "General Flight Operations",
    overlay_profiles: list[str] | None = None,
    enable_nfz_overlay: bool = False,
    viewpoint_summary: str | None = None,
    trusted_modality: str | None = None,
    risk_level: str | None = None,
) -> dict[str, str]:
    primary_degradation = degradation_labels[0] if degradation_labels else "clean_visual"
    policy = POLICY_PACKS.get(policy_family, POLICY_PACKS["General Flight Operations"])
    resolved_overlay_profiles = set(overlay_profiles or [])
    has_nfz_overlay = enable_nfz_overlay or "NFZ Edge Caution" in resolved_overlay_profiles or "nfz_edge_caution" in resolved_overlay_profiles

    severity_to_usability = {
        "low": "high",
        "medium": "medium",
        "high": "low",
        "critical": "unusable",
    }
    severity_to_risk = {
        "low": "low",
        "medium": "medium",
        "high": "high",
        "critical": "critical",
    }

    if trusted_modality is not None:
        resolved_trusted_modality = trusted_modality
    elif primary_degradation in {"clean_visual"} and source_domain_bucket == "airport_like":
        resolved_trusted_modality = "pose"
    elif protocol_context == "Smart Infrastructure Interaction" and degradation_severity in {"low", "medium"}:
        resolved_trusted_modality = "pose"
    elif primary_degradation in {"dust_veil", "heat_shimmer", "high_glare", "noise", "data_loss"}:
        resolved_trusted_modality = "lidar"
    elif source_domain_bucket == "town_like" and degradation_severity in {"low", "medium"}:
        resolved_trusted_modality = "multi"
    else:
        resolved_trusted_modality = "lidar"

    if risk_level is not None:
        resolved_risk_level = risk_level
    else:
        resolved_risk_level = severity_to_risk.get(degradation_severity, "medium")
        if policy.risk_floor:
            resolved_risk_level = max_risk_level(resolved_risk_level, policy.risk_floor)
        if has_nfz_overlay:
            resolved_risk_level = max_risk_level(resolved_risk_level, "high")
        if application_theme == "Emergency, Ethical & Public-Safety Events":
            resolved_risk_level = max_risk_level(resolved_risk_level, "high")

    if has_nfz_overlay:
        risk_zone_label = "nfz_edge"
    elif policy.family_name != "General Flight Operations":
        risk_zone_label = policy.risk_zone_label
    elif source_domain_bucket == "airport_like":
        risk_zone_label = "restricted_corridor"
    elif source_domain_bucket == "town_like":
        risk_zone_label = "urban_canyon"
    elif source_domain_bucket == "island_coastal_like":
        risk_zone_label = "boundary_sensitive_zone"
    else:
        risk_zone_label = "open_corridor"

    if viewpoint_summary is not None:
        resolved_viewpoint_summary = viewpoint_summary
    elif resolved_risk_level == "critical":
        resolved_viewpoint_summary = "dangerous_viewpoint"
    elif resolved_risk_level == "high":
        resolved_viewpoint_summary = "constrained_viewpoint"
    elif policy.safety_priority or has_nfz_overlay:
        resolved_viewpoint_summary = "policy_limited_viewpoint"
    elif source_domain_bucket == "valley_open_area_like":
        resolved_viewpoint_summary = "low_visibility_viewpoint"
    else:
        resolved_viewpoint_summary = "mission_favorable_viewpoint"

    cause_label = DEGRADATION_CAUSE_LABELS.get(primary_degradation, "unknown_cause")

    return {
        "protocol_context_profile": protocol_context or source_domain_bucket,
        "primary_degradation": primary_degradation,
        "usability": severity_to_usability.get(degradation_severity, "medium"),
        "trusted_modality": resolved_trusted_modality,
        "risk_level": resolved_risk_level,
        "risk_zone_label": risk_zone_label,
        "viewpoint_summary": resolved_viewpoint_summary,
        "cause_label": cause_label,
    }
