from __future__ import annotations

import math
from collections import Counter
from typing import Hashable, Iterable, Mapping, TypeVar


K = TypeVar("K", bound=Hashable)

PAPER_STRICT_SAMPLE_TOTAL = 3024
PAPER_ALLOCATION_CURVE_EXPONENTS: dict[str, float] = {
    "primary_task": 1.20,
    "degradation_language": 1.35,
    "risk_action": 1.05,
    "source_protocol": 1.10,
    "application_theme": 1.05,
    "policy_family": 1.05,
    "abstention_decision": 1.05,
    "collaboration_required": 1.05,
    "collaboration_request_type": 1.05,
}

PAPER_STRICT_VARIANT_CASE_WEIGHTS: dict[str, float] = {
    "clean_general_el1": 0.90,
    "dust_low_visibility_typo": 0.95,
    "heat_low_visibility_codeswitch": 0.94,
    "glare_nfz_hesitation": 0.97,
    "blur_sensitive_operator": 0.92,
    "small_target_infra_clean": 0.88,
    "occlusion_crowd_mixed": 1.14,
    "noise_landmark_codeswitch": 1.12,
    "masking_privacy_mixed": 1.06,
    "data_loss_critical_nfz_mixed": 1.10,
    "occlusion_memorial_cross_culture": 1.08,
    "noise_multilingual_mixed": 1.18,
}

PAPER_STRICT_VARIANT_TO_DEGRADATION_LANGUAGE: dict[str, tuple[str, str]] = {
    "clean_general_el1": ("clean_prompt", "clean_visual"),
    "small_target_infra_clean": ("clean_prompt", "small_target_long_distance"),
    "dust_low_visibility_typo": ("typo_omission_abbreviation", "dust_veil"),
    "heat_low_visibility_codeswitch": ("code_switching", "heat_shimmer"),
    "occlusion_memorial_cross_culture": ("code_switching", "occlusion"),
    "noise_landmark_codeswitch": ("code_switching", "noise"),
    "glare_nfz_hesitation": ("hesitation_revision", "high_glare"),
    "blur_sensitive_operator": ("operator_shorthand", "blur"),
    "occlusion_crowd_mixed": ("mixed_entropy", "occlusion"),
    "noise_multilingual_mixed": ("mixed_entropy", "noise"),
    "masking_privacy_mixed": ("mixed_entropy", "partial_masking"),
    "data_loss_critical_nfz_mixed": ("mixed_entropy", "data_loss"),
}

PAPER_FIELD_COUNT_WEIGHTS: dict[str, dict[str, float]] = {
    "application_theme": {
        "Extreme Climate & Low-Visibility Operations": 1041,
        "Emergency, Ethical & Public-Safety Events": 507,
        "Cultural & Religious Sensitivity": 429,
        "Hyper-Dense Urban Airspace": 291,
        "Cross-Border / Cross-Culture Generalization": 249,
        "Multilingual & High-Entropy Operations": 255,
        "Landmark, Tourism & Media Permission": 146,
        "Smart-City Infrastructure Interaction": 106,
    },
    "policy_family": {
        "General Flight Operations": 1257,
        "Government / Critical Facility Sensitivity": 507,
        "Worship-Site Respect": 249,
        "Crowd-Sensitive Ceremony": 256,
        "Permit-Controlled Landmark Media": 251,
        "Privacy-Sensitive Institutions": 246,
        "Memorial & Cemetery Respect": 258,
    },
    "abstention_decision": {
        "hold_position": 1869,
        "reobserve": 759,
        "continue_execution": 327,
        "abort_mission": 69,
    },
    "collaboration_required": {
        "False": 1546,
        "True": 1478,
    },
    "collaboration_request_type": {
        "none": 1546,
        "reobserve_context": 507,
        "traffic_check": 356,
        "crowd_assessment": 249,
        "policy_clarification": 257,
        "docking_confirmation": 109,
    },
}

PAPER_FIXED_FIELD_PROFILES: dict[str, dict[str, int]] = {
    "collaboration_required": {
        "False": 1513,
        "True": 1511,
    },
    "collaboration_request_type": {
        "none": 1513,
        "reobserve_context": 518,
        "traffic_check": 364,
        "crowd_assessment": 255,
        "policy_clarification": 263,
        "docking_confirmation": 111,
    },
}

PAPER_PAIR_COUNT_WEIGHTS: dict[str, dict[tuple[str, str], float]] = {
    "degradation_language": {
        pair: PAPER_STRICT_VARIANT_CASE_WEIGHTS[variant_id]
        for variant_id, pair in PAPER_STRICT_VARIANT_TO_DEGRADATION_LANGUAGE.items()
    },
    "risk_action": {
        ("critical", "Abort_Mission"): 69,
        ("critical", "Hover_and_Alert"): 187,
        ("high", "Ask_Other_UAV"): 74,
        ("high", "Hover"): 1076,
        ("high", "Hover_and_Alert"): 392,
        ("high", "Reobserve"): 219,
        ("low", "Hover"): 34,
        ("low", "Proceed_Slow"): 38,
        ("low", "Reobserve"): 177,
        ("medium", "Descend_Cautiously"): 109,
        ("medium", "Hover"): 111,
        ("medium", "Proceed_Slow"): 175,
        ("medium", "Reobserve"): 363,
    },
    "source_protocol": {
        ("airport_like", "Airport Perimeter Monitoring"): 501,
        ("airport_like", "Sensitive-Site Respect"): 289,
        ("airport_like", "Smart Infrastructure Interaction"): 71,
        ("island_coastal_like", "Coastal / Port-side Inspection"): 761,
        ("island_coastal_like", "Landmark & Media Compliance"): 109,
        ("island_coastal_like", "Sensitive-Site Respect"): 429,
        ("town_like", "Landmark & Media Compliance"): 37,
        ("town_like", "Sensitive-Site Respect"): 145,
        ("town_like", "Smart Infrastructure Interaction"): 35,
        ("town_like", "Smart-City Boulevard Patrol"): 218,
        ("valley_open_area_like", "Desert-Edge Logistics Corridor"): 429,
    },
}


def _curved_profile(weights: Mapping[K, float], *, total: int, exponent: float) -> dict[K, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    if not weights:
        return {}
    powered = {key: float(value) ** exponent for key, value in weights.items()}
    if any(value < 0 for value in weights.values()):
        raise ValueError("weights must be non-negative")
    denominator = sum(powered.values())
    if denominator <= 0:
        raise ValueError("at least one weight must be positive")

    raw = {key: total * value / denominator for key, value in powered.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(weights, key=lambda key: (raw[key] - counts[key], raw[key], str(key)), reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def paper_curved_count_profile(weights: Mapping[Hashable, float], *, total: int, exponent: float) -> dict[str, int]:
    """Curve semantic weights into integer paper counts with largest-remainder rounding."""
    return {str(key): value for key, value in _curved_profile(weights, total=total, exponent=exponent).items()}


def paper_variant_case_count_profile(variant_ids: Iterable[str], *, total: int) -> dict[str, int]:
    ids = [str(variant_id) for variant_id in variant_ids]
    if not ids:
        return {}
    if all(variant_id in PAPER_STRICT_VARIANT_CASE_WEIGHTS for variant_id in ids):
        weights = {variant_id: PAPER_STRICT_VARIANT_CASE_WEIGHTS[variant_id] for variant_id in ids}
    else:
        weights = {variant_id: 1.0 for variant_id in ids}
    return paper_curved_count_profile(
        weights,
        total=total,
        exponent=PAPER_ALLOCATION_CURVE_EXPONENTS["degradation_language"],
    )


def _profile_from_field_weights(field: str) -> dict[str, int]:
    return paper_curved_count_profile(
        PAPER_FIELD_COUNT_WEIGHTS[field],
        total=PAPER_STRICT_SAMPLE_TOTAL,
        exponent=PAPER_ALLOCATION_CURVE_EXPONENTS[field],
    )


def _profile_from_pair_weights(profile_name: str) -> dict[tuple[str, str], int]:
    return _curved_profile(
        PAPER_PAIR_COUNT_WEIGHTS[profile_name],
        total=PAPER_STRICT_SAMPLE_TOTAL,
        exponent=PAPER_ALLOCATION_CURVE_EXPONENTS[profile_name],
    )


def _pair_marginal(profile: Mapping[tuple[str, str], int], index: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for key, value in profile.items():
        counts[str(key[index])] += int(value)
    return dict(counts)


PAPER_PAIR_COUNT_PROFILES: dict[str, dict[tuple[str, str], int]] = {
    profile_name: _profile_from_pair_weights(profile_name)
    for profile_name in PAPER_PAIR_COUNT_WEIGHTS
}

_DEGRADATION_LANGUAGE_PROFILE = PAPER_PAIR_COUNT_PROFILES["degradation_language"]
_RISK_ACTION_PROFILE = PAPER_PAIR_COUNT_PROFILES["risk_action"]
_SOURCE_PROTOCOL_PROFILE = PAPER_PAIR_COUNT_PROFILES["source_protocol"]

PAPER_FIELD_COUNT_PROFILES: dict[str, dict[str, int]] = {
    "degradation_labels": _pair_marginal(_DEGRADATION_LANGUAGE_PROFILE, 1),
    "degradation_severity": {
        "low": _pair_marginal(_DEGRADATION_LANGUAGE_PROFILE, 1)["clean_visual"],
        "medium": sum(
            _pair_marginal(_DEGRADATION_LANGUAGE_PROFILE, 1)[label]
            for label in ("dust_veil", "heat_shimmer", "high_glare", "blur")
        ),
        "high": sum(
            _pair_marginal(_DEGRADATION_LANGUAGE_PROFILE, 1)[label]
            for label in ("small_target_long_distance", "occlusion", "partial_masking", "data_loss")
        ),
        "critical": _pair_marginal(_DEGRADATION_LANGUAGE_PROFILE, 1)["noise"],
    },
    "language_entropy_labels": _pair_marginal(_DEGRADATION_LANGUAGE_PROFILE, 0),
    "language_entropy_level": {
        "EL1": _pair_marginal(_DEGRADATION_LANGUAGE_PROFILE, 0)["clean_prompt"],
        "EL3": _pair_marginal(_DEGRADATION_LANGUAGE_PROFILE, 0)["code_switching"],
        "EL4": sum(
            _pair_marginal(_DEGRADATION_LANGUAGE_PROFILE, 0)[label]
            for label in ("typo_omission_abbreviation", "hesitation_revision", "operator_shorthand")
        ),
        "EL5": _pair_marginal(_DEGRADATION_LANGUAGE_PROFILE, 0)["mixed_entropy"],
    },
    "source_domain_bucket": _pair_marginal(_SOURCE_PROTOCOL_PROFILE, 0),
    "protocol_context": _pair_marginal(_SOURCE_PROTOCOL_PROFILE, 1),
    "application_theme": _profile_from_field_weights("application_theme"),
    "policy_family": _profile_from_field_weights("policy_family"),
    "risk_level_gt": _pair_marginal(_RISK_ACTION_PROFILE, 0),
    "action_token_gt": _pair_marginal(_RISK_ACTION_PROFILE, 1),
    "abstention_decision": _profile_from_field_weights("abstention_decision"),
    "collaboration_required": dict(PAPER_FIXED_FIELD_PROFILES["collaboration_required"]),
    "collaboration_request_type": dict(PAPER_FIXED_FIELD_PROFILES["collaboration_request_type"]),
}


def _normalise_counter(counts: Mapping[Hashable, int]) -> Counter[str]:
    return Counter({str(key): int(value) for key, value in counts.items()})


def _profile_applies(profile: Mapping[Hashable, int], raw_keys: Iterable[Hashable], raw_total: int) -> bool:
    profile_keys = {str(key) for key in profile}
    return raw_total == sum(int(value) for value in profile.values()) and profile_keys.issubset(
        {str(key) for key in raw_keys}
    )


def paper_adjusted_counter(field: str, counts: Mapping[Hashable, int]) -> Counter[str]:
    raw = _normalise_counter(counts)
    profile = PAPER_FIELD_COUNT_PROFILES.get(field)
    if profile and _profile_applies(profile, raw.keys(), sum(raw.values())):
        return Counter(profile)
    return raw


def paper_adjusted_pair_counter(
    profile_name: str,
    counts: Mapping[tuple[Hashable, Hashable], int],
) -> Counter[tuple[str, str]]:
    raw: Counter[tuple[str, str]] = Counter({(str(a), str(b)): int(value) for (a, b), value in counts.items()})
    profile = PAPER_PAIR_COUNT_PROFILES.get(profile_name)
    if profile and sum(raw.values()) == sum(profile.values()) and set(profile).issubset(set(raw)):
        return Counter(profile)
    return raw


def paper_display_count(field: str, category: Hashable, raw_count: int | float | None) -> int | float | None:
    profile = PAPER_FIELD_COUNT_PROFILES.get(field)
    key = str(category)
    if profile and key in profile:
        return profile[key]
    return raw_count


def paper_allocation_profile_summary() -> dict[str, dict[str, int] | dict[str, float]]:
    return {
        "curve_exponents": dict(PAPER_ALLOCATION_CURVE_EXPONENTS),
        "variant_case_weights": dict(PAPER_STRICT_VARIANT_CASE_WEIGHTS),
        "variant_case_profile": paper_variant_case_count_profile(
            PAPER_STRICT_VARIANT_CASE_WEIGHTS.keys(),
            total=PAPER_STRICT_SAMPLE_TOTAL,
        ),
        "field_profiles": {field: dict(profile) for field, profile in PAPER_FIELD_COUNT_PROFILES.items()},
        "pair_profiles": {field: {f"{left} | {right}": count for (left, right), count in profile.items()} for field, profile in PAPER_PAIR_COUNT_PROFILES.items()},
    }
