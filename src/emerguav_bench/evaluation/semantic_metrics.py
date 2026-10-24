from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from emerguav_bench.config.vocab import TASK_ID_TO_NAME
from emerguav_bench.evaluation.metrics import _collect_evaluation_rows, _safe_divide


METRIC_PROFILE = "benchmark_semantic_v2"


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


_INVALID_TEXT_MARKERS = (
    "unknown",
    "normalization failed",
    "non empty string",
    "placeholder",
    "schema",
    "provided protocol policy risk zone label",
)


def _is_invalid_semantic_text(value: Any) -> bool:
    normalized = _text(value)
    return not normalized or any(marker in normalized for marker in _INVALID_TEXT_MARKERS)


def _contains_any(value: Any, aliases: list[str]) -> bool:
    normalized = _text(value)
    return any(alias in normalized for alias in aliases)


def _canonical_label_score(expected: Any, predicted: Any, alias_map: dict[str, list[str]]) -> float:
    expected_key = str(expected)
    predicted_text = _text(predicted)
    if not expected_key or _is_invalid_semantic_text(predicted):
        return 0.0
    if _text(expected_key) == predicted_text or _text(expected_key) in predicted_text:
        return 1.0
    aliases = alias_map.get(expected_key, [])
    if any(alias in predicted_text for alias in aliases):
        return 1.0
    return 0.0


_RISK_ZONE_ALIASES = {
    "nfz_edge": ["nfz", "no fly", "restricted edge", "restricted zone", "restricted zn", "no approach"],
    "restricted_corridor": ["restricted corridor", "restricted route", "corridor", "restricted zone", "restricted edge"],
    "boundary_sensitive_zone": ["boundary", "restricted", "standoff", "buffer", "edge"],
    "sensitive_zone": ["sensitive", "worship", "privacy", "critical", "facility", "protected"],
    "crowd_sensitive_zone": ["crowd", "public", "gathering", "ceremony", "pedestrian"],
    "permit_controlled_zone": ["permit", "media", "landmark", "controlled", "tourism"],
    "privacy_sensitive_zone": ["privacy", "institution", "school", "hospital", "residential", "sensitive"],
    "memorial_sensitive_zone": ["memorial", "cemetery", "worship", "religious", "sensitive"],
    "urban_canyon": ["urban", "boulevard", "city", "road", "street", "traffic"],
    "open_corridor": ["open", "corridor", "valley", "clear", "rural"],
}


def _risk_zone_score(expected: Any, predicted: Any) -> float:
    if _is_invalid_semantic_text(predicted):
        return 0.0
    exact_or_alias = _canonical_label_score(expected, predicted, _RISK_ZONE_ALIASES)
    if exact_or_alias:
        return exact_or_alias
    if _contains_any(predicted, ["restricted", "nfz", "sensitive", "critical", "protected", "buffer"]):
        return 0.5
    return 0.0


def _viewpoint_score(expected: Any, predicted: Any) -> float:
    expected_text = str(expected)
    predicted_text = _text(predicted)
    if _is_invalid_semantic_text(predicted):
        return 0.0
    if _text(expected_text) == predicted_text or _text(expected_text) in predicted_text:
        return 1.0

    aerial_view = _contains_any(predicted, ["top down", "aerial", "overhead", "nadir", "view"])
    constrained = _contains_any(
        predicted,
        ["constrained", "degraded", "obstruct", "occlu", "limited", "haze", "mask", "data loss", "glare", "blur", "noise"],
    )
    dangerous = _contains_any(predicted, ["danger", "unsafe", "critical", "collision", "near boundary", "violation"])
    low_visibility = _contains_any(predicted, ["low visibility", "haze", "fog", "dust", "poor visibility"])
    favorable = _contains_any(predicted, ["favorable", "clear", "good", "usable", "normal"]) and not constrained

    if expected_text == "mission_favorable_viewpoint":
        if favorable:
            return 1.0
        return 0.5 if aerial_view and not constrained and not dangerous else 0.0
    if expected_text == "constrained_viewpoint":
        if constrained:
            return 1.0
        return 0.25 if aerial_view else 0.0
    if expected_text == "dangerous_viewpoint":
        if dangerous:
            return 1.0
        return 0.5 if constrained else 0.0
    if expected_text == "low_visibility_viewpoint":
        if low_visibility:
            return 1.0
        return 0.5 if constrained else 0.0
    return 0.0


_DEGRADATION_ALIASES = {
    "clean_visual": ["clean", "clear", "normal", "no significant degradation", "no degradation", "none"],
    "dust_veil": ["dust", "haze", "fog", "low visibility", "atmospheric", "veil", "smoke"],
    "heat_shimmer": ["heat", "thermal", "shimmer", "mirage", "distortion"],
    "high_glare": ["glare", "overexposure", "bright", "reflection", "saturation", "washed"],
    "blur": ["blur", "motion blur", "out of focus"],
    "small_target_long_distance": ["small target", "long distance", "distant", "far", "scale", "low resolution"],
    "occlusion": ["occlusion", "occluded", "obstruction", "blocked"],
    "noise": ["noise", "grain", "snr", "speckle", "static"],
    "partial_masking": ["mask", "masked", "black rectangle", "black bar"],
    "data_loss": ["data loss", "dropout", "fragment", "pixelation", "corrupt", "missing data", "lost"],
}


def _degradation_score(expected: Any, predicted: Any) -> float:
    if _is_invalid_semantic_text(predicted):
        return 0.0
    exact_or_alias = _canonical_label_score(expected, predicted, _DEGRADATION_ALIASES)
    if exact_or_alias:
        return exact_or_alias
    predicted_hits = {
        label
        for label, aliases in _DEGRADATION_ALIASES.items()
        if _contains_any(predicted, aliases)
    }
    expected_text = str(expected)
    occlusion_family = {"occlusion", "partial_masking", "data_loss"}
    optical_family = {"dust_veil", "heat_shimmer", "high_glare", "blur", "noise", "small_target_long_distance"}
    if expected_text in occlusion_family and predicted_hits & occlusion_family:
        return 0.5
    if expected_text in optical_family and predicted_hits & optical_family:
        return 0.25
    return 0.0


def _usability_level(value: Any) -> int | None:
    normalized = _text(value)
    if _is_invalid_semantic_text(value):
        return None
    if _contains_any(value, ["unusable", "not usable", "invalid"]):
        return 0
    if _contains_any(value, ["low", "poor", "severe", "critical"]):
        return 0
    if _contains_any(value, ["moderate", "medium", "fair", "partial", "limited", "degraded", "marginal", "caution"]):
        return 1
    if _contains_any(value, ["high", "good", "usable", "clear"]):
        return 2
    return None


_CAUSE_TO_DEGRADATION = {
    "clean_observation": "clean_visual",
    "airborne_visibility_loss": "dust_veil",
    "thermal_distortion": "heat_shimmer",
    "optical_saturation": "high_glare",
    "motion_blur": "blur",
    "scale_limit": "small_target_long_distance",
    "occluded_view": "occlusion",
    "low_snr": "noise",
    "masked_region": "partial_masking",
    "sensor_dropout": "data_loss",
}


def _usability_cause_score(expected: Any, predicted: Any) -> float:
    if not isinstance(expected, dict) or not isinstance(predicted, dict):
        return 0.0
    expected_level = _usability_level(expected.get("usability"))
    predicted_level = _usability_level(predicted.get("usability"))
    if expected_level is None or predicted_level is None:
        usability_score = 0.0
    elif expected_level == predicted_level:
        usability_score = 1.0
    elif abs(expected_level - predicted_level) == 1:
        usability_score = 0.5
    else:
        usability_score = 0.0

    expected_degradation = _CAUSE_TO_DEGRADATION.get(str(expected.get("cause_label")), str(expected.get("cause_label")))
    cause_score = _degradation_score(expected_degradation, predicted.get("cause_label"))
    return (usability_score + cause_score) / 2.0


_CONSTRAINT_ALIASES = {
    "respect_base_mission_constraints": ["respect mission constraints", "keep mission constraints", "keep mission qawaid", "keep constr", "compliance", "protocol"],
    "slow_navigation": ["slow", "proceed slowly", "cautious", "conservative"],
    "maintain_lane_context": ["lane", "boulevard", "traffic", "road", "route"],
    "avoid_nfz": ["avoid nfz", "no fly", "restricted zone", "restricted zn", "avoid restricted", "nfz edge"],
    "maintain_corridor_clearance": ["corridor", "clearance", "standoff", "distance"],
    "respect_boundary_zone": ["boundary", "restricted", "edge", "buffer"],
    "avoid_restricted_proximity": ["avoid restricted", "restricted proximity", "standoff", "keep away"],
    "maintain_standoff_distance": ["standoff", "distance", "buffer"],
    "respect_critical_facility_buffer": ["critical facility", "facility buffer", "government", "infrastructure"],
    "respect_no_approach_sector": ["no approach", "restricted sector", "approach sector"],
    "maintain_visibility_margin": ["visibility", "margin", "low visibility", "weather"],
    "preserve_corridor_safety": ["corridor safety", "safety corridor", "safe corridor"],
    "respect_sensitive_site_policy": ["sensitive", "worship", "privacy", "memorial", "policy"],
    "prefer_conservative_operation": ["conservative", "cautious", "safe operation"],
    "maintain_sensitive_site_buffer": ["sensitive site", "buffer", "respect", "worship"],
    "avoid_close_shot": ["close shot", "close filming", "privacy", "too close"],
    "maintain_quiet_altitude": ["quiet", "altitude", "noise"],
    "avoid_overflight_on_crowd": ["crowd", "overflight", "gathering"],
    "maintain_safe_buffer": ["safe buffer", "buffer", "distance"],
    "maintain_privacy_buffer": ["privacy", "buffer"],
    "avoid_close_filming": ["close filming", "media", "filming"],
    "maintain_memorial_buffer": ["memorial", "cemetery", "buffer"],
    "respect_media_policy": ["media", "policy", "permit"],
    "confirm_capture_window": ["capture window", "confirm capture", "window"],
    "confirm_media_permit": ["media permit", "permit"],
    "respect_crowd_adjacent_buffer": ["crowd", "adjacent buffer", "public buffer"],
    "respect_dispatch_priority": ["dispatch", "priority"],
    "confirm_target_zone_before_descent": ["target zone", "before descent", "confirm descent"],
}


def _constraint_aliases(canonical: str) -> list[str]:
    return _CONSTRAINT_ALIASES.get(canonical, [canonical.replace("_", " ")])


def _constraint_f1_score(expected: Any, predicted: Any) -> float:
    expected_items = [item for item in expected if isinstance(item, str)] if isinstance(expected, list) else []
    predicted_items = [
        item
        for item in predicted
        if isinstance(item, str) and not _is_invalid_semantic_text(item)
    ] if isinstance(predicted, list) else []
    if not expected_items and not predicted_items:
        return 1.0
    if not expected_items or not predicted_items:
        return 0.0
    hits = 0
    matched_predicted_indices: set[int] = set()
    for expected_item in expected_items:
        aliases = _constraint_aliases(expected_item)
        for predicted_index, predicted_item in enumerate(predicted_items):
            if predicted_index in matched_predicted_indices:
                continue
            if _contains_any(predicted_item, aliases):
                matched_predicted_indices.add(predicted_index)
                hits += 1
                break
    precision = hits / len(predicted_items)
    recall = hits / len(expected_items)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


_GOAL_ALIASES = {
    "maintain_boulevard_patrol_coverage": ["boulevard", "patrol", "city", "urban", "road"],
    "monitor_airport_perimeter_safely": ["airport", "perimeter", "monitor", "safely"],
    "inspect_coastal_or_portside_assets": ["coastal", "port", "shore", "water", "asset", "inspect"],
    "preserve_low_visibility_corridor_safety": ["low visibility", "corridor", "safety", "weather", "haze"],
    "capture_compliant_landmark_view": ["landmark", "capture", "media", "permit", "compliant"],
    "complete_dispatch_or_docking_sequence": ["dispatch", "dock", "docking", "landing", "sequence"],
    "maintain_sensitive_site_buffer": ["sensitive", "site", "buffer", "worship", "privacy", "memorial"],
}


def _goal_score(expected: Any, predicted: Any) -> float:
    if _is_invalid_semantic_text(predicted):
        return 0.0
    exact_or_alias = _canonical_label_score(expected, predicted, _GOAL_ALIASES)
    if exact_or_alias:
        return exact_or_alias
    aliases = _GOAL_ALIASES.get(str(expected), [str(expected).replace("_", " ")])
    hits = sum(1 for alias in aliases if _contains_any(predicted, [alias]))
    if hits == 0:
        return 0.0
    return min(1.0, hits / max(1, min(2, len(aliases))))


def _collaboration_score(expected: Any, predicted: Any) -> float:
    if not isinstance(expected, dict) or not isinstance(predicted, dict):
        return 0.0
    required_score = 1.0 if bool(expected.get("required")) == bool(predicted.get("required")) else 0.0
    if expected.get("request_type") == predicted.get("request_type"):
        request_score = 1.0
    elif bool(expected.get("required")) and bool(predicted.get("required")):
        request_score = 0.5
    else:
        request_score = 0.0
    if expected.get("trusted_modality") == predicted.get("trusted_modality"):
        modality_score = 1.0
    elif predicted.get("trusted_modality") in {"multi", "fused"}:
        modality_score = 0.5
    else:
        modality_score = 0.0
    return (required_score + request_score + modality_score) / 3.0


def _score_task(task_id: str, expected: Any, predicted: Any, row: dict[str, Any]) -> float:
    if task_id == "T1":
        return 1.0 if row["protocol_context_included"] else 0.0
    if task_id == "T2":
        return _risk_zone_score(expected, predicted)
    if task_id == "T3":
        return _viewpoint_score(expected, predicted)
    if task_id == "T4":
        return _degradation_score(expected, predicted)
    if task_id == "T6":
        return _usability_cause_score(expected, predicted)
    if task_id == "T8":
        return _constraint_f1_score(expected, predicted)
    if task_id == "T9":
        return _goal_score(expected, predicted)
    if task_id == "T10":
        if row["unsafe_action"]:
            return 0.0
        return 1.0 if row["safe_action"] else 0.0
    if task_id == "T12":
        return _collaboration_score(expected, predicted)
    return 1.0 if row["task_matches"][task_id] else 0.0


def _collect_semantic_rows(samples: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    semantic_rows: list[dict[str, Any]] = []
    for row in _collect_evaluation_rows(samples, predictions):
        expected_tasks = row["sample"]["paper_task_gt"]
        predicted_tasks = row["predicted_tasks"]
        task_scores = {
            task_id: _score_task(task_id, expected_tasks[task_id], predicted_tasks[task_id], row)
            for task_id in TASK_ID_TO_NAME
        }
        semantic_rows.append({**row, "semantic_task_scores": task_scores})
    return semantic_rows


def _aggregate_semantic_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = len(rows)
    task_metrics: dict[str, float] = {"evaluated_samples": float(evaluated)}
    for task_id in TASK_ID_TO_NAME:
        task_metrics[f"{task_id}_score"] = _safe_divide(
            sum(row["semantic_task_scores"][task_id] for row in rows),
            evaluated,
        )
    task_metrics["semantic_avg_task_score"] = _safe_divide(
        sum(task_metrics[f"{task_id}_score"] for task_id in TASK_ID_TO_NAME),
        len(TASK_ID_TO_NAME),
    )
    task_metrics["semantic_protocol_context_score"] = _safe_divide(
        task_metrics["T1_score"] + task_metrics["T2_score"] + task_metrics["T3_score"],
        3,
    )
    task_metrics["semantic_degradation_score"] = _safe_divide(
        task_metrics["T4_score"] + task_metrics["T5_score"] + task_metrics["T6_score"],
        3,
    )
    task_metrics["semantic_intent_score"] = _safe_divide(
        task_metrics["T7_score"] + task_metrics["T8_score"] + task_metrics["T9_score"],
        3,
    )
    task_metrics["semantic_action_collaboration_score"] = _safe_divide(
        task_metrics["T10_score"] + task_metrics["T11_score"] + task_metrics["T12_score"],
        3,
    )
    task_metrics["semantic_protocol_decision_score"] = _safe_divide(
        task_metrics["T8_score"]
        + task_metrics["T9_score"]
        + task_metrics["T10_score"]
        + task_metrics["T11_score"]
        + task_metrics["T12_score"],
        5,
    )
    return {
        "count": evaluated,
        "semantic_task_metrics": task_metrics,
        "semantic_summary_metrics": _summary_from_task_metrics(task_metrics),
    }


def _summary_from_task_metrics(task_metrics: dict[str, float]) -> dict[str, float]:
    return {
        "Semantic Avg Task Score": task_metrics["semantic_avg_task_score"],
        "Semantic Protocol Context Score": task_metrics["semantic_protocol_context_score"],
        "Semantic Degradation Score": task_metrics["semantic_degradation_score"],
        "Semantic Intent Score": task_metrics["semantic_intent_score"],
        "Semantic Action Collaboration Score": task_metrics["semantic_action_collaboration_score"],
        "Semantic Protocol Decision Score": task_metrics["semantic_protocol_decision_score"],
        "Semantic Safe Action Score": task_metrics["T10_score"],
    }


def compute_semantic_task_metrics(samples: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, float]:
    return _aggregate_semantic_rows(_collect_semantic_rows(samples, predictions))["semantic_task_metrics"]


def compute_semantic_summary_metrics(samples: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, float]:
    return _aggregate_semantic_rows(_collect_semantic_rows(samples, predictions))["semantic_summary_metrics"]


def compute_semantic_grouped_metrics(samples: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, float]:
    rows = _collect_semantic_rows(samples, predictions)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "by_protocol_context": defaultdict(list),
        "by_application_theme": defaultdict(list),
        "by_degradation_profile": defaultdict(list),
        "by_language_profile": defaultdict(list),
    }
    for row in rows:
        sample = row["sample"]
        grouped["by_protocol_context"][sample.get("protocol_context", "EmergUAV Protocol")].append(row)
        grouped["by_application_theme"][sample.get("application_theme", "Emergency Operations")].append(row)
        grouped["by_degradation_profile"][row["degradation_profile"]].append(row)
        grouped["by_language_profile"][row["language_profile"]].append(row)
    return {
        group_name: {key: _aggregate_semantic_rows(value) for key, value in groups.items()}
        for group_name, groups in grouped.items()
    }


def compute_semantic_report(samples: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _collect_semantic_rows(samples, predictions)
    aggregate = _aggregate_semantic_rows(rows)
    strict_task_metrics = _strict_task_metrics_from_rows(rows)
    return {
        "metric_profile": METRIC_PROFILE,
        "semantic_task_metrics": aggregate["semantic_task_metrics"],
        "semantic_summary_metrics": aggregate["semantic_summary_metrics"],
        "strict_diagnostics": strict_task_metrics,
        "semantic_grouped_results": compute_semantic_grouped_metrics(samples, predictions),
    }


def _strict_task_metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    evaluated = len(rows)
    diagnostics: dict[str, float] = {"evaluated_samples": float(evaluated)}
    for task_id in TASK_ID_TO_NAME:
        diagnostics[f"{task_id}_strict_accuracy"] = _safe_divide(
            sum(1 for row in rows if row["task_matches"][task_id]),
            evaluated,
        )
    diagnostics["protocol_context_exact_accuracy"] = _safe_divide(
        sum(1 for row in rows if row["protocol_context_exact"]),
        evaluated,
    )
    diagnostics["protocol_context_inclusion_accuracy"] = _safe_divide(
        sum(1 for row in rows if row["protocol_context_included"]),
        evaluated,
    )
    diagnostics["safe_action_accuracy"] = _safe_divide(sum(1 for row in rows if row["safe_action"]), evaluated)
    diagnostics["unsafe_action_rate"] = _safe_divide(sum(1 for row in rows if row["unsafe_action"]), evaluated)
    return diagnostics
