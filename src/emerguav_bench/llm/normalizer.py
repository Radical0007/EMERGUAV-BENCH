from __future__ import annotations

import json
import re
from typing import Any

from emerguav_bench.config.vocab import (
    ABSTENTION_DECISIONS,
    ACTION_TOKENS,
    COLLABORATION_REQUEST_TYPES,
    RISK_LEVELS,
    TRUSTED_MODALITIES,
)

JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def _extract_balanced_json_at(candidate: str, start: int) -> str | None:
    depth = 0
    for index in range(start, len(candidate)):
        char = candidate[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : index + 1]
    return None


def iter_json_candidates(raw_text: str) -> list[str]:
    candidates: list[str] = []
    fence_match = JSON_FENCE_PATTERN.search(raw_text)
    if fence_match:
        candidates.append(fence_match.group(1))
    for match in re.finditer(r"\{", raw_text):
        candidate = _extract_balanced_json_at(raw_text, match.start())
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _truncate(text: str, *, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated, total {len(text)} chars]"


def extract_json_candidate(raw_text: str) -> str:
    candidates = iter_json_candidates(raw_text)
    if not candidates:
        snippet = _truncate(raw_text)
        raise ValueError(f"Model response does not contain a JSON object. Raw response: {snippet}")
    return candidates[0]


def _clean_text_for_json(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"\bNaN\b", "null", text)
    text = re.sub(r"\b-?Infinity\b", "null", text)
    text = _sanitize_json_text(text)
    return text


def _attempt_repair_truncated_json(text: str) -> dict[str, Any] | None:
    text = _clean_text_for_json(text)

    stack: list[str] = []
    in_string = False
    escape = False
    for char in text:
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append(char)
        elif char == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif char == "]":
            if stack and stack[-1] == "[":
                stack.pop()

    if not stack:
        return None

    stripped = text.rstrip(",\n\r\t ")
    if in_string:
        if escape:
            stripped = stripped[:-1]
        stripped += '"'
    base_closers = "".join("}" if opener == "{" else "]" for opener in reversed(stack))

    for extra in range(6):
        closers = base_closers + "}" * extra
        repaired = stripped + closers
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            continue
    return None


def _attempt_repair_json_syntax(text: str, *, max_attempts: int = 10) -> dict[str, Any] | None:
    for _ in range(max_attempts):
        text = text.rstrip(",\n\r\t ")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            msg = exc.msg
            pos = exc.pos
            near_end = pos >= len(text) - 2
            if "Expecting ',' delimiter" in msg:
                text = text[:pos] + ("}" if near_end else ",") + text[pos:]
            elif "Expecting ':' delimiter" in msg:
                text = text[:pos] + ":" + text[pos:]
            elif "Expecting value" in msg:
                text = text[:pos] + "null" + text[pos:]
            elif "Expecting property name" in msg:
                text = text[:pos] + "}" + text[pos:]
            elif "Unterminated string" in msg:
                text += '"'
            elif "Invalid \\escape" in msg:
                text = _sanitize_json_text(text)
            else:
                return None
    return None


_INVALID_UNICODE_ESCAPE = re.compile(r"(?<!\\)\\u(?![\da-fA-F]{4})")
_INVALID_ESCAPE_CHAR = re.compile(r'(?<!\\)\\(?!["\\/bfnrtu])')


def _sanitize_json_text(text: str) -> str:
    text = _INVALID_UNICODE_ESCAPE.sub(r"\\\\u", text)
    text = _INVALID_ESCAPE_CHAR.sub(r"\\\\", text)
    return text


_ANCHOR_TRUSTED_MODALITY: dict[str, list[str]] = {
    "rgb":    ["rgb", "camera", "visual", "vision", "image", "photo", "color", "appearance"],
    "lidar":  ["lidar", "laser", "pointcloud", "point_cloud", "depth", "range", "scan", "beam"],
    "pose":   ["pose", "gps", "position", "gnss", "localization", "imu", "coordinate", "location", "odometry"],
    "multi":  ["multi", "fusion", "fused", "combined", "both", "all", "hybrid", "ensemble"],
}

_ANCHOR_RISK_LEVEL: dict[str, list[str]] = {
    "low":      ["low", "safe", "minor", "negligible", "minimal", "clear", "benign", "normal"],
    "medium":   ["medium", "moderate", "mid", "average", "some", "partial", "fair"],
    "high":     ["high", "severe", "serious", "significant", "substantial", "major", "heavy", "elevated"],
    "critical": ["critical", "extreme", "catastrophic", "imminent", "danger", "fatal", "lethal"],
}

_ANCHOR_ACTION_TOKEN: dict[str, list[str]] = {
    "Hover":              ["hover", "hold", "stay", "wait", "maintain", "pause", "remain", "stop"],
    "Proceed_Slow":       ["proceed", "move", "advance", "forward", "continue", "go"],
    "Ascend_Cautiously":  ["ascend", "climb", "rise", "up", "elevate", "increase altitude"],
    "Descend_Cautiously": ["descend", "lower", "down", "decrease altitude", "dive", "sink"],
    "Turn_Left_Slow":     ["turn left", "left turn", "rotate left", "bank left", "yaw left"],
    "Turn_Right_Slow":    ["turn right", "right turn", "rotate right", "bank right", "yaw right"],
    "Reobserve":          ["reobserve", "re-observe", "re_observe", "observe again", "rescan", "reacquire", "re-scan"],
    "Ask_Other_UAV":      ["ask", "request", "query", "other uav", "another drone", "coordinate", "consult", "delegate"],
    "Abort_Mission":      ["abort", "abandon", "terminate", "cancel", "return home", "rtl", "rtb", "return to launch"],
    "Hover_and_Alert":    ["alert", "warn", "notify", "broadcast", "signal", "hover and alert"],
}

_ANCHOR_ABSTENTION: dict[str, list[str]] = {
    "continue_execution": ["continue", "proceed", "execut", "go ahead", "resume", "carry on"],
    "reobserve":          ["reobserve", "re-observe", "re_observe", "again", "rescan", "reacquire", "re-scan", "second look"],
    "hold_position":      ["hold", "wait", "stay", "pause", "maintain", "stop", "remain", "freeze"],
    "abort_mission":      ["abort", "abandon", "terminate", "cancel", "return", "rtl", "rtb"],
}

_ANCHOR_COLLABORATION: dict[str, list[str]] = {
    "none":                  ["none", "no", "not required", "unnecessary", "n/a"],
    "reobserve_context":     ["reobserve", "re-observe", "re_observe", "context", "scene", "surrounding"],
    "traffic_check":         ["traffic", "air traffic", "collision", "deconfliction"],
    "policy_clarification":  ["policy", "regulation", "rule", "legal", "compliance", "law", "ordinance"],
    "crowd_assessment":      ["crowd", "people", "public", "gathering", "assembly", "population", "pedestrian", "bystander"],
    "docking_confirmation":  ["dock", "land", "approach", "pad", "touchdown", "helipad", "vertiport"],
}


def _resolve_controlled(
    raw_value: Any,
    canonical: set[str],
    anchor_map: dict[str, list[str]],
    default: str,
) -> tuple[str, str]:
    if isinstance(raw_value, str) and raw_value in canonical:
        return raw_value, "exact"

    if isinstance(raw_value, str):
        low = raw_value.lower().strip()
        for c in canonical:
            if c.lower() == low:
                return c, "normalized"

        best_match: str | None = None
        best_len = 0
        for canonical_value, keywords in anchor_map.items():
            for kw in keywords:
                if kw in low and len(kw) > best_len:
                    best_match = canonical_value
                    best_len = len(kw)
        if best_match is not None:
            return best_match, "normalized"

    return default, "default"


_SECTION_KEYS = frozenset({"Observation_Protocol_Assessment", "Intent_Extraction", "Action_Output", "Rationale_Short"})


def _looks_like_model_output(payload: dict[str, Any]) -> bool:
    return bool(_SECTION_KEYS & set(payload.keys()))


def _lenient_normalize(payload: dict[str, Any], *, is_truncated: bool = False) -> dict[str, Any]:
    status: dict[str, str] = {}
    result: dict[str, Any] = {}

    protocol_assessment = payload.get("Observation_Protocol_Assessment")
    if isinstance(protocol_assessment, dict):
        assessment: dict[str, Any] = {}
        for field in (
            "protocol_context_profile", "primary_degradation", "usability",
            "risk_zone_label", "viewpoint_summary", "cause_label",
        ):
            val = protocol_assessment.get(field)
            if isinstance(val, str) and val.strip():
                assessment[field] = val
                status[f"Observation_Protocol_Assessment.{field}"] = "exact"
            else:
                assessment[field] = ""
                status[f"Observation_Protocol_Assessment.{field}"] = "default"

        tm, tm_s = _resolve_controlled(
            protocol_assessment.get("trusted_modality"), TRUSTED_MODALITIES, _ANCHOR_TRUSTED_MODALITY, "multi",
        )
        assessment["trusted_modality"] = tm
        status["Observation_Protocol_Assessment.trusted_modality"] = tm_s

        rl, rl_s = _resolve_controlled(
            protocol_assessment.get("risk_level"), RISK_LEVELS, _ANCHOR_RISK_LEVEL, "medium",
        )
        assessment["risk_level"] = rl
        status["Observation_Protocol_Assessment.risk_level"] = rl_s
        result["Observation_Protocol_Assessment"] = assessment
    else:
        result["Observation_Protocol_Assessment"] = {
            "protocol_context_profile": "",
            "primary_degradation": "",
            "usability": "",
            "trusted_modality": "multi",
            "risk_level": "medium",
            "risk_zone_label": "",
            "viewpoint_summary": "",
            "cause_label": "",
        }
        for k in result["Observation_Protocol_Assessment"]:
            status[f"Observation_Protocol_Assessment.{k}"] = "default"

    intent = payload.get("Intent_Extraction")
    if isinstance(intent, dict):
        ie: dict[str, Any] = {}

        val = intent.get("cleaned_instruction")
        ie["cleaned_instruction"] = str(val) if isinstance(val, str) and val else ""
        status["Intent_Extraction.cleaned_instruction"] = "exact" if isinstance(val, str) and val else "default"

        target = intent.get("target_object")
        if target is None or (isinstance(target, str) and target.strip()):
            ie["target_object"] = target
            status["Intent_Extraction.target_object"] = "exact"
        else:
            ie["target_object"] = None
            status["Intent_Extraction.target_object"] = "default"

        constraints = intent.get("constraints")
        if isinstance(constraints, list) and all(isinstance(item, str) for item in constraints):
            ie["constraints"] = constraints
            status["Intent_Extraction.constraints"] = "exact"
        else:
            ie["constraints"] = []
            status["Intent_Extraction.constraints"] = "default"

        goal = intent.get("target_goal")
        ie["target_goal"] = str(goal) if isinstance(goal, str) and goal else "unknown"
        status["Intent_Extraction.target_goal"] = "exact" if isinstance(goal, str) and goal else "default"

        result["Intent_Extraction"] = ie
    else:
        result["Intent_Extraction"] = {
            "cleaned_instruction": "",
            "target_object": None,
            "constraints": [],
            "target_goal": "unknown",
        }
        for k in result["Intent_Extraction"]:
            status[f"Intent_Extraction.{k}"] = "default"

    action = payload.get("Action_Output")
    if isinstance(action, dict):
        ao: dict[str, Any] = {}

        at, at_s = _resolve_controlled(
            action.get("action_token"), ACTION_TOKENS, _ANCHOR_ACTION_TOKEN, "Hover",
        )
        ao["action_token"] = at
        status["Action_Output.action_token"] = at_s

        confidence = action.get("confidence")
        if isinstance(confidence, (int, float)) and 0.0 <= float(confidence) <= 1.0:
            ao["confidence"] = float(confidence)
            status["Action_Output.confidence"] = "exact"
        else:
            ao["confidence"] = 0.5
            status["Action_Output.confidence"] = "default"

        ad, ad_s = _resolve_controlled(
            action.get("abstention_decision"), ABSTENTION_DECISIONS, _ANCHOR_ABSTENTION, "hold_position",
        )
        ao["abstention_decision"] = ad
        status["Action_Output.abstention_decision"] = ad_s

        cr = action.get("collaboration_request")
        if isinstance(cr, dict):
            crd: dict[str, Any] = {}
            required = cr.get("required")
            crd["required"] = bool(required) if isinstance(required, bool) else False
            status["Action_Output.collaboration_request.required"] = "exact" if isinstance(required, bool) else "default"

            rt, rt_s = _resolve_controlled(
                cr.get("request_type"), COLLABORATION_REQUEST_TYPES, _ANCHOR_COLLABORATION, "none",
            )
            crd["request_type"] = rt
            status["Action_Output.collaboration_request.request_type"] = rt_s

            ctm, ctm_s = _resolve_controlled(
                cr.get("trusted_modality"), TRUSTED_MODALITIES, _ANCHOR_TRUSTED_MODALITY, "multi",
            )
            crd["trusted_modality"] = ctm
            status["Action_Output.collaboration_request.trusted_modality"] = ctm_s

            ao["collaboration_request"] = crd
        else:
            ao["collaboration_request"] = {
                "required": False, "request_type": "none", "trusted_modality": "multi",
            }
            status["Action_Output.collaboration_request.required"] = "default"
            status["Action_Output.collaboration_request.request_type"] = "default"
            status["Action_Output.collaboration_request.trusted_modality"] = "default"

        result["Action_Output"] = ao
    else:
        result["Action_Output"] = {
            "action_token": "Hover",
            "confidence": 0.5,
            "abstention_decision": "hold_position",
            "collaboration_request": {
                "required": False, "request_type": "none", "trusted_modality": "multi",
            },
        }
        for k in ("action_token", "confidence", "abstention_decision", "collaboration_request"):
            status[f"Action_Output.{k}"] = "default"

    rationale = payload.get("Rationale_Short")
    if isinstance(rationale, str) and rationale.strip():
        result["Rationale_Short"] = rationale
        status["Rationale_Short"] = "exact"
    else:
        result["Rationale_Short"] = ""
        status["Rationale_Short"] = "default"

    result["_truncated"] = is_truncated
    result["_field_status"] = status
    result["_normalized"] = is_truncated or any(s != "exact" for s in status.values())
    return result


def normalize_model_response(raw_response: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_response, dict):
        return _lenient_normalize(raw_response)

    candidates = iter_json_candidates(raw_response)
    if candidates:
        for candidate in reversed(candidates):
            try:
                payload = json.loads(_clean_text_for_json(candidate))
            except json.JSONDecodeError:
                continue
            if not _looks_like_model_output(payload):
                continue
            return _lenient_normalize(payload)

        for candidate in reversed(candidates):
            repaired = _attempt_repair_truncated_json(candidate)
            if repaired is not None and _looks_like_model_output(repaired):
                return _lenient_normalize(repaired, is_truncated=True)

    last_error = "no balanced JSON candidates"
    last_payload = None

    brace_positions = [m.start() for m in re.finditer(r"\{", raw_response)]
    for brace_pos in reversed(brace_positions):
        repaired = _attempt_repair_truncated_json(raw_response[brace_pos:])
        if repaired is not None and _looks_like_model_output(repaired):
            return _lenient_normalize(repaired, is_truncated=True)

    cleaned_full = _clean_text_for_json(raw_response)
    try:
        last_payload = json.loads(cleaned_full)
    except json.JSONDecodeError as exc:
        last_error = f"json.JSONDecodeError: {exc}"
    else:
        if _looks_like_model_output(last_payload):
            return _lenient_normalize(last_payload)
        last_error = "balanced JSON parsed but no section keys matched"

    if last_payload is None:
        repaired = _attempt_repair_json_syntax(cleaned_full)
        if repaired is not None and _looks_like_model_output(repaired):
            return _lenient_normalize(repaired)

    snippet = _truncate(raw_response)
    raise ValueError(
        f"No valid model output found in {len(candidates)} JSON candidate(s) "
        f"(repair also failed). Last-chance diagnostics: {last_error}. "
        f"Raw response: {snippet}"
    )
