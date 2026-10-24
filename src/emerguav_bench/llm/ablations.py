from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Literal

from emerguav_bench.llm.prompts import (
    SYSTEM_PROMPT,
    build_default_prompt_messages,
    render_output_contract,
)
from emerguav_bench.llm.types import ensure_llm_input_record

AblationCondition = Literal["baseline", "text_only", "image_only", "primary_rgb_only"]
ABLATION_CONDITIONS: tuple[AblationCondition, ...] = (
    "baseline",
    "text_only",
    "image_only",
    "primary_rgb_only",
)
# Preserve the serialized version tag used by existing exported manifests.
ABLATION_CONTRACT_VERSION = "tnnls_multimodal_v1"

_FORBIDDEN_MODEL_VISIBLE_FIELD_NAMES = (
    "degradation_labels",
    "degradation_severity",
    "observation_severity_code",
    "observation_protocol_gt",
    "intent_extraction_gt",
    "action_token_gt",
    "acceptable_safe_alternatives",
    "forbidden_actions",
    "risk_level_gt",
    "collaboration_gt",
    "paper_task_gt",
)


def source_anchor_id(sample_id: str) -> str:
    parts = str(sample_id).split("::")
    if len(parts) < 2:
        raise ValueError(f"sample_id does not contain a source anchor: {sample_id}")
    return "::".join(parts[:2])


def stage1_strata(record: dict[str, Any]) -> set[tuple[str, str]]:
    sample = record.get("metadata", {}).get("sample_record")
    if not isinstance(sample, dict):
        sample = record
    collaboration = sample.get("collaboration_gt") or {}
    action = str(sample.get("action_token_gt", "unknown"))
    degradation_labels = sample.get("degradation_labels") or ["none"]
    return {
        ("sequence", str(sample.get("sequence_id", "unknown"))),
        ("protocol", str(sample.get("protocol_context", "unknown"))),
        *(("degradation", str(label)) for label in degradation_labels),
        ("language_level", str(sample.get("language_entropy_level", "unknown"))),
        ("action", action),
        ("risk", str(sample.get("risk_level_gt", "unknown"))),
        ("collaboration_required", str(bool(collaboration.get("required")))),
        ("collaboration_type", str(collaboration.get("request_type", "none"))),
        ("abstention", str(action in {"Hover", "Reobserve"})),
    }


def select_stratified_stage1_records(
    records: Iterable[dict[str, Any]],
    *,
    minimum_per_stratum: int = 2,
) -> list[dict[str, Any]]:
    if minimum_per_stratum <= 0:
        raise ValueError("minimum_per_stratum must be positive")
    remaining = sorted(records, key=lambda item: str(item["sample_id"]))
    if not remaining:
        return []

    universe = set().union(*(stage1_strata(record) for record in remaining))
    required = {stratum: minimum_per_stratum for stratum in universe}
    selected: list[dict[str, Any]] = []
    while any(count > 0 for count in required.values()):
        best = min(
            remaining,
            key=lambda record: (
                -sum(1 for stratum in stage1_strata(record) if required[stratum] > 0),
                str(record["sample_id"]),
            ),
        )
        gain = {stratum for stratum in stage1_strata(best) if required[stratum] > 0}
        if not gain:
            missing = sorted(stratum for stratum, count in required.items() if count > 0)
            raise ValueError(
                "Cannot satisfy stratified Stage 1 coverage with distinct source anchors; "
                f"missing strata: {missing}"
            )
        selected.append(copy.deepcopy(best))
        for stratum in stage1_strata(best):
            required[stratum] = max(0, required[stratum] - 1)
        selected_anchor = source_anchor_id(str(best["sample_id"]))
        remaining = [
            record
            for record in remaining
            if source_anchor_id(str(record["sample_id"])) != selected_anchor
        ]
        if not remaining and any(count > 0 for count in required.values()):
            missing = sorted(stratum for stratum, count in required.items() if count > 0)
            raise ValueError(
                "Cannot satisfy stratified Stage 1 coverage with distinct source anchors; "
                f"missing strata: {missing}"
            )
    return selected


def summarize_stage1_coverage(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in rows:
        for group, value in stage1_strata(record):
            counts[group][value] += 1
    return {
        "record_count": len(rows),
        "source_anchor_count": len({source_anchor_id(str(record["sample_id"])) for record in rows}),
        "strata": {
            group: dict(sorted(value_counts.items()))
            for group, value_counts in sorted(counts.items())
        },
    }


def _ordered_image_lines(record: dict[str, Any]) -> list[str]:
    return [
        f"- image_{index}: role={image_input['role']}"
        for index, image_input in enumerate(record.get("image_inputs", []), start=1)
    ]


def _text_only_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    metadata = record["metadata"]
    system = SYSTEM_PROMPT.replace(
        "Use the provided images and mission context",
        "No images are supplied in this text-only ablation. Use the provided mission context",
    ).replace(
        "Use images for observation quality, observation degradation, viewpoint, and safety cues; ",
        "Infer only what the supplied text supports and report visual uncertainty conservatively; ",
    )
    user = "\n".join(
        [
            "Mission context:",
            f"- protocol_context: {metadata.get('protocol_context', 'unknown')}",
            f"- application_theme: {metadata.get('application_theme', 'unknown')}",
            f"- policy_family: {metadata.get('policy_family', 'unknown')}",
            f"- language_profile: {metadata.get('language_entropy_level', 'unknown')}",
            f"- primary_instruction: {metadata.get('instruction_text', '')}",
            "Protocol / policy context:",
            "- Treat protocol_context, application_theme, policy_family, and mission constraints as provided protocol context.",
            "- Use provided protocol context for policy semantics.",
            "Observation use:",
            "- No image input is available in this text-only ablation.",
            "- Do not invent visual degradation, viewpoint, or immediate visual safety evidence.",
            render_output_contract(),
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _image_only_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are a UAV image-only decision model. Use only the supplied RGB images. "
        "Mission protocol, policy family, operator instruction, language condition, and rule constraints are unavailable. "
        "Return only the requested JSON object and mark unsupported textual or policy fields as unknown."
    )
    user = "\n".join(
        [
            "Image-only ablation:",
            "- Use pixels for observation quality, degradation, viewpoint usability, and immediate visual safety cues.",
            "- Do not infer protocol or policy semantics from image handles, ordering, paths, frame ids, or sequence ids.",
            "Ordered image inputs:",
            *_ordered_image_lines(record),
            render_output_contract(),
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _prepare_record(record: dict[str, Any], *, condition: AblationCondition) -> dict[str, Any]:
    transformed = copy.deepcopy(record)
    transformed["sft_targets"] = []
    transformed["preference_pairs"] = []
    metadata = transformed.setdefault("metadata", {})
    metadata["ablation_condition"] = condition
    metadata["ablation_contract_version"] = ABLATION_CONTRACT_VERSION
    metadata["source_anchor_id"] = source_anchor_id(transformed["sample_id"])

    if condition == "baseline":
        llm_record = ensure_llm_input_record(transformed)
        transformed["prompt_messages"] = build_default_prompt_messages(llm_record)
    elif condition == "text_only":
        transformed["image_inputs"] = []
        transformed["prompt_messages"] = _text_only_messages(transformed)
    elif condition == "image_only":
        for key in (
            "protocol_context",
            "application_theme",
            "policy_family",
            "language_entropy_level",
            "language_entropy_labels",
            "observation_severity_code",
            "instruction_source",
            "instruction_text",
            "sample_record",
        ):
            metadata.pop(key, None)
        transformed["prompt_messages"] = _image_only_messages(transformed)
    elif condition == "primary_rgb_only":
        primary_images = [item for item in transformed.get("image_inputs", []) if item.get("role") == "primary_frame"]
        if len(primary_images) != 1:
            raise ValueError(f"Expected one primary frame for {transformed['sample_id']}, found {len(primary_images)}")
        transformed["image_inputs"] = primary_images
        llm_record = ensure_llm_input_record(transformed)
        transformed["prompt_messages"] = build_default_prompt_messages(llm_record)
    else:
        raise ValueError(f"Unsupported ablation condition: {condition}")
    return transformed


def build_ablation_manifest(
    records: Iterable[dict[str, Any]],
    *,
    condition: AblationCondition,
) -> list[dict[str, Any]]:
    if condition not in ABLATION_CONDITIONS:
        raise ValueError(f"Unsupported condition {condition!r}; expected one of {ABLATION_CONDITIONS}")
    return [_prepare_record(record, condition=condition) for record in records]


def _visual_condition(record: dict[str, Any]) -> tuple[str, ...]:
    sample = record.get("metadata", {}).get("sample_record", {})
    labels = sample.get("degradation_labels") or []
    return tuple(str(label) for label in labels)


def _pair_priority(first: dict[str, Any], second: dict[str, Any]) -> tuple[int, int, int, str, str]:
    first_sample = first.get("metadata", {}).get("sample_record", {})
    second_sample = second.get("metadata", {}).get("sample_record", {})
    action_change = int(first_sample.get("action_token_gt") != second_sample.get("action_token_gt"))
    constraint_change = int(set(first_sample.get("mission_constraints", [])) != set(second_sample.get("mission_constraints", [])))
    forbidden_change = int(set(first_sample.get("forbidden_actions", [])) != set(second_sample.get("forbidden_actions", [])))
    return (
        -action_change,
        -constraint_change,
        -forbidden_change,
        str(first["sample_id"]),
        str(second["sample_id"]),
    )


def build_protocol_counterfactual_manifest(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(source_anchor_id(record["sample_id"]), _visual_condition(record))].append(record)

    candidates_by_anchor: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for (anchor, _), group in grouped.items():
        for first, second in combinations(sorted(group, key=lambda item: item["sample_id"]), 2):
            first_protocol = first.get("metadata", {}).get("protocol_context")
            second_protocol = second.get("metadata", {}).get("protocol_context")
            if first_protocol and second_protocol and first_protocol != second_protocol:
                candidates_by_anchor[anchor].append((first, second))

    manifest: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for anchor in sorted(candidates_by_anchor):
        first, second = min(candidates_by_anchor[anchor], key=lambda pair: _pair_priority(*pair))
        visual_inputs = copy.deepcopy(first.get("image_inputs", []))
        first_row = _prepare_record(first, condition="baseline")
        second_row = _prepare_record(second, condition="baseline")
        first_row["image_inputs"] = copy.deepcopy(visual_inputs)
        second_row["image_inputs"] = copy.deepcopy(visual_inputs)
        first_row["metadata"]["ablation_condition"] = "protocol_counterfactual"
        second_row["metadata"]["ablation_condition"] = "protocol_counterfactual"
        first_row["metadata"]["counterfactual_pair_role"] = "visual_source"
        second_row["metadata"]["counterfactual_pair_role"] = "protocol_counterfactual"
        first_row["prompt_messages"] = build_default_prompt_messages(ensure_llm_input_record(first_row))
        second_row["prompt_messages"] = build_default_prompt_messages(ensure_llm_input_record(second_row))
        manifest.extend([first_row, second_row])
        pairs.append(
            {
                "pair_id": f"protocol_cf::{anchor}",
                "source_anchor_id": anchor,
                "visual_condition": list(_visual_condition(first)),
                "visual_source_sample_id": first_row["sample_id"],
                "counterfactual_sample_id": second_row["sample_id"],
                "visual_source_protocol": first_row["metadata"].get("protocol_context"),
                "counterfactual_protocol": second_row["metadata"].get("protocol_context"),
                "visual_source_action": first_row.get("metadata", {}).get("sample_record", {}).get("action_token_gt"),
                "counterfactual_action": second_row.get("metadata", {}).get("sample_record", {}).get("action_token_gt"),
            }
        )
    return manifest, pairs


def prompt_leakage_audit(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    checked = 0
    for record in records:
        checked += 1
        prompt = "\n".join(str(message.get("content", "")) for message in record.get("prompt_messages", []))
        lowered = prompt.casefold()
        hits = [field for field in _FORBIDDEN_MODEL_VISIBLE_FIELD_NAMES if field.casefold() in lowered]
        if hits:
            violations.append({"sample_id": record.get("sample_id"), "field_name_hits": hits})
    return {
        "contract_version": ABLATION_CONTRACT_VERSION,
        "checked_record_count": checked,
        "forbidden_field_names": list(_FORBIDDEN_MODEL_VISIBLE_FIELD_NAMES),
        "violation_count": len(violations),
        "violations": violations,
        "passed": not violations,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
