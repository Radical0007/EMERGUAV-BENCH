from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from emerguav_bench.config.vocab import ACTION_TOKENS, derive_abstention_decision
from emerguav_bench.llm.prompts import build_default_prompt_messages, render_user_prompt, serialize_output_json
from emerguav_bench.llm.types import ImageInput, LLMInputRecord, PreferencePair, SFTTarget
from emerguav_bench.utils.progress import progress_iter
from emerguav_bench.validation.json_validator import assert_valid_sample_record


def _frame_sort_key(frame_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", frame_id)
    return (int(match.group(1)) if match else 0, frame_id)


def _sample_to_output(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "Observation_Protocol_Assessment": dict(sample["observation_protocol_gt"]),
        "Intent_Extraction": dict(sample["intent_extraction_gt"]),
        "Action_Output": {
            "action_token": sample["action_token_gt"],
            "confidence": 1.0,
            "abstention_decision": sample["paper_task_gt"]["T11"],
            "collaboration_request": dict(sample["collaboration_gt"]),
        },
        "Rationale_Short": "Ground-truth structured target for EmergUAV-Bench.",
    }


def _safe_alternative_output(sample: dict[str, Any], action_token: str) -> dict[str, Any]:
    output = _sample_to_output(sample)
    collaboration_request = dict(sample["collaboration_gt"])
    if action_token == "Ask_Other_UAV":
        collaboration_request["required"] = True
        if collaboration_request.get("request_type") == "none":
            collaboration_request["request_type"] = "reobserve_context"
    output["Action_Output"]["action_token"] = action_token
    output["Action_Output"]["abstention_decision"] = derive_abstention_decision(action_token)
    output["Action_Output"]["collaboration_request"] = collaboration_request
    output["Rationale_Short"] = f"Safe alternative action {action_token} accepted by benchmark policy."
    return output


def _derive_hard_negative_actions(sample: dict[str, Any]) -> list[str]:
    negatives = list(sample.get("forbidden_actions", []))
    if sample.get("risk_level_gt") in {"high", "critical"}:
        negatives.extend(["Proceed_Slow", "Descend_Cautiously"])
    constraints = set(sample.get("mission_constraints", []))
    if {"avoid_nfz", "maintain_sensitive_site_buffer", "avoid_close_shot"} & constraints:
        negatives.extend(["Descend_Cautiously", "Proceed_Slow"])
    if sample.get("observation_protocol_gt", {}).get("trusted_modality") != "rgb":
        negatives.extend(["Turn_Left_Slow", "Turn_Right_Slow"])

    safe_actions = {sample["action_token_gt"], *sample.get("acceptable_safe_alternatives", [])}
    deduped: list[str] = []
    for action_token in negatives:
        if action_token in ACTION_TOKENS and action_token not in safe_actions and action_token not in deduped:
            deduped.append(action_token)
    return deduped


def _context_roles(count: int) -> list[str]:
    roles: list[str] = []
    for offset in range(1, count + 1):
        roles.extend([f"context_prev_{offset}", f"context_next_{offset}"])
    return roles


def _placeholder_image_path(sample: dict[str, Any], role: str) -> str:
    return f"placeholder://context/{sample['sequence_id']}/{sample['frame_id']}/{role}.jpg"


def _assemble_image_inputs(
    sample: dict[str, Any],
    *,
    sequence_samples: list[dict[str, Any]],
    context_frames: int,
    min_image_inputs: int,
) -> list[ImageInput]:
    sorted_samples = sorted(sequence_samples, key=lambda item: _frame_sort_key(item["frame_id"]))
    sample_index = next(index for index, item in enumerate(sorted_samples) if item["sample_id"] == sample["sample_id"])

    image_inputs = [
        ImageInput(
            path=sample["rgb_frame_path"],
            role="primary_frame",
            frame_id=sample["frame_id"],
            source_sample_id=sample["sample_id"],
            is_placeholder=str(sample["rgb_frame_path"]).startswith("placeholder://"),
        )
    ]

    for offset in range(1, context_frames + 1):
        for direction, role_name in ((-1, f"context_prev_{offset}"), (1, f"context_next_{offset}")):
            neighbor_index = sample_index + (offset * direction)
            if 0 <= neighbor_index < len(sorted_samples):
                neighbor = sorted_samples[neighbor_index]
                image_inputs.append(
                    ImageInput(
                        path=neighbor["rgb_frame_path"],
                        role=role_name,
                        frame_id=neighbor["frame_id"],
                        source_sample_id=neighbor["sample_id"],
                        is_placeholder=str(neighbor["rgb_frame_path"]).startswith("placeholder://"),
                    )
                )

    for role_name in _context_roles(context_frames):
        if len(image_inputs) >= min_image_inputs:
            break
        if any(image_input.role == role_name for image_input in image_inputs):
            continue
        image_inputs.append(
            ImageInput(
                path=_placeholder_image_path(sample, role_name),
                role=role_name,
                frame_id=sample["frame_id"],
                source_sample_id=sample["sample_id"],
                is_placeholder=True,
            )
        )

    return image_inputs


def build_llm_input_manifest(
    samples: list[dict[str, Any]],
    *,
    split_name: str = "dev",
    instruction_source: str = "noisy",
    context_frames: int = 2,
    min_image_inputs: int = 3,
) -> list[dict[str, Any]]:
    sequence_to_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in progress_iter(samples, desc="validate llm samples", total=len(samples), unit="sample"):
        assert_valid_sample_record(sample)
        sequence_to_samples[sample["sequence_id"]].append(sample)

    records: list[dict[str, Any]] = []
    for sample in progress_iter(samples, desc="build llm manifest", total=len(samples), unit="record"):
        image_inputs = _assemble_image_inputs(
            sample,
            sequence_samples=sequence_to_samples[sample["sequence_id"]],
            context_frames=context_frames,
            min_image_inputs=min_image_inputs,
        )
        instruction_text = sample["noisy_instruction"] if instruction_source == "noisy" else sample["clean_instruction"]
        record = LLMInputRecord(
            sample_id=sample["sample_id"],
            sequence_id=sample["sequence_id"],
            split_name=split_name,
            image_inputs=image_inputs,
            prompt_messages=[],
            ground_truth_output=_sample_to_output(sample),
            metadata={
                "source_domain_bucket": sample["source_domain_bucket"],
                "protocol_context": sample["protocol_context"],
                "application_theme": sample["application_theme"],
                "policy_family": sample["policy_family"],
                "language_entropy_level": sample["language_entropy_level"],
                "language_entropy_labels": list(sample["language_entropy_labels"]),
                "observation_severity_code": sample["observation_severity_code"],
                "instruction_source": instruction_source,
                "instruction_text": instruction_text,
                "sample_record": sample,
            },
        )
        record.prompt_messages = build_default_prompt_messages(record)
        record.sft_targets = [
            SFTTarget(
                target_text=serialize_output_json(record.ground_truth_output),
                training_weight=1.0,
                variant="primary_ground_truth",
                action_token=sample["action_token_gt"],
            )
        ]
        for alternative in sample["acceptable_safe_alternatives"]:
            record.sft_targets.append(
                SFTTarget(
                    target_text=serialize_output_json(_safe_alternative_output(sample, alternative)),
                    training_weight=0.5,
                    variant="safe_alternative",
                    action_token=alternative,
                )
            )

        prompt_text = render_user_prompt(record)
        primary_text = serialize_output_json(record.ground_truth_output)
        for alternative in [sample["action_token_gt"], *sample["acceptable_safe_alternatives"]]:
            chosen_text = primary_text if alternative == sample["action_token_gt"] else serialize_output_json(_safe_alternative_output(sample, alternative))
            pair_type_prefix = "ground_truth" if alternative == sample["action_token_gt"] else "safe_alternative"
            for negative in _derive_hard_negative_actions(sample):
                record.preference_pairs.append(
                    PreferencePair(
                        prompt_text=prompt_text,
                        chosen_text=chosen_text,
                        rejected_text=serialize_output_json(_safe_alternative_output(sample, negative)),
                        pair_type=f"{pair_type_prefix}_vs_negative",
                        chosen_action_token=alternative,
                        rejected_action_token=negative,
                        pair_weight=1.0 if alternative == sample["action_token_gt"] else 0.75,
                    )
                )

        records.append(record.to_dict())
    return records


def build_sft_records(llm_manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in progress_iter(llm_manifest, desc="build sft records", total=len(llm_manifest), unit="record"):
        prompt_messages = [dict(message) for message in record["prompt_messages"]]
        for target in record.get("sft_targets", []):
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "sequence_id": record["sequence_id"],
                    "split_name": record["split_name"],
                    "messages": prompt_messages,
                    "image_inputs": list(record["image_inputs"]),
                    "target_text": target["target_text"],
                    "training_weight": float(target["training_weight"]),
                    "variant": target["variant"],
                    "action_token": target["action_token"],
                    "metadata": dict(record.get("metadata", {})),
                }
            )
    return rows


def build_preference_records(
    llm_manifest: list[dict[str, Any]],
    *,
    hardest_fraction: float = 0.3,
) -> list[dict[str, Any]]:
    scored_records: list[tuple[float, dict[str, Any]]] = []
    for record in progress_iter(llm_manifest, desc="score preference records", total=len(llm_manifest), unit="record"):
        sample = record.get("metadata", {}).get("sample_record", {})
        risk_score = 2.0 if sample.get("risk_level_gt") in {"high", "critical"} else 0.0
        modality_score = 1.0 if sample.get("observation_protocol_gt", {}).get("trusted_modality") != "rgb" else 0.0
        forbidden_score = 1.0 if sample.get("forbidden_actions") else 0.0
        compliance_score = 1.0 if sample.get("policy_family") != "General Flight Operations" else 0.0
        scored_records.append((risk_score + modality_score + forbidden_score + compliance_score, record))

    scored_records.sort(key=lambda item: item[0], reverse=True)
    cutoff = max(1, int(len(scored_records) * hardest_fraction)) if scored_records else 0
    selected_records = [record for _, record in scored_records[:cutoff]]

    rows: list[dict[str, Any]] = []
    for record in progress_iter(selected_records, desc="build preference records", total=len(selected_records), unit="record"):
        for pair in record.get("preference_pairs", []):
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "sequence_id": record["sequence_id"],
                    "split_name": record["split_name"],
                    "prompt_text": pair["prompt_text"],
                    "chosen_text": pair["chosen_text"],
                    "rejected_text": pair["rejected_text"],
                    "pair_type": pair["pair_type"],
                    "chosen_action_token": pair["chosen_action_token"],
                    "rejected_action_token": pair["rejected_action_token"],
                    "pair_weight": float(pair.get("pair_weight", 1.0)),
                    "image_inputs": list(record["image_inputs"]),
                    "metadata": dict(record.get("metadata", {})),
                }
            )
    return rows
