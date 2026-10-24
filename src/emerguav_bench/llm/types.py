from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ImageInput:
    path: str
    role: str
    frame_id: str
    source_sample_id: str
    is_placeholder: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SFTTarget:
    target_text: str
    training_weight: float
    variant: str
    action_token: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreferencePair:
    prompt_text: str
    chosen_text: str
    rejected_text: str
    pair_type: str
    chosen_action_token: str
    rejected_action_token: str
    pair_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LLMInputRecord:
    sample_id: str
    sequence_id: str
    split_name: str
    image_inputs: list[ImageInput]
    prompt_messages: list[dict[str, str]]
    ground_truth_output: dict[str, Any]
    sft_targets: list[SFTTarget] = field(default_factory=list)
    preference_pairs: list[PreferencePair] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "sequence_id": self.sequence_id,
            "split_name": self.split_name,
            "image_inputs": [image_input.to_dict() for image_input in self.image_inputs],
            "prompt_messages": [dict(message) for message in self.prompt_messages],
            "ground_truth_output": self.ground_truth_output,
            "sft_targets": [target.to_dict() for target in self.sft_targets],
            "preference_pairs": [pair.to_dict() for pair in self.preference_pairs],
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class AdapterCapability:
    family: str
    supports_vision: bool
    supports_multi_image: bool
    supports_few_shot: bool
    preferred_prompt_modes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelPreset:
    model_id: str
    model_name_or_path: str
    family: str
    capability: AdapterCapability
    processor_class_hint: str
    model_class_hint: str
    processor_load_kwargs: dict[str, Any] = field(default_factory=dict)
    model_load_kwargs: dict[str, Any] = field(default_factory=dict)
    default_generation_kwargs: dict[str, Any] = field(default_factory=dict)
    training_priority: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["capability"] = self.capability.to_dict()
        return payload


def ensure_llm_input_record(record: dict[str, Any]) -> LLMInputRecord:
    image_inputs = [
        ImageInput(
            path=image_input["path"],
            role=image_input["role"],
            frame_id=image_input["frame_id"],
            source_sample_id=image_input["source_sample_id"],
            is_placeholder=bool(image_input.get("is_placeholder")),
        )
        for image_input in record.get("image_inputs", [])
    ]
    sft_targets = [
        SFTTarget(
            target_text=target["target_text"],
            training_weight=float(target["training_weight"]),
            variant=target["variant"],
            action_token=target["action_token"],
        )
        for target in record.get("sft_targets", [])
    ]
    preference_pairs = [
        PreferencePair(
            prompt_text=pair["prompt_text"],
            chosen_text=pair["chosen_text"],
            rejected_text=pair["rejected_text"],
            pair_type=pair["pair_type"],
            chosen_action_token=pair["chosen_action_token"],
            rejected_action_token=pair["rejected_action_token"],
            pair_weight=float(pair.get("pair_weight", 1.0)),
        )
        for pair in record.get("preference_pairs", [])
    ]
    return LLMInputRecord(
        sample_id=record["sample_id"],
        sequence_id=record["sequence_id"],
        split_name=record["split_name"],
        image_inputs=image_inputs,
        prompt_messages=[dict(message) for message in record.get("prompt_messages", [])],
        ground_truth_output=dict(record["ground_truth_output"]),
        sft_targets=sft_targets,
        preference_pairs=preference_pairs,
        metadata=dict(record.get("metadata", {})),
    )


def validate_llm_input_record(record: LLMInputRecord) -> list[str]:
    errors: list[str] = []
    if not record.sample_id:
        errors.append("sample_id must be non-empty")
    if not record.sequence_id:
        errors.append("sequence_id must be non-empty")
    if not record.split_name:
        errors.append("split_name must be non-empty")
    allows_no_images = record.metadata.get("ablation_condition") == "text_only"
    if not record.image_inputs and not allows_no_images:
        errors.append("image_inputs must be non-empty unless this is a text_only ablation")
    elif record.image_inputs and record.image_inputs[0].role != "primary_frame":
        errors.append("image_inputs[0] must be the primary_frame")
    if not record.prompt_messages:
        errors.append("prompt_messages must be non-empty")
    if not isinstance(record.ground_truth_output, dict):
        errors.append("ground_truth_output must be an object")
    for target in record.sft_targets:
        if target.training_weight <= 0:
            errors.append("sft target training_weight must be positive")
    for pair in record.preference_pairs:
        if pair.chosen_text == pair.rejected_text:
            errors.append("preference pair chosen_text and rejected_text must differ")
    return errors
