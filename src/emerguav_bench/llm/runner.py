from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from emerguav_bench.config.vocab import derive_abstention_decision
from emerguav_bench.llm.io import save_json
from emerguav_bench.llm.normalizer import normalize_model_response
from emerguav_bench.llm.prompts import build_few_shot_messages, build_zero_shot_messages
from emerguav_bench.llm.registry import create_adapter
from emerguav_bench.llm.types import LLMInputRecord, ensure_llm_input_record, validate_llm_input_record
from emerguav_bench.utils.gpu import clear_cuda_memory, is_cuda_out_of_memory_error, recommended_inference_batch_size
from emerguav_bench.utils.progress import make_progress_bar

_LLM_PROGRESS_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]"


def _stable_generation_seed(seed: int, sample_ids: list[str]) -> int:
    payload = f"{seed}\n" + "\n".join(sample_ids)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _seed_live_generation(adapter: Any, seed: int, sample_ids: list[str]) -> None:
    if getattr(adapter, "runtime", None) != "live":
        return
    import torch

    batch_seed = _stable_generation_seed(seed, sample_ids)
    torch.manual_seed(batch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(batch_seed)


def _truncate_metadata_text(text: str, *, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated, total {len(text)} chars]"


def _raw_response_preview(raw_response: Any) -> str:
    if isinstance(raw_response, str):
        return _truncate_metadata_text(raw_response)
    try:
        text = json.dumps(raw_response, ensure_ascii=False)
    except TypeError:
        text = repr(raw_response)
    return _truncate_metadata_text(text)


def _normalization_failure_field_status() -> dict[str, str]:
    fields = [
        "Observation_Protocol_Assessment.protocol_context_profile",
        "Observation_Protocol_Assessment.primary_degradation",
        "Observation_Protocol_Assessment.usability",
        "Observation_Protocol_Assessment.trusted_modality",
        "Observation_Protocol_Assessment.risk_level",
        "Observation_Protocol_Assessment.risk_zone_label",
        "Observation_Protocol_Assessment.viewpoint_summary",
        "Observation_Protocol_Assessment.cause_label",
        "Intent_Extraction.cleaned_instruction",
        "Intent_Extraction.target_object",
        "Intent_Extraction.constraints",
        "Intent_Extraction.target_goal",
        "Action_Output.action_token",
        "Action_Output.confidence",
        "Action_Output.abstention_decision",
        "Action_Output.collaboration_request.required",
        "Action_Output.collaboration_request.request_type",
        "Action_Output.collaboration_request.trusted_modality",
        "Rationale_Short",
    ]
    return dict.fromkeys(fields, "default")


def _build_normalization_failure_output(
    record: LLMInputRecord,
    raw_response: Any,
    error: ValueError,
) -> dict[str, Any]:
    action_token = "Reobserve"
    return {
        "Observation_Protocol_Assessment": {
            "protocol_context_profile": "unknown",
            "primary_degradation": "unknown",
            "usability": "unknown",
            "trusted_modality": "multi",
            "risk_level": "medium",
            "risk_zone_label": "unknown",
            "viewpoint_summary": "unknown",
            "cause_label": "normalization_failed",
        },
        "Intent_Extraction": {
            "cleaned_instruction": str(record.metadata.get("instruction_text") or ""),
            "target_object": None,
            "constraints": [],
            "target_goal": "unknown",
        },
        "Action_Output": {
            "action_token": action_token,
            "confidence": 0.0,
            "abstention_decision": derive_abstention_decision(action_token),
            "collaboration_request": {
                "required": True,
                "request_type": "reobserve_context",
                "trusted_modality": "multi",
            },
        },
        "Rationale_Short": "Model output could not be parsed as the required JSON contract.",
        "_truncated": False,
        "_field_status": _normalization_failure_field_status(),
        "_normalized": True,
        "_normalization_failed": True,
        "_normalization_error": _truncate_metadata_text(str(error)),
        "_raw_response_preview": _raw_response_preview(raw_response),
    }


def _prediction_has_normalization_failure(prediction: dict[str, Any]) -> bool:
    output = prediction.get("output")
    return isinstance(output, dict) and bool(output.get("_normalization_failed"))


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    rounded = int(round(seconds))
    minutes, remaining_seconds = divmod(rounded, 60)
    if minutes < 60:
        return f"{minutes}m{remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h{remaining_minutes:02d}m"


class LLMInferenceRunner:
    def __init__(
        self,
        *,
        model_id: str,
        runtime: str = "mock",
        generation_kwargs: dict[str, Any] | None = None,
        processor_max_pixels: int | None = None,
        batch_size: int = 1,
        auto_batch: bool = True,
        precision: str | None = None,
    ) -> None:
        if batch_size < 0:
            raise ValueError("batch_size must be >= 0")
        self.model_id = model_id
        self.requested_batch_size = batch_size
        self.batch_size = recommended_inference_batch_size(model_id) if batch_size == 0 else max(1, batch_size)
        self.auto_batch = auto_batch
        self.precision = precision
        self.adapter = create_adapter(model_id, runtime=runtime, generation_kwargs=generation_kwargs, precision=precision)
        self.adapter.processor_max_pixels = processor_max_pixels
        self.last_run_metadata: dict[str, Any] = {
            "model_id": model_id,
            "requested_batch_size": batch_size,
            "initial_batch_size": self.batch_size,
            "effective_batch_size": self.batch_size,
            "auto_batch": auto_batch,
            "oom_fallback_count": 0,
            "precision": precision,
            "normalization_failure_count": 0,
            "normalization_failure_sample_ids": [],
        }

    def _select_few_shot_records(
        self,
        records: list[LLMInputRecord],
        *,
        current_sample_id: str,
        few_shot_count: int,
        seed: int,
    ) -> list[LLMInputRecord]:
        candidates = [record for record in records if record.sample_id != current_sample_id]
        rng = random.Random(seed)
        rng.shuffle(candidates)
        return candidates[:few_shot_count]

    def run(
        self,
        llm_manifest: list[dict[str, Any] | LLMInputRecord],
        *,
        prompt_mode: str = "zero_shot",
        few_shot_count: int = 0,
        seed: int = 0,
        existing_predictions: list[dict[str, Any]] | None = None,
        on_prediction: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> list[dict[str, Any]]:
        records = [record if isinstance(record, LLMInputRecord) else ensure_llm_input_record(record) for record in llm_manifest]
        existing_by_sample_id = {
            str(prediction["sample_id"]): prediction
            for prediction in existing_predictions or []
            if isinstance(prediction, dict) and "sample_id" in prediction
        }
        predictions: list[dict[str, Any]] = []
        current_batch_size = self.batch_size
        oom_fallback_count = 0
        normalization_failure_count = 0
        normalization_failure_sample_ids: list[str] = []

        def record_normalization_failure(sample_id: str) -> None:
            nonlocal normalization_failure_count
            normalization_failure_count += 1
            if len(normalization_failure_sample_ids) < 50:
                normalization_failure_sample_ids.append(sample_id)
            self.last_run_metadata["normalization_failure_count"] = normalization_failure_count
            self.last_run_metadata["normalization_failure_sample_ids"] = list(normalization_failure_sample_ids)

        def messages_for(record: LLMInputRecord) -> list[dict[str, str]]:
            if prompt_mode == "few_shot":
                few_shot_records = self._select_few_shot_records(
                    records,
                    current_sample_id=record.sample_id,
                    few_shot_count=few_shot_count,
                    seed=seed,
                )
                return build_few_shot_messages(record, few_shot_records=few_shot_records)
            return build_zero_shot_messages(record)

        with make_progress_bar(
            total=len(records),
            desc=f"{self.model_id} inference",
            unit="sample",
            bar_format=_LLM_PROGRESS_BAR_FORMAT,
        ) as progress_bar:
            index = 0
            while index < len(records):
                record = records[index]
                errors = validate_llm_input_record(record)
                if errors:
                    raise ValueError(f"Invalid llm input record for {record.sample_id}: {'; '.join(errors)}")
                if record.sample_id in existing_by_sample_id:
                    existing_prediction = existing_by_sample_id[record.sample_id]
                    predictions.append(existing_prediction)
                    if _prediction_has_normalization_failure(existing_prediction):
                        record_normalization_failure(record.sample_id)
                    progress_bar.set_postfix(
                        sample=record.sample_id,
                        split=record.split_name,
                        item=f"{index + 1}/{len(records)}",
                        batch=current_batch_size,
                        refresh=False,
                    )
                    progress_bar.update(1)
                    index += 1
                    continue

                batch_records: list[LLMInputRecord] = []
                batch_messages: list[list[dict[str, str]]] = []
                lookahead = index
                while lookahead < len(records) and len(batch_records) < current_batch_size:
                    candidate = records[lookahead]
                    if candidate.sample_id in existing_by_sample_id:
                        break
                    candidate_errors = validate_llm_input_record(candidate)
                    if candidate_errors:
                        raise ValueError(f"Invalid llm input record for {candidate.sample_id}: {'; '.join(candidate_errors)}")
                    batch_records.append(candidate)
                    batch_messages.append(messages_for(candidate))
                    lookahead += 1

                while True:
                    batch_start = time.perf_counter()
                    try:
                        _seed_live_generation(
                            self.adapter,
                            seed,
                            [batch_record.sample_id for batch_record in batch_records],
                        )
                        raw_responses = self.adapter.generate_batch(batch_records, batch_messages)
                        break
                    except Exception as exc:
                        if not (self.auto_batch and current_batch_size > 1 and is_cuda_out_of_memory_error(exc)):
                            if is_cuda_out_of_memory_error(exc):
                                clear_cuda_memory()
                            raise
                        clear_cuda_memory()
                        current_batch_size = max(1, current_batch_size // 2)
                        oom_fallback_count += 1
                        batch_records = batch_records[:current_batch_size]
                        batch_messages = batch_messages[:current_batch_size]
                        self.last_run_metadata["effective_batch_size"] = current_batch_size
                        self.last_run_metadata["oom_fallback_count"] = oom_fallback_count

                if len(raw_responses) != len(batch_records):
                    raise RuntimeError(f"Adapter returned {len(raw_responses)} responses for {len(batch_records)} records")
                for batch_record, raw_response in zip(batch_records, raw_responses, strict=True):
                    try:
                        normalized_output = normalize_model_response(raw_response)
                    except ValueError as exc:
                        record_normalization_failure(batch_record.sample_id)
                        normalized_output = _build_normalization_failure_output(batch_record, raw_response, exc)
                    predictions.append({"sample_id": batch_record.sample_id, "output": normalized_output})
                    if on_prediction is not None:
                        on_prediction(predictions)
                batch_seconds = max(0.0, time.perf_counter() - batch_start)
                batch_end_index = index + len(batch_records)
                remaining_new_records = sum(
                    1 for future_record in records[batch_end_index:] if future_record.sample_id not in existing_by_sample_id
                )
                seconds_per_sample = batch_seconds / max(1, len(batch_records))
                last_batch_record = batch_records[-1]
                postfix = {
                    "sample": last_batch_record.sample_id,
                    "split": last_batch_record.split_name,
                    "item": f"{batch_end_index}/{len(records)}",
                    "batch": len(batch_records),
                    "batch_time": f"{batch_seconds:.2f}s/batch",
                    "throughput": f"{len(batch_records) / batch_seconds:.2f}sample/s" if batch_seconds > 0 else "inf sample/s",
                    "eta": _format_seconds(seconds_per_sample * remaining_new_records),
                }
                if len(batch_records) != current_batch_size:
                    postfix["target_batch"] = current_batch_size
                progress_bar.set_postfix(**postfix, refresh=False)
                progress_bar.update(len(batch_records))
                index = batch_end_index
        self.last_run_metadata.update(
            {
                "effective_batch_size": current_batch_size,
                "oom_fallback_count": oom_fallback_count,
                "normalization_failure_count": normalization_failure_count,
                "normalization_failure_sample_ids": normalization_failure_sample_ids,
                "generated_prediction_count": len(predictions) - len(existing_by_sample_id),
                "total_prediction_count": len(predictions),
            }
        )
        return predictions


def save_predictions_tree(
    predictions: list[dict[str, Any]],
    *,
    results_root: str | Path,
    model_id: str,
    split_name: str,
    prompt_mode: str,
    seed: int,
) -> Path:
    output_dir = Path(results_root) / model_id / split_name / prompt_mode / f"seed_{seed}"
    output_path = output_dir / "predictions.json"
    save_json(output_path, predictions)
    return output_path
