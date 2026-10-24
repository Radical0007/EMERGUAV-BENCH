from __future__ import annotations

import json
import re
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from emerguav_bench.llm.types import LLMInputRecord, ModelPreset, ensure_llm_input_record
from emerguav_bench.utils.gpu import is_cuda_out_of_memory_error


def _apply_chat_template_with_prefill(
    processor: Any,
    multimodal_messages: list[dict[str, Any]],
    *,
    images: list[Any],
    fallback_text: str,
    processor_kwargs: dict[str, Any] | None = None,
    prefill_json: bool = True,
) -> tuple[dict[str, Any], bool]:
    try:
        chat_template_kwargs: dict[str, Any] = {"add_generation_prompt": True, "tokenize": False}
        if processor_kwargs is not None:
            chat_template_kwargs["processor_kwargs"] = processor_kwargs
        prompt_text: str = processor.apply_chat_template(multimodal_messages, **chat_template_kwargs)
        if prefill_json:
            prompt_text += "{"
        processor_inputs: dict[str, Any] = {"text": [prompt_text], "return_tensors": "pt"}
        if images:
            processor_inputs["images"] = images
        return processor(**processor_inputs), prefill_json
    except TypeError:
        if prefill_json:
            fallback_text += "{"
        processor_inputs = {"text": [fallback_text], "return_tensors": "pt"}
        if images:
            processor_inputs["images"] = images
        return processor(**processor_inputs), prefill_json


def _render_chat_template_text_with_prefill(
    processor: Any,
    multimodal_messages: list[dict[str, Any]],
    *,
    fallback_text: str,
    prefill_json: bool = True,
) -> tuple[str, bool]:
    try:
        prompt_text: str = processor.apply_chat_template(
            multimodal_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    except TypeError:
        prompt_text = fallback_text
    return (prompt_text + "{", True) if prefill_json else (prompt_text, False)


def _transformers_prefers_dtype_keyword(transformers: Any) -> bool:
    version = str(getattr(transformers, "__version__", ""))
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        return False
    major = int(match.group(1))
    minor = int(match.group(2))
    return major > 4 or (major == 4 and minor >= 56)


def _normalize_model_dtype_keyword(transformers: Any, model_kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(model_kwargs)
    if _transformers_prefers_dtype_keyword(transformers) and "torch_dtype" in kwargs and "dtype" not in kwargs:
        kwargs["dtype"] = kwargs.pop("torch_dtype")
    return kwargs


def _model_dtype_keyword(model_kwargs: dict[str, Any]) -> str | None:
    if "dtype" in model_kwargs:
        return "dtype"
    if "torch_dtype" in model_kwargs:
        return "torch_dtype"
    return None


def _smolvlm_chat_template_processor_kwargs(processor: Any) -> dict[str, Any]:
    video_processor = getattr(processor, "video_processor", None)
    return {
        "num_frames": getattr(video_processor, "num_frames", None),
        "fps": getattr(video_processor, "fps", None),
    }


def _apply_precision_to_model_kwargs(transformers: Any, model_kwargs: dict[str, Any], precision: str | None) -> dict[str, Any]:
    if precision is None:
        return _normalize_model_dtype_keyword(transformers, model_kwargs)
    normalized = precision.lower()
    kwargs = dict(model_kwargs)
    if normalized == "auto":
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = "auto"
        return _normalize_model_dtype_keyword(transformers, kwargs)

    import torch

    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized in dtype_map:
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = dtype_map[normalized]
        return _normalize_model_dtype_keyword(transformers, kwargs)
    if normalized in {"int8", "8bit"}:
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(load_in_8bit=True)
        return _normalize_model_dtype_keyword(transformers, kwargs)
    if normalized in {"int4", "4bit"}:
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        return _normalize_model_dtype_keyword(transformers, kwargs)
    raise ValueError(f"Unsupported precision: {precision}")


def torch_inference_context(torch_module: Any) -> Any:
    context_factory = getattr(torch_module, "inference_mode", None) or getattr(torch_module, "no_grad", None)
    return context_factory() if context_factory is not None else nullcontext()


def _lazy_import_pil() -> Any:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required for live multimodal inference") from exc
    return Image, ImageDraw


def _candidate_local_paths(path: str) -> list[Path]:
    candidates: list[Path] = []
    raw_path = str(path)
    normalized = raw_path.replace("\\", "/")

    for candidate in (Path(raw_path), Path(normalized)):
        if candidate not in candidates:
            candidates.append(candidate)

    cwd = Path.cwd()
    cwd_name = cwd.name
    for marker in ("artifacts/", "data/", "outputs/"):
        if marker in normalized:
            suffix = normalized.split(marker, 1)[1]
            candidate = cwd / marker.rstrip("/") / suffix
            if candidate not in candidates:
                candidates.append(candidate)

    project_marker = f"/{cwd_name}/"
    if project_marker in normalized:
        suffix = normalized.split(project_marker, 1)[1]
        candidate = cwd / suffix
        if candidate not in candidates:
            candidates.append(candidate)

    drive_match = re.match(r"^[A-Za-z]:/(.+)$", normalized)
    if drive_match:
        suffix = drive_match.group(1)
        for marker in ("artifacts/", "data/", "outputs/"):
            if marker in suffix:
                relative_suffix = suffix.split(marker, 1)[1]
                candidate = cwd / marker.rstrip("/") / relative_suffix
                if candidate not in candidates:
                    candidates.append(candidate)

    return candidates


def resolve_existing_local_path(path: str) -> Path | None:
    for candidate in _candidate_local_paths(path):
        if candidate.is_file():
            return candidate
    return None


def load_image_for_adapter(path: str, *, role: str) -> Any:
    Image, ImageDraw = _lazy_import_pil()
    if path.startswith("placeholder://"):
        image = Image.new("RGB", (224, 224), color=(242, 240, 229))
        drawer = ImageDraw.Draw(image)
        drawer.text((12, 12), role, fill=(35, 35, 35))
        drawer.text((12, 40), "opaque placeholder", fill=(60, 60, 60))
        return image
    resolved_path = resolve_existing_local_path(path)
    if resolved_path is None:
        tried = ", ".join(str(candidate) for candidate in _candidate_local_paths(path))
        raise FileNotFoundError(f"Image path not found: {path}. Tried: {tried}")
    return Image.open(resolved_path).convert("RGB")


def serialize_messages(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for message in messages:
        lines.append(f"{message['role'].upper()}: {message['content']}")
    return "\n\n".join(lines)


def build_multimodal_chat_messages(messages: list[dict[str, str]], images: list[Any]) -> list[dict[str, Any]]:
    chat_messages: list[dict[str, Any]] = []
    last_user_index = max((index for index, message in enumerate(messages) if message.get("role") == "user"), default=None)
    for index, message in enumerate(messages):
        content_parts: list[dict[str, Any]] = []
        if index == last_user_index:
            content_parts.extend({"type": "image", "image": image} for image in images)
        content_parts.append({"type": "text", "text": str(message.get("content", ""))})
        chat_messages.append({"role": message.get("role", "user"), "content": content_parts})
    if last_user_index is None and images:
        chat_messages.append({"role": "user", "content": [{"type": "image", "image": image} for image in images]})
    return chat_messages


def live_vision_model_class_hints(model_class_hint: str) -> list[str]:
    canonical_auto_hints = [
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForMultimodalLM",
        "AutoModelForCausalLM",
        "AutoModel",
    ]
    ordered_hints: list[str] = []
    if model_class_hint:
        ordered_hints.append(model_class_hint)
    ordered_hints.extend(hint for hint in canonical_auto_hints if hint not in ordered_hints)
    return ordered_hints


def _infer_model_input_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _move_inputs_to_model_device(inputs: Any, model: Any) -> Any:
    device = _infer_model_input_device(model)
    if device is None:
        return inputs
    if hasattr(inputs, "to"):
        try:
            return inputs.to(device)
        except Exception:
            pass
    if isinstance(inputs, dict):
        moved = {}
        for key, value in inputs.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved
    return inputs


def _input_prompt_length(inputs: Any) -> int | None:
    input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else getattr(inputs, "input_ids", None)
    if input_ids is None:
        return None
    shape = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[-1])
    try:
        first_row = input_ids[0]
        return len(first_row)
    except Exception:
        return None


def trim_generated_prompt_tokens(outputs: Any, inputs: Any) -> Any:
    prompt_length = _input_prompt_length(inputs)
    if prompt_length is None:
        return outputs
    output_shape = getattr(outputs, "shape", None)
    if output_shape is not None and len(output_shape) >= 2 and int(output_shape[-1]) <= prompt_length:
        return outputs
    try:
        return outputs[:, prompt_length:]
    except Exception:
        pass
    if isinstance(outputs, list):
        trimmed = []
        for output in outputs:
            try:
                trimmed.append(output[prompt_length:] if len(output) > prompt_length else output)
            except Exception:
                trimmed.append(output)
        return trimmed
    return outputs


def apply_processor_memory_limits(processor: Any, *, max_pixels: int | None = None) -> None:
    if max_pixels is None:
        return
    for target in (processor, getattr(processor, "image_processor", None)):
        if target is None:
            continue
        try:
            setattr(target, "max_pixels", max_pixels)
        except Exception:
            pass
        size = getattr(target, "size", None)
        if size is None:
            continue
        try:
            if isinstance(size, dict):
                size["longest_edge"] = max_pixels
            elif hasattr(size, "longest_edge"):
                size.longest_edge = max_pixels
        except Exception:
            pass


def configure_tokenizer_for_generation(tokenizer: Any) -> None:
    if tokenizer is None:
        return
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            tokenizer.pad_token_id = eos_token_id
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"


def _first_token_id(token_id: Any) -> Any:
    if isinstance(token_id, (list, tuple)):
        return token_id[0] if token_id else None
    return token_id


def resolve_generation_pad_token_id(model: Any, tokenizer: Any) -> Any:
    for source in (tokenizer, getattr(model, "generation_config", None), model):
        if source is None:
            continue
        pad_token_id = _first_token_id(getattr(source, "pad_token_id", None))
        if pad_token_id is not None:
            return pad_token_id
    for source in (tokenizer, getattr(model, "generation_config", None), model):
        if source is None:
            continue
        eos_token_id = _first_token_id(getattr(source, "eos_token_id", None))
        if eos_token_id is not None:
            return eos_token_id
    return None


class BaseAdapter:
    def __init__(
        self,
        preset: ModelPreset,
        *,
        runtime: str = "mock",
        generation_kwargs: dict[str, Any] | None = None,
        precision: str | None = None,
    ) -> None:
        self.preset = preset
        self.runtime = runtime
        self.generation_kwargs = {**preset.default_generation_kwargs, **(generation_kwargs or {})}
        self.precision = precision
        self._loaded = False
        self._processor: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self.processor_max_pixels: int | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "model_id": self.preset.model_id,
            "model_name_or_path": self.preset.model_name_or_path,
            "family": self.preset.family,
            "capability": self.preset.capability.to_dict(),
            "runtime": self.runtime,
        }

    def load(self) -> None:
        if self._loaded:
            return
        if self.runtime == "live":
            self._load_live_backend()
        self._loaded = True

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False

    def __del__(self) -> None:
        try:
            self.unload()
        except Exception:
            pass

    @staticmethod
    def _try_load_vision_model(model_classes: list[tuple[str, Any]], model_name_or_path: str, model_kwargs: dict[str, Any]) -> tuple[Any | None, list[str]]:
        errors: list[str] = []
        for class_hint, model_class in model_classes:
            try:
                return model_class.from_pretrained(model_name_or_path, **model_kwargs), []
            except Exception as exc:
                errors.append(f"{class_hint}: {exc}")
                continue
        return None, errors

    @staticmethod
    def _resolve_attn_implementation() -> str:
        try:
            import flash_attn
            return "flash_attention_2"
        except ImportError:
            return "sdpa"

    def _allow_dtype_fallback(self) -> bool:
        return self.precision in {None, "auto"}

    def _generation_kwargs_for_generate(self, tokenizer: Any) -> dict[str, Any]:
        kwargs = dict(self.generation_kwargs)
        if kwargs.get("pad_token_id") is None:
            pad_token_id = resolve_generation_pad_token_id(self._model, tokenizer)
            if pad_token_id is not None:
                kwargs["pad_token_id"] = pad_token_id
        return kwargs

    def _load_live_backend(self) -> None:
        warnings.filterwarnings("ignore", message=".*fast path is not available.*")

        try:
            import transformers
        except ImportError as exc:
            raise RuntimeError("transformers is required for live adapter loading") from exc

        processor_kwargs = {"trust_remote_code": True, **self.preset.processor_load_kwargs}
        base_model_kwargs: dict[str, Any] = {}
        if "attn_implementation" not in self.preset.model_load_kwargs:
            base_model_kwargs["attn_implementation"] = self._resolve_attn_implementation()
        model_kwargs = _apply_precision_to_model_kwargs(
            transformers,
            {"trust_remote_code": True, **base_model_kwargs, **self.preset.model_load_kwargs},
            self.precision,
        )
        if self.preset.model_id == "phi4_mm":
            phi4mm_config_class = getattr(transformers, "Phi4MultimodalConfig", None)
            if phi4mm_config_class is not None:
                remote_config = transformers.AutoConfig.from_pretrained(self.preset.model_name_or_path, trust_remote_code=True)
                model_config = phi4mm_config_class.from_dict(remote_config.to_dict())
                attn_implementation = self.preset.model_load_kwargs.get("attn_implementation")
                if attn_implementation is not None:
                    setattr(model_config, "_attn_implementation", attn_implementation)
                model_kwargs = {key: value for key, value in model_kwargs.items() if key != "attn_implementation"}
                model_kwargs["config"] = model_config
                model_kwargs["ignore_mismatched_sizes"] = True
                model_kwargs["trust_remote_code"] = False
        if self.preset.capability.supports_vision:
            processor_error: Exception | None = None
            try:
                self._processor = transformers.AutoProcessor.from_pretrained(self.preset.model_name_or_path, **processor_kwargs)
                apply_processor_memory_limits(self._processor, max_pixels=self.processor_max_pixels)
            except Exception as exc:
                processor_error = exc
                self._processor = None
            class_hints = live_vision_model_class_hints(self.preset.model_class_hint)
            model_classes = []
            for class_hint in class_hints:
                model_class = getattr(transformers, class_hint, None)
                if model_class is not None and all(existing_class is not model_class for _, existing_class in model_classes):
                    model_classes.append((class_hint, model_class))

            self._model, class_errors = self._try_load_vision_model(model_classes, self.preset.model_name_or_path, model_kwargs)
            dtype_key = _model_dtype_keyword(model_kwargs)
            if self._model is None and class_errors and dtype_key is not None and self._allow_dtype_fallback():
                fp32_kwargs = {key: value for key, value in model_kwargs.items() if key != dtype_key}
                fp32_model, fp32_errors = self._try_load_vision_model(model_classes, self.preset.model_name_or_path, fp32_kwargs)
                if fp32_model is not None:
                    self._model = fp32_model
                    class_errors = []
                else:
                    class_errors.extend(f"fp32 fallback: {err}" for err in fp32_errors)

            if self._model is None:
                diagnostics = []
                if processor_error is not None:
                    diagnostics.append(f"processor: {processor_error}")
                diagnostics.extend(class_errors)
                diagnostic_text = "; ".join(diagnostics) if diagnostics else "no compatible model class found"
                raise RuntimeError(f"Unable to load live backend for {self.preset.model_id}: {diagnostic_text}")
            if self._processor is None:
                raise RuntimeError(
                    f"Unable to load live processor for {self.preset.model_id}: {processor_error}"
                )
        else:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.preset.model_name_or_path, **processor_kwargs)
            try:
                self._model = transformers.AutoModelForCausalLM.from_pretrained(self.preset.model_name_or_path, **model_kwargs)
            except Exception:
                dtype_key = _model_dtype_keyword(model_kwargs)
                if dtype_key is None or not self._allow_dtype_fallback():
                    raise
                fp32_kwargs = {key: value for key, value in model_kwargs.items() if key != dtype_key}
                self._model = transformers.AutoModelForCausalLM.from_pretrained(self.preset.model_name_or_path, **fp32_kwargs)

        tokenizer_refs = [
            getattr(self._processor, "tokenizer", None) if self._processor is not None else None,
            self._tokenizer,
        ]
        seen_tokenizer_ids: set[int] = set()
        for tokenizer_ref in tokenizer_refs:
            if tokenizer_ref is None or id(tokenizer_ref) in seen_tokenizer_ids:
                continue
            seen_tokenizer_ids.add(id(tokenizer_ref))
            configure_tokenizer_for_generation(tokenizer_ref)

    def _mock_generate(self, record: LLMInputRecord, messages: list[dict[str, str]]) -> str:
        del messages
        return json.dumps(record.ground_truth_output, ensure_ascii=False)

    def _live_generate(self, record: LLMInputRecord, messages: list[dict[str, str]]) -> str:
        import torch

        prompt_text = serialize_messages(messages)
        json_prefilled = False
        if self.preset.capability.supports_vision:
            if self._processor is None or self._model is None:
                raise RuntimeError("Adapter must be loaded before live generation")
            images = [load_image_for_adapter(image_input.path, role=image_input.role) for image_input in record.image_inputs]
            try:
                if hasattr(self._processor, "apply_chat_template"):
                    multimodal_messages = build_multimodal_chat_messages(messages, images)
                    chat_template_processor_kwargs = (
                        _smolvlm_chat_template_processor_kwargs(self._processor)
                        if self.preset.family == "smolvlm"
                        else None
                    )
                    inputs, json_prefilled = _apply_chat_template_with_prefill(
                        self._processor,
                        multimodal_messages,
                        images=images,
                        fallback_text=prompt_text,
                        processor_kwargs=chat_template_processor_kwargs,
                        prefill_json=not self.preset.model_id.startswith("qwen3_vl"),
                    )
                else:
                    prompt_text += "{"
                    json_prefilled = True
                    processor_inputs = {"text": prompt_text, "return_tensors": "pt"}
                    if images:
                        processor_inputs["images"] = images
                    try:
                        inputs = self._processor(**processor_inputs)
                    except TypeError:
                        processor_inputs.pop("text")
                        inputs = self._processor(prompt_text, **processor_inputs)
            finally:
                for image in images:
                    image.close()
            inputs = _move_inputs_to_model_device(inputs, self._model)
            generation_kwargs = self._generation_kwargs_for_generate(getattr(self._processor, "tokenizer", None))
            with torch_inference_context(torch):
                outputs = self._model.generate(**inputs, **generation_kwargs)
            outputs = trim_generated_prompt_tokens(outputs, inputs)
            if hasattr(self._processor, "batch_decode"):
                decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)
            else:
                decoded = [str(outputs)]
            result = decoded[0]
            if json_prefilled:
                result = "{" + result
            return result

        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Adapter must be loaded before live generation")
        prompt_text += "{"
        json_prefilled = True
        inputs = self._tokenizer(prompt_text, return_tensors="pt")
        inputs = _move_inputs_to_model_device(inputs, self._model)
        generation_kwargs = self._generation_kwargs_for_generate(self._tokenizer)
        with torch_inference_context(torch):
            outputs = self._model.generate(**inputs, **generation_kwargs)
        outputs = trim_generated_prompt_tokens(outputs, inputs)
        result = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        if json_prefilled:
            result = "{" + result
        return result

    def _live_generate_batch(
        self,
        records: list[LLMInputRecord],
        messages_batch: list[list[dict[str, str]]],
    ) -> list[str]:
        import torch

        if len(records) == 1:
            return [self._live_generate(records[0], messages_batch[0])]

        if self.preset.capability.supports_vision:
            if self._processor is None or self._model is None:
                raise RuntimeError("Adapter must be loaded before live generation")
            prompt_texts: list[str] = []
            image_batches: list[list[Any]] = []
            opened_images: list[Any] = []
            json_prefilled = True
            try:
                for record, messages in zip(records, messages_batch, strict=True):
                    images = [load_image_for_adapter(image_input.path, role=image_input.role) for image_input in record.image_inputs]
                    opened_images.extend(images)
                    image_batches.append(images)
                    prompt_text = serialize_messages(messages)
                    if hasattr(self._processor, "apply_chat_template"):
                        multimodal_messages = build_multimodal_chat_messages(messages, images)
                        prompt_text, json_prefilled = _render_chat_template_text_with_prefill(
                            self._processor,
                            multimodal_messages,
                            fallback_text=prompt_text,
                            prefill_json=not self.preset.model_id.startswith("qwen3_vl"),
                        )
                    else:
                        prompt_text += "{"
                    prompt_texts.append(prompt_text)
                processor_inputs = {"text": prompt_texts, "return_tensors": "pt"}
                if any(image_batches):
                    processor_inputs["images"] = image_batches
                try:
                    inputs = self._processor(**processor_inputs, processor_kwargs={"padding": True})
                except TypeError:
                    inputs = self._processor(**processor_inputs)
            finally:
                for image in opened_images:
                    image.close()
            inputs = _move_inputs_to_model_device(inputs, self._model)
            generation_kwargs = self._generation_kwargs_for_generate(getattr(self._processor, "tokenizer", None))
            with torch_inference_context(torch):
                outputs = self._model.generate(**inputs, **generation_kwargs)
            outputs = trim_generated_prompt_tokens(outputs, inputs)
            if hasattr(self._processor, "batch_decode"):
                decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)
            else:
                decoded = [str(output) for output in outputs]
            if len(decoded) != len(records):
                raise RuntimeError(f"Batch decode returned {len(decoded)} outputs for {len(records)} records")
            return ["{" + result if json_prefilled else result for result in decoded]

        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Adapter must be loaded before live generation")
        prompt_texts = [serialize_messages(messages) + "{" for messages in messages_batch]
        inputs = self._tokenizer(prompt_texts, return_tensors="pt", padding=True)
        inputs = _move_inputs_to_model_device(inputs, self._model)
        generation_kwargs = self._generation_kwargs_for_generate(self._tokenizer)
        with torch_inference_context(torch):
            outputs = self._model.generate(**inputs, **generation_kwargs)
        outputs = trim_generated_prompt_tokens(outputs, inputs)
        decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if len(decoded) != len(records):
            raise RuntimeError(f"Batch decode returned {len(decoded)} outputs for {len(records)} records")
        return ["{" + result for result in decoded]

    def generate_batch(
        self,
        records: list[dict[str, Any] | LLMInputRecord],
        messages_batch: list[list[dict[str, str]]],
    ) -> list[str]:
        if len(records) != len(messages_batch):
            raise ValueError("records and messages_batch must have the same length")
        llm_records = [record if isinstance(record, LLMInputRecord) else ensure_llm_input_record(record) for record in records]
        self.load()
        if self.runtime != "live":
            return [self._mock_generate(record, messages) for record, messages in zip(llm_records, messages_batch, strict=True)]
        try:
            return self._live_generate_batch(llm_records, messages_batch)
        except Exception as exc:
            if is_cuda_out_of_memory_error(exc):
                raise
            if len(llm_records) == 1:
                raise
            warnings.warn(
                f"{self.preset.model_id} rejected native batch generation; falling back to single-sample generation: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return [self._live_generate(record, messages) for record, messages in zip(llm_records, messages_batch, strict=True)]

    def generate(self, record: dict[str, Any] | LLMInputRecord, messages: list[dict[str, str]]) -> str:
        return self.generate_batch([record], [messages])[0]
