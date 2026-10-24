from __future__ import annotations

from typing import Any

from emerguav_bench.llm.adapters.base import (
    BaseAdapter,
    _apply_precision_to_model_kwargs,
    _infer_model_input_device,
    apply_processor_memory_limits,
    configure_tokenizer_for_generation,
    load_image_for_adapter,
    serialize_messages,
    torch_inference_context,
)
from emerguav_bench.llm.types import AdapterCapability, LLMInputRecord, ModelPreset


def _capability(family: str) -> AdapterCapability:
    return AdapterCapability(
        family=family,
        supports_vision=True,
        supports_multi_image=True,
        supports_few_shot=True,
        preferred_prompt_modes=["zero_shot", "few_shot"],
    )


_AUTO_LOAD = {"device_map": "auto", "torch_dtype": "auto"}
_GEMMA4_LOAD = {**_AUTO_LOAD, "attn_implementation": "sdpa"}
_MINICPM_LOAD = {**_AUTO_LOAD, "attn_implementation": "sdpa"}
_OVIS_LOAD = {**_AUTO_LOAD, "multimodal_max_length": 32768}
_IMAGE_TEXT_MODEL_CLASS = "AutoModelForImageTextToText"
_MINICPM_POST_INIT_PATCH_FLAG = "_emerguav_bench_minicpm_post_init_patch"
_MINICPM_TOKEN_TEXT_ATTRS = {
    "im_start": "<image>",
    "im_end": "</image>",
    "slice_start": "<slice>",
    "slice_end": "</slice>",
    "im_id_start": "<image_id>",
    "im_id_end": "</image_id>",
}
_MINICPM_TOKEN_ID_ATTRS = {
    "im_start_id": "im_start",
    "im_end_id": "im_end",
    "slice_start_id": "slice_start",
    "slice_end_id": "slice_end",
    "im_id_start_id": "im_id_start",
    "im_id_end_id": "im_id_end",
}


def _get_dynamic_module_class_loader(transformers: Any) -> Any | None:
    dynamic_module_utils = getattr(transformers, "dynamic_module_utils", None)
    loader = getattr(dynamic_module_utils, "get_class_from_dynamic_module", None)
    if loader is not None:
        return loader
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except Exception:
        return None
    return get_class_from_dynamic_module


def _patch_minicpm_remote_model_init(model_class: Any) -> None:
    if getattr(model_class, _MINICPM_POST_INIT_PATCH_FLAG, False):
        return
    original_init = model_class.__init__

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if hasattr(self, "all_tied_weights_keys"):
            return
        post_init = getattr(self, "post_init", None)
        if callable(post_init):
            post_init()

    patched_init.__name__ = getattr(original_init, "__name__", "patched_init")
    patched_init.__qualname__ = getattr(original_init, "__qualname__", patched_init.__qualname__)
    model_class.__init__ = patched_init
    setattr(model_class, _MINICPM_POST_INIT_PATCH_FLAG, True)


def _set_attr_if_missing(target: Any, name: str, value: Any) -> None:
    try:
        getattr(target, name)
        return
    except AttributeError:
        pass
    try:
        setattr(target, name, value)
    except Exception:
        pass


def _ensure_minicpm_tokenizer_compat(tokenizer: Any) -> None:
    if tokenizer is None:
        return
    for name, value in _MINICPM_TOKEN_TEXT_ATTRS.items():
        _set_attr_if_missing(tokenizer, name, value)

    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert_tokens_to_ids):
        for id_attr, text_attr in _MINICPM_TOKEN_ID_ATTRS.items():
            token_text = getattr(tokenizer, text_attr, _MINICPM_TOKEN_TEXT_ATTRS[text_attr])
            token_id = convert_tokens_to_ids(token_text)
            if token_id is not None:
                _set_attr_if_missing(tokenizer, id_attr, token_id)
        newline_id = convert_tokens_to_ids("\n")
        if newline_id is not None:
            _set_attr_if_missing(tokenizer, "newline_id", newline_id)

    for alias, source in (("bos_id", "bos_token_id"), ("eos_id", "eos_token_id"), ("unk_id", "unk_token_id")):
        source_id = getattr(tokenizer, source, None)
        if source_id is not None:
            _set_attr_if_missing(tokenizer, alias, source_id)


def _prepare_minicpm_remote_class_for_auto_device_map(transformers: Any, model_name_or_path: str) -> Any | None:
    auto_config = getattr(transformers, "AutoConfig", None)
    if auto_config is None:
        return None
    config = auto_config.from_pretrained(model_name_or_path, trust_remote_code=True)
    auto_map = getattr(config, "auto_map", {}) or {}
    class_ref = auto_map.get("AutoModel")
    if isinstance(class_ref, (list, tuple)):
        class_ref = next((ref for ref in class_ref if isinstance(ref, str)), None)
    if not isinstance(class_ref, str):
        return None
    loader = _get_dynamic_module_class_loader(transformers)
    if loader is None:
        return None
    model_class = loader(class_ref, model_name_or_path, trust_remote_code=True)
    _patch_minicpm_remote_model_init(model_class)
    register = getattr(getattr(transformers, "AutoModel", None), "register", None)
    if callable(register):
        try:
            register(config.__class__, model_class, exist_ok=True)
        except TypeError:
            try:
                register(config.__class__, model_class)
            except Exception:
                pass
        except Exception:
            pass
    return config

EDGE_VLM_PRESETS = {
    "gemma4_e2b_it": ModelPreset(
        model_id="gemma4_e2b_it",
        model_name_or_path="google/gemma-4-E2B-it",
        family="gemma4",
        capability=_capability("gemma4"),
        processor_class_hint="AutoProcessor",
        model_class_hint=_IMAGE_TEXT_MODEL_CLASS,
        model_load_kwargs=dict(_GEMMA4_LOAD),
        default_generation_kwargs={"max_new_tokens": 2048, "do_sample": False},
        training_priority=9,
    ),
    "gemma4_e4b_it": ModelPreset(
        model_id="gemma4_e4b_it",
        model_name_or_path="google/gemma-4-E4B-it",
        family="gemma4",
        capability=_capability("gemma4"),
        processor_class_hint="AutoProcessor",
        model_class_hint=_IMAGE_TEXT_MODEL_CLASS,
        model_load_kwargs=dict(_GEMMA4_LOAD),
        default_generation_kwargs={"max_new_tokens": 2048, "do_sample": False},
        training_priority=10,
    ),
    "gemma4_26b_a4b_it": ModelPreset(
        model_id="gemma4_26b_a4b_it",
        model_name_or_path="google/gemma-4-26B-A4B-it",
        family="gemma4",
        capability=_capability("gemma4"),
        processor_class_hint="AutoProcessor",
        model_class_hint=_IMAGE_TEXT_MODEL_CLASS,
        model_load_kwargs=dict(_GEMMA4_LOAD),
        default_generation_kwargs={"max_new_tokens": 2048, "do_sample": False},
    ),
    "gemma4_31b_it": ModelPreset(
        model_id="gemma4_31b_it",
        model_name_or_path="google/gemma-4-31B-it",
        family="gemma4",
        capability=_capability("gemma4"),
        processor_class_hint="AutoProcessor",
        model_class_hint=_IMAGE_TEXT_MODEL_CLASS,
        model_load_kwargs=dict(_GEMMA4_LOAD),
        default_generation_kwargs={"max_new_tokens": 2048, "do_sample": False},
    ),
    "minicpm_v_4_6": ModelPreset(
        model_id="minicpm_v_4_6",
        model_name_or_path="openbmb/MiniCPM-V-4_6",
        family="minicpm_v",
        capability=_capability("minicpm_v"),
        processor_class_hint="AutoTokenizer",
        model_class_hint="AutoModel",
        model_load_kwargs=dict(_MINICPM_LOAD),
        default_generation_kwargs={"max_new_tokens": 2048, "do_sample": False},
    ),
    "minicpm_v_4_5": ModelPreset(
        model_id="minicpm_v_4_5",
        model_name_or_path="openbmb/MiniCPM-V-4_5",
        family="minicpm_v",
        capability=_capability("minicpm_v"),
        processor_class_hint="AutoTokenizer",
        model_class_hint="AutoModel",
        model_load_kwargs=dict(_MINICPM_LOAD),
        default_generation_kwargs={"max_new_tokens": 2048, "do_sample": False},
        training_priority=11,
    ),
    "glm4_6v_flash": ModelPreset(
        model_id="glm4_6v_flash",
        model_name_or_path="zai-org/GLM-4.6V-Flash",
        family="glm_v",
        capability=_capability("glm_v"),
        processor_class_hint="AutoProcessor",
        model_class_hint=_IMAGE_TEXT_MODEL_CLASS,
        model_load_kwargs=dict(_AUTO_LOAD),
        default_generation_kwargs={"max_new_tokens": 2048, "do_sample": False},
        training_priority=12,
    ),
    "glm4_6v": ModelPreset(
        model_id="glm4_6v",
        model_name_or_path="zai-org/GLM-4.6V",
        family="glm_v",
        capability=_capability("glm_v"),
        processor_class_hint="AutoProcessor",
        model_class_hint=_IMAGE_TEXT_MODEL_CLASS,
        model_load_kwargs=dict(_AUTO_LOAD),
        default_generation_kwargs={"max_new_tokens": 2048, "do_sample": False},
    ),
    "smolvlm2_2_2b": ModelPreset(
        model_id="smolvlm2_2_2b",
        model_name_or_path="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        family="smolvlm",
        capability=_capability("smolvlm"),
        processor_class_hint="AutoProcessor",
        model_class_hint="AutoModelForImageTextToText",
        model_load_kwargs=dict(_AUTO_LOAD),
        default_generation_kwargs={"max_new_tokens": 2048, "do_sample": False},
        training_priority=13,
    ),
    "ovis2_8b": ModelPreset(
        model_id="ovis2_8b",
        model_name_or_path="AIDC-AI/Ovis2-8B",
        family="ovis2",
        capability=_capability("ovis2"),
        processor_class_hint="AutoProcessor",
        model_class_hint="AutoModelForCausalLM",
        model_load_kwargs=dict(_OVIS_LOAD),
        default_generation_kwargs={"max_new_tokens": 2048, "do_sample": False},
    ),
    "ovis2_16b": ModelPreset(
        model_id="ovis2_16b",
        model_name_or_path="AIDC-AI/Ovis2-16B",
        family="ovis2",
        capability=_capability("ovis2"),
        processor_class_hint="AutoProcessor",
        model_class_hint="AutoModelForCausalLM",
        model_load_kwargs=dict(_OVIS_LOAD),
        default_generation_kwargs={"max_new_tokens": 2048, "do_sample": False},
    ),
}


class EdgeVLMAdapter(BaseAdapter):
    def _load_live_backend(self) -> None:
        if self.preset.family == "minicpm_v":
            self._load_minicpm_live_backend()
            return
        if self.preset.family == "ovis2":
            self._load_ovis_live_backend()
            return
        super()._load_live_backend()

    def _load_minicpm_live_backend(self) -> None:
        try:
            import transformers
        except ImportError as exc:
            raise RuntimeError("transformers is required for MiniCPM live adapter loading") from exc

        load_kwargs = _apply_precision_to_model_kwargs(
            transformers,
            {"trust_remote_code": True, **self.preset.model_load_kwargs},
            self.precision,
        )
        tokenizer_kwargs = {"trust_remote_code": True, **self.preset.processor_load_kwargs}
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.preset.model_name_or_path, **tokenizer_kwargs)
        self._processor = transformers.AutoProcessor.from_pretrained(self.preset.model_name_or_path, **tokenizer_kwargs)
        apply_processor_memory_limits(self._processor, max_pixels=self.processor_max_pixels)
        _ensure_minicpm_tokenizer_compat(self._tokenizer)
        _ensure_minicpm_tokenizer_compat(getattr(self._processor, "tokenizer", None))
        model_config = (
            _prepare_minicpm_remote_class_for_auto_device_map(transformers, self.preset.model_name_or_path)
            if load_kwargs.get("device_map") == "auto"
            else None
        )
        if model_config is not None:
            load_kwargs = {**load_kwargs, "config": model_config}
        self._model = transformers.AutoModel.from_pretrained(self.preset.model_name_or_path, **load_kwargs)
        if hasattr(self._model, "eval"):
            self._model = self._model.eval()
        configure_tokenizer_for_generation(self._tokenizer)
        configure_tokenizer_for_generation(getattr(self._processor, "tokenizer", None))

    def _load_ovis_live_backend(self) -> None:
        try:
            import transformers
        except ImportError as exc:
            raise RuntimeError("transformers is required for Ovis live adapter loading") from exc

        load_kwargs = _apply_precision_to_model_kwargs(
            transformers,
            {"trust_remote_code": True, **self.preset.model_load_kwargs},
            self.precision,
        )
        self._model = transformers.AutoModelForCausalLM.from_pretrained(self.preset.model_name_or_path, **load_kwargs)
        if hasattr(self._model, "eval"):
            self._model = self._model.eval()
        if not hasattr(self._model, "preprocess_inputs"):
            raise RuntimeError(f"Ovis model {self.preset.model_name_or_path} does not expose preprocess_inputs")
        self._tokenizer = self._model.get_text_tokenizer()
        configure_tokenizer_for_generation(self._tokenizer)

    @staticmethod
    def _minicpm_messages(messages: list[dict[str, str]], images: list[Any]) -> list[dict[str, Any]]:
        system_text = "\n\n".join(str(message.get("content", "")) for message in messages if message.get("role") == "system")
        non_system_messages = [message for message in messages if message.get("role") != "system"]
        last_user_index = max(
            (index for index, message in enumerate(non_system_messages) if message.get("role") == "user"),
            default=None,
        )
        chat_messages: list[dict[str, Any]] = []
        for index, message in enumerate(non_system_messages):
            role = message.get("role", "user")
            content = str(message.get("content", ""))
            if index == last_user_index:
                if system_text:
                    content = f"{system_text}\n\n{content}"
                chat_messages.append({"role": "user", "content": [*images, content]})
            elif role == "assistant":
                chat_messages.append({"role": "assistant", "content": [content]})
            else:
                chat_messages.append({"role": "user", "content": [content]})
        if last_user_index is None:
            content = system_text or serialize_messages(messages)
            chat_messages.append({"role": "user", "content": [*images, content]})
        return chat_messages

    def _minicpm_live_generate(self, record: LLMInputRecord, messages: list[dict[str, str]]) -> str:
        import torch

        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Adapter must be loaded before MiniCPM live generation")
        images = [load_image_for_adapter(image_input.path, role=image_input.role) for image_input in record.image_inputs]
        try:
            chat_messages = self._minicpm_messages(messages, images)
            generation_kwargs = dict(self.generation_kwargs)
            processor = self._processor
            processor_tokenizer = getattr(processor, "tokenizer", None)
            chat_tokenizer = processor_tokenizer or self._tokenizer
            _ensure_minicpm_tokenizer_compat(self._tokenizer)
            _ensure_minicpm_tokenizer_compat(processor_tokenizer)
            chat_kwargs: dict[str, Any] = {
                "processor": processor,
                "sampling": bool(generation_kwargs.get("do_sample", False)),
                "enable_thinking": False,
            }
            if "max_new_tokens" in generation_kwargs:
                chat_kwargs["max_new_tokens"] = generation_kwargs["max_new_tokens"]
            attempts = [
                {"image": None, **chat_kwargs},
                dict(chat_kwargs),
                {"image": None},
                {},
            ]
            last_error: TypeError | None = None
            for kwargs in attempts:
                try:
                    with torch_inference_context(torch):
                        response = self._model.chat(msgs=chat_messages, tokenizer=chat_tokenizer, **kwargs)
                    if isinstance(response, tuple):
                        response = response[0]
                    return str(response)
                except TypeError as exc:
                    last_error = exc
            raise last_error if last_error is not None else RuntimeError("MiniCPM chat failed")
        finally:
            for image in images:
                image.close()

    @staticmethod
    def _ovis_query(messages: list[dict[str, str]], image_count: int) -> str:
        image_prefix = "\n".join(f"Image {index + 1}: <image>" for index in range(image_count))
        prompt_text = serialize_messages(messages)
        return f"{image_prefix}\n{prompt_text}\n{{" if image_prefix else f"{prompt_text}\n{{"

    def _ovis_live_generate(self, record: LLMInputRecord, messages: list[dict[str, str]]) -> str:
        import torch

        if self._model is None:
            raise RuntimeError("Adapter must be loaded before Ovis live generation")
        text_tokenizer = self._model.get_text_tokenizer()
        visual_tokenizer = self._model.get_visual_tokenizer()
        images = [load_image_for_adapter(image_input.path, role=image_input.role) for image_input in record.image_inputs]
        try:
            query = self._ovis_query(messages, len(images))
            _, input_ids, pixel_values = self._model.preprocess_inputs(query, images, max_partition=9)
            attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
            device = _infer_model_input_device(self._model)
            if device is not None:
                input_ids = input_ids.unsqueeze(0).to(device=device)
                attention_mask = attention_mask.unsqueeze(0).to(device=device)
            else:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
            if pixel_values is not None:
                pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
            generation_kwargs = dict(self.generation_kwargs)
            generation_kwargs.setdefault("eos_token_id", getattr(self._model.generation_config, "eos_token_id", None))
            generation_kwargs.setdefault("pad_token_id", text_tokenizer.pad_token_id)
            generation_kwargs.setdefault("use_cache", True)
            with torch_inference_context(torch):
                output_ids = self._model.generate(
                    input_ids,
                    pixel_values=[pixel_values],
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )[0]
            return "{" + text_tokenizer.decode(output_ids, skip_special_tokens=True)
        finally:
            for image in images:
                image.close()

    def _live_generate(self, record: LLMInputRecord, messages: list[dict[str, str]]) -> str:
        if self.preset.family == "minicpm_v":
            return self._minicpm_live_generate(record, messages)
        if self.preset.family == "ovis2":
            return self._ovis_live_generate(record, messages)
        return super()._live_generate(record, messages)

    def _live_generate_batch(
        self,
        records: list[LLMInputRecord],
        messages_batch: list[list[dict[str, str]]],
    ) -> list[str]:
        if self.preset.family in {"minicpm_v", "ovis2", "smolvlm"}:
            return [self._live_generate(record, messages) for record, messages in zip(records, messages_batch, strict=True)]
        return super()._live_generate_batch(records, messages_batch)
