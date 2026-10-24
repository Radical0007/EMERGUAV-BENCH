from __future__ import annotations

from emerguav_bench.llm.adapters.base import (
    BaseAdapter,
    _move_inputs_to_model_device,
    load_image_for_adapter,
    torch_inference_context,
    trim_generated_prompt_tokens,
)
from emerguav_bench.llm.types import AdapterCapability, LLMInputRecord, ModelPreset

PHI4_MM_PRESETS = {
    "phi4_mm": ModelPreset(
        model_id="phi4_mm",
        model_name_or_path="microsoft/Phi-4-multimodal-instruct",
        family="phi4_mm",
        capability=AdapterCapability(
            family="phi4_mm",
            supports_vision=True,
            supports_multi_image=True,
            supports_few_shot=True,
            preferred_prompt_modes=["zero_shot", "few_shot"],
        ),
        processor_class_hint="AutoProcessor",
        model_class_hint="AutoModelForCausalLM",
        model_load_kwargs={"device_map": "auto", "torch_dtype": "auto", "attn_implementation": "sdpa"},
        default_generation_kwargs={"max_new_tokens": 2048, "do_sample": False},
    )
}


class Phi4MMAdapter(BaseAdapter):
    @staticmethod
    def _normalize_processor_inputs(inputs: dict[str, object]) -> dict[str, object]:
        normalized = dict(inputs)
        if "input_image_embeds" in normalized and "image_pixel_values" not in normalized:
            normalized["image_pixel_values"] = normalized.pop("input_image_embeds")
        if "input_audio_embeds" in normalized and "audio_input_features" not in normalized:
            normalized["audio_input_features"] = normalized.pop("input_audio_embeds")
        normalized.pop("input_mode", None)
        return normalized

    @staticmethod
    def _chat_messages_with_image_placeholders(
        messages: list[dict[str, str]],
        image_count: int,
    ) -> list[dict[str, str]]:
        placeholders = "".join(f"<|image_{index + 1}|>" for index in range(image_count))
        last_user_index = max((index for index, message in enumerate(messages) if message.get("role") == "user"), default=None)
        chat_messages: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            content = str(message.get("content", ""))
            if index == last_user_index:
                content = f"{placeholders}{content}"
            chat_messages.append({"role": message.get("role", "user"), "content": content})
        return chat_messages

    @staticmethod
    def _manual_prompt(chat_messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for message in chat_messages:
            role = message.get("role", "user")
            if role not in {"system", "user", "assistant"}:
                role = "user"
            parts.append(f"<|{role}|>{message.get('content', '')}<|end|>")
        parts.append("<|assistant|>")
        return "".join(parts)

    def _live_generate(self, record: LLMInputRecord, messages: list[dict[str, str]]) -> str:  # pragma: no cover
        import torch

        if self._processor is None or self._model is None:
            raise RuntimeError("Adapter must be loaded before Phi-4-MM live generation")

        images = [load_image_for_adapter(image_input.path, role=image_input.role) for image_input in record.image_inputs]
        try:
            chat_messages = self._chat_messages_with_image_placeholders(messages, len(images))
            tokenizer = getattr(self._processor, "tokenizer", None)
            if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
                if prompt.endswith("<|endoftext|>"):
                    prompt = prompt.removesuffix("<|endoftext|>")
            else:
                prompt = self._manual_prompt(chat_messages)
            prompt += "{"
            image_payload = images[0] if len(images) == 1 else images
            inputs = self._processor(text=prompt, images=image_payload, return_tensors="pt")
            inputs = self._normalize_processor_inputs(inputs)
            inputs = _move_inputs_to_model_device(inputs, self._model)
            generation_kwargs = self._generation_kwargs_for_generate(tokenizer)
            with torch_inference_context(torch):
                outputs = self._model.generate(**inputs, **generation_kwargs)
            outputs = trim_generated_prompt_tokens(outputs, inputs)
            decoded = self._processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return "{" + decoded[0]
        finally:
            for image in images:
                image.close()

    def _live_generate_batch(
        self,
        records: list[LLMInputRecord],
        messages_batch: list[list[dict[str, str]]],
    ) -> list[str]:  # pragma: no cover
        return [self._live_generate(record, messages) for record, messages in zip(records, messages_batch, strict=True)]
