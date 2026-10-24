from __future__ import annotations

from emerguav_bench.llm.adapters.base import BaseAdapter
from emerguav_bench.llm.types import AdapterCapability, ModelPreset

LLAVA_NEXT_PRESETS = {
    "llava_next": ModelPreset(
        model_id="llava_next",
        model_name_or_path="llava-hf/llava-v1.6-mistral-7b-hf",
        family="llava_next",
        capability=AdapterCapability(
            family="llava_next",
            supports_vision=True,
            supports_multi_image=True,
            supports_few_shot=True,
            preferred_prompt_modes=["zero_shot", "few_shot"],
        ),
        processor_class_hint="AutoProcessor",
        model_class_hint="AutoModelForImageTextToText",
        model_load_kwargs={"device_map": "auto", "torch_dtype": "auto"},
        default_generation_kwargs={"max_new_tokens": 2048, "do_sample": False},
    )
}


class LlavaNextAdapter(BaseAdapter):
    pass
