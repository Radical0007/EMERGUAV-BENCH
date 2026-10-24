from __future__ import annotations

from emerguav_bench.llm.adapters.base import BaseAdapter
from emerguav_bench.llm.types import AdapterCapability, ModelPreset

IDEFICS3_PRESETS = {
    "idefics3": ModelPreset(
        model_id="idefics3",
        model_name_or_path="HuggingFaceM4/Idefics3-8B-Llama3",
        family="idefics3",
        capability=AdapterCapability(
            family="idefics3",
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


class Idefics3Adapter(BaseAdapter):
    pass
