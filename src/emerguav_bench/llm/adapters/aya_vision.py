from __future__ import annotations

from emerguav_bench.llm.adapters.base import BaseAdapter
from emerguav_bench.llm.types import AdapterCapability, ModelPreset

AYA_VISION_PRESETS = {
    "aya_vision": ModelPreset(
        model_id="aya_vision",
        model_name_or_path="CohereLabs/aya-vision-8b",
        family="aya_vision",
        capability=AdapterCapability(
            family="aya_vision",
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


class AyaVisionAdapter(BaseAdapter):
    pass
