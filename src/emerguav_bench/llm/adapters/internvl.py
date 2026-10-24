from __future__ import annotations

from emerguav_bench.llm.adapters.base import BaseAdapter
from emerguav_bench.llm.types import AdapterCapability, ModelPreset

_INTERNVL_CAPABILITY = AdapterCapability(
    family="internvl",
    supports_vision=True,
    supports_multi_image=True,
    supports_few_shot=True,
    preferred_prompt_modes=["zero_shot", "few_shot"],
)

INTERNVL_PRESETS = {
    "internvl3_5_8b": ModelPreset(
        model_id="internvl3_5_8b",
        model_name_or_path="OpenGVLab/InternVL3_5-8B-HF",
        family="internvl",
        capability=_INTERNVL_CAPABILITY,
        processor_class_hint="AutoProcessor",
        model_class_hint="AutoModelForImageTextToText",
        model_load_kwargs={"device_map": "auto", "torch_dtype": "auto"},
        default_generation_kwargs={"max_new_tokens": 2048, "do_sample": False},
        training_priority=8,
    ),
    "internvl3_5": ModelPreset(
        model_id="internvl3_5",
        model_name_or_path="OpenGVLab/InternVL3_5-8B-HF",
        family="internvl",
        capability=_INTERNVL_CAPABILITY,
        processor_class_hint="AutoProcessor",
        model_class_hint="AutoModelForImageTextToText",
        model_load_kwargs={"device_map": "auto", "torch_dtype": "auto"},
        default_generation_kwargs={"max_new_tokens": 2048, "do_sample": False},
        training_priority=4,
    )
}


class InternVLAdapter(BaseAdapter):
    pass
