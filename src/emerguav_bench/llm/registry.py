from __future__ import annotations

from typing import Any

from emerguav_bench.llm.adapters.aya_vision import AYA_VISION_PRESETS, AyaVisionAdapter
from emerguav_bench.llm.adapters.edge_vlm import EDGE_VLM_PRESETS, EdgeVLMAdapter
from emerguav_bench.llm.adapters.idefics3 import IDEFICS3_PRESETS, Idefics3Adapter
from emerguav_bench.llm.adapters.internvl import INTERNVL_PRESETS, InternVLAdapter
from emerguav_bench.llm.adapters.llava_next import LLAVA_NEXT_PRESETS, LlavaNextAdapter
from emerguav_bench.llm.adapters.phi4_mm import PHI4_MM_PRESETS, Phi4MMAdapter
from emerguav_bench.llm.adapters.qwen_family import (
    QWEN_FAMILY_PRESETS, QWEN_IMPLEMENTATION_ORDER, QwenFamilyAdapter,
)
from emerguav_bench.llm.types import ModelPreset

MODEL_PRESETS: dict[str, ModelPreset] = {
    **QWEN_FAMILY_PRESETS,
    **INTERNVL_PRESETS,
    **EDGE_VLM_PRESETS,
    **PHI4_MM_PRESETS,
    **AYA_VISION_PRESETS,
    **LLAVA_NEXT_PRESETS,
    **IDEFICS3_PRESETS,
}

MODEL_TO_ADAPTER = {
    **{model_id: QwenFamilyAdapter for model_id in QWEN_FAMILY_PRESETS},
    **{model_id: InternVLAdapter for model_id in INTERNVL_PRESETS},
    **{model_id: EdgeVLMAdapter for model_id in EDGE_VLM_PRESETS},
    "qwen3_5": QwenFamilyAdapter,
    "qwen3_vl": QwenFamilyAdapter,
    "qwen2_5_vl": QwenFamilyAdapter,
    "internvl3_5": InternVLAdapter,
    "phi4_mm": Phi4MMAdapter,
    "aya_vision": AyaVisionAdapter,
    "llava_next": LlavaNextAdapter,
    "idefics3": Idefics3Adapter,
}


def create_adapter(
    model_id: str,
    *,
    runtime: str = "mock",
    generation_kwargs: dict[str, Any] | None = None,
    precision: str | None = None,
):
    try:
        preset = MODEL_PRESETS[model_id]
    except KeyError as exc:
        raise KeyError(f"Unsupported model_id: {model_id}") from exc
    adapter_class = MODEL_TO_ADAPTER[model_id]
    return adapter_class(preset, runtime=runtime, generation_kwargs=generation_kwargs, precision=precision)


def describe_model_registry() -> dict[str, Any]:
    return {
        "models": {model_id: preset.to_dict() for model_id, preset in MODEL_PRESETS.items()},
        "qwen_default_priority": list(QWEN_IMPLEMENTATION_ORDER),
    }
