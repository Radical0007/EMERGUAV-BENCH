import contextlib
import json
import os
import shutil
import subprocess
import types
import sys
import unittest
import warnings
from pathlib import Path
from PIL import ImageChops
from typing import Any
from unittest.mock import patch
from uuid import uuid4
import torch

from emerguav_bench.llm.adapters.base import (
    BaseAdapter,
    _apply_chat_template_with_prefill,
    apply_processor_memory_limits,
    build_multimodal_chat_messages,
    live_vision_model_class_hints,
    load_image_for_adapter,
    resolve_existing_local_path,
    trim_generated_prompt_tokens,
)
from emerguav_bench.llm.adapters.phi4_mm import Phi4MMAdapter
from emerguav_bench.llm.exporters import build_llm_input_manifest, build_preference_records, build_sft_records
from emerguav_bench.llm.normalizer import normalize_model_response
from emerguav_bench.llm.prompts import build_zero_shot_messages, render_output_contract, serialize_output_json
from emerguav_bench.llm.registry import MODEL_PRESETS, create_adapter, describe_model_registry
from emerguav_bench.llm.runner import LLMInferenceRunner, _stable_generation_seed, save_predictions_tree
from emerguav_bench.llm.training import PreferenceTrainingConfig, PreferenceTrainingWrapper, SFTTrainingConfig, SFTTrainingWrapper
from emerguav_bench.llm.types import AdapterCapability, ModelPreset, ensure_llm_input_record
from scripts.build_samples import build_seed_samples_from_manifest
from scripts.evaluate_predictions import evaluate_predictions
from scripts.run_model_suite import (
    OOM_SKIP_RETURN_CODE,
    completed_prediction_file,
    format_failure_summary,
    run_model_suite,
)
from emerguav_bench.uavscenes.manifest import make_placeholder_sequence


def make_workspace_temp_dir(prefix: str) -> Path:
    base_dir = Path.cwd() / ".test_tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = base_dir / f"{prefix}_{uuid4().hex}"
    temp_dir.mkdir()
    return temp_dir


def make_sample(*, sample_id: str, sequence_id: str, frame_id: str, rgb_suffix: str, risk_level: str = "high", policy_family: str = "Government / Critical Facility Sensitivity") -> dict:
    action_token = "Hover_and_Alert" if risk_level in {"high", "critical"} else "Proceed_Slow"
    return {
        "sample_id": sample_id,
        "source_domain_bucket": "airport_like",
        "protocol_context": "Airport Perimeter Monitoring",
        "application_theme": "Hyper-Dense Urban Airspace",
        "policy_family": policy_family,
        "sequence_id": sequence_id,
        "frame_id": frame_id,
        "rgb_frame_path": f"placeholder://rgb/{sequence_id}/{rgb_suffix}.jpg",
        "lidar_frame_path": f"placeholder://lidar/{sequence_id}/{rgb_suffix}.bin",
        "pose_6dof": {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "orientation": {"roll": 0.0, "pitch": 0.1, "yaw": 0.2},
        },
        "map_id": f"map::{sequence_id}",
        "degradation_labels": ["dust_veil"],
        "degradation_severity": "high",
        "observation_severity_code": "SV4",
        "language_entropy_labels": ["typo_omission_abbreviation"],
        "language_entropy_level": "EL3",
        "clean_instruction": "Descend cautiously to 30m and avoid NFZ edge.",
        "noisy_instruction": "descnd cautoisly 30m avoid nfz edg",
        "mission_constraints": ["avoid_nfz", "slow_descent"],
        "observation_protocol_gt": {
            "protocol_context_profile": "Airport Perimeter Monitoring",
            "primary_degradation": "dust_veil",
            "usability": "low",
            "trusted_modality": "lidar",
            "risk_level": risk_level,
            "risk_zone_label": "nfz_edge",
            "viewpoint_summary": "restricted_viewpoint",
            "cause_label": "airborne_visibility_loss",
        },
        "intent_extraction_gt": {
            "cleaned_instruction": "descend cautiously to 30m and avoid NFZ edge",
            "target_object": None,
            "constraints": ["avoid_nfz", "slow_descent"],
            "target_goal": "monitor_airport_perimeter_safely",
        },
        "action_token_gt": action_token,
        "acceptable_safe_alternatives": ["Hover", "Reobserve"],
        "forbidden_actions": ["Descend_Cautiously"],
        "risk_level_gt": risk_level,
        "collaboration_gt": {"required": True, "request_type": "traffic_check", "trusted_modality": "lidar"},
        "paper_task_gt": {
            "T1": "Airport Perimeter Monitoring",
            "T2": "nfz_edge",
            "T3": "restricted_viewpoint",
            "T4": "dust_veil",
            "T5": "lidar",
            "T6": {"usability": "low", "cause_label": "airborne_visibility_loss"},
            "T7": "descend cautiously to 30m and avoid NFZ edge",
            "T8": ["avoid_nfz", "slow_descent"],
            "T9": "monitor_airport_perimeter_safely",
            "T10": action_token,
            "T11": "hold_position" if action_token == "Hover_and_Alert" else "continue_execution",
            "T12": {"required": True, "request_type": "traffic_check", "trusted_modality": "lidar"},
        },
        "geometry_anchor": {"waypoint": [1.0, 2.0, 3.0]},
        "protocol_metadata": {"scene_family": "HKairport", "overlay_profiles": ["nfz_edge_caution"]},
    }


class RegistryAndExporterTests(unittest.TestCase):
    def test_registry_contains_expected_model_ids(self) -> None:
        required = {
            "qwen3_5",
            "qwen3_vl",
            "qwen2_5_vl",
            "internvl3_5",
            "phi4_mm",
            "aya_vision",
            "llava_next",
            "idefics3",
            "qwen3_vl_2b",
            "qwen3_vl_4b",
            "qwen3_vl_8b",
            "qwen3_vl_32b",
            "qwen3_vl_30b_a3b",
            "qwen3_5_0_8b",
            "qwen3_5_2b",
            "qwen3_5_4b",
            "qwen3_5_9b",
            "qwen3_5_27b",
            "qwen3_5_27b_fp8",
            "qwen3_5_35b_a3b",
            "qwen3_5_35b_a3b_fp8",
            "qwen3_6_27b",
            "qwen3_6_27b_fp8",
            "qwen3_6_35b_a3b",
            "gemma4_e2b_it",
            "gemma4_e4b_it",
            "gemma4_26b_a4b_it",
            "gemma4_31b_it",
            "internvl3_5_8b",
            "minicpm_v_4_6",
            "minicpm_v_4_5",
            "glm4_6v_flash",
            "glm4_6v",
            "smolvlm2_2_2b",
            "ovis2_8b",
            "ovis2_16b",
        }
        self.assertTrue(required.issubset(set(MODEL_PRESETS)))
        self.assertEqual(
            describe_model_registry()["qwen_default_priority"],
            ["qwen3_5_4b", "qwen3_5_2b", "qwen3_vl_4b", "qwen3_vl_2b", "qwen2_5_vl"],
        )

    def test_model_presets_include_live_load_defaults(self) -> None:
        preset = MODEL_PRESETS["qwen3_5_4b"]

        self.assertEqual(preset.model_name_or_path, "Qwen/Qwen3.5-4B")
        self.assertEqual(preset.model_load_kwargs["device_map"], "auto")
        self.assertEqual(preset.model_load_kwargs["torch_dtype"], "auto")
        self.assertEqual(preset.processor_load_kwargs, {})
        self.assertEqual(MODEL_PRESETS["qwen3_6_27b"].model_class_hint, "AutoModelForImageTextToText")
        self.assertEqual(MODEL_PRESETS["gemma4_e4b_it"].model_class_hint, "AutoModelForImageTextToText")
        self.assertEqual(MODEL_PRESETS["phi4_mm"].model_class_hint, "AutoModelForCausalLM")
        self.assertEqual(MODEL_PRESETS["idefics3"].model_class_hint, "AutoModelForImageTextToText")
        self.assertEqual(MODEL_PRESETS["glm4_6v_flash"].model_class_hint, "AutoModelForImageTextToText")
        self.assertEqual(MODEL_PRESETS["glm4_6v"].model_class_hint, "AutoModelForImageTextToText")
        self.assertEqual(MODEL_PRESETS["minicpm_v_4_5"].model_name_or_path, "openbmb/MiniCPM-V-4_5")
        self.assertEqual(MODEL_PRESETS["minicpm_v_4_5"].model_class_hint, "AutoModel")
        self.assertEqual(MODEL_PRESETS["ovis2_8b"].model_class_hint, "AutoModelForCausalLM")
        for model_id in ("gemma4_e2b_it", "gemma4_e4b_it", "gemma4_26b_a4b_it", "gemma4_31b_it"):
            self.assertEqual(MODEL_PRESETS[model_id].model_load_kwargs["attn_implementation"], "sdpa")
        self.assertEqual(MODEL_PRESETS["minicpm_v_4_5"].model_load_kwargs["attn_implementation"], "sdpa")
        self.assertEqual(MODEL_PRESETS["phi4_mm"].model_load_kwargs["attn_implementation"], "sdpa")
        self.assertEqual(MODEL_PRESETS["ovis2_8b"].model_load_kwargs["multimodal_max_length"], 32768)

    def test_phi4_mm_live_loader_uses_builtin_phi4multimodal_config(self) -> None:
        captured: dict[str, Any] = {}

        class FakeRemoteConfig:
            def to_dict(self):
                return {"audio_processor": {"config": {"nemo_conv_settings": {"subsampling": "striding_conv1d"}}}}

        class FakeBuiltInConfig:
            def __init__(self):
                self._attn_implementation = "flash_attention_2"

            @classmethod
            def from_dict(cls, payload):
                captured["builtin_config_payload"] = payload
                return cls()

        class FakeProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                captured["processor_args"] = args
                captured["processor_kwargs"] = kwargs
                return cls()

        class FakeConfigLoader:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                captured["config_args"] = args
                captured["config_kwargs"] = kwargs
                return FakeRemoteConfig()

        class AutoModelForCausalLM:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                captured["model_class"] = "AutoModelForCausalLM"
                captured["model_args"] = args
                captured["model_kwargs"] = kwargs
                return cls()

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=FakeProcessor,
            AutoConfig=FakeConfigLoader,
            Phi4MultimodalConfig=FakeBuiltInConfig,
            AutoModelForCausalLM=AutoModelForCausalLM,
        )

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            BaseAdapter(MODEL_PRESETS["phi4_mm"], runtime="live").load()

        self.assertEqual(captured["processor_args"], ("microsoft/Phi-4-multimodal-instruct",))
        self.assertEqual(captured["config_args"], ("microsoft/Phi-4-multimodal-instruct",))
        self.assertEqual(captured["config_kwargs"], {"trust_remote_code": True})
        self.assertEqual(captured["builtin_config_payload"], FakeRemoteConfig().to_dict())
        self.assertEqual(captured["model_class"], "AutoModelForCausalLM")
        self.assertEqual(captured["model_kwargs"]["config"]._attn_implementation, "sdpa")
        self.assertTrue(captured["model_kwargs"]["ignore_mismatched_sizes"])
        self.assertFalse(captured["model_kwargs"]["trust_remote_code"])
        self.assertNotIn("attn_implementation", captured["model_kwargs"])

    def test_phi4_mm_live_generate_renames_processor_outputs_for_builtin_model(self) -> None:
        captured: dict[str, Any] = {}

        class DummyImage:
            def close(self) -> None:
                return None

        class FakeProcessor:
            tokenizer = types.SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "prompt")

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def __call__(self, *args, **kwargs):
                del args
                del kwargs
                return {
                    "input_ids": torch.tensor([[1, 2]]),
                    "input_image_embeds": torch.tensor([1.0]),
                    "image_sizes": torch.tensor([[3, 4]]),
                    "image_attention_mask": torch.tensor([[[1]]]),
                    "input_audio_embeds": torch.tensor([2.0]),
                    "audio_embed_sizes": torch.tensor([0]),
                    "audio_attention_mask": None,
                    "attention_mask": torch.tensor([[1, 1]]),
                    "input_mode": torch.tensor([1]),
                }

            def batch_decode(self, outputs, **kwargs):
                del kwargs
                captured["decoded_outputs"] = outputs
                return ["decoded"]

        class FakeModel:
            device = torch.device("cpu")

            def generate(self, **kwargs):
                captured["generate_kwargs"] = kwargs
                return torch.tensor([[1, 2]])

        record = ensure_llm_input_record(
            build_llm_input_manifest([make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001")])[0]
        )
        adapter = Phi4MMAdapter(MODEL_PRESETS["phi4_mm"], runtime="live")
        adapter._processor = FakeProcessor()
        adapter._model = FakeModel()

        with patch("emerguav_bench.llm.adapters.phi4_mm.load_image_for_adapter", return_value=DummyImage()):
            response = adapter._live_generate(record, record.prompt_messages)

        self.assertEqual(response, "{decoded")
        self.assertIn("image_pixel_values", captured["generate_kwargs"])
        self.assertIn("audio_input_features", captured["generate_kwargs"])
        self.assertNotIn("input_image_embeds", captured["generate_kwargs"])
        self.assertNotIn("input_audio_embeds", captured["generate_kwargs"])
        self.assertNotIn("input_mode", captured["generate_kwargs"])

    def test_model_suite_config_uses_registered_models(self) -> None:
        suite_path = Path("configs/model_suites.json")
        suites = json.loads(suite_path.read_text(encoding="utf-8"))

        self.assertIn("core", suites)
        self.assertIn("extended", suites)
        self.assertIn("qwen3_5_9b", suites["core"]["model_ids"])
        self.assertIn("gemma4_e4b_it", suites["core"]["model_ids"])
        self.assertIn("minicpm_v_4_5", suites["core"]["model_ids"])
        self.assertNotIn("minicpm_v_4_6", suites["core"]["model_ids"])
        self.assertIn("qwen3_5_27b_fp8", suites["extended"]["model_ids"])
        self.assertIn("qwen3_6_27b", suites["extended"]["model_ids"])
        self.assertIn("gemma4_31b_it", suites["extended"]["model_ids"])
        self.assertNotIn("qwen3_vl_235b_a22b", suites["extended"]["model_ids"])
        for suite in suites.values():
            for model_id in suite["model_ids"]:
                self.assertIn(model_id, MODEL_PRESETS)

    def test_live_core_model_dependencies_include_remote_processor_requirements(self) -> None:
        pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")

        for dependency in ("num2words", "scipy", "backoff"):
            self.assertIn(f'"{dependency}>=', pyproject_text)

    def test_each_adapter_supports_mock_generation(self) -> None:
        manifest = build_llm_input_manifest([make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001")])
        record = manifest[0]
        for model_id in MODEL_PRESETS:
            adapter = create_adapter(model_id, runtime="mock")
            response = adapter.generate(record, record["prompt_messages"])
            normalized = normalize_model_response(response)
            self.assertEqual(normalized["Action_Output"]["action_token"], record["ground_truth_output"]["Action_Output"]["action_token"])

    def test_prompt_frames_protocol_context_as_decision_condition(self) -> None:
        manifest = build_llm_input_manifest(
            [
                make_sample(
                    sample_id="SEQ_AIRPORT_SECRET::0001",
                    sequence_id="SEQ_AIRPORT_SECRET",
                    frame_id="0001",
                    rgb_suffix="0001",
                )
            ]
        )
        record = ensure_llm_input_record(manifest[0])

        messages = build_zero_shot_messages(record)
        prompt_text = "\n".join(message["content"] for message in messages)
        contract_text = render_output_contract()

        self.assertIn("protocol-conditioned UAV decision model", prompt_text)
        self.assertIn("Protocol, policy, and protected-zone semantics must come from the provided mission context", prompt_text)
        self.assertIn("Protocol / policy context", prompt_text)
        self.assertIn("Observation use", prompt_text)
        self.assertIn("provided protocol context", contract_text)
        self.assertIn("Use pixels to assess degradation, visibility, viewpoint usability, and immediate safety cues", prompt_text)
        self.assertIn("Image inputs are opaque handles", prompt_text)
        self.assertIn("image_1", prompt_text)
        self.assertNotIn("placeholder://", prompt_text)
        self.assertNotIn("SEQ_AIRPORT_SECRET::0001", prompt_text)
        self.assertNotIn("SEQ_AIRPORT_SECRET", prompt_text)
        self.assertNotIn("frame_id=0001", prompt_text)

    def test_placeholder_image_loader_does_not_render_semantic_path_text(self) -> None:
        first = load_image_for_adapter("placeholder://context/SEQ_AIRPORT_SECRET/0001/previous_frame.jpg", role="previous_frame")
        second = load_image_for_adapter("placeholder://context/SEQ_NFZ_SECRET/9999/previous_frame.jpg", role="previous_frame")

        self.assertIsNone(ImageChops.difference(first, second).getbbox())

    def test_live_loader_reports_underlying_class_errors(self) -> None:
        class FailingProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise RuntimeError("processor failed")

        class FailingModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise RuntimeError("model failed")

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=FailingProcessor,
            AutoModelForImageTextToText=FailingModel,
        )
        preset = ModelPreset(
            model_id="broken_live",
            model_name_or_path="example/broken",
            family="broken",
            capability=AdapterCapability(
                family="broken",
                supports_vision=True,
                supports_multi_image=True,
                supports_few_shot=False,
                preferred_prompt_modes=["zero_shot"],
            ),
            processor_class_hint="AutoProcessor",
            model_class_hint="AutoModelForImageTextToText",
        )

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            with self.assertRaisesRegex(RuntimeError, "processor failed.*AutoModelForImageTextToText.*model failed"):
                BaseAdapter(preset, runtime="live").load()

    def test_live_vision_loader_respects_explicit_auto_hint_order(self) -> None:
        self.assertEqual(live_vision_model_class_hints("AutoModelForCausalLM")[0], "AutoModelForCausalLM")
        self.assertEqual(live_vision_model_class_hints("AutoModelForImageTextToText")[0], "AutoModelForImageTextToText")
        self.assertIn("AutoModel", live_vision_model_class_hints("AutoModelForCausalLM"))

        class FakeProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                image_parts: list[dict[str, Any]] = []
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        image_parts = [part for part in msg["content"] if part.get("type") == "image"]
                        break
                if not image_parts:
                    raise AssertionError("multimodal chat template did not receive image content parts")
                self.image_part_count = len(image_parts)
                if not kwargs.get("tokenize", True):
                    return "<|im_start|>assistant\n"
                return {
                    "input_ids": [[1]],
                    "pixel_values": "pixels",
                    "image_grid_thw": "grid",
                    "mm_token_type_ids": "mm-types",
                }

            def __call__(self, *args, **kwargs):
                return {
                    "input_ids": [[1]],
                    "pixel_values": "pixels",
                    "image_grid_thw": "grid",
                    "mm_token_type_ids": "mm-types",
                }

            def batch_decode(self, outputs, skip_special_tokens=True):
                if outputs != [[2]]:
                    raise AssertionError(f"prompt tokens were not trimmed: {outputs}")
                return ["decoded"]

        class CausalLM:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def generate(self, **kwargs):
                raise ValueError("unused image kwargs reached causal lm")

        class ImageTextToText:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def generate(self, **kwargs):
                if "pixel_values" not in kwargs or "image_grid_thw" not in kwargs:
                    raise AssertionError("image kwargs were dropped")
                return [[1, 2]]

        class FakeInferenceContext:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return None

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=FakeProcessor,
            AutoModelForImageTextToText=ImageTextToText,
            AutoModelForCausalLM=CausalLM,
        )
        fake_torch = types.SimpleNamespace(inference_mode=FakeInferenceContext)
        preset = ModelPreset(
            model_id="image_text_preferred",
            model_name_or_path="example/image-text",
            family="example",
            capability=AdapterCapability(
                family="example",
                supports_vision=True,
                supports_multi_image=True,
                supports_few_shot=False,
                preferred_prompt_modes=["zero_shot"],
            ),
            processor_class_hint="AutoProcessor",
            model_class_hint="AutoModelForImageTextToText",
        )
        record = build_llm_input_manifest([make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001")])[0]

        with patch.dict(sys.modules, {"transformers": fake_transformers, "torch": fake_torch}):
            response = BaseAdapter(preset, runtime="live").generate(record, record["prompt_messages"])

        self.assertEqual(response, "{decoded")

    def test_live_vision_loader_uses_causal_lm_when_registry_requests_it(self) -> None:
        loaded_classes: list[str] = []

        class FakeProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        class ImageTextToText:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                loaded_classes.append("AutoModelForImageTextToText")
                raise AssertionError("image-text auto should not be tried before the explicit causal hint")

        class CausalLM:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                loaded_classes.append("AutoModelForCausalLM")
                return cls()

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=FakeProcessor,
            AutoModelForImageTextToText=ImageTextToText,
            AutoModelForCausalLM=CausalLM,
        )
        preset = ModelPreset(
            model_id="causal_preferred",
            model_name_or_path="example/causal",
            family="example",
            capability=AdapterCapability(
                family="example",
                supports_vision=True,
                supports_multi_image=True,
                supports_few_shot=False,
                preferred_prompt_modes=["zero_shot"],
            ),
            processor_class_hint="AutoProcessor",
            model_class_hint="AutoModelForCausalLM",
        )

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            BaseAdapter(preset, runtime="live").load()

        self.assertEqual(loaded_classes, ["AutoModelForCausalLM"])

    def test_live_loader_accepts_bf16_precision_override(self) -> None:
        captured_model_kwargs: dict[str, Any] = {}

        class FakeProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        class FakeModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                captured_model_kwargs.update(kwargs)
                return cls()

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=FakeProcessor,
            AutoModelForImageTextToText=FakeModel,
        )
        fake_torch = types.SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32")
        preset = ModelPreset(
            model_id="bf16_live",
            model_name_or_path="example/bf16",
            family="example",
            capability=AdapterCapability(
                family="example",
                supports_vision=True,
                supports_multi_image=True,
                supports_few_shot=False,
                preferred_prompt_modes=["zero_shot"],
            ),
            processor_class_hint="AutoProcessor",
            model_class_hint="AutoModelForImageTextToText",
            model_load_kwargs={"device_map": "auto", "torch_dtype": "auto"},
        )

        with patch.dict(sys.modules, {"transformers": fake_transformers, "torch": fake_torch}):
            BaseAdapter(preset, runtime="live", precision="bf16").load()

        self.assertEqual(captured_model_kwargs["torch_dtype"], "bf16")

    def test_live_loader_uses_dtype_keyword_for_modern_transformers(self) -> None:
        captured_model_kwargs: dict[str, Any] = {}

        class FakeProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        class FakeModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                captured_model_kwargs.update(kwargs)
                return cls()

        fake_transformers = types.SimpleNamespace(
            __version__="4.56.1",
            AutoProcessor=FakeProcessor,
            AutoModelForImageTextToText=FakeModel,
        )
        fake_torch = types.SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32")
        preset = ModelPreset(
            model_id="bf16_live",
            model_name_or_path="example/bf16",
            family="example",
            capability=AdapterCapability(
                family="example",
                supports_vision=True,
                supports_multi_image=True,
                supports_few_shot=False,
                preferred_prompt_modes=["zero_shot"],
            ),
            processor_class_hint="AutoProcessor",
            model_class_hint="AutoModelForImageTextToText",
            model_load_kwargs={"device_map": "auto", "torch_dtype": "auto"},
        )

        with patch.dict(sys.modules, {"transformers": fake_transformers, "torch": fake_torch}):
            BaseAdapter(preset, runtime="live", precision="bf16").load()

        self.assertNotIn("torch_dtype", captured_model_kwargs)
        self.assertEqual(captured_model_kwargs["dtype"], "bf16")

    def test_live_loader_does_not_override_preset_attention_backend(self) -> None:
        captured_model_kwargs: dict[str, Any] = {}

        class FakeProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        class FakeModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                captured_model_kwargs.update(kwargs)
                return cls()

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=FakeProcessor,
            AutoModelForImageTextToText=FakeModel,
        )
        preset = ModelPreset(
            model_id="sdpa_live",
            model_name_or_path="example/sdpa",
            family="example",
            capability=AdapterCapability(
                family="example",
                supports_vision=True,
                supports_multi_image=True,
                supports_few_shot=False,
                preferred_prompt_modes=["zero_shot"],
            ),
            processor_class_hint="AutoProcessor",
            model_class_hint="AutoModelForImageTextToText",
            model_load_kwargs={"device_map": "auto", "attn_implementation": "sdpa"},
        )

        with patch.dict(sys.modules, {"transformers": fake_transformers, "flash_attn": types.SimpleNamespace()}):
            BaseAdapter(preset, runtime="live").load()

        self.assertEqual(captured_model_kwargs["attn_implementation"], "sdpa")

    def test_minicpm_live_adapter_uses_chat_api(self) -> None:
        captured: dict[str, Any] = {}

        class FakeInferenceContext:
            def __enter__(self):
                captured["inference_context_active"] = True

            def __exit__(self, exc_type, exc, tb):
                captured["inference_context_active"] = False

        class FakeTokenizer:
            pad_token_id = None
            eos_token_id = 99
            padding_side = "right"

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                captured["tokenizer_args"] = args
                captured["tokenizer_kwargs"] = kwargs
                return cls()

            def convert_tokens_to_ids(self, token):
                return {
                    "<image>": 11,
                    "</image>": 12,
                    "<slice>": 13,
                    "</slice>": 14,
                    "<image_id>": 15,
                    "</image_id>": 16,
                    "\n": 17,
                }.get(token, 0)

        class FakeProcessor:
            def __init__(self):
                self.tokenizer = FakeTokenizer()

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                captured["processor_args"] = args
                captured["processor_kwargs"] = kwargs
                return cls()

        class FakeModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                captured["model_args"] = args
                captured["model_kwargs"] = kwargs
                return cls()

            def eval(self):
                return self

            def chat(self, **kwargs):
                if not captured.get("inference_context_active", False):
                    raise AssertionError("MiniCPM chat must run inside torch inference context")
                captured["chat_kwargs"] = kwargs
                return '{"Action_Output": {"action_token": "Hover"}}'

        fake_transformers = types.SimpleNamespace(AutoTokenizer=FakeTokenizer, AutoProcessor=FakeProcessor, AutoModel=FakeModel)
        fake_torch = types.SimpleNamespace(inference_mode=FakeInferenceContext)
        record = build_llm_input_manifest([make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001")])[0]

        with patch.dict(sys.modules, {"transformers": fake_transformers, "torch": fake_torch}):
            response = create_adapter("minicpm_v_4_5", runtime="live").generate(record, record["prompt_messages"])

        self.assertEqual(captured["model_args"], ("openbmb/MiniCPM-V-4_5",))
        self.assertEqual(captured["model_kwargs"]["attn_implementation"], "sdpa")
        self.assertIs(captured["chat_kwargs"]["image"], None)
        self.assertFalse(captured["chat_kwargs"]["sampling"])
        self.assertFalse(captured["chat_kwargs"]["enable_thinking"])
        msgs = captured["chat_kwargs"]["msgs"]
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertTrue(hasattr(msgs[-1]["content"][0], "close"))
        self.assertIn("Return ONLY one JSON object", msgs[-1]["content"][-1])
        self.assertEqual(response, '{"Action_Output": {"action_token": "Hover"}}')

    def test_minicpm_live_loader_patches_missing_tied_weight_keys_before_auto_device_map(self) -> None:
        captured: dict[str, Any] = {}

        class FakeConfig:
            auto_map = {"AutoModel": "modeling_minicpmv.MiniCPMV"}

        class FakeAutoConfig:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                captured["config_args"] = args
                captured["config_kwargs"] = kwargs
                return FakeConfig()

        class RemoteMiniCPM:
            def __init__(self, config):
                self.config = config
                self.post_init_called = False

            def post_init(self):
                self.post_init_called = True
                self.all_tied_weights_keys = []

            def eval(self):
                return self

        class FakeDynamicModuleUtils:
            @staticmethod
            def get_class_from_dynamic_module(*args, **kwargs):
                captured["dynamic_class_args"] = args
                captured["dynamic_class_kwargs"] = kwargs
                return RemoteMiniCPM

        class FakeTokenizer:
            pad_token_id = None
            eos_token_id = 99
            padding_side = "right"

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def convert_tokens_to_ids(self, token):
                return {
                    "<image>": 11,
                    "</image>": 12,
                    "<slice>": 13,
                    "</slice>": 14,
                    "<image_id>": 15,
                    "</image_id>": 16,
                    "\n": 17,
                }.get(token, 0)

        class FakeProcessor:
            def __init__(self):
                self.tokenizer = FakeTokenizer()

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        class FakeAutoModel:
            @classmethod
            def register(cls, *args, **kwargs):
                captured["registered"] = (args, kwargs)

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                model = RemoteMiniCPM(kwargs.get("config", FakeConfig()))
                if kwargs.get("device_map") == "auto":
                    len(model.all_tied_weights_keys)
                captured["model_kwargs"] = kwargs
                captured["model"] = model
                return model

        fake_transformers = types.SimpleNamespace(
            AutoConfig=FakeAutoConfig,
            AutoTokenizer=FakeTokenizer,
            AutoProcessor=FakeProcessor,
            AutoModel=FakeAutoModel,
            dynamic_module_utils=FakeDynamicModuleUtils,
        )

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            create_adapter("minicpm_v_4_5", runtime="live").load()

        self.assertEqual(captured["config_args"], ("openbmb/MiniCPM-V-4_5",))
        self.assertTrue(captured["config_kwargs"]["trust_remote_code"])
        self.assertEqual(captured["dynamic_class_args"], ("modeling_minicpmv.MiniCPMV", "openbmb/MiniCPM-V-4_5"))
        self.assertTrue(captured["dynamic_class_kwargs"]["trust_remote_code"])
        self.assertIn("config", captured["model_kwargs"])
        self.assertTrue(captured["model"].post_init_called)
        self.assertEqual(captured["model"].all_tied_weights_keys, [])

    def test_minicpm_live_adapter_passes_processor_with_tokenizer_compat_aliases(self) -> None:
        captured: dict[str, Any] = {}

        class FakeInferenceContext:
            def __enter__(self):
                captured["inference_context_active"] = True

            def __exit__(self, exc_type, exc, tb):
                captured["inference_context_active"] = False

        class BareTokenizer:
            pad_token_id = None
            eos_token_id = 2
            bos_token_id = 1
            unk_token_id = 0
            padding_side = "right"

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def convert_tokens_to_ids(self, token):
                return {
                    "<image>": 11,
                    "</image>": 12,
                    "<slice>": 13,
                    "</slice>": 14,
                    "<image_id>": 15,
                    "</image_id>": 16,
                    "\n": 17,
                }[token]

        class FakeProcessor:
            def __init__(self):
                self.tokenizer = BareTokenizer()

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                captured["processor_args"] = args
                captured["processor_kwargs"] = kwargs
                return cls()

        class FakeModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def eval(self):
                return self

            def chat(self, **kwargs):
                if not captured.get("inference_context_active", False):
                    raise AssertionError("MiniCPM chat must run inside torch inference context")
                processor = kwargs["processor"]
                tokenizer = kwargs["tokenizer"]
                captured["processor_tokenizer_attrs"] = {
                    "im_start_id": processor.tokenizer.im_start_id,
                    "im_end_id": processor.tokenizer.im_end_id,
                    "slice_start_id": processor.tokenizer.slice_start_id,
                    "slice_end_id": processor.tokenizer.slice_end_id,
                    "bos_id": processor.tokenizer.bos_id,
                    "eos_id": processor.tokenizer.eos_id,
                }
                captured["chat_tokenizer_attrs"] = {
                    "im_start_id": tokenizer.im_start_id,
                    "slice_start_id": tokenizer.slice_start_id,
                }
                return '{"Action_Output": {"action_token": "Hover"}}'

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=BareTokenizer,
            AutoProcessor=FakeProcessor,
            AutoModel=FakeModel,
        )
        fake_torch = types.SimpleNamespace(inference_mode=FakeInferenceContext)
        record = build_llm_input_manifest([make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001")])[0]

        with patch.dict(sys.modules, {"transformers": fake_transformers, "torch": fake_torch}):
            response = create_adapter("minicpm_v_4_5", runtime="live").generate(record, record["prompt_messages"])

        self.assertEqual(captured["processor_args"], ("openbmb/MiniCPM-V-4_5",))
        self.assertTrue(captured["processor_kwargs"]["trust_remote_code"])
        self.assertEqual(
            captured["processor_tokenizer_attrs"],
            {
                "im_start_id": 11,
                "im_end_id": 12,
                "slice_start_id": 13,
                "slice_end_id": 14,
                "bos_id": 1,
                "eos_id": 2,
            },
        )
        self.assertEqual(captured["chat_tokenizer_attrs"], {"im_start_id": 11, "slice_start_id": 13})
        self.assertEqual(response, '{"Action_Output": {"action_token": "Hover"}}')

    def test_smolvlm_live_adapter_uses_single_sample_generation_without_batch_probe(self) -> None:
        captured: dict[str, Any] = {"chat_template_kwargs": [], "processor_calls": []}

        class FakeInferenceContext:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return None

        class FakeProcessor:
            tokenizer = types.SimpleNamespace(pad_token_id=None, eos_token_id=99, padding_side="right")

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def apply_chat_template(self, messages, **kwargs):
                del messages
                captured["chat_template_kwargs"].append(kwargs)
                if not kwargs.get("processor_kwargs"):
                    warnings.warn("Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`")
                return "prompt"

            def __call__(self, *args, **kwargs):
                del args
                texts = kwargs["text"]
                captured["processor_calls"].append(texts)
                if isinstance(texts, list) and len(texts) > 1:
                    raise AssertionError("SmolVLM2 live adapter should not probe native batch generation")
                return {"input_ids": [[1]], "pixel_values": "pixels"}

            def batch_decode(self, outputs, skip_special_tokens=True):
                del skip_special_tokens
                return [f"decoded-{output[-1]}" for output in outputs]

        class FakeModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def generate(self, **kwargs):
                if "pixel_values" not in kwargs:
                    raise AssertionError("image inputs were not forwarded")
                return [[1, len(captured["processor_calls"])]]

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=FakeProcessor,
            AutoModelForImageTextToText=FakeModel,
        )
        fake_torch = types.SimpleNamespace(inference_mode=FakeInferenceContext)
        records = build_llm_input_manifest(
            [
                make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001"),
                make_sample(sample_id="s2", sequence_id="SEQ_A", frame_id="0002", rgb_suffix="0002"),
            ]
        )
        messages_batch = [record["prompt_messages"] for record in records]

        with patch.dict(sys.modules, {"transformers": fake_transformers, "torch": fake_torch}):
            with warnings.catch_warnings(record=True) as emitted_warnings:
                warnings.simplefilter("always")
                responses = create_adapter("smolvlm2_2_2b", runtime="live").generate_batch(records, messages_batch)

        self.assertEqual(responses, ["{decoded-1", "{decoded-2"])
        self.assertEqual(
            [kwargs["processor_kwargs"] for kwargs in captured["chat_template_kwargs"]],
            [{"num_frames": None, "fps": None}, {"num_frames": None, "fps": None}],
        )
        self.assertEqual(len(captured["processor_calls"]), 2)
        self.assertFalse(
            [
                warning
                for warning in emitted_warnings
                if "rejected native batch generation" in str(warning.message)
            ]
        )
        self.assertFalse(
            [
                warning
                for warning in emitted_warnings
                if "Kwargs passed to `processor.__call__`" in str(warning.message)
            ]
        )

    def test_phi4_mm_prompt_inserts_image_placeholders(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "seen"},
            {"role": "user", "content": "second"},
        ]

        chat_messages = Phi4MMAdapter._chat_messages_with_image_placeholders(messages, image_count=2)
        prompt = Phi4MMAdapter._manual_prompt(chat_messages)

        self.assertEqual(chat_messages[-1]["content"], "<|image_1|><|image_2|>second")
        self.assertNotIn("<|image_1|>first", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>"))

    def test_live_loader_uses_left_padding_for_generation_tokenizer(self) -> None:
        fake_tokenizer = types.SimpleNamespace(pad_token_id=None, eos_token_id=99, padding_side="right")

        class FakeProcessor:
            tokenizer = fake_tokenizer

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        class FakeModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=FakeProcessor,
            AutoModelForImageTextToText=FakeModel,
        )
        preset = ModelPreset(
            model_id="left_pad_live",
            model_name_or_path="example/left-pad",
            family="example",
            capability=AdapterCapability(
                family="example",
                supports_vision=True,
                supports_multi_image=True,
                supports_few_shot=False,
                preferred_prompt_modes=["zero_shot"],
            ),
            processor_class_hint="AutoProcessor",
            model_class_hint="AutoModelForImageTextToText",
        )

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            BaseAdapter(preset, runtime="live").load()

        self.assertEqual(fake_tokenizer.pad_token_id, 99)
        self.assertEqual(fake_tokenizer.padding_side, "left")

    def test_live_batch_generate_passes_pad_token_id_to_transformers_generate(self) -> None:
        captured_generate_kwargs: dict[str, Any] = {}
        fake_tokenizer = types.SimpleNamespace(pad_token_id=None, eos_token_id=123, padding_side="right")

        class FakeProcessor:
            tokenizer = fake_tokenizer

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def apply_chat_template(self, messages, **kwargs):
                return "<|im_start|>assistant\n"

            def __call__(self, *args, **kwargs):
                text_batch = kwargs["text"]
                return {"input_ids": [[1] for _ in text_batch]}

            def batch_decode(self, outputs, skip_special_tokens=True):
                self.decoded_outputs = outputs
                return ["decoded" for _ in outputs]

        class FakeModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def generate(self, **kwargs):
                captured_generate_kwargs.update(kwargs)
                return [[1, 2], [1, 2]]

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=FakeProcessor,
            AutoModelForImageTextToText=FakeModel,
        )
        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        preset = ModelPreset(
            model_id="pad_generate_live",
            model_name_or_path="example/pad-generate",
            family="example",
            capability=AdapterCapability(
                family="example",
                supports_vision=True,
                supports_multi_image=True,
                supports_few_shot=False,
                preferred_prompt_modes=["zero_shot"],
            ),
            processor_class_hint="AutoProcessor",
            model_class_hint="AutoModelForImageTextToText",
        )
        record = build_llm_input_manifest([make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001")])[0]

        with patch.dict(sys.modules, {"transformers": fake_transformers, "torch": fake_torch}):
            responses = BaseAdapter(preset, runtime="live").generate_batch(
                [record, record],
                [record["prompt_messages"], record["prompt_messages"]],
            )

        self.assertEqual(responses, ["{decoded", "{decoded"])
        self.assertEqual(captured_generate_kwargs["pad_token_id"], 123)

    def test_multimodal_chat_messages_insert_image_parts_before_user_text(self) -> None:
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "question"}]
        chat_messages = build_multimodal_chat_messages(messages, images=["img-a", "img-b"])

        self.assertEqual(chat_messages[0]["content"], [{"type": "text", "text": "sys"}])
        self.assertEqual(chat_messages[1]["content"][0], {"type": "image", "image": "img-a"})
        self.assertEqual(chat_messages[1]["content"][1], {"type": "image", "image": "img-b"})
        self.assertEqual(chat_messages[1]["content"][-1], {"type": "text", "text": "question"})

    def test_qwen3_vl_uses_official_sampling_defaults_with_stable_sample_seed(self) -> None:
        qwen3_vl_kwargs = MODEL_PRESETS["qwen3_vl_8b"].default_generation_kwargs
        qwen3_5_kwargs = MODEL_PRESETS["qwen3_5_9b"].default_generation_kwargs

        self.assertTrue(qwen3_vl_kwargs["do_sample"])
        self.assertEqual(qwen3_vl_kwargs["top_k"], 20)
        self.assertEqual(qwen3_vl_kwargs["top_p"], 0.8)
        self.assertEqual(qwen3_vl_kwargs["temperature"], 0.7)
        self.assertTrue(qwen3_vl_kwargs["remove_invalid_values"])
        self.assertTrue(qwen3_vl_kwargs["renormalize_logits"])
        self.assertEqual(MODEL_PRESETS["qwen3_vl_8b"].model_load_kwargs["attn_implementation"], "sdpa")
        self.assertNotIn("attn_implementation", MODEL_PRESETS["qwen3_5_9b"].model_load_kwargs)
        self.assertFalse(qwen3_5_kwargs["do_sample"])
        self.assertEqual(
            _stable_generation_seed(0, ["sample-a"]),
            _stable_generation_seed(0, ["sample-a"]),
        )
        self.assertNotEqual(
            _stable_generation_seed(0, ["sample-a"]),
            _stable_generation_seed(0, ["sample-b"]),
        )

    def test_chat_template_can_generate_without_json_prefill(self) -> None:
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                return "assistant-header"

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return {"input_ids": [[1]]}

        processor = Processor()
        inputs, prefilled = _apply_chat_template_with_prefill(
            processor,
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            images=[],
            fallback_text="hello",
            prefill_json=False,
        )

        self.assertEqual(inputs, {"input_ids": [[1]]})
        self.assertFalse(prefilled)
        self.assertEqual(processor.kwargs["text"], ["assistant-header"])

    def test_generated_token_trimming_keeps_already_trimmed_outputs(self) -> None:
        self.assertEqual(trim_generated_prompt_tokens([[7, 8]], {"input_ids": [[1, 2, 3]]}), [[7, 8]])

    def test_processor_memory_limit_updates_common_image_processor_attrs(self) -> None:
        image_processor = types.SimpleNamespace(
            max_pixels=10,
            max_dynamic_patch=8,
            size=types.SimpleNamespace(shortest_edge=256, longest_edge=8192),
        )
        processor = types.SimpleNamespace(image_processor=image_processor)

        apply_processor_memory_limits(processor, max_pixels=4096)

        self.assertEqual(processor.max_pixels, 4096)
        self.assertEqual(image_processor.max_pixels, 4096)
        self.assertEqual(image_processor.size.shortest_edge, 256)
        self.assertEqual(image_processor.size.longest_edge, 4096)

    def test_processor_memory_limit_updates_mapping_size(self) -> None:
        image_processor = types.SimpleNamespace(size={"shortest_edge": 256, "longest_edge": 8192})
        processor = types.SimpleNamespace(image_processor=image_processor)

        apply_processor_memory_limits(processor, max_pixels=4096)

        self.assertEqual(image_processor.size, {"shortest_edge": 256, "longest_edge": 4096})

    def test_normalizer_ignores_prompt_schema_when_answer_json_follows(self) -> None:
        record = build_llm_input_manifest([make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001")])[0]
        schema_reminder = serialize_output_json(
            {
                "Observation_Protocol_Assessment": {"trusted_modality": "rgb|lidar|pose|multi", "risk_level": "low|medium|high|critical"},
                "Action_Output": {"action_token": "controlled token", "confidence": "0.0-1.0"},
            }
        )
        valid_answer = serialize_output_json(record["ground_truth_output"])

        normalized = normalize_model_response(f"{schema_reminder}\n\nassistant:\n{valid_answer}")

        self.assertEqual(normalized["Action_Output"]["action_token"], record["ground_truth_output"]["Action_Output"]["action_token"])

    def test_normalizer_defaults_modality_to_multi_not_rgb(self) -> None:
        normalized = normalize_model_response(
            {
                "Observation_Protocol_Assessment": {"risk_level": "medium"},
                "Intent_Extraction": {},
                "Action_Output": {"collaboration_request": {}},
                "Rationale_Short": "",
            }
        )

        self.assertEqual(normalized["Observation_Protocol_Assessment"]["trusted_modality"], "multi")
        self.assertEqual(normalized["Action_Output"]["collaboration_request"]["trusted_modality"], "multi")

    def test_prompt_contract_does_not_embed_json_schema_object(self) -> None:
        record = build_llm_input_manifest([make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001")])[0]
        prompt_text = record["prompt_messages"][-1]["content"]

        self.assertNotIn('"controlled token"', prompt_text)
        self.assertNotIn('"rgb|lidar|pose|multi"', prompt_text)
        self.assertIn("action_token: one of", prompt_text)
        self.assertIn("Observation_Protocol_Assessment (object, required)", prompt_text)
        self.assertIn("Intent_Extraction (object, required)", prompt_text)
        self.assertIn("Rationale_Short: non-empty string", prompt_text)

    def test_runner_refreshes_stale_manifest_prompt_schema(self) -> None:
        record = build_llm_input_manifest([make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001")])[0]
        record["prompt_messages"][-1]["content"] = 'Return only JSON. Follow this schema:\n{"Action_Output": {"action_token": "controlled token"}}'

        messages = build_zero_shot_messages(ensure_llm_input_record(record))

        self.assertNotIn('"controlled token"', messages[-1]["content"])
        self.assertIn("action_token: one of", messages[-1]["content"])

    def test_runner_refreshes_prompt_missing_protocol_contract(self) -> None:
        record = build_llm_input_manifest([make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001")])[0]
        record["prompt_messages"] = [
            {"role": "system", "content": "You are a protocol-conditioned UAV decision model."},
            {"role": "user", "content": "Return JSON with only Action_Output."},
        ]

        messages = build_zero_shot_messages(ensure_llm_input_record(record))

        self.assertIn("Observation_Protocol_Assessment (object, required)", messages[-1]["content"])
        self.assertIn("protocol_context_profile", messages[-1]["content"])
        self.assertIn("Protocol / policy context", messages[-1]["content"])

    def test_live_image_loader_resolves_backslash_artifact_paths(self) -> None:
        tmp_dir = make_workspace_temp_dir("path_resolution")
        try:
            image_path = tmp_dir / "artifacts" / "images" / "frame.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image

            Image.new("RGB", (8, 8), color=(1, 2, 3)).save(image_path)
            backslash_path = str(image_path).replace("/", "\\")

            resolved = resolve_existing_local_path(backslash_path)
            loaded = load_image_for_adapter(backslash_path, role="primary_frame")

            self.assertEqual(resolved, image_path)
            self.assertEqual(loaded.size, (8, 8))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_suite_failure_summary_includes_stderr_tail(self) -> None:
        summary = {
            "runs": [
                {
                    "model_id": "qwen3_5_4b",
                    "status": "failed",
                    "leaderboard_returncode": 1,
                    "leaderboard_stderr": "old line\nreal root cause",
                }
            ]
        }

        text = format_failure_summary(summary, tail_lines=1)

        self.assertIn("qwen3_5_4b", text)
        self.assertIn("real root cause", text)
        self.assertNotIn("old line", text)

    def test_completed_prediction_file_requires_all_manifest_samples(self) -> None:
        tmp_dir = make_workspace_temp_dir("completed_predictions")
        try:
            manifest_path = tmp_dir / "manifest.json"
            predictions_path = tmp_dir / "predictions.json"
            manifest_path.write_text(
                json.dumps([{"sample_id": "s1"}, {"sample_id": "s2"}], ensure_ascii=False),
                encoding="utf-8",
            )
            predictions_path.write_text(
                json.dumps([{"sample_id": "s1", "output": {}}], ensure_ascii=False),
                encoding="utf-8",
            )

            self.assertFalse(completed_prediction_file(predictions_path, manifest_path))

            predictions_path.write_text(
                json.dumps([{"sample_id": "s1", "output": {}}, {"sample_id": "s2", "output": {}}], ensure_ascii=False),
                encoding="utf-8",
            )
            self.assertTrue(completed_prediction_file(predictions_path, manifest_path))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_manifest_orders_primary_then_context_images(self) -> None:
        samples = [
            make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001"),
            make_sample(sample_id="s2", sequence_id="SEQ_A", frame_id="0002", rgb_suffix="0002"),
            make_sample(sample_id="s3", sequence_id="SEQ_A", frame_id="0003", rgb_suffix="0003"),
        ]
        manifest = build_llm_input_manifest(samples, context_frames=1, min_image_inputs=3)
        middle = next(record for record in manifest if record["sample_id"] == "s2")
        roles = [item["role"] for item in middle["image_inputs"]]
        self.assertEqual(roles[0], "primary_frame")
        self.assertEqual(roles[1:], ["context_prev_1", "context_next_1"])

    def test_manifest_fills_placeholder_context_when_neighbors_missing(self) -> None:
        manifest = build_llm_input_manifest([make_sample(sample_id="solo", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001")], context_frames=2, min_image_inputs=4)
        record = manifest[0]
        self.assertEqual(record["image_inputs"][0]["role"], "primary_frame")
        self.assertEqual(len(record["image_inputs"]), 4)
        self.assertTrue(all(item["is_placeholder"] for item in record["image_inputs"][1:]))

    def test_sft_export_keeps_primary_and_safe_alternative_weights(self) -> None:
        manifest = build_llm_input_manifest([make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001")])
        rows = build_sft_records(manifest)
        weights = sorted({row["training_weight"] for row in rows})
        self.assertEqual(weights, [0.5, 1.0])
        self.assertTrue(any(row["variant"] == "primary_ground_truth" for row in rows))
        self.assertTrue(any(row["variant"] == "safe_alternative" for row in rows))

    def test_preference_export_selects_hard_cases(self) -> None:
        hard = make_sample(sample_id="hard", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001", risk_level="critical")
        easy = make_sample(
            sample_id="easy",
            sequence_id="SEQ_B",
            frame_id="0001",
            rgb_suffix="0001",
            risk_level="low",
            policy_family="General Flight Operations",
        )
        easy["forbidden_actions"] = []
        easy["observation_protocol_gt"]["trusted_modality"] = "rgb"
        easy["collaboration_gt"] = {"required": False, "request_type": "none", "trusted_modality": "rgb"}
        easy["paper_task_gt"]["T12"] = {"required": False, "request_type": "none", "trusted_modality": "rgb"}
        manifest = build_llm_input_manifest([hard, easy])
        rows = build_preference_records(manifest, hardest_fraction=0.5)
        self.assertTrue(rows)
        self.assertTrue(all(row["sample_id"] == "hard" for row in rows))


class RunnerAndWrapperTests(unittest.TestCase):
    def test_runner_produces_evaluator_compatible_predictions(self) -> None:
        samples = [
            make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001"),
            make_sample(sample_id="s2", sequence_id="SEQ_A", frame_id="0002", rgb_suffix="0002"),
        ]
        manifest = build_llm_input_manifest(samples)
        runner = LLMInferenceRunner(model_id="qwen3_vl", runtime="mock")
        predictions = runner.run(manifest, prompt_mode="zero_shot")
        report = evaluate_predictions(samples, predictions)
        self.assertEqual(report["summary_metrics"]["PACS"], 1.0)

        predictions_few = runner.run(manifest, prompt_mode="few_shot", few_shot_count=1, seed=7)
        report_few = evaluate_predictions(samples, predictions_few)
        self.assertEqual(report_few["summary_metrics"]["Safe Action Accuracy"], 1.0)

    def test_runner_resumes_existing_sample_predictions_and_preserves_order(self) -> None:
        samples = [
            make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001"),
            make_sample(sample_id="s2", sequence_id="SEQ_A", frame_id="0002", rgb_suffix="0002"),
            make_sample(sample_id="s3", sequence_id="SEQ_A", frame_id="0003", rgb_suffix="0003"),
        ]
        manifest = build_llm_input_manifest(samples)
        existing_predictions = [
            {"sample_id": "s2", "output": {"Action_Output": {"action_token": "existing"}}},
        ]
        generated_sample_ids: list[str] = []

        runner = LLMInferenceRunner(model_id="qwen3_vl", runtime="mock")
        original_generate_batch = runner.adapter.generate_batch

        def tracking_generate_batch(records, messages_batch):
            generated_sample_ids.extend(record.sample_id for record in records)
            return original_generate_batch(records, messages_batch)

        runner.adapter.generate_batch = tracking_generate_batch
        predictions = runner.run(manifest, prompt_mode="zero_shot", existing_predictions=existing_predictions)

        self.assertEqual(generated_sample_ids, ["s1", "s3"])
        self.assertEqual([item["sample_id"] for item in predictions], ["s1", "s2", "s3"])
        resumed = next(item for item in predictions if item["sample_id"] == "s2")
        self.assertEqual(resumed["output"]["Action_Output"]["action_token"], "existing")

    def test_runner_calls_prediction_callback_for_new_predictions_only(self) -> None:
        samples = [
            make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001"),
            make_sample(sample_id="s2", sequence_id="SEQ_A", frame_id="0002", rgb_suffix="0002"),
        ]
        manifest = build_llm_input_manifest(samples)
        existing_predictions = [{"sample_id": "s1", "output": {"Action_Output": {"action_token": "existing"}}}]
        snapshots: list[list[str]] = []

        runner = LLMInferenceRunner(model_id="qwen3_vl", runtime="mock")
        runner.run(
            manifest,
            prompt_mode="zero_shot",
            existing_predictions=existing_predictions,
            on_prediction=lambda predictions: snapshots.append([item["sample_id"] for item in predictions]),
        )

        self.assertEqual(snapshots, [["s1", "s2"]])

    def test_runner_batches_new_records_with_mock_adapter(self) -> None:
        samples = [
            make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001"),
            make_sample(sample_id="s2", sequence_id="SEQ_A", frame_id="0002", rgb_suffix="0002"),
            make_sample(sample_id="s3", sequence_id="SEQ_A", frame_id="0003", rgb_suffix="0003"),
        ]
        manifest = build_llm_input_manifest(samples)
        calls: list[list[str]] = []

        runner = LLMInferenceRunner(model_id="qwen3_vl", runtime="mock", batch_size=2)

        def tracking_generate_batch(records, messages_batch):
            del messages_batch
            calls.append([record.sample_id for record in records])
            return [json.dumps(record.ground_truth_output, ensure_ascii=False) for record in records]

        runner.adapter.generate_batch = tracking_generate_batch
        predictions = runner.run(manifest, prompt_mode="zero_shot")

        self.assertEqual(calls, [["s1", "s2"], ["s3"]])
        self.assertEqual([item["sample_id"] for item in predictions], ["s1", "s2", "s3"])
        self.assertEqual(runner.last_run_metadata["effective_batch_size"], 2)

    def test_runner_records_normalization_failure_and_continues(self) -> None:
        samples = [
            make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001"),
            make_sample(sample_id="s2", sequence_id="SEQ_A", frame_id="0002", rgb_suffix="0002"),
        ]
        manifest = build_llm_input_manifest(samples)
        runner = LLMInferenceRunner(model_id="qwen3_vl", runtime="mock", batch_size=2)
        bad_response = "{래퍼의 요청을 분석합니다.\n\n1. **Observation_Protocol_Assessment**: not JSON"

        def generate_batch(records, messages_batch):
            del messages_batch
            return [bad_response, json.dumps(records[1].ground_truth_output, ensure_ascii=False)]

        runner.adapter.generate_batch = generate_batch
        predictions = runner.run(manifest, prompt_mode="zero_shot")
        report = evaluate_predictions(samples, predictions)

        self.assertEqual([item["sample_id"] for item in predictions], ["s1", "s2"])
        failed_output = predictions[0]["output"]
        self.assertTrue(failed_output["_normalization_failed"])
        self.assertEqual(failed_output["Action_Output"]["action_token"], "Reobserve")
        self.assertEqual(failed_output["Action_Output"]["confidence"], 0.0)
        self.assertIn("No valid model output", failed_output["_normalization_error"])
        self.assertIn("래퍼", failed_output["_raw_response_preview"])
        self.assertEqual(runner.last_run_metadata["normalization_failure_count"], 1)
        self.assertEqual(runner.last_run_metadata["normalization_failure_sample_ids"], ["s1"])
        self.assertEqual(report["task_metrics"]["evaluated_samples"], 2.0)

    def test_runner_counts_resumed_normalization_failures_in_metadata(self) -> None:
        samples = [
            make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001"),
            make_sample(sample_id="s2", sequence_id="SEQ_A", frame_id="0002", rgb_suffix="0002"),
        ]
        manifest = build_llm_input_manifest(samples)
        resumed_failure_output = dict(manifest[0]["ground_truth_output"])
        resumed_failure_output["_normalization_failed"] = True
        existing_predictions = [{"sample_id": "s1", "output": resumed_failure_output}]

        runner = LLMInferenceRunner(model_id="qwen3_vl", runtime="mock")
        predictions = runner.run(manifest, prompt_mode="zero_shot", existing_predictions=existing_predictions)

        self.assertEqual([item["sample_id"] for item in predictions], ["s1", "s2"])
        self.assertEqual(runner.last_run_metadata["normalization_failure_count"], 1)
        self.assertEqual(runner.last_run_metadata["normalization_failure_sample_ids"], ["s1"])

    def test_runner_reduces_batch_after_cuda_oom(self) -> None:
        samples = [
            make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001"),
            make_sample(sample_id="s2", sequence_id="SEQ_A", frame_id="0002", rgb_suffix="0002"),
            make_sample(sample_id="s3", sequence_id="SEQ_A", frame_id="0003", rgb_suffix="0003"),
        ]
        manifest = build_llm_input_manifest(samples)
        calls: list[list[str]] = []

        runner = LLMInferenceRunner(model_id="qwen3_vl", runtime="mock", batch_size=4, auto_batch=True)

        def flaky_generate_batch(records, messages_batch):
            del messages_batch
            calls.append([record.sample_id for record in records])
            if len(calls) == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
            return [json.dumps(record.ground_truth_output, ensure_ascii=False) for record in records]

        runner.adapter.generate_batch = flaky_generate_batch
        with patch("emerguav_bench.llm.runner.clear_cuda_memory") as clear_memory:
            predictions = runner.run(manifest, prompt_mode="zero_shot")

        self.assertEqual(calls, [["s1", "s2", "s3"], ["s1", "s2"], ["s3"]])
        self.assertEqual([item["sample_id"] for item in predictions], ["s1", "s2", "s3"])
        self.assertEqual(runner.last_run_metadata["effective_batch_size"], 2)
        self.assertEqual(runner.last_run_metadata["oom_fallback_count"], 1)
        clear_memory.assert_called()

    def test_runner_raises_batch_one_cuda_oom_after_backoff(self) -> None:
        samples = [
            make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001"),
            make_sample(sample_id="s2", sequence_id="SEQ_A", frame_id="0002", rgb_suffix="0002"),
        ]
        manifest = build_llm_input_manifest(samples)
        runner = LLMInferenceRunner(model_id="qwen3_vl", runtime="mock", batch_size=2, auto_batch=True)

        def always_oom(records, messages_batch):
            del records, messages_batch
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")

        runner.adapter.generate_batch = always_oom
        with patch("emerguav_bench.llm.runner.clear_cuda_memory"):
            with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                runner.run(manifest, prompt_mode="zero_shot")

        self.assertEqual(runner.last_run_metadata["effective_batch_size"], 1)
        self.assertEqual(runner.last_run_metadata["oom_fallback_count"], 1)

    def test_training_wrappers_support_dry_run(self) -> None:
        tmp_dir = make_workspace_temp_dir("llm_training")
        try:
            manifest = build_llm_input_manifest([make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001")])
            sft_rows = build_sft_records(manifest)
            preference_rows = build_preference_records(manifest, hardest_fraction=1.0)
            sft_path = tmp_dir / "train.jsonl"
            pref_path = tmp_dir / "preference.jsonl"
            with sft_path.open("w", encoding="utf-8") as handle:
                for row in sft_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            with pref_path.open("w", encoding="utf-8") as handle:
                for row in preference_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            sft_summary = SFTTrainingWrapper(SFTTrainingConfig(model_id="qwen3_vl", output_dir=str(tmp_dir))).dry_run(sft_path)
            pref_summary = PreferenceTrainingWrapper(PreferenceTrainingConfig(model_id="qwen3_vl", output_dir=str(tmp_dir), algorithm="DPO")).dry_run(pref_path)
            self.assertGreaterEqual(sft_summary["weighted_record_count"], sft_summary["record_count"])
            self.assertEqual(pref_summary["algorithm"], "DPO")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class ScriptSmokeTests(unittest.TestCase):
    def test_cli_flow_from_samples_to_reports(self) -> None:
        tmp_dir = make_workspace_temp_dir("llm_cli")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(Path.cwd() / "src"), str(Path.cwd())])
        try:
            placeholder_manifest = [make_placeholder_sequence(sequence_id="SEQ_SMOKE", scene_family="HKairport", frame_count=2)]
            samples = build_seed_samples_from_manifest(placeholder_manifest, language_entropy_labels=["code_switching"])
            samples_path = tmp_dir / "samples.json"
            manifest_path = tmp_dir / "llm_manifest.json"
            sft_path = tmp_dir / "sft.jsonl"
            pref_path = tmp_dir / "pref.jsonl"
            results_root = tmp_dir / "results"
            report_path = tmp_dir / "report.json"
            with samples_path.open("w", encoding="utf-8") as handle:
                json.dump(samples, handle, ensure_ascii=False, indent=2)

            commands = [
                [sys.executable, "scripts/export_manifest.py", "--samples", str(samples_path), "--manifest-out", str(manifest_path)],
                [sys.executable, "scripts/export_sft_jsonl.py", "--manifest", str(manifest_path), "--sft-out", str(sft_path)],
                [sys.executable, "scripts/export_preference_jsonl.py", "--manifest", str(manifest_path), "--preference-out", str(pref_path), "--hardest-fraction", "1.0"],
                [sys.executable, "scripts/run_inference.py", "--manifest", str(manifest_path), "--model-id", "qwen3_vl", "--results-root", str(results_root), "--runtime", "mock"],
                [sys.executable, "scripts/train_sft.py", "--train-jsonl", str(sft_path), "--model-id", "qwen3_vl", "--output-dir", str(tmp_dir / "sft_out")],
                [sys.executable, "scripts/train_preference.py", "--preference-jsonl", str(pref_path), "--model-id", "qwen3_vl", "--output-dir", str(tmp_dir / "pref_out")],
            ]
            for command in commands:
                subprocess.run(command, cwd=Path.cwd(), env=env, check=True)

            predictions_path = results_root / "qwen3_vl" / "dev" / "zero_shot" / "seed_0" / "predictions.json"
            with predictions_path.open("r", encoding="utf-8") as handle:
                predictions = json.load(handle)
            report = evaluate_predictions(samples, predictions)
            with report_path.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)

            self.assertTrue(manifest_path.exists())
            self.assertTrue(sft_path.exists())
            self.assertTrue(pref_path.exists())
            self.assertTrue(predictions_path.exists())
            self.assertEqual(report["summary_metrics"]["PACS"], 1.0)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_leaderboard_cli_resumes_partial_predictions(self) -> None:
        tmp_dir = make_workspace_temp_dir("llm_resume_cli")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(Path.cwd() / "src"), str(Path.cwd())])
        try:
            samples = [
                make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001"),
                make_sample(sample_id="s2", sequence_id="SEQ_A", frame_id="0002", rgb_suffix="0002"),
            ]
            manifest = build_llm_input_manifest(samples)
            manifest_path = tmp_dir / "llm_manifest.json"
            results_root = tmp_dir / "results"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            save_predictions_tree(
                [{"sample_id": "s1", "output": manifest[0]["ground_truth_output"]}],
                results_root=results_root,
                model_id="qwen3_vl",
                split_name="dev",
                prompt_mode="zero_shot",
                seed=0,
            )

            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_inference.py",
                    "--manifest",
                    str(manifest_path),
                    "--model-id",
                    "qwen3_vl",
                    "--results-root",
                    str(results_root),
                    "--runtime",
                    "mock",
                    "--resume",
                    "--flush-every",
                    "1",
                ],
                cwd=Path.cwd(),
                env=env,
                check=True,
            )

            predictions_path = results_root / "qwen3_vl" / "dev" / "zero_shot" / "seed_0" / "predictions.json"
            predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
            self.assertEqual([item["sample_id"] for item in predictions], ["s1", "s2"])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_leaderboard_cli_accepts_generation_memory_options(self) -> None:
        tmp_dir = make_workspace_temp_dir("llm_generation_options_cli")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(Path.cwd() / "src"), str(Path.cwd())])
        try:
            samples = [make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001")]
            manifest = build_llm_input_manifest(samples)
            manifest_path = tmp_dir / "llm_manifest.json"
            results_root = tmp_dir / "results"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_inference.py",
                    "--manifest",
                    str(manifest_path),
                    "--model-id",
                    "qwen3_vl",
                    "--results-root",
                    str(results_root),
                    "--runtime",
                    "mock",
                    "--max-new-tokens",
                    "256",
                    "--processor-max-pixels",
                    "262144",
                    "--batch-size",
                    "2",
                    "--precision",
                    "bf16",
                ],
                cwd=Path.cwd(),
                env=env,
                check=True,
            )

            predictions_path = results_root / "qwen3_vl" / "dev" / "zero_shot" / "seed_0" / "predictions.json"
            metadata_path = predictions_path.parent / "run_metadata.json"
            self.assertTrue(predictions_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["requested_batch_size"], 2)
            self.assertEqual(metadata["precision"], "bf16")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_leaderboard_cli_flushes_without_resume_flag_before_later_failure(self) -> None:
        tmp_dir = make_workspace_temp_dir("llm_flush_cli")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(Path.cwd() / "src"), str(Path.cwd())])
        try:
            samples = [
                make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001"),
                make_sample(sample_id="s2", sequence_id="SEQ_A", frame_id="0002", rgb_suffix="0002"),
            ]
            manifest = build_llm_input_manifest(samples)
            manifest[1]["sample_id"] = ""
            manifest_path = tmp_dir / "llm_manifest.json"
            results_root = tmp_dir / "results"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_inference.py",
                    "--manifest",
                    str(manifest_path),
                    "--model-id",
                    "qwen3_vl",
                    "--results-root",
                    str(results_root),
                    "--runtime",
                    "mock",
                    "--flush-every",
                    "1",
                ],
                cwd=Path.cwd(),
                env=env,
                text=True,
                capture_output=True,
            )

            predictions_path = results_root / "qwen3_vl" / "dev" / "zero_shot" / "seed_0" / "predictions.json"
            self.assertNotEqual(result.returncode, 0)
            predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
            self.assertEqual([item["sample_id"] for item in predictions], ["s1"])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_model_suite_cli_runs_selected_models_and_writes_reports(self) -> None:
        tmp_dir = make_workspace_temp_dir("llm_suite_cli")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(Path.cwd() / "src"), str(Path.cwd())])
        try:
            placeholder_manifest = [make_placeholder_sequence(sequence_id="SEQ_SUITE", scene_family="HKairport", frame_count=2)]
            samples = build_seed_samples_from_manifest(placeholder_manifest, language_entropy_labels=["code_switching"])
            manifest = build_llm_input_manifest(samples)
            samples_path = tmp_dir / "samples.json"
            manifest_path = tmp_dir / "llm_manifest.json"
            results_root = tmp_dir / "suite_results"
            samples_path.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_model_suite.py",
                    "--manifest",
                    str(manifest_path),
                    "--samples",
                    str(samples_path),
                    "--suite",
                    "core",
                    "--models",
                    "qwen3_vl_2b,internvl3_5_8b",
                    "--results-root",
                    str(results_root),
                    "--runtime",
                    "mock",
                ],
                cwd=Path.cwd(),
                env=env,
                check=True,
            )

            for model_id in ("qwen3_vl_2b", "internvl3_5_8b"):
                output_dir = results_root / model_id / "dev" / "zero_shot" / "seed_0"
                self.assertTrue((output_dir / "predictions.json").exists())
                self.assertTrue((output_dir / "evaluation_report.json").exists())
            summary = json.loads((results_root / "suite_core_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["suite"], "core")
            self.assertEqual({item["model_id"] for item in summary["runs"]}, {"qwen3_vl_2b", "internvl3_5_8b"})
            self.assertTrue(all(item["status"] == "passed" for item in summary["runs"]))
            self.assertTrue(all(item["run_metadata"]["precision"] == "bf16" for item in summary["runs"]))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_leaderboard_run_preserves_partial_predictions_when_batch_one_oom(self) -> None:
        from scripts import run_inference as leaderboard

        tmp_dir = make_workspace_temp_dir("llm_partial_oom")
        try:
            samples = [
                make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001"),
                make_sample(sample_id="s2", sequence_id="SEQ_A", frame_id="0002", rgb_suffix="0002"),
            ]
            manifest = build_llm_input_manifest(samples)
            manifest_path = tmp_dir / "llm_manifest.json"
            results_root = tmp_dir / "results"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

            class FakeAdapter:
                generation_kwargs = {"max_new_tokens": 768}

                def unload(self) -> None:
                    pass

            class FakeRunner:
                def __init__(self, **kwargs) -> None:
                    self.adapter = FakeAdapter()
                    self.last_run_metadata = {
                        "model_id": kwargs["model_id"],
                        "requested_batch_size": kwargs["batch_size"],
                        "initial_batch_size": 1,
                        "effective_batch_size": 1,
                        "auto_batch": kwargs["auto_batch"],
                        "oom_fallback_count": 0,
                        "precision": kwargs["precision"],
                    }

                def run(self, records, **kwargs):
                    partial = [{"sample_id": records[0]["sample_id"], "output": records[0]["ground_truth_output"]}]
                    kwargs["on_prediction"](partial)
                    raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")

            args = types.SimpleNamespace(
                manifest=str(manifest_path),
                model_id="qwen3_vl",
                results_root=str(results_root),
                prompt_mode="zero_shot",
                few_shot_count=0,
                seed=0,
                runtime="live",
                resume=False,
                flush_every=0,
                max_samples=None,
                max_new_tokens=None,
                processor_max_pixels=None,
                batch_size=1,
                disable_auto_batch=False,
                precision="bf16",
            )

            with patch.object(leaderboard, "LLMInferenceRunner", FakeRunner), patch.object(leaderboard, "clear_cuda_memory"):
                returncode = leaderboard.run_leaderboard(args)

            predictions_path = results_root / "qwen3_vl" / "dev" / "zero_shot" / "seed_0" / "predictions.json"
            metadata_path = predictions_path.parent / "run_metadata.json"
            self.assertEqual(returncode, OOM_SKIP_RETURN_CODE)
            predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual([item["sample_id"] for item in predictions], ["s1"])
            self.assertEqual(metadata["status"], "skipped_oom")
            self.assertIn("CUDA out of memory", metadata["error"])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_model_suite_counts_batch_one_oom_as_skipped(self) -> None:
        tmp_dir = make_workspace_temp_dir("llm_suite_oom_skip")
        try:
            manifest_path = tmp_dir / "llm_manifest.json"
            manifest_path.write_text(json.dumps([{"sample_id": "s1", "split_name": "dev"}]), encoding="utf-8")

            def fake_run_command(command, **kwargs):
                del command, kwargs
                return subprocess.CompletedProcess(args=[], returncode=OOM_SKIP_RETURN_CODE, stdout="", stderr="CUDA out of memory")

            with patch("scripts.run_model_suite.run_command", side_effect=fake_run_command):
                summary = run_model_suite(
                    manifest=manifest_path,
                    samples=None,
                    suite="core",
                    models=["qwen3_vl_2b"],
                    results_root=tmp_dir / "results",
                    runtime="live",
                    stream_subprocess=False,
                )

            self.assertEqual(summary["skipped_model_count"], 1)
            self.assertEqual(summary["failed_model_count"], 0)
            self.assertEqual(summary["runs"][0]["status"], "skipped_oom")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_model_suite_sets_pytorch_cuda_allocator_env(self) -> None:
        tmp_dir = make_workspace_temp_dir("llm_suite_cuda_alloc_env")
        captured_envs: list[dict[str, str]] = []
        try:
            manifest_path = tmp_dir / "llm_manifest.json"
            manifest_path.write_text(json.dumps([{"sample_id": "s1", "split_name": "dev"}]), encoding="utf-8")

            def fake_run_command(command, **kwargs):
                del command
                captured_envs.append(dict(kwargs["env"]))
                return subprocess.CompletedProcess(args=[], returncode=OOM_SKIP_RETURN_CODE, stdout="", stderr="CUDA out of memory")

            with patch("scripts.run_model_suite.run_command", side_effect=fake_run_command):
                run_model_suite(
                    manifest=manifest_path,
                    samples=None,
                    suite="core",
                    models=["qwen3_vl_2b"],
                    results_root=tmp_dir / "results",
                    runtime="live",
                    cuda_alloc_conf="expandable_segments:True",
                    stream_subprocess=False,
                )

            self.assertEqual(captured_envs[0].get("PYTORCH_CUDA_ALLOC_CONF"), "expandable_segments:True")
            self.assertNotIn("PYTORCH_ALLOC_CONF", captured_envs[0])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_model_suite_backfills_normalization_failure_metadata_for_completed_predictions(self) -> None:
        tmp_dir = make_workspace_temp_dir("llm_suite_metadata_backfill")
        try:
            samples = [
                make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001"),
                make_sample(sample_id="s2", sequence_id="SEQ_A", frame_id="0002", rgb_suffix="0002"),
            ]
            manifest = build_llm_input_manifest(samples)
            manifest_path = tmp_dir / "llm_manifest.json"
            results_root = tmp_dir / "suite_results"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

            resumed_failure_output = dict(manifest[0]["ground_truth_output"])
            resumed_failure_output["_normalization_failed"] = True
            predictions_path = save_predictions_tree(
                [
                    {"sample_id": "s1", "output": resumed_failure_output},
                    {"sample_id": "s2", "output": manifest[1]["ground_truth_output"]},
                ],
                results_root=results_root,
                model_id="qwen3_vl_2b",
                split_name="dev",
                prompt_mode="zero_shot",
                seed=0,
            )
            metadata_path = predictions_path.parent / "run_metadata.json"
            metadata_path.write_text(json.dumps({"status": "passed", "precision": "bf16"}), encoding="utf-8")

            with patch("scripts.run_model_suite.run_command", side_effect=AssertionError("completed model should be skipped")):
                summary = run_model_suite(
                    manifest=manifest_path,
                    samples=None,
                    suite="core",
                    models=["qwen3_vl_2b"],
                    results_root=results_root,
                    runtime="mock",
                    skip_existing=True,
                    stream_subprocess=False,
                )

            run_metadata = summary["runs"][0]["run_metadata"]
            rewritten_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["runs"][0]["status"], "passed")
            self.assertEqual(run_metadata["normalization_failure_count"], 1)
            self.assertEqual(run_metadata["normalization_failure_sample_ids"], ["s1"])
            self.assertEqual(rewritten_metadata["normalization_failure_count"], 1)
            self.assertEqual(rewritten_metadata["normalization_failure_sample_ids"], ["s1"])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_model_suite_cli_forwards_resume_options(self) -> None:
        tmp_dir = make_workspace_temp_dir("llm_suite_resume_cli")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(Path.cwd() / "src"), str(Path.cwd())])
        try:
            samples = [
                make_sample(sample_id="s1", sequence_id="SEQ_A", frame_id="0001", rgb_suffix="0001"),
                make_sample(sample_id="s2", sequence_id="SEQ_A", frame_id="0002", rgb_suffix="0002"),
            ]
            manifest = build_llm_input_manifest(samples)
            samples_path = tmp_dir / "samples.json"
            manifest_path = tmp_dir / "llm_manifest.json"
            results_root = tmp_dir / "suite_results"
            samples_path.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            save_predictions_tree(
                [{"sample_id": "s1", "output": manifest[0]["ground_truth_output"]}],
                results_root=results_root,
                model_id="qwen3_vl_2b",
                split_name="dev",
                prompt_mode="zero_shot",
                seed=0,
            )

            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_model_suite.py",
                    "--manifest",
                    str(manifest_path),
                    "--samples",
                    str(samples_path),
                    "--suite",
                    "core",
                    "--models",
                    "qwen3_vl_2b",
                    "--results-root",
                    str(results_root),
                    "--runtime",
                    "mock",
                    "--skip-existing",
                    "--resume-model",
                    "--flush-every",
                    "1",
                ],
                cwd=Path.cwd(),
                env=env,
                check=True,
            )

            predictions_path = results_root / "qwen3_vl_2b" / "dev" / "zero_shot" / "seed_0" / "predictions.json"
            predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
            self.assertEqual([item["sample_id"] for item in predictions], ["s1", "s2"])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


