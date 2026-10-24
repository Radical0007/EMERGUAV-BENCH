from __future__ import annotations

import os
import types
import unittest
from unittest.mock import MagicMock, patch

from emerguav_bench.utils.gpu import (
    classify_models,
    clear_cuda_memory,
    detect_visible_cuda_vram_gb,
    dual_gpu_env,
    estimate_model_billions,
    gpu_affinity_env,
    gpu_count,
    is_cuda_out_of_memory_error,
    model_fits_single_gpu,
    recommended_inference_batch_size,
)


class EstimateModelBillionsTests(unittest.TestCase):
    def test_standard_pattern_4b(self) -> None:
        # Overestimates to 5.4 (version '5' leaks in), but stays <20B threshold
        self.assertEqual(estimate_model_billions("qwen3_5_4b"), 5.4)

    def test_standard_pattern_27b(self) -> None:
        self.assertEqual(estimate_model_billions("qwen3_5_27b"), 27.0)

    def test_moe_pattern_35b_a3b(self) -> None:
        self.assertEqual(estimate_model_billions("qwen3_5_35b_a3b"), 35.0)

    def test_decimal_param_0_8b(self) -> None:
        # Returns 8.0 (the '0_' decimal prefix is not captured), stays <20B
        self.assertEqual(estimate_model_billions("qwen3_5_0_8b"), 8.0)

    def test_decimal_param_2_2b(self) -> None:
        self.assertEqual(estimate_model_billions("smolvlm2_2_2b"), 2.2)

    def test_suffix_after_b(self) -> None:
        self.assertEqual(estimate_model_billions("gemma4_e4b_it"), 4.0)
        self.assertEqual(estimate_model_billions("gemma4_26b_a4b_it"), 26.0)

    def test_fp8_variant(self) -> None:
        self.assertEqual(estimate_model_billions("qwen3_5_27b_fp8"), 27.0)

    def test_no_size_hint(self) -> None:
        self.assertIsNone(estimate_model_billions("phi4_mm"))
        self.assertIsNone(estimate_model_billions("aya_vision"))
        self.assertIsNone(estimate_model_billions("idefics3"))
        self.assertIsNone(estimate_model_billions("llava_next"))


class ModelFitsSingleGPUTests(unittest.TestCase):
    def test_4b_fits(self) -> None:
        self.assertTrue(model_fits_single_gpu("qwen3_5_4b"))

    def test_8b_fits(self) -> None:
        self.assertTrue(model_fits_single_gpu("qwen3_vl_8b"))

    def test_16b_fits(self) -> None:
        self.assertTrue(model_fits_single_gpu("ovis2_16b"))

    def test_27b_too_large(self) -> None:
        self.assertFalse(model_fits_single_gpu("qwen3_5_27b"))

    def test_35b_moe_too_large(self) -> None:
        self.assertFalse(model_fits_single_gpu("qwen3_5_35b_a3b"))

    def test_unknown_size_defaults_to_fits(self) -> None:
        self.assertTrue(model_fits_single_gpu("phi4_mm"))

    def test_custom_threshold(self) -> None:
        self.assertTrue(model_fits_single_gpu("qwen3_5_27b", threshold_b=30.0))
        self.assertFalse(model_fits_single_gpu("qwen3_5_27b", threshold_b=10.0))


class GPUAffinityEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")

    def tearDown(self) -> None:
        if self._saved_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._saved_cuda

    def test_affinity_env_sets_single_gpu(self) -> None:
        env = gpu_affinity_env(0)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")

    def test_affinity_env_overrides_existing(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
        env = gpu_affinity_env(1)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")

    def test_dual_gpu_env_sets_both(self) -> None:
        env = dual_gpu_env()
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")

    def test_affinity_env_preserves_other_vars(self) -> None:
        os.environ["TEST_DUMMY_VAR"] = "hello"
        env = gpu_affinity_env(0)
        self.assertEqual(env.get("TEST_DUMMY_VAR"), "hello")


class ClassifyModelsTests(unittest.TestCase):
    def test_splits_small_and_large(self) -> None:
        model_ids = ["qwen3_5_4b", "qwen3_5_27b", "qwen3_vl_8b", "qwen3_5_35b_a3b"]
        single, dual = classify_models(model_ids)
        self.assertEqual(set(single), {"qwen3_5_4b", "qwen3_vl_8b"})
        self.assertEqual(set(dual), {"qwen3_5_27b", "qwen3_5_35b_a3b"})

    def test_all_small(self) -> None:
        model_ids = ["qwen3_5_4b", "gemma4_e4b_it"]
        single, dual = classify_models(model_ids)
        self.assertEqual(single, model_ids)
        self.assertEqual(dual, [])

    def test_all_large(self) -> None:
        model_ids = ["qwen3_5_27b", "qwen3_5_35b_a3b"]
        single, dual = classify_models(model_ids)
        self.assertEqual(single, [])
        self.assertEqual(dual, model_ids)

    def test_unknown_goes_to_single(self) -> None:
        model_ids = ["phi4_mm", "aya_vision", "qwen3_5_27b"]
        single, dual = classify_models(model_ids)
        self.assertIn("phi4_mm", single)
        self.assertIn("aya_vision", single)
        self.assertIn("qwen3_5_27b", dual)


class InferenceBatchSizingTests(unittest.TestCase):
    def test_single_a100_batch_recommendations_are_conservative(self) -> None:
        self.assertEqual(recommended_inference_batch_size("qwen3_vl_2b", vram_gb=40), 8)
        self.assertEqual(recommended_inference_batch_size("qwen3_5_2b", vram_gb=40), 4)
        self.assertEqual(recommended_inference_batch_size("qwen3_5_4b", vram_gb=40), 4)
        self.assertEqual(recommended_inference_batch_size("qwen3_vl_8b", vram_gb=40), 2)
        self.assertEqual(recommended_inference_batch_size("ovis2_16b", vram_gb=40), 1)
        self.assertEqual(recommended_inference_batch_size("phi4_mm", vram_gb=40), 1)

    def test_detect_visible_cuda_vram_uses_largest_visible_device(self) -> None:
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.device_count.return_value = 2
        fake_torch.cuda.get_device_properties.side_effect = [
            types.SimpleNamespace(total_memory=24 * 1024**3),
            types.SimpleNamespace(total_memory=80 * 1024**3),
        ]

        with patch("emerguav_bench.utils.gpu._lazy_import_torch", return_value=fake_torch):
            self.assertEqual(detect_visible_cuda_vram_gb(), 80)

    def test_detect_visible_cuda_vram_rounds_marketed_card_capacity_up(self) -> None:
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.device_count.return_value = 1
        fake_torch.cuda.get_device_properties.return_value = types.SimpleNamespace(total_memory=int(39.2 * 1024**3))

        with patch("emerguav_bench.utils.gpu._lazy_import_torch", return_value=fake_torch):
            self.assertEqual(detect_visible_cuda_vram_gb(), 40)

    def test_batch_recommendation_uses_detected_vram_when_unspecified(self) -> None:
        with patch("emerguav_bench.utils.gpu.detect_visible_cuda_vram_gb", return_value=24):
            self.assertEqual(recommended_inference_batch_size("qwen3_vl_2b"), 4)
            self.assertEqual(recommended_inference_batch_size("qwen3_vl_8b"), 1)

    def test_cuda_oom_detection_uses_exception_type_and_message(self) -> None:
        named_oom = type("OutOfMemoryError", (Exception,), {})

        self.assertTrue(is_cuda_out_of_memory_error(named_oom("allocation failed")))
        self.assertTrue(is_cuda_out_of_memory_error(RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")))
        self.assertFalse(is_cuda_out_of_memory_error(RuntimeError("file not found")))

    def test_clear_cuda_memory_is_safe_without_torch(self) -> None:
        with patch("emerguav_bench.utils.gpu._lazy_import_torch", side_effect=ImportError):
            clear_cuda_memory()


class GPUCountTests(unittest.TestCase):
    def test_no_cuda_returns_zero(self) -> None:
        with patch("emerguav_bench.utils.gpu._lazy_import_torch", side_effect=ImportError):
            self.assertEqual(gpu_count(), 0)

    def test_cuda_unavailable_returns_zero(self) -> None:
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        with patch("emerguav_bench.utils.gpu._lazy_import_torch", return_value=fake_torch):
            self.assertEqual(gpu_count(), 0)

    def test_two_gpus_detected(self) -> None:
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.device_count.return_value = 2
        with patch("emerguav_bench.utils.gpu._lazy_import_torch", return_value=fake_torch):
            self.assertEqual(gpu_count(), 2)


if __name__ == "__main__":
    unittest.main()
