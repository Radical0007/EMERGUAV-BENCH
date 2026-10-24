import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from emerguav_bench.degradation.variants import STRICT_VARIANT_MATRIX, build_strict_variant_samples
from emerguav_bench.llm.exporters import build_llm_input_manifest
from emerguav_bench.llm.runner import LLMInferenceRunner
from emerguav_bench.uavscenes.manifest import make_placeholder_sequence
from scripts.build_samples import build_seed_samples_from_manifest
from scripts.run_model_suite import run_model_suite


class FakeProgressBar:
    instances = []

    def __init__(self, *, total=None, desc=None, unit=None, **kwargs):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.kwargs = kwargs
        self.count = 0
        self.updates = []
        self.postfixes = []
        self.descriptions = [desc]
        FakeProgressBar.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def update(self, amount=1):
        self.updates.append(amount)
        self.count += amount

    def set_description(self, desc=None, refresh=True):
        del refresh
        self.desc = desc
        self.descriptions.append(desc)

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        del refresh
        payload = {}
        if ordered_dict:
            payload.update(dict(ordered_dict))
        payload.update(kwargs)
        self.postfixes.append(payload)


def make_base_sample(tmp_dir: Path) -> dict:
    rgb_path = tmp_dir / "rgb.jpg"
    lidar_path = tmp_dir / "lidar.bin"
    Image.new("RGB", (32, 32), color=(120, 120, 120)).save(rgb_path)
    lidar_path.write_text("1 2 3\n", encoding="utf-8")
    sample = build_seed_samples_from_manifest(
        [make_placeholder_sequence(sequence_id="SEQ_PROGRESS", scene_family="HKairport", frame_count=1)]
    )[0]
    sample["rgb_frame_path"] = str(rgb_path)
    sample["lidar_frame_path"] = str(lidar_path)
    sample["protocol_metadata"]["placeholder_source"] = False
    sample["protocol_metadata"]["uavscenes"]["materialized_paths"]["rgb"] = str(rgb_path)
    return sample


class ProgressBarTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeProgressBar.instances = []
        self._previous_disable_tqdm = os.environ.get("EMERGUAV_DISABLE_TQDM")
        os.environ["EMERGUAV_DISABLE_TQDM"] = "1"

    def tearDown(self) -> None:
        if self._previous_disable_tqdm is None:
            os.environ.pop("EMERGUAV_DISABLE_TQDM", None)
        else:
            os.environ["EMERGUAV_DISABLE_TQDM"] = self._previous_disable_tqdm

    def test_strict_variant_builder_updates_progress_for_each_variant(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_dir = Path(raw_tmp)
            base_sample = make_base_sample(tmp_dir)
            cases = STRICT_VARIANT_MATRIX[:2]

            with patch("emerguav_bench.degradation.variants.make_progress_bar", FakeProgressBar):
                build_strict_variant_samples([base_sample], output_root=tmp_dir / "strict", variant_matrix=cases)

            self.assertEqual(FakeProgressBar.instances[0].total, len(cases))
            self.assertEqual(FakeProgressBar.instances[0].count, len(cases))

    def test_llm_runner_wraps_records_with_progress(self) -> None:
        samples = build_seed_samples_from_manifest(
            [make_placeholder_sequence(sequence_id="SEQ_LLM", scene_family="HKairport", frame_count=2)]
        )

        manifest = build_llm_input_manifest(samples)
        with patch("emerguav_bench.llm.runner.make_progress_bar", FakeProgressBar):
            LLMInferenceRunner(model_id="qwen3_vl", runtime="mock").run(manifest)

        self.assertEqual(FakeProgressBar.instances[0].desc, "qwen3_vl inference")
        self.assertEqual(FakeProgressBar.instances[0].total, 2)
        self.assertEqual(FakeProgressBar.instances[0].unit, "sample")
        self.assertEqual(FakeProgressBar.instances[0].count, 2)
        self.assertEqual(len(FakeProgressBar.instances[0].postfixes), 2)
        self.assertIn("sample", FakeProgressBar.instances[0].postfixes[0])

    def test_llm_runner_updates_generated_progress_by_batch(self) -> None:
        samples = build_seed_samples_from_manifest(
            [make_placeholder_sequence(sequence_id="SEQ_LLM_BATCH", scene_family="HKairport", frame_count=3)]
        )
        manifest = build_llm_input_manifest(samples)

        with (
            patch("emerguav_bench.llm.runner.make_progress_bar", FakeProgressBar),
            patch("emerguav_bench.llm.runner.time.perf_counter", side_effect=[100.0, 104.0, 104.0, 107.0]),
        ):
            LLMInferenceRunner(model_id="qwen3_vl", runtime="mock", batch_size=2).run(manifest)

        progress = FakeProgressBar.instances[0]
        self.assertIn("bar_format", progress.kwargs)
        self.assertNotIn("rate_fmt", progress.kwargs["bar_format"])
        self.assertEqual(progress.updates, [2, 1])
        self.assertEqual(progress.postfixes[0]["item"], "2/3")
        self.assertEqual(progress.postfixes[0]["batch_time"], "4.00s/batch")
        self.assertEqual(progress.postfixes[0]["throughput"], "0.50sample/s")
        self.assertEqual(progress.postfixes[0]["eta"], "2.00s")
        self.assertEqual(progress.postfixes[1]["item"], "3/3")
        self.assertEqual(progress.postfixes[1]["batch_time"], "3.00s/batch")
        self.assertEqual(progress.postfixes[1]["eta"], "0.00s")

    def test_model_suite_updates_progress_for_each_model(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_dir = Path(raw_tmp)
            manifest_path = tmp_dir / "manifest.json"
            suite_config_path = tmp_dir / "suite.json"
            manifest_path.write_text('[{"split_name": "dev"}]', encoding="utf-8")
            suite_config_path.write_text('{"core": {"model_ids": ["qwen3_vl_2b", "internvl3_5_8b"]}}', encoding="utf-8")

            with patch("scripts.run_model_suite.make_progress_bar", FakeProgressBar):
                with patch("scripts.run_model_suite.run_command") as run_command:
                    run_command.return_value.returncode = 0
                    run_command.return_value.stdout = ""
                    run_command.return_value.stderr = ""
                    run_model_suite(
                        manifest=manifest_path,
                        samples=None,
                        suite="core",
                        suite_config=suite_config_path,
                        results_root=tmp_dir / "results",
                        runtime="mock",
                    )

            self.assertEqual(FakeProgressBar.instances[0].total, 2)
            self.assertEqual(FakeProgressBar.instances[0].count, 2)
            self.assertIn("stage", FakeProgressBar.instances[0].postfixes[-1])
            self.assertTrue(any("qwen3_vl_2b" in str(desc) for desc in FakeProgressBar.instances[0].descriptions))
            self.assertTrue(run_command.call_args.kwargs["stream_output"])


if __name__ == "__main__":
    unittest.main()

