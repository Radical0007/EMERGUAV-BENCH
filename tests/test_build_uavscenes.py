import json
import shutil
import unittest
from pathlib import Path

from scripts.build_uavscenes import DEFAULT_UAVSCENES_ROOT, build_materialized_bench_dataset
from test_uavscenes_materialize import make_fixture_uavscenes_root, make_workspace_temp_dir


class UAVScenesBuildScriptTests(unittest.TestCase):
    def test_default_uavscenes_root_is_not_windows_specific(self) -> None:
        self.assertNotIn("\\", DEFAULT_UAVSCENES_ROOT)
        self.assertNotRegex(DEFAULT_UAVSCENES_ROOT, r"^[A-Za-z]:")

    def test_build_materialized_bench_dataset_writes_pipeline_outputs(self) -> None:
        source_root = make_fixture_uavscenes_root()
        output_root = make_workspace_temp_dir("uavscenes_build")
        try:
            report = build_materialized_bench_dataset(
                uavscenes_root=source_root,
                output_root=output_root,
                target_frames_per_sequence=3,
                sequence_ids=["AMtown03"],
                context_frames=1,
                min_image_inputs=2,
            )

            expected_files = [
                "samples.json",
                "predictions.json",
                "llm_manifest.json",
                "sft.jsonl",
                "preference.jsonl",
                "build_report.json",
            ]
            for filename in expected_files:
                self.assertTrue((output_root / filename).exists(), filename)

            self.assertEqual(report["sample_count"], 3)
            self.assertEqual(report["llm_record_count"], 3)
            self.assertEqual(report["sequences"][0]["selected_frame_count"], 3)

            samples = json.loads((output_root / "samples.json").read_text(encoding="utf-8"))
            self.assertEqual(len(samples), 3)
            first_metadata = samples[0]["protocol_metadata"]["uavscenes"]
            self.assertTrue(Path(first_metadata["materialized_paths"]["rgb"]).exists())
            self.assertTrue(Path(first_metadata["materialized_paths"]["camera_label"]).exists())
            self.assertTrue(Path(first_metadata["materialized_paths"]["lidar_label"]).exists())
        finally:
            shutil.rmtree(source_root, ignore_errors=True)
            shutil.rmtree(output_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

