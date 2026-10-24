import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from emerguav_bench.uavscenes.calibration import get_calibration, has_calibration, list_calibrated_scenes
from emerguav_bench.uavscenes.indexer import UAVScenesIndexer
from emerguav_bench.uavscenes.manifest import make_placeholder_frame, make_placeholder_sequence


def make_workspace_temp_dir(prefix: str) -> Path:
    base_dir = Path.cwd() / ".test_tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = base_dir / f"{prefix}_{uuid4().hex}"
    temp_dir.mkdir()
    return temp_dir


class UAVScenesPlaceholderTests(unittest.TestCase):
    def test_get_calibration_returns_numeric_payload(self) -> None:
        calibration = get_calibration("HKairport01")
        self.assertIn("camera_intrinsic", calibration)
        self.assertEqual(len(calibration["camera_intrinsic"]), 9)
        self.assertTrue(all(isinstance(value, (int, float)) for value in calibration["camera_ext_t"]))

    def test_calibration_registry_helpers_work(self) -> None:
        self.assertTrue(has_calibration("AMtown01"))
        self.assertFalse(has_calibration("SEQ_AIRPORT_01"))
        self.assertIn("HKairport01", list_calibrated_scenes())

    def test_make_placeholder_frame_uses_placeholder_paths(self) -> None:
        frame = make_placeholder_frame(sequence_id="SEQ_AIRPORT_01", frame_id="placeholder_0001")
        self.assertTrue(frame.is_placeholder)
        self.assertTrue(frame.rgb_frame_path.startswith("placeholder://rgb/"))
        self.assertEqual(frame.geometry_anchor["type"], "placeholder_waypoint")

    def test_make_placeholder_sequence_creates_requested_frame_count(self) -> None:
        sequence = make_placeholder_sequence(sequence_id="SEQ_TOWN_01", scene_family="AMtown", frame_count=3)
        self.assertTrue(sequence.placeholder_state.enabled)
        self.assertEqual(len(sequence.frames), 3)
        self.assertTrue(all(frame.is_placeholder for frame in sequence.frames))

    def test_indexer_uses_placeholder_mode_without_root(self) -> None:
        indexer = UAVScenesIndexer()
        manifests = indexer.build_manifest()
        self.assertEqual(len(manifests), 4)
        self.assertTrue(all(manifest.placeholder_state.enabled for manifest in manifests))

    def test_indexer_scans_existing_root_in_placeholder_directory_mode(self) -> None:
        root = make_workspace_temp_dir("indexer_root")
        try:
            (root / "HKairport01").mkdir()
            (root / "AMtown01").mkdir()

            indexer = UAVScenesIndexer(str(root))
            manifests = indexer.build_manifest()

            sequence_ids = {manifest.sequence_id for manifest in manifests}
            self.assertEqual(sequence_ids, {"HKairport01", "AMtown01"})
            self.assertTrue(all(manifest.placeholder_state.enabled for manifest in manifests))
            airport_manifest = next(manifest for manifest in manifests if manifest.sequence_id == "HKairport01")
            self.assertTrue(airport_manifest.calibration_available)
            self.assertEqual(airport_manifest.metadata["source_domain_bucket_hint"], "airport_like")
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


