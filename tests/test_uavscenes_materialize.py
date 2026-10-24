import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from emerguav_bench.uavscenes.indexer import UAVScenesIndexer
from emerguav_bench.uavscenes.materialize import materialize_sequence, select_uniform_frames
from emerguav_bench.validation.json_validator import assert_valid_sample_record
from scripts.build_samples import build_seed_samples_from_manifest


def make_workspace_temp_dir(prefix: str) -> Path:
    base_dir = Path.cwd() / ".test_tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = base_dir / f"{prefix}_{uuid4().hex}"
    temp_dir.mkdir()
    return temp_dir


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_bytes(path: Path, content: bytes = b"fixture") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def make_sample_info(scene_name: str, image_name: str, sorted_id: int) -> dict:
    return {
        "OriginalSceneName": scene_name,
        "OriginalImageName": image_name,
        "SortedImageID": sorted_id,
        "T4x4": [
            [1.0, 0.0, 0.0, float(sorted_id)],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "P3x3": [[10.0, 0.0, 5.0], [0.0, 11.0, 6.0], [0.0, 0.0, 1.0]],
        "K1": 0.1,
        "K2": 0.2,
        "K3": 0.3,
        "P1": 0.4,
        "P2": 0.5,
        "Width": 2448,
        "Height": 2048,
    }


def make_fixture_uavscenes_root() -> Path:
    root = make_workspace_temp_dir("uavscenes_real")
    sequence_dir = root / "interval1_AMtown03"
    sample_infos = []

    for index in range(5):
        timestamp = f"1658131847.{index:09d}"
        lidar_timestamp = f"1658131847.{index + 100:09d}"
        image_name = f"{timestamp}.jpg"
        lidar_name = f"image{timestamp}_lidar{lidar_timestamp}.txt"

        write_bytes(sequence_dir / "interval1_CAM" / image_name)
        write_text(sequence_dir / "interval1_LIDAR" / lidar_name, "1.0 2.0 3.0\n")
        write_bytes(
            root
            / "interval1_CAM_label"
            / "interval1_AMtown03"
            / "interval1_CAM_label_color"
            / f"{timestamp}.png",
            b"label",
        )
        write_text(
            root
            / "interval1_LIDAR_label"
            / "interval1_AMtown03"
            / "interval1_LIDAR_label_color"
            / lidar_name,
            "0 0 0\n",
        )
        sample_infos.append(make_sample_info("AMtown03", image_name, index))

    write_text(sequence_dir / "sampleinfos_interpolated.json", json.dumps(sample_infos))
    write_text(sequence_dir / "rtk_positions_raw.csv", "scenename,headerstamp,lat,lon,alt,easting,northing\n")
    return root


class UAVScenesMaterializationTests(unittest.TestCase):
    def test_indexer_reads_real_frames_and_preserves_label_metadata(self) -> None:
        root = make_fixture_uavscenes_root()
        try:
            manifests = UAVScenesIndexer(root).build_manifest(sequence_ids=["AMtown03"])

            self.assertEqual(len(manifests), 1)
            manifest = manifests[0]
            self.assertFalse(manifest.placeholder_state.enabled)
            self.assertEqual(manifest.sequence_id, "AMtown03")
            self.assertEqual(manifest.scene_family, "AMtown")
            self.assertTrue(manifest.calibration_available)
            self.assertEqual(len(manifest.frames), 5)

            frame = manifest.frames[0]
            self.assertFalse(frame.is_placeholder)
            self.assertEqual(frame.frame_id, "000000")
            self.assertTrue(Path(frame.rgb_frame_path).exists())
            self.assertTrue(Path(frame.lidar_frame_path).exists())
            self.assertTrue(Path(frame.metadata["camera_label_path"]).exists())
            self.assertTrue(Path(frame.metadata["lidar_label_path"]).exists())
            self.assertEqual(frame.geometry_anchor["T4x4"][0][3], 0.0)
            self.assertEqual(frame.geometry_anchor["image_size"], {"width": 2448, "height": 2048})
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_materialize_sequence_copies_uniform_subset_and_metadata(self) -> None:
        root = make_fixture_uavscenes_root()
        output_root = make_workspace_temp_dir("uavscenes_out")
        try:
            manifest = UAVScenesIndexer(root).build_manifest(sequence_ids=["AMtown03"])[0]
            selected = select_uniform_frames(manifest.frames, target_count=3)
            materialized = materialize_sequence(manifest, selected, output_root=output_root)

            self.assertEqual([frame.frame_id for frame in selected], ["000000", "000002", "000004"])
            self.assertEqual(len(materialized.frames), 3)
            first = materialized.frames[0]
            self.assertTrue(Path(first.rgb_frame_path).exists())
            self.assertTrue(Path(first.lidar_frame_path).exists())
            self.assertTrue(Path(first.metadata["camera_label_path"]).exists())
            self.assertTrue(Path(first.metadata["lidar_label_path"]).exists())
            self.assertTrue(Path(first.metadata["frame_metadata_path"]).exists())
            self.assertEqual(first.metadata["source"]["sequence_id"], "AMtown03")
        finally:
            shutil.rmtree(root, ignore_errors=True)
            shutil.rmtree(output_root, ignore_errors=True)

    def test_materialized_manifest_builds_valid_samples_with_uavscenes_metadata(self) -> None:
        root = make_fixture_uavscenes_root()
        output_root = make_workspace_temp_dir("uavscenes_samples")
        try:
            manifest = UAVScenesIndexer(root).build_manifest(sequence_ids=["AMtown03"])[0]
            materialized = materialize_sequence(
                manifest,
                select_uniform_frames(manifest.frames, target_count=3),
                output_root=output_root,
            )

            samples = build_seed_samples_from_manifest([materialized])

            self.assertEqual(len(samples), 3)
            for sample in samples:
                assert_valid_sample_record(sample)
                self.assertEqual(sample["geometry_anchor"]["type"], "uavscenes_frame")
                uavscenes_metadata = sample["protocol_metadata"]["uavscenes"]
                self.assertFalse(uavscenes_metadata["is_placeholder"])
                self.assertTrue(Path(uavscenes_metadata["materialized_paths"]["rgb"]).exists())
                self.assertTrue(Path(uavscenes_metadata["materialized_paths"]["camera_label"]).exists())
                self.assertTrue(Path(uavscenes_metadata["materialized_paths"]["lidar_label"]).exists())
                self.assertEqual(uavscenes_metadata["sample_info"]["OriginalSceneName"], "AMtown03")
        finally:
            shutil.rmtree(root, ignore_errors=True)
            shutil.rmtree(output_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

