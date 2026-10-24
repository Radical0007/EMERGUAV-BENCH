from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from emerguav_bench.protocol_context.mapper import SCENE_FAMILY_TO_SOURCE_DOMAIN_BUCKET
from emerguav_bench.uavscenes.calibration import get_calibration, has_calibration
from emerguav_bench.uavscenes.manifest import (
    FrameManifest,
    PlaceholderState,
    SequenceManifest,
    make_placeholder_sequence,
)
from emerguav_bench.utils.progress import progress_iter

DEFAULT_PLACEHOLDER_SEQUENCES = [
    ("SEQ_AIRPORT_01", "HKairport"),
    ("SEQ_TOWN_01", "AMtown"),
    ("SEQ_ISLAND_01", "HKisland"),
    ("SEQ_VALLEY_01", "AMvalley"),
]


def _normalize_sequence_id(path_name: str) -> str:
    return path_name.removeprefix("interval1_")


def _derive_scene_family(sequence_id: str) -> str:
    if sequence_id.startswith("AMtown"):
        return "AMtown"
    if sequence_id.startswith("AMvalley"):
        return "AMvalley"
    if sequence_id.startswith("HKairport"):
        return "HKairport"
    if sequence_id.startswith("HKisland"):
        return "HKisland"
    match = re.match(r"SEQ_([A-Z]+)_", sequence_id)
    if match:
        token = match.group(1).lower()
        if token == "airport":
            return "HKairport"
        if token == "town":
            return "AMtown"
        if token == "island":
            return "HKisland"
        if token == "valley":
            return "AMvalley"
    raise KeyError(f"Cannot infer UAVScenes scene family from sequence id: {sequence_id}")


def _is_label_root(path: Path) -> bool:
    return path.name in {"interval1_CAM_label", "interval1_LIDAR_label"}


def _is_real_sequence_dir(path: Path) -> bool:
    return (path / "interval1_CAM").is_dir() and (path / "sampleinfos_interpolated.json").is_file()


def _load_sample_infos(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in UAVScenes sample info file: {path}")
    return [dict(item) for item in payload if isinstance(item, dict)]


def _find_single_match(directory: Path, pattern: str) -> Path | None:
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def _distortion_from_sample_info(sample_info: dict[str, Any]) -> dict[str, float | None]:
    return {
        "K1": sample_info.get("K1"),
        "K2": sample_info.get("K2"),
        "K3": sample_info.get("K3"),
        "P1": sample_info.get("P1"),
        "P2": sample_info.get("P2"),
    }


def _geometry_anchor_from_sample_info(sequence_id: str, sample_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "uavscenes_frame",
        "source_dataset": "UAVScenes",
        "sequence_id": sequence_id,
        "original_scene_name": sample_info.get("OriginalSceneName"),
        "original_image_name": sample_info.get("OriginalImageName"),
        "sorted_image_id": sample_info.get("SortedImageID"),
        "T4x4": sample_info.get("T4x4"),
        "P3x3": sample_info.get("P3x3"),
        "distortion": _distortion_from_sample_info(sample_info),
        "image_size": {
            "width": sample_info.get("Width"),
            "height": sample_info.get("Height"),
        },
    }


class UAVScenesIndexer:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else None

    def build_manifest(self, sequence_ids: list[str] | None = None) -> list[SequenceManifest]:
        requested = set(sequence_ids or [])
        if self.root is None:
            return self._build_default_placeholder_manifest(requested=requested)
        return self._build_root_placeholder_manifest(requested=requested)

    def _build_default_placeholder_manifest(self, *, requested: set[str]) -> list[SequenceManifest]:
        manifests: list[SequenceManifest] = []
        for sequence_id, scene_family in DEFAULT_PLACEHOLDER_SEQUENCES:
            if requested and sequence_id not in requested:
                continue
            manifests.append(
                make_placeholder_sequence(
                    sequence_id=sequence_id,
                    scene_family=scene_family,
                    metadata={"source_domain_bucket_hint": SCENE_FAMILY_TO_SOURCE_DOMAIN_BUCKET[scene_family]},
                )
            )
        return manifests

    def _build_root_placeholder_manifest(self, *, requested: set[str]) -> list[SequenceManifest]:
        if self.root is None or not self.root.exists():
            return self._build_default_placeholder_manifest(requested=requested)

        manifests: list[SequenceManifest] = []
        children = sorted(item for item in self.root.iterdir() if item.is_dir())
        for child in progress_iter(children, desc="index sequences", total=len(children), unit="sequence"):
            if _is_label_root(child):
                continue
            sequence_id = _normalize_sequence_id(child.name)
            if requested and sequence_id not in requested and child.name not in requested:
                continue
            if _is_real_sequence_dir(child):
                manifests.append(self._build_real_sequence_manifest(child, sequence_id=sequence_id))
                continue
            scene_family = _derive_scene_family(sequence_id)
            manifests.append(
                make_placeholder_sequence(
                    sequence_id=sequence_id,
                    scene_family=scene_family,
                    calibration_available=has_calibration(sequence_id),
                    metadata={
                        "source_domain_bucket_hint": SCENE_FAMILY_TO_SOURCE_DOMAIN_BUCKET[scene_family],
                        "source_sequence_dir": str(child),
                    },
                )
            )
        return manifests

    def _build_real_sequence_manifest(self, sequence_dir: Path, *, sequence_id: str) -> SequenceManifest:
        if self.root is None:
            raise ValueError("Cannot build real UAVScenes manifest without a dataset root")

        scene_family = _derive_scene_family(sequence_id)
        sample_infos = _load_sample_infos(sequence_dir / "sampleinfos_interpolated.json")
        frames: list[FrameManifest] = []
        cam_label_dir = self.root / "interval1_CAM_label" / sequence_dir.name / "interval1_CAM_label_color"
        lidar_label_dir = self.root / "interval1_LIDAR_label" / sequence_dir.name / "interval1_LIDAR_label_color"
        lidar_dir = sequence_dir / "interval1_LIDAR"

        sorted_sample_infos = sorted(sample_infos, key=lambda item: int(item.get("SortedImageID", 0)))
        for fallback_index, sample_info in enumerate(
            progress_iter(
                sorted_sample_infos,
                desc=f"index {sequence_id}",
                total=len(sorted_sample_infos),
                unit="frame",
            )
        ):
            image_name = str(sample_info.get("OriginalImageName", ""))
            if not image_name:
                continue
            rgb_path = sequence_dir / "interval1_CAM" / image_name
            if not rgb_path.is_file():
                continue
            image_timestamp = Path(image_name).stem
            lidar_path = _find_single_match(lidar_dir, f"image{image_timestamp}_lidar*.txt")
            cam_label_path = cam_label_dir / f"{image_timestamp}.png"
            lidar_label_path = _find_single_match(lidar_label_dir, f"image{image_timestamp}_lidar*.txt")
            sorted_image_id = int(sample_info.get("SortedImageID", fallback_index))
            frame_id = f"{sorted_image_id:06d}"
            frames.append(
                FrameManifest(
                    sequence_id=sequence_id,
                    frame_id=frame_id,
                    rgb_frame_path=str(rgb_path),
                    lidar_frame_path=str(lidar_path) if lidar_path else f"missing://lidar/{sequence_id}/{frame_id}",
                    map_id=f"uavscenes://map/{sequence_id}",
                    geometry_anchor=_geometry_anchor_from_sample_info(sequence_id, sample_info),
                    is_placeholder=False,
                    metadata={
                        "placeholder_source": False,
                        "source_sequence_dir": str(sequence_dir),
                        "source_rgb_path": str(rgb_path),
                        "source_lidar_path": str(lidar_path) if lidar_path else None,
                        "camera_label_path": str(cam_label_path) if cam_label_path.is_file() else None,
                        "lidar_label_path": str(lidar_label_path) if lidar_label_path else None,
                        "sample_info": sample_info,
                    },
                )
            )

        return SequenceManifest(
            sequence_id=sequence_id,
            scene_family=scene_family,
            frames=frames,
            placeholder_state=PlaceholderState(enabled=False, reason="real UAVScenes assets indexed"),
            calibration_available=has_calibration(sequence_id),
            metadata={
                "source_domain_bucket_hint": SCENE_FAMILY_TO_SOURCE_DOMAIN_BUCKET[scene_family],
                "source_sequence_dir": str(sequence_dir),
                "sampleinfos_path": str(sequence_dir / "sampleinfos_interpolated.json"),
                "rtk_positions_raw_path": str(sequence_dir / "rtk_positions_raw.csv"),
                "calibration": get_calibration(sequence_id) if has_calibration(sequence_id) else None,
            },
        )
