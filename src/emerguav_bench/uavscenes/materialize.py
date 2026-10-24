from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from emerguav_bench.uavscenes.manifest import FrameManifest, PlaceholderState, SequenceManifest
from emerguav_bench.utils.progress import progress_iter


def select_uniform_frames(frames: list[FrameManifest], *, target_count: int) -> list[FrameManifest]:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if len(frames) <= target_count:
        return list(frames)
    if target_count == 1:
        return [frames[0]]

    last_index = len(frames) - 1
    selected_indices: list[int] = []
    for index in range(target_count):
        selected = round(index * last_index / (target_count - 1))
        if selected not in selected_indices:
            selected_indices.append(selected)
    return [frames[index] for index in selected_indices]


def _copy_optional_file(source: str | None, target: Path) -> str | None:
    if not source:
        return None
    source_path = Path(source)
    if not source_path.is_file():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target)
    return str(target)


def _frame_metadata_payload(sequence: SequenceManifest, frame: FrameManifest) -> dict[str, Any]:
    return {
        "sequence_id": sequence.sequence_id,
        "frame_id": frame.frame_id,
        "geometry_anchor": frame.geometry_anchor,
        "frame_metadata": frame.metadata,
        "sequence_metadata": sequence.metadata,
    }


def _write_frame_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def materialize_sequence(
    sequence: SequenceManifest,
    selected_frames: list[FrameManifest],
    *,
    output_root: str | Path,
) -> SequenceManifest:
    target_root = Path(output_root)
    materialized_frames: list[FrameManifest] = []

    for frame in progress_iter(
        selected_frames,
        desc=f"materialize {sequence.sequence_id}",
        total=len(selected_frames),
        unit="frame",
    ):
        frame_root = target_root / "assets" / sequence.sequence_id / frame.frame_id
        rgb_target = frame_root / "rgb" / Path(frame.rgb_frame_path).name
        lidar_target = frame_root / "lidar" / Path(frame.lidar_frame_path).name
        camera_label_target = frame_root / "labels" / "camera" / Path(str(frame.metadata.get("camera_label_path"))).name
        lidar_label_target = frame_root / "labels" / "lidar" / Path(str(frame.metadata.get("lidar_label_path"))).name
        metadata_target = frame_root / "metadata" / "frame_metadata.json"

        copied_rgb = _copy_optional_file(frame.rgb_frame_path, rgb_target)
        copied_lidar = _copy_optional_file(frame.lidar_frame_path, lidar_target)
        copied_camera_label = _copy_optional_file(frame.metadata.get("camera_label_path"), camera_label_target)
        copied_lidar_label = _copy_optional_file(frame.metadata.get("lidar_label_path"), lidar_label_target)
        source_metadata = {
            "sequence_id": sequence.sequence_id,
            "frame_id": frame.frame_id,
            "rgb_frame_path": frame.rgb_frame_path,
            "lidar_frame_path": frame.lidar_frame_path,
            "camera_label_path": frame.metadata.get("camera_label_path"),
            "lidar_label_path": frame.metadata.get("lidar_label_path"),
        }
        materialized_metadata = {
            **frame.metadata,
            "source": source_metadata,
            "source_rgb_path": frame.rgb_frame_path,
            "source_lidar_path": frame.lidar_frame_path,
            "source_camera_label_path": frame.metadata.get("camera_label_path"),
            "source_lidar_label_path": frame.metadata.get("lidar_label_path"),
            "camera_label_path": copied_camera_label,
            "lidar_label_path": copied_lidar_label,
            "frame_metadata_path": str(metadata_target),
            "materialized_root": str(frame_root),
        }
        materialized_frame = FrameManifest(
            sequence_id=sequence.sequence_id,
            frame_id=frame.frame_id,
            rgb_frame_path=copied_rgb or frame.rgb_frame_path,
            lidar_frame_path=copied_lidar or frame.lidar_frame_path,
            map_id=frame.map_id,
            geometry_anchor=dict(frame.geometry_anchor),
            is_placeholder=False,
            metadata=materialized_metadata,
        )
        _write_frame_metadata(metadata_target, _frame_metadata_payload(sequence, materialized_frame))
        materialized_frames.append(materialized_frame)

    return SequenceManifest(
        sequence_id=sequence.sequence_id,
        scene_family=sequence.scene_family,
        frames=materialized_frames,
        placeholder_state=PlaceholderState(enabled=False, reason="materialized UAVScenes subset"),
        calibration_available=sequence.calibration_available,
        metadata={
            **sequence.metadata,
            "materialized_output_root": str(target_root),
            "materialized_frame_count": len(materialized_frames),
        },
    )
