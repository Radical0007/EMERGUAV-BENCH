from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PlaceholderState:
    enabled: bool
    reason: str = ""


@dataclass(slots=True)
class FrameManifest:
    sequence_id: str
    frame_id: str
    rgb_frame_path: str
    lidar_frame_path: str
    map_id: str
    geometry_anchor: dict[str, Any]
    is_placeholder: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SequenceManifest:
    sequence_id: str
    scene_family: str
    frames: list[FrameManifest]
    placeholder_state: PlaceholderState
    calibration_available: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def make_placeholder_frame(sequence_id: str, frame_id: str = "placeholder_0001") -> FrameManifest:
    return FrameManifest(
        sequence_id=sequence_id,
        frame_id=frame_id,
        rgb_frame_path=f"placeholder://rgb/{sequence_id}/{frame_id}.jpg",
        lidar_frame_path=f"placeholder://lidar/{sequence_id}/{frame_id}.txt",
        map_id=f"placeholder://map/{sequence_id}",
        geometry_anchor={
            "type": "placeholder_waypoint",
            "sequence_id": sequence_id,
            "frame_id": frame_id,
        },
        is_placeholder=True,
        metadata={"placeholder_source": True},
    )


def make_placeholder_sequence(
    *,
    sequence_id: str,
    scene_family: str,
    frame_count: int = 1,
    calibration_available: bool = False,
    metadata: dict[str, Any] | None = None,
) -> SequenceManifest:
    frames = [
        make_placeholder_frame(sequence_id=sequence_id, frame_id=f"placeholder_{index + 1:04d}")
        for index in range(frame_count)
    ]
    return SequenceManifest(
        sequence_id=sequence_id,
        scene_family=scene_family,
        frames=frames,
        placeholder_state=PlaceholderState(enabled=True, reason="placeholder manifest"),
        calibration_available=calibration_available,
        metadata=dict(metadata or {}),
    )
