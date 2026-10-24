from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from emerguav_bench.config.vocab import (
    DEFAULT_LANGUAGE_ENTROPY_LABEL,
    resolve_language_entropy_level,
    resolve_observation_severity_code,
)
from emerguav_bench.language.entropy import make_placeholder_noisy_instruction
from emerguav_bench.protocol_context.mapper import assign_protocol_context
from emerguav_bench.truth.action import derive_action_truth
from emerguav_bench.truth.observation_protocol import derive_observation_protocol_truth
from emerguav_bench.truth.intent import derive_intent_truth
from emerguav_bench.truth.tasks import derive_task_truth
from emerguav_bench.uavscenes.indexer import UAVScenesIndexer
from emerguav_bench.uavscenes.manifest import SequenceManifest
from emerguav_bench.utils.progress import make_progress_bar, progress_iter

DEFAULT_CLEAN_INSTRUCTION = "Proceed carefully while respecting mission constraints and prefer a conservative action near restricted zones."


def _default_pose() -> dict[str, dict[str, float]]:
    return {
        "position": {"x": 0.0, "y": 0.0, "z": 30.0},
        "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    }


def _pose_from_geometry_anchor(geometry_anchor: dict[str, Any]) -> dict[str, dict[str, float]]:
    matrix = geometry_anchor.get("T4x4")
    if not isinstance(matrix, list) or len(matrix) < 3:
        return _default_pose()
    try:
        r00 = float(matrix[0][0])
        r10 = float(matrix[1][0])
        r20 = float(matrix[2][0])
        r21 = float(matrix[2][1])
        r22 = float(matrix[2][2])
        x = float(matrix[0][3])
        y = float(matrix[1][3])
        z = float(matrix[2][3])
    except (TypeError, ValueError, IndexError):
        return _default_pose()

    clamped = max(-1.0, min(1.0, -r20))
    pitch = math.asin(clamped)
    roll = math.atan2(r21, r22)
    yaw = math.atan2(r10, r00)
    return {
        "position": {"x": x, "y": y, "z": z},
        "orientation": {"roll": roll, "pitch": pitch, "yaw": yaw},
    }


def _uavscenes_metadata(manifest: SequenceManifest, frame: Any) -> dict[str, Any]:
    frame_metadata = dict(getattr(frame, "metadata", {}) or {})
    source = dict(frame_metadata.get("source", {}) or {})
    return {
        "is_placeholder": bool(getattr(frame, "is_placeholder", False) or manifest.placeholder_state.enabled),
        "sequence_id": manifest.sequence_id,
        "scene_family": manifest.scene_family,
        "source_sequence_dir": manifest.metadata.get("source_sequence_dir"),
        "sampleinfos_path": manifest.metadata.get("sampleinfos_path"),
        "rtk_positions_raw_path": manifest.metadata.get("rtk_positions_raw_path"),
        "calibration": manifest.metadata.get("calibration"),
        "sample_info": frame_metadata.get("sample_info"),
        "source_paths": {
            "rgb": frame_metadata.get("source_rgb_path") or source.get("rgb_frame_path") or frame.rgb_frame_path,
            "lidar": frame_metadata.get("source_lidar_path") or source.get("lidar_frame_path") or frame.lidar_frame_path,
            "camera_label": frame_metadata.get("source_camera_label_path") or source.get("camera_label_path"),
            "lidar_label": frame_metadata.get("source_lidar_label_path") or source.get("lidar_label_path"),
        },
        "materialized_paths": {
            "rgb": frame.rgb_frame_path,
            "lidar": frame.lidar_frame_path,
            "camera_label": frame_metadata.get("camera_label_path"),
            "lidar_label": frame_metadata.get("lidar_label_path"),
            "frame_metadata": frame_metadata.get("frame_metadata_path"),
        },
    }


def build_seed_samples_from_manifest(
    manifests: list[SequenceManifest],
    *,
    protocol_context: str | None = None,
    application_theme: str | None = None,
    policy_family: str | None = None,
    overlay_profiles: list[str] | None = None,
    enable_nfz_overlay: bool = False,
    degradation_label: str = "dust_veil",
    degradation_severity: str = "high",
    language_entropy_labels: list[str] | None = None,
    language_entropy_level: str | None = None,
    observation_severity_code: str | None = None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    resolved_language_entropy_labels = list(language_entropy_labels or [DEFAULT_LANGUAGE_ENTROPY_LABEL])

    total_frames = sum(len(manifest.frames) for manifest in manifests)
    with make_progress_bar(total=total_frames, desc="build samples", unit="sample") as progress_bar:
        for manifest in manifests:
            protocol_assignment = assign_protocol_context(
                sequence_id=manifest.sequence_id,
                scene_family=manifest.scene_family,
                protocol_context=protocol_context,
                application_theme=application_theme,
                policy_family=policy_family,
                enable_nfz_overlay=enable_nfz_overlay,
                overlay_profiles=overlay_profiles,
            )

            for frame in manifest.frames:
                sample_id = f"{manifest.sequence_id}::{frame.frame_id}"
                target_goal = protocol_assignment.protocol_metadata["default_target_goal"]
                clean_instruction = f"{DEFAULT_CLEAN_INSTRUCTION} Mission objective: {target_goal}."
                noisy_instruction = make_placeholder_noisy_instruction(clean_instruction, resolved_language_entropy_labels)
                constraints = list(protocol_assignment.mission_constraints)
                observation_protocol_truth = derive_observation_protocol_truth(
                    degradation_labels=[degradation_label],
                    degradation_severity=degradation_severity,
                    source_domain_bucket=protocol_assignment.source_domain_bucket,
                    protocol_context=protocol_assignment.protocol_context,
                    application_theme=protocol_assignment.application_theme,
                    policy_family=protocol_assignment.policy_family,
                    overlay_profiles=protocol_assignment.overlay_profiles,
                    enable_nfz_overlay=enable_nfz_overlay,
                )
                intent_truth = derive_intent_truth(
                    clean_instruction=clean_instruction,
                    mission_constraints=constraints,
                    protocol_context=protocol_assignment.protocol_context,
                    application_theme=protocol_assignment.application_theme,
                    policy_family=protocol_assignment.policy_family,
                    target_goal=target_goal,
                )
                action_truth = derive_action_truth(
                    source_domain_bucket=protocol_assignment.source_domain_bucket,
                    risk_level=observation_protocol_truth["risk_level"],
                    trusted_modality=observation_protocol_truth["trusted_modality"],
                    protocol_context=protocol_assignment.protocol_context,
                    application_theme=protocol_assignment.application_theme,
                    policy_family=protocol_assignment.policy_family,
                    overlay_profiles=protocol_assignment.overlay_profiles,
                    enable_nfz_overlay=enable_nfz_overlay,
                )
                paper_task_truth = derive_task_truth(
                    observation_protocol_truth=observation_protocol_truth,
                    intent_truth=intent_truth,
                    action_truth=action_truth,
                )

                sample = {
                    "sample_id": sample_id,
                    "source_domain_bucket": protocol_assignment.source_domain_bucket,
                    "protocol_context": protocol_assignment.protocol_context,
                    "application_theme": protocol_assignment.application_theme,
                    "policy_family": protocol_assignment.policy_family,
                    "sequence_id": manifest.sequence_id,
                    "frame_id": frame.frame_id,
                    "rgb_frame_path": frame.rgb_frame_path,
                    "lidar_frame_path": frame.lidar_frame_path,
                    "pose_6dof": _pose_from_geometry_anchor(frame.geometry_anchor),
                    "map_id": frame.map_id,
                    "degradation_labels": [degradation_label],
                    "degradation_severity": degradation_severity,
                    "observation_severity_code": observation_severity_code
                    or resolve_observation_severity_code(degradation_severity, [degradation_label]),
                    "language_entropy_labels": list(resolved_language_entropy_labels),
                    "language_entropy_level": resolve_language_entropy_level(
                        resolved_language_entropy_labels,
                        requested_level=language_entropy_level,
                    ),
                    "clean_instruction": clean_instruction,
                    "noisy_instruction": noisy_instruction,
                    "mission_constraints": constraints,
                    "observation_protocol_gt": observation_protocol_truth,
                    "intent_extraction_gt": intent_truth,
                    "action_token_gt": action_truth["action_token_gt"],
                    "acceptable_safe_alternatives": action_truth["acceptable_safe_alternatives"],
                    "forbidden_actions": action_truth["forbidden_actions"],
                    "risk_level_gt": observation_protocol_truth["risk_level"],
                    "collaboration_gt": action_truth["collaboration_gt"],
                    "paper_task_gt": paper_task_truth,
                    "geometry_anchor": frame.geometry_anchor,
                    "protocol_metadata": {
                        **protocol_assignment.protocol_metadata,
                        "scene_family": manifest.scene_family,
                        "placeholder_source": manifest.placeholder_state.enabled,
                        "calibration_available": manifest.calibration_available,
                        "overlay_profiles": list(protocol_assignment.overlay_profiles),
                        "uavscenes": _uavscenes_metadata(manifest, frame),
                    },
                }
                samples.append(sample)
                progress_bar.update(1)

    return samples


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_prediction_stub(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions = []
    for sample in progress_iter(samples, desc="prediction stubs", total=len(samples), unit="sample"):
        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "output": {
                    "Observation_Protocol_Assessment": dict(sample["observation_protocol_gt"]),
                    "Intent_Extraction": dict(sample["intent_extraction_gt"]),
                    "Action_Output": {
                        "action_token": sample["action_token_gt"],
                        "confidence": 0.5,
                        "abstention_decision": sample["paper_task_gt"]["T11"],
                        "collaboration_request": dict(sample["collaboration_gt"]),
                    },
                    "Rationale_Short": "placeholder prediction stub",
                },
            }
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a placeholder-compatible EmergUAV-Bench seed set.")
    parser.add_argument("--uavscenes-root", default=None, help="Optional external UAVSCENES_ROOT directory")
    parser.add_argument("--sequence-id", action="append", dest="sequence_ids", help="Sequence ID to include; repeatable")
    parser.add_argument("--protocol-context", default=None, help="Optional protocol context override")
    parser.add_argument("--application-theme", default=None, help="Optional application theme override")
    parser.add_argument("--policy-family", default=None, help="Optional policy family override")
    parser.add_argument("--overlay-profile", action="append", dest="overlay_profiles", help="Optional overlay profile; repeatable")
    parser.add_argument("--enable-nfz-overlay", action="store_true", help="Attach NFZ Edge Caution overlay")
    parser.add_argument("--degradation-label", default="dust_veil", help="Primary degradation label")
    parser.add_argument("--degradation-severity", default="high", help="Degradation severity")
    parser.add_argument(
        "--language-entropy-label",
        action="append",
        dest="language_entropy_labels",
        help="Language entropy label; repeatable",
    )
    parser.add_argument("--language-entropy-level", default=None, help="Language entropy level placeholder")
    parser.add_argument("--observation-severity-code", default=None, help="Optional observation severity code placeholder")
    parser.add_argument("--samples-out", required=True, help="Output path for generated samples JSON")
    parser.add_argument("--predictions-out", help="Optional output path for prediction stub JSON")
    args = parser.parse_args()

    manifests = UAVScenesIndexer(args.uavscenes_root).build_manifest(sequence_ids=args.sequence_ids)
    samples = build_seed_samples_from_manifest(
        manifests,
        protocol_context=args.protocol_context,
        application_theme=args.application_theme,
        policy_family=args.policy_family,
        overlay_profiles=args.overlay_profiles,
        enable_nfz_overlay=args.enable_nfz_overlay,
        degradation_label=args.degradation_label,
        degradation_severity=args.degradation_severity,
        language_entropy_labels=args.language_entropy_labels,
        language_entropy_level=args.language_entropy_level,
        observation_severity_code=args.observation_severity_code,
    )
    save_json(args.samples_out, samples)

    if args.predictions_out:
        save_json(args.predictions_out, build_prediction_stub(samples))


if __name__ == "__main__":
    main()
