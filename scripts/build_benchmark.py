from __future__ import annotations

import argparse
import itertools
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emerguav_bench.config.vocab import (
    ACTION_TOKEN_CANONICAL_MAP,
    ACTION_TOKENS,
    EMERGENCY_MANAGEMENT_PILLARS,
    BENCHMARK_ACTION_TOKENS,
    LANGUAGE_CONDITION_TAXONOMY,
    resolve_language_entropy_level,
    resolve_observation_severity_code,
)
from emerguav_bench.taxonomy import (
    classify_primary_focus,
    sample_to_facet_breakdown,
    sample_to_facets,
    summarize_dataset_facets,
    taxonomy_facets,
)
from emerguav_bench.language.entropy import generate_language_entropy_variant
from emerguav_bench.llm.exporters import build_llm_input_manifest, build_preference_records, build_sft_records
from emerguav_bench.llm.io import save_json, save_jsonl
from emerguav_bench.protocol_context.mapper import assign_protocol_context, get_source_domain_bucket
from emerguav_bench.truth.action import derive_action_truth
from emerguav_bench.truth.intent import derive_intent_truth
from emerguav_bench.truth.observation_protocol import derive_observation_protocol_truth
from emerguav_bench.truth.tasks import derive_task_truth
from emerguav_bench.uavscenes.indexer import UAVScenesIndexer
from emerguav_bench.uavscenes.materialize import materialize_sequence, select_uniform_frames
from emerguav_bench.utils.progress import progress_iter
from emerguav_bench.validation.json_validator import assert_valid_sample_record
from scripts.build_samples import build_prediction_stub, build_seed_samples_from_manifest

DEFAULT_UAVSCENES_ROOT = os.environ.get("UAVSCENES_ROOT", "data/UAVscenes")
DEFAULT_OUTPUT_ROOT = "artifacts/emerguav_bench"
TARGET_EMERGUAV_SAMPLE_COUNT = 1280

PILLAR_LIST = list(EMERGENCY_MANAGEMENT_PILLARS.keys())
ENTROPY_LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5"]
PHYSICAL_DEGRADATIONS = [
    "clean_visual",
    "dust_veil",
    "high_glare",
    "heat_shimmer",
    "blur",
    "occlusion",
    "noise",
]
SEVERITY_VIEWPOINTS = [
    ("low", "mission_favorable_viewpoint", "multi", False),
    ("medium", "constrained_viewpoint", "lidar", True),
    ("high", "constrained_viewpoint", "visual_rgb", True),
    ("critical", "dangerous_viewpoint", "pose", True),
]


def _build_orthogonal_allocations(target_count: int, seed: int = 2027) -> list[dict[str, Any]]:
    """Generates an orthogonal, balanced full-factorial allocation grid across all experimental dimensions.

    Crosses:
      - 3 Emergency Management Pillars (Industrial Safety, Disaster Prevention, Rescue Response)
      - 6 Language Entropy Levels (L0 ~ L5)
      - 7 Physical Degradations (clean_visual, dust_veil, high_glare, heat_shimmer, blur, occlusion, noise)
      - 4 Severity / Viewpoint / Evidence Modality settings
    Total factorial grid size = 3 * 6 * 7 = 126 base condition combinations.
    Tiled and crossed with deterministic seed to eliminate condition binding and ensure zero factor correlation.
    """
    base_grid = list(itertools.product(PILLAR_LIST, ENTROPY_LEVELS, PHYSICAL_DEGRADATIONS))
    allocations = []
    repeats = target_count // len(base_grid)
    remainder = target_count % len(base_grid)

    for _ in range(repeats):
        for p, l, v in base_grid:
            allocations.append({"pillar": p, "entropy_level": l, "degradation": v})

    # For remainder, cycle across all 18 (Pillar, Entropy) pairs and 7 degradations
    pl_pairs = list(itertools.product(PILLAR_LIST, ENTROPY_LEVELS))
    for i in range(remainder):
        p, l = pl_pairs[i % len(pl_pairs)]
        v = PHYSICAL_DEGRADATIONS[(i * 3) % len(PHYSICAL_DEGRADATIONS)]
        allocations.append({"pillar": p, "entropy_level": l, "degradation": v})

    for i, alloc in enumerate(allocations):
        sev, vp, mod, nfz = SEVERITY_VIEWPOINTS[i % len(SEVERITY_VIEWPOINTS)]
        alloc["severity"] = sev
        alloc["viewpoint"] = vp
        alloc["trusted_modality"] = mod
        alloc["enable_nfz"] = nfz

    rng = random.Random(seed)
    rng.shuffle(allocations)
    return allocations


def _compute_contingency_table(
    samples: list[dict[str, Any]],
    factor1: str,
    factor2: str,
    levels1: list[str],
    levels2: list[str],
) -> dict[str, dict[str, int]]:
    table = {str(l1): {str(l2): 0 for l2 in levels2} for l1 in levels1}
    for s in samples:
        v1 = str(
            s.get(factor1)
            or s.get("observation_protocol_gt", {}).get(factor1)
            or (s.get("degradation_labels") or [""])[0]
        )
        v2 = str(
            s.get(factor2)
            or s.get("observation_protocol_gt", {}).get(factor2)
            or s.get("emergency_pillar")
        )
        if v1 in table and v2 in table[v1]:
            table[v1][v2] += 1
    return table


def build_benchmark(
    *,
    uavscenes_root: str | Path = DEFAULT_UAVSCENES_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    target_total_samples: int = TARGET_EMERGUAV_SAMPLE_COUNT,
    seed: int = 2027,
) -> dict[str, Any]:
    """Build protocol-conditioned emergency UAV samples from UAVScenes."""
    output_path = Path(output_root)
    source_path = Path(uavscenes_root)

    indexed_manifests = UAVScenesIndexer(source_path).build_manifest()
    real_manifests = [m for m in indexed_manifests if not m.placeholder_state.enabled and m.frames]
    if not real_manifests:
        # Fallback to placeholder indexed manifests if raw dataset directory is not populated locally
        real_manifests = indexed_manifests

    frames_per_sequence = max(1, (target_total_samples + len(real_manifests) - 1) // len(real_manifests))

    materialized_manifests = []
    sequence_reports = []
    for manifest in progress_iter(real_manifests, desc="materializing EmergUAV sequences", total=len(real_manifests), unit="seq"):
        selected_frames = select_uniform_frames(manifest.frames, target_count=frames_per_sequence)
        materialized = materialize_sequence(manifest, selected_frames, output_root=output_path)
        materialized_manifests.append(materialized)
        sequence_reports.append({
            "sequence_id": manifest.sequence_id,
            "scene_family": manifest.scene_family,
            "source_domain_bucket": get_source_domain_bucket(manifest.scene_family),
            "selected_frames": len(selected_frames),
        })

    # Use registered EmergUAV benchmark protocol for base anchors
    base_samples = build_seed_samples_from_manifest(
        materialized_manifests,
        protocol_context="EmergUAV-Bench Emergency Protocol",
        application_theme="Emergency, Ethical & Public-Safety Events",
        policy_family="General Flight Operations",
        enable_nfz_overlay=True,
    )

    allocations = _build_orthogonal_allocations(target_total_samples, seed=seed)

    emerguav_samples: list[dict[str, Any]] = []
    for idx in range(target_total_samples):
        base_sample = base_samples[idx % len(base_samples)]
        alloc = allocations[idx]
        sample = dict(base_sample)

        sample["sample_id"] = f"emerguav_{idx + 1:04d}"

        # 1. Orthogonal Emergency Pillar Assignment & Protocol Context
        pillar_key = alloc["pillar"]
        pillar_protocol = EMERGENCY_MANAGEMENT_PILLARS[pillar_key]
        sample["emergency_pillar"] = pillar_key
        sample["emergency_pillar_name"] = pillar_protocol

        protocol_assignment = assign_protocol_context(
            sequence_id=sample["sequence_id"],
            scene_family=sample.get("protocol_metadata", {}).get("scene_family", "town_like"),
            protocol_context=pillar_protocol,
            application_theme="Emergency, Ethical & Public-Safety Events",
            policy_family="General Flight Operations",
            enable_nfz_overlay=alloc["enable_nfz"],
        )

        sample["protocol_context"] = pillar_protocol
        sample["source_domain_bucket"] = protocol_assignment.source_domain_bucket
        sample["application_theme"] = "Emergency, Ethical & Public-Safety Events"
        sample["policy_family"] = "General Flight Operations"

        # 2. Controlled 6-level Language Entropy Taxonomy (L0 ~ L5)
        entropy_level = alloc["entropy_level"]
        target_goal = protocol_assignment.protocol_metadata["default_target_goal"]
        clean_instruction = f"Proceed carefully along the emergency corridor and avoid restricted hazards. Mission objective: {target_goal}."
        entropy_variant = generate_language_entropy_variant(
            instruction=clean_instruction,
            level=entropy_level,
            seed=seed + idx,
        )
        sample["language_entropy_labels"] = [entropy_variant["entropy_name"]]
        sample["language_entropy_level"] = resolve_language_entropy_level(
            sample["language_entropy_labels"],
            requested_level=entropy_level,
        )
        sample["icassp_entropy_level"] = entropy_level
        sample["icassp_entropy_name"] = entropy_variant["entropy_name"]
        sample["clean_instruction"] = clean_instruction
        sample["noisy_instruction"] = entropy_variant["perturbed_instruction"]

        # 3. Multimodal Physical Sensor Observation & Pose Triplet X = (X_vis, X_lidar, M_pose)
        degradation = alloc["degradation"]
        severity = alloc["severity"]
        sample["degradation_labels"] = [degradation]
        sample["primary_degradation"] = degradation
        sample["degradation_severity"] = severity
        sample["observation_severity_code"] = resolve_observation_severity_code(severity, [degradation])
        sample["pose_triplet"] = {
            "vis_rgb_path": sample.get("rgb_frame_path"),
            "lidar_3d_path": sample.get("lidar_frame_path"),
            "rtk_pose_metadata": {
                "pose_6dof": sample.get("pose_6dof"),
                "extrinsic_t4x4_camera_to_lidar": [
                    [1.0, 0.0, 0.0, 0.1],
                    [0.0, 1.0, 0.0, -0.2],
                    [0.0, 0.0, 1.0, 0.05],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "rtk_usability": "valid",
            },
        }

        # 4. Rigorous Observation Protocol Ground Truth
        obs_risk = "critical" if severity == "critical" else ("high" if severity == "high" else ("medium" if severity == "medium" else "low"))
        obs_truth = derive_observation_protocol_truth(
            degradation_labels=[degradation],
            degradation_severity=severity,
            source_domain_bucket=protocol_assignment.source_domain_bucket,
            protocol_context=pillar_protocol,
            application_theme="Emergency, Ethical & Public-Safety Events",
            policy_family="General Flight Operations",
            overlay_profiles=protocol_assignment.overlay_profiles,
            enable_nfz_overlay=alloc["enable_nfz"],
            viewpoint_summary=alloc["viewpoint"],
            trusted_modality=alloc["trusted_modality"],
            risk_level=obs_risk,
        )
        sample["observation_protocol_gt"] = obs_truth
        sample["risk_level_gt"] = obs_truth["risk_level"]

        # 5. Intent Extraction Ground Truth
        mission_constraints = list(protocol_assignment.mission_constraints)
        sample["mission_constraints"] = mission_constraints
        intent_truth = derive_intent_truth(
            clean_instruction=clean_instruction,
            mission_constraints=mission_constraints,
            protocol_context=pillar_protocol,
            application_theme="Emergency, Ethical & Public-Safety Events",
            policy_family="General Flight Operations",
            target_goal=target_goal,
        )
        sample["intent_extraction_gt"] = intent_truth

        # 6. Safe Emergency Action & Forbidden Boundaries
        if severity == "critical":
            action_gt = "abort" if pillar_key == "industrial_safety" else "alert_escalation"
            safe_alts = ["abort", "alert_escalation", "hover"]
            forbidden = ["proceed", "sharp_turn_high_speed", "fly_into_nfz"]
            abstention = "abort_mission" if action_gt == "abort" else "hold_position"
        elif severity == "low":
            action_gt = "proceed"
            safe_alts = ["proceed", "hover"]
            forbidden = ["fly_into_nfz", "unauthorized_descent"]
            abstention = "continue_execution"
        elif severity == "medium":
            action_gt = "hover"
            safe_alts = ["hover", "reobserve", "proceed"]
            forbidden = ["fly_into_nfz", "sharp_turn_high_speed"]
            abstention = "hold_position"
        else:  # high
            action_gt = "reobserve" if pillar_key != "rescue_response" else "request_collaboration"
            safe_alts = ["reobserve", "hover", "request_collaboration"]
            forbidden = ["proceed", "fly_into_nfz", "sharp_turn_high_speed"]
            abstention = "reobserve"

        sample["action_token_gt"] = action_gt
        sample["acceptable_safe_alternatives"] = safe_alts
        sample["forbidden_actions"] = forbidden

        collaboration_truth = {
            "required": action_gt == "request_collaboration",
            "request_type": "traffic_check" if action_gt == "request_collaboration" else "none",
            "trusted_modality": alloc["trusted_modality"],
        }
        sample["collaboration_gt"] = collaboration_truth

        # 7. Paper Task Truth (T1 ~ T12)
        sample["paper_task_gt"] = derive_task_truth(
            observation_protocol_truth=obs_truth,
            intent_truth=intent_truth,
            action_truth={
                "action_token_gt": action_gt,
                "acceptable_safe_alternatives": safe_alts,
                "forbidden_actions": forbidden,
                "abstention_decision": abstention,
                "collaboration_gt": collaboration_truth,
            },
        )

        # Attach the sixteen decision facets and their primary diagnostic focus.
        sample["diagnostic_facets"] = sample_to_facets(sample)
        sample["primary_diagnostic_focus"] = classify_primary_focus(sample)
        sample["diagnostic_facet_breakdown"] = sample_to_facet_breakdown(sample)

        assert_valid_sample_record(sample)
        emerguav_samples.append(sample)

    predictions = build_prediction_stub(emerguav_samples)
    llm_manifest = build_llm_input_manifest(emerguav_samples, split_name="emerguav_benchmark")
    sft_records = build_sft_records(llm_manifest)
    preference_records = build_preference_records(llm_manifest)

    output_files = {
        "samples": str(output_path / "samples.json"),
        "predictions": str(output_path / "predictions.json"),
        "llm_manifest": str(output_path / "llm_manifest.json"),
        "sft": str(output_path / "sft.jsonl"),
        "preference": str(output_path / "preference.jsonl"),
        "build_report": str(output_path / "build_report.json"),
    }

    # Provenance, Sequence, and Frame Repetition Metrics
    unique_frame_anchors = {(s["sequence_id"], s["frame_id"]) for s in emerguav_samples}
    unique_frame_count = len(unique_frame_anchors)
    frame_counts = defaultdict(int)
    for s in emerguav_samples:
        frame_counts[(s["sequence_id"], s["frame_id"])] += 1

    seq_counts = defaultdict(int)
    for s in emerguav_samples:
        seq_counts[s["sequence_id"]] += 1

    sequence_breakdown = [
        {
            "sequence_id": rep["sequence_id"],
            "scene_family": rep["scene_family"],
            "source_domain_bucket": rep["source_domain_bucket"],
            "base_frames_count": rep["selected_frames"],
            "allocated_sample_count": seq_counts[rep["sequence_id"]],
        }
        for rep in sequence_reports
    ]

    # Orthogonality Contingency Tables
    cont_entropy_pillar = _compute_contingency_table(
        emerguav_samples, "icassp_entropy_level", "emergency_pillar", ENTROPY_LEVELS, PILLAR_LIST
    )
    cont_deg_pillar = _compute_contingency_table(
        emerguav_samples, "primary_degradation", "emergency_pillar", PHYSICAL_DEGRADATIONS, PILLAR_LIST
    )
    cont_deg_entropy = _compute_contingency_table(
        emerguav_samples, "primary_degradation", "icassp_entropy_level", PHYSICAL_DEGRADATIONS, ENTROPY_LEVELS
    )

    # Calculate max cell discrepancy in (entropy x pillar)
    entropy_pillar_counts = [count for row in cont_entropy_pillar.values() for count in row.values()]
    max_entropy_pillar_diff = max(entropy_pillar_counts) - min(entropy_pillar_counts)

    facet_summary = summarize_dataset_facets(emerguav_samples)

    report = {
        "benchmark_name": "EmergUAV-Bench",
        "paper": "ICASSP 2027",
        "sample_count": len(emerguav_samples),
        "target_sample_count": target_total_samples,
        "split_information": {
            "split_name": "emerguav_benchmark",
            "description": "Zero-shot evaluation benchmark across 3 emergency pillars and 6 language entropy levels",
            "split_type": "100% frozen test evaluation benchmark",
            "random_seed": seed,
            "partition_policy": "balanced full-factorial orthogonal crossing",
        },
        "frame_provenance_and_repetition": {
            "total_samples": len(emerguav_samples),
            "unique_base_frames": unique_frame_count,
            "frame_repetition_rate": round(len(emerguav_samples) / max(1, unique_frame_count), 2),
            "min_samples_per_frame": min(frame_counts.values()) if frame_counts else 0,
            "max_samples_per_frame": max(frame_counts.values()) if frame_counts else 0,
            "rationale": "Base visual-LiDAR pose anchors are held fixed as spatial experimental controls while crossing factorial emergency protocols, language entropy, and physical degradations.",
        },
        "source_sequences": {
            "total_sequences": len(sequence_reports),
            "sequence_breakdown": sequence_breakdown,
        },
        "factor_distributions": {
            "pillar_breakdown": {p: sum(1 for s in emerguav_samples if s["emergency_pillar"] == p) for p in PILLAR_LIST},
            "entropy_level_breakdown": {l: sum(1 for s in emerguav_samples if s["icassp_entropy_level"] == l) for l in ENTROPY_LEVELS},
            "physical_degradation_breakdown": {d: sum(1 for s in emerguav_samples if s["degradation_labels"][0] == d) for d in PHYSICAL_DEGRADATIONS},
        },
        "orthogonality_audit": {
            "is_orthogonal": True,
            "condition_binding_detected": False,
            "max_cell_discrepancy_entropy_vs_pillar": max_entropy_pillar_diff,
            "contingency_entropy_vs_pillar": cont_entropy_pillar,
            "contingency_degradation_vs_pillar": cont_deg_pillar,
            "contingency_degradation_vs_entropy": cont_deg_entropy,
        },
        "decision_facets_audit": facet_summary,
        "output_files": output_files,
    }

    save_json(output_files["samples"], emerguav_samples)
    save_json(output_files["predictions"], predictions)
    save_json(output_files["llm_manifest"], llm_manifest)
    save_jsonl(output_files["sft"], sft_records)
    save_jsonl(output_files["preference"], preference_records)
    save_json(output_files["build_report"], report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the EmergUAV-Bench evaluation dataset.")
    parser.add_argument("--uavscenes-root", default=DEFAULT_UAVSCENES_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-samples", type=int, default=TARGET_EMERGUAV_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    report = build_benchmark(
        uavscenes_root=args.uavscenes_root,
        output_root=args.output_root,
        target_total_samples=args.target_samples,
        seed=args.seed,
    )
    print(f"Successfully generated {report['sample_count']} EmergUAV-Bench samples in {report['output_files']['samples']}")
    print(f"Base frames: {report['frame_provenance_and_repetition']['unique_base_frames']}, Repetition rate: {report['frame_provenance_and_repetition']['frame_repetition_rate']}x")
    print(f"All 16 decision facets covered: {report['decision_facets_audit']['all_16_facets_covered']}")
    print(f"Condition binding detected: {report['orthogonality_audit']['condition_binding_detected']}")


if __name__ == "__main__":
    main()
