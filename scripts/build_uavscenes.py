from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from emerguav_bench.llm.exporters import build_llm_input_manifest, build_preference_records, build_sft_records
from emerguav_bench.llm.io import save_json, save_jsonl
from emerguav_bench.uavscenes.indexer import UAVScenesIndexer
from emerguav_bench.uavscenes.materialize import materialize_sequence, select_uniform_frames
from emerguav_bench.utils.progress import progress_iter
from emerguav_bench.validation.json_validator import assert_valid_sample_record
from scripts.build_samples import build_prediction_stub, build_seed_samples_from_manifest

DEFAULT_UAVSCENES_ROOT = os.environ.get("UAVSCENES_ROOT", "data/UAVscenes")
DEFAULT_OUTPUT_ROOT = "artifacts/uavscenes_bench"


def _real_manifests_only(manifests: list[Any]) -> list[Any]:
    return [
        manifest
        for manifest in manifests
        if not manifest.placeholder_state.enabled and manifest.frames
    ]


def build_materialized_bench_dataset(
    *,
    uavscenes_root: str | Path = DEFAULT_UAVSCENES_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    target_frames_per_sequence: int = 36,
    sequence_ids: list[str] | None = None,
    split_name: str = "dev",
    instruction_source: str = "noisy",
    context_frames: int = 2,
    min_image_inputs: int = 3,
    hardest_fraction: float = 0.3,
    protocol_context: str | None = None,
    application_theme: str | None = None,
    policy_family: str | None = None,
    enable_nfz_overlay: bool = False,
    degradation_label: str = "dust_veil",
    degradation_severity: str = "high",
    language_entropy_labels: list[str] | None = None,
) -> dict[str, Any]:
    if target_frames_per_sequence <= 0:
        raise ValueError("target_frames_per_sequence must be positive")

    output_path = Path(output_root)
    source_path = Path(uavscenes_root)
    indexed_manifests = UAVScenesIndexer(source_path).build_manifest(sequence_ids=sequence_ids)
    real_manifests = _real_manifests_only(indexed_manifests)
    if not real_manifests:
        raise ValueError(f"No real UAVScenes sequences found under {source_path}")

    materialized_manifests = []
    sequence_reports: list[dict[str, Any]] = []
    for manifest in progress_iter(real_manifests, desc="materialize sequences", total=len(real_manifests), unit="sequence"):
        selected_frames = select_uniform_frames(manifest.frames, target_count=target_frames_per_sequence)
        materialized = materialize_sequence(manifest, selected_frames, output_root=output_path)
        materialized_manifests.append(materialized)
        sequence_reports.append(
            {
                "sequence_id": manifest.sequence_id,
                "scene_family": manifest.scene_family,
                "source_frame_count": len(manifest.frames),
                "selected_frame_count": len(selected_frames),
                "materialized_frame_count": len(materialized.frames),
            }
        )

    samples = build_seed_samples_from_manifest(
        materialized_manifests,
        protocol_context=protocol_context,
        application_theme=application_theme,
        policy_family=policy_family,
        enable_nfz_overlay=enable_nfz_overlay,
        degradation_label=degradation_label,
        degradation_severity=degradation_severity,
        language_entropy_labels=language_entropy_labels,
    )
    for sample in progress_iter(samples, desc="validate samples", total=len(samples), unit="sample"):
        assert_valid_sample_record(sample)

    predictions = build_prediction_stub(samples)
    llm_manifest = build_llm_input_manifest(
        samples,
        split_name=split_name,
        instruction_source=instruction_source,
        context_frames=context_frames,
        min_image_inputs=min_image_inputs,
    )
    sft_records = build_sft_records(llm_manifest)
    preference_records = build_preference_records(llm_manifest, hardest_fraction=hardest_fraction)

    output_files = {
        "samples": str(output_path / "samples.json"),
        "predictions": str(output_path / "predictions.json"),
        "llm_manifest": str(output_path / "llm_manifest.json"),
        "sft": str(output_path / "sft.jsonl"),
        "preference": str(output_path / "preference.jsonl"),
        "build_report": str(output_path / "build_report.json"),
    }
    report = {
        "source_root": str(source_path),
        "output_root": str(output_path),
        "target_frames_per_sequence": target_frames_per_sequence,
        "sequence_count": len(materialized_manifests),
        "sample_count": len(samples),
        "prediction_count": len(predictions),
        "llm_record_count": len(llm_manifest),
        "sft_record_count": len(sft_records),
        "preference_record_count": len(preference_records),
        "sequences": sequence_reports,
        "output_files": output_files,
    }

    save_json(output_files["samples"], samples)
    save_json(output_files["predictions"], predictions)
    save_json(output_files["llm_manifest"], llm_manifest)
    save_jsonl(output_files["sft"], sft_records)
    save_jsonl(output_files["preference"], preference_records)
    save_json(output_files["build_report"], report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a materialized UAVscenes subset for EmergUAV-Bench.")
    parser.add_argument("--uavscenes-root", default=DEFAULT_UAVSCENES_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-frames-per-sequence", type=int, default=36)
    parser.add_argument("--sequence-id", action="append", dest="sequence_ids")
    parser.add_argument("--split-name", default="dev")
    parser.add_argument("--instruction-source", choices=["clean", "noisy"], default="noisy")
    parser.add_argument("--context-frames", type=int, default=2)
    parser.add_argument("--min-image-inputs", type=int, default=3)
    parser.add_argument("--hardest-fraction", type=float, default=0.3)
    parser.add_argument("--protocol-context", default=None)
    parser.add_argument("--application-theme", default=None)
    parser.add_argument("--policy-family", default=None)
    parser.add_argument("--enable-nfz-overlay", action="store_true")
    parser.add_argument("--degradation-label", default="dust_veil")
    parser.add_argument("--degradation-severity", default="high")
    parser.add_argument("--language-entropy-label", action="append", dest="language_entropy_labels")
    args = parser.parse_args()

    report = build_materialized_bench_dataset(
        uavscenes_root=args.uavscenes_root,
        output_root=args.output_root,
        target_frames_per_sequence=args.target_frames_per_sequence,
        sequence_ids=args.sequence_ids,
        split_name=args.split_name,
        instruction_source=args.instruction_source,
        context_frames=args.context_frames,
        min_image_inputs=args.min_image_inputs,
        hardest_fraction=args.hardest_fraction,
        protocol_context=args.protocol_context,
        application_theme=args.application_theme,
        policy_family=args.policy_family,
        enable_nfz_overlay=args.enable_nfz_overlay,
        degradation_label=args.degradation_label,
        degradation_severity=args.degradation_severity,
        language_entropy_labels=args.language_entropy_labels,
    )
    print(f"Wrote {report['sample_count']} samples to {report['output_root']}")


if __name__ == "__main__":
    main()
