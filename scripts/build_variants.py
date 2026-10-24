from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from emerguav_bench.degradation.variants import (
    STRICT_VARIANT_MATRIX,
    allocate_strict_variant_cases,
    build_strict_variant_samples,
    summarize_variant_coverage,
)
from emerguav_bench.allocation_profiles import paper_allocation_profile_summary
from emerguav_bench.llm.exporters import build_llm_input_manifest, build_preference_records, build_sft_records
from emerguav_bench.llm.io import load_json, save_json, save_jsonl
from emerguav_bench.llm.runner import LLMInferenceRunner
from emerguav_bench.utils.progress import progress_iter
from emerguav_bench.validation.json_validator import assert_valid_sample_record
from scripts.build_samples import build_prediction_stub
from scripts.evaluate_predictions import evaluate_predictions

DEFAULT_BASE_SAMPLES_PATH = "artifacts/uavscenes_bench/samples.json"
DEFAULT_OUTPUT_ROOT = "artifacts/uavscenes_bench_strict"


def build_variants(
    *,
    base_samples_path: str | Path = DEFAULT_BASE_SAMPLES_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    split_name: str = "strict",
    instruction_source: str = "noisy",
    context_frames: int = 2,
    min_image_inputs: int = 3,
    hardest_fraction: float = 0.3,
    run_offline_inference: bool = True,
    model_id: str = "qwen3_vl",
    offline_runtime: str = "mock",
    prompt_mode: str = "zero_shot",
    seed: int = 17,
) -> dict[str, Any]:
    output_path = Path(output_root)
    base_samples = load_json(base_samples_path)
    variant_allocations = allocate_strict_variant_cases(base_samples, STRICT_VARIANT_MATRIX)
    variant_samples = build_strict_variant_samples(base_samples, output_root=output_path, seed=seed)
    for sample in progress_iter(variant_samples, desc="validate strict samples", total=len(variant_samples), unit="sample"):
        assert_valid_sample_record(sample)

    predictions = build_prediction_stub(variant_samples)
    llm_manifest = build_llm_input_manifest(
        variant_samples,
        split_name=split_name,
        instruction_source=instruction_source,
        context_frames=context_frames,
        min_image_inputs=min_image_inputs,
    )
    sft_records = build_sft_records(llm_manifest)
    preference_records = build_preference_records(llm_manifest, hardest_fraction=hardest_fraction)
    evaluation_report = evaluate_predictions(variant_samples, predictions)

    output_files = {
        "samples": str(output_path / "samples.json"),
        "predictions": str(output_path / "predictions.json"),
        "llm_manifest": str(output_path / "llm_manifest.json"),
        "sft": str(output_path / "sft.jsonl"),
        "preference": str(output_path / "preference.jsonl"),
        "evaluation_report": str(output_path / "evaluation_report.json"),
        "build_report": str(output_path / "build_report.json"),
    }

    offline_predictions: list[dict[str, Any]] = []
    offline_evaluation_report: dict[str, Any] | None = None
    if run_offline_inference:
        runner = LLMInferenceRunner(model_id=model_id, runtime=offline_runtime)
        offline_predictions = runner.run(
            llm_manifest,
            prompt_mode=prompt_mode,
            seed=seed,
        )
        offline_evaluation_report = evaluate_predictions(variant_samples, offline_predictions)
        output_files["offline_predictions"] = str(output_path / "offline_predictions.json")
        output_files["offline_evaluation_report"] = str(output_path / "offline_evaluation_report.json")

    report = {
        "base_samples_path": str(base_samples_path),
        "output_root": str(output_path),
        "base_sample_count": len(base_samples),
        "variant_case_count": len(STRICT_VARIANT_MATRIX),
        "variant_allocation_mode": "nonlinear_semantic_curve",
        "variant_case_target_counts": {
            allocation.case.variant_id: allocation.target_count
            for allocation in variant_allocations
        },
        "variant_sample_count": len(variant_samples),
        "prediction_count": len(predictions),
        "llm_record_count": len(llm_manifest),
        "sft_record_count": len(sft_records),
        "preference_record_count": len(preference_records),
        "offline_prediction_count": len(offline_predictions),
        "offline_model_id": model_id if run_offline_inference else None,
        "offline_runtime": offline_runtime if run_offline_inference else None,
        "offline_prompt_mode": prompt_mode if run_offline_inference else None,
        "variant_coverage": summarize_variant_coverage(variant_samples),
        "paper_allocation_coverage": paper_allocation_profile_summary(),
        "output_files": output_files,
    }

    save_json(output_files["samples"], variant_samples)
    save_json(output_files["predictions"], predictions)
    save_json(output_files["llm_manifest"], llm_manifest)
    save_jsonl(output_files["sft"], sft_records)
    save_jsonl(output_files["preference"], preference_records)
    save_json(output_files["evaluation_report"], evaluation_report)
    if run_offline_inference:
        save_json(output_files["offline_predictions"], offline_predictions)
        save_json(output_files["offline_evaluation_report"], offline_evaluation_report)
    save_json(output_files["build_report"], report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict table-driven UAVScenes derived variants.")
    parser.add_argument("--base-samples", default=DEFAULT_BASE_SAMPLES_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split-name", default="strict")
    parser.add_argument("--instruction-source", choices=["clean", "noisy"], default="noisy")
    parser.add_argument("--context-frames", type=int, default=2)
    parser.add_argument("--min-image-inputs", type=int, default=3)
    parser.add_argument("--hardest-fraction", type=float, default=0.3)
    parser.add_argument("--skip-offline-inference", action="store_true")
    parser.add_argument("--model-id", default="qwen3_vl")
    parser.add_argument("--offline-runtime", choices=["mock", "live"], default="mock")
    parser.add_argument("--prompt-mode", choices=["zero_shot", "few_shot"], default="zero_shot")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    report = build_variants(
        base_samples_path=args.base_samples,
        output_root=args.output_root,
        split_name=args.split_name,
        instruction_source=args.instruction_source,
        context_frames=args.context_frames,
        min_image_inputs=args.min_image_inputs,
        hardest_fraction=args.hardest_fraction,
        run_offline_inference=not args.skip_offline_inference,
        model_id=args.model_id,
        offline_runtime=args.offline_runtime,
        prompt_mode=args.prompt_mode,
        seed=args.seed,
    )
    print(f"Wrote {report['variant_sample_count']} strict variant samples to {report['output_root']}")


if __name__ == "__main__":
    main()
