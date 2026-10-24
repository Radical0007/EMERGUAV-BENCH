from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emerguav_bench.llm.io import load_json, save_json
from emerguav_bench.llm.runner import LLMInferenceRunner, save_predictions_tree
from emerguav_bench.utils.gpu import OOM_SKIP_RETURN_CODE, clear_cuda_memory, is_cuda_out_of_memory_error


def _metadata_payload(args: argparse.Namespace, runner: LLMInferenceRunner, *, status: str, error: str | None = None) -> dict:
    payload = {
        **runner.last_run_metadata,
        "status": status,
        "runtime": args.runtime,
        "prompt_mode": args.prompt_mode,
        "few_shot_count": args.few_shot_count,
        "seed": args.seed,
        "max_new_tokens": runner.adapter.generation_kwargs.get("max_new_tokens"),
        "generation_kwargs": dict(runner.adapter.generation_kwargs),
        "processor_max_pixels": args.processor_max_pixels,
    }
    if error is not None:
        payload["error"] = error
    return payload


def run_leaderboard(args: argparse.Namespace) -> int:
    manifest = load_json(args.manifest)
    if args.max_samples is not None:
        manifest = manifest[: args.max_samples]
    split_name = manifest[0]["split_name"] if manifest else "unknown"
    output_path = (
        Path(args.results_root)
        / args.model_id
        / split_name
        / args.prompt_mode
        / f"seed_{args.seed}"
        / "predictions.json"
    )
    existing_predictions = load_json(output_path) if args.resume and output_path.exists() else None

    generated_since_flush = 0
    current_predictions = list(existing_predictions or [])

    def persist_predictions(predictions: list[dict]) -> None:
        save_predictions_tree(
            predictions,
            results_root=args.results_root,
            model_id=args.model_id,
            split_name=split_name,
            prompt_mode=args.prompt_mode,
            seed=args.seed,
        )

    def track_and_flush(predictions: list[dict]) -> None:
        nonlocal generated_since_flush, current_predictions
        current_predictions = list(predictions)
        generated_since_flush += 1
        if args.flush_every and generated_since_flush >= args.flush_every:
            persist_predictions(current_predictions)
            generated_since_flush = 0

    generation_kwargs = {}
    if args.max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = args.max_new_tokens
    elif args.runtime == "live":
        generation_kwargs["max_new_tokens"] = 768
    runner = LLMInferenceRunner(
        model_id=args.model_id,
        runtime=args.runtime,
        generation_kwargs=generation_kwargs or None,
        processor_max_pixels=args.processor_max_pixels,
        batch_size=args.batch_size,
        auto_batch=not args.disable_auto_batch,
        precision=args.precision,
    )
    try:
        try:
            predictions = runner.run(
                manifest,
                prompt_mode=args.prompt_mode,
                few_shot_count=args.few_shot_count,
                seed=args.seed,
                existing_predictions=existing_predictions,
                on_prediction=track_and_flush,
            )
        except Exception as exc:
            if not is_cuda_out_of_memory_error(exc):
                raise
            persist_predictions(current_predictions)
            save_json(
                output_path.parent / "run_metadata.json",
                _metadata_payload(args, runner, status="skipped_oom", error=str(exc)),
            )
            return OOM_SKIP_RETURN_CODE
    finally:
        runner.adapter.unload()
        clear_cuda_memory()

    persist_predictions(predictions)
    save_json(output_path.parent / "run_metadata.json", _metadata_payload(args, runner, status="passed"))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a raw EmergUAV-Bench LLM/VLM leaderboard pass.")
    parser.add_argument("--manifest", required=True, help="Input LLM manifest JSON path")
    parser.add_argument("--model-id", required=True, help="Registry model id")
    parser.add_argument("--results-root", required=True, help="Root directory for leaderboard outputs")
    parser.add_argument("--prompt-mode", choices=["zero_shot", "few_shot"], default="zero_shot")
    parser.add_argument("--few-shot-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runtime", choices=["mock", "live"], default="mock")
    parser.add_argument("--resume", action="store_true", help="Resume at sample level from an existing predictions.json file.")
    parser.add_argument("--flush-every", type=int, default=0, help="When resuming, write partial predictions every N newly generated samples.")
    parser.add_argument("--max-samples", type=int, help="Run only the first N manifest samples for smoke testing.")
    parser.add_argument("--max-new-tokens", type=int, help="Override generation max_new_tokens for memory-constrained runs.")
    parser.add_argument("--processor-max-pixels", type=int, help="Set common processor image max_pixels attributes for memory-constrained VLM runs.")
    parser.add_argument("--batch-size", type=int, default=0, help="Samples per model batch. Use 0 for model-size heuristic.")
    parser.add_argument("--disable-auto-batch", action="store_true", help="Disable CUDA OOM batch-size reduction and retry.")
    parser.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16", "fp32", "int8", "int4"],
        default="bf16",
        help="Live model load precision. Mock runtime records the value but does not load weights.",
    )
    args = parser.parse_args()
    if args.flush_every < 0:
        parser.error("--flush-every must be >= 0")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be > 0")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be > 0")
    if args.processor_max_pixels is not None and args.processor_max_pixels <= 0:
        parser.error("--processor-max-pixels must be > 0")
    if args.batch_size < 0:
        parser.error("--batch-size must be >= 0")

    sys.exit(run_leaderboard(args))


if __name__ == "__main__":
    main()
