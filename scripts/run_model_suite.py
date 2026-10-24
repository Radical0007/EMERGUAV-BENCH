from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from emerguav_bench.llm.registry import MODEL_PRESETS
from emerguav_bench.utils.gpu import OOM_SKIP_RETURN_CODE, classify_models, dual_gpu_env, gpu_affinity_env, gpu_count as get_gpu_count
from emerguav_bench.utils.progress import make_progress_bar

DEFAULT_SUITE_CONFIG = "configs/model_suites.json"


def load_suites(path: str | Path = DEFAULT_SUITE_CONFIG) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_name(manifest_path: str | Path) -> str:
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return str(manifest[0]["split_name"]) if manifest else "unknown"


def parse_model_override(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_suite_models(
    suites: dict[str, dict[str, Any]],
    *,
    suite_name: str,
    model_override: list[str] | None = None,
) -> list[str]:
    if suite_name not in suites:
        raise KeyError(f"Unknown suite: {suite_name}")
    model_ids = model_override or list(suites[suite_name].get("model_ids", []))
    unknown = [model_id for model_id in model_ids if model_id not in MODEL_PRESETS]
    if unknown:
        raise KeyError(f"Suite contains unsupported model ids: {', '.join(unknown)}")
    return model_ids


def prediction_path(results_root: str | Path, *, model_id: str, split_name: str, prompt_mode: str, seed: int) -> Path:
    return Path(results_root) / model_id / split_name / prompt_mode / f"seed_{seed}" / "predictions.json"


def completed_prediction_file(path: str | Path, manifest: str | Path, *, max_samples: int | None = None) -> bool:
    prediction_path = Path(path)
    if not prediction_path.exists():
        return False
    try:
        with prediction_path.open("r", encoding="utf-8") as handle:
            predictions = json.load(handle)
        with Path(manifest).open("r", encoding="utf-8") as handle:
            records = json.load(handle)
    except Exception:
        return False
    if max_samples is not None:
        records = records[:max_samples]
    expected_sample_ids = {str(record.get("sample_id")) for record in records}
    predicted_sample_ids = {
        str(prediction.get("sample_id"))
        for prediction in predictions
        if isinstance(prediction, dict) and prediction.get("sample_id") is not None
    }
    return expected_sample_ids.issubset(predicted_sample_ids)


def normalization_failure_metadata_from_predictions(path: str | Path) -> dict[str, Any] | None:
    prediction_path = Path(path)
    if not prediction_path.exists():
        return None
    try:
        with prediction_path.open("r", encoding="utf-8") as handle:
            predictions = json.load(handle)
    except Exception:
        return None
    if not isinstance(predictions, list):
        return None

    failed_sample_ids: list[str] = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        output = prediction.get("output")
        if isinstance(output, dict) and output.get("_normalization_failed"):
            failed_sample_ids.append(str(prediction.get("sample_id")))
    return {
        "normalization_failure_count": len(failed_sample_ids),
        "normalization_failure_sample_ids": failed_sample_ids[:50],
    }


def load_run_metadata_with_prediction_backfill(metadata_path: str | Path, predictions_path: str | Path) -> dict[str, Any] | None:
    metadata_file = Path(metadata_path)
    metadata: dict[str, Any] | None = None
    if metadata_file.exists():
        try:
            with metadata_file.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                metadata = loaded
            else:
                metadata = {"metadata_error": "run_metadata.json must be an object"}
        except Exception as exc:
            metadata = {"metadata_error": str(exc)}

    failure_metadata = normalization_failure_metadata_from_predictions(predictions_path)
    if failure_metadata is None:
        return metadata

    if metadata is None:
        metadata = {"metadata_source": "predictions_json_backfill"}
    metadata.update(failure_metadata)
    write_json(metadata_file, metadata)
    return metadata


def _pump_stream(stream: Any, target: Any, chunks: list[str]) -> None:
    while True:
        chunk = stream.read(1)
        if not chunk:
            break
        chunks.append(chunk)
        target.write(chunk)
        target.flush()


def run_command(command: list[str], *, cwd: Path, stream_output: bool = False, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    if not stream_output:
        return subprocess.run(command, cwd=cwd, text=True, capture_output=True, env=env or os.environ)

    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        env=env or os.environ,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    threads = [
        threading.Thread(target=_pump_stream, args=(process.stdout, sys.stdout, stdout_chunks), daemon=True),
        threading.Thread(target=_pump_stream, args=(process.stderr, sys.stderr, stderr_chunks), daemon=True),
    ]
    for thread in threads:
        thread.start()
    returncode = process.wait()
    for thread in threads:
        thread.join()
    return subprocess.CompletedProcess(
        args=command,
        returncode=returncode,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _tail(text: str, *, tail_lines: int) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-tail_lines:])


def format_failure_summary(summary: dict[str, Any], *, tail_lines: int = 60) -> str:
    failed_runs = [run for run in summary.get("runs", []) if run.get("status") != "passed"]
    if not failed_runs:
        return ""
    blocks = ["Failed model diagnostics:"]
    for run in failed_runs:
        model_id = run.get("model_id", "<unknown>")
        blocks.append("=" * 80)
        blocks.append(
            f"{model_id} failed "
            f"(leaderboard_rc={run.get('leaderboard_returncode')}, eval_rc={run.get('evaluation_returncode')})"
        )
        stderr = str(run.get("leaderboard_stderr") or run.get("evaluation_stderr") or "").strip()
        stdout = str(run.get("leaderboard_stdout") or run.get("evaluation_stdout") or "").strip()
        if stderr:
            blocks.append("--- stderr tail ---")
            blocks.append(_tail(stderr, tail_lines=tail_lines))
        if stdout:
            blocks.append("--- stdout tail ---")
            blocks.append(_tail(stdout, tail_lines=max(10, tail_lines // 2)))
        if not stderr and not stdout:
            blocks.append("<no subprocess output captured>")
    return "\n".join(blocks)


def _run_single_model(
    *,
    model_id: str,
    manifest: str | Path,
    samples: str | Path | None,
    results_root: str | Path,
    split_name: str,
    prompt_mode: str,
    few_shot_count: int,
    seed: int,
    runtime: str,
    skip_existing: bool,
    resume_model: bool,
    flush_every: int,
    max_samples_per_model: int | None,
    max_new_tokens: int | None,
    processor_max_pixels: int | None,
    batch_size: int,
    auto_batch: bool,
    precision: str,
    cuda_alloc_conf: str | None,
    stream_output: bool,
    repo_root: Path,
    env_override: dict[str, str] | None = None,
) -> dict[str, Any]:
    output_predictions = prediction_path(
        results_root,
        model_id=model_id,
        split_name=split_name,
        prompt_mode=prompt_mode,
        seed=seed,
    )
    run_record: dict[str, Any] = {
        "model_id": model_id,
        "status": "pending",
        "predictions_path": str(output_predictions),
        "evaluation_report_path": None,
    }
    model_started_at = time.perf_counter()
    if skip_existing and completed_prediction_file(output_predictions, manifest, max_samples=max_samples_per_model):
        leaderboard_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="skipped existing predictions", stderr="")
        leaderboard_seconds = 0.0
    else:
        leaderboard_started_at = time.perf_counter()
        command = [
            sys.executable,
            "scripts/run_inference.py",
            "--manifest",
            str(manifest),
            "--model-id",
            model_id,
            "--results-root",
            str(results_root),
            "--prompt-mode",
            prompt_mode,
            "--few-shot-count",
            str(few_shot_count),
            "--seed",
            str(seed),
            "--runtime",
            runtime,
        ]
        if resume_model:
            command.append("--resume")
        if flush_every:
            command.extend(["--flush-every", str(flush_every)])
        if max_samples_per_model is not None:
            command.extend(["--max-samples", str(max_samples_per_model)])
        if max_new_tokens is not None:
            command.extend(["--max-new-tokens", str(max_new_tokens)])
        if processor_max_pixels is not None:
            command.extend(["--processor-max-pixels", str(processor_max_pixels)])
        command.extend(["--batch-size", str(batch_size)])
        if not auto_batch:
            command.append("--disable-auto-batch")
        command.extend(["--precision", precision])
        env = (env_override or os.environ).copy()
        if cuda_alloc_conf:
            env["PYTORCH_CUDA_ALLOC_CONF"] = cuda_alloc_conf
        leaderboard_result = run_command(
            command,
            cwd=repo_root,
            stream_output=stream_output,
            env=env,
        )
        leaderboard_seconds = time.perf_counter() - leaderboard_started_at
    run_record["leaderboard_returncode"] = leaderboard_result.returncode
    run_record["leaderboard_stdout"] = leaderboard_result.stdout
    run_record["leaderboard_stderr"] = leaderboard_result.stderr
    run_record["leaderboard_seconds"] = round(leaderboard_seconds, 3)
    metadata_path = output_predictions.parent / "run_metadata.json"
    run_record["run_metadata"] = load_run_metadata_with_prediction_backfill(metadata_path, output_predictions)

    if leaderboard_result.returncode == OOM_SKIP_RETURN_CODE:
        run_record["evaluation_seconds"] = None
        run_record["status"] = "skipped_oom"
    elif leaderboard_result.returncode == 0 and samples is not None:
        evaluation_path = output_predictions.parent / "evaluation_report.json"
        evaluation_started_at = time.perf_counter()
        eval_result = run_command(
            [
                sys.executable,
                "scripts/evaluate_predictions.py",
                "--samples",
                str(samples),
                "--predictions",
                str(output_predictions),
                "--report-out",
                str(evaluation_path),
            ],
            cwd=repo_root,
            stream_output=stream_output,
            env=env_override,
        )
        run_record["evaluation_returncode"] = eval_result.returncode
        run_record["evaluation_stdout"] = eval_result.stdout
        run_record["evaluation_stderr"] = eval_result.stderr
        run_record["evaluation_seconds"] = round(time.perf_counter() - evaluation_started_at, 3)
        run_record["evaluation_report_path"] = str(evaluation_path)
        run_record["status"] = "passed" if eval_result.returncode == 0 else "failed"
    else:
        run_record["evaluation_seconds"] = None
        run_record["status"] = "passed" if leaderboard_result.returncode == 0 else "failed"

    run_record["total_seconds"] = round(time.perf_counter() - model_started_at, 3)
    return run_record


def _run_models_parallel(
    *,
    model_ids: list[str],
    manifest: str | Path,
    samples: str | Path | None,
    results_root: str | Path,
    split_name: str,
    prompt_mode: str,
    few_shot_count: int,
    seed: int,
    runtime: str,
    skip_existing: bool,
    resume_model: bool,
    flush_every: int,
    max_samples_per_model: int | None,
    max_new_tokens: int | None,
    processor_max_pixels: int | None,
    batch_size: int,
    auto_batch: bool,
    precision: str,
    cuda_alloc_conf: str | None,
    fail_fast: bool,
    repo_root: Path,
) -> list[dict[str, Any]]:
    single_models, dual_models = classify_models(model_ids)
    results: dict[str, dict[str, Any]] = {}
    results_lock = threading.Lock()
    stop_event = threading.Event()

    total = len(model_ids)
    completed = [0]

    def _worker(model_queue: list[str], gpu_id: int) -> None:
        env = gpu_affinity_env(gpu_id)
        for mid in model_queue:
            if stop_event.is_set():
                break
            run_record = _run_single_model(
                model_id=mid,
                manifest=manifest,
                samples=samples,
                results_root=results_root,
                split_name=split_name,
                prompt_mode=prompt_mode,
                few_shot_count=few_shot_count,
                seed=seed,
                runtime=runtime,
                skip_existing=skip_existing,
                resume_model=resume_model,
                flush_every=flush_every,
                max_samples_per_model=max_samples_per_model,
                max_new_tokens=max_new_tokens,
                processor_max_pixels=processor_max_pixels,
                batch_size=batch_size,
                auto_batch=auto_batch,
                precision=precision,
                cuda_alloc_conf=cuda_alloc_conf,
                stream_output=False,
                repo_root=repo_root,
                env_override=env,
            )
            with results_lock:
                results[mid] = run_record
                completed[0] += 1
                if fail_fast and run_record["status"] != "passed":
                    stop_event.set()
            gpu_label = f"GPU {gpu_id}"
            _ts = run_record["total_seconds"]
            print(f"[{gpu_label}] {mid}: {run_record['status']} ({completed[0]}/{total}) | leaderboard {run_record['leaderboard_seconds']}s | total {_ts}s")

    if single_models:
        gpu0_queue = single_models[::2]
        gpu1_queue = single_models[1::2]
        if not stop_event.is_set():
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(_worker, gpu0_queue, 0),
                    executor.submit(_worker, gpu1_queue, 1),
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()

    if dual_models and not stop_event.is_set():
        env = dual_gpu_env()
        for mid in dual_models:
            if stop_event.is_set():
                break
            run_record = _run_single_model(
                model_id=mid,
                manifest=manifest,
                samples=samples,
                results_root=results_root,
                split_name=split_name,
                prompt_mode=prompt_mode,
                few_shot_count=few_shot_count,
                seed=seed,
                runtime=runtime,
                skip_existing=skip_existing,
                resume_model=resume_model,
                flush_every=flush_every,
                max_samples_per_model=max_samples_per_model,
                max_new_tokens=max_new_tokens,
                processor_max_pixels=processor_max_pixels,
                batch_size=batch_size,
                auto_batch=auto_batch,
                precision=precision,
                cuda_alloc_conf=cuda_alloc_conf,
                stream_output=False,
                repo_root=repo_root,
                env_override=env,
            )
            with results_lock:
                results[mid] = run_record
                completed[0] += 1
                if fail_fast and run_record["status"] != "passed":
                    stop_event.set()
            _ts = run_record["total_seconds"]
            print(f"[GPU 0,1] {mid}: {run_record['status']} ({completed[0]}/{total}) | leaderboard {run_record['leaderboard_seconds']}s | total {_ts}s")

    return [results[mid] for mid in model_ids if mid in results]


def run_model_suite(
    *,
    manifest: str | Path,
    samples: str | Path | None,
    suite: str,
    results_root: str | Path,
    suite_config: str | Path = DEFAULT_SUITE_CONFIG,
    runtime: str = "mock",
    prompt_mode: str = "zero_shot",
    few_shot_count: int = 0,
    seed: int = 0,
    models: list[str] | None = None,
    skip_existing: bool = False,
    resume_model: bool = False,
    flush_every: int = 0,
    max_samples_per_model: int | None = None,
    max_new_tokens: int | None = None,
    processor_max_pixels: int | None = None,
    batch_size: int = 0,
    auto_batch: bool = True,
    precision: str = "bf16",
    cuda_alloc_conf: str | None = None,
    fail_fast: bool = False,
    stream_subprocess: bool = True,
    parallel: bool = False,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    suites = load_suites(suite_config)
    model_ids = resolve_suite_models(suites, suite_name=suite, model_override=models)
    split_name = load_split_name(manifest)

    if parallel:
        if get_gpu_count() >= 2:
            runs = _run_models_parallel(
                model_ids=model_ids,
                manifest=manifest,
                samples=samples,
                results_root=results_root,
                split_name=split_name,
                prompt_mode=prompt_mode,
                few_shot_count=few_shot_count,
                seed=seed,
                runtime=runtime,
                skip_existing=skip_existing,
                resume_model=resume_model,
                flush_every=flush_every,
                max_samples_per_model=max_samples_per_model,
                max_new_tokens=max_new_tokens,
                processor_max_pixels=processor_max_pixels,
                batch_size=batch_size,
                auto_batch=auto_batch,
                precision=precision,
                cuda_alloc_conf=cuda_alloc_conf,
                fail_fast=fail_fast,
                repo_root=repo_root,
            )
        else:
            print("[warn] --parallel requested but only 1 GPU detected; falling back to sequential", file=sys.stderr)
            parallel = False

    if not parallel:
        runs: list[dict[str, Any]] = []
        passed_count = 0
        skipped_count = 0
        failed_count = 0

        with make_progress_bar(total=len(model_ids), desc=f"{suite} suite", unit="model") as progress_bar:
            for index, model_id in enumerate(model_ids, start=1):
                progress_bar.set_description(f"{suite} suite: {model_id}")
                progress_bar.set_postfix(stage="prepare", passed=passed_count, skipped=skipped_count, failed=failed_count, item=f"{index}/{len(model_ids)}")
                run_record = _run_single_model(
                    model_id=model_id,
                    manifest=manifest,
                    samples=samples,
                    results_root=results_root,
                    split_name=split_name,
                    prompt_mode=prompt_mode,
                    few_shot_count=few_shot_count,
                    seed=seed,
                    runtime=runtime,
                    skip_existing=skip_existing,
                    resume_model=resume_model,
                    flush_every=flush_every,
                    max_samples_per_model=max_samples_per_model,
                    max_new_tokens=max_new_tokens,
                    processor_max_pixels=processor_max_pixels,
                    batch_size=batch_size,
                    auto_batch=auto_batch,
                    precision=precision,
                    cuda_alloc_conf=cuda_alloc_conf,
                    stream_output=stream_subprocess,
                    repo_root=repo_root,
                )
                runs.append(run_record)
                if run_record["status"] == "passed":
                    passed_count += 1
                elif run_record["status"] == "skipped_oom":
                    skipped_count += 1
                else:
                    failed_count += 1
                progress_bar.set_postfix(stage=run_record["status"], passed=passed_count, skipped=skipped_count, failed=failed_count, item=f"{index}/{len(model_ids)}")
                progress_bar.update(1)
                if fail_fast and run_record["status"] != "passed":
                    break

    summary = {
        "suite": suite,
        "suite_config": str(suite_config),
        "runtime": runtime,
        "prompt_mode": prompt_mode,
        "few_shot_count": few_shot_count,
        "seed": seed,
        "batch_size": batch_size,
        "auto_batch": auto_batch,
        "precision": precision,
        "split_name": split_name,
        "requested_model_count": len(model_ids),
        "passed_model_count": sum(1 for run in runs if run["status"] == "passed"),
        "skipped_model_count": sum(1 for run in runs if run["status"] == "skipped_oom"),
        "failed_model_count": sum(1 for run in runs if run["status"] not in {"passed", "skipped_oom"}),
        "runs": runs,
    }
    write_json(Path(results_root) / f"suite_{suite}_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a configured EmergUAV-Bench LLM/VLM model suite.")
    parser.add_argument("--manifest", required=True, help="Input LLM manifest JSON path")
    parser.add_argument("--samples", help="Optional benchmark samples JSON for evaluator round-trip")
    parser.add_argument("--suite", choices=["core", "extended"], default="core")
    parser.add_argument("--suite-config", default=DEFAULT_SUITE_CONFIG)
    parser.add_argument("--models", help="Comma-separated model ids to run instead of the suite default")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--prompt-mode", choices=["zero_shot", "few_shot"], default="zero_shot")
    parser.add_argument("--few-shot-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runtime", choices=["mock", "live"], default="mock")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume-model", action="store_true", help="Resume each model at sample level from partial predictions.json files.")
    parser.add_argument("--flush-every", type=int, default=0, help="With --resume-model, flush partial predictions every N newly generated samples.")
    parser.add_argument("--max-samples-per-model", type=int, help="Run only the first N manifest samples per model for smoke testing.")
    parser.add_argument("--max-new-tokens", type=int, help="Override generation max_new_tokens for all suite models.")
    parser.add_argument("--processor-max-pixels", type=int, help="Set common processor image max_pixels attributes for all suite models.")
    parser.add_argument("--batch-size", type=int, default=0, help="Samples per model batch. Use 0 for model-size heuristic.")
    parser.add_argument("--disable-auto-batch", action="store_true", help="Disable CUDA OOM batch-size reduction and retry.")
    parser.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16", "fp32", "int8", "int4"],
        default="bf16",
        help="Live model load precision. Mock runtime records the value but does not load weights.",
    )
    parser.add_argument("--cuda-alloc-conf", help="Set PYTORCH_CUDA_ALLOC_CONF for leaderboard/evaluation subprocesses.")
    parser.add_argument(
        "--expandable-cuda-segments",
        action="store_true",
        help="Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for subprocesses unless --cuda-alloc-conf is provided.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--quiet-subprocess", action="store_true", help="Capture subprocess output without streaming per-model progress live.")
    parser.add_argument("--parallel", action="store_true", dest="parallel", default=False, help="Run models in parallel across dual GPUs (one model per GPU).")
    parser.add_argument("--no-parallel", action="store_false", dest="parallel", help="Disable parallel execution (default).")
    args = parser.parse_args()
    if args.flush_every < 0:
        parser.error("--flush-every must be >= 0")
    if args.max_samples_per_model is not None and args.max_samples_per_model <= 0:
        parser.error("--max-samples-per-model must be > 0")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be > 0")
    if args.processor_max_pixels is not None and args.processor_max_pixels <= 0:
        parser.error("--processor-max-pixels must be > 0")
    if args.batch_size < 0:
        parser.error("--batch-size must be >= 0")
    cuda_alloc_conf = args.cuda_alloc_conf
    if args.expandable_cuda_segments and not cuda_alloc_conf:
        cuda_alloc_conf = "expandable_segments:True"

    summary = run_model_suite(
        manifest=args.manifest,
        samples=args.samples,
        suite=args.suite,
        results_root=args.results_root,
        suite_config=args.suite_config,
        runtime=args.runtime,
        prompt_mode=args.prompt_mode,
        few_shot_count=args.few_shot_count,
        seed=args.seed,
        models=parse_model_override(args.models),
        skip_existing=args.skip_existing,
        resume_model=args.resume_model,
        flush_every=args.flush_every,
        max_samples_per_model=args.max_samples_per_model,
        max_new_tokens=args.max_new_tokens,
        processor_max_pixels=args.processor_max_pixels,
        batch_size=args.batch_size,
        auto_batch=not args.disable_auto_batch,
        precision=args.precision,
        cuda_alloc_conf=cuda_alloc_conf,
        fail_fast=args.fail_fast,
        stream_subprocess=not args.quiet_subprocess,
        parallel=args.parallel,
    )
    print(
        f"Suite {summary['suite']} finished: "
        f"{summary['passed_model_count']} passed, "
        f"{summary['skipped_model_count']} skipped, "
        f"{summary['failed_model_count']} failed"
    )
    failure_summary = format_failure_summary(summary)
    if failure_summary:
        print(failure_summary, file=sys.stderr)
    return 1 if summary["failed_model_count"] or summary["skipped_model_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
