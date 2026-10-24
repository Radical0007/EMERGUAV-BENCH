from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

from emerguav_bench.evaluation.semantic_metrics import METRIC_PROFILE, compute_semantic_report
from scripts.evaluate_predictions import evaluate_predictions


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Any) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _prediction_paths(leaderboard_root: Path, *, split_name: str, prompt_mode: str, seed: int) -> list[Path]:
    pattern = f"*/{split_name}/{prompt_mode}/seed_{seed}/predictions.json"
    return sorted(leaderboard_root.glob(pattern))


def _read_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        metadata = load_json(metadata_path)
    except Exception as exc:
        return {"metadata_error": str(exc)}
    return metadata if isinstance(metadata, dict) else {"metadata_error": "run_metadata.json must be an object"}


def _float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _format_score(value: Any) -> str:
    return f"{_float(value):.4f}"


def _csv_text(rows: list[dict[str, Any]], columns: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue()


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body]) + "\n"


def _main_row(model_id: str, metadata: dict[str, Any], strict_report: dict[str, Any], semantic_report: dict[str, Any]) -> dict[str, Any]:
    strict_task = strict_report["task_metrics"]
    strict_summary = strict_report["summary_metrics"]
    semantic_task = semantic_report["semantic_task_metrics"]
    semantic_summary = semantic_report["semantic_summary_metrics"]
    return {
        "Model": model_id,
        "Status": metadata.get("status", "unknown"),
        "N": str(int(semantic_task["evaluated_samples"])),
        "Semantic Protocol Decision Score": _format_score(semantic_summary["Semantic Protocol Decision Score"]),
        "Semantic Avg Task Score": _format_score(semantic_summary["Semantic Avg Task Score"]),
        "Safe Action Accuracy": _format_score(strict_summary["Safe Action Accuracy"]),
        "Unsafe Action Rate": _format_score(strict_summary["Unsafe Action Rate"]),
        "PACS": _format_score(strict_summary["PACS"]),
        "MAD": _format_score(strict_summary["MAD"]),
        "Strict Avg Task Accuracy": _format_score(strict_task["avg_task_accuracy"]),
        "T1": _format_score(strict_task["T1_accuracy"]),
        "T1 Exact": _format_score(strict_task["protocol_context_exact_accuracy"]),
        "Norm Failures": str(int(metadata.get("normalization_failure_count", 0) or 0)),
    }


def _task_row(
    model_id: str,
    metadata: dict[str, Any],
    strict_report: dict[str, Any],
    semantic_report: dict[str, Any],
) -> dict[str, Any]:
    strict_task = strict_report["task_metrics"]
    semantic_task = semantic_report["semantic_task_metrics"]
    row = {
        "Model": model_id,
        "Status": metadata.get("status", "unknown"),
        "Semantic Protocol Decision Score": _format_score(semantic_task["semantic_protocol_decision_score"]),
        "T1": _format_score(semantic_task["T1_score"]),
        "T1 Exact": _format_score(strict_task["protocol_context_exact_accuracy"]),
    }
    for task_index in range(2, 13):
        row[f"T{task_index}"] = _format_score(semantic_task[f"T{task_index}_score"])
    row["Semantic Avg Task Score"] = _format_score(semantic_task["semantic_avg_task_score"])
    row["Strict Avg Task Accuracy"] = _format_score(strict_task["avg_task_accuracy"])
    return row


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row.get("Status") != "passed",
            -_float(row.get("Semantic Protocol Decision Score")),
            row.get("Model", ""),
        ),
    )


def evaluate_leaderboard(
    *,
    samples_path: str | Path,
    leaderboard_root: str | Path,
    split_name: str = "strict",
    prompt_mode: str = "zero_shot",
    seed: int = 0,
) -> dict[str, Any]:
    samples = load_json(samples_path)
    root = Path(leaderboard_root)
    main_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []

    for predictions_path in _prediction_paths(root, split_name=split_name, prompt_mode=prompt_mode, seed=seed):
        model_id = predictions_path.relative_to(root).parts[0]
        run_dir = predictions_path.parent
        predictions = load_json(predictions_path)
        metadata = _read_metadata(run_dir)

        strict_report = evaluate_predictions(samples, predictions)
        strict_report_path = run_dir / "evaluation_report.json"
        save_json(strict_report_path, strict_report)

        semantic_report = compute_semantic_report(samples, predictions)
        semantic_report["strict_report_path"] = str(strict_report_path)
        semantic_report["predictions_path"] = str(predictions_path)
        semantic_report_path = run_dir / "semantic_evaluation_report.json"
        save_json(semantic_report_path, semantic_report)

        main_row = _main_row(model_id, metadata, strict_report, semantic_report)
        task_row = _task_row(model_id, metadata, strict_report, semantic_report)
        main_rows.append(main_row)
        task_rows.append(task_row)
        runs.append(
            {
                "model_id": model_id,
                "status": metadata.get("status", "unknown"),
                "predictions_path": str(predictions_path),
                "evaluation_report_path": str(strict_report_path),
                "semantic_evaluation_report_path": str(semantic_report_path),
                "evaluated_samples": int(strict_report["task_metrics"]["evaluated_samples"]),
                "semantic_protocol_decision_score": semantic_report["semantic_task_metrics"]["semantic_protocol_decision_score"],
                "semantic_avg_task_score": semantic_report["semantic_task_metrics"]["semantic_avg_task_score"],
                "strict_avg_task_accuracy": strict_report["task_metrics"]["avg_task_accuracy"],
            }
        )

    main_rows = _sort_rows(main_rows)
    task_rows = _sort_rows(task_rows)
    main_columns = [
        "Model",
        "Status",
        "N",
        "Semantic Protocol Decision Score",
        "Semantic Avg Task Score",
        "Safe Action Accuracy",
        "Unsafe Action Rate",
        "PACS",
        "MAD",
        "Strict Avg Task Accuracy",
        "T1",
        "T1 Exact",
        "Norm Failures",
    ]
    task_columns = [
        "Model",
        "T1",
        "T1 Exact",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
        "T8",
        "T9",
        "T10",
        "T11",
        "T12",
        "Semantic Avg Task Score",
        "Strict Avg Task Accuracy",
    ]
    (root / "semantic_main_leaderboard.csv").write_text(_csv_text(main_rows, main_columns), encoding="utf-8")
    (root / "semantic_main_leaderboard.md").write_text(_markdown_table(main_rows, main_columns), encoding="utf-8")
    (root / "semantic_task_breakdown.csv").write_text(_csv_text(task_rows, task_columns), encoding="utf-8")
    (root / "semantic_task_breakdown.md").write_text(_markdown_table(task_rows, task_columns), encoding="utf-8")

    summary = {
        "metric_profile": METRIC_PROFILE,
        "samples_path": str(samples_path),
        "leaderboard_root": str(root),
        "split_name": split_name,
        "prompt_mode": prompt_mode,
        "seed": seed,
        "recomputed_model_count": len(runs),
        "runs": runs,
        "semantic_main_leaderboard_path": str(root / "semantic_main_leaderboard.md"),
        "semantic_task_breakdown_path": str(root / "semantic_task_breakdown.md"),
    }
    save_json(root / "semantic_recompute_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute strict and semantic leaderboard reports from existing predictions only."
    )
    parser.add_argument("--samples", required=True, help="Benchmark samples JSON path.")
    parser.add_argument("--leaderboard-root", required=True, help="Existing leaderboard root containing model prediction trees.")
    parser.add_argument("--split-name", default="strict")
    parser.add_argument("--prompt-mode", default="zero_shot")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = evaluate_leaderboard(
        samples_path=args.samples,
        leaderboard_root=args.leaderboard_root,
        split_name=args.split_name,
        prompt_mode=args.prompt_mode,
        seed=args.seed,
    )
    print(f"Recomputed {summary['recomputed_model_count']} model reports with {summary['metric_profile']}.")


if __name__ == "__main__":
    main()
