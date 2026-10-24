from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from emerguav_bench.evaluation.metrics import compute_grouped_metrics, compute_summary_metrics, compute_task_metrics
from emerguav_bench.utils.progress import progress_iter
from emerguav_bench.validation.json_validator import assert_valid_model_output, assert_valid_sample_record


def evaluate_predictions(samples: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    for sample in progress_iter(samples, desc="validate eval samples", total=len(samples), unit="sample"):
        assert_valid_sample_record(sample)

    for prediction in progress_iter(predictions, desc="validate predictions", total=len(predictions), unit="prediction"):
        if not isinstance(prediction, dict) or "sample_id" not in prediction or "output" not in prediction:
            raise ValueError("Each prediction must contain 'sample_id' and 'output'")
        assert_valid_model_output(prediction["output"])

    task_metrics = compute_task_metrics(samples, predictions)
    summary_metrics = compute_summary_metrics(samples, predictions)
    grouped_results = compute_grouped_metrics(samples, predictions)

    return {
        "task_metrics": task_metrics,
        "summary_metrics": summary_metrics,
        "grouped_results": grouped_results,
    }


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EmergUAV-Bench prediction JSON against sample JSON.")
    parser.add_argument("--samples", required=True, help="Path to samples JSON")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON")
    parser.add_argument("--report-out", required=True, help="Path to evaluation report JSON")
    args = parser.parse_args()

    samples = load_json(args.samples)
    predictions = load_json(args.predictions)
    report = evaluate_predictions(samples, predictions)
    save_json(args.report_out, report)


if __name__ == "__main__":
    main()
